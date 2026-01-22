#!/usr/bin/env python
"""
Example script for getting and setting robot poses using RTDE. -samy kamkar

Usage:
    # Get current pose and save to file
    python pose.py -g [-f poses.json] [-n "home"] [-F]

    # Move to pose from file
    python pose.py -s -f poses.json -n "home"

    # Move to pose from command line
    python pose.py -s -p "0.3, 0.2, 0.4, 0, 3.14, 0"

    # List all poses in file
    python pose.py -l [-f poses.json]
"""

import sys
import os
import json
import argparse
import time
import socket
import math
import numpy as np

sys.path.append("..")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

ROBOT_HOST = os.getenv("UR_HOST", "localhost")
RTDE_PORT = 30004
SCRIPT_PORT = 30002
DEFAULT_POSE_FILE = "poses.json"

# RTDE configuration for reading robot state
RTDE_CONFIG = """<?xml version="1.0"?>
<rtde_config>
	<recipe key="out">
		<field name="actual_q" type="VECTOR6D"/>
		<field name="actual_TCP_pose" type="VECTOR6D"/>
		<field name="actual_TCP_speed" type="VECTOR6D"/>
	</recipe>
</rtde_config>"""


class RobotPoseController:
    """Robot pose controller using RTDE for reading and secondary client for commands"""

    def __init__(self, host, rtde_port=30004, script_port=30002):
        self.host = host
        self.rtde_port = rtde_port
        self.script_port = script_port

        # Setup RTDE connection for reading
        print(f"Connecting RTDE to {host}:{rtde_port}...")
        self.rtde = rtde.RTDE(host, rtde_port)
        self.rtde.connect()
        self.rtde.get_controller_version()

        # Parse RTDE config
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(RTDE_CONFIG)
            config_file = f.name

        try:
            conf = rtde_config.ConfigFile(config_file)
            output_names, output_types = conf.get_recipe("out")

            # Setup output recipe
            if not self.rtde.send_output_setup(
                output_names, output_types, frequency=125
            ):
                raise RuntimeError("Failed to setup RTDE output recipe")

            # Start data synchronization
            if not self.rtde.send_start():
                raise RuntimeError("Failed to start RTDE synchronization")

            print("RTDE connected and started")
        finally:
            os.unlink(config_file)

        # Setup secondary client for commands
        print(f"Connecting to secondary client {host}:{script_port}...")
        self.script_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.script_socket.settimeout(2)
        self.script_socket.connect((host, script_port))
        print("Secondary client connected")

    def _update_state(self):
        """Update internal state from RTDE"""
        state = self.rtde.receive()
        if state is None:
            return False
        self._state = state
        return True

    def _send_script(self, script):
        """Send URScript command to robot"""
        script_bytes = script.encode("utf-8") + b"\n"
        self.script_socket.send(script_bytes)
        time.sleep(0.01)

    def get_pose(self):
        """Get current TCP pose as [x, y, z, rx, ry, rz]"""
        self._update_state()
        pose = list(self._state.actual_TCP_pose)
        return [float(v) for v in pose]

    def get_joints(self):
        """Get current joint positions in radians"""
        self._update_state()
        joints = list(self._state.actual_q)
        return [float(j) for j in joints]

    def movel(self, pose, acc=0.3, vel=0.05, wait=True):
        """Move linearly to pose [x, y, z, rx, ry, rz]"""
        x, y, z, rx, ry, rz = [float(v) for v in pose[:6]]
        script = f"movel(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={acc}, v={vel})"
        self._send_script(script)
        if wait:
            self._wait_for_move()

    def movej(self, joints, acc=0.8, vel=0.2, wait=True):
        """Move to joint positions"""
        joints_list = [float(j) for j in joints]
        script = f"movej({joints_list}, a={acc}, v={vel})"
        self._send_script(script)
        if wait:
            self._wait_for_move()

    def _wait_for_move(self, timeout=10):
        """Wait for move to complete"""
        start_time = time.time()
        initial_pose = self.get_pose()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            current_pose = self.get_pose()
            # Check if position changed significantly
            if (
                np.linalg.norm(np.array(current_pose[:3]) - np.array(initial_pose[:3]))
                < 0.001
            ):
                # Check velocity
                self._update_state()
                if np.linalg.norm(self._state.actual_TCP_speed[:3]) < 0.01:
                    return
        print("Warning: Move timeout")

    def close(self):
        """Close connections"""
        if hasattr(self, "rtde"):
            self.rtde.send_pause()
            self.rtde.disconnect()
        if hasattr(self, "script_socket"):
            self.script_socket.close()


def load_poses_file(filename):
    """Load poses from JSON file"""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading poses file: {e}")
        return {}


def save_poses_file(poses, filename):
    """Save poses to JSON file"""
    with open(filename, "w") as f:
        json.dump(poses, f, indent=2)


def format_pose(pose):
    """Format pose as string"""
    return f"[{pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}, {pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}]"


def format_joints(joints, show_degrees=True):
    """Format joint positions as string with optional degree conversion"""
    joint_str = f"[{joints[0]:.6f}, {joints[1]:.6f}, {joints[2]:.6f}, {joints[3]:.6f}, {joints[4]:.6f}, {joints[5]:.6f}]"
    if show_degrees:
        degrees = [math.degrees(j) for j in joints]
        joint_str += f"  (degrees: [{degrees[0]:.2f}°, {degrees[1]:.2f}°, {degrees[2]:.2f}°, {degrees[3]:.2f}°, {degrees[4]:.2f}°, {degrees[5]:.2f}°])"
    return joint_str


def explain_motion_limits():
    """Print explanation of motion parameter limits"""
    print("\n" + "=" * 80)
    print("MOTION PARAMETER LIMITS")
    print("=" * 80)
    print("For movel (Cartesian/TCP motion):")
    print("  - Velocity (v): in m/s, typical max ~1.0 m/s")
    print("  - Acceleration (a): in m/s², typical max ~2.5 m/s²")
    print("\nFor movej (Joint space motion):")
    print("  - Velocity (v): in rad/s, typical max ~3.14 rad/s (180°/s)")
    print("  - Acceleration (a): in rad/s², typical max ~14 rad/s² (800°/s²)")
    print("\nNote: Actual limits depend on:")
    print("  - Robot model (UR3, UR5, UR10, UR16, etc.)")
    print("  - Payload mass and center of gravity")
    print("  - Joint configuration and current pose")
    print("  - Safety settings and motion version")
    print("  - The controller may scale down values that exceed safe limits")
    print("=" * 80 + "\n")


def explain_joint_positions():
    """Print explanation of joint positions"""
    print("\n" + "=" * 80)
    print("POSE vs JOINT POSITIONS")
    print("=" * 80)
    print("POSE [x, y, z, rx, ry, rz]:")
    print("  - Cartesian position and orientation of the TCP (Tool Center Point)")
    print("  - x, y, z: Position in meters")
    print("  - rx, ry, rz: Orientation as axis-angle (rotation vector in radians)")
    print("  - Robot uses inverse kinematics to find joint angles")
    print("  - Multiple joint configurations can reach the same pose!")
    print("    (e.g., elbow up vs elbow down)")
    print("\nJOINT POSITIONS [j1, j2, j3, j4, j5, j6]:")
    print("  - Joint angles in radians for each of the 6 robot joints:")
    print("    Joint 1 (Base):     Rotation around base axis")
    print("    Joint 2 (Shoulder): Up/down movement of shoulder")
    print("    Joint 3 (Elbow):    Up/down movement of elbow")
    print("    Joint 4 (Wrist 1):  Rotation of first wrist joint")
    print("    Joint 5 (Wrist 2):  Rotation of second wrist joint")
    print("    Joint 6 (Wrist 3):  End effector rotation")
    print("  - Unambiguous: exact arm configuration")
    print("  - If tool/payload changes, same joints = different TCP pose")
    print("\nWHICH ONE DO YOU NEED?")
    print("  - You only need ONE (pose OR joints), not both!")
    print(
        "  - Use POSE when: You care about where the tool is, not how the arm gets there"
    )
    print("  - Use JOINTS when: You need the exact same arm configuration")
    print("    (e.g., to avoid singularities, ensure elbow up/down, etc.)")
    print("=" * 80 + "\n")


def parse_pose_string(pose_str):
    """Parse pose from string like '0.3, 0.2, 0.4, 0, 3.14, 0'"""
    try:
        values = [float(x.strip()) for x in pose_str.split(",")]
        if len(values) != 6:
            raise ValueError("Pose must have 6 values")
        return values
    except ValueError as e:
        raise ValueError(f"Invalid pose format: {e}")


def validate_motion_params(acc, vel, use_joints=False):
    """Validate and warn about motion parameters"""
    warnings = []

    if use_joints:
        # movej: acceleration in rad/s², velocity in rad/s
        if acc > 14.0:
            warnings.append(
                f"Warning: Acceleration {acc} rad/s² exceeds typical max (~14 rad/s² for movej)"
            )
        if vel > 3.14:
            warnings.append(
                f"Warning: Velocity {vel} rad/s exceeds typical max (~3.14 rad/s = 180°/s for movej)"
            )
        if acc < 0.1:
            warnings.append(
                f"Warning: Very low acceleration {acc} rad/s² may result in very slow motion"
            )
        if vel < 0.1:
            warnings.append(
                f"Warning: Very low velocity {vel} rad/s may result in very slow motion"
            )
    else:
        # movel: acceleration in m/s², velocity in m/s
        if acc > 2.5:
            warnings.append(
                f"Warning: Acceleration {acc} m/s² exceeds typical max (~2.5 m/s² for movel)"
            )
        if vel > 1.0:
            warnings.append(
                f"Warning: Velocity {vel} m/s exceeds typical max (~1.0 m/s for movel)"
            )
        if acc < 0.1:
            warnings.append(
                f"Warning: Very low acceleration {acc} m/s² may result in very slow motion"
            )
        if vel < 0.01:
            warnings.append(
                f"Warning: Very low velocity {vel} m/s may result in very slow motion"
            )

    for warning in warnings:
        print(warning)

    return len(warnings) == 0


def get_pose(robot, filename, pose_name="current", force=False, explain=False):
    """Get current pose and save to file"""
    if explain:
        explain_joint_positions()

    print("Getting current pose...")
    pose = robot.get_pose()
    joints = robot.get_joints()

    # Format pose for output
    pose_str = format_pose(pose)
    joints_str = format_joints(joints)
    print(f"Current pose (Cartesian): {pose_str}")
    print(f"  x, y, z: Position in meters")
    print(f"  rx, ry, rz: Orientation as axis-angle (rotation vector)")
    print(f"Current joints: {joints_str}")
    print(
        f"  Joints 1-6: Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3 (angles in radians)"
    )

    # Load existing poses
    poses = load_poses_file(filename)

    # Check if pose name already exists
    if pose_name in poses and not force:
        response = input(
            f"Pose '{pose_name}' already exists in {filename}. Overwrite? [y/N]: "
        )
        if response.lower() not in ["y", "yes"]:
            print("Cancelled. Pose not saved.")
            return pose

    # Add or update pose (save both, but either can be used independently)
    poses[pose_name] = {
        "pose": pose,
        "joints": joints,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Either 'pose' or 'joints' can be used to move to this position",
    }

    # Save to file
    save_poses_file(poses, filename)
    print(f"Pose '{pose_name}' saved to {filename}")

    return pose


def set_pose(
    robot,
    filename=None,
    pose_name=None,
    pose_str=None,
    acc=0.3,
    vel=0.05,
    prefer_joints=False,
):
    """Move robot to a pose or joint configuration"""
    if pose_str:
        # Parse pose from command line string
        pose = parse_pose_string(pose_str)
        print(f"Moving to pose from command line: {format_pose(pose)}")
        validate_motion_params(acc, vel, use_joints=False)
        robot.movel(pose, acc=acc, vel=vel)
        print("Move completed")
        return True
    elif filename and pose_name:
        # Load pose from file
        poses = load_poses_file(filename)
        if pose_name not in poses:
            print(f"Error: Pose '{pose_name}' not found in {filename}")
            print(f"Available poses: {', '.join(poses.keys())}")
            return False

        pose_data = poses[pose_name]

        # Check if both pose and joints are available
        has_pose = "pose" in pose_data
        has_joints = "joints" in pose_data

        if not has_pose and not has_joints:
            print(f"Error: No pose or joint data found for '{pose_name}'")
            return False

        # Use joints if preferred or if pose not available
        if prefer_joints and has_joints:
            print(
                f"Moving to joint positions for '{pose_name}': {format_joints(pose_data['joints'])}"
            )
            validate_motion_params(acc, vel, use_joints=True)
            robot.movej(pose_data["joints"], acc=acc, vel=vel)
            print("Move completed")
            return True
        elif has_pose:
            print(f"Moving to pose '{pose_name}': {format_pose(pose_data['pose'])}")
            if has_joints:
                print(
                    f"  (Note: Joint data also available, but using pose. Use --joints to use joint positions instead)"
                )
            validate_motion_params(acc, vel, use_joints=False)
            robot.movel(pose_data["pose"], acc=acc, vel=vel)
            print("Move completed")
            return True
        elif has_joints:
            # Only joints available
            print(
                f"Moving to joint positions for '{pose_name}': {format_joints(pose_data['joints'])}"
            )
            validate_motion_params(acc, vel, use_joints=True)
            robot.movej(pose_data["joints"], acc=acc, vel=vel)
            print("Move completed")
            return True
    else:
        print("Error: Must provide either --pose or --file with --pose-name")
        return False


def list_poses(filename, explain=False):
    """List all poses in file"""
    if explain:
        explain_joint_positions()

    poses = load_poses_file(filename)
    if not poses:
        print(f"No poses found in {filename}")
        return

    print(f"Poses in {filename}:")
    print("-" * 80)
    for name, data in poses.items():
        print(f"  {name}:")
        if "pose" in data:
            print(f"    Pose (Cartesian): {format_pose(data['pose'])}")
        if "joints" in data:
            print(f"    Joints: {format_joints(data['joints'])}")
        if "timestamp" in data:
            print(f"    Saved: {data['timestamp']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Get and set robot poses using RTDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-H",
        "--host",
        default=ROBOT_HOST,
        help=f"Robot hostname or IP (default: from UR_HOST env or {ROBOT_HOST})",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=DEFAULT_POSE_FILE,
        help=f"Pose file (default: {DEFAULT_POSE_FILE})",
    )

    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "-g", "--get", action="store_true", help="Get current pose and save to file"
    )
    action_group.add_argument("-s", "--set", action="store_true", help="Move to a pose")
    action_group.add_argument(
        "-l", "--list", action="store_true", help="List all poses in file"
    )

    # Pose specification arguments
    parser.add_argument(
        "-n",
        "--pose-name",
        default="current",
        help="Name for pose when getting, or name to use when setting (default: 'current')",
    )
    parser.add_argument(
        "-p", "--pose", help="Pose as comma-separated values: 'x, y, z, rx, ry, rz'"
    )

    # Movement parameters
    parser.add_argument(
        "-a",
        "--acc",
        type=float,
        default=0.3,
        help="Acceleration: movel in m/s² (max ~2.5), movej in rad/s² (max ~14) (default: 0.3)",
    )
    parser.add_argument(
        "-v",
        "--vel",
        type=float,
        default=0.05,
        help="Velocity: movel in m/s (max ~1.0), movej in rad/s (max ~3.14) (default: 0.05)",
    )

    # Force overwrite
    parser.add_argument(
        "-F",
        "--force",
        action="store_true",
        help="Force overwrite existing pose without confirmation",
    )

    # Explanation
    parser.add_argument(
        "-e",
        "--explain",
        action="store_true",
        help="Show explanation of joint positions and poses",
    )
    parser.add_argument(
        "--explain-limits",
        action="store_true",
        help="Show explanation of motion parameter limits",
    )

    # Prefer joints over pose
    parser.add_argument(
        "-j",
        "--joints",
        action="store_true",
        help="When setting pose, prefer joint positions over Cartesian pose",
    )

    args = parser.parse_args()

    # Show limits explanation if requested
    if args.explain_limits:
        explain_motion_limits()
        if not (args.get or args.set or args.list):
            return  # Just show explanation and exit

    # Handle list action (doesn't need robot connection)
    if args.list:
        list_poses(args.file, args.explain)
        return

    # Connect to robot
    robot = RobotPoseController(args.host)

    try:
        if args.get:
            get_pose(robot, args.file, args.pose_name, args.force, args.explain)
        elif args.set:
            success = set_pose(
                robot,
                args.file,
                args.pose_name,
                args.pose,
                args.acc,
                args.vel,
                args.joints,
            )
            if not success:
                sys.exit(1)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
