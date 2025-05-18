import numpy as np
from scipy.spatial.transform import Rotation as R


class URIKSolver:
    def __init__(self, a, d, alpha, w):
        # DH parameters for UR5
        #self.a = [0, -0.425, -0.39225, 0, 0, 0]
        #self.d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        #self.alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
        #going to have dh parameters for UR3 plus it's extension
        self.a = a
        self.d = d
        self.alpha = alpha

        # Weights for selecting closest solution
        #self.weights = np.array([6, 5, 4, 3, 2, 1])
        self.weights = np.array(w)

    def solve(self, posEul, q_previous):
        """
        Solve inverse kinematics for UR5 robot

        Parameters:
        pos (np.array): Desired end-effector position [x, y, z]
        eul (np.array): Desired end-effector orientation as Euler angles (ZYX) [rz, ry, rx]
        q_previous (np.array): Previous joint angles for solution selection

        Returns:
        np.array: Joint angles that achieve the desired pose
        """
        pos = posEul[:3]
        eul = posEul[-3:]
        print(pos)
        print(eul)

        rot = R.from_euler('zyx', eul).as_matrix()
        T06 = np.eye(4)
        T06[:3, :3] = rot
        T06[:3, 3] = pos
        theta1_ = self.calculateTheta1(T06)
        theta5_ = self.calculateTheta5(T06, theta1_)
        theta6_ = self.calculateTheta6(T06, theta1_, theta5_)
        theta3_, P14_, T14_ = self.calculateTheta3(T06, theta1_, theta5_, theta6_)
        theta2_ = self.calculateTheta2(theta3_, P14_)
        theta4_ = self.calculateTheta4(theta2_, theta3_, T14_)
        solutions = self._generate_possible_solutions(theta1_, theta2_, theta3_, theta4_, theta5_, theta6_)
        solution = self._closest_solution(solutions, q_previous)

        return solution

    def calculateTheta1(self, T06):
        """Calculate theta1 solutions"""
        P05 = T06 @ np.array([0, 0, -self.d[5], 1])
        theta1_ = []

        theta1P = np.arctan2(P05[1], P05[0]) + np.pi / 2
        theta1M = theta1P

        if P05[0] != 0 or P05[1] != 0:
            acos_arg = self.d[3] / np.sqrt(P05[0] ** 2 + P05[1] ** 2)
            # Handle potential numerical issues with acos
            acos_arg = np.clip(acos_arg, -1, 1)
            theta1P = theta1P + np.arccos(acos_arg)
            theta1M = theta1M - np.arccos(acos_arg)

        return np.array([theta1P, theta1M])

    def calculateTheta5(self, T06, theta1_):
        """Calculate theta5 solutions"""
        theta5_ = np.zeros(4)
        idx = 0

        for theta1 in theta1_:
            acos_value = (T06[0, 3] * np.sin(theta1) - T06[1, 3] * np.cos(theta1) - self.d[3]) / self.d[5]
            acos_value = np.clip(acos_value, -1, 1)

            if abs(acos_value) > 1:
                raise ValueError('Theta5 cannot be determined. Value inside acos is outside [-1, 1].')

            for sign in [1, -1]:
                theta5_[idx] = sign * np.arccos(acos_value)
                idx += 1

        return theta5_

    def calculateTheta6(self, T06, theta1_, theta5_):
        T60 = np.linalg.inv(T06)
        X60 = T60[:3, 0]
        Y60 = T60[:3, 1]

        theta6_ = np.zeros(4)
        t5 = [0, 2]

        idx = 0
        for t1 in range(len(theta1_)):
            theta6_[idx] = self._calc_theta6_helper(X60, Y60, theta1_[t1], theta5_[t5[0]])
            theta6_[idx + 1] = self._calc_theta6_helper(X60, Y60, theta1_[t1], theta5_[t5[1]])
            idx += 2

        return theta6_

    def _calc_theta6_helper(self, X60, Y60, theta1, theta5):
        """Helper function to calculate theta6"""
        if np.sin(theta5) != 0:
            left_num = -X60[1] * np.sin(theta1) + Y60[1] * np.cos(theta1)
            right_num = X60[0] * np.sin(theta1) - Y60[0] * np.cos(theta1)
            denom = np.sin(theta5)
            return np.arctan2(left_num / denom, right_num / denom)
        else:
            return 0

    def calculateTheta3(self, T06, theta1_, theta5_, theta6_):
        """Calculate theta3 solutions"""
        theta3_ = np.zeros(8)
        P14_ = np.zeros((8, 3))
        T14_ = np.zeros((8, 4, 4))

        idx = 0
        t5 = [0, 2]  # Indices for theta5 solutions

        for t1 in range(len(theta1_)):
            for sign in [1, -1]:
                # First theta5 solution for this theta1
                theta3, P14, T14 = self._calc_theta3_helper(T06, theta1_[t1], theta5_[t5[0]], theta6_[t5[0]])
                theta3_[idx] = sign * theta3
                P14_[idx] = P14
                T14_[idx] = T14
                idx += 1

                # Second theta5 solution for this theta1
                theta3, P14, T14 = self._calc_theta3_helper(T06, theta1_[t1], theta5_[t5[1]], theta6_[t5[1]])
                theta3_[idx] = sign * theta3
                P14_[idx] = P14
                T14_[idx] = T14
                idx += 1

        # Reorder to match MATLAB implementation
        reorder_idx = [0, 2, 1, 3, 4, 6, 5, 7]
        theta3_ = theta3_[reorder_idx]
        P14_ = P14_[reorder_idx]
        T14_ = T14_[reorder_idx]

        return theta3_, P14_, T14_

    def _calc_theta3_helper(self, T06, theta1, theta5, theta6):
        """Helper function to calculate theta3"""
        P14, T14 = self._calculate_P14(T06, theta1, theta5, theta6)
        P14_xz_length = np.linalg.norm([P14[0], P14[2]])

        conditions = [abs(self.a[1] - self.a[2]), abs(self.a[1] + self.a[2])]

        if conditions[0] < P14_xz_length < conditions[1]:
            acos_arg = (P14_xz_length ** 2 - self.a[1] ** 2 - self.a[2] ** 2) / (2 * self.a[1] * self.a[2])
            acos_arg = np.clip(acos_arg, -1, 1)
            theta3 = np.arccos(acos_arg)
        else:
            raise ValueError('Theta3 cannot be determined. Conditions are not met.')

        return theta3, P14, T14

    def _calculate_P14(self, T06, theta1, theta5, theta6):
        """Calculate P14 vector and T14 transformation matrix"""
        T01 = self._dh_to_tform(0, 0, self.d[0], theta1)
        T10 = np.linalg.inv(T01)

        T45 = self._dh_to_tform(self.alpha[3], self.a[3], self.d[4], theta5)
        T54 = np.linalg.inv(T45)

        T56 = self._dh_to_tform(self.alpha[4], self.a[4], self.d[5], theta6)
        T65 = np.linalg.inv(T56)

        T14 = T10 @ T06 @ T65 @ T54
        P14 = T14[:3, 3]

        return P14, T14

    def calculateTheta2(self, theta3_, P14_):
        """Calculate theta2 solutions"""
        theta2_ = np.zeros(8)

        for i in range(len(theta3_)):
            P14_xz_length = np.linalg.norm([P14_[i, 0], P14_[i, 2]])
            asin_arg = -self.a[2] * np.sin(theta3_[i]) / P14_xz_length
            asin_arg = np.clip(asin_arg, -1, 1)

            theta2 = np.arctan2(-P14_[i, 2], -P14_[i, 0]) - np.arcsin(asin_arg)
            theta2_[i] = theta2

        return theta2_

    def calculateTheta4(self, theta2_, theta3_, T14_):
        """Calculate theta4 solutions"""
        theta4_ = np.zeros(8)

        for i in range(len(theta2_)):
            T12 = self._dh_to_tform(self.alpha[0], self.a[0], self.d[1], theta2_[i])
            T21 = np.linalg.inv(T12)

            T23 = self._dh_to_tform(self.alpha[1], self.a[1], self.d[2], theta3_[i])
            T32 = np.linalg.inv(T23)

            T34 = T32 @ T21 @ T14_[i]
            X34 = T34[:3, 0]

            theta4 = np.arctan2(X34[1], X34[0])
            theta4_[i] = theta4

        return theta4_

    def _generate_possible_solutions(self, theta1_, theta2_, theta3_, theta4_, theta5_, theta6_):
        """Generate all possible joint angle solutions"""
        solutions = np.array([
            [theta1_[0], theta2_[0], theta3_[0], theta4_[0], theta5_[0], theta6_[0]],
            [theta1_[0], theta2_[2], theta3_[2], theta4_[2], theta5_[1], theta6_[1]],
            [theta1_[1], theta2_[4], theta3_[4], theta4_[4], theta5_[2], theta6_[2]],
            [theta1_[1], theta2_[6], theta3_[6], theta4_[6], theta5_[3], theta6_[3]],

            [theta1_[0], theta2_[1], theta3_[1], theta4_[1], theta5_[0], theta6_[0]],
            [theta1_[0], theta2_[3], theta3_[3], theta4_[3], theta5_[1], theta6_[1]],
            [theta1_[1], theta2_[5], theta3_[5], theta4_[5], theta5_[2], theta6_[2]],
            [theta1_[1], theta2_[7], theta3_[7], theta4_[7], theta5_[3], theta6_[3]]
        ])

        return solutions

    def _closest_solution(self, solutions, q_previous):
        """Select solution closest to previous joint angles"""
        best_distance = np.inf
        best_solution = solutions[0]

        for sol in solutions:
            distance = np.sum(((sol - q_previous) * self.weights) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_solution = sol

        return best_solution

    def _dh_to_tform(self, alpha, a, d, theta):
        tform = np.eye(4)

        # Row 1
        tform[0, 0] = np.cos(theta)
        tform[0, 1] = -np.sin(theta)
        tform[0, 3] = a

        # Row 2
        tform[1, 0] = np.sin(theta) * np.cos(alpha)
        tform[1, 1] = np.cos(theta) * np.cos(alpha)
        tform[1, 2] = -np.sin(alpha)
        tform[1, 3] = -np.sin(alpha) * d

        # Row 3
        tform[2, 0] = np.sin(theta) * np.sin(alpha)
        tform[2, 1] = np.cos(theta) * np.sin(alpha)
        tform[2, 2] = np.cos(alpha)
        tform[2, 3] = np.cos(alpha) * d

        return tform