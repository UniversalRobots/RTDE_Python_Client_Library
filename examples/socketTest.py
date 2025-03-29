import socket

robot_ip = "10.0.12.245"
port = 30004  # Replace with the port you want to test

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, port))
    print(f"Connection to port {port} successful!")
except Exception as e:
    print(f"Connection to port {port} failed: {e}")
finally:
    s.close()