# RTDE client library - Python
Library implements API for Universal Robots RTDE realtime interface.

Full RTDE description is available on [Universal Robots documentation site](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html)
# Project structure
## rtde
RTDE core library

- rtde.py:
RTDE connection management object

- rtde_config.py:
XML configuration files parser

- csv_writer.py, csv_reader.py: 
read and write rtde data objects to text csv files

## examples
- record.py - example of recording realtime data from selected channels.
- example_control_loop.py - example for controlling robot motion. Program moves robot between 2 setpoints.
Copy rtde_control_loop.urp to the robot. Start python script before starting program.
- example_plotting.py - example for using csv_reader, and plotting selected data.

### Running examples
It's recommended to run examples in [virtual environment](https://docs.python.org/3/library/venv.html) or [devcontainer](#using-devcontainer).
```bash
# Example for recording realtime data from the robot
# NOTE: RTDE interface has to be enabled in the robot security settings
cd examples
python record.py -h
python record.py --host 192.168.0.1 --frequency 10
```
# Using robot simulator in a Docker
RTDE can connect from host system or [devcontainer](#using-devcontainer) to controller running in a Docker
when RTDE port 30004 is forwarded.
```bash
# 1. Get latest ursim docker image
docker pull universalrobots/ursim_e-series
# 2. Run docker container: 
docker run --rm -dit -p 30004:30004 -p 5900:5900 -p 6080:6080 universalrobots/ursim_e-series
# 3. open vnc client in browser, and confirm safety: http://localhost:6080/vnc.html?host=docker_ip&port=6080
```

More information about ursim docker image is available on [Dockerhub](https://hub.docker.com/r/universalrobots/ursim_e-series)

# Using robot simulator in VirtualBox
RTDE can connect from host system to controller running in VirtualBox
when RTDE port 30004 is forwarded.
1. Download simulator from [Universal Robots support site](https://www.universal-robots.com/support/)
2. Run simulator in VirtualBox
3. Open menu Devices->Network Settings
4. Open Advanced settings for NAT
5. Open Port Forwarding
6. Add new rule, setting host, and guest ports to 30004. 
Leave host, and guest IP fields blank.

# Using rtde library
Importing locally into project and install
```bash
git clone https://github.com/UniversalRobots/RTDE_Python_Client_Library
pip install RTDE_Python_Client_Library
```
Install latest github commit
```bash
pip install git+https://github.com/UniversalRobots/RTDE_Python_Client_Library.git@main
```
Use [pre-built package](https://github.com/UniversalRobots/RTDE_Python_Client_Library/releases) from github
```bash
pip install git+https://github.com/UniversalRobots/RTDE_Python_Client_Library.git@<version-tag> # vX.X.X
```

Library is compatible with Python 2.7+ and Python 3.6+

# Build release package
```bash
mvn package
```
## Using pre-built package with virtual environment
Create virtual environment, and install wheel package

### Linux & MacOS
```bash
python -m venv venv
source venv/bin/activate
pip install wheel
# Install pre-built rtde package
pip install target/rtde-<version>-release.zip
```

### Windows PowerShell
If Python3 is not installed, then just run python3 from powershell. Microsoft store will launch the installation.

Permission to run scripts in console is needed to activate virtual envrionment.
```PowerShell
set-executionpolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
python -m venv venv
venv/Scripts/Activate.ps1
pip install wheel
# Install pre-built rtde package
pip install target/rtde-<version>-release.zip
```

## Using devcontainer
Open project in VSCode and select to "reopen in devcontainer".
Execute build command from terminal

Running record.py against simulator:
```bash
# first start simulator exposing RTDE port 30004
# docker run --rm -dit -p 30004:30004 -p 5900:5900 -p 6080:6080 universalrobots/ursim_e-series

# in devcontainer terminal type
cd examples
./record.py --host controller --frequency 10 --verbose
```

# Contributor guidelines
Code is formatted with [black](https://github.com/psf/black).
Run code formatter before submitting pull request.

```bash
# open project in devcontainer
python -m black .
```
