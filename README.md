# RTDE client library - Python
Library implements API for Universal Robots RTDE realtime interface.

Full RTDE description is available on [Universal Robots support site](https://www.universal-robots.com/support/)
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
It's recommended to run examples in [virtual environment](https://docs.python.org/3/library/venv.html).
Some require additional libraries.
```
python record.py -h
python record.py --host 192.168.0.1 --frequency 10
```
# Using robot simulator in Docker
RTDE can connect from host system to controller running in Docker
when RTDE port 30004 is forwarded.
1. Get latest ursim docker image: docker pull universalrobots/ursim_e-series
2. Run docker container: docker run --rm -dit -p 30004:30004 -p 5900:5900 -p 6080:6080 universalrobots/ursim_e-series
3. open vnc client in browser, and confirm safet: http://localhost:6080/vnc.html?host=docker_ip&port=6080

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
Copy rtde folder python project
Library is compatible with Python 2.7+, and Python 3.6+

# Build release package
```
mvn package
```
## Using with virtual environment
Create virtual environment, and install wheel package
```
python -m venv venv
source venv/bin/activate
pip install wheel
```
Install rtde package
```
pip install target/rtde-<version>-release.zip
```

# Contributor guidelines
Code is formatted with [black](https://github.com/psf/black).
Run code formatter before submitting pull request.

