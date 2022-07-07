#!/bin/bash
# http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

# Create catkin workspace
mkdir -p ./catkin_ws/src
catkin_make -C ./catkin_ws

# Clone JdeRobot/CustomRobots as package in the workspace
mkdir ./catkin_ws/src/jderobot_assets && cd ./catkin_ws/src/jderobot_assets
git init
git remote add -f origin https://github.com/JdeRobot/CustomRobots.git
git config core.sparseCheckout true
echo 'f1' >> .git/info/sparse-checkout
git pull origin noetic-devel
cd ../..

# Next activate catkin workspace and build jderobot_assets package:
source /devel/setup.bash
catkin_make install