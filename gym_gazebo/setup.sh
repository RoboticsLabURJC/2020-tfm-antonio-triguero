#!/bin/bash
##########################

INSTALLATION_DIRECTORY=$(pwd)

# [1] - Install Gazebo
printf '[2] - Installing Gazebo...\n'

sudo apt-get install -y curl
curl -sSL http://get.gazebosim.org | sh
echo "source /usr/share/gazebo/setup.sh" >> ~/.bashrc
source ~/.bashrc

# [2] - Install ROS Noetic
printf '[2] - Installing ROS Noetic...\n'

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt-get update -y
sudo apt-get install -y ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update

# [3] - Install gazebo_ros packages
printf '[3] - Installing dependencies...\n'

sudo apt-get install -y ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
