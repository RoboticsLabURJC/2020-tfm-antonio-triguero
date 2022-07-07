#!/bin/bash
# http://wiki.ros.org/noetic/Installation/Ubuntu#noetic.2FInstallation.2FPostInstall.Tutorials
# http://wiki.ros.org/ROS/Installation/TwoLineInstall/

wget -c https://raw.githubusercontent.com/qboticslabs/ros_install_noetic/master/ros_install_noetic.sh && chmod +x ./ros_install_noetic.sh && ./ros_install_noetic.sh
rm ros_install_noetic.sh

echo ""
echo "#######################################################################################################################"
echo ">>> {Step 8: Dependencies for building packages.}"
echo ""
echo "#######################################################################################################################"

sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install -y python3-rosdep
sudo rosdep init
rosdep update