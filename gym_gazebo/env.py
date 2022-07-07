from gym_gazebo.publisher import VelocityPublisher
import roslaunch
import rospy
from sensor_msgs.msg import Image
import cv2 as cv
import numpy as np
import gym
from gym_gazebo.service import ResetService
from gym_gazebo.subscriber import ImageSubscriber


class Env(gym.Env):
  def __init__(self, package, launchfile):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    cli_args = [package, launchfile]
    roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)

    self.launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    self.launch.start()

    rospy.init_node('gym_env', anonymous=True)

    self.reset_service = ResetService()

  def reset(self):
    self.reset_service.call()
  
  def close(self):
    self.launch.shutdown()


class F1ROS(Env):
  def __init__(self, launchfile: str):
    super().__init__("jderobot_assets", launchfile)
    self.img_subs = ImageSubscriber("/F1ROS/cameraL/image_raw", Image)
    self.vel_pub = VelocityPublisher("/F1ROS/cmd_vel")
    self.img = None

  def __new_img(self):
    self.img = self.img_subs.get_data()
    return self.img

  def reset(self):
    super().reset()
    return self.__new_img()

  def step(self, action):
    self.vel_pub.publish(*action)
    img = self.__new_img()
    return self._get_obs(img), self._get_done(img), self._get_reward(img), self._get_info(img)

  def render(self):
    if self.img is not None:
      cv.imshow('image', self.img)
      cv.waitKey(1)

  def _get_obs(self):
    raise NotImplementedError

  def _get_done(self):
    raise NotImplementedError

  def _get_info(self):
    return {}

  def _get_reward(self):
    raise NotImplementedError
