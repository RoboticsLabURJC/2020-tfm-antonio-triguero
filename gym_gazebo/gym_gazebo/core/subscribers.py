from threading import Event
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

class Subscriber(rospy.topics.Subscriber):
	def __init__(self, name, data_class):
		def set_data(message):
			self._data = message
			self._data_lock.set()
			self._data_lock = Event()
		
		super().__init__(name, data_class, set_data, queue_size=1)
		self._data_lock = Event()
		self._data = None

	def get_data(self):
		self._data_lock.wait()
		return self._data

class ImageSubscriber(Subscriber):
	def __init__(self, name):
		super().__init__(name, Image)
		self._bridge = CvBridge()

	def get_data(self, scale: float):
		data = super().get_data()
		image = self._bridge.imgmsg_to_cv2(data, 'bgr8').astype(np.uint8)

		new_size = tuple((np.array(image.shape[:-1][::-1]) * scale).astype(np.uint8))
		image = cv.resize(image, new_size)

		return image