import threading
import rospy
from cv_bridge import CvBridge


class Subscriber(rospy.Subscriber):
    def __init__(self, name: str, data_class):
        self.data = None
        self.data_event = threading.Event()
        self.subscriber = rospy.Subscriber(name, data_class, self.__save_data)

    def __save_data(self, data):
        self.data = data
        self.data_event.set()

    def get_data(self):
        self.data_event.wait()
        self.data_event.clear()
        return self.data


class ImageSubscriber(Subscriber):
    def __init__(self, name: str, data_class):
       super().__init__(name, data_class)
       self.bridge = CvBridge()

    def get_data(self):
       data = super().get_data()
       return self.bridge.imgmsg_to_cv2(data, "bgr8")
