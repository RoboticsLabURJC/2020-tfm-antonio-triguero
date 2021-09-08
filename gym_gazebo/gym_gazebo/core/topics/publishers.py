from gym_gazebo.core.launchers import MasterPortLock
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty


class Publisher:
    def __init__(self, master_port: int):
        self.master_port = master_port
        self.master_port_lock = MasterPortLock()

    def publish(self, *args):
        self.master_port_lock.acquire(self.master_port)

        rospy.init_node('publisher', anonymous=True, log_level=rospy.FATAL)
        self._send_data(*args)

        self.master_port_lock.release()

    def _send_data(self):
        raise NotImplementedError

class ServicePublisher(Publisher):
    def __init__(self, master_port: int, service_name: str, service = Empty):
        super().__init__(master_port)
        self.service_name = service_name
        self.service = service

    def _send_data(self, message = None):
        rospy.wait_for_service(self.service_name)
        self._proxy = rospy.ServiceProxy(self.service_name, self.service)
        return self._proxy(message) if message else self._proxy()

class VelocityPublisher(Publisher):
    def __init__(self, master_port: int, model_name: str):
        super().__init__(master_port)
        self._publisher = rospy.Publisher(model_name, Twist, queue_size=1)

    def _send_data(self, x: float, z: float):
        message = Twist()
        message.linear.x = x
        message.angular.z = z
        self._publisher.publish(message)

class PositionPublisher(ServicePublisher):
    def __init__(self, master_port: int, model_name: str):
        super().__init__(master_port, '/gazebo/set_model_state', SetModelState)
        self.model_name = model_name
        
    def _send_data(self, x: float, y: float, z: float, ox: float, oy: float, oz: float, ow: float):
        message = ModelState()

        message.model_name = self.model_name
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.position.z = z
        message.pose.orientation.x = ox
        message.pose.orientation.y = oy
        message.pose.orientation.z = oz
        message.pose.orientation.w = ow

        super()._send_data(message)