import rospy
from std_srvs.srv import Empty

class Service(rospy.ServiceProxy):
    def call(self, *args, **kwds):
        rospy.wait_for_service(self.resolved_name)
        return super().call(*args, **kwds)


class ResetService(Service):
    def __init__(self, persistent=False, headers=None):
        super().__init__("/gazebo/reset_world", Empty, persistent, headers)