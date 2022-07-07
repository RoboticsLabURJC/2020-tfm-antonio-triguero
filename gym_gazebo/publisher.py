import rospy
from geometry_msgs.msg import Twist


class VelocityPublisher(rospy.Publisher):
    def __init__(self, name, subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        super().__init__(name, Twist, subscriber_listener, tcp_nodelay, latch, headers, queue_size)

    def publish(self, x, z):
        twist = Twist()
        twist.linear.x = x
        twist.angular.z = z
        return super().publish(twist)