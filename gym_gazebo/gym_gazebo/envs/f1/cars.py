from gym_gazebo.envs.f1.cameras import Camera
from gym_gazebo.core.topics.publishers import PositionPublisher, VelocityPublisher


class Car:
	def __init__(self, model_name: str, camera: Camera, master_port: int):
		self.camera = camera
		self._velocity_publisher = VelocityPublisher(master_port, '/F1ROS/cmd_vel')
		self._position_publisher = PositionPublisher(master_port, model_name)

	def velocity(self, linear_vel: float, angular_vel: float):
		self._velocity_publisher.publish(linear_vel, angular_vel)

	def position(self, x, y, z, ox, oy, oz, ow):
		self._position_publisher.publish(x, y, z, ox, oy, oz, ow)

	def image(self):
		return self.camera.get_image()

class F1Renault(Car):
	def __init__(self, master_port: int):
		camera = Camera('/F1ROS/cameraL/image_raw', 0.25)
		super().__init__('f1_renault', camera, master_port)