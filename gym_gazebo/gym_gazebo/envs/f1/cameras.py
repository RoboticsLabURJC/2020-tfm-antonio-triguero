from gym_gazebo.core.topics.subscribers import ImageSubscriber

class Camera:
	def __init__(self, name: str, scale: float, master_port):
		self._subscriber = ImageSubscriber(name, master_port)

		self.scale = scale
		self.shape = (int(480 * scale), int(640 * scale), 3)

	def get_image(self):
		return self._subscriber.get_data(scale=self.scale)