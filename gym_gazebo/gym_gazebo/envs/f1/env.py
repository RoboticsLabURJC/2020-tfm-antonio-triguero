from gym_gazebo.envs.f1.cars import F1Renault
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2 as cv
from gym_gazebo.envs.env import Env
from gym.spaces import Box, Space

class F1Env(Env):

	def __init__(self, launchfile: str, args={}):
		super().__init__(launchfile=launchfile, args=args)
		self.car = F1Renault(master_port=self._id)
		self.render_pool = ThreadPoolExecutor(max_workers=1)

	def reset(self):
		super().reset()
		return self.car.image()
		
	def render(self, mode='human'):
		def run():
			if mode == 'human':
				image = self.car.image()
				cv.imshow("Camera View", image)
				cv.waitKey(10)
		
		self.render_pool.submit(run)

	def step(self, action: np.array):
		super().step(action)
		self.car.velocity(*action)
		image = self.car.image()
		return image, None, None, None

class MontmeloLineEnv(F1Env):
	def __init__(self, args={}):
		super().__init__('montmelo_line.launch', args=args)

	def reset(self):
		super(F1Env, self).reset()
		# https://quaternions.online/
		positions = [
			(102, 0, 0.004, 0, 0, -0.704, 0.710),
			(108, -6, 0.004, 0, 0, -0.711, -0.703),
			(-71, -31.5, 0.004, 0, 0, -1, 0),
			(105, -3.5, 0.004, 0, 0, -1, 0),
			(100, 20, 0.004, 0, 0, -1, 10),
			(79, 27.75, 0.004, 0, 0, -0.990, -0.200),
			(80, 5, 0.004, 0, 0, -0.853, -0.521),
			(84.2, -8, 0.004, 0, 0, -0.580, 0.815),
			(77, -11.5, 0.004, 0, 0, -0.287, 0.958),
			(16, 25.2, 0.004, 0, 0, -0.965, -0.264),
			(-6.5, 22, 0.004, 0, 0, -0.467, -0.884),
			(-22.7, -1, 0.004, 0, 0, -0.897, 0.443),
			(-32, -18.5, 0.004, 0, 0, 0, -1),
			(-47, -18.5, 0.004, 0, 0, -1, 0),
			(-52, -17.5, 0.004, 0, 0, -0.164, 0.986),
			(-78, -2.5, 0.004, 0, 0, -0.986, -0.168),
			(-75, 8.7, 0.004, 0, 0, -1, 0),
			(-50, 8.7, 0.004, 0, 0, 0, 1),
			(-45, 31.6, 0.004, 0, 0, 0, 1),
			(-85, 31.6, 0.004, 0, 0, -1, 0),
			(-85, -12.85, 0.004, 0, 0, -0.997, -0.082),
			(-85, -12.85, 0.004, 0, 0, -0.089, 0.996),
			(-80, -20, 0.004, 0, 0, -0.732, 0.681),
			(-80, -20, 0.004, 0, 0, -0.682, -0.731),
		]
		position = random.choice(positions)
		if position is not None:
			self.car.position(*position)

		return self.car.image()