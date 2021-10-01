from gym_gazebo.envs.f1.cars import F1Renault
import random
import numpy as np
import cv2 as cv
from gym_gazebo.envs.env import Env
from gym.spaces import Box

class ImageProcessor:
	def process(self, image: np.array):
		mask = self.skinny_line(image)
		degrees, radians = self.angle(mask)
		return {
			'mask': mask,
			'angle': {
				'radians': radians,
				'degrees': degrees
			}
		}

	def draw_contours(self, img: np.array):
		mask = np.zeros_like(img)
		contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(mask, contours, 0, 255, 1)
		return mask

	def skinny_line(self, img: np.array):
		# Segment line by color
		mask = cv.inRange(img, (41, 41, 218), (42, 42, 219))
		if not mask.any():
			return mask

		# Draw borders of line
		contours_mask = self.draw_contours(mask)
		rows, _ = contours_mask.nonzero()
		if rows.size < 21:
			return mask

		# Extract center of line
		mask = np.zeros_like(mask)
		cols = np.array([np.mean(contours_mask[row].nonzero()[0][[0, -1]]).astype(int) for row in rows])

		p1, p2 = np.array((rows[0], cols[0])), np.array((rows[20], cols[20]))
		m = (p2[1] - p1[1]) / (p2[0] - p1[0])
		x = img.shape[0]
		if not (np.isnan(m) or np.isnan(p1).any()):
			p3 = (x, int(m * (x - p1[0]) + p1[1]))
			cv.line(mask, tuple(p1[::-1]), tuple(p3[::-1]), 255, 1)
		
		return mask

	def angle(self, mask: np.array):
		rows, cols = mask.nonzero()
		if rows.size > 3:
			pl_1, pl_2 = np.array(list(zip(rows[[0, -1]], cols[[0, -1]])))
			pc_1, pc_2 = np.array(list(zip([mask.shape[0] - 10, mask.shape[0] - 80], [mask.shape[1] // 2] * 2)))
			v1, v2 = (pl_1 - pl_2), (pc_2 - pc_1)
			unit_v1, unit_v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)

			dot_dir = np.dot(unit_v1, unit_v2)
			radians = np.arccos(dot_dir)
			if cols[-1] < mask.shape[1] // 2:
				radians = -radians
			degrees = radians * 180 / np.pi
			return degrees, radians
		return 0, 0


class MontmeloLineEnv(Env):

	def __init__(self, args={}):
		super().__init__(launchfile='montmelo_line.launch', args=args)
		self.car = F1Renault(master_port=self._id)

		# because action space normalization
		self.action_high = np.array([12, 1])
		self.action_low = np.array([0, -1])

		self.action_space = Box(low=-1, high=1, shape=(2,), dtype="float32") # normalized
		self.observation_space=Box(low=0, high=255, shape=self.car.camera.shape, dtype=np.uint8)
		
		self.last_action = None
		self.image_processor = ImageProcessor()

	# OpenAI Gym methods

	def reset(self):
		super().reset()
		return self.car.image() // 255
		
	def render(self, mode='human'):
		distance_color, font_color, lines_color = (255, 0, 0), (0, 0, 0), (255, 255, 255)

		image = self.car.image()
		'''if not info:
			return

		image[info['mask'] == 255] = lines_color
		image[:, image.shape[1] // 2] = lines_color

		middle = image.shape[1] // 2
		if info['distance'] > 0:
			image[-1, :middle] = distance_color
		else:
			image[-1, middle:] = distance_color

		shape = image.shape[:-1][::-1]
		image = cv.resize(image, (shape[0] * 4, shape[1] * 4))

		font_scale, thickness = image.shape[0] * 0.001, 1
		org = (image.shape[0] // 20, image.shape[1] // 20)
		angle = info['angle_radians']
		image = cv.putText(image, f'radians: {angle:.2f}', 
								org, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv.LINE_AA)
		org = (image.shape[0] // 20, image.shape[1] // 12)
		angle = info['angle_degrees']
		image = cv.putText(image, f'degrees: {angle:.2f}', 
								org, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv.LINE_AA)
		org = (image.shape[0] // 20, int(image.shape[1] / 8.5))
		distance = info['distance']
		image = cv.putText(image, f'distance: {distance:.2f}', 
								org, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv.LINE_AA)
		org = (image.shape[0] // 20, int(image.shape[1] / 6.5))
		norm_distance = info['norm_distance']
		image = cv.putText(image, f'normaliced distance: {norm_distance:.2f}', 
								org, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv.LINE_AA)'''

		cv.imshow("Camera View", image)
		cv.waitKey(10)

	def step(self, action):
		super().step(action)

		denormalized = (np.array(action) + 1) * (self.action_high - self.action_low) / 2 + self.action_low
		self.last_action = denormalized
		self.car.velocity(*denormalized)

		image = self.car.image()
		result = self.image_processor.process(image)

		mask = result['mask']
		done = not mask[-1].any()
		
		radians = result['angle']['radians']
		velocity_x = self.last_action[0] / self.action_high[0]
		reward = velocity_x * (np.cos(radians) - np.sin(radians))

		return image, reward, done, result