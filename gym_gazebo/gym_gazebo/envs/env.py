import os
from gym import Env
from gym.error import ClosedEnvironmentError, InvalidAction
from gym_gazebo.core.launcher import GazeboROSLauncher, LAUNCHFILES_PATH
from gym_gazebo.core.topics.publishers import ServicePublisher

class Env(Env):
    
    def __init__(self, launchfile: str, args: dict):
        self._launcher = GazeboROSLauncher()
        launchfile_path = os.path.join(LAUNCHFILES_PATH, 'f1', 'launch', launchfile)
        self._id = self._launcher.launch(launchfile_path, args)
        self._closed = False

        self._publishers = {
            'reset': ServicePublisher(master_port=self._id, service_name='/gazebo/reset_simulation'),
            'pause': ServicePublisher(master_port=self._id, service_name='/gazebo/pause_physics'),
            'unpause': ServicePublisher(master_port=self._id, service_name='/gazebo/unpause_physics'),
        }

    def _check_closed(self):
        if self._closed:
            raise ClosedEnvironmentError()

    def _publish(self, id: str):
        self._check_closed()
        self._publishers[id].publish()
		
    def reset(self):
        self._publish('reset')

    def pause(self):
        self._publish('pause')

    def unpause(self):
        self._publish('unpause')

    def close(self):
        self._closed = True
        self._launcher.stop(self._id)
        
    def step(self, action):
        self._check_closed()
        if not self.action_space.contains(action):
            raise InvalidAction()