import os
import signal
import sys
import roslaunch
import socket
import rospy
from contextlib import closing
from threading import Lock


LAUNCHFILES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'launchfiles')


class Singleton(object):
  _instances = {}
  def __new__(class_, *args, **kwargs):
    if class_ not in class_._instances:
        class_._instances[class_] = super(Singleton, class_).__new__(class_, *args, **kwargs)
    return class_._instances[class_]

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class MasterPortLock(Singleton):
    def __init__(self):
        self._lock = Lock()

    def acquire(self, port: int) -> None:
        self._lock.acquire()
        self._url = os.environ['ROS_MASTER_URI'].split(':')
        os.environ['ROS_MASTER_URI'] = ':'.join(self._url[:-1] + [str(port)])

    def release(self) -> None:
        os.environ['ROS_MASTER_URI'] = ':'.join(self._url)
        return self._lock.release()

class GazeboROSLauncher(Singleton):
	def __init__(self):
		self._processes = {}
		self._master_port_lock = MasterPortLock()

		os.environ['GAZEBO_MODEL_PATH'] += os.path.join(LAUNCHFILES_PATH, 'f1', 'models')
		os.environ['GAZEBO_RESOURCE_PATH'] += os.path.join(LAUNCHFILES_PATH, 'f1', 'worlds')
		os.environ['ROSCONSOLE_CONFIG_FILE'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rosconsole.conf')
		os.environ['IGN_IP'] = '127.0.0.1'

		signal.signal(signal.SIGTERM, lambda a, b: self.stop_all() or exit())
		signal.signal(signal.SIGINT, lambda a, b: self.stop_all() or exit())

	def _find_free_port(self):
		with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
			s.bind(('', 0))
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			return s.getsockname()[1]

	def _roslaunch(self, launchfile_path: str, port: int, args: dict):
		uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
		cli_args = [launchfile_path, *[f'{key}:={value}' for key, value in args.items()]]
		roslaunch_args = cli_args[1:]
		roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
		launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, port=port, is_core=True)
		self._master_port_lock.acquire(port)
		print(f'Starting roslaunch process at port {port}...', end='')
		with HiddenPrints():
			launch.start()
			rospy.init_node('gym_gazebo', anonymous=True, log_level=rospy.FATAL)
		self._master_port_lock.release()
		print(' DONE!')
		return launch

	def launch(self, launchfile_path: str, args: dict = {}) -> int:
		port = self._find_free_port()
		launch = self._roslaunch(launchfile_path, port, args)
		self._processes[port] = launch
		return port

	def stop(self, port: int):
		if port not in self._processes.keys():
			return
		with HiddenPrints():
			self._processes[port].shutdown()
		del self._processes[port]
	
	def stop_all(self):
		keys = list(self._processes.keys())
		for port in keys:
			self.stop(port)
