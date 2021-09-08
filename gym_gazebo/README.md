# Gym Gazebo

<h2>Prerequistes</h2>

1. Ubuntu 20.04<

<h2>Installation</h2>

1. Install ROS Noetic and Gym Gazebo 11 with next command:

```
$ sudo chmod +x setup.sh
$ ./setup.sh
```

If it doesn't work, try with ```sudo``` for executing ```setup.sh``` file.

2. Create Python virtual environment and activate it
3. Move to gym_gazebo directory and install gym gazebo with next line:

```
$ sudo pip install -e .
```

<h2 id="available-envs">Available environments</h2>

| env-id              | preview       |
| ------------------- |:-------------:|
| MontmeloLine-v0     |               |

<h2>Q&A</h2>

- **Q**: How to create an environment?
- **A**: You can create an virtual environment with OpenAi gym make method as next:

```python
import gym

gym.make('gym_gazebo:<env-id>')
```

With ```env-id``` as gym gazebo environment name from [list](#available-envs)
