---
title: "Week 5-6. Q Learning for solving follow-line problem"
date: 2021-09-10T19:28:00-04:00
categories:
  - logbook
tags:
  - log
  - update
---

As a first solution of the problem of line follower with car, I suggest to use Q Learning as minimal algorithm that 
can resolve the problem. 
Our environment offer a discrete observation space and a continous action space. The observation space is discrete becuase
the observations are RGB images with integer values between 0 and 255. The action space is cotinous defined by two actions: linear velocity and angular velocity. This actions can be taken at the same time and with any float value. Q Learning is an algorithm that 
estimate the function Q of MDPs (Markov Decision Process). This function return the expected return that we can obtain if we choose the action _a_ at state _s_ and the agent acts according to the policy. The policy suggest the actions that the agent takes in each state. In Q Learning the policy is defined by the action that maximices the Q function. If you want to learn more about Q learning, you can visit next [link](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#key-concepts).
But our problem for applying Q Learning to our problem is that we need a discrete action space. We need to map states to actions. Another problem is that this algorithm need to know all the markov graph (states and possible actions) for updating the Q function.
The first approach that you can think is to discretize the action space and it is easy becuase you can take the max values and min values and divide linear velocity space in parts with the same shape and equals for angular velocity and extract all combinations that you can do. In my case, I use parts of shape 1 for linear velocity and shape 0.1 for angular velocity.
Another problem is the observation space. It is discrete but the number of _classes_ is very high because it is an images with three channels and each channel with integer values between 0 and 255. The number of different states is enormous. So I tried to extract information about the observation and use this information as state. The first approach is to extract the angle between the direction
of the car and the axis of the line. Now we can use angle for define an state. This space is continous but we can discretize it more easy than the crude images. I define batches of shape five degrees for each state. Carefully because the angle has to be negative in one side and positive to another side.