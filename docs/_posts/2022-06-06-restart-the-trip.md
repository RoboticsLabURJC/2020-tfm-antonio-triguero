---
title: "Restarting"
date: 2021-06-06T22:58:00-04:00
categories:
  - logbook
tags:
  - log
  - update
---

Yes, I have resumed the TFM. Since so much time has passed, I have been forced to review all the technology I was using. Taking advantage of the situation, I have used this restart to further work on the knowledge I had about ROS and Gazebo. I did this because I remembered that in the gym_gazebo system I had implemented there were execution errors and I would like to see why.

As for Gazebo, the task has been simple since it has not been necessary to review much since it is a simulator that integrates with ROS. Since my task here is not to design the worlds that run inside this simulator, what I have done is to focus on its connection with ROS, although this task is something that has been left half done this week since I wanted to review ROS before this.

As far as ROS is concerned, I reviewed how it works underneath and the most common commands although I have stayed at the point of seeing how to implement services and messages. Also, after the latter, I would like to see how exactly the integration with Python (rospy) works, in order to properly refactor my code.

During this research process I have realized that the implementation of the gym_gazebo module is probably not the right one as I believe, with what I have seen so far, that the best option would be to create a ROS package containing the necessary node/s to run the simulation environment and, later on, develop a gym environment that communicates with this/these ROS node/s. I would like to do this since, as I am going to resume this work for next season, I think I have enough time to deliver quality work.
