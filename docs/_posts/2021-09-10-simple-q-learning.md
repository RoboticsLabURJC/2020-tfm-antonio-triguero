---
title: "Week 5-6. Simple Q learning"
date: 2021-09-10T19:28:00-04:00
categories:
  - logbook
tags:
  - log
  - update
---

As a first approach to the problem at hand, I proceed to build a new environment from the original MontmeloLine environment. In this new environment the action space is reduced to only modeling the angular velocity while the linear velocity is kept constant so that it is low enough to be simpler. The observation space is reduced to the separation between the center line of the image and the position of the line in the image. In turn, this space is formed by ranges, so that if I am 74 pixels away from the center line of the screen, I will be in the range 60-80 for example. The same happens with the new action space, it is divided into ranges of angular velocities.
So far I find myself with a model that pilots adequately on the straights but not in the curves. To see if this one works in the corners I proceed to place the car at the beginning of a corner instead of at the finish line each time the environment is reset. I will be giving more details on the results throughout this week.