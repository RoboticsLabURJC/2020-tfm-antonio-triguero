from gym.envs.registration import register

register(
    id='montmelo-line-v0',
    entry_point='gym_gazebo.envs.f1.env:MontmeloLineEnv',
)
