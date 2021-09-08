from gym.envs.registration import register

register(
    id='montmelo-line-v0',
    entry_point='gym_gazebo.envs.f1:MontmeloLineEnv',
)
register(
    id='f1-env-v0',
    entry_point='gym_gazebo.envs.f1:F1Env',
)
register(
    id='env-v0',
    entry_point='gym_gazebo.envs.core:Env',
)
