import numpy as np
import robosuite as suite
print(suite.__file__)

# create environment instance
env = suite.make(
    env_name="Stack", # try with other tasks like "Stack" and "Door"
    robots="RB3",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="OnRobotRG2Gripper",  # try with other grippers like "OnRobotRG2Gripper"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
obs, done = env.reset(), False
print(env.action_spec)

num_epis = 1000

for i in range(num_epis):
    j = 0
    while not done:
        j += 1
        action = np.random.randn(env.robots[0].dof) # sample random action
        #action = np.zeros(env.robots[0].dof)
        #action[6] = j / 10
        #action[:6] = env.robots[0].init_qpos[:6]
        #action[7:-1] = env.robots[0].init_qpos[6:]
        #action = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0, -0.96, 1.16, 0, -0.3, 0, 0.0084])
        #action[:] = 0.0
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
    obs, done = env.reset(), False