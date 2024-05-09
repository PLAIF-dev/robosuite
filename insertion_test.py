import numpy as np
import robosuite as suite
print(suite.__file__)

def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def quat2euler(quat):
    """ Convert Quaternions to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    ysqr = qy * qy

    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = np.clip(t2, -1.0, 1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (ysqr + qz * qz)
    Z = np.arctan2(t3, t4)

    return np.stack((X, Y, Z), axis=-1)

# create environment instance
env = suite.make(
    env_name="TwoArmACInsertion", 
    env_configuration="single-arm-plaif-ac",  
    robots=["RB3", "RB3"],
    controller_configs={
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2
        },
    gripper_types=["OnRobotRG2Gripper", "OnRobotRG2Gripper"],  # try with other grippers like "OnRobotRG2Gripper"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_gpu_device_id=0,
)

# reset the environment
obs, done = env.reset(), False
print(env.action_spec)

num_epis = 1000
gripped = False

for i in range(num_epis):
    j = 0
    while not done:
        j += 1
        action = np.random.randn(env.robots[0].dof + env.robots[1].dof) # sample random action

        action[:] = 0
        action[:3] = env._gripper0_to_socket
        action[3] = 0.01
        #action[3:6] = quat2euler(obs['plug_quat'] - env._eef0_xquat)

        action[7:10] = env._gripper1_to_plug


        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
    obs, done = env.reset(), False