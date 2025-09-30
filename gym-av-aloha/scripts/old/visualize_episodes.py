def main():
    import gym_av_aloha
    import gymnasium as gym
    import time
    import numpy as np
    import os
    from glob import glob
    import pickle

    env = "gym_av_aloha/pour-test-tube-v1"
    traj_dir = os.path.join("results", "sim_pour_test_tube")
    swap_action = False

    env = gym.make(env, cameras={}, fps=25)
    observation, info = env.reset(seed=42)

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
        env.unwrapped.MIDDLE_POSE
    ])

    num_episodes = len(glob(os.path.join(traj_dir, 'episode_*.pkl')))

    for i in range(num_episodes):
        traj_file = os.path.join(traj_dir, f'episode_{i}.pkl')
        with open(traj_file, 'rb') as f:
            data = pickle.load(f)

        env.reset()
        env.unwrapped.set_qpos(data[0]['qpos'])

        for frame in data:
            

            if swap_action:
                qpos = frame['qpos']
                tmp = frame['ctrl'].copy()
                action = np.concatenate([
                    tmp[1:7],
                    tmp[0:1],
                    tmp[8:14],
                    tmp[7:8],
                    tmp[14:21],
                ])
            else:
                qpos = frame['qpos']
                action = frame['action']

            env.step(action)
            # env.unwrapped.set_qpos(qpos)
            env.unwrapped.get_reward()
            env.unwrapped.render_viewer()
            time.sleep(0.01)

        if env.unwrapped.get_reward() == env.unwrapped.max_reward:
            print(f"Episode {i} reached max reward!")
        else:
            print(f"Episode {i} did not reach max reward.")

        

        

if __name__ == '__main__':
    main()