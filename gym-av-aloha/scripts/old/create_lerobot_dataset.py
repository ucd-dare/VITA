def main():
    import gym_av_aloha
    import gymnasium as gym
    import time
    import numpy as np
    import os
    from glob import glob
    import pickle
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import cv2

    env = "gym_av_aloha/pour-test-tube-v1"
    traj_dir = os.path.join("results", "sim_pour_test_tube")
    swap_action = False
    repo_id = "iantc104/av_aloha_sim_pour_test_tube"
    description = "pour the red ball inside the right test tube into the left test tube"
    fps = 25
    add_eye_data = False
    
    traj_file = os.path.join(traj_dir, f'episode_0.pkl')
    with open(traj_file, 'rb') as f:
        data = pickle.load(f)
    qpos_len = len(data[0]['qpos'])

    if add_eye_data:
        eye_keys = {
            "left_eye": {"dtype": "float32", "shape": (2,), 'names': None},
            "right_eye": {"dtype": "float32", "shape": (2,), 'names': None},
        }
    else:
        eye_keys = {}

    env = gym.make(env, fps=25)
    observation, info = env.reset(seed=42)

    dataset = LeRobotDataset.create(
        repo_id,
        fps,
        root=f'outputs/{repo_id}',
        features = {
            "action": {"dtype": "float32", "shape": (21,), 'names': None},
            "observation.state": {"dtype": "float32", "shape": (21,), 'names': None},
            "observation.environment_state": {"dtype": "float32", "shape": (env.unwrapped.ENV_STATE_DIM,), 'names': None},
            "observation.qpos": {"dtype": "float32", "shape": (qpos_len,), 'names': None},
            "observation.images.zed_cam_left" : {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']},
            "observation.images.zed_cam_right" : {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']},
            "observation.images.wrist_cam_left" : {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']},
            "observation.images.wrist_cam_right" : {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']},
            "observation.images.overhead_cam" : {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']},
            "observation.images.worms_eye_cam" : {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']},
            **eye_keys,
        },
        image_writer_threads=4 * 2
    )

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
        env.unwrapped.MIDDLE_POSE
    ])

    num_episodes = len(glob(os.path.join(traj_dir, 'episode_*.pkl')))

    while dataset.num_episodes < num_episodes:
        i = dataset.num_episodes
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

            env.unwrapped.set_qpos(qpos)
            env.unwrapped.get_reward()
            obs = env.unwrapped.get_obs()

            if add_eye_data:
                eye_dict = {
                    "left_eye": frame['left_eye'].astype(np.float32),
                    "right_eye": frame['right_eye'].astype(np.float32),
                }
            else:
                eye_dict = {}

            frame = {
                "action": action.astype(np.float32),
                "observation.state": obs['agent_pos'].astype(np.float32),
                "observation.environment_state": obs['environment_state'].astype(np.float32),
                "observation.qpos": qpos.astype(np.float32),
                **{f"observation.images.{k}": v for k,v in obs['pixels'].items()},
                'task': description,
                **eye_dict,
            }

            # left_img = cv2.cvtColor(frame["observation.images.zed_cam_left"], cv2.COLOR_BGR2RGB)
            # right_img = cv2.cvtColor(frame["observation.images.zed_cam_right"], cv2.COLOR_BGR2RGB)
            # left_eye_x = (frame['left_eye'][0] + 1)/2 * 640
            # left_eye_y = (frame['left_eye'][1] + 1) /2 * 480
            # right_eye_x = (frame['right_eye'][0] + 1) /2 * 640
            # right_eye_y = (frame['right_eye'][1] + 1) /2 * 480
            # # Draw left eye coordinates on the left image
            # left_img = cv2.circle(left_img, (int(left_eye_x), int(left_eye_y)), radius=5, color=(0, 255, 0), thickness=-1)
            # # Draw right eye coordinates on the right image
            # right_img = cv2.circle(right_img, (int(right_eye_x), int(right_eye_y)), radius=5, color=(0, 255, 0), thickness=-1)

            # # Update the frame with the modified images
            # frame["observation.images.zed_cam_left"] = left_img
            # frame["observation.images.zed_cam_right"] = right_img


            dataset.add_frame(frame)

        if env.unwrapped.get_reward() == env.unwrapped.max_reward:
            print(f"Episode {i}")
        else:
            print(f"Episode {i} did not reach max reward.")

        dataset.save_episode()

    dataset.push_to_hub()


if __name__ == '__main__':
    main()