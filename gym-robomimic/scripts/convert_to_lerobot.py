def main():
    import gym_robomimic
    import gymnasium as gym
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import os
    import numpy as np

    env = "gym_robomimic/square-v0"
    hdf5_path = "/home/ianchuang/projects/wmil/gym-robomimic/datasets/square/ph/low_dim_v15.hdf5"
    repo_id = "iantc104/robomimic_sim_square"
    description = "pick up the red cube"
    fps = 20
    
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ian/miniconda3/lib/

    # open file
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    num_episodes = len(demos)
    print("hdf5 file {} has {} demonstrations".format(hdf5_path, num_episodes))

    env = gym.make(env, shift_wrist_camera=True)
    observation, info = env.reset()
    
    image_features = {}
    for key, dim in zip(env.unwrapped.image_keys, env.unwrapped.image_dims):
        image_features[f"observation.images.{key}"] = {'dtype': 'video', 'shape': dim, 'names': ['height', 'width', 'channel']}
    
    dataset = LeRobotDataset.create(
        repo_id,
        fps,
        root=f'outputs/{repo_id}',
        features = {
            "action": {"dtype": "float32", "shape": (env.unwrapped.action_dim,), 'names': None},
            "action.delta": {"dtype": "float32", "shape": (env.unwrapped.action_dim,), 'names': None},
            "action.absolute": {"dtype": "float32", "shape": (env.unwrapped.action_dim,), 'names': None},
            "observation.state": {"dtype": "float32", "shape": (env.unwrapped.state_dim,), 'names': None},
            "observation.environment_state": {"dtype": "float32", "shape": (env.unwrapped.environment_state_dim,), 'names': None},
            **image_features,
        },
        image_writer_threads=4 * 2
    )

    for i in range(num_episodes):

        print(f"Processing episode {i}")

        demo_key = demos[i]
        init_state = f["data/{}/states".format(demo_key)][0]
        model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
        initial_state_dict = dict(states=init_state, model=model_xml)
        env.reset_to(initial_state_dict)
        actions = f["data/{}/actions".format(demo_key)][:]            

        success = False
        for action in actions:
            obs, reward, is_success, _, info = env.step(action)
            frame = {
                "action": action.astype(np.float32),
                "action.absolute": info['absolute_action'].astype(np.float32),
                "action.delta": action.astype(np.float32),
                "observation.state": obs['agent_pos'].astype(np.float32),
                "observation.environment_state": obs['environment_state'].astype(np.float32),
                **{f"observation.images.{k}": v for k,v in obs['pixels'].items()},
            }
            dataset.add_frame(frame, task=description)
            success = success or is_success

        if not success:
            print(f"Episode {i} did not succeed")
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        print(f"Episode {i} saved\n\n\n")

    dataset.push_to_hub()


if __name__ == '__main__':
    main()




