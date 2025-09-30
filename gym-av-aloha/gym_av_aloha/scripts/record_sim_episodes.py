import numpy as np
import torch

import gym_av_aloha
from gym_av_aloha.vr.headset import Headset, WebRTCHeadset
from gym_av_aloha.vr.headset_control import HeadsetControl
from gym_av_aloha.vr.headset_utils import HeadsetFeedback, HeadsetData
from gym_av_aloha.utils.dataset_utils import interpolate_data
from gym_av_aloha.scripts.teleoperate import TeleopEnv
from gym_av_aloha.env.sim_config import SIM_DT

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import time
import os
import pickle
from tqdm import tqdm
import glob
from termcolor import colored

def prompt(env: TeleopEnv, message):
    while True:
        user_input = input(message).strip()

        if env.prompts and user_input not in env.prompts:
            print(colored(f"Invalid input. Please enter one of: {', '.join(env.prompts)}", 'red'))
            continue

        print(colored(f"You entered: '{user_input}'", 'green'))
        confirm = input("Is this correct? (y/n): ").strip().lower()
        if confirm == 'y':
            return user_input

def run_episode(env: TeleopEnv, headset: Headset, task, episode_idx):
    # reset the environment
    ts, info = env.reset()

    # state variables
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_data = HeadsetData()
    headset_control.reset()
    
    # wait for user to start the episode
    print("Waiting for user to start the episode...")
    while True:
        start_time = time.time()
        headset_data = headset.receive_data()
        if headset_data is not None:
            headset_action, feedback = headset_control.run(
                headset_data, 
                left_arm_pose=info['left_arm_pose'],
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            )
            # start the episode if the user clicks the right button and the headset is in sync
            if headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.left_out_of_sync == False and feedback.right_out_of_sync == False:
                headset_control.start(
                    headset_data, 
                    info['middle_arm_pose'],
                )
                break
            if headset_data.l_button_thumbstick:
                task = prompt(env, "Enter the task name (e.g., 'insert peg'): ")
                print(colored(f"Task set to '{task}'", 'green'))
                for i in range(5):
                    headset.receive_data()
                    time.sleep(0.1)
        
        # send feedback to the headset
        feedback.info = f"Starting Episode {episode_idx}.\nTask: {task}.\nAlign and hold A to start. Press Left Thumbstick to change task."
        headset.send_feedback(feedback)
        headset.send_left_image(ts['pixels']['zed_cam_left'], 0)
        headset.send_right_image(ts['pixels']['zed_cam_right'], 0)

        # Rudimentary time keeping
        time.sleep(max(0, SIM_DT - (time.time() - start_time)))

    # run the episode
    print(f"Starting episode {episode_idx}...")
    env.set_prompt(task)
    data = []
    eye_data = []
    reward = []
    step_idx = 0
    l_h, l_w = env.cameras['zed_cam_left']
    r_h, r_w = env.cameras['zed_cam_right']
    while True:
        start_time = time.time()    
        # Receive data from the headset
        headset_data = headset.receive_data()
        if headset_data is not None:
            # get the action and feedback from the headset control
            headset_action, feedback = headset_control.run(
                headset_data, 
                left_arm_pose=info['left_arm_pose'],
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            )
            if headset_data.r_button_one == False:
                print(f"Episode {episode_idx} ended by user.")
                break
            # save the eye data
            eye_frame = {}
            eye_frame['left_eye'] = headset_data.l_eye.copy()
            eye_frame['right_eye'] = headset_data.r_eye.copy()
            eye_frame['left_eye_frame_id'] = headset_data.l_eye_frame_id
            eye_frame['right_eye_frame_id'] = headset_data.r_eye_frame_id       
            eye_frame['left_eye'][0] = (eye_frame['left_eye'][0] / l_w) * 2 - 1
            eye_frame['left_eye'][1] = (eye_frame['left_eye'][1] / l_h) * 2 - 1
            eye_frame['right_eye'][0] = (eye_frame['right_eye'][0] / r_w) * 2 - 1
            eye_frame['right_eye'][1] = (eye_frame['right_eye'][1] / r_h) * 2 - 1
            eye_data.append(eye_frame)

        # save the data
        frame = {}
        frame['observation.state'] = ts['agent_pos'].copy()
        frame['observation.environment_state'] = ts['environment_state'].copy()
        frame['left_arm_pose'] = info['left_arm_pose'].copy()
        frame['right_arm_pose'] = info['right_arm_pose'].copy()
        frame['middle_arm_pose'] = info['middle_arm_pose'].copy()
        ts, r, _, _, info = env.step(**headset_action)
        frame['action'] = info['action'].copy()
        data.append(frame)
        reward.append(r)

        # send feedback to the headset
        feedback.info = f"Episode {episode_idx}, Step: {str(step_idx).zfill(5)}"
        headset.send_feedback(feedback)
        headset.send_left_image(ts['pixels']['zed_cam_left'], step_idx)
        headset.send_right_image(ts['pixels']['zed_cam_right'], step_idx)

        # Rudimentary time keeping
        time.sleep(max(0, SIM_DT - (time.time() - start_time)))
        step_idx += 1

    if max(reward) != env.max_reward:
        print(f"Episode {episode_idx} failed. Reward: {max(reward)}")
        return [], [], task, False

    return data, eye_data, task, True
      
def confirm_episode(headset: Headset, task, episode_idx):
    # wait for user to redo or do next episode
    feedback = HeadsetFeedback()
    print("Waiting for user to redo or do next episode...")
    while True:
        start_time = time.time()
        headset_data = headset.receive_data()
        if headset_data is not None:
            if headset_data.l_button_one == True:
                return True
            elif headset_data.l_button_two == True:
                return False    
        feedback.info = f"Episode {episode_idx} completed.\nTask: {task}.\nPress X to start next episode or Y to redo. "
        headset.send_feedback(feedback)   
        # Rudimentary time keeping
        time.sleep(max(0, SIM_DT - (time.time() - start_time)))

def combine_data(data, eye_data):
    aligned_left_eye = interpolate_data(
        data = np.array([frame['left_eye'] for frame in eye_data]),
        valid_indices = np.array([frame['left_eye_frame_id'] for frame in eye_data]),
        total_len = len(data),
    )
    aligned_right_eye = interpolate_data(
        data = np.array([frame['right_eye'] for frame in eye_data]),
        valid_indices = np.array([frame['right_eye_frame_id'] for frame in eye_data]),
        total_len = len(data),
    )
    for i, frame in enumerate(data):
        frame['left_eye'] = aligned_left_eye[i]
        frame['right_eye'] = aligned_right_eye[i]
    return data

def collect_data(args):

    num_episodes = args["num_episodes"]
    env_name = args["env_name"]
    repo_id = args["repo_id"]
    root = args["root"]
    task = args["task"]

    headset = WebRTCHeadset()
    headset.run_in_thread()
    
    env = TeleopEnv(
        env_name=env_name,
        cameras={
            "zed_cam_left": [480, 640],
            "zed_cam_right": [480, 640],
        },
    )
    if env.prompts:
        if len(env.prompts) == 1:
            task = env.prompts[0]
            print(colored(f"Using default task: '{task}'", 'green'))
        assert task in env.prompts, \
            f"Task '{task}' is not in the list of available tasks: {env.prompts}. Please choose a valid task."

    # init save dir for trajectories
    traj_save_dir = os.path.join(root, 'trajectories', repo_id)
    os.makedirs(traj_save_dir, exist_ok=True)
    episode_idx = len(glob.glob(os.path.join(traj_save_dir, 'episode_*.pkl')))
    print(colored(f"Collecting {num_episodes} episodes for task '{task}' in environment '{env_name}'", 'green'))
    while episode_idx < num_episodes:
        # run the episode
        data, eye_data, task, ok = run_episode(env, headset, task, episode_idx)
        if not ok:
            continue
        # confirm if the episode is to be saved
        ok = confirm_episode(headset, task, episode_idx)
        if not ok:
            continue
        data = combine_data(data, eye_data)
        # save the episode as pkl
        episode_save_path = os.path.join(traj_save_dir, f'episode_{episode_idx}.pkl')
        with open(episode_save_path, 'wb') as f:
            pickle.dump(
                {
                    'task': task,
                    'data': data,
                }, 
                f,
            )
        episode_idx += 1
    headset.close()
    print("Data collection complete.")

    # create env with all cameras for dataset
    env = TeleopEnv(
        env_name=env_name,
        cameras={
            "zed_cam_left": [480, 640],
            "zed_cam_right": [480, 640],
            "wrist_cam_left": [480, 640],
            "wrist_cam_right": [480, 640],
            "overhead_cam": [480, 640],
            "worms_eye_cam": [480, 640],
        },
    )
    ts, info = env.reset()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=os.path.join(root, repo_id),
        fps=round(1.0/SIM_DT),
        features={
            "observation.images.zed_cam_left": {
                "dtype": "video",
                "shape": (env.cameras['zed_cam_left'][0], env.cameras['zed_cam_left'][1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.zed_cam_right": {
                "dtype": "video",
                "shape": (env.cameras['zed_cam_right'][0], env.cameras['zed_cam_right'][1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_cam_left": {
                "dtype": "video",
                "shape": (env.cameras['wrist_cam_left'][0], env.cameras['wrist_cam_left'][1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_cam_right": {
                "dtype": "video",
                "shape": (env.cameras['wrist_cam_right'][0], env.cameras['wrist_cam_right'][1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.overhead_cam": {
                "dtype": "video",
                "shape": (env.cameras['overhead_cam'][0], env.cameras['overhead_cam'][1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.worms_eye_cam": {
                "dtype": "video",
                "shape": (env.cameras['worms_eye_cam'][0], env.cameras['worms_eye_cam'][1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (21,),
                "names": None,
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (ts['environment_state'].shape[0],),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (21,),
                "names": None,
            },
            "left_eye": {
                "dtype": "float32",
                "shape": (2,),
                "names": None,
            },
            "right_eye": {
                "dtype": "float32",
                "shape": (2,),
                "names": None,
            },
            "left_arm_pose": {
                "dtype": "float32",
                "shape": (16,),
                "names": None,
            },
            "right_arm_pose": {
                "dtype": "float32",
                "shape": (16,),
                "names": None,
            },
            "middle_arm_pose": {
                "dtype": "float32",
                "shape": (16,),
                "names": None,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    while True:
        if dataset.num_episodes >= num_episodes:
            break
        # get the episode index
        episode_idx = dataset.num_episodes
        # open
        episode_path = os.path.join(root, 'trajectories', repo_id, f'episode_{episode_idx}.pkl')
        with open(episode_path, 'rb') as f:
            filedata = pickle.load(f)
            task = filedata['task']
            data = filedata['data']
        # add the episode to the dataset
        for frame in tqdm(data):
            env.set_state(frame['observation.state'], frame['observation.environment_state'])
            ts = env.get_obs()
            f = { k: torch.tensor(v.reshape(-1).copy().astype(np.float32)) for k, v in frame.items() }
            for key in ts['pixels']:
                f[f'observation.images.{key}'] = torch.from_numpy(ts['pixels'][key].copy())
            f['task'] = task
            dataset.add_frame(f)
        dataset.save_episode()

    dataset.push_to_hub(
        private=False,
    )

if __name__ == "__main__":
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Record simulation episodes for AV Aloha.")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--env_name", type=str, default="thread-needle-v1", help="Environment task to run.")
    parser.add_argument("--repo-id", type=str, default="iantc104/av_aloha_sim_peg_insertion_test", help="Repository ID for the dataset.")
    parser.add_argument("--root", type=str, default="outputs", help="Root directory for the dataset.")
    parser.add_argument("--task", type=str, default="pick red cube", help="Task name for the dataset.")
    args = parser.parse_args()
    args_dict = vars(args)
    collect_data(args_dict)