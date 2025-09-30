import gym_av_aloha
import gymnasium as gym
from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.kinematics.diff_ik import DiffIK, DiffIKConfig
from gym_av_aloha.kinematics.grad_ik import GradIK, GradIKConfig
import numpy as np
import time
from gym_av_aloha.vr.headset import Headset, WebRTCHeadset
from gym_av_aloha.vr.headset_control import HeadsetControl
from gym_av_aloha.vr.headset_utils import HeadsetData, HeadsetFeedback

FPS = 25
CAMERAS = {
    "zed_cam_left": [480, 640],
    "zed_cam_right": [480, 640],
}

class TeleopEnv():
    def __init__(self, env_name, fps=FPS, cameras=CAMERAS):
        self.cameras = cameras
        gym_env = gym.make(f"gym_av_aloha/{env_name}", fps=fps, cameras=cameras)
        self.env: AVAlohaEnv = gym_env.unwrapped

        self.left_controller = GradIK(
            config=GradIKConfig(),
            physics=self.env.physics,
            joints=self.env.left_joints,
            eef_site=self.env.left_eef_site,
        )
        self.right_controller = GradIK(
            config=GradIKConfig(),
            physics=self.env.physics,
            joints=self.env.right_joints,
            eef_site=self.env.right_eef_site,
        )
        self.middle_controller = DiffIK(
            config=DiffIKConfig(),
            physics=self.env.physics,
            joints=self.env.middle_joints,
            eef_site=self.env.middle_eef_site,
        )

    def step(
        self,
        left_pose,
        left_gripper,
        right_pose,
        right_gripper,
        middle_pose,
    ):
        left_joints = self.left_controller.run(
            q=self.env.left_arm.get_joint_positions(),
            target_pos=left_pose[:3, 3],
            target_mat=left_pose[:3, :3],
        )
        right_joints = self.right_controller.run(
            q=self.env.right_arm.get_joint_positions(),
            target_pos=right_pose[:3, 3],
            target_mat=right_pose[:3, :3],
        )
        middle_joints = self.middle_controller.run(
            q=self.env.middle_arm.get_joint_positions(),
            target_pos=middle_pose[:3, 3],
            target_mat=middle_pose[:3, :3],
        )
        action = np.zeros(21)
        action[:6] = left_joints
        action[6] = left_gripper
        action[7:13] = right_joints
        action[13] = right_gripper
        action[14:21] = middle_joints

        # Apply the action to the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        info = self.get_info()
        info['action'] = action

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, self.get_info()
    
    def render_viewer(self):
        self.env.render_viewer()

    def get_info(self):
        return {
            'left_arm_pose': self.env.left_arm.get_eef_pose(),
            'left_gripper': self.env.left_arm.get_gripper_position()[0],
            'right_arm_pose': self.env.right_arm.get_eef_pose(),
            'right_gripper': self.env.right_arm.get_gripper_position()[0],
            'middle_arm_pose': self.env.middle_arm.get_eef_pose(),
        }
    
    def get_obs(self):
        return self.env.get_obs()
    
    def set_state(self, state, environment_state):
        self.env.set_state(state, environment_state)

    def set_prompt(self, prompt):
        self.env.set_prompt(prompt)

    @property
    def prompts(self):
        return self.env.PROMPTS
    
    @property
    def max_reward(self):
        return self.env.max_reward

    

if __name__ == "__main__":
    headset = Headset()
    headset.run_in_thread()
    env = TeleopEnv(
        env_name="peg-insertion-v1",
        cameras=CAMERAS,
    )

    obs, info = env.reset()
    action = {
        'left_pose': info['left_arm_pose'],
        'left_gripper': info['left_gripper'],
        'right_pose': info['right_arm_pose'],
        'right_gripper': info['right_gripper'],
        'middle_pose': info['middle_arm_pose'],
    }

    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_data = HeadsetData()
    headset_control.reset()
    while True:
        start_time = time.time()
        obs, _, _, _, info = env.step(**action)

        # get the headset data
        headset_data = headset.receive_data()
        if headset_data is not None:
            headset_action, feedback = headset_control.run(
                headset_data=headset_data, 
                left_arm_pose=info['left_arm_pose'],
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            ) 
            # start the episode if the user clicks the right button and the headset is in sync
            if headset_data.r_button_one == True and not headset_control.is_running():
                headset_control.start(
                    headset_data, 
                    info['middle_arm_pose'],
                )
            
            if headset_control.is_running():
                action = headset_action

        # send feedback to the headset
        headset.send_feedback(feedback)
        headset.send_left_image(obs['pixels']['zed_cam_left'], 0)
        headset.send_right_image(obs['pixels']['zed_cam_right'], 0)

        end_time = time.time()
        print(f"Step time: {end_time - start_time:.4f} seconds")
        time.sleep(max(0, 1.0 / FPS - (end_time - start_time)))




