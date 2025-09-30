from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.env.sim_config import XML_DIR
import numpy as np
import os
from gymnasium import spaces


class PegInsertionEnv(AVAlohaEnv):
    XML = os.path.join(XML_DIR, 'task_peg_insertion.xml')
    LEFT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    LEFT_GRIPPER_POSE = 1
    RIGHT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    RIGHT_GRIPPER_POSE = 1
    MIDDLE_POSE = [0, -0.6, 0.5, 0, 0.5, 0, 0]
    ENV_STATE_DIM = 14
    PROMPTS = [
        "insert peg into socket"
    ]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_reward = 4

        self.peg_joint = self.mjcf_root.find('joint', 'peg_joint')
        self.hole_joint = self.mjcf_root.find('joint', 'hole_joint')

        self.observation_space_dict['environment_state'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        obs['environment_state'] = np.concatenate([
            self.physics.bind(self.peg_joint).qpos,
            self.physics.bind(self.hole_joint).qpos,
        ])
        return obs
    
    def set_state(self, state, environment_state):
        super().set_state(state, environment_state)
        self.physics.bind(self.peg_joint).qpos = environment_state[:7]
        self.physics.bind(self.hole_joint).qpos = environment_state[7:14]
        self.physics.forward()

    def get_reward(self):
        touch_left_gripper = False
        touch_right_gripper = False
        peg_touch_table = False
        hole_touch_table = False
        peg_touch_hole = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self.physics.data.ncon):
            id_geom_1 = self.physics.data.contact[i_contact].geom1
            id_geom_2 = self.physics.data.contact[i_contact].geom2
            geom1 = self.physics.model.id2name(id_geom_1, 'geom')
            geom2 = self.physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "peg" and geom2.startswith("right"):
                touch_right_gripper = True

            if geom1.startswith("hole-") and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "peg":
                peg_touch_table = True

            if geom1 == "table" and geom2.startswith("hole-"):
                hole_touch_table = True

            if geom1 == "peg" and geom2.startswith("hole-"):
                peg_touch_hole = True

            if geom1 == "peg" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not hole_touch_table):  # grasp both
            reward = 2
        if peg_touch_hole and (not peg_touch_table) and (not hole_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward

    def reset(self, seed=None, options=None) -> tuple:
        observation, info = super().reset(seed=seed, options=options)

        # reset physics
        x_range = [0.1, 0.2]
        y_range = [-0.1, 0.1]
        z_range = [0.01, 0.01]
        ranges = np.vstack([x_range, y_range, z_range])
        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.1, -0.2]
        y_range = [-0.1, 0.1]
        z_range = [0.021, 0.021]
        ranges = np.vstack([x_range, y_range, z_range])
        hole_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        hole_quat = np.array([1, 0, 0, 0])

        self.physics.bind(self.peg_joint).qpos = np.concatenate([peg_position, peg_quat])
        self.physics.bind(self.hole_joint).qpos = np.concatenate([hole_position, hole_quat])

        self.physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info


def main():
    import gym_av_aloha
    from gym_av_aloha.env.sim_env import SIM_DT
    import gymnasium as gym
    import time

    env = gym.make("gym_av_aloha/peg-insertion-v1", cameras={}, fps=8.33)

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
        env.unwrapped.MIDDLE_POSE
    ])

    options_list = [
        {"randomize_light": True},
        {"distractors": True},
        {"adverse": True},
        {}
    ]

    observation, info = env.reset(seed=42, options=options_list[0])

    i = 0
    j = 0
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        env.unwrapped.render_viewer()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

        if i % 10 == 0:
            env.reset(seed=42, options=options_list[j % len(options_list)])
            j += 1

        i += 1


if __name__ == '__main__':
    main()
