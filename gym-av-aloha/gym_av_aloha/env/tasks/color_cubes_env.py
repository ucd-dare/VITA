from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.env.sim_config import XML_DIR
import numpy as np
import os
from gymnasium import spaces


class ColorCubesEnv(AVAlohaEnv):
    XML = os.path.join(XML_DIR, 'task_color_cubes.xml')
    LEFT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    LEFT_GRIPPER_POSE = 1
    RIGHT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    RIGHT_GRIPPER_POSE = 1
    MIDDLE_POSE = [0, -0.6, 0.5, 0, 0.5, 0, 0]
    ENV_STATE_DIM = 21
    PROMPTS = [
        "pick red cube",
        "pick blue cube",
        "pick yellow cube",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_reward = 4
        self._grasp_right = False

        self.red_cube_joint = self.mjcf_root.find('joint', 'red_cube_joint')
        self.blue_cube_joint = self.mjcf_root.find('joint', 'blue_cube_joint')
        self.yellow_cube_joint = self.mjcf_root.find('joint', 'yellow_cube_joint')

        self.distractor1_geom = self.mjcf_root.find('geom', 'distractor1')
        self.distractor2_geom = self.mjcf_root.find('geom', 'distractor2')
        self.distractor3_geom = self.mjcf_root.find('geom', 'distractor3')
        self.distractor1_joint = self.mjcf_root.find('joint', 'distractor1_joint')
        self.distractor2_joint = self.mjcf_root.find('joint', 'distractor2_joint')
        self.distractor3_joint = self.mjcf_root.find('joint', 'distractor3_joint')

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
            self.physics.bind(self.red_cube_joint).qpos,
            self.physics.bind(self.blue_cube_joint).qpos,
            self.physics.bind(self.yellow_cube_joint).qpos,
        ])
        return obs
    
    def set_state(self, state, environment_state):
        super().set_state(state, environment_state)
        self.physics.bind(self.red_cube_joint).qpos = environment_state[:7]
        self.physics.bind(self.blue_cube_joint).qpos = environment_state[7:14]
        self.physics.bind(self.yellow_cube_joint).qpos = environment_state[14:21]
        self.physics.forward()

    def get_reward(self):

        if self.prompt == "pick red cube":
            geom_name = "red_cube"
        elif self.prompt == "pick blue cube":
            geom_name = "blue_cube"
        elif self.prompt == "pick yellow cube":
            geom_name = "yellow_cube"
        else:
            raise ValueError(f"Unknown prompt: {self.prompt}")

        touch_left_gripper = False
        touch_right_gripper = False
        cube_touch_table = False

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
            if geom1 == geom_name and geom2.startswith("right"):
                touch_right_gripper = True

            if geom1 == geom_name and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == geom_name:
                cube_touch_table = True

        reward = 0
        if touch_right_gripper:  # touch
            reward = 1
        if touch_right_gripper and not cube_touch_table and not touch_left_gripper:  # grasp
            reward = 2
            self._grasp_right = True
        if touch_right_gripper and touch_left_gripper and not cube_touch_table and self._grasp_right:  # transfer
            reward = 3
        if touch_left_gripper and not cube_touch_table and not touch_right_gripper and self._grasp_right:
            reward = 4
        return reward

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)

        use_distractors = options and options.get('distractors', False)

        # reset physics
        scale=2.5
        x_range = [-0.05*scale, 0.05*scale]
        y_range = [0.1 - scale*0.05, 0.1 + scale*0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        min_distance = 0.08  # Minimum distance to maintain

        # reset distractors
        distractor_geoms = [self.distractor1_geom, self.distractor2_geom, self.distractor3_geom]
        distractor_joints = [self.distractor1_joint, self.distractor2_joint, self.distractor3_joint]
        position = np.array([0.0, 0.0, -1.0])
        quat = np.array([1, 0, 0, 0])
        for geom, joint in zip(distractor_geoms, distractor_joints):
            self.physics.bind(joint).qpos = np.concatenate([position, quat])
            if not use_distractors:
                self.physics.bind(geom).contype = 0
                self.physics.bind(geom).conaffinity = 0
                self.physics.bind(joint).damping = 1e8
            else:
                self.physics.bind(geom).contype = 1
                self.physics.bind(geom).conaffinity = 1
                self.physics.bind(joint).damping = 0

        # add items 
        joints = [self.red_cube_joint, self.blue_cube_joint, self.yellow_cube_joint]
        if use_distractors: joints += distractor_joints
        positions = []
        max_tries = 50
        while len(positions) < len(joints):
            for i in range(max_tries):
                random_pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
                if all(
                    np.linalg.norm(random_pos - pos) > min_distance for pos in positions
                ):
                    positions.append(random_pos)
                    break
            else:
                positions = []

        quats = []
        for _ in range(len(joints)):
            yaw = np.random.uniform(0, 2 * np.pi)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            quats.append(quat)

        # Assign positions to distractors
        for i, joint in enumerate(joints):
            self.physics.bind(joint).qpos = np.concatenate([positions[i], quats[i]])

        self._grasp_right = False

        self.physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info


def main():
    import gym_av_aloha
    from gym_av_aloha.env.sim_env import SIM_DT
    import gymnasium as gym
    import time

    env = gym.make("gym_av_aloha/color-cubes-v1", cameras={}, fps=8.33)

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
        env.unwrapped.MIDDLE_POSE
    ])

    options_list = [
        # {"randomize_light": True},
        {"distractors": True, "prompt": "pick blue cube"},
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

        if i % 1000 == 0:
            env.reset(seed=42, options=options_list[j % len(options_list)])
            j += 1

        i += 1


if __name__ == '__main__':
    main()
