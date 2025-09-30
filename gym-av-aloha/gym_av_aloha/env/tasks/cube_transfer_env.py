from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.env.sim_config import XML_DIR
import numpy as np
import os
from gymnasium import spaces


class CubeTransferEnv(AVAlohaEnv):
    XML = os.path.join(XML_DIR, 'task_cube_transfer.xml')
    LEFT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    LEFT_GRIPPER_POSE = 1
    RIGHT_POSE = [0, -0.082, 1.06, 0, -0.953, 0]
    RIGHT_GRIPPER_POSE = 1
    MIDDLE_POSE = [0, -0.6, 0.5, 0, 0.5, 0, 0]
    ENV_STATE_DIM = 7
    PROMPTS = [
        "pick red cube"
    ]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_reward = 4
        self._grasp_right = False

        self.cube_joint = self.mjcf_root.find('joint', 'cube_joint')

        self.distractor1_geom = self.mjcf_root.find('geom', 'distractor1')
        self.distractor2_geom = self.mjcf_root.find('geom', 'distractor2')
        self.distractor3_geom = self.mjcf_root.find('geom', 'distractor3')
        self.adverse_geom = self.mjcf_root.find('geom', 'adverse')
        self.distractor1_joint = self.mjcf_root.find('joint', 'distractor1_joint')
        self.distractor2_joint = self.mjcf_root.find('joint', 'distractor2_joint')
        self.distractor3_joint = self.mjcf_root.find('joint', 'distractor3_joint')
        self.adverse_joint = self.mjcf_root.find('joint', 'adverse_joint')

        self.observation_space_dict['environment_state'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        obs['environment_state'] = self.physics.bind(self.cube_joint).qpos
        return obs

    def set_state(self, state, environment_state):
        super().set_state(state, environment_state)
        self.physics.bind(self.cube_joint).qpos = environment_state
        self.physics.forward()

    def get_reward(self):
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
            if geom1 == "cube" and geom2.startswith("right"):
                touch_right_gripper = True

            if geom1 == "cube" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "cube":
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

        # reset physics
        scale = 1
        x_range = [-0.05*scale, 0.05*scale]
        y_range = [0.1 - scale*0.05, 0.1 + scale*0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        cube_quat = np.array([1, 0, 0, 0])
        self.physics.bind(self.cube_joint).qpos = np.concatenate([cube_position, cube_quat])

        # reset distractors
        distractor_geoms = [self.distractor1_geom, self.distractor2_geom, self.distractor3_geom, self.adverse_geom]
        distractor_joints = [self.distractor1_joint, self.distractor2_joint, self.distractor3_joint, self.adverse_joint]
        position = np.array([0.0, 0.0, -1.0])
        quat = np.array([1, 0, 0, 0])
        for geom, joint in zip(distractor_geoms, distractor_joints):
            self.physics.bind(geom).contype = 0
            self.physics.bind(geom).conaffinity = 0
            self.physics.bind(joint).damping = 1e8
            self.physics.bind(joint).qpos = np.concatenate([position, quat])

        if (options and options.get('distractors', False)):
            distractor_geoms = [self.distractor1_geom, self.distractor2_geom, self.distractor3_geom]
            distractor_joints = [self.distractor1_joint, self.distractor2_joint, self.distractor3_joint]
            for geom, joint in zip(distractor_geoms, distractor_joints):
                self.physics.bind(geom).contype = 1
                self.physics.bind(geom).conaffinity = 1
                self.physics.bind(joint).damping = 0

            # find random positions that are not too close to each other or cube
            distractor_positions = []
            min_distance = 0.08  # Minimum distance to maintain

            distractor_scale = scale * 2
            x_range = [-0.05*distractor_scale, 0.05*distractor_scale]
            y_range = [0.1 - distractor_scale*0.05, 0.1 + distractor_scale*0.05]
            z_range = [0.0, 0.0]
            ranges = np.vstack([x_range, y_range, z_range])

            max_tries = 50
            while len(distractor_positions) < len(distractor_geoms):
                for i in range(max_tries):
                    d_pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
                    if np.linalg.norm(d_pos - cube_position) > min_distance and all(
                        np.linalg.norm(d_pos - dp) > min_distance for dp in distractor_positions
                    ):
                        distractor_positions.append(d_pos)
                        break
                else:
                    distractor_positions = []

            random_quats = []
            for _ in range(3):
                yaw = np.random.uniform(0, 2 * np.pi)
                quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                random_quats.append(quat)

            # Assign positions to distractors
            for i, joint in enumerate(distractor_joints):
                self.physics.bind(joint).qpos = np.concatenate([distractor_positions[i], random_quats[i]])

        if (options and options.get('adverse', False)):
            self.physics.bind(self.adverse_geom).contype = 1
            self.physics.bind(self.adverse_geom).conaffinity = 1
            self.physics.bind(self.adverse_joint).damping = 0

            distractor_scale = scale * 2
            x_range = [-0.05*distractor_scale, 0.05*distractor_scale]
            y_range = [0.1 - distractor_scale*0.05, 0.1 + distractor_scale*0.05]
            z_range = [0.0, 0.0]
            ranges = np.vstack([x_range, y_range, z_range])

            min_distance = 0.08  # Minimum distance to maintain
            while True:
                d_pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
                if np.linalg.norm(d_pos - cube_position) > min_distance:
                    adverse_position = d_pos
                    yaw = np.random.uniform(0, 2 * np.pi)
                    adverse_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                    self.physics.bind(self.adverse_joint).qpos = np.concatenate([adverse_position, adverse_quat])
                    break

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

    env = gym.make("gym_av_aloha/cube-transfer-v1", cameras={}, fps=8.33)

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
