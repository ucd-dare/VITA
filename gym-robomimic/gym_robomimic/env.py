
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite
from typing import Optional
from scipy.spatial.transform import Rotation as R

class RoboMimicEnv(gym.Env):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    def __init__(
        self, 
        env_meta,
        lite_physics=False,
        input_type: str="delta",
        shift_wrist_camera: bool=True,
    ):
        super().__init__()
        self.shift_wrist_camera = shift_wrist_camera
        self.n_robots = len(env_meta["env_kwargs"]["robots"])
        env_meta["env_kwargs"]["lite_physics"] = lite_physics
        self.env = robosuite.make(env_meta["env_name"], **env_meta["env_kwargs"])

        # action space
        self.action_dim = self.env.action_spec[0].shape[0]
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float64
        )

        # observation space
        self.obs_spec = self.env.observation_spec()
        self.image_keys = [key for key in self.obs_spec if key.endswith("_image")]
        self.image_dims = [self.obs_spec[key].shape for key in self.image_keys]
        self.state_keys = [key for key in self.obs_spec if key.endswith("_proprio-state")]
        self.state_dim = sum([self.obs_spec[key].shape[0] for key in self.state_keys])
        assert self.state_dim > 0, "No proprioceptive state found in observation spec"
        assert "object-state" in self.obs_spec, "No object-state found in observation spec"
        self.environment_state_dim = self.obs_spec['object-state'].shape[0]
        self.observation_space_dict = {}
        self.observation_space_dict['agent_pos'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float64,
        )
        self.observation_space_dict['environment_state'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.environment_state_dim,),
            dtype=np.float64,
        )
        if len(self.image_keys) > 0:
            self.observation_space_dict['pixels'] = spaces.Dict(
                {
                    camera: spaces.Box(
                        low=0,
                        high=255,
                        shape=dim,
                        dtype=np.uint8,
                    )
                    for camera, dim in zip(self.image_keys, self.image_dims)
                }
            )
        self.observation_space = spaces.Dict(self.observation_space_dict)

        self._fill_in_osc_controllers()
        self.set_input_type(input_type)
        if self.shift_wrist_camera:
            self.move_wrist_cameras()

    def _fill_in_osc_controllers(self):
        self.osc_controllers = []
        for robot in self.env.robots:
            for k, v in robot.composite_controller.part_controllers.items():
                if isinstance(v, robosuite.controllers.parts.arm.osc.OperationalSpaceController):
                    self.osc_controllers.append(v)

    def filter_robosuite_obs(self, obs):
        """
        Filter robosuite observations to return only the relevant keys
        """
        filtered_obs = {
            'environment_state': obs['object-state'],
            'agent_pos': np.concatenate(
                [obs[key] for key in self.state_keys]
            ),
            'pixels': {key: obs[key][::-1] for key in self.image_keys}
        }
        return filtered_obs
    
    # def process_action(self, action):
    #     new_action = []
    #     assert action.shape[0] == 10 * self.n_robots
    #     for i in range(self.n_robots):
    #         robot_action = action[i*10:(i+1)*10]
    #         new_action.extend(robot_action[:3]) # position
    #         if self.osc_controllers[i].input_type == "delta":
    #             new_action.extend(robot_action[3:6])
    #         else:
    #             new_action.extend(np_axis_angle_from_rotation_6d(robot_action[3:9]))
    #         new_action.append(robot_action[9]) # gripper
    #     return np.array(new_action)
    
    def set_input_type(self, input_type):
        assert input_type in ["delta", "absolute"]
        for controller in self.osc_controllers:
            controller.input_type = input_type

        print(f"Set input type to {input_type}")

    def get_absolute_action(self, action):
        absolute_action = []
        for i, controller in enumerate(self.osc_controllers):
            absolute_action.extend(controller.goal_pos)
            absolute_action.extend(R.from_matrix(controller.goal_ori).as_rotvec())
            absolute_action.extend([action[i*7+6]])
        return np.array(absolute_action)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)

        terminated = self.is_success()
        truncated = done
        info = {
            "is_success": terminated,
            "absolute_action": self.get_absolute_action(action),
        }

        return self.filter_robosuite_obs(observation), reward, terminated, truncated, info
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        obs = self.env.reset()
        if self.shift_wrist_camera:
            self.move_wrist_cameras()
        info = {"is_success": False}
        return self.filter_robosuite_obs(obs), info

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        if "model" in state:
            if state.get("ep_meta", None) is not None:
                # set relevant episode information
                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = {}

            self.env.set_ep_meta(ep_meta)
            # this reset is necessary.
            # while the call to env.reset_from_xml_string does call reset,
            # that is only a "soft" reset that doesn't actually reload the model.
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not (robosuite.__version__.split(".")[0] == "1"):
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()

        if "goal" in state:
            self.set_goal(**state["goal"])

        self._fill_in_osc_controllers()

        if self.shift_wrist_camera:
            self.move_wrist_cameras()

    def render(self, height=240, width=304, camera_name="agentview"):
        return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        info = dict(model=xml, states=state)
        return info

    def is_success(self):
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return succ
    
    def set_camera_pose(self, pos, quat, camera_name="robot0_eye_in_hand"):
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        self.env.sim.model.cam_pos[cam_id] = pos
        self.env.sim.model.cam_quat[cam_id] = quat

    def move_wrist_cameras(self, 
        pos=[0.13, 0, 0.025], 
        quat=[0.2126311, 0.6743797, 0.6743797, 0.2126311],
    ):
        wrist_cameras = [x for x in self.env.sim.model.camera_names if "eye_in_hand" in x]
        for camera in wrist_cameras:
            self.set_camera_pose(
                pos=pos,
                quat=quat,
                camera_name=camera
            )
        self.env.sim.forward()