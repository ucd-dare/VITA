from gymnasium.envs.registration import register

PICK_PLACE_CAN_CONFIG = {
    "env_name": "PickPlaceCan",
    "env_version": "1.5.0",
    "type": 1,
    "env_kwargs": {
        "hard_reset": False,
        "ignore_done": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "input_ref_frame": "world",
                    "gripper": {"type": "GRIP"},
                }
            },
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 256,
        "camera_widths": 256,
        "reward_shaping": False,
        "lite_physics": False,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand",
        ],
    },
}

LIFT_CONFIG = {
    "env_name": "Lift",
    "env_version": "1.5.0",
    "type": 1,
    "env_kwargs": {
        "hard_reset": False,
        "ignore_done": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "input_ref_frame": "world",
                    "gripper": {"type": "GRIP"},
                }
            },
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 256,
        "camera_widths": 256,
        "reward_shaping": False,
        "lite_physics": False,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand",
        ],
    },
}

NUT_ASSEMBLY_SQUARE_CONFIG = {
    "env_name": "NutAssemblySquare",
    "env_version": "1.5.0",
    "type": 1,
    "env_kwargs": {
        "hard_reset": False,
        "ignore_done": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "input_ref_frame": "world",
                    "gripper": {"type": "GRIP"},
                }
            },
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 256,
        "camera_widths": 256,
        "reward_shaping": False,
        "lite_physics": False,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand",
        ],
    },
}

TOOL_HANG_CONFIG = {
    "env_name": "ToolHang",
    "env_version": "1.5.0",
    "type": 1,
    "env_kwargs": {
        "hard_reset": False,
        "ignore_done": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "input_ref_frame": "world",
                    "gripper": {"type": "GRIP"},
                }
            },
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 256,
        "camera_widths": 256,
        "reward_shaping": False,
        "lite_physics": False,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand",
        ],
    },
}

TWO_ARM_TRANSPORT_CONFIG = {
    "env_name": "TwoArmTransport",
    "env_version": "1.5.0",
    "type": 1,
    "env_kwargs": {
        "hard_reset": False,
        "ignore_done": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "input_ref_frame": "world",
                    "gripper": {"type": "GRIP"},
                }
            },
        },
        "robots": ["Panda", "Panda"],
        "env_configuration": "single-arm-opposed",
        "camera_depths": False,
        "camera_heights": 256,
        "camera_widths": 256,
        "reward_shaping": False,
        "lite_physics": False,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand",
            "robot1_eye_in_hand"
        ],
    },
}

register(
    id="gym_robomimic/can-v0",
    entry_point=f"gym_robomimic.env:RoboMimicEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={
        "env_meta": PICK_PLACE_CAN_CONFIG,
    }
)

register(
    id="gym_robomimic/lift-v0",
    entry_point=f"gym_robomimic.env:RoboMimicEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={
        "env_meta": LIFT_CONFIG,
    }
)

register(
    id="gym_robomimic/square-v0",
    entry_point=f"gym_robomimic.env:RoboMimicEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={
        "env_meta": NUT_ASSEMBLY_SQUARE_CONFIG,
    }
)

register(
    id="gym_robomimic/tool-hang-v0",
    entry_point=f"gym_robomimic.env:RoboMimicEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={
        "env_meta": TOOL_HANG_CONFIG,
    }
)

register(
    id="gym_robomimic/transport-v0",
    entry_point=f"gym_robomimic.env:RoboMimicEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={
        "env_meta": TWO_ARM_TRANSPORT_CONFIG,
    }
)
