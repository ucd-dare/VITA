import pathlib
import os
print("sim_config loaded")
# task parameters
XML_DIR = os.path.join(str(pathlib.Path(__file__).parent.resolve()), '../assets')

CAMERAS = {
    "zed_cam_left": [480, 640],
    "zed_cam_right": [480, 640],
    "wrist_cam_left": [480, 640],
    "wrist_cam_right": [480, 640],
    "overhead_cam": [480, 640],
    "worms_eye_cam": [480, 640],
}

RENDER_CAMERA = "zed_cam_left"

# physics parameters
SIM_PHYSICS_DT = 0.002
SIM_DT = 0.04
SIM_PHYSICS_ENV_STEP_RATIO = int(SIM_DT/SIM_PHYSICS_DT)
SIM_DT = SIM_PHYSICS_DT * SIM_PHYSICS_ENV_STEP_RATIO

# robot parameters
AV_STATE_DIM = 21
STATE_DIM = 14
AV_ACTION_DIM = 21
ACTION_DIM = 14

LEFT_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
LEFT_GRIPPER_JOINT_NAMES = ["left_left_finger", "left_right_finger"]
RIGHT_JOINT_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]
RIGHT_GRIPPER_JOINT_NAMES = ["right_right_finger", "right_left_finger"]
MIDDLE_JOINT_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
    "middle_wrist_3_joint",
]
LEFT_ACTUATOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
LEFT_GRIPPER_ACTUATOR_NAME = "left_gripper"
RIGHT_ACTUATOR_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]
RIGHT_GRIPPER_ACTUATOR_NAME = "right_gripper"
MIDDLE_ACTUATOR_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
    "middle_wrist_3_joint",
]
LEFT_EEF_SITE = "left_gripper_control"
RIGHT_EEF_SITE = "right_gripper_control"
MIDDLE_EEF_SITE = "middle_zed_camera_center"

# for hiding the middle arm
MIDDLE_BASE_LINK = "middle_base_link"

LIGHT_NAME = "light"
