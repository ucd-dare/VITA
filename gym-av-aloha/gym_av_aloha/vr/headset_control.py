import numpy as np
from gym_av_aloha.vr.headset_utils import (
    convert_right_to_left_coordinates,
    HeadsetFeedback
)
from gym_av_aloha.utils.transform_utils import (
    align_rotation_to_z_axis, 
    within_pose_threshold, 
    quat2mat,
    pose2mat,
    mat2pose,
    transform_coordinates,
)

class HeadsetControl():
    def __init__(
            self,
            start_ctrl_position_threshold=0.04,
            start_ctrl_rotation_threshold=0.3,
            start_head_position_threshold=0.03,
            start_head_rotation_threshold=0.2,
            ctrl_position_threshold=0.04,
            ctrl_rotation_threshold=0.3,
            head_position_threshold=0.05,
            head_rotation_threshold=0.3
        ):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold
        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold
        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
    
    def reset(self):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

    def is_running(self):
        return self.started

    def start(self, headset_data, middle_arm_pose):
        aligned_headset_pose = np.eye(4)
        aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat))
        aligned_headset_pose[:3, 3] = headset_data.h_pos
        self.start_headset_pose = aligned_headset_pose

        aligned_middle_arm_pose = np.eye(4)
        aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
        aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
        self.start_middle_arm_pose = aligned_middle_arm_pose

        self.started = True

    def run(self, headset_data, left_arm_pose, right_arm_pose, middle_arm_pose):
        headset_pose = pose2mat(headset_data.h_pos, headset_data.h_quat)
        left_pose = pose2mat(headset_data.l_pos, headset_data.l_quat)
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat)

        if self.started:
            start_headset_pose = self.start_headset_pose
            start_middle_arm_pose = self.start_middle_arm_pose
        else:
            aligned_headset_pose = np.eye(4)
            aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(headset_pose[:3, :3])
            aligned_headset_pose[:3, 3] = headset_pose[:3, 3]
            start_headset_pose = aligned_headset_pose

            aligned_middle_arm_pose = np.eye(4)
            aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
            aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
            start_middle_arm_pose = aligned_middle_arm_pose

        # calculate offset between current and saved headset pose
        new_middle_arm_pose = transform_coordinates(headset_pose, start_headset_pose, start_middle_arm_pose)
        new_left_arm_pose = transform_coordinates(left_pose, start_headset_pose, start_middle_arm_pose)
        new_right_arm_pose = transform_coordinates(right_pose, start_headset_pose, start_middle_arm_pose)

        # grippers 
        new_left_gripper = 1 - headset_data.l_index_trigger
        new_right_gripper = 1 - headset_data.r_index_trigger

        action = {
            "left_pose": new_left_arm_pose,
            "left_gripper": new_left_gripper,
            "right_pose": new_right_arm_pose,
            "right_gripper": new_right_gripper,
            "middle_pose": new_middle_arm_pose,
        }

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        unity_middle_arm_pose = transform_coordinates(middle_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_left_arm_pose = transform_coordinates(left_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_right_arm_pose = transform_coordinates(right_arm_pose, start_middle_arm_pose, start_headset_pose)        

        unity_left_arm_pos, unity_left_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_left_arm_pose))
        unity_right_arm_pos, unity_right_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_right_arm_pose))
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_pose))
        
        headOutOfSync = not within_pose_threshold(
            middle_arm_pose[:3, 3],
            middle_arm_pose[:3, :3],
            new_middle_arm_pose[:3, 3], 
            new_middle_arm_pose[:3, :3],
            self.head_position_threshold if self.started else self.start_head_position_threshold,
            self.head_rotation_threshold if self.started else self.start_head_rotation_threshold
        )
        leftOutOfSync = not within_pose_threshold(
            left_arm_pose[:3, 3],
            left_arm_pose[:3, :3],
            new_left_arm_pose[:3, 3], 
            new_left_arm_pose[:3, :3],
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        )
        rightOutOfSync = not within_pose_threshold(
            right_arm_pose[:3, 3],
            right_arm_pose[:3, :3],
            new_right_arm_pose[:3, 3], 
            new_right_arm_pose[:3, :3],
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        )

        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = headOutOfSync
        feedback.left_out_of_sync = leftOutOfSync
        feedback.right_out_of_sync = rightOutOfSync
        feedback.left_arm_position = unity_left_arm_pos
        feedback.left_arm_rotation = unity_left_arm_quat
        feedback.right_arm_position = unity_right_arm_pos
        feedback.right_arm_rotation = unity_right_arm_quat
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat        

        return action, feedback