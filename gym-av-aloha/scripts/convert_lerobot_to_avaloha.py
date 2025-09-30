from gym_av_aloha.datasets.av_aloha_dataset import create_av_aloha_dataset_from_lerobot

def main():
    # episodes = {
    #     "iantc104/av_aloha_sim_peg_insertion_v0": list(range(0, 50)),
    #     "iantc104/av_aloha_sim_thread_needle": list(range(0, 50)),
    # }
    # repo_id = "iantc104/av_aloha_sim"
    # remove_keys = [
    #     "observation.images.wrist_cam_left",
    #     "observation.images.wrist_cam_right",
    #     "observation.images.worms_eye_cam",
    #     "observation.images.overhead_cam",
    # ]
    # image_size = (240, 320)

    episodes = {
        "iantc104/av_aloha_sim_cube_transfer": list(range(0, 100)),
        "iantc104/av_aloha_sim_thread_needle": list(range(0, 100)),
        "iantc104/av_aloha_sim_pour_test_tube": list(range(0, 100)),
        "iantc104/av_aloha_sim_slot_insertion": list(range(0, 100)),
        "iantc104/av_aloha_sim_hook_package": list(range(0, 100)),
    }
    repo_id = "iantc104/av_aloha_sim"
    remove_keys = [
        "observation.images.wrist_cam_left",
        "observation.images.wrist_cam_right",
        "observation.images.worms_eye_cam",
        "observation.images.overhead_cam",
        "observation.environment_state",
    ]
    image_size = (240, 320)

    create_av_aloha_dataset_from_lerobot(
        episodes=episodes,
        repo_id=repo_id,
        remove_keys=remove_keys,
        image_size=image_size,
    )

if __name__ == "__main__":
    main()