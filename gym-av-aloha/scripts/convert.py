# Convert datasets from LeRobot to AV-ALOHA which is MUCH FASTER for training
# Usage:
# * Listing all available datasets
#       python convert.py -l
# * Converting a single task dataset
#       python convert.py -r iantc104/av_aloha_sim_thread_needle
# * Display help message
#       python convert.py -h

import argparse
from gym_av_aloha.datasets.av_aloha_dataset import create_av_aloha_dataset_from_lerobot


DATASET_CONFIGS = {
    # gym-av-aloha tasks
    "iantc104/av_aloha_sim_cube_transfer": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_thread_needle": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_pour_test_tube": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_slot_insertion": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_hook_package": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    # robomimic tasks
    "iantc104/robomimic_sim_square": {
        "episodes": list(range(0, 174)),
        "remove_keys": ["observation.images.robot0_eye_in_hand"],
        "image_size": (256, 256),
    },
    "iantc104/robomimic_sim_can": {
        "episodes": list(range(0, 191)),
        "remove_keys": ["observation.images.robot0_eye_in_hand"],
        "image_size": (256, 256),
    },
    # pusht
    "lerobot/pusht": {
        "episodes": list(range(0, 206)),
        "remove_keys": [],
        "image_size": (96, 96),
    },
}


def list_datasets():
    print("--- Available Dataset Repository IDs (Repo IDs) ---")
    for repo_id in DATASET_CONFIGS:
        print(f"  - {repo_id}")
    print("--------------------------------------------------")


def convert_dataset(repo_id: str):
    if repo_id not in DATASET_CONFIGS:
        print(f"Error: Repository ID '{repo_id}' not found in configurations.")
        list_datasets()
        return

    config = DATASET_CONFIGS[repo_id]

    episodes_dict = {repo_id: config["episodes"]}

    print(f"--- Converting Dataset: {repo_id} ---")
    print(f"Episodes to process: {len(config['episodes'])}")
    print(f"Keys to remove: {config['remove_keys']}")
    print(f"Target image size: {config['image_size']}")
    print("------------------------------------------")

    create_av_aloha_dataset_from_lerobot(
        episodes=episodes_dict,
        repo_id=repo_id,
        remove_keys=config["remove_keys"],
        image_size=config["image_size"],
    )

    print(f"--- Successfully completed conversion for: {repo_id} ---")


def main():
    parser = argparse.ArgumentParser(
        description="A script to convert AV-ALOHA and Robomimic datasets from Hugging Face.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-l", "--ls",
        action="store_true",
        help="List all available dataset repository IDs (repo IDs).",
    )
    group.add_argument(
        "-r", "--repo",
        type=str,
        metavar="REPO_ID",
        help="Specify the single dataset REPO_ID to convert (e.g., iantc104/robomimic_sim_transport).",
    )

    args = parser.parse_args()

    if args.ls:
        list_datasets()
    elif args.repo:
        convert_dataset(args.repo)


if __name__ == "__main__":
    main()
