# gym-av-aloha
Gym environment for AV-ALOHA simulation experiments.

# AV ALOHA Simulation Datasets

| Dataset | Eye Data | Episodes | Visualization |
|---------|----------|----------|--------------|
| [AV ALOHA Sim Peg Insertion](https://huggingface.co/datasets/iantc104/av_aloha_sim_peg_insertion) | ✅ | 50 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_peg_insertion&episode=0) |
| [AV ALOHA Sim Cube Transfer](https://huggingface.co/datasets/iantc104/av_aloha_sim_cube_transfer) | ✅ | 200 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_cube_transfer&episode=0) |
| [AV ALOHA Sim Thread Needle](https://huggingface.co/datasets/iantc104/av_aloha_sim_thread_needle) | ✅ | 200 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_thread_needle&episode=0) |
| [AV ALOHA Sim Pour Test Tube](https://huggingface.co/datasets/iantc104/av_aloha_sim_pour_test_tube) | ❌ | 50 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_pour_test_tube&episode=0) |
| [AV ALOHA Sim Hook Package](https://huggingface.co/datasets/iantc104/av_aloha_sim_hook_package) | ❌ | 50 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_hook_package&episode=0) |
| [AV ALOHA Sim Slot Insertion](https://huggingface.co/datasets/iantc104/av_aloha_sim_slot_insertion) | ❌ | 50 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_slot_insertion&episode=0) |

# Data Collection

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Soltanilara/gym_av_aloha.git -b teleoperation
pip install -e ./gym_av_aloha
```

Install additional dependencies:

```bash
pip install asyncio numba google-cloud-firestore
pip install git+https://github.com/huggingface/lerobot.git@bed90e3a41c43758c619dba66158ddd5798d361a
pip install git+https://github.com/ian-chuang/aiortc.git@91cdb627b2510dba80786f9236277f103617c87a
```

## Configuration

Add the following files to `gym_av_aloha/vr/`:

* `serviceAccountKey.json` (for Firebase authentication)
* `signalingSettings.json` (for WebRTC signaling configuration)

## Available Environments

* `peg-insertion-v1`
* `cube-transfer-v1`
* `color-cubes-v1`
* `thread-needle-v1`
* `hook-package-v1`
* `pour-test-tube-v1`
* `slot-insertion-v1`

## Example: Recording Simulation Episodes

Navigate to the `scripts/` directory:

```bash
cd gym_av_aloha/scripts
```

Run the recording script:

```bash
python record_sim_episodes.py \
    --env_name hook-package-v1 \
    --num-episodes 100 \
    --repo-id iantc104/av_aloha_sim_hook_package \
    --root outputs \
    --task "hook package"
```