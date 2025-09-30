import os
import re
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_latest_checkpoint(checkpoint_dir):
    """Get latest checkpoint directory (step_* format)."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for step_* directories
    step_dirs = list(checkpoint_dir.glob("step_*"))
    if not step_dirs:
        logger.warning(f"No step_* directories found in {checkpoint_dir}")
        return None

    # Extract step numbers and find maximum
    step_numbers = []
    for step_dir in step_dirs:
        match = re.search(r'step_(\d+)', step_dir.name)
        if match:
            step_numbers.append((int(match.group(1)), step_dir))

    if not step_numbers:
        logger.warning(f"No valid step directories found in {checkpoint_dir}")
        return None

    # Return directory with highest step number
    latest_step, latest_dir = max(step_numbers, key=lambda x: x[0])
    logger.info(f"Found latest checkpoint: {latest_dir} (step {latest_step})")
    return latest_dir


def get_best_checkpoint(checkpoint_dir, best_name="best_success_rate"):
    """Get best checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for best_* directories
    if best_name == "best":
        # Find any best_* directory
        best_dirs = list(checkpoint_dir.glob("best_*"))
        if not best_dirs:
            return None
        # Prioritize success_rate, then score, then most recent
        for priority in ["best_success_rate", "best_score", "best_reward"]:
            for best_dir in best_dirs:
                if priority in best_dir.name:
                    return best_dir
        return best_dirs[0]  # Return first found if no priority match
    else:
        best_dir = checkpoint_dir / best_name
        return best_dir if best_dir.exists() else None


def parse_checkpoint_patterns(checkpoint_dir, ckpt_pattern):
    """Parse checkpoint pattern into list of checkpoint paths."""
    if isinstance(ckpt_pattern, int):
        step_dir = Path(checkpoint_dir) / f"step_{ckpt_pattern:010d}"
        return [step_dir] if step_dir.exists() else []

    if ckpt_pattern == "latest":
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        return [latest_checkpoint] if latest_checkpoint else []

    if ckpt_pattern.startswith("best"):
        best_ckpt = get_best_checkpoint(checkpoint_dir, best_name=ckpt_pattern)
        return [best_ckpt] if best_ckpt else []

    if ":" in ckpt_pattern:
        parts = ckpt_pattern.split(":")
        if len(parts) not in [2, 3]:
            raise ValueError(f"Invalid slicing syntax: {ckpt_pattern}")

        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1

        checkpoint_dir = Path(checkpoint_dir)
        ckpt_paths = []
        for i in range(start, end + 1, step):
            step_dir = checkpoint_dir / f"step_{i:010d}"
            if step_dir.exists():
                ckpt_paths.append(step_dir)
        return ckpt_paths

    # Try to parse as step number
    try:
        step_num = int(ckpt_pattern)
        step_dir = Path(checkpoint_dir) / f"step_{step_num:010d}"
        return [step_dir] if step_dir.exists() else []
    except ValueError:
        logger.warning(f"Unknown checkpoint pattern: {ckpt_pattern}")
        return []


def get_checkpoint_paths(checkpoint_dir, pattern):
    """Get list of valid checkpoint paths matching pattern."""
    ckpt_list = parse_checkpoint_patterns(checkpoint_dir, pattern)
    if not ckpt_list:
        logger.warning(f"No checkpoints found for pattern: {pattern}")
        return []

    valid_paths = []
    for ckpt_path in ckpt_list:
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue
        # Verify it's a valid checkpoint directory
        if not (ckpt_path / "training_state.pt").exists():
            logger.warning(f"Invalid checkpoint directory (missing training_state.pt): {ckpt_path}")
            continue
        valid_paths.append(ckpt_path)

    return valid_paths
