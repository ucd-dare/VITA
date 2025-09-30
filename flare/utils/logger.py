import os
import imageio
import wandb
import logging
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def setup_logging(save_dir, log_file="train.log", debug=False):
    """Configure basic logging to file and console"""
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler
    os.makedirs(save_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(save_dir, log_file))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class FlareLogger:
    """
    FlareLogger handles logging to console, TensorBoard, and WandB logging
    Used during training and evaluation to log metrics, figures, videos, etc.
    """

    def __init__(self, config, log_dir=None):
        """
        Initialize loggers based on configuration
        """
        self.config = config
        self.log_dir = log_dir or config.log_dir

        # Initialize TensorBoard if enabled
        self.use_tensorboard = config.tensorboard.enable
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

        # Initialize WandB if enabled
        self.use_wandb = config.wandb.enable
        if self.use_wandb:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name,
                tags=list(config.wandb.tags),
                notes=config.wandb.notes,
                group=config.wandb.group,
                config=OmegaConf.to_container(config, resolve=True),
            )

    def log_metrics(self, metrics, prefix='', step=None, epoch=None):
        """
        Log metrics to all active loggers

        Args:
            metrics: Dictionary of metric values
            prefix: Prefix for metric names
            step: Step number (defaults to current step counter)
            epoch: Epoch number (optional)
        """

        # Prepend prefix to metric names
        log_data = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            log_data[key] = v

        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            for k, v in log_data.items():
                self.writer.add_scalar(k, v, step)

        # Log to WandB
        if self.use_wandb:
            log_data = {**log_data}
            if epoch is not None:
                log_data['epoch'] = epoch
            wandb.log(log_data, step=step)

    def log_figures(self, figures, prefix='', step=None, epoch=None):
        """
        Log figures to visualization platforms

        Args:
            figures: Dictionary of figure objects
            prefix: Prefix for figure names
            step: Step number (defaults to current step counter)
            epoch: Epoch number (optional)
        """
        for name, fig in figures.items():
            tag = f"{prefix}/{name}" if prefix else name

            # Log to TensorBoard
            if self.use_tensorboard and self.writer is not None:
                self.writer.add_figure(tag, fig, step)

            # Log to WandB
            if self.use_wandb:
                log_data = {tag: wandb.Image(fig)}
                if epoch is not None:
                    log_data['epoch'] = epoch
                wandb.log(log_data, step=step)

        # Close figures to prevent memory leaks
        for fig in figures.values():
            plt.close(fig)

    def log_video_frames(self, videos, prefix='', fps=10, step=None, epoch=None):
        """
        Log videos frames

        Args:
            videos: Dictionary of video arrays
            prefix: Prefix for video names
            fps: Frames per second
            step: Step number (defaults to current step counter)
            epoch: Epoch number (optional)
        """
        for name, video in videos.items():
            tag = f"{prefix}/{name}" if prefix else name

            # Log to TensorBoard
            if self.use_tensorboard and self.writer is not None:
                # Convert to TensorBoard format: [N, T, C, H, W]
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # => [T, C, H, W]
                video_tensor = video_tensor.unsqueeze(0)  # => [1, T, C, H, W]
                self.writer.add_video(tag, video_tensor, step, fps=fps)

            # Log to WandB
            if self.use_wandb:
                log_data = {tag: wandb.Video(video, fps=fps)}
                if epoch is not None:
                    log_data['epoch'] = epoch
                wandb.log(log_data, step=step)

    def log_video_files(self, video_files, prefix='', fps=10, step=None, epoch=None):
        """
        Upload videos files from disk

        Args:
            videos: Dictionary of video file paths
            prefix: Prefix for video names
            fps: Frames per second
            step: Step number (defaults to current step counter)
            epoch: Epoch number (optional)
        """
        for name, file_path in video_files.items():
            tag = f"{prefix}/{name}" if prefix else name

            try:
                # Log to TensorBoard
                if self.use_tensorboard and self.writer is not None:
                    reader = imageio.get_reader('video.mp4', 'ffmpeg')
                    frames = []
                    for frame in reader:
                        frame_tensor = torch.tensor(frame).permute(2, 0, 1) / 255.0  # [C, H, W]
                        frames.append(frame_tensor)
                    reader.close()

                    video_tensor = torch.stack(frames)  # [T, C, H, W]
                    video_tensor = video_tensor.unsqueeze(0)  # [1, T, C, H, W]
                    self.writer.add_video(tag, video_tensor, step, fps=fps)

                # Log to WandB
                if self.use_wandb:
                    log_data = {tag: wandb.Video(file_path, format="mp4")}
                    if epoch is not None:
                        log_data['epoch'] = epoch
                    wandb.log(log_data, step=step)

            except Exception as e:
                logging.warning(f"Failed to log video {file_path}: {e}")

    def log_checkpoint(self, path, epoch, is_best=False):
        """
        Log checkpoint as artifact

        Args:
            path: Path to checkpoint file
            epoch: Epoch number
            is_best: Whether this is the best checkpoint (optional)
        """
        if self.use_wandb and self.config.wandb.upload_checkpoints:
            name = "best" if is_best else f"epoch_{epoch}"

            artifact = wandb.Artifact(
                name=name,
                type="model",
                description=f"Model checkpoint at epoch {epoch}"
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def log_hyperparams(self, params):
        """Log hyperparameters"""
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_hparams(params, {})

        if self.use_wandb:
            wandb.config.update(params)

    def close(self):
        """Close all loggers"""
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()

        if self.use_wandb:
            if wandb.run:
                wandb.finish()
