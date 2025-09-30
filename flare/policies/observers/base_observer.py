import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseObserver(nn.Module):
    def __init__(
        self,
        state_key: str = 'observation.state',
        image_keys: list[str] = ['observation.image'],
        resize_shape: tuple[int, int] = (240, 320),
        crop_shape: tuple[int, int] = (224, 308),
    ):
        super().__init__()
        self.state_key = state_key
        self.image_keys = image_keys
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

    def get_states(self, batch):
        return batch[self.state_key]

    def get_images(self, batch):
        images = torch.stack([batch[key] for key in self.image_keys], dim=2)
        images = einops.rearrange(images, 'b s n c h w -> (b s n) c h w')

        if images.shape[-2:] != tuple(self.resize_shape):
            images = F.interpolate(images, size=tuple(self.resize_shape), mode='bilinear', align_corners=False)

        ch, cw = self.crop_shape
        _, _, H, W = images.shape
        top = torch.randint(0, H - ch + 1, (1,)).item()
        left = torch.randint(0, W - cw + 1, (1,)).item()
        images = images[..., top:top+ch, left:left+cw]
        return images

    def observe(self, batch):
        raise NotImplementedError("observe() should be implemented.")

    def forward(self, batch):
        features = self.observe(batch)
        return features
