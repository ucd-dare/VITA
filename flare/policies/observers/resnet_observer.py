import einops
import torch
import torchvision
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from flare.policies.observers.base_observer import BaseObserver


class ResNetObserver(BaseObserver):
    """
    Extracts 1D or 2D embeddings from a ResNet backbone and concat with states.

    When tokenize=False: Returns concatenated 1D features
    When tokenize=True: Returns 2D token features where ResNet features are spatial tokens
                       and state is added as an additional token.
    """

    def __init__(
        self,
        state_key: str = 'observation.state',
        image_keys: list[str] = ['observation.image'],
        resize_shape: tuple[int, int] = (240, 320),
        crop_shape: tuple[int, int] = (224, 308),
        state_dim: int = 21,
        tokenize: bool = False,
    ):
        super().__init__(
            state_key=state_key,
            image_keys=image_keys,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
        )

        self.tokenize = tokenize
        self.state_dim = state_dim

        # ResNet backbone setup
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})

        if self.tokenize:
            self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
            self.state_projector = nn.Linear(state_dim, 512)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def observe(self, batch):
        if self.tokenize:
            return self._observe_tokenized(batch)
        else:
            return self._observe_flattened(batch)

    def _observe_tokenized(self, batch):
        b = batch[self.state_key].shape[0]
        s = batch[self.image_keys[0]].shape[1]
        n = len(self.image_keys)

        images = self.get_images(batch)
        img_features = self.backbone(images)["feature_map"]
        img_features = self.spatial_pool(img_features)

        img_tokens = einops.rearrange(img_features, 'bsn c h w -> bsn (h w) c')

        img_tokens = einops.rearrange(
            img_tokens, '(b s n) hw c -> b (s n hw) c', b=b, s=s, n=n
        )

        states = self.get_states(batch).flatten(start_dim=1)
        state_tokens = self.state_projector(states).unsqueeze(1)

        tokens = torch.cat([state_tokens, img_tokens], dim=1)

        return tokens

    def _observe_flattened(self, batch):
        features = []
        features.append(self.get_states(batch).flatten(start_dim=1))

        b = batch[self.state_key].shape[0]
        s = batch[self.state_key].shape[1]
        n = len(self.image_keys)

        images = self.get_images(batch)
        img_features = self.pool(self.backbone(images)["feature_map"])
        features.append(
            einops.rearrange(
                img_features, '(b s n) c h w -> b (s n c h w)', b=b, s=s, n=n
            )
        )

        features = torch.cat(features, dim=1)
        return features
