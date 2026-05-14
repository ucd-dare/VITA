import torch
import torch.nn.functional as F


def compute_contrastive_loss(image_features, action_features, temperature=0.07):
    # VITA contrastive loss between image and action feautres (InfoNCE)
    # Can provide an additional boost on top of FLD and FLC
    
    # Normalize features
    batch_size = image_features.size(0)
    image_features = F.normalize(image_features, dim=1)
    action_features = F.normalize(action_features, dim=1)

    # Compute similarity matrix
    logits = torch.matmul(image_features, action_features.T) / temperature

    # Symmetric contrastive loss (image-to-action + action-to-image)
    labels = torch.arange(batch_size, device=logits.device)
    loss_i2a = F.cross_entropy(logits, labels)
    loss_a2i = F.cross_entropy(logits.T, labels)

    return (loss_i2a + loss_a2i) / 2
