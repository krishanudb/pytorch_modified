import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets, masks = None):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    if masks:
        masks = torch.flip(masks, [-1])
        return images, masks, targets
    else:
        return images, targets
