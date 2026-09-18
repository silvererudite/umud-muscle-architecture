"""Losses for the segmentation and orientation branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice on the sigmoid probabilities.

    Aponeuroses and fascicle segments occupy 1-3 % of a frame, so a pure BCE
    optimum sits uncomfortably close to predicting background everywhere; Dice
    supplies the gradient that keeps the positive class alive.
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = (probs * target).sum(dims)
    cardinality = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def orientation_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cosine distance between predicted and target doubled-angle directors.

    Both tensors are unit vectors, so ``1 - <pred, target>`` equals
    ``1 - cos(2 dtheta)`` — zero when the predicted fascicle direction matches,
    and blind to the 180-degree flip that carries no anatomical meaning.

    The loss is evaluated only where an annotator actually drew a fascicle:
    orientation is undefined in the surrounding speckle, and supervising it
    there would teach the head to regress the background texture.
    """
    weight = mask.float()
    total = weight.sum()
    if total < 1.0:
        return pred.sum() * 0.0
    cosine = (pred * target).sum(dim=1, keepdim=True)
    return ((1.0 - cosine) * weight).sum() / total


def combined_loss(
    outputs: dict,
    mask_target: torch.Tensor,
    orientation_target: torch.Tensor | None,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    orientation_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    logits = outputs["mask"]
    bce = F.binary_cross_entropy_with_logits(logits, mask_target)
    dice = soft_dice_loss(logits, mask_target)
    loss = bce_weight * bce + dice_weight * dice
    parts = {"bce": bce.detach().item(), "dice": dice.detach().item(), "orientation": 0.0}

    if orientation_target is not None and "orientation" in outputs:
        ori = orientation_loss(outputs["orientation"], orientation_target, mask_target)
        loss = loss + orientation_weight * ori
        parts["orientation"] = ori.detach().item()

    parts["total"] = loss.detach().item()
    return loss, parts


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (2, 3)
    intersection = (pred * target).sum(dims)
    cardinality = pred.sum(dims) + target.sum(dims)
    # Frames with an empty ground-truth mask score 1 when the prediction is also
    # empty, which is the behaviour we want for the handful of blank annotations.
    dice = torch.where(cardinality > 0, 2.0 * intersection / cardinality, torch.ones_like(cardinality))
    return float(dice.mean())


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (2, 3)
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - intersection
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return float(iou.mean())


@torch.no_grad()
def orientation_error_deg(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    """Mean absolute angular error in degrees, on the true (undoubled) angle."""
    weight = mask.float()
    if weight.sum() < 1.0:
        return float("nan")
    cosine = (pred * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    error = 0.5 * torch.acos(cosine) * 180.0 / torch.pi
    return float((error * weight).sum() / weight.sum())
