"""Training loop for the geometry-aware segmentation model.

Runs unchanged on a Kaggle GPU kernel and on a laptop; the only difference is
where ``config.out_dir`` points.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig
from .data import Sample, list_samples, load_pair, orientation_target, split_samples
from .losses import combined_loss, dice_score, iou_score, orientation_error_deg
from .models import build_model


class UltrasoundDataset(Dataset):
    """Paired frames with light, anatomy-preserving augmentation.

    The augmentation menu is deliberately short.  Horizontal flip is safe — a
    mirrored scan is a scan of the other leg, and it flips the fascicle
    orientation in a way we can express exactly.  Vertical flip is *not* safe:
    it would put the deep aponeurosis above the superficial one, inverting the
    anatomy the geometry module relies on.  Rotation is likewise avoided,
    because pennation angle is measured against the probe axis.
    """

    def __init__(self, samples: list[Sample], config: TrainConfig, train: bool):
        self.samples = samples
        self.config = config
        self.train = train
        self.rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image, mask = load_pair(sample, self.config.image_size)
        image = image.astype(np.float32) / 255.0

        if self.train:
            rng = np.random.default_rng(self.config.seed * 100003 + index * 7919 + int(time.time_ns() % 65536))
            if rng.random() < 0.5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            # Gain and gamma jitter stand in for scanner preset differences,
            # which is the dominant nuisance variable across these 14 datasets.
            image = np.clip(image * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06), 0, 1)
            image = np.power(image, rng.uniform(0.8, 1.25))

        tensors = {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
        }
        if self.config.uses_orientation:
            tensors["orientation"] = torch.from_numpy(orientation_target(mask))
        return tensors


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, config, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "orientation": 0.0}
    dices, ious, ori_errors = [], [], []
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ori = batch.get("orientation")
        if ori is not None:
            ori = ori.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            autocast = torch.autocast(
                device_type=device.type,
                enabled=config.amp and device.type == "cuda",
                dtype=torch.float16,
            )
            with autocast:
                outputs = model(image)
                loss, parts = combined_loss(
                    outputs, mask, ori,
                    dice_weight=config.dice_weight,
                    bce_weight=config.bce_weight,
                    orientation_weight=config.orientation_weight,
                )

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        for key in totals:
            totals[key] += parts.get(key, 0.0)
        n_batches += 1

        if not train:
            logits = outputs["mask"].float()
            dices.append(dice_score(logits, mask))
            ious.append(iou_score(logits, mask))
            if ori is not None and "orientation" in outputs:
                ori_errors.append(
                    orientation_error_deg(outputs["orientation"].float(), ori, mask)
                )

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if not train:
        metrics["dice"] = float(np.mean(dices)) if dices else float("nan")
        metrics["iou"] = float(np.mean(ious)) if ious else float("nan")
        metrics["orientation_mae_deg"] = (
            float(np.nanmean(ori_errors)) if ori_errors else float("nan")
        )
    return metrics


def train(config: TrainConfig, data_root: Path | None = None) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    samples = list_samples(config.task, data_root)
    train_samples, val_samples = split_samples(
        samples, config.val_fraction, config.seed, task=config.task
    )
    print(f"[{config.task}] {len(samples)} pairs -> {len(train_samples)} train / {len(val_samples)} val", flush=True)

    device = _device()
    print(f"[{config.task}] device: {device}", flush=True)

    train_loader = DataLoader(
        UltrasoundDataset(train_samples, config, train=True),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        UltrasoundDataset(val_samples, config, train=False),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.epochs * max(len(train_loader), 1),
        pct_start=0.25,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"umud_{config.task}.pt"

    history, best_dice = [], -1.0
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, config, device, optimizer, scaler)
        for _ in range(len(train_loader)):
            if scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()
        val_metrics = run_epoch(model, val_loader, config, device)

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 1),
            **{f"train_{k}": round(v, 5) for k, v in train_metrics.items()},
            **{f"val_{k}": round(v, 5) for k, v in val_metrics.items()},
        }
        history.append(record)
        print(
            f"[{config.task}] epoch {epoch:3d}/{config.epochs}  "
            f"train {train_metrics['total']:.4f}  val {val_metrics['total']:.4f}  "
            f"dice {val_metrics['dice']:.4f}  iou {val_metrics['iou']:.4f}  "
            f"ori {val_metrics['orientation_mae_deg']:.2f}deg  ({record['seconds']}s)",
            flush=True,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {**asdict(config), "out_dir": str(config.out_dir)},
                    "epoch": epoch,
                    "val": val_metrics,
                },
                weights_path,
            )

    summary = {
        "task": config.task,
        "best_val_dice": best_dice,
        "final": history[-1] if history else {},
        "history": history,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "weights": str(weights_path),
        "encoder_pretrained": model.encoder.pretrained,
    }
    (out_dir / f"history_{config.task}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{config.task}] best val dice {best_dice:.4f} -> {weights_path}", flush=True)
    return summary
