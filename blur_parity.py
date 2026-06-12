# SPDX-License-Identifier: Apache-2.0
"""Post-hoc parity evaluation for the exposure-trajectory blur model.

Renders held-out frames from a checkpoint twice — single-instant (sharp)
and K-sample exposure-averaged (parity with the training forward model) —
and reports both PSNRs plus a sharpness statistic, stratified by the
frame's within-readout GT rotation. PSNR against blurred ground truth
penalizes sharp renders exactly where the blur model helps, so the parity
render is the apples-to-apples number while the sharpness gain is the
deliverable. Side-by-side JPEGs (GT | sharp | exposure-averaged) are
written for the mandatory qualitative check.

Run from the 3dgrut repo root with the project venv active:
    python blur_parity.py --ckpt <ours_N/ckpt_N.pt> \
        --scene <window>/colmap --out /tmp/parity --blur-samples 4
"""

import argparse
import json
import os
from dataclasses import replace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy.spatial.transform import Rotation

from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.trainer import Trainer3DGRUT


def _rotation_deg_by_name(sparse_dir):
    def load(path):
        poses = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                qw, qx, qy, qz = (float(v) for v in parts[1:5])
                poses[parts[9]] = Rotation.from_quat([qx, qy, qz, qw])
        return poses

    start = load(os.path.join(sparse_dir, "images.txt"))
    end = load(os.path.join(sparse_dir, "images_end.txt"))
    return {
        name: float(np.degrees((start[name].inv() * end[name]).magnitude()))
        for name in start
        if name in end
    }


def _psnr(pred, gt):
    mse = float(torch.mean((pred - gt) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def _laplacian_var(image_hw3):
    gray = image_hw3.mean(dim=-1)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
    ).reshape(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(gray[None, None], kernel)
    return float(lap.var())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scene", required=True, help="COLMAP root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--blur-samples", type=int, default=4)
    parser.add_argument("--n-slow", type=int, default=6)
    parser.add_argument("--n-fast", type=int, default=6)
    parser.add_argument("--test-split-interval", type=int, default=8)
    parser.add_argument(
        "--override", action="append", default=[], help="extra hydra overrides"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.abspath(config_dir)
    ):
        conf = compose(
            config_name="apps/colmap_3dgrt.yaml",
            overrides=[
                f"path={args.scene}",
                f"out_dir={args.out}",
                "experiment_name=blur_parity",
                f"resume={args.ckpt}",
                "use_wandb=false",
                "num_workers=0",
                "initialization=random",
                "dataset.downsample_factor=1",
                f"dataset.test_split_interval={args.test_split_interval}",
                "n_iterations=7000",
                "val_frequency=7000",
                *args.override,
            ],
        )

    trainer = Trainer3DGRUT(conf)
    trainer.model.build_acc(rebuild=True)  # resume leaves gasAABB zeroed

    eval_dataset = ColmapDataset(
        args.scene,
        split="val",
        downsample_factor=1,
        test_split_interval=args.test_split_interval,
        shutter_type=conf.dataset.get("shutter_type", "GLOBAL"),
        rs_ray_injection=True,
        blur_samples=args.blur_samples,
    )
    rotations = _rotation_deg_by_name(
        os.path.join(args.scene, "sparse", "0")
    )

    loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    frames = []
    for batch in loader:
        name = "/".join(
            batch["image_path"][0].replace("\\", "/").split("/")[-2:]
        )
        frames.append((rotations.get(name, -1.0), name, batch))
    frames.sort(key=lambda item: item[0])
    picked = frames[: args.n_slow] + frames[-args.n_fast :]

    rows = []
    for rotation_deg, name, batch in picked:
        gpu_batch = eval_dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            sharp = trainer.model(gpu_batch, train=False)["pred_rgb"][0]
            intr = int(eval_dataset._first_scalar(batch["intr"]))
            cache = eval_dataset._lazy_worker_intrinsics_cache()
            rays_dir_cam = cache[intr][2]
            rays_ori_k, rays_dir_k = eval_dataset.build_blur_bundles(
                name, rays_dir_cam
            )
            blur_batch = replace(
                gpu_batch, rays_ori=rays_ori_k, rays_dir=rays_dir_k
            )
            blurred = (
                trainer.model(blur_batch, train=False)["pred_rgb"]
                .mean(dim=0)
            )
        gt = gpu_batch.rgb_gt[0]
        row = {
            "name": name,
            "rotation_deg_per_readout": round(rotation_deg, 3),
            "psnr_sharp": round(_psnr(sharp, gt), 4),
            "psnr_parity": round(_psnr(blurred, gt), 4),
            "lapvar_sharp": _laplacian_var(sharp),
            "lapvar_parity": _laplacian_var(blurred),
            "lapvar_gt": _laplacian_var(gt),
        }
        rows.append(row)
        print(
            f"{name} rot={row['rotation_deg_per_readout']} "
            f"sharp={row['psnr_sharp']} parity={row['psnr_parity']} "
            f"lap(sharp/gt)={row['lapvar_sharp'] / max(row['lapvar_gt'], 1e-12):.2f}"
        )
        side = torch.cat([gt, sharp.clamp(0, 1), blurred.clamp(0, 1)], dim=1)
        side_u8 = (side.cpu().numpy() * 255).astype(np.uint8)
        try:
            import cv2

            cv2.imwrite(
                os.path.join(
                    args.out, name.replace("/", "_") + "_gt_sharp_parity.jpg"
                ),
                side_u8[:, :, ::-1],
            )
        except ImportError:
            pass

    with open(
        os.path.join(args.out, "parity_summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=1)
    print(f"wrote {args.out}/parity_summary.json ({len(rows)} frames)")


if __name__ == "__main__":
    main()
