# Standalone minimal repro for the 3DGRT OptiX grad-tracked-ray forward/backward
# deadlock. Builds the real model from a clean checkpoint, builds the BVH, fires
# ONE forward+backward through the OptiX Tracer, and exits. Gaussians always
# require grad (matches clean training, which does NOT hang). The --case flag
# additionally toggles grad on a geometric INPUT to the autograd Function:
#   clean : nothing extra            -> expected SUCCESS (reproduces clean)
#   rays  : rays_ori/rays_dir grad   -> expected HANG  (reproduces recovery)
#   twld  : T_to_world grad          -> THE DECISIVE TEST (rays-specific vs any input)
# Run each case in its OWN process under `timeout -s ABRT` (a libcuda wedge
# poisons the context, so cases cannot share a process).
import argparse
import os

import torch
from hydra import compose, initialize_config_dir

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians


def build_conf(rig_path: str):
    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(
            config_name="apps/colmap_3dgrt.yaml",
            overrides=[
                f"path={rig_path}",
                "out_dir=/tmp/minrepro",
                "experiment_name=minrepro",
                "use_wandb=false",
            ],
        )


def load_model(conf, ckpt_path: str) -> MixtureOfGaussians:
    model = MixtureOfGaussians(conf, scene_extent=1.0)
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state = ckpt["model"] if "positions" not in ckpt else ckpt
    model.init_from_checkpoint(state, setup_optimizer=False)
    model = model.to("cuda")
    for attr in (
        "positions",
        "rotation",
        "scale",
        "density",
        "features_albedo",
        "features_specular",
    ):
        getattr(model, attr).requires_grad_(True)
    model.build_acc(rebuild=True)
    return model


def make_batch(conf, case: str):
    """Build a REAL dataset batch (rays actually hit the scene), then toggle
    grad on the geometric input under test."""
    train_dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    sample = train_dataset[0]
    collated = torch.utils.data.default_collate([sample])
    gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(collated)
    if case == "rays":
        gpu_batch.rays_ori = gpu_batch.rays_ori.detach().clone().requires_grad_(True)
        gpu_batch.rays_dir = gpu_batch.rays_dir.detach().clone().requires_grad_(True)
    elif case == "twld":
        gpu_batch.T_to_world = gpu_batch.T_to_world.detach().clone().requires_grad_(True)
    return gpu_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["clean", "rays", "twld"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rig", required=True)
    args = ap.parse_args()

    torch.cuda.init()
    conf = build_conf(args.rig)
    model = load_model(conf, args.ckpt)
    batch = make_batch(conf, args.case)

    print(f"[{args.case}] forward...", flush=True)
    out = model.renderer.render(model, batch, train=True, frame_id=0)
    loss = out["pred_rgb"].float().mean()
    print(f"[{args.case}] backward...", flush=True)
    loss.backward()
    torch.cuda.synchronize()
    print(f"[{args.case}] SUCCESS loss={loss.item():.6f}", flush=True)


if __name__ == "__main__":
    main()
