# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

from threedgrut.utils.misc import quaternion_to_so3, so3_to_quaternion_wxyz, to_np

SURFACE_PCA_AUDIT_MAX_SAMPLES = 65_536
SURFACE_PCA_COVARIANCE_CHUNK_SIZE = 4_096


class SurfaceAlignedPCAConfig(NamedTuple):
    """Validated geometry parameters for surface-aligned initialization."""

    num_support_points: int
    max_neighbor_radius_m: float
    max_normal_to_mid_ratio: float
    min_mid_to_max_ratio: float
    min_mid_eigenvalue_m2: float
    min_thickness_ratio: float
    query_chunk_size: int


class SurfaceAlignedPCAAudit(NamedTuple):
    """Compact deterministic audit data for one PCA initialization.

    Rejection counts are exact. Quantiles use at most 65,536 source rows,
    sampled at a deterministic stride recorded by ``quantile_sample_count``.
    """

    total_points: int
    finite_point_count: int
    reliable_point_count: int
    flattened_point_count: int
    nonfinite_input_count: int
    invalid_incumbent_scale_count: int
    insufficient_support_count: int
    nonfinite_geometry_count: int
    materially_negative_eigenvalue_count: int
    radius_rejection_count: int
    mid_eigenvalue_rejection_count: int
    normal_ratio_rejection_count: int
    line_like_rejection_count: int
    rotation_rejection_count: int
    quantile_sample_count: int
    radius_quantiles_m: tuple[float, float, float] | None
    mid_eigenvalue_quantiles_m2: tuple[float, float, float] | None
    normal_to_mid_ratio_quantiles: tuple[float, float, float] | None
    mid_to_max_ratio_quantiles: tuple[float, float, float] | None
    tangent_scale_quantiles_m: tuple[float, float, float] | None
    normal_scale_quantiles_m: tuple[float, float, float] | None
    anisotropy_quantiles: tuple[float, float, float] | None


class SurfaceAlignedPCAResult(NamedTuple):
    """Final initializer values and the rows on which PCA was reliable."""

    physical_scales: torch.Tensor
    raw_rotations_wxyz: torch.Tensor
    reliable_mask: torch.Tensor
    audit: SurfaceAlignedPCAAudit


def _sampled_quantiles(
    chunks: list[npt.NDArray[np.float64]],
) -> tuple[float, float, float] | None:
    if not chunks:
        return None
    values = np.concatenate(chunks)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    quantiles = np.quantile(values, (0.5, 0.95, 0.99))
    return (
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


def _empty_surface_pca_audit(
    *,
    total_points: int,
    finite_point_count: int,
    nonfinite_input_count: int,
    invalid_incumbent_scale_count: int,
    insufficient_support_count: int,
) -> SurfaceAlignedPCAAudit:
    return SurfaceAlignedPCAAudit(
        total_points=total_points,
        finite_point_count=finite_point_count,
        reliable_point_count=0,
        flattened_point_count=0,
        nonfinite_input_count=nonfinite_input_count,
        invalid_incumbent_scale_count=invalid_incumbent_scale_count,
        insufficient_support_count=insufficient_support_count,
        nonfinite_geometry_count=0,
        materially_negative_eigenvalue_count=0,
        radius_rejection_count=0,
        mid_eigenvalue_rejection_count=0,
        normal_ratio_rejection_count=0,
        line_like_rejection_count=0,
        rotation_rejection_count=0,
        quantile_sample_count=0,
        radius_quantiles_m=None,
        mid_eigenvalue_quantiles_m2=None,
        normal_to_mid_ratio_quantiles=None,
        mid_to_max_ratio_quantiles=None,
        tangent_scale_quantiles_m=None,
        normal_scale_quantiles_m=None,
        anisotropy_quantiles=None,
    )


def _validate_surface_pca_inputs(
    points: torch.Tensor,
    incumbent_physical_scales: torch.Tensor,
    incumbent_raw_rotations_wxyz: torch.Tensor,
    config: SurfaceAlignedPCAConfig,
) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {tuple(points.shape)}")
    expected_scales_shape = (points.shape[0], 3)
    if incumbent_physical_scales.shape != expected_scales_shape:
        raise ValueError(
            "Expected incumbent scales with shape "
            f"{expected_scales_shape}, got "
            f"{tuple(incumbent_physical_scales.shape)}"
        )
    expected_rotations_shape = (points.shape[0], 4)
    if incumbent_raw_rotations_wxyz.shape != expected_rotations_shape:
        raise ValueError(
            "Expected incumbent rotations with shape "
            f"{expected_rotations_shape}, got "
            f"{tuple(incumbent_raw_rotations_wxyz.shape)}"
        )
    if points.device != incumbent_physical_scales.device or points.device != incumbent_raw_rotations_wxyz.device:
        raise ValueError("Points, scales, and rotations must share a device")
    if config.num_support_points < 3:
        raise ValueError("num_support_points must be at least 3")
    if config.query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")


def surface_aligned_pca_initialize(
    points: torch.Tensor,
    incumbent_physical_scales: torch.Tensor,
    incumbent_raw_rotations_wxyz: torch.Tensor,
    config: SurfaceAlignedPCAConfig,
) -> SurfaceAlignedPCAResult:
    """Align reliable Gaussian ellipsoids to local point-cloud surfaces.

    Neighborhood search and eigendecomposition run deterministically on CPU
    float64 data. Each support contains the anchor and its nearest
    ``num_support_points - 1`` other source rows. Unreliable rows retain their
    incumbent scales and raw rotations exactly.
    """
    _validate_surface_pca_inputs(
        points,
        incumbent_physical_scales,
        incumbent_raw_rotations_wxyz,
        config,
    )
    total_points = points.shape[0]
    final_scales = incumbent_physical_scales.detach().clone()
    final_rotations = incumbent_raw_rotations_wxyz.detach().clone()
    reliable_mask_np = np.zeros(total_points, dtype=np.bool_)

    points_np = points.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    point_finite = np.isfinite(points_np).all(axis=1)
    incumbent_finite = (
        (torch.isfinite(incumbent_physical_scales).all(dim=1) & torch.isfinite(incumbent_raw_rotations_wxyz).all(dim=1))
        .detach()
        .cpu()
        .numpy()
    )
    incumbent_scale_valid = (
        (
            (incumbent_physical_scales > 0.0).all(dim=1)
            & (incumbent_physical_scales == incumbent_physical_scales[:, :1]).all(dim=1)
        )
        .detach()
        .cpu()
        .numpy()
    )
    incumbent_valid = incumbent_finite & incumbent_scale_valid
    finite_source_indices = np.flatnonzero(point_finite)
    finite_point_count = int(finite_source_indices.size)
    nonfinite_input_count = int(np.count_nonzero(~(point_finite & incumbent_finite)))
    invalid_incumbent_scale_count = int(np.count_nonzero(incumbent_finite & ~incumbent_scale_valid))

    if total_points < config.num_support_points or finite_point_count < config.num_support_points:
        audit = _empty_surface_pca_audit(
            total_points=total_points,
            finite_point_count=finite_point_count,
            nonfinite_input_count=nonfinite_input_count,
            invalid_incumbent_scale_count=(invalid_incumbent_scale_count),
            insufficient_support_count=total_points,
        )
        return SurfaceAlignedPCAResult(
            physical_scales=final_scales,
            raw_rotations_wxyz=final_rotations,
            reliable_mask=torch.from_numpy(reliable_mask_np).to(points.device),
            audit=audit,
        )

    finite_points = points_np[finite_source_indices]
    tree = sklearn.neighbors.KDTree(finite_points, metric="euclidean")
    query_neighbors = min(
        config.num_support_points + 1,
        finite_point_count,
    )
    sample_stride = max(
        1,
        int(np.ceil(total_points / SURFACE_PCA_AUDIT_MAX_SAMPLES)),
    )

    nonfinite_geometry_count = 0
    materially_negative_count = 0
    radius_rejection_count = 0
    mid_eigenvalue_rejection_count = 0
    normal_ratio_rejection_count = 0
    line_like_rejection_count = 0
    rotation_rejection_count = 0
    flattened_point_count = 0
    quantile_sample_count = 0
    radius_samples: list[npt.NDArray[np.float64]] = []
    mid_eigenvalue_samples: list[npt.NDArray[np.float64]] = []
    normal_ratio_samples: list[npt.NDArray[np.float64]] = []
    mid_to_max_ratio_samples: list[npt.NDArray[np.float64]] = []
    tangent_scale_samples: list[npt.NDArray[np.float64]] = []
    normal_scale_samples: list[npt.NDArray[np.float64]] = []
    anisotropy_samples: list[npt.NDArray[np.float64]] = []

    for query_start in range(
        0,
        finite_point_count,
        config.query_chunk_size,
    ):
        query_end = min(
            query_start + config.query_chunk_size,
            finite_point_count,
        )
        distances, neighbor_tree_indices = tree.query(
            finite_points[query_start:query_end],
            k=query_neighbors,
            return_distance=True,
            dualtree=False,
            breadth_first=False,
            sort_results=True,
        )
        anchor_tree_indices = np.arange(query_start, query_end)
        anchor_columns = neighbor_tree_indices == anchor_tree_indices[:, None]
        non_anchor_distances = distances.copy()
        non_anchor_distances[anchor_columns] = np.inf
        neighbor_order = np.argsort(
            non_anchor_distances,
            axis=1,
            kind="stable",
        )[:, : config.num_support_points - 1]
        other_tree_indices = np.take_along_axis(
            neighbor_tree_indices,
            neighbor_order,
            axis=1,
        )
        other_distances = np.take_along_axis(
            distances,
            neighbor_order,
            axis=1,
        )

        for block_start in range(
            0,
            query_end - query_start,
            SURFACE_PCA_COVARIANCE_CHUNK_SIZE,
        ):
            block_end = min(
                block_start + SURFACE_PCA_COVARIANCE_CHUNK_SIZE,
                query_end - query_start,
            )
            block_anchor_tree_indices = anchor_tree_indices[block_start:block_end]
            block_source_indices = finite_source_indices[block_anchor_tree_indices]
            block_other_tree_indices = other_tree_indices[block_start:block_end]
            support_tree_indices = np.concatenate(
                (
                    block_anchor_tree_indices[:, None],
                    block_other_tree_indices,
                ),
                axis=1,
            )
            support = finite_points[support_tree_indices]
            centered = support - np.mean(support, axis=1, keepdims=True)
            covariance = np.einsum(
                "nki,nkj->nij",
                centered,
                centered,
                optimize=True,
            ) / float(config.num_support_points)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            radius = np.max(
                other_distances[block_start:block_end],
                axis=1,
            )
            covariance_scale = np.max(
                np.abs(covariance),
                axis=(1, 2),
            )
            roundoff_tolerance = (
                128.0
                * np.finfo(np.float64).eps
                * np.maximum(
                    covariance_scale,
                    config.min_mid_eigenvalue_m2,
                )
            )
            materially_negative = eigenvalues[:, 0] < -roundoff_tolerance
            clamped_min = np.maximum(eigenvalues[:, 0], 0.0)
            mid_eigenvalue = eigenvalues[:, 1]
            max_eigenvalue = eigenvalues[:, 2]
            normal_to_mid_ratio: npt.NDArray[np.float64] = np.full(
                block_end - block_start,
                np.inf,
                dtype=np.float64,
            )
            positive_mid = mid_eigenvalue > 0.0
            normal_to_mid_ratio[positive_mid] = clamped_min[positive_mid] / mid_eigenvalue[positive_mid]
            mid_to_max_ratio: npt.NDArray[np.float64] = np.full(
                block_end - block_start,
                -np.inf,
                dtype=np.float64,
            )
            positive_max = max_eigenvalue > 0.0
            mid_to_max_ratio[positive_max] = mid_eigenvalue[positive_max] / max_eigenvalue[positive_max]
            geometry_finite = (
                np.isfinite(radius)
                & np.isfinite(eigenvalues).all(axis=1)
                & np.isfinite(eigenvectors).all(axis=(1, 2))
                & np.isfinite(normal_to_mid_ratio)
                & np.isfinite(mid_to_max_ratio)
            )
            block_incumbent_valid = incumbent_valid[block_source_indices]
            reliable = (
                geometry_finite
                & block_incumbent_valid
                & ~materially_negative
                & (radius <= config.max_neighbor_radius_m)
                & (mid_eigenvalue > config.min_mid_eigenvalue_m2)
                & (normal_to_mid_ratio <= config.max_normal_to_mid_ratio)
                & (mid_to_max_ratio >= config.min_mid_to_max_ratio)
            )

            nonfinite_geometry_count += int(np.count_nonzero(~geometry_finite))
            materially_negative_count += int(np.count_nonzero(materially_negative))
            radius_rejection_count += int(np.count_nonzero(geometry_finite & (radius > config.max_neighbor_radius_m)))
            mid_eigenvalue_rejection_count += int(
                np.count_nonzero(geometry_finite & (mid_eigenvalue <= config.min_mid_eigenvalue_m2))
            )
            normal_ratio_rejection_count += int(
                np.count_nonzero(geometry_finite & (normal_to_mid_ratio > config.max_normal_to_mid_ratio))
            )
            line_like_rejection_count += int(
                np.count_nonzero(geometry_finite & (mid_to_max_ratio < config.min_mid_to_max_ratio))
            )

            reliable_rows = np.flatnonzero(reliable)
            normal_scales: npt.NDArray[np.float64] = np.empty(
                0,
                dtype=np.float64,
            )
            tangent_scales: npt.NDArray[np.float64] = np.empty(
                0,
                dtype=np.float64,
            )
            reliable_source_indices: npt.NDArray[np.int64] = np.empty(
                0,
                dtype=np.int64,
            )
            if reliable_rows.size > 0:
                normals = eigenvectors[reliable_rows, :, 0]
                tangent_u = eigenvectors[reliable_rows, :, 2]
                tangent_v = np.cross(normals, tangent_u)
                rotations = np.stack(
                    (tangent_u, tangent_v, normals),
                    axis=2,
                )
                determinants = np.linalg.det(rotations)
                negative_determinant = determinants < 0.0
                rotations[negative_determinant, :, 1] *= -1.0
                determinants = np.linalg.det(rotations)
                rotation_finite = (
                    np.isfinite(rotations).all(axis=(1, 2))
                    & np.isfinite(determinants)
                    & (determinants > 0.0)
                    & (np.abs(determinants - 1.0) <= 1.0e-10)
                )

                rotation_tensor = torch.from_numpy(rotations)
                quaternion = so3_to_quaternion_wxyz(rotation_tensor)
                roundtrip = quaternion_to_so3(quaternion)
                roundtrip_error = torch.amax(
                    torch.abs(roundtrip - rotation_tensor),
                    dim=(1, 2),
                ).numpy()
                rotation_finite &= torch.isfinite(quaternion).all(dim=1).numpy() & (roundtrip_error <= 1.0e-10)
                rotation_rejection_count += int(np.count_nonzero(~rotation_finite))
                rejected_rows = reliable_rows[~rotation_finite]
                reliable[rejected_rows] = False
                reliable_rows = reliable_rows[rotation_finite]
                quaternion = quaternion[rotation_finite]

                if reliable_rows.size > 0:
                    reliable_source_indices = block_source_indices[reliable_rows]
                    source_index_tensor = torch.from_numpy(reliable_source_indices).to(device=points.device)
                    tangent_scale_tensor = incumbent_physical_scales.index_select(
                        0,
                        source_index_tensor,
                    )[:, 0]
                    tangent_scales = tangent_scale_tensor.detach().to(device="cpu", dtype=torch.float64).numpy()
                    normal_scales = np.minimum(
                        tangent_scales,
                        np.maximum(
                            config.min_thickness_ratio * tangent_scales,
                            np.sqrt(clamped_min[reliable_rows]),
                        ),
                    )
                    normal_scale_tensor = torch.from_numpy(normal_scales).to(
                        device=points.device,
                        dtype=incumbent_physical_scales.dtype,
                    )
                    replacement_scales = torch.stack(
                        (
                            tangent_scale_tensor,
                            tangent_scale_tensor,
                            normal_scale_tensor,
                        ),
                        dim=1,
                    )
                    final_scales.index_copy_(
                        0,
                        source_index_tensor,
                        replacement_scales,
                    )
                    final_rotations.index_copy_(
                        0,
                        source_index_tensor,
                        quaternion.to(
                            device=points.device,
                            dtype=incumbent_raw_rotations_wxyz.dtype,
                        ),
                    )
                    reliable_mask_np[reliable_source_indices] = True
                    flattened_point_count += int(np.count_nonzero(normal_scales < tangent_scales))

            sampled = (block_source_indices % sample_stride == 0) & geometry_finite
            quantile_sample_count += int(np.count_nonzero(sampled))
            if np.any(sampled):
                radius_samples.append(radius[sampled])
                mid_eigenvalue_samples.append(mid_eigenvalue[sampled])
                normal_ratio_samples.append(normal_to_mid_ratio[sampled])
                mid_to_max_ratio_samples.append(mid_to_max_ratio[sampled])
            if reliable_source_indices.size > 0:
                reliable_sampled = reliable_source_indices % sample_stride == 0
                if np.any(reliable_sampled):
                    sampled_tangent = tangent_scales[reliable_sampled]
                    sampled_normal = normal_scales[reliable_sampled]
                    tangent_scale_samples.append(sampled_tangent)
                    normal_scale_samples.append(sampled_normal)
                    anisotropy_samples.append(sampled_tangent / sampled_normal)

    audit = SurfaceAlignedPCAAudit(
        total_points=total_points,
        finite_point_count=finite_point_count,
        reliable_point_count=int(np.count_nonzero(reliable_mask_np)),
        flattened_point_count=flattened_point_count,
        nonfinite_input_count=nonfinite_input_count,
        invalid_incumbent_scale_count=invalid_incumbent_scale_count,
        insufficient_support_count=0,
        nonfinite_geometry_count=nonfinite_geometry_count,
        materially_negative_eigenvalue_count=materially_negative_count,
        radius_rejection_count=radius_rejection_count,
        mid_eigenvalue_rejection_count=(mid_eigenvalue_rejection_count),
        normal_ratio_rejection_count=normal_ratio_rejection_count,
        line_like_rejection_count=line_like_rejection_count,
        rotation_rejection_count=rotation_rejection_count,
        quantile_sample_count=quantile_sample_count,
        radius_quantiles_m=_sampled_quantiles(radius_samples),
        mid_eigenvalue_quantiles_m2=_sampled_quantiles(mid_eigenvalue_samples),
        normal_to_mid_ratio_quantiles=_sampled_quantiles(normal_ratio_samples),
        mid_to_max_ratio_quantiles=_sampled_quantiles(mid_to_max_ratio_samples),
        tangent_scale_quantiles_m=_sampled_quantiles(tangent_scale_samples),
        normal_scale_quantiles_m=_sampled_quantiles(normal_scale_samples),
        anisotropy_quantiles=_sampled_quantiles(anisotropy_samples),
    )
    return SurfaceAlignedPCAResult(
        physical_scales=final_scales,
        raw_rotations_wxyz=final_rotations,
        reliable_mask=torch.from_numpy(reliable_mask_np).to(points.device),
        audit=audit,
    )


def apply_points_transform(
    points: torch.Tensor,
    transform: np.ndarray | torch.Tensor | None,
) -> torch.Tensor:
    """Apply a 4x4 affine transform to an ``(N, 3)`` point tensor."""
    if transform is None:
        return points
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")

    transform_tensor = torch.as_tensor(transform, dtype=points.dtype, device=points.device)
    if transform_tensor.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {tuple(transform_tensor.shape)}")
    if not torch.isfinite(transform_tensor).all():
        raise ValueError("transform must contain only finite values")
    return points @ transform_tensor[:3, :3].T + transform_tensor[:3, 3]


def k_nearest_neighbors(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = sklearn.neighbors.NearestNeighbors(
        n_neighbors=K,
        metric="euclidean",
    ).fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def nearest_neighbors(pts_src, k=2):
    pts_src_np = to_np(pts_src)

    # distance from a point set to itself
    pts_target_np = pts_src_np

    # Build the tree
    kd_tree = sklearn.neighbors.KDTree(pts_target_np)

    # Query it
    _, neighbors = kd_tree.query(pts_src_np, k=k)

    # Mask out self element
    mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

    # make sure we mask out exactly one element in each row, in rare case of many duplicate points
    mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
    neighbors = neighbors[mask].reshape((neighbors.shape[0], k - 1))

    # recompute distances in torch, so the function is differentiable
    neigh_inds = torch.tensor(neighbors, device=pts_src.device, dtype=torch.int64)
    return neigh_inds


def nearest_neighbor_dist_cpuKD(pts_src, pts_target=None):
    """
    Compute the distance to the nearest neighbor, using a CPU kd-tree
    Passing one arg computes from a point set to itself,
    to args computes distance from each point in src to target
    """

    pts_src_np = to_np(pts_src)

    if pts_target is None:
        # distance from a point set to itself
        on_self = True
        k = 2
        pts_target = pts_src
        pts_target_np = pts_src_np
    else:
        # distance between two point sets
        on_self = False
        k = 1
        pts_target_np = to_np(pts_target)

    # Build the tree
    kd_tree = sklearn.neighbors.KDTree(pts_target_np)

    # Query it
    _, neighbors = kd_tree.query(pts_src_np, k=k)

    # Mask out self element
    if on_self:
        mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

        # make sure we mask out exactly one element in each row, in rare case of many duplicate points
        mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
        neighbors = neighbors[mask].reshape((neighbors.shape[0],))
    else:
        neighbors = neighbors[:, 0]

    # recompute distances in torch, so the function is differentiable
    neigh_inds = torch.tensor(neighbors, device=pts_src.device, dtype=torch.int64)
    dists = torch.linalg.norm(pts_src - pts_target[neigh_inds, :], dim=-1)

    return dists


def safe_normalize(vecs):
    norms = torch.linalg.norm(vecs, dim=-1)
    norms = torch.where(norms > 0.0, norms, 1.0)
    return vecs / norms[..., None]
