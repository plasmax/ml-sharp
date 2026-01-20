"""Contains functionality for exporting Gaussian splats to OpenEXR deep data.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import OpenEXR
import torch

from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData

LOGGER = logging.getLogger(__name__)


class DeepSample(NamedTuple):
    """A single deep sample."""

    z: float
    r: float
    g: float
    b: float
    a: float


def _project_gaussians_vectorized(
    gaussians: Gaussians3D,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D screen space (vectorized).

    Returns:
        means2d: [N, 2] - 2D centers
        cov2d: [N, 2, 2] - 2D covariances
        depths: [N] - depths
        radii: [N] - projected radii
        valid_mask: [N] - visibility mask
    """
    from sharp.utils.gaussians import compose_covariance_matrices

    means = gaussians.mean_vectors.squeeze(0)  # [N, 3]
    quats = gaussians.quaternions.squeeze(0)  # [N, 4]
    scales = gaussians.singular_values.squeeze(0)  # [N, 3]

    # Build 3D covariance matrices
    cov3d = compose_covariance_matrices(quats, scales)

    # Transform to camera space
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    means_cam = means @ R.T + t

    depths = means_cam[:, 2]
    valid_mask = depths > 0.01

    # Perspective projection
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z_safe = depths.clamp(min=0.01)
    means2d = torch.zeros(means.shape[0], 2, device=means.device, dtype=means.dtype)
    means2d[:, 0] = fx * means_cam[:, 0] / z_safe + cx
    means2d[:, 1] = fy * means_cam[:, 1] / z_safe + cy

    # Compute 2D covariance via EWA splatting
    J = torch.zeros(means.shape[0], 2, 3, device=means.device, dtype=means.dtype)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * means_cam[:, 0] / (z_safe**2)
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * means_cam[:, 1] / (z_safe**2)

    cov_cam = torch.einsum("ij,njk,lk->nil", R, cov3d, R)
    cov2d = torch.einsum("nij,njk,nlk->nil", J, cov_cam, J)

    # Add anti-aliasing filter
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # Compute radii from eigenvalues
    trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp(min=1e-8)
    discriminant = (trace**2 / 4 - det).clamp(min=0)
    max_eigenval = trace / 2 + torch.sqrt(discriminant)
    radii = 3.0 * torch.sqrt(max_eigenval.clamp(min=1e-8))

    return means2d, cov2d, depths, radii, valid_mask


def _process_gaussian_batch_vectorized(
    batch_indices: np.ndarray,
    means2d: np.ndarray,
    cov2d_inv: np.ndarray,
    depths: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    width: int,
    height: int,
    contribution_threshold: float,
    pixel_samples: dict,
) -> None:
    """Process a batch of Gaussians using vectorized numpy operations."""
    for i in batch_indices:
        cx, cy = means2d[i]
        r = radii[i]

        # Compute bounding box
        x_min = max(0, int(cx - r))
        x_max = min(width - 1, int(cx + r))
        y_min = max(0, int(cy - r))
        y_max = min(height - 1, int(cy + r))

        if x_min > x_max or y_min > y_max:
            continue

        # Get parameters for this Gaussian
        cov_inv = cov2d_inv[i]
        depth = depths[i]
        color = colors[i]
        opacity = opacities[i]

        # Create pixel coordinate grids for the bounding box
        xs = np.arange(x_min, x_max + 1) + 0.5
        ys = np.arange(y_min, y_max + 1) + 0.5

        # Compute deltas from center
        dx = xs - cx  # [W]
        dy = ys - cy  # [H]

        # Compute Mahalanobis distance squared for all pixels in bbox
        # mahal_sq = dx^T * cov_inv * dx
        # For 2D: cov_inv[0,0]*dx^2 + 2*cov_inv[0,1]*dx*dy + cov_inv[1,1]*dy^2
        dx_grid, dy_grid = np.meshgrid(dx, dy)  # [H, W]

        mahal_sq = (
            cov_inv[0, 0] * dx_grid**2
            + 2 * cov_inv[0, 1] * dx_grid * dy_grid
            + cov_inv[1, 1] * dy_grid**2
        )

        # Mask to 3-sigma cutoff
        valid_mask = mahal_sq <= 9.0

        # Compute contributions
        contrib = np.exp(-0.5 * mahal_sq)
        alpha = opacity * contrib

        # Apply threshold
        valid_mask &= alpha >= contribution_threshold

        # Get valid pixel coordinates and values
        valid_ys, valid_xs = np.where(valid_mask)

        for j in range(len(valid_ys)):
            py = y_min + valid_ys[j]
            px = x_min + valid_xs[j]
            a = alpha[valid_ys[j], valid_xs[j]]

            key = (py, px)
            if key not in pixel_samples:
                pixel_samples[key] = []

            pixel_samples[key].append(
                DeepSample(
                    z=float(depth),
                    r=float(color[0]),
                    g=float(color[1]),
                    b=float(color[2]),
                    a=float(a),
                )
            )


def export_deep_exr(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    viewmat: torch.Tensor | None = None,
    max_samples_per_pixel: int = 64,
    contribution_threshold: float = 0.01,
    batch_size: int = 5000,
) -> None:
    """Export Gaussian splats to OpenEXR deep data format.

    This function converts a 3D Gaussian splat representation to an OpenEXR
    deep image where each pixel contains multiple depth samples corresponding
    to the Gaussians that contribute to that pixel.

    Args:
        gaussians: The 3D Gaussians to export
        metadata: Scene metadata including focal length and resolution
        output_path: Path to write the output .exr file
        viewmat: Camera view matrix [4, 4]. If None, uses identity.
        max_samples_per_pixel: Maximum number of samples to store per pixel
        contribution_threshold: Minimum Gaussian contribution to include
        batch_size: Number of Gaussians to process at once

    The output deep EXR contains the following channels:
        - R, G, B: Color values (in sRGB color space)
        - A: Alpha/opacity values
        - Z: Depth values (distance from camera)
    """
    device = gaussians.mean_vectors.device
    width, height = metadata.resolution_px
    width, height = int(width), int(height)
    fx = metadata.focal_length_px

    # Build intrinsics matrix
    K = torch.tensor(
        [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    )

    if viewmat is None:
        viewmat = torch.eye(4, device=device, dtype=torch.float32)

    LOGGER.info(
        "Projecting %d Gaussians to 2D for %dx%d image...",
        gaussians.mean_vectors.shape[1],
        width,
        height,
    )

    # Project all Gaussians
    means2d, cov2d, depths, radii, valid_mask = _project_gaussians_vectorized(
        gaussians, viewmat, K
    )

    # Filter to valid Gaussians only
    means2d = means2d[valid_mask]
    cov2d = cov2d[valid_mask]
    depths = depths[valid_mask]
    radii = radii[valid_mask]

    colors = gaussians.colors.squeeze(0)[valid_mask]  # [N, 3]
    opacities = gaussians.opacities.squeeze(0)[valid_mask]  # [N]

    # Convert colors to sRGB if needed
    if metadata.color_space == "linearRGB":
        colors = cs_utils.linearRGB2sRGB(colors)

    n_gaussians = means2d.shape[0]
    LOGGER.info("Processing %d visible Gaussians...", n_gaussians)

    # Compute inverse covariances
    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp(min=1e-8)
    cov2d_inv = torch.zeros_like(cov2d)
    cov2d_inv[:, 0, 0] = cov2d[:, 1, 1] / det
    cov2d_inv[:, 1, 1] = cov2d[:, 0, 0] / det
    cov2d_inv[:, 0, 1] = -cov2d[:, 0, 1] / det
    cov2d_inv[:, 1, 0] = -cov2d[:, 1, 0] / det

    # Use a dictionary to accumulate samples per pixel
    pixel_samples: dict[tuple[int, int], list[DeepSample]] = {}

    # Move to CPU for pixel-level processing
    means2d_cpu = means2d.cpu().numpy()
    cov2d_inv_cpu = cov2d_inv.cpu().numpy()
    depths_cpu = depths.cpu().numpy()
    radii_cpu = radii.cpu().numpy()
    colors_cpu = colors.cpu().numpy()
    opacities_cpu = opacities.cpu().numpy()

    # Process Gaussians in batches
    for batch_start in range(0, n_gaussians, batch_size):
        batch_end = min(batch_start + batch_size, n_gaussians)

        if batch_start % 50000 == 0:
            LOGGER.info(
                "Processing Gaussians %d-%d (%.1f%%)...",
                batch_start,
                batch_end,
                100 * batch_start / n_gaussians,
            )

        batch_indices = np.arange(batch_start, batch_end)
        _process_gaussian_batch_vectorized(
            batch_indices=batch_indices,
            means2d=means2d_cpu,
            cov2d_inv=cov2d_inv_cpu,
            depths=depths_cpu,
            radii=radii_cpu,
            colors=colors_cpu,
            opacities=opacities_cpu,
            width=width,
            height=height,
            contribution_threshold=contribution_threshold,
            pixel_samples=pixel_samples,
        )

    LOGGER.info("Building deep arrays for %d pixels with samples...", len(pixel_samples))

    # Allocate deep data arrays
    R = np.empty((height, width), dtype=object)
    G = np.empty((height, width), dtype=object)
    B = np.empty((height, width), dtype=object)
    A = np.empty((height, width), dtype=object)
    Z = np.empty((height, width), dtype=object)

    # Initialize all pixels with empty arrays
    for y in range(height):
        for x in range(width):
            R[y, x] = np.array([], dtype=np.float32)
            G[y, x] = np.array([], dtype=np.float32)
            B[y, x] = np.array([], dtype=np.float32)
            A[y, x] = np.array([], dtype=np.float32)
            Z[y, x] = np.array([], dtype=np.float32)

    # Fill in pixels with samples
    total_samples = 0
    for (y, x), samples in pixel_samples.items():
        # Sort by depth (front to back)
        samples.sort(key=lambda s: s.z)

        # Limit samples
        if len(samples) > max_samples_per_pixel:
            samples = samples[:max_samples_per_pixel]

        total_samples += len(samples)

        R[y, x] = np.array([s.r for s in samples], dtype=np.float32)
        G[y, x] = np.array([s.g for s in samples], dtype=np.float32)
        B[y, x] = np.array([s.b for s in samples], dtype=np.float32)
        A[y, x] = np.array([s.a for s in samples], dtype=np.float32)
        Z[y, x] = np.array([s.z for s in samples], dtype=np.float32)

    LOGGER.info(
        "Total samples: %d (avg %.2f per non-empty pixel)",
        total_samples,
        total_samples / max(1, len(pixel_samples)),
    )

    # Write deep EXR
    LOGGER.info("Writing deep EXR to %s...", output_path)

    header = {
        "type": OpenEXR.deepscanline,
        "compression": OpenEXR.ZIPS_COMPRESSION,
    }

    channels = {"R": R, "G": G, "B": B, "A": A, "Z": Z}

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(str(output_path))

    LOGGER.info("Deep EXR export complete.")


def export_deep_exr_from_ply(
    ply_path: Path,
    output_path: Path,
    viewmat: torch.Tensor | None = None,
    max_samples_per_pixel: int = 64,
    contribution_threshold: float = 0.01,
    downsample_gaussians: int | None = None,
) -> None:
    """Export a PLY file containing Gaussian splats to OpenEXR deep data.

    Args:
        ply_path: Path to the input .ply file
        output_path: Path to write the output .exr file
        viewmat: Camera view matrix [4, 4]. If None, uses identity.
        max_samples_per_pixel: Maximum number of samples to store per pixel
        contribution_threshold: Minimum Gaussian contribution to include
        downsample_gaussians: If set, randomly sample this many Gaussians
    """
    from sharp.utils.gaussians import load_ply

    LOGGER.info("Loading PLY from %s...", ply_path)
    gaussians, metadata = load_ply(ply_path)

    if downsample_gaussians is not None:
        n_total = gaussians.mean_vectors.shape[1]
        if downsample_gaussians < n_total:
            LOGGER.info(
                "Downsampling from %d to %d Gaussians...", n_total, downsample_gaussians
            )
            indices = torch.randperm(n_total)[:downsample_gaussians]
            gaussians = Gaussians3D(
                mean_vectors=gaussians.mean_vectors[:, indices],
                singular_values=gaussians.singular_values[:, indices],
                quaternions=gaussians.quaternions[:, indices],
                colors=gaussians.colors[:, indices],
                opacities=gaussians.opacities[:, indices],
            )

    export_deep_exr(
        gaussians=gaussians,
        metadata=metadata,
        output_path=output_path,
        viewmat=viewmat,
        max_samples_per_pixel=max_samples_per_pixel,
        contribution_threshold=contribution_threshold,
    )
