#!/usr/bin/env python3
"""Convert Alembic camera data to SHARP intrinsics/extrinsics and align a Gaussian .ply.

This helper does three things:
1) Reads camera intrinsics from an Alembic camera sample.
2) Obtains a world-from-camera transform (either from JSON or a user-edit section).
3) Transforms a SHARP Gaussian .ply into that world frame and exports an aligned .ply.

Coordinate convention used by SHARP/OpenCV in this repository:
- x right, y down, z forward

Example usage:
    python tools/convert_alembic_camera_to_sharp.py \
      --abc camera.abc \
      --camera-path /cam \
      --sample-index 0 \
      --input-ply input.ply \
      --output-ply aligned.ply \
      --image-width 1920 \
      --image-height 1080 \
      --extrinsics-json camera_transforms.json

JSON schema for --extrinsics-json:
{
  "world_from_camera": {
    "0": [[...4 floats...], [...], [...], [...]],
    "1": [[...], [...], [...], [...]]
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sharp.utils.gaussians import apply_transform, load_ply, save_ply



def _require_alembic_modules():
    try:
        from alembic.Abc import IArchive  # type: ignore
        from alembic.AbcGeom import ICamera  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Alembic Python bindings are required. Install a package that provides "
            "`alembic.Abc` and `alembic.AbcGeom` in your environment."
        ) from exc
    return IArchive, ICamera


def read_camera_sample(abc_path: Path, camera_path: str, sample_index: int):
    """Read Alembic CameraSample and return basic camera parameters."""
    IArchive, ICamera = _require_alembic_modules()

    archive = IArchive(str(abc_path))
    camera_name = camera_path.strip("/")
    camera_obj = ICamera(archive.getTop(), camera_name)
    if not camera_obj.valid():
        raise ValueError(f"Invalid camera path in archive: {camera_path}")

    schema = camera_obj.getSchema()
    num_samples = schema.getNumSamples()
    if num_samples <= 0:
        raise ValueError("Camera schema has no samples.")
    if sample_index < 0 or sample_index >= num_samples:
        raise ValueError(
            f"sample_index={sample_index} out of range [0, {num_samples - 1}]"
        )

    from alembic.Abc import ISampleSelector  # type: ignore

    sample = schema.getValue(ISampleSelector(sample_index))
    return sample, num_samples


def camera_sample_to_intrinsics_px(sample, image_width: int, image_height: int) -> np.ndarray:
    """Convert Alembic camera sample to 3x3 OpenCV intrinsics (pixels)."""
    focal_mm = float(sample.getFocalLength())

    # Alembic apertures/offsets are in centimeters (as in standard Alembic camera schema).
    h_aperture_cm = float(sample.getHorizontalAperture())
    v_aperture_cm = float(sample.getVerticalAperture())
    h_offset_cm = float(sample.getHorizontalFilmOffset())
    v_offset_cm = float(sample.getVerticalFilmOffset())

    lens_squeeze = float(sample.getLensSqueezeRatio())
    if lens_squeeze == 0:
        raise ValueError("Lens squeeze ratio is zero; cannot compute fx.")

    h_aperture_mm = h_aperture_cm * 10.0
    v_aperture_mm = v_aperture_cm * 10.0
    h_offset_mm = h_offset_cm * 10.0
    v_offset_mm = v_offset_cm * 10.0

    fx = (focal_mm / lens_squeeze) * (image_width / h_aperture_mm)
    fy = focal_mm * (image_height / v_aperture_mm)

    cx = image_width * 0.5 + (h_offset_mm / h_aperture_mm) * image_width
    cy = image_height * 0.5 + (v_offset_mm / v_aperture_mm) * image_height

    intrinsics = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsics


def load_world_from_camera(
    sample_index: int,
    extrinsics_json_path: Path | None,
) -> np.ndarray:
    """Load a 4x4 world-from-camera matrix for a sample.

    Priority:
    1) --extrinsics-json path, if provided.
    2) User-edit injection block in this function.
    """
    if extrinsics_json_path is not None:
        payload = json.loads(extrinsics_json_path.read_text())
        by_frame = payload.get("world_from_camera", {})
        if str(sample_index) not in by_frame:
            raise KeyError(
                f"Sample {sample_index} not found in {extrinsics_json_path}. "
                "Expected key under world_from_camera."
            )
        matrix = np.asarray(by_frame[str(sample_index)], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Expected 4x4 matrix for sample {sample_index}, got {matrix.shape}."
            )
        return matrix

    # -------------------------------------------------------------------------
    # INJECT YOUR CAMERA TRANSFORMS HERE
    # -------------------------------------------------------------------------
    # If you don't pass --extrinsics-json, edit this block to return the
    # world_from_camera matrix for your chosen sample.
    #
    # Expected convention: OpenCV / SHARP (x right, y down, z forward).
    # Matrix convention (column-vector):
    #   X_world = world_from_camera @ X_camera_homogeneous
    #
    # Example:
    # if sample_index == 0:
    #     return np.array([
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 0, 1, 2],
    #         [0, 0, 0, 1],
    #     ], dtype=np.float64)
    # -------------------------------------------------------------------------

    return np.eye(4, dtype=np.float64)


def align_ply_to_camera_world(input_ply: Path, output_ply: Path, world_from_camera: np.ndarray):
    """Transform Gaussian means/orientations into target world frame and save."""
    gaussians, metadata = load_ply(input_ply)

    transform = torch.from_numpy(world_from_camera[:3]).to(dtype=torch.float32)
    aligned = apply_transform(gaussians, transform)

    # save_ply stores only a scalar focal and centered principal point metadata.
    # Alignment itself is encoded in Gaussian coordinates via the transform above.
    save_ply(aligned, metadata.focal_length_px, metadata.resolution_px[::-1], output_ply)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abc", type=Path, required=True, help="Path to Alembic .abc")
    parser.add_argument(
        "--camera-path",
        type=str,
        required=True,
        help="Alembic camera object path relative to archive top, e.g. /cam",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        default=None,
        help="Optional JSON with world_from_camera matrices by frame index.",
    )

    args = parser.parse_args()

    sample, num_samples = read_camera_sample(args.abc, args.camera_path, args.sample_index)
    k = camera_sample_to_intrinsics_px(sample, args.image_width, args.image_height)
    world_from_camera = load_world_from_camera(args.sample_index, args.extrinsics_json)

    align_ply_to_camera_world(args.input_ply, args.output_ply, world_from_camera)

    print("Converted camera sample:")
    print(f"- Alembic camera samples available: {num_samples}")
    print(f"- Selected sample index: {args.sample_index}")
    print("- Intrinsics K (pixels):")
    print(np.array2string(k, precision=6, suppress_small=False))
    print("- world_from_camera (4x4):")
    print(np.array2string(world_from_camera, precision=6, suppress_small=False))
    print(f"- Wrote aligned ply: {args.output_ply}")


if __name__ == "__main__":
    main()
