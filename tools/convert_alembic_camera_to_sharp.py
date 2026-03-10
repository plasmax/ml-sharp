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
      --camera-path /Camera01/camera/.../render_:cameraLeft_LOCShape \
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


def _get_child_by_name(parent, child_name: str):
    """Return direct child object by name or None."""
    for child_idx in range(parent.getNumChildren()):
        child = parent.getChild(child_idx)
        if child.getName() == child_name:
            return child
    return None


def _resolve_object_by_path(root, object_path: str):
    """Resolve an Alembic object from an absolute/relative path."""
    current = root
    path_parts = [part for part in object_path.strip("/").split("/") if part]
    for part in path_parts:
        next_obj = _get_child_by_name(current, part)
        if next_obj is None:
            return None
        current = next_obj
    return current


def _open_camera_by_full_path(root, camera_path: str, ICamera):
    """Construct ICamera from full path using (parent, child_name) ctor."""
    normalized = "/" + camera_path.strip("/")
    parent_path, _, child_name = normalized.rpartition("/")
    if not child_name:
        raise ValueError(f"Invalid camera path: {camera_path}")

    parent_obj = root if parent_path in {"", "/"} else _resolve_object_by_path(root, parent_path)
    if parent_obj is None:
        raise ValueError(f"Camera parent path not found: {parent_path}")

    try:
        return ICamera(parent_obj, child_name)
    except RuntimeError as exc:
        raise ValueError(
            "Failed to open camera at path "
            f"{normalized}. Ensure the path points to a camera schema object "
            "(e.g. ...Shape), or run --list-cameras to choose a valid path."
        ) from exc


def _collect_camera_paths(root, ICamera, prefix: str = "") -> list[str]:
    """List camera object paths under `root`."""
    camera_paths: list[str] = []

    def _dfs(obj, current_prefix: str) -> None:
        path = f"{current_prefix}/{obj.getName()}" if current_prefix else f"/{obj.getName()}"
        if ICamera.matches(obj.getHeader()):
            camera_paths.append(path)
        for child_idx in range(obj.getNumChildren()):
            _dfs(obj.getChild(child_idx), path)

    _dfs(root, prefix)
    return camera_paths


def _open_camera_from_path(archive, camera_path: str, ICamera):
    """Open ICamera robustly from a full object path."""
    root = archive.getTop()
    obj = _resolve_object_by_path(root, camera_path)
    if obj is None:
        available: list[str] = []
        for child_idx in range(root.getNumChildren()):
            available.extend(_collect_camera_paths(root.getChild(child_idx), ICamera))
        sample = "\n".join(f"  - {p}" for p in available[:20])
        raise ValueError(
            f"Camera path not found: {camera_path}\n"
            "Use --list-cameras to inspect available camera objects.\n"
            f"First available camera paths:\n{sample}"
        )

    if ICamera.matches(obj.getHeader()):
        return _open_camera_by_full_path(root, camera_path, ICamera)

    # Fallback: user may pass a parent transform path. Pick first camera below it.
    subtree_cameras = _collect_camera_paths(obj, ICamera, prefix=camera_path.rsplit("/", 1)[0])
    if len(subtree_cameras) == 1:
        return _open_camera_by_full_path(root, subtree_cameras[0], ICamera)
    if len(subtree_cameras) > 1:
        candidates = "\n".join(f"  - {p}" for p in subtree_cameras)
        raise ValueError(
            f"Path exists but is not a camera object: {camera_path}\n"
            "Multiple camera objects found under that subtree; pass one exact camera path:\n"
            f"{candidates}"
        )

    raise ValueError(
        f"Path exists but is not a camera object and has no camera descendants: {camera_path}"
    )


def read_camera_sample(abc_path: Path, camera_path: str, sample_index: int):
    """Read Alembic CameraSample and return basic camera parameters."""
    IArchive, ICamera = _require_alembic_modules()

    archive = IArchive(str(abc_path))
    camera_obj = _open_camera_from_path(archive, camera_path, ICamera)
    if not camera_obj.valid():
        raise ValueError(f"Invalid camera object in archive: {camera_path}")

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
        required=False,
        default=None,
        help=(
            "Full camera object path, e.g. "
            "/Camera01/camera/.../render_:cameraLeft_LOCShape"
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List camera object paths inside the Alembic archive and exit.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-width", type=int, required=False)
    parser.add_argument("--image-height", type=int, required=False)
    parser.add_argument("--input-ply", type=Path, required=False)
    parser.add_argument("--output-ply", type=Path, required=False)
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        default=None,
        help="Optional JSON with world_from_camera matrices by frame index.",
    )

    args = parser.parse_args()
    IArchive, ICamera = _require_alembic_modules()
    archive = IArchive(str(args.abc))

    if args.list_cameras:
        camera_paths: list[str] = []
        root = archive.getTop()
        for child_idx in range(root.getNumChildren()):
            camera_paths.extend(_collect_camera_paths(root.getChild(child_idx), ICamera))
        if not camera_paths:
            print("No camera objects were found in this archive.")
        else:
            print("Camera objects:")
            for path in camera_paths:
                print(path)
        return

    required_when_converting = [
        ("--camera-path", args.camera_path),
        ("--image-width", args.image_width),
        ("--image-height", args.image_height),
        ("--input-ply", args.input_ply),
        ("--output-ply", args.output_ply),
    ]
    missing = [name for name, value in required_when_converting if value is None]
    if missing:
        raise ValueError(
            "Missing required argument(s) for conversion: " + ", ".join(missing)
        )

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
