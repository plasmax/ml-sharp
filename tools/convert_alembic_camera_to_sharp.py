#!/usr/bin/env python3
"""Convert Alembic camera data to SHARP intrinsics/extrinsics and align a Gaussian .ply.

By default, this tool derives `world_from_camera` from the Alembic xform chain
above the selected camera object and converts intrinsics from camera sample parameters
(including lens squeeze applied as horizontal aperture scaling).
You can override extrinsics with --extrinsics-json.
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
        from alembic.Abc import IArchive, ISampleSelector  # type: ignore
        from alembic.AbcGeom import ICamera, IXform  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Alembic Python bindings are required. Install a package that provides "
            "`alembic.Abc` and `alembic.AbcGeom` in your environment."
        ) from exc
    return IArchive, ICamera, IXform, ISampleSelector


def _get_child_by_name(parent, child_name: str):
    for child_idx in range(parent.getNumChildren()):
        child = parent.getChild(child_idx)
        if child.getName() == child_name:
            return child
    return None


def _resolve_object_by_path(root, object_path: str):
    current = root
    for part in [part for part in object_path.strip("/").split("/") if part]:
        next_obj = _get_child_by_name(current, part)
        if next_obj is None:
            return None
        current = next_obj
    return current


def _open_camera_by_full_path(root, camera_path: str, ICamera):
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
        return _open_camera_by_full_path(root, camera_path, ICamera), camera_path

    subtree_cameras = _collect_camera_paths(obj, ICamera, prefix=camera_path.rsplit("/", 1)[0])
    if len(subtree_cameras) == 1:
        resolved_path = subtree_cameras[0]
        return _open_camera_by_full_path(root, resolved_path, ICamera), resolved_path
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


def _matrix44_to_numpy(matrix44) -> np.ndarray:
    """Convert Alembic M44* to a numpy matrix with translation in last column.

    Alembic/imath bindings may expose M44 indexing in a convention that appears
    transposed relative to DCC world matrices (e.g. Nuke). We normalize to the
    common 4x4 form used by this script:

        [ R00 R01 R02 tx ]
        [ R10 R11 R12 ty ]
        [ R20 R21 R22 tz ]
        [  0   0   0   1 ]
    """
    try:
        m = np.array([[float(matrix44[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    except Exception:
        flat = list(matrix44)
        if len(flat) != 16:
            raise ValueError("Unable to convert Alembic 4x4 matrix to numpy array.")
        m = np.asarray(flat, dtype=np.float64).reshape(4, 4)

    # If translation shows up in the last row instead of the last column,
    # transpose to match standard world-matrix layout.
    if np.allclose(m[:3, 3], 0.0) and not np.allclose(m[3, :3], 0.0):
        m = m.T

    return m


def world_from_camera_from_alembic(archive, camera_path: str, sample_index: int, IXform, ISampleSelector) -> np.ndarray:
    """Compute world_from_camera by composing IXform matrices along the camera path."""
    root = archive.getTop()
    current = root
    world_from_camera = np.eye(4, dtype=np.float64)

    parts = [part for part in camera_path.strip("/").split("/") if part]
    traversed_parts: list[str] = []
    for part in parts:
        child = _get_child_by_name(current, part)
        if child is None:
            raise ValueError(f"Path segment not found while extracting xforms: {'/'.join(traversed_parts + [part])}")

        traversed_parts.append(part)
        if IXform.matches(child.getHeader()):
            xform = IXform(current, part)
            schema = xform.getSchema()
            num_samples = schema.getNumSamples()
            idx = min(sample_index, max(0, num_samples - 1))
            sample = schema.getValue(ISampleSelector(idx))
            xform_matrix = _matrix44_to_numpy(sample.getMatrix())
            world_from_camera = world_from_camera @ xform_matrix

        current = child

    return world_from_camera


def read_camera_sample(archive, camera_path: str, sample_index: int, ICamera, ISampleSelector):
    camera_obj, resolved_path = _open_camera_from_path(archive, camera_path, ICamera)
    if not camera_obj.valid():
        raise ValueError(f"Invalid camera object in archive: {camera_path}")

    schema = camera_obj.getSchema()
    num_samples = schema.getNumSamples()
    if num_samples <= 0:
        raise ValueError("Camera schema has no samples.")
    if sample_index < 0 or sample_index >= num_samples:
        raise ValueError(f"sample_index={sample_index} out of range [0, {num_samples - 1}]")

    sample = schema.getValue(ISampleSelector(sample_index))
    return sample, num_samples, resolved_path


def camera_sample_to_intrinsics_px(
    sample,
    image_width: int,
    image_height: int,
    override_focal_mm: float | None = None,
    ignore_film_offset: bool = False,
) -> np.ndarray:
    focal_mm = float(sample.getFocalLength()) if override_focal_mm is None else float(override_focal_mm)
    h_aperture_cm = float(sample.getHorizontalAperture())
    v_aperture_cm = float(sample.getVerticalAperture())
    h_offset_cm = float(sample.getHorizontalFilmOffset())
    v_offset_cm = float(sample.getVerticalFilmOffset())

    lens_squeeze = float(sample.getLensSqueezeRatio())
    if lens_squeeze == 0:
        raise ValueError("Lens squeeze ratio is zero; cannot compute fx.")

    h_aperture_mm = h_aperture_cm * 10.0
    v_aperture_mm = v_aperture_cm * 10.0
    h_offset_mm = 0.0 if ignore_film_offset else (h_offset_cm * 10.0)
    v_offset_mm = 0.0 if ignore_film_offset else (v_offset_cm * 10.0)

    # Alembic lens squeeze scales horizontal aperture (anamorphic behaviour).
    # Use effective aperture in camera space: h_aperture / lens_squeeze.
    h_aperture_effective_mm = h_aperture_mm / lens_squeeze
    h_offset_effective_mm = h_offset_mm / lens_squeeze

    fx = focal_mm * (image_width / h_aperture_effective_mm)
    fy = focal_mm * (image_height / v_aperture_mm)
    cx = image_width * 0.5 + (h_offset_effective_mm / h_aperture_effective_mm) * image_width
    cy = image_height * 0.5 + (v_offset_mm / v_aperture_mm) * image_height

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_world_from_camera(
    archive,
    resolved_camera_path: str,
    sample_index: int,
    extrinsics_json_path: Path | None,
    IXform,
    ISampleSelector,
) -> tuple[np.ndarray, str]:
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
            raise ValueError(f"Expected 4x4 matrix for sample {sample_index}, got {matrix.shape}.")
        return matrix, "json"

    matrix = world_from_camera_from_alembic(
        archive, resolved_camera_path, sample_index, IXform=IXform, ISampleSelector=ISampleSelector
    )
    return matrix, "alembic_xform_chain"


def convert_world_from_camera_frame(world_from_camera: np.ndarray, flip_camera_yz: bool) -> np.ndarray:
    """Convert camera basis to SHARP/OpenCV camera basis if requested."""
    if not flip_camera_yz:
        return world_from_camera
    camera_basis_fix = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    return world_from_camera @ camera_basis_fix


def align_ply_to_camera_world(input_ply: Path, output_ply: Path, world_from_camera: np.ndarray):
    gaussians, metadata = load_ply(input_ply)
    transform = torch.from_numpy(world_from_camera[:3]).to(dtype=torch.float32)
    aligned = apply_transform(gaussians, transform)
    save_ply(aligned, metadata.focal_length_px, metadata.resolution_px[::-1], output_ply)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abc", type=Path, required=True, help="Path to Alembic .abc")
    parser.add_argument(
        "--camera-path",
        type=str,
        required=False,
        default=None,
        help="Full camera object path, e.g. /Camera01/.../cameraLeftShape",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List camera object paths and exit.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--override-focal-length-mm",
        type=float,
        default=None,
        help="Optional focal length override in mm (uses Alembic focal when omitted).",
    )
    parser.add_argument(
        "--ignore-film-offset",
        action="store_true",
        help="Ignore Alembic horizontal/vertical film offsets (window translate).",
    )
    parser.add_argument(
        "--no-camera-yz-flip",
        action="store_true",
        help="Disable conversion from DCC camera basis to OpenCV basis via camera Y/Z sign flip.",
    )
    parser.add_argument("--image-width", type=int, required=False)
    parser.add_argument("--image-height", type=int, required=False)
    parser.add_argument("--input-ply", type=Path, required=False)
    parser.add_argument("--output-ply", type=Path, required=False)
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        default=None,
        help="Optional JSON override with world_from_camera matrices by frame index.",
    )

    args = parser.parse_args()
    IArchive, ICamera, IXform, ISampleSelector = _require_alembic_modules()
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

    required = [
        ("--camera-path", args.camera_path),
        ("--image-width", args.image_width),
        ("--image-height", args.image_height),
        ("--input-ply", args.input_ply),
        ("--output-ply", args.output_ply),
    ]
    missing = [name for name, value in required if value is None]
    if missing:
        raise ValueError("Missing required argument(s) for conversion: " + ", ".join(missing))

    sample, num_samples, resolved_camera_path = read_camera_sample(
        archive, args.camera_path, args.sample_index, ICamera=ICamera, ISampleSelector=ISampleSelector
    )
    k = camera_sample_to_intrinsics_px(
        sample,
        args.image_width,
        args.image_height,
        override_focal_mm=args.override_focal_length_mm,
        ignore_film_offset=args.ignore_film_offset,
    )
    world_from_camera, extrinsics_source = load_world_from_camera(
        archive,
        resolved_camera_path,
        args.sample_index,
        args.extrinsics_json,
        IXform=IXform,
        ISampleSelector=ISampleSelector,
    )

    world_from_camera = convert_world_from_camera_frame(
        world_from_camera,
        flip_camera_yz=(not args.no_camera_yz_flip),
    )

    align_ply_to_camera_world(args.input_ply, args.output_ply, world_from_camera)

    print("Converted camera sample:")
    print(f"- Alembic camera samples available: {num_samples}")
    print(f"- Selected sample index: {args.sample_index}")
    print(f"- Resolved camera path: {resolved_camera_path}")
    print(f"- Extrinsics source: {extrinsics_source}")
    print(f"- Camera Y/Z flip applied: {not args.no_camera_yz_flip}")
    print(f"- Film offset ignored: {args.ignore_film_offset}")
    print(f"- Focal override mm: {args.override_focal_length_mm}")
    print("- Camera sample parameters:")
    print(f"  focal_length_mm={float(sample.getFocalLength())}")
    print(f"  lens_squeeze_ratio={float(sample.getLensSqueezeRatio())}")
    print(f"  horizontal_aperture_cm={float(sample.getHorizontalAperture())}")
    print(f"  vertical_aperture_cm={float(sample.getVerticalAperture())}")
    print(f"  horizontal_film_offset_cm={float(sample.getHorizontalFilmOffset())}")
    print(f"  vertical_film_offset_cm={float(sample.getVerticalFilmOffset())}")
    print("- Intrinsics K (pixels):")
    print(np.array2string(k, precision=6, suppress_small=False))
    print("- world_from_camera (4x4):")
    print(np.array2string(world_from_camera, precision=6, suppress_small=False))
    print(f"- Wrote aligned ply: {args.output_ply}")


if __name__ == "__main__":
    main()
