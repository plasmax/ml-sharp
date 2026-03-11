#!/usr/bin/env python3
"""Convert Alembic camera data to SHARP intrinsics/extrinsics and align Gaussian .ply files."""

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
        current = _get_child_by_name(current, part)
        if current is None:
            return None
    return current


def _open_camera_by_full_path(root, camera_path: str, ICamera):
    normalized = "/" + camera_path.strip("/")
    parent_path, _, child_name = normalized.rpartition("/")
    if not child_name:
        raise ValueError(f"Invalid camera path: {camera_path}")
    parent_obj = root if parent_path in {"", "/"} else _resolve_object_by_path(root, parent_path)
    if parent_obj is None:
        raise ValueError(f"Camera parent path not found: {parent_path}")
    return ICamera(parent_obj, child_name)


def _collect_camera_paths(root, ICamera, prefix: str = "") -> list[str]:
    paths: list[str] = []

    def _dfs(obj, current_prefix: str) -> None:
        path = f"{current_prefix}/{obj.getName()}" if current_prefix else f"/{obj.getName()}"
        if ICamera.matches(obj.getHeader()):
            paths.append(path)
        for child_idx in range(obj.getNumChildren()):
            _dfs(obj.getChild(child_idx), path)

    _dfs(root, prefix)
    return paths


def _open_camera_from_path(archive, camera_path: str, ICamera):
    root = archive.getTop()
    obj = _resolve_object_by_path(root, camera_path)
    if obj is None:
        available: list[str] = []
        for child_idx in range(root.getNumChildren()):
            available.extend(_collect_camera_paths(root.getChild(child_idx), ICamera))
        sample = "\n".join(f"  - {p}" for p in available[:20])
        raise ValueError(f"Camera path not found: {camera_path}\nFirst available:\n{sample}")

    if ICamera.matches(obj.getHeader()):
        return _open_camera_by_full_path(root, camera_path, ICamera), camera_path

    subtree = _collect_camera_paths(obj, ICamera, prefix=camera_path.rsplit("/", 1)[0])
    if len(subtree) == 1:
        return _open_camera_by_full_path(root, subtree[0], ICamera), subtree[0]
    if len(subtree) > 1:
        candidates = "\n".join(f"  - {p}" for p in subtree)
        raise ValueError(f"Ambiguous camera path {camera_path}. Candidates:\n{candidates}")
    raise ValueError(f"Path exists but has no camera descendants: {camera_path}")


def _matrix44_to_numpy(matrix44) -> np.ndarray:
    try:
        m = np.array([[float(matrix44[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    except Exception:
        flat = list(matrix44)
        if len(flat) != 16:
            raise ValueError("Unable to convert Alembic 4x4 matrix.")
        m = np.asarray(flat, dtype=np.float64).reshape(4, 4)
    if np.allclose(m[:3, 3], 0.0) and not np.allclose(m[3, :3], 0.0):
        m = m.T
    return m


def _matrix33_to_numpy(matrix33) -> np.ndarray:
    try:
        m = np.array([[float(matrix33[i][j]) for j in range(3)] for i in range(3)], dtype=np.float64)
    except Exception:
        flat = list(matrix33)
        if len(flat) != 9:
            raise ValueError("Unable to convert Alembic 3x3 matrix.")
        m = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    if np.allclose(m[:2, 2], 0.0) and not np.allclose(m[2, :2], 0.0):
        m = m.T
    return m


def _sample_selector(ISampleSelector, sample_index: int | None = None, sample_time: float | None = None):
    if sample_time is not None:
        return ISampleSelector(float(sample_time))
    if sample_index is None:
        sample_index = 0
    return ISampleSelector(int(sample_index))


def _apply_filmback_matrix(sample, h_aperture_mm, v_aperture_mm, h_offset_mm, v_offset_mm, apply_tx):
    if not hasattr(sample, "getFilmBackMatrix"):
        return h_aperture_mm, v_aperture_mm, h_offset_mm, v_offset_mm

    filmback = _matrix33_to_numpy(sample.getFilmBackMatrix())
    half_w = 0.5 * h_aperture_mm
    half_h = 0.5 * v_aperture_mm
    corners = np.array(
        [
            [h_offset_mm - half_w, v_offset_mm - half_h, 1.0],
            [h_offset_mm + half_w, v_offset_mm - half_h, 1.0],
            [h_offset_mm - half_w, v_offset_mm + half_h, 1.0],
            [h_offset_mm + half_w, v_offset_mm + half_h, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = (filmback @ corners.T).T
    x_min, y_min = transformed[:, 0].min(), transformed[:, 1].min()
    x_max, y_max = transformed[:, 0].max(), transformed[:, 1].max()
    h_new = x_max - x_min
    v_new = y_max - y_min
    h_off = 0.5 * (x_min + x_max)
    v_off = 0.5 * (y_min + y_max)
    if not apply_tx:
        h_off = 0.0
        v_off = 0.0
    return h_new, v_new, h_off, v_off


def camera_sample_to_intrinsics_px(
    sample,
    image_width,
    image_height,
    override_focal_mm=None,
    ignore_film_offset=True,
    apply_filmback_translation=False,
):
    focal_mm = float(sample.getFocalLength()) if override_focal_mm is None else float(override_focal_mm)
    lens_squeeze = float(sample.getLensSqueezeRatio())
    if lens_squeeze == 0:
        raise ValueError("Lens squeeze ratio is zero.")

    h_aperture_mm = float(sample.getHorizontalAperture()) * 10.0
    v_aperture_mm = float(sample.getVerticalAperture()) * 10.0
    h_offset_mm = 0.0 if ignore_film_offset else float(sample.getHorizontalFilmOffset()) * 10.0
    v_offset_mm = 0.0 if ignore_film_offset else float(sample.getVerticalFilmOffset()) * 10.0

    h_aperture_mm, v_aperture_mm, h_offset_mm, v_offset_mm = _apply_filmback_matrix(
        sample,
        h_aperture_mm,
        v_aperture_mm,
        h_offset_mm,
        v_offset_mm,
        apply_tx=apply_filmback_translation and (not ignore_film_offset),
    )

    h_aperture_effective_mm = h_aperture_mm / lens_squeeze
    h_offset_effective_mm = h_offset_mm / lens_squeeze

    fx = focal_mm * (float(image_width) / h_aperture_effective_mm)
    fy = focal_mm * (float(image_height) / v_aperture_mm)
    cx = float(image_width) * 0.5 + (h_offset_effective_mm / h_aperture_effective_mm) * float(image_width)
    cy = float(image_height) * 0.5 + (v_offset_mm / v_aperture_mm) * float(image_height)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def read_camera_sample(archive, camera_path, ICamera, ISampleSelector, sample_index=0, sample_time=None):
    camera_obj, resolved_path = _open_camera_from_path(archive, camera_path, ICamera)
    schema = camera_obj.getSchema()
    num_samples = schema.getNumSamples()
    if num_samples <= 0:
        raise ValueError("Camera schema has no samples.")
    if sample_time is None and (sample_index < 0 or sample_index >= num_samples):
        raise ValueError(f"sample_index={sample_index} out of range [0, {num_samples - 1}]")
    sample = schema.getValue(_sample_selector(ISampleSelector, sample_index=sample_index, sample_time=sample_time))
    return sample, num_samples, resolved_path


def world_from_camera_from_alembic(
    archive,
    camera_path,
    IXform,
    ISampleSelector,
    sample_index=0,
    sample_time=None,
):
    root = archive.getTop()
    current = root
    world_from_camera = np.eye(4, dtype=np.float64)

    for part in [part for part in camera_path.strip("/").split("/") if part]:
        child = _get_child_by_name(current, part)
        if child is None:
            raise ValueError(f"Missing path segment while extracting xforms: {part}")
        if IXform.matches(child.getHeader()):
            xform = IXform(current, part)
            schema = xform.getSchema()
            value = schema.getValue(_sample_selector(ISampleSelector, sample_index=sample_index, sample_time=sample_time))
            world_from_camera = world_from_camera @ _matrix44_to_numpy(value.getMatrix())
        current = child
    return world_from_camera


def load_world_from_camera(
    archive,
    resolved_camera_path,
    IXform,
    ISampleSelector,
    sample_index=0,
    sample_time=None,
    extrinsics_json_path=None,
    frame_key=None,
):
    if extrinsics_json_path is not None:
        payload = json.loads(Path(extrinsics_json_path).read_text())
        by_frame = payload.get("world_from_camera", {})
        key = str(frame_key if frame_key is not None else sample_index)
        if key not in by_frame:
            raise KeyError(f"Key {key} not in {extrinsics_json_path} under world_from_camera")
        matrix = np.asarray(by_frame[key], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix for key {key}, got {matrix.shape}")
        return matrix, "json"

    return (
        world_from_camera_from_alembic(
            archive,
            resolved_camera_path,
            IXform,
            ISampleSelector,
            sample_index=sample_index,
            sample_time=sample_time,
        ),
        "alembic_xform_chain",
    )


def convert_world_from_camera_frame(world_from_camera: np.ndarray, flip_camera_yz: bool) -> np.ndarray:
    if not flip_camera_yz:
        return world_from_camera
    return world_from_camera @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def _scale_transform(world_scale: float, anchor_xyz: np.ndarray) -> torch.Tensor:
    linear = np.eye(3, dtype=np.float32) * float(world_scale)
    offset = ((1.0 - float(world_scale)) * anchor_xyz).astype(np.float32)
    return torch.from_numpy(np.concatenate([linear, offset[:, None]], axis=1))


def _intrinsics_remap_transform(source_focal_px: float, source_resolution_px: tuple[int, int], target_k: np.ndarray):
    src_w, src_h = source_resolution_px
    fx_src = float(source_focal_px)
    fy_src = float(source_focal_px)
    cx_src = src_w * 0.5
    cy_src = src_h * 0.5

    fx_tgt = float(target_k[0, 0])
    fy_tgt = float(target_k[1, 1])
    cx_tgt = float(target_k[0, 2])
    cy_tgt = float(target_k[1, 2])

    return torch.from_numpy(
        np.array(
            [
                [fx_src / fx_tgt, 0.0, (cx_src - cx_tgt) / fx_tgt, 0.0],
                [0.0, fy_src / fy_tgt, (cy_src - cy_tgt) / fy_tgt, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )


def align_ply_to_camera_world(
    input_ply,
    output_ply,
    world_from_camera,
    target_k,
    apply_intrinsics_remap,
    world_scale,
    scale_anchor,
):
    gaussians, metadata = load_ply(input_ply)
    if apply_intrinsics_remap:
        gaussians = apply_transform(
            gaussians,
            _intrinsics_remap_transform(metadata.focal_length_px, metadata.resolution_px, target_k),
        )

    gaussians = apply_transform(gaussians, torch.from_numpy(world_from_camera[:3]).to(dtype=torch.float32))

    if world_scale != 1.0:
        if scale_anchor == "camera":
            anchor_xyz = np.asarray(world_from_camera[:3, 3], dtype=np.float32)
        elif scale_anchor == "origin":
            anchor_xyz = np.zeros(3, dtype=np.float32)
        else:
            raise ValueError(f"Invalid scale_anchor={scale_anchor}")
        gaussians = apply_transform(gaussians, _scale_transform(world_scale, anchor_xyz))

    save_ply(gaussians, metadata.focal_length_px, metadata.resolution_px[::-1], output_ply)


def _path_for_frame(path_template: Path, frame: int, require_placeholder: bool) -> Path:
    text = str(path_template)
    has_placeholder = "{frame" in text
    if require_placeholder and not has_placeholder:
        raise ValueError(f"Expected '{{frame}}' placeholder in path template: {path_template}")
    return Path(text.format(frame=frame)) if has_placeholder else path_template


def _process_single(args, archive, ICamera, IXform, ISampleSelector):
    sample, num_samples, resolved_camera_path = read_camera_sample(
        archive,
        args.camera_path,
        ICamera,
        ISampleSelector,
        sample_index=args.sample_index,
        sample_time=None,
    )

    ignore_film_offset = args.ignore_film_offset or (not args.use_film_offset)
    k = camera_sample_to_intrinsics_px(
        sample,
        args.image_width,
        args.image_height,
        override_focal_mm=args.override_focal_length_mm,
        ignore_film_offset=ignore_film_offset,
        apply_filmback_translation=args.apply_filmback_translation,
    )

    world_from_camera, source = load_world_from_camera(
        archive,
        resolved_camera_path,
        IXform,
        ISampleSelector,
        sample_index=args.sample_index,
        sample_time=None,
        extrinsics_json_path=args.extrinsics_json,
    )
    world_from_camera = convert_world_from_camera_frame(world_from_camera, flip_camera_yz=(not args.no_camera_yz_flip))

    align_ply_to_camera_world(
        args.input_ply,
        args.output_ply,
        world_from_camera,
        target_k=k,
        apply_intrinsics_remap=(not args.no_intrinsics_remap),
        world_scale=args.world_scale,
        scale_anchor=args.scale_anchor,
    )

    print("Converted camera sample:")
    print(f"- Alembic camera samples available: {num_samples}")
    print(f"- Selected sample index: {args.sample_index}")
    print(f"- Resolved camera path: {resolved_camera_path}")
    print(f"- Extrinsics source: {source}")
    print(f"- Camera Y/Z flip applied: {not args.no_camera_yz_flip}")
    print(f"- Intrinsics remap applied: {not args.no_intrinsics_remap}")
    print(f"- Film offset ignored: {ignore_film_offset}")
    print(f"- Filmback translation applied: {args.apply_filmback_translation}")
    print(f"- Focal override mm: {args.override_focal_length_mm}")
    print(f"- World scale: {args.world_scale}")
    print(f"- Scale anchor: {args.scale_anchor}")
    print("- Intrinsics K (pixels):")
    print(np.array2string(k, precision=6, suppress_small=False))
    print("- world_from_camera (4x4):")
    print(np.array2string(world_from_camera, precision=6, suppress_small=False))
    print(f"- Wrote aligned ply: {args.output_ply}")


def _process_sequence(args, archive, ICamera, IXform, ISampleSelector):
    require_placeholder = True
    ignore_film_offset = args.ignore_film_offset or (not args.use_film_offset)

    _, _, resolved_camera_path = read_camera_sample(
        archive,
        args.camera_path,
        ICamera,
        ISampleSelector,
        sample_index=args.sample_index,
        sample_time=None,
    )

    for frame in range(args.start_frame, args.end_frame + 1):
        sample_time = float(frame) / float(args.fps)
        sample, _, _ = read_camera_sample(
            archive,
            args.camera_path,
            ICamera,
            ISampleSelector,
            sample_time=sample_time,
        )
        k = camera_sample_to_intrinsics_px(
            sample,
            args.image_width,
            args.image_height,
            override_focal_mm=args.override_focal_length_mm,
            ignore_film_offset=ignore_film_offset,
            apply_filmback_translation=args.apply_filmback_translation,
        )
        world_from_camera, _ = load_world_from_camera(
            archive,
            resolved_camera_path,
            IXform,
            ISampleSelector,
            sample_time=sample_time,
            extrinsics_json_path=args.extrinsics_json,
            frame_key=frame,
        )
        world_from_camera = convert_world_from_camera_frame(
            world_from_camera,
            flip_camera_yz=(not args.no_camera_yz_flip),
        )

        input_ply = _path_for_frame(args.input_ply, frame, require_placeholder=require_placeholder)
        output_ply = _path_for_frame(args.output_ply, frame, require_placeholder=require_placeholder)
        if not input_ply.exists():
            print(f"[WARN] missing input for frame {frame}: {input_ply}")
            continue

        output_ply.parent.mkdir(parents=True, exist_ok=True)
        align_ply_to_camera_world(
            input_ply,
            output_ply,
            world_from_camera,
            target_k=k,
            apply_intrinsics_remap=(not args.no_intrinsics_remap),
            world_scale=args.world_scale,
            scale_anchor=args.scale_anchor,
        )
        print(f"[OK] frame {frame}: {input_ply} -> {output_ply}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--camera-path", type=str, required=False, default=None)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--override-focal-length-mm", type=float, default=None)
    parser.add_argument("--ignore-film-offset", action="store_true")
    parser.add_argument(
        "--use-film-offset",
        action="store_true",
        help="Use Alembic film/window offsets in principal point. Default behavior ignores them.",
    )
    parser.add_argument("--no-camera-yz-flip", action="store_true")
    parser.add_argument("--no-intrinsics-remap", action="store_true")
    parser.add_argument("--apply-filmback-translation", action="store_true")
    parser.add_argument("--world-scale", type=float, default=10.0)
    parser.add_argument("--scale-anchor", choices=["camera", "origin"], default="camera")
    parser.add_argument("--image-width", type=int, required=False)
    parser.add_argument("--image-height", type=int, required=False)
    parser.add_argument("--input-ply", type=Path, required=False)
    parser.add_argument("--output-ply", type=Path, required=False)
    parser.add_argument("--extrinsics-json", type=Path, default=None)

    args = parser.parse_args()

    IArchive, ICamera, IXform, ISampleSelector = _require_alembic_modules()
    archive = IArchive(str(args.abc))

    if args.list_cameras:
        root = archive.getTop()
        camera_paths: list[str] = []
        for child_idx in range(root.getNumChildren()):
            camera_paths.extend(_collect_camera_paths(root.getChild(child_idx), ICamera))
        if not camera_paths:
            print("No camera objects found.")
        else:
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
        raise ValueError("Missing required argument(s): " + ", ".join(missing))

    if (args.start_frame is None) ^ (args.end_frame is None):
        raise ValueError("Provide both --start-frame and --end-frame, or neither.")

    if args.start_frame is not None:
        if args.end_frame < args.start_frame:
            raise ValueError("--end-frame must be >= --start-frame")
        _process_sequence(args, archive, ICamera, IXform, ISampleSelector)
    else:
        _process_single(args, archive, ICamera, IXform, ISampleSelector)


if __name__ == "__main__":
    main()
