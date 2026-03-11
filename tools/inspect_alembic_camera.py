#!/usr/bin/env python3
"""Inspect Alembic camera sample and report intrinsics conversion candidates.

This script is for debugging focal-length / window-translate mismatches between
DCC exports (e.g. Nuke) and SHARP/OpenCV conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require_alembic_modules():
    from alembic.Abc import IArchive, ISampleSelector  # type: ignore
    from alembic.AbcGeom import ICamera  # type: ignore

    return IArchive, ICamera, ISampleSelector


def _get_child_by_name(parent, child_name: str):
    for child_idx in range(parent.getNumChildren()):
        child = parent.getChild(child_idx)
        if child.getName() == child_name:
            return child
    return None


def _resolve_object_by_path(root, object_path: str):
    current = root
    for part in [part for part in object_path.strip("/").split("/") if part]:
        nxt = _get_child_by_name(current, part)
        if nxt is None:
            return None
        current = nxt
    return current


def _open_camera(root, camera_path: str, ICamera):
    normalized = "/" + camera_path.strip("/")
    parent_path, _, child_name = normalized.rpartition("/")
    parent_obj = root if parent_path in {"", "/"} else _resolve_object_by_path(root, parent_path)
    if parent_obj is None:
        raise ValueError(f"Camera parent path not found: {parent_path}")
    return ICamera(parent_obj, child_name)


def _matrix33_to_numpy(matrix33) -> np.ndarray:
    try:
        m = np.array([[float(matrix33[i][j]) for j in range(3)] for i in range(3)], dtype=np.float64)
    except Exception:
        flat = list(matrix33)
        if len(flat) != 9:
            raise ValueError("Unable to convert Alembic 3x3 matrix to numpy array.")
        m = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    if np.allclose(m[:2, 2], 0.0) and not np.allclose(m[2, :2], 0.0):
        m = m.T
    return m


def _apply_filmback_matrix(
    sample,
    h_aperture_mm: float,
    v_aperture_mm: float,
    h_offset_mm: float,
    v_offset_mm: float,
    apply_translation: bool,
) -> tuple[float, float, float, float]:
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
    if not apply_translation:
        h_off = 0.0
        v_off = 0.0
    return h_new, v_new, h_off, v_off


def compute_k(sample, width: int, height: int, ignore_film_offset: bool, apply_filmback_translation: bool):
    focal_mm = float(sample.getFocalLength())
    lens_squeeze = float(sample.getLensSqueezeRatio())

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
        apply_translation=apply_filmback_translation and (not ignore_film_offset),
    )

    h_aperture_effective_mm = h_aperture_mm / lens_squeeze
    h_offset_effective_mm = h_offset_mm / lens_squeeze

    fx = focal_mm * (width / h_aperture_effective_mm)
    fy = focal_mm * (height / v_aperture_mm)
    cx = width * 0.5 + (h_offset_effective_mm / h_aperture_effective_mm) * width
    cy = height * 0.5 + (v_offset_mm / v_aperture_mm) * height

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--camera-path", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    args = parser.parse_args()

    IArchive, ICamera, ISampleSelector = _require_alembic_modules()
    archive = IArchive(str(args.abc))
    root = archive.getTop()
    cam = _open_camera(root, args.camera_path, ICamera)
    schema = cam.getSchema()
    sample = schema.getValue(ISampleSelector(args.sample_index))

    print("Camera sample raw values")
    print(f"- focal_length_mm: {float(sample.getFocalLength())}")
    print(f"- lens_squeeze_ratio: {float(sample.getLensSqueezeRatio())}")
    print(f"- horizontal_aperture_cm: {float(sample.getHorizontalAperture())}")
    print(f"- vertical_aperture_cm: {float(sample.getVerticalAperture())}")
    print(f"- horizontal_film_offset_cm: {float(sample.getHorizontalFilmOffset())}")
    print(f"- vertical_film_offset_cm: {float(sample.getVerticalFilmOffset())}")
    if hasattr(sample, "getFilmBackMatrix"):
        print("- filmback_matrix_3x3:")
        print(np.array2string(_matrix33_to_numpy(sample.getFilmBackMatrix()), precision=6))

    modes = [
        ("raw_offsets_no_filmback_translation", False, False),
        ("raw_offsets_with_filmback_translation", False, True),
        ("ignore_offsets", True, False),
    ]
    print("\nCandidate OpenCV intrinsics K:")
    for name, ignore_offset, apply_fb_tx in modes:
        k = compute_k(
            sample,
            width=args.image_width,
            height=args.image_height,
            ignore_film_offset=ignore_offset,
            apply_filmback_translation=apply_fb_tx,
        )
        print(f"\n[{name}] ignore_film_offset={ignore_offset} apply_filmback_translation={apply_fb_tx}")
        print(np.array2string(k, precision=6, suppress_small=False))


if __name__ == "__main__":
    main()
