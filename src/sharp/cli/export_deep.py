"""Contains `sharp export-deep` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch

from sharp.utils import logging as logging_utils
from sharp.utils.deep_exr import export_deep_exr_from_ply

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the PLY file or directory of PLY files.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    help="Path to save the deep EXR files.",
    required=True,
)
@click.option(
    "--max-samples",
    type=int,
    default=64,
    help="Maximum number of samples per pixel.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.01,
    help="Minimum alpha contribution threshold.",
)
@click.option(
    "--downsample",
    type=int,
    default=None,
    help="Downsample to this many Gaussians (for testing/faster export).",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def export_deep_cli(
    input_path: Path,
    output_path: Path,
    max_samples: int,
    threshold: float,
    downsample: int | None,
    verbose: bool,
):
    """Export Gaussian splats to OpenEXR deep data format.

    This command converts 3D Gaussian splat PLY files to OpenEXR deep images
    where each pixel contains multiple depth samples with RGBA values.
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if input_path.suffix == ".ply":
        scene_paths = [input_path]
        if output_path.suffix != ".exr":
            output_path.mkdir(exist_ok=True, parents=True)
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)

    LOGGER.info("Found %d PLY files to process.", len(scene_paths))

    for scene_path in scene_paths:
        if output_path.suffix == ".exr":
            out_file = output_path
        else:
            out_file = (output_path / scene_path.stem).with_suffix(".exr")

        LOGGER.info("Exporting %s -> %s", scene_path, out_file)

        export_deep_exr_from_ply(
            ply_path=scene_path,
            output_path=out_file,
            max_samples_per_pixel=max_samples,
            contribution_threshold=threshold,
            downsample_gaussians=downsample,
        )

    LOGGER.info("Export complete.")
