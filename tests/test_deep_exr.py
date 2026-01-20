"""Tests for deep EXR export functionality.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import tempfile
from pathlib import Path

import numpy as np
import OpenEXR
import pytest
import torch

from sharp.utils.deep_exr import export_deep_exr, export_deep_exr_from_ply
from sharp.utils.gaussians import Gaussians3D, SceneMetaData


def create_synthetic_gaussians(
    n_gaussians: int = 100,
    device: torch.device = torch.device("cpu"),
) -> tuple[Gaussians3D, SceneMetaData]:
    """Create synthetic Gaussian splat data for testing."""
    # Create Gaussians distributed in a 3D volume in front of the camera
    mean_vectors = torch.rand(1, n_gaussians, 3, device=device) * 2 - 1  # [-1, 1]
    mean_vectors[:, :, 2] = mean_vectors[:, :, 2].abs() + 1.0  # Z in [1, 2]

    # Random quaternions (normalized)
    quaternions = torch.randn(1, n_gaussians, 4, device=device)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    # Small scales
    singular_values = torch.ones(1, n_gaussians, 3, device=device) * 0.1

    # Random colors
    colors = torch.rand(1, n_gaussians, 3, device=device)

    # Random opacities
    opacities = torch.rand(1, n_gaussians, device=device) * 0.5 + 0.5  # [0.5, 1.0]

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )

    metadata = SceneMetaData(
        focal_length_px=500.0,
        resolution_px=(640, 480),
        color_space="linearRGB",
    )

    return gaussians, metadata


def composite_deep_image(exr_path: Path) -> np.ndarray:
    """Composite a deep EXR image to a flat image using front-to-back blending."""
    with OpenEXR.File(str(exr_path)) as f:
        Z = f.channels()["Z"].pixels
        # OpenEXR may combine RGBA into a single channel
        rgba_key = "RGBA" if "RGBA" in f.channels() else "R"

        if rgba_key == "RGBA":
            RGBA = f.channels()["RGBA"].pixels
        else:
            R = f.channels()["R"].pixels
            G = f.channels()["G"].pixels
            B = f.channels()["B"].pixels
            A = f.channels()["A"].pixels

        height, width = Z.shape
        result = np.zeros((height, width, 4), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                if Z[y, x] is None or len(Z[y, x]) == 0:
                    continue

                if rgba_key == "RGBA":
                    rgba_samples = RGBA[y, x]
                else:
                    rgba_samples = np.stack(
                        [R[y, x], G[y, x], B[y, x], A[y, x]], axis=-1
                    )

                # Front-to-back compositing
                accumulated_color = np.zeros(3, dtype=np.float32)
                accumulated_alpha = 0.0

                for i in range(len(rgba_samples)):
                    if rgba_key == "RGBA":
                        r, g, b, a = rgba_samples[i]
                    else:
                        r, g, b, a = rgba_samples[i]

                    accumulated_color += (1 - accumulated_alpha) * a * np.array(
                        [r, g, b]
                    )
                    accumulated_alpha += (1 - accumulated_alpha) * a

                    if accumulated_alpha > 0.999:
                        break

                result[y, x, :3] = accumulated_color
                result[y, x, 3] = accumulated_alpha

    return result


class TestDeepExrExport:
    """Tests for deep EXR export."""

    def test_export_creates_valid_exr_file(self):
        """Test that export creates a valid deep EXR file."""
        gaussians, metadata = create_synthetic_gaussians(n_gaussians=50)

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_deep_exr(
                gaussians=gaussians,
                metadata=metadata,
                output_path=output_path,
                max_samples_per_pixel=32,
                contribution_threshold=0.01,
            )

            # Verify file exists
            assert output_path.exists(), "EXR file was not created"

            # Verify it's a valid EXR file
            with OpenEXR.File(str(output_path)) as f:
                channels = list(f.channels().keys())
                # OpenEXR may combine channels
                assert "Z" in channels, "Z channel missing"
                assert (
                    "RGBA" in channels or "R" in channels
                ), "Color channels missing"

        finally:
            output_path.unlink(missing_ok=True)

    def test_export_produces_deep_samples(self):
        """Test that export produces multiple samples per pixel."""
        # Use larger scale to ensure overlap
        gaussians, metadata = create_synthetic_gaussians(n_gaussians=100)
        # Increase scale to ensure overlap
        gaussians = Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values * 2,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_deep_exr(
                gaussians=gaussians,
                metadata=metadata,
                output_path=output_path,
                max_samples_per_pixel=32,
                contribution_threshold=0.001,
            )

            # Read and verify deep structure
            with OpenEXR.File(str(output_path)) as f:
                Z = f.channels()["Z"].pixels

                # Find max samples per pixel
                max_samples = 0
                pixels_with_multiple = 0
                for y in range(Z.shape[0]):
                    for x in range(Z.shape[1]):
                        if Z[y, x] is not None and len(Z[y, x]) > 0:
                            n = len(Z[y, x])
                            max_samples = max(max_samples, n)
                            if n > 1:
                                pixels_with_multiple += 1

                # With overlapping Gaussians, we should have multiple samples
                assert (
                    pixels_with_multiple > 0
                ), "No pixels have multiple depth samples"

        finally:
            output_path.unlink(missing_ok=True)

    def test_samples_sorted_by_depth(self):
        """Test that samples are sorted by depth (front to back)."""
        gaussians, metadata = create_synthetic_gaussians(n_gaussians=100)
        gaussians = Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values * 2,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_deep_exr(
                gaussians=gaussians,
                metadata=metadata,
                output_path=output_path,
                max_samples_per_pixel=32,
                contribution_threshold=0.001,
            )

            with OpenEXR.File(str(output_path)) as f:
                Z = f.channels()["Z"].pixels

                for y in range(Z.shape[0]):
                    for x in range(Z.shape[1]):
                        if Z[y, x] is not None and len(Z[y, x]) > 1:
                            depths = Z[y, x]
                            # Verify depths are sorted
                            assert np.all(
                                depths[:-1] <= depths[1:]
                            ), f"Depths not sorted at ({y}, {x}): {depths}"

        finally:
            output_path.unlink(missing_ok=True)

    def test_composite_produces_valid_image(self):
        """Test that compositing the deep image produces a valid result."""
        gaussians, metadata = create_synthetic_gaussians(n_gaussians=200)

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_deep_exr(
                gaussians=gaussians,
                metadata=metadata,
                output_path=output_path,
                max_samples_per_pixel=32,
                contribution_threshold=0.01,
            )

            # Composite the image
            result = composite_deep_image(output_path)

            # Verify result shape
            width, height = metadata.resolution_px
            assert result.shape == (
                height,
                width,
                4,
            ), f"Unexpected shape: {result.shape}"

            # Verify values are in valid range
            assert np.all(result[:, :, :3] >= 0), "Negative color values"
            assert np.all(result[:, :, :3] <= 1), "Color values > 1"
            assert np.all(result[:, :, 3] >= 0), "Negative alpha values"
            assert np.all(result[:, :, 3] <= 1), "Alpha values > 1"

            # Verify we have some coverage (not all black)
            assert np.any(result[:, :, 3] > 0), "No visible content"

        finally:
            output_path.unlink(missing_ok=True)

    def test_export_respects_max_samples(self):
        """Test that max_samples_per_pixel is respected."""
        gaussians, metadata = create_synthetic_gaussians(n_gaussians=100)
        gaussians = Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values * 3,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

        max_samples = 5

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_deep_exr(
                gaussians=gaussians,
                metadata=metadata,
                output_path=output_path,
                max_samples_per_pixel=max_samples,
                contribution_threshold=0.001,
            )

            with OpenEXR.File(str(output_path)) as f:
                Z = f.channels()["Z"].pixels

                for y in range(Z.shape[0]):
                    for x in range(Z.shape[1]):
                        if Z[y, x] is not None:
                            assert (
                                len(Z[y, x]) <= max_samples
                            ), f"Too many samples at ({y}, {x}): {len(Z[y, x])}"

        finally:
            output_path.unlink(missing_ok=True)


class TestExportFromPly:
    """Tests for PLY-to-deep-EXR export."""

    def test_export_from_ply_works(self, tmp_path):
        """Test that export_deep_exr_from_ply works with a real PLY file."""
        ply_path = Path("/workspace/ml-sharp/output/test_image.ply")

        if not ply_path.exists():
            pytest.skip("Test PLY file not found")

        output_path = tmp_path / "test_deep.exr"

        export_deep_exr_from_ply(
            ply_path=ply_path,
            output_path=output_path,
            downsample_gaussians=1000,  # Small for fast test
            max_samples_per_pixel=16,
        )

        assert output_path.exists(), "EXR file was not created"

        with OpenEXR.File(str(output_path)) as f:
            channels = list(f.channels().keys())
            assert "Z" in channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
