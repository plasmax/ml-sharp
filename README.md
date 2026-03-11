# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### FAQ: camera alignment and external camera tracks

- **Can I convert Alembic camera data to extrinsics/intrinsics and export a `.ply` aligned to that camera?**
  - Yes. Use `tools/convert_alembic_camera_to_sharp.py` to:
    1. read an Alembic camera sample,
    2. convert camera parameters to OpenCV/SHARP intrinsics (x right, y down, z forward), and
    3. transform Gaussian means/orientations into your target world frame before writing a new `.ply`.
  - Example:

```bash
python tools/convert_alembic_camera_to_sharp.py \
  --abc camera.abc \
  --camera-path /Camera01/camera/.../render_:cameraLeft_LOCShape \
  --sample-index 0 \
  --image-width 1920 \
  --image-height 1080 \
  --input-ply input.ply \
  --output-ply aligned.ply
```

  - By default, the script reads extrinsics from the Alembic xform chain above the selected camera object.
  - By default, it applies a camera-basis conversion (`Y`/`Z` sign flip) to map common DCC camera axes to SHARP/OpenCV. Disable with `--no-camera-yz-flip` if your source is already OpenCV-style.
  - The tool auto-normalizes Alembic 4x4 matrix layout when translation is stored in the last row (as commonly seen via Python bindings), so translation is applied correctly during `.ply` alignment.
  - By default, filmBack translation channels are not applied to principal point (this matches many Nuke exports where window translate is a comp-space adjustment). Use `--apply-filmback-translation` if you want those offsets baked into `K`.
  - If your DCC camera has window translate / film offsets you do not want in projection matching, pass `--ignore-film-offset`.
  - Intrinsics now apply Alembic `lens_squeeze_ratio` as horizontal aperture scaling (anamorphic), which fixes common focal mismatch issues when converting to OpenCV `fx`.
  - If you need to match a calibrated focal directly, pass `--override-focal-length-mm <value>`.
  - Optional override: pass `--extrinsics-json` with `world_from_camera` matrices keyed by frame index.
  - Use `--world-scale` to uniformly scale the aligned `.ply` around an anchor (`--scale-anchor camera|origin`). Example: `--world-scale 10 --scale-anchor camera`.
  - Tip for nested Alembic rigs: run `python tools/convert_alembic_camera_to_sharp.py --abc camera.abc --list-cameras` and copy one full camera path into `--camera-path`.
  - Note: `save_ply()` currently writes identity extrinsics metadata; alignment is encoded by transformed Gaussian coordinates.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
