# Reproduce the Portable v5 Model

Release: <https://github.com/Gabric-cau/ind5003-vision-ii/releases/tag/v2026.10.28-portable-v5>

Download `ind5003_offline_video_v5_20261028.zip` from Assets and extract it. Run
commands inside `portable_bundle_v5`. The archive contains source, pretrained
visual weights, the fitted downstream heads, dependency specifications, an
installer, and SHA-256 checksums. GitHub's automatic source archive does not
contain the model weights. Original match videos are supplied separately.

## Windows

Install 64-bit Python 3.11 from the Python distribution and verify
`py -3.11 --version`. In PowerShell:

```powershell
py -3.11 verify_bundle.py
py -3.11 install.py --torch cu118
.\.venv\Scripts\python.exe run_video.py --video C:\videos\penalty.mp4 --output runs\penalty_001 --device cuda:0 --threads 2 --batch-size 1
```

The installer creates `.venv` and `.venv-pose33` inside the extracted bundle.
The MediaPipe/shoe environment remains separate because the serialized shoe
classifier and its dependencies use a different numerical stack. Activation
scripts are not required. Installation must complete both environments.

`cu118` keeps PyTorch 2.7.0 while using its CUDA 11.8 wheels, suitable for testing
the Dell RTX 2060's existing driver. Newer GPUs such as RTX 5060 should use
`--torch cu128` and a compatible NVIDIA driver. CPU installation uses
`--torch cpu`, followed by `--device cpu` for inference. These wheel variants
are listed in the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/).

## Linux and macOS

```sh
python3.11 verify_bundle.py
python3.11 install.py --torch cpu
.venv/bin/python run_video.py --video /path/to/penalty.mp4 --output runs/penalty_001 --device cpu --threads 2 --batch-size 1
```

Linux NVIDIA machines can select `--torch cu118` or `--torch cu128` and run with
`--device cuda:0`. On Ubuntu, OpenCV may additionally require the distribution's
`libgl1` and `libglib2.0-0` system packages. macOS uses CPU in this release;
MPS execution is not a validated profile. Initial installation requires network
access to the package indexes. Normal inference uses the included local weights.

## Input and Results

Supply one manually isolated penalty event as MP4, beginning with the original
shooting camera. Include the approach, strike and visible result. This entry
point does not detect every penalty automatically from a complete match.
Inference receives the video without human contact times, actor boxes, labels,
player names or outcomes.

Use a new output directory for every run. The main result is `analysis.json`:

- `required_outputs`: structured evidence and candidates for the requested features.
- `final_feature_table`: scalar values after the selected auxiliary models.
- `stage_receipts` and `logs/`: actual stage completion and timings.
- `accuracy_validated`: remains false; installation or execution is not an accuracy test.

Unknown and censored values are retained. Sparse ground projections do not
establish a complete top-down run-up, and the last visible ball point does not
establish its physical resting position. The current goalkeeper-height and
continuous-tracking limitations remain visible in the output.

## Version Scope

This release carries the full-shot causal contact v3 weights and the currently
selected shoe, goalkeeper, ball-category/height and camera heads. It includes
the measured contact-actor fallback, result-field consistency fixes, Windows
interpreter resolution, and an adjustable visual-stage batch size.

The experimental ball detector, ROI/identity repairs and local VLM were not
promoted automatically. The latest human-review delivery has been inspected but
has not been used to retrain this release. Existing foot-reference conflicts
must be adjudicated before presenting those development comparisons as final
accuracy. This version does not establish 90% performance on the course outputs.

## Verification Status

The v5 archive passed a fresh installation into two empty virtual environments
on Windows 11 with an RTX 2060 (6 GB), driver 555.99 and Python 3.11.9. Both
environments passed their dependency/import checks, and a real CUDA operation
passed using PyTorch 2.7.0+cu118. One known development video then completed all
14 stages in 220.95 seconds, without supplied human annotations or an image-size
reduction. See the [Dell validation report](DELL_PORTABLE_V5_VALIDATION_2026-10-28.md)
for the exact archive/input hashes and the scope of the check.

A successful installer writes `environment_check.json`; it records environment
checks and deliberately does not claim video execution or accuracy. The separate
validation report records the subsequent video test. The tested ZIP remains
immutable, so its build manifest still describes the checks made at packaging
time. This does not establish 90% accuracy or a fresh installation on every
other operating system/GPU variant. The lab's existing Linux RTX 4090
environments completed the earlier v4 inference runs.
