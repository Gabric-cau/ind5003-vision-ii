# M5 Pro Training Probe

## Result

Actual YOLO11m training passed on the local Apple M5 Pro with 24 GB unified
memory. A fixed 32-frame training-only subset covering 13 training events was
used for two epochs at 1920 pixels. Both batch sizes completed two real AdamW
steps, changed an observed FP32 non-bias parameter, and retained the requested
batch size without an observed OOM retry. Loss values were finite.

| Configuration | Warm Batch Median | Second Epoch, 32 Images | Maximum Sampled MPS Driver Allocation |
| --- | ---: | ---: | ---: |
| Batch 2 | 0.776 s | 12.795 s | 11.787 GB (10.98 GiB) |
| Batch 1 | 0.386 s | 12.850 s | 6.068 GB (5.65 GiB) |

Batch 1 is recommended for a later full run on this laptop: observed per-image
speed is similar and driver allocation is about 49% lower. Keep the 1920-pixel
input and existing match partition. The batch change is an experimental
configuration change; gradient accumulation does not guarantee identical
optimization or BatchNorm behavior across devices and batch sizes.

The last-epoch times imply approximately 89 minutes for 442 images and 30
epochs. This is a linear estimate, not a completed full training measurement;
it excludes the complete model comparisons and may change with sustained
load, memory pressure, thermal behavior and data loading. The 4090 experiment's
remote completion and checkpoint availability remain unverified.

The first batch-2 run coincided with system-wide swap usage increasing from
3.06 GB to 9.11 GB. A later batch-2 run changed from 8.81 to 8.92 GB; batch 1
changed from 8.78 to 8.76 GB. These are whole-system observations, including
other applications, rather than process-attributed swap measurements. Driver
allocation is sampled after batches and is not a strict instantaneous peak.

## Acceptance Evidence

- Training parameters were physically on `mps`, using FP32 with AMP disabled.
- `PYTORCH_ENABLE_MPS_FALLBACK=0` prevented implicit unsupported-operator CPU fallback.
- Maximum observed change in the same FP32 non-bias tensor was 0.000110075
  for batch 2 and 0.000110060 for batch 1, measured after actual optimizer steps.
- Saved checkpoints were reloaded and executed in full-frame and tiled modes
  through the actual MPS predictor backend.
- Two-epoch probe checkpoints produced zero candidates on the selected smoke
  image. This is a compatibility experiment, not an accuracy result or a
  replacement for the existing trained model.
- The previous four-event refit was additionally loaded on MPS and produced
  one candidate in each inference mode on the same training image. This also
  exercised the existing nonempty-candidate CPU NMS merge path. Detection
  correctness was not scored.
- No held-out images or label files were loaded by the probe. The framework's
  required final validation used only the same 32 training images; its fit
  metrics must not be interpreted as generalization accuracy.

## Artifacts And Corrections

The complete batch-1 run is in
`outputs/mps_ball_probe_2026-10-28_1920_b1/report.json`, with
`previous_model_inference.json` recording the existing-model smoke test.

Batch-2 training is recorded in
`outputs/mps_ball_probe_2026-10-28_1920_b2_v3/report.json`. Its original
`completed=false` records a probe assertion that inspected the original CPU
model object instead of Ultralytics' copied MPS predictor backend. This check
was corrected. `checkpoint_verification.json` separately verifies the same
checkpoint, binds both the checkpoint SHA256 and original report SHA256, and
confirms MPS inference. Read these two files together; the original error
record has been preserved.

Two earlier probe implementation issues were also corrected: deriving legacy
video IDs from source filenames and using absolute output paths. The framework
had saved the earlier successful training checkpoint beneath `runs/detect`;
that generated fit directory is preserved as
`outputs/mps_ball_probe_2026-10-28_1920_b2_v2/fit_recovered/`.

Probe weights are experimental only. The main pipeline's default weights were
not replaced, and no new project-accuracy claim is made by this test.

## Environment And Reproduction

The isolated environment is `/Users/gabric/Downloads/IND5003_mps_env`:
Python 3.12, PyTorch 2.7.0, torchvision 0.22.0, Ultralytics 8.4.175 and NumPy
1.26.0. `pip check` passed. Direct requirements are pinned in
`requirements-mps-training.txt`; full installed versions are retained in each
probe report. The existing general-purpose environments were not modified.

From the repository root, using a fresh output directory:

```bash
python3.12 -m venv .venv-inference
.venv-inference/bin/python -m pip install -r requirements-mps-training.txt
mkdir -p .runtime-cache/mps-config
env YOLO_CONFIG_DIR="$PWD/.runtime-cache/mps-config" \
    YOLO_OFFLINE=true PYTORCH_ENABLE_MPS_FALLBACK=0 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    .venv-inference/bin/python scripts/probe_mps_ball_training.py \
    --batch 1 --frames 32 --epochs 2 --imgsz 1920 \
    --output outputs/mps_probe_new_run
```

The local source manifests and pretrained checkpoint named by the script must
be present. For this probe, data-loader workers were zero and the MPS process
memory fraction was 0.65 of the framework-recommended limit. The short run's
effective warmup was one epoch; the later 30-epoch protocol must retain its
full warmup and independently record the final result.
