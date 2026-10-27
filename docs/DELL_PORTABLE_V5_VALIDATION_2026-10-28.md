# Dell RTX 2060 Portable v5 Validation

Archive: `ind5003_offline_video_v5_20261028.zip`

SHA-256: `5264516d0c28b57f002875523bb1971e59792577ba8f56b9a76dfb6fea41085e`

## Scope

This checks installation and complete video execution on a second machine. It
does not measure prediction accuracy. The input is a known development clip,
provided without its human labels, contact time, actor boxes or outcome.

The existing operating system, Python installation and NVIDIA driver were
retained. Both virtual environments were absent before the successful attempt.
Only downloaded wheel-cache files were reused from earlier failed attempts;
installed packages and virtual environments were not copied from the lab or
from those attempts.

## Machine and Installation

| Item | Observed value |
|---|---|
| Operating system | Windows 11, x64 |
| Base interpreter | Python 3.11.9, Windows Store distribution |
| GPU | NVIDIA GeForce RTX 2060, 6 GB |
| Driver | 555.99, unchanged |
| Main PyTorch | 2.7.0+cu118 |
| Torch CUDA runtime | 11.8 |
| GPU capability | 7.5 |
| Main environment | Fresh `.venv`, direct pins/imports/`pip check` passed |
| Shoe environment | Fresh `.venv-pose33`, direct pins/imports/`pip check` passed |
| CUDA operation | Actual GPU tensor operation and synchronization passed |

The successful command was `py -3.11 install.py --torch cu118`. The installer
returned zero and wrote `environment_check.json` with
`all_environment_checks_passed=true`. That receipt deliberately leaves
`end_to_end_inference_tested` and `accuracy_validated` false: those are separate
claims, not installation checks.

Earlier attempts exposed Windows Store Python failures while resolving existing
user cache directories and an inherited Conda PATH entry. The released installer
uses project-local pip/Matplotlib/Ultralytics/Torch caches and a restricted PATH
for its Windows child processes. Video stages also use project-local runtime
caches. Existing user configuration and other Python environments were not
modified. The original failed-attempt logs remain local.

## Complete Video Check

Input SHA-256:
`d965503c165650f129096e61e805ac9bbc481726cf3c9fcc613936fc7fcb0d98`

The clip is 15.0 seconds long: 450 frames at 30 fps.

```powershell
.\.venv\Scripts\python.exe run_video.py --video ..\..\inputs\penalty.mp4 --output runs\dell_core01 --device cuda:0 --threads 2 --batch-size 1
```

The complete invocation returned zero. All 14 stage processes returned zero,
including the separate shoe environment. The pipeline recorded 220.9514 seconds
of end-to-end processing; the external process wall time was 221.5761 seconds.
The final result was written to `runs/dell_core01/analysis.json`. No prediction
cache was reused, no CUDA out-of-memory failure occurred, and the model image
sizes and frozen contact-extractor settings were retained. The selected device
was `cuda:0`; several numerical heads and MediaPipe stages run on CPU.

These are execution results for one known clip. They do not establish accuracy
on new footage or completion of every course feature. Unknown and censored
values remain part of the result. The latest human-review labels have not been
used to retrain these weights.

The output includes right-foot and three-visible-step candidates, an
`image_left` goalkeeper dive and a `low` qualitative height candidate. It
retains an unknown outcome, no projected run-up points, and a camera-cut-censored
ball endpoint. These candidates have not been scored against independent
labels. Per-stage times and a sanitized machine receipt are in the
[JSON record](DELL_PORTABLE_V5_VALIDATION_2026-10-28.json). The raw analysis hash
identifies this particular local run, including its paths and timings; it is
not a required byte-identical result hash for other computers.

## Source Checks

The source and weight bindings passed the bundle verifier for all 366 listed
files. The corresponding source revision is
`0b8849485ec62ccabe735d94013fbcc526b5c694`.

[GitHub CI](https://github.com/Gabric-cau/ind5003-vision-ii/actions/runs/40543095657)
passed both jobs. The evidence suite reported 1,422 passed, 4 skipped and 40
subtests passed; the analysis job separately reported 43 passed.

The build manifest's `clean_install_tested=false` describes packaging before
this machine check. The tested ZIP is kept immutable; this report records the
subsequent machine validation against its exact hash. Other operating systems
and CUDA wheel variants have not received this same clean-install check.
