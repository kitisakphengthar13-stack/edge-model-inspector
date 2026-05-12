# PyTorch Model Inspection and ONNX Preparation Toolkit

Local CLI toolkit for inspecting PyTorch checkpoints, defining model specs,
exporting ONNX when possible, and validating ONNX inference before edge
deployment.

This project does not replace framework-specific exporters such as Anomalib,
Ultralytics, OpenVINO tools, or TensorRT `trtexec`. It provides a structured
workflow to inspect models and prepare validated ONNX artifacts for downstream
deployment.

## What This Project Is / Is Not

This project is:

- a PyTorch checkpoint inspection toolkit
- a model spec validation workflow
- a guarded PyTorch dry-run tool
- an ONNX export and ONNX Runtime validation workflow
- an optional TensorRT deployment planning helper

This project is not:

- a universal `.pt`/`.ckpt` converter
- a replacement for Anomalib, Ultralytics, OpenVINO, or TensorRT
- a TensorRT engine builder in the core PC workflow
- a tool that guarantees every model can be exported automatically

## Export Strategy

This project uses a source-first export strategy:

1. Prefer the official exporter from the source framework/library.
   - For Anomalib models, use Anomalib export when available.
   - For Ultralytics models, use Ultralytics export when available.
   - For other frameworks, use their official export path when available.
2. Use this toolkit's generic PyTorch-to-ONNX exporter as a fallback only when:
   - the model can be instantiated from module/class
   - checkpoint loading is defined
   - input/output spec is provided
   - the model can pass dry-run forward
3. If neither works, the toolkit still provides checkpoint inspection and
   deployment analysis to explain what is missing.

Official source-library exporters and this toolkit's generic exporter are
separate routes. Use `assess-export` to make that distinction explicit before
choosing an export path.

The goal is not to force every model through one converter. The goal is to
produce a reliable, validated ONNX artifact using the most appropriate export
path.

## Core Workflow

- Phase 1: Checkpoint Inspection
- Phase 2: Model Spec Validation
- Phase 3: Checkpoint Loading Analysis
- Phase 4: Safe PyTorch Dry Run
- Phase 5: ONNX Export
- Phase 6: ONNX Validation

Optional deployment notes/helpers:

- TensorRT build planning helper
- Jetson `trtexec` documentation
- target-side TensorRT build on the target device or matching runtime

## Usage

```bash
python -m converter.cli inspect models/model.pt
python -m converter.cli inspect models/model.ckpt --max-items 120
python -m converter.cli inspect models/model.pt --unsafe-load

python -m converter.cli validate-spec specs/example_simple_classifier_dryrun.yaml
python -m converter.cli assess-export specs/patchcore_cable_coreset_0_1.yaml
python -m converter.cli assess-export specs/yolo26n_task_detect.yaml
python -m converter.cli assess-export specs/example_simple_classifier_dryrun.yaml
python -m converter.cli check-checkpoint specs/example_patchcore_cable.yaml --max-items 80
python -m converter.cli plan-load specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli dry-run-model specs/example_simple_classifier_dryrun.yaml --allow-imports
python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports
python -m converter.cli validate-onnx artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml
```

## Checkpoint Inspection

PyTorch checkpoint files are not all the same. A `.pt`, `.pth`, or `.ckpt` file
may contain:

- a full PyTorch model object
- a raw `state_dict`
- a training checkpoint dictionary
- a PyTorch Lightning checkpoint
- a TorchScript archive
- a custom dictionary

Raw `state_dict` files cannot be exported by themselves without matching model
architecture code and loading rules.

## Model Specs

Checkpoint inspection alone is not enough for export. A spec describes the
missing contract:

- model identity
- checkpoint loading rule
- model construction hints
- conversion strategy
- input shape and dtype
- output names
- optional preprocessing, postprocessing, runtime assets, and metadata

The spec schema is intentionally generic and extensible. Unknown custom tasks
and extra sections are allowed.

## Export Capability Assessment

`assess-export` evaluates export-route options without running exporters,
importing model code, instantiating models, or creating deployment artifacts.
It reports:

- detected or declared source framework and model family
- whether an official source-library exporter route is known, likely, unknown,
  blocked, or not applicable
- whether this toolkit's generic PyTorch-to-ONNX exporter has enough spec
  information to be attempted
- recommended route, evidence, blockers, and unknowns

Spec-based assessment is preferred because the spec contains the model
construction, checkpoint loading, input, output, and target-format contract:

```bash
python -m converter.cli assess-export specs/patchcore_cable_coreset_0_1.yaml
python -m converter.cli assess-export specs/example_simple_classifier_dryrun.yaml
```

Checkpoint-path assessment is preliminary and lower confidence because a
checkpoint alone usually does not contain the full export contract:

```bash
python -m converter.cli assess-export path/to/model.pt --unsafe-load
```

The assessment is conservative. `known_from_registry` means the framework has a
known official route in principle; it does not mean the installed version or
specific model has been verified. `likely` is not `verified`. External exporters
such as Anomalib or Ultralytics are not executed by this command.

## Dry Run and ONNX Export

`dry-run-model` and `export-onnx` may import local user code, so both are guarded
with explicit flags. `export-onnx` requires `--allow-imports` and only works when
the spec contains enough information to instantiate the PyTorch model and run a
forward pass.

Actual ONNX export requires `onnx` and `onnxscript` from `requirements.txt`.
ONNX Runtime validation uses `onnxruntime`.

## ONNX Validation

`validate-onnx` checks exported ONNX files before downstream deployment. It
loads the ONNX file, runs `onnx.checker.check_model`, creates an ONNX Runtime
CPU session, builds dummy input, and runs inference.

It does not import user modules, load checkpoints, execute custom loaders, or
build TensorRT. PyTorch-vs-ONNX numerical comparison can be added later.

## Optional: TensorRT Deployment Planning

`plan-tensorrt` generates a TensorRT/`trtexec` build plan from an ONNX file. It
does not run `trtexec`, does not require TensorRT, does not require Jetson
hardware, and does not create engine files.

```bash
python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx
python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16
python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16 --workspace-mb 2048
python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16 --min-shape 1x3x2x2 --opt-shape 1x3x2x2 --max-shape 4x3x2x2
```

Precision behavior:

- `fp32`: no `trtexec` precision flag
- `fp16`: adds `--fp16`
- `int8`: adds `--int8`, but calibration or explicit quantization is not
  implemented

TensorRT `.engine` files are target-specific artifacts and are intentionally not
produced by the core PC-side workflow. Build them on the actual target device,
such as NVIDIA Jetson, using TensorRT tools like `trtexec`.

## Future TensorFlow/TFLite Direction

TFLite and TensorFlow SavedModel may become future downstream planning or
validation targets, especially for Raspberry Pi-class deployments. They are not
implemented now, and TensorFlow is not a base dependency.

The project remains ONNX-core. Future TFLite support should remain source-first
when libraries such as Ultralytics provide official TFLite exporters. Generic
ONNX-to-TFLite bridges are not current core scope.

## Real PatchCore Checkpoint Testing

A real PatchCore inspection/planning spec is available at
`specs/patchcore_cable_coreset_0_1.yaml`. See
[docs/PATCHCORE_REAL_TEST.md](docs/PATCHCORE_REAL_TEST.md).

For PatchCore, the preferred path is source-first export: use Anomalib's
official ONNX export when available, then use this toolkit for ONNX validation
and optional TensorRT planning. Direct generic export is expected to require
future Anomalib/PatchCore-specific loader support.

## Real YOLO Detection Checkpoint Testing

A real Ultralytics YOLO detection example is documented at
[docs/YOLO_REAL_TEST.md](docs/YOLO_REAL_TEST.md). The spec is
`specs/yolo26n_task_detect.yaml`.

This case demonstrates checkpoint inspection, official Ultralytics ONNX export,
ONNX validation with this toolkit, and export-route assessment. It complements
the PatchCore real test with a different source framework, task, and output
structure.

## Jetson TensorRT Build Notes

PC can export and validate ONNX. Jetson or a matching target runtime should
build TensorRT engines. See [docs/JETSON.md](docs/JETSON.md) for transfer
guidance and manual `trtexec` examples.
