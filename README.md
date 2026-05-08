# Edge Model Converter

This project is a generic PyTorch checkpoint-to-deployment preparation toolkit.
Phase 1 inspects generic PyTorch `.pt`, `.pth`, and `.ckpt` files. Phase 2 adds
generic `spec.yaml` files that describe how a model should be handled in future
conversion phases. Phase 3 adds a safe loading plan layer and checkpoint-only
loading utilities. Phase 4 adds trusted local model dry-runs for checking model
instantiation, checkpoint loading, dummy input creation, and forward execution.
Phase 5 adds PyTorch-to-ONNX export.

The current goal is to understand checkpoint contents and define the missing
conversion contract. It does not export ONNX, build TensorRT engines, run a web
API, or assume any specific model family.

## Why Inspection Matters

PyTorch checkpoint files are not all the same. A `.pt` file may contain:

- a full PyTorch model object
- a raw `state_dict` only
- a checkpoint dictionary
- a PyTorch Lightning checkpoint
- a TorchScript archive
- a custom dictionary

Raw `state_dict` files cannot be converted by themselves without the matching
model architecture and the code needed to instantiate it.

Some checkpoints also contain non-neural-network runtime data. For example,
PatchCore-style anomaly detection checkpoints may include a feature memory bank,
image or pixel thresholds, normalization ranges, and post-processing settings.
Those values can be important for deployment even though they are not ordinary
network weights.

## Usage

```bash
python -m converter.cli inspect models/model.pt
python -m converter.cli inspect models/model.ckpt --max-items 120
python -m converter.cli inspect models/model.pt --unsafe-load

python -m converter.cli validate-spec specs/example_patchcore_cable.yaml
python -m converter.cli validate-spec specs/example_classification.yaml
python -m converter.cli validate-spec specs/example_custom_model.yaml

python -m converter.cli plan-load specs/example_patchcore_cable.yaml
python -m converter.cli plan-load specs/example_classification.yaml
python -m converter.cli plan-load specs/example_custom_model.yaml

python -m converter.cli check-checkpoint specs/example_classification.yaml --max-items 50
python -m converter.cli check-checkpoint specs/example_patchcore_cable.yaml --max-items 80

python -m converter.cli dry-run-model specs/example_classification.yaml
python -m converter.cli dry-run-model specs/example_classification.yaml --allow-imports
python -m converter.cli dry-run-model specs/example_classification.yaml --allow-imports --no-strict
python -m converter.cli dry-run-model specs/example_classification.yaml --allow-imports --prefix-to-strip model.
python -m converter.cli dry-run-model specs/example_custom_model.yaml --allow-imports
python -m converter.cli dry-run-model specs/example_patchcore_cable.yaml --allow-imports

python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml
python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports
python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports --output artifacts/simple_classifier_dryrun/simple_classifier.onnx
python -m converter.cli export-onnx specs/example_custom_model.yaml --allow-imports
python -m converter.cli export-onnx specs/example_patchcore_cable.yaml --allow-imports
```

By default, the CLI tries PyTorch safe loading with `weights_only=True` when the
installed PyTorch version supports it. If safe loading fails, inspect the source
of the file before using `--unsafe-load`. Unsafe loading uses Python pickle and
should only be used for trusted local files.

The inspector prints file information, the top-level Python object type, a
detected checkpoint kind, common metadata, state dictionary entries when
available, notable deployment signals, and possible task hints.

Task hints are heuristic and may be wrong. They are based only on key names and
should not be treated as proof of the model task.

Exact conversion still requires:

- model architecture
- checkpoint loading rule
- input shape
- output names
- preprocessing and postprocessing rules

## Conversion Specs

Checkpoint inspection alone is not enough for conversion. `.pt`, `.pth`, and
`.ckpt` files do not always contain model architecture, model construction
arguments, input rules, or output names.

A conversion spec describes that missing contract:

- model identity
- checkpoint loading rule
- model construction hints
- conversion strategy
- input shape and dtype
- output names
- optional preprocessing, postprocessing, runtime assets, and metadata

The spec schema is intentionally generic and extensible. It is not limited to
YOLO, Anomalib, PatchCore, classification, or any other single model family.
Unknown custom tasks are allowed. Extra sections are allowed and reported
without failing validation. The examples in `specs/` are examples only.

`validate-spec` checks only generic structure. It does not instantiate PyTorch
models, import user modules, load checkpoints, or export anything.

## Loading Plans

`plan-load` explains how the project would load a model in future phases without
executing user code. It reads and validates the spec, then reports the checkpoint
path, checkpoint kind, load mode, model architecture hints, module/class strings,
conversion strategy, custom loader requirements, unsafe loading requirements,
wrapper likelihood, and status fields:

- `can_plan`
- `can_load_checkpoint`
- `can_instantiate_model`
- `can_export_now`

In Phase 3, `can_export_now` is always `no` because ONNX export is not
implemented. `plan-load` never loads checkpoints, imports user modules,
executes custom loaders, instantiates models, exports ONNX, or builds TensorRT.

## Checkpoint Checks

`check-checkpoint` explicitly loads only the checkpoint declared by a valid spec
and tries to extract a `state_dict` when possible. It does not instantiate
models, import model modules, execute custom loaders, export ONNX, or build
TensorRT.

Checkpoint loading follows `checkpoint.load_mode`:

- `safe_weights_only`: uses `torch.load(..., weights_only=True)` when supported.
- `unsafe_trusted_local`: uses Python pickle behavior and should only be used
  for trusted local files.
- `custom_loader` or `external_loader`: declared but not executed in Phase 3.

Model instantiation is still not automatic. Future loader support will require
explicit module/class handling, construction arguments, wrappers, and security
boundaries. A `custom_loader` value in the spec is metadata only for now.

## Model Dry Runs

`dry-run-model` is the first command that may import local user code. Importing
Python modules and instantiating classes can execute code, so `--allow-imports`
is required. Without that flag, the command refuses before importing or
instantiating anything.

With `--allow-imports`, `dry-run-model` verifies whether a spec can instantiate
a model, load checkpoint weights when declared, create a dummy input from
`spec.input`, and run a forward pass on CPU. It prints the state_dict source,
strict loading mode, dummy input shape and dtype, forward status, and output
structure. It still does not export ONNX and still does not build TensorRT.

Placeholder example specs may fail dry-run because their modules and classes
are examples only. Custom loaders and custom wrappers are declared in specs but
are not executed yet. Some model families, such as PatchCore-style anomaly
models, may require framework-specific loaders or wrappers before dry-run or
export can work.

The repository includes `specs/example_simple_classifier_dryrun.yaml` and a tiny
`model_zoo.simple_classifier` module only for local self-checking. It declares
`checkpoint.kind: none`, so dry-run uses randomly initialized weights and does
not load a checkpoint.

## ONNX Export

`export-onnx` is the first command that writes an ONNX artifact. It still
requires `--allow-imports` because it imports local model code and instantiates
the class declared in the spec.

The command validates the spec, instantiates the model, optionally loads a
checkpoint, creates a dummy input, runs a forward pass, and then calls
`torch.onnx.export`. It uses `spec.input.name` and `spec.output.names` for ONNX
I/O names, and uses `conversion.opset` unless `--opset` is provided. Output path
priority is `--output`, then `conversion.output_path`, then
`artifacts/<spec.name>/model.onnx`.

Actual ONNX export requires the `onnx` and `onnxscript` packages listed in
`requirements.txt`. ONNX Runtime validation is intentionally not included yet;
it is planned for a later phase. TensorRT engine building is also intentionally
not included in Phase 5.

Expected behavior:

- without `--allow-imports`, export refuses
- the simple classifier demo exports successfully
- custom loader and custom wrapper specs are refused until implemented
- PatchCore-style specs need a framework-specific loader or module/class before
  export
- ONNX Runtime validation is planned for a later phase

Phase 5 does not validate ONNX with ONNX Runtime, does not build TensorRT, and
does not execute `custom_loader` or `custom_wrapper` code. The exporter remains
generic and is not specific to YOLO, Anomalib, PatchCore, classification, or any
other model family.

## Conversion Strategies

- `full_model`: use when the model forward can be exported directly.
- `feature_extractor_only`: use when only part of the model should be exported,
  such as a backbone or feature extractor.
- `module_subgraph`: use when a named submodule or subgraph should be exported.
- `custom_wrapper`: use when a custom forward wrapper is required.
- `torchscript_existing`: use when the file is already TorchScript.
- `external_exporter`: use when another framework or tool handles export.

## Planned Phases

- Phase 3: generic model loading plan and checkpoint loading utilities
- Phase 4: safe model instantiation and dry-run forward execution
- Phase 5: PyTorch to ONNX export
- Phase 6: ONNX Runtime validation
- Phase 7: TensorRT build on Jetson target devices
- Phase 8: benchmarking and deployment metadata
