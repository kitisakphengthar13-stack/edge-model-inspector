# Pre-Jetson Checklist

- [ ] `git status` is clean
- [ ] `validate-spec` passes
- [ ] `dry-run-model` passes
- [ ] `export-onnx` passes
- [ ] `validate-onnx` passes
- [ ] `plan-tensorrt` passes
- [ ] `artifacts/simple_classifier_dryrun/model.onnx` exists
- [ ] `docs/JETSON.md` exists
- [ ] repo is pushed or copied to Jetson
- [ ] do not copy large `.pt`/`.pth`/`.ckpt` files unless needed

## Smoke Test Commands

Run from the project root:

```bash
python -m converter.cli validate-spec specs/example_simple_classifier_dryrun.yaml

python -m converter.cli dry-run-model specs/example_simple_classifier_dryrun.yaml --allow-imports

python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports

python -m converter.cli validate-onnx artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml

python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16

python -m compileall converter model_zoo
```
