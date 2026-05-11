# Pre-Jetson Checklist

- [ ] `git status` is clean
- [ ] `validate-spec` passes
- [ ] `dry-run-model` passes when source code is available and trusted
- [ ] `export-onnx` passes, or an official source-framework exporter produced ONNX
- [ ] ONNX artifact exists
- [ ] `validate-onnx` passes
- [ ] optional `plan-tensorrt` command checked
- [ ] target-side TensorRT build will be performed on Jetson with `trtexec`
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
