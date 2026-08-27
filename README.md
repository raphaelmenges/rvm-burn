# rvm-burn
Inferring the [Robust Video Matting (RVM)](https://peterl1n.github.io/RobustVideoMatting) model with [burn](https://github.com/tracel-ai/burn) and [ort](https://github.com/pykeio/ort).

Model `src/model/rvmopset20.onnx` taken from [RobustVideoMatting repository](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx) and converted to opset20 by [@hedeshy](https://github.com/hedeshy).

Input image `Lenna.png` taken from [Wikipedia](https://en.wikipedia.org/wiki/Lenna#/media/File:Lenna_(test_image).png).

## Building

The ONNX Runtime hardware execution provider is a mutually exclusive crate feature, because prebuilt binaries do not host all combinations at once. The default is WebGPU, which works on all platforms:

```sh
cargo run --release
cargo run --release --no-default-features --features ort-directml   # Windows
cargo run --release --no-default-features --features ort-coreml     # macOS
```
