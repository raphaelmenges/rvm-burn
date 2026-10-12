mod burn;
mod input;
mod measure;
mod model;
mod ort;
mod output;
mod resolution;

fn main() -> ::ort::Result<()> {
    println!("burn {}, ONNX Runtime {}", burn::version(), ort::version());
    burn::run_all();
    ort::run_all()
}
