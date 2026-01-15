use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/rvmopset20.onnx")
        .out_dir("models/")
        .run_from_script();
}
