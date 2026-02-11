use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/rvmopset20.onnx")
        .out_dir("model/")
        .run_from_script();
}
