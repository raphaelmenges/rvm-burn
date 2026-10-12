use burn_onnx::ModelGen;
use std::fs;

fn main() {
    // Pass the resolved burn version on to the binary, so that it prints the
    // version it was actually compiled against. The build script runs in the
    // package directory.
    let lock = fs::read_to_string("Cargo.lock").expect("lock file is readable");
    let burn_version = lock
        .lines()
        .skip_while(|line| line.trim() != "name = \"burn\"")
        .nth(1)
        .and_then(|line| line.trim().strip_prefix("version = \""))
        .and_then(|version| version.strip_suffix('"'))
        .expect("lock file contains the burn version");
    println!("cargo:rustc-env=BURN_VERSION={burn_version}");

    ModelGen::new()
        .input("src/model/rvmopset20.onnx")
        .out_dir("model/")
        .run_from_script();
}
