mod burn;
mod common;
mod model;
mod ort;

fn main() -> ::ort::Result<()> {
    burn::run_all();
    ort::run_all()
}
