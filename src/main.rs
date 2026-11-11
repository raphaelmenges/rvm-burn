mod burn;
mod input;
mod measure;
mod model;
mod ort;
mod output;
mod resolution;

// Keeps freed memory for reuse, which spares burn-flex page faults on fresh output buffers.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // burn-flex runs this model fastest on a few threads.
    rayon::ThreadPoolBuilder::new()
        .num_threads(4_usize)
        .build_global()?;
    println!("burn {}, ONNX Runtime {}", burn::version(), ort::version());
    burn::run_all();
    ort::run_all()?;
    Ok(())
}
