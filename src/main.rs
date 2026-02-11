#![recursion_limit = "256"]

mod model;

use burn::{
    Tensor,
    backend::{Wgpu, ndarray::NdArray},
    tensor::backend::Backend,
};
use image::{GrayImage, ImageReader, Luma};
use model::rvmopset20::Model;
use std::time::Instant;

struct Resolution {
    name: &'static str,
    src_width: usize,
    src_height: usize,
    r1_width: usize,
    r1_height: usize,
    r2_width: usize,
    r2_height: usize,
    r3_width: usize,
    r3_height: usize,
    r4_width: usize,
    r4_height: usize,
}

const FAST: Resolution = Resolution {
    name: "fast",
    src_width: 120,
    src_height: 90,
    r1_width: 60,
    r1_height: 45,
    r2_width: 30,
    r2_height: 23,
    r3_width: 15,
    r3_height: 12,
    r4_width: 8,
    r4_height: 6,
};

const BALANCED: Resolution = Resolution {
    name: "balanced",
    src_width: 160,
    src_height: 120,
    r1_width: 80,
    r1_height: 60,
    r2_width: 40,
    r2_height: 30,
    r3_width: 20,
    r3_height: 15,
    r4_width: 10,
    r4_height: 8,
};

const ACCURATE: Resolution = Resolution {
    name: "accurate",
    src_width: 320,
    src_height: 240,
    r1_width: 160,
    r1_height: 120,
    r2_width: 80,
    r2_height: 60,
    r3_width: 40,
    r3_height: 30,
    r4_width: 20,
    r4_height: 15,
};

fn run<B: Backend>(backend_name: &str, res: &Resolution) {
    let device = B::Device::default();
    let model: Model<B> = Model::default();

    // Load input image.
    let img = ImageReader::open("Lenna.png")
        .unwrap()
        .decode()
        .unwrap()
        .resize_exact(
            res.src_width as u32,
            res.src_height as u32,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb32f();
    let chw: Vec<f32> = (0..3usize)
        .flat_map(|c| img.pixels().map(move |p| p.0[c]))
        .collect();

    // Initial recurrent states.
    let mut r1i = Tensor::<B, 4>::zeros([1, 16, res.r1_height, res.r1_width], &device);
    let mut r2i = Tensor::<B, 4>::zeros([1, 20, res.r2_height, res.r2_width], &device);
    let mut r3i = Tensor::<B, 4>::zeros([1, 40, res.r3_height, res.r3_width], &device);
    let mut r4i = Tensor::<B, 4>::zeros([1, 64, res.r4_height, res.r4_width], &device);

    // Repeated inference.
    let downsample_ratio = vec![1_f32];
    let iterations = 10;
    let mut total = std::time::Duration::ZERO;
    for i in 0..=iterations {
        let src = Tensor::<B, 1>::from_floats(chw.as_slice(), &device).reshape([
            1,
            3,
            res.src_height,
            res.src_width,
        ]);
        let downsample_ratio = Tensor::<B, 1>::from_floats(downsample_ratio.as_slice(), &device);

        // Do the inference.
        let start = Instant::now();
        let (_, pha, r1o, r2o, r3o, r4o) = model.forward(src, r1i, r2i, r3i, r4i, downsample_ratio);
        let elapsed = start.elapsed();

        // Update recurrent states.
        r1i = r1o;
        r2i = r2o;
        r3i = r3o;
        r4i = r4o;

        // Let first run be warm-up.
        if i == 0 {
            continue;
        }

        total += elapsed;

        // Save last output to disk.
        if i == iterations - 1 {
            let pha: Vec<f32> = pha
                .reshape([res.src_height, res.src_width])
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let img = GrayImage::from_fn(res.src_width as u32, res.src_height as u32, |x, y| {
                let v = pha[(y as usize) * res.src_width + x as usize];
                Luma([(v.clamp(0_f32, 1_f32) * 255_f32) as u8])
            });
            img.save(format!("output_{}_{}.png", backend_name, res.name))
                .unwrap();
        }
    }

    println!(
        "[{backend_name}/{}] Average: {}ms",
        res.name,
        total.as_millis() / iterations
    );
}

fn main() {
    run::<NdArray<f32>>("ndarray", &FAST);
    run::<NdArray<f32>>("ndarray", &BALANCED);
    run::<NdArray<f32>>("ndarray", &ACCURATE);
    // run::<Cpu>("cpu", &FAST); // Too slow.
    // run::<Cpu>("cpu", &BALANCED);
    // run::<Cpu>("cpu", &ACCURATE);
    run::<Wgpu>("wgpu", &FAST);
    run::<Wgpu>("wgpu", &BALANCED);
    run::<Wgpu>("wgpu", &ACCURATE);
}
