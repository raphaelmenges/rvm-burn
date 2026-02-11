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

fn run<B: Backend>(name: &str) {
    let device = B::Device::default();
    let model: Model<B> = Model::default();

    // Load input image.
    let img = ImageReader::open("Lenna.png")
        .unwrap()
        .decode()
        .unwrap()
        .resize_exact(120, 90, image::imageops::FilterType::CatmullRom)
        .to_rgb32f();
    let chw: Vec<f32> = (0..3usize)
        .flat_map(|c| img.pixels().map(move |p| p.0[c]))
        .collect();

    // Initial recurrent states.
    let mut r1i = Tensor::<B, 4>::zeros([1, 16, 45, 60], &device);
    let mut r2i = Tensor::<B, 4>::zeros([1, 20, 23, 30], &device);
    let mut r3i = Tensor::<B, 4>::zeros([1, 40, 12, 15], &device);
    let mut r4i = Tensor::<B, 4>::zeros([1, 64, 6, 8], &device);

    // Repeated inference.
    let downsample_ratio = vec![1_f32];
    let iterations = 10;
    let mut total = std::time::Duration::ZERO;
    for i in 0..=iterations {
        let src = Tensor::<B, 1>::from_floats(chw.as_slice(), &device).reshape([1, 3, 90, 120]);
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
            let pha: Vec<f32> = pha.reshape([90, 120]).into_data().to_vec::<f32>().unwrap();
            let img = GrayImage::from_fn(120, 90, |x, y| {
                let v = pha[(y as usize) * 120 + x as usize];
                Luma([(v.clamp(0_f32, 1_f32) * 255_f32) as u8])
            });
            img.save(format!("output_{name}.png")).unwrap();
        }
    }

    println!("[{name}] Average: {}ms", total.as_millis() / iterations);
}

fn main() {
    run::<NdArray<f32>>("ndarray");
    run::<Wgpu>("wgpu");
}
