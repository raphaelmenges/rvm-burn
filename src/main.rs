mod model;

use burn::{Tensor, backend::ndarray::NdArray};
use image::{GrayImage, ImageReader, Luma};
use model::rvmopset20::Model;
use std::time::Instant;

fn main() {
    type Backend = NdArray<f32>;

    // Get a default device for the backend.
    let device = <NdArray<f32> as burn::tensor::backend::Backend>::Device::default();

    // Create model instance and load weights from target dir default device.
    let model: Model<Backend> = Model::default();

    // Prepare src input.
    let img = ImageReader::open("Lenna.png")
        .unwrap()
        .decode()
        .unwrap()
        .resize_exact(120, 90, image::imageops::FilterType::CatmullRom)
        .to_rgb32f();
    let chw: Vec<f32> = (0..3usize)
        .flat_map(|c| img.pixels().map(move |p| p.0[c]))
        .collect();
    let src = Tensor::<Backend, 1>::from_floats(chw.as_slice(), &device).reshape([1, 3, 90, 120]);

    // Prepare recurrent inputs.
    let r1i = Tensor::<Backend, 4>::zeros([1, 16, 45, 60], &device);
    let r2i = Tensor::<Backend, 4>::zeros([1, 20, 23, 30], &device);
    let r3i = Tensor::<Backend, 4>::zeros([1, 40, 12, 15], &device);
    let r4i = Tensor::<Backend, 4>::zeros([1, 64, 6, 8], &device);

    // Prepare downsample ratio input.
    let downsample_ratio_value = vec![1_f32];
    let downsample_ratio =
        Tensor::<Backend, 1>::from_floats(downsample_ratio_value.as_slice(), &device);

    // Infer.
    let start = Instant::now();
    let output = model.forward(src, r1i, r2i, r3i, r4i, downsample_ratio);
    println!("Inference took: {}ms", start.elapsed().as_millis());

    // Store matting as output.
    let pha: Vec<f32> = output
        .1
        .reshape([90, 120])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let img = GrayImage::from_fn(120, 90, |x, y| {
        let v = pha[(y as usize) * 120 + x as usize];
        Luma([(v.clamp(0_f32, 1_f32) * 255_f32) as u8])
    });
    img.save("output.png").unwrap();
}
