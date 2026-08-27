use image::{GrayImage, ImageReader, Luma};
use std::time::Duration;

/// Number of untimed iterations before measurement begins.
pub const WARMUP: usize = 10;

/// Number of timed iterations.
pub const ITERATIONS: usize = 25;

pub struct Resolution {
    pub name: &'static str,
    pub src_width: usize,
    pub src_height: usize,
    pub r1_width: usize,
    pub r1_height: usize,
    pub r2_width: usize,
    pub r2_height: usize,
    pub r3_width: usize,
    pub r3_height: usize,
    pub r4_width: usize,
    pub r4_height: usize,
}

pub const FAST: Resolution = Resolution {
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

pub const BALANCED: Resolution = Resolution {
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

pub const ACCURATE: Resolution = Resolution {
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

/// Loads the input image, resizes it to the given resolution, and returns the
/// pixel data in CHW layout.
pub fn load_src(res: &Resolution) -> Vec<f32> {
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
    (0_usize..3_usize)
        .flat_map(|c| img.pixels().map(move |p| p.0[c]))
        .collect()
}

/// Saves the alpha matte as a grayscale image.
pub fn save_alpha(pha: &[f32], res: &Resolution, backend_name: &str) {
    let img = GrayImage::from_fn(res.src_width as u32, res.src_height as u32, |x, y| {
        let v = pha[(y as usize) * res.src_width + x as usize];
        Luma([(v.clamp(0_f32, 1_f32) * 255_f32) as u8])
    });
    img.save(format!("output_{}_{}.png", backend_name, res.name))
        .unwrap();
}

/// Prints the average duration of a timed iteration.
pub fn report(backend_name: &str, res: &Resolution, total: Duration) {
    println!(
        "[{backend_name}/{}] Average: {}ms",
        res.name,
        total.as_millis() / ITERATIONS as u128
    );
}
