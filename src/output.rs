use crate::resolution::Resolution;
use image::{GrayImage, ImageReader, Luma};

/// Maximum allowed absolute difference per pixel between output and reference.
pub const PIXEL_TOLERANCE: f32 = 0.1_f32;

/// Maximum allowed fraction of pixels exceeding the tolerance.
pub const MAX_BAD_PIXEL_FRACTION: f32 = 0.01_f32;

/// Saves the alpha matte as a grayscale image.
pub fn save(pha: &[f32], res: &Resolution, backend_name: &str) {
    let img = GrayImage::from_fn(res.src_width as u32, res.src_height as u32, |x, y| {
        let v = pha[(y as usize) * res.src_width + x as usize];
        Luma([(v.clamp(0_f32, 1_f32) * 255_f32) as u8])
    });
    img.save(format!("output_{}_{}.png", backend_name, res.name))
        .unwrap();
}

/// Compares the alpha matte against the stored reference image and prints
/// whether the output matches within tolerance.
pub fn check(pha: &[f32], res: &Resolution, backend_name: &str) {
    let reference = ImageReader::open(format!("reference/{}.png", res.name))
        .unwrap()
        .decode()
        .unwrap()
        .to_luma8();
    assert_eq!(reference.width() as usize, res.src_width);
    assert_eq!(reference.height() as usize, res.src_height);
    let bad = pha
        .iter()
        .zip(reference.pixels())
        .filter(|(v, p)| (v.clamp(0_f32, 1_f32) - p.0[0] as f32 / 255_f32).abs() > PIXEL_TOLERANCE)
        .count();
    let fraction = bad as f32 / pha.len() as f32;
    if fraction > MAX_BAD_PIXEL_FRACTION {
        println!(
            "[{backend_name}/{}] Output check FAILED: {:.1}% of pixels deviate more than {PIXEL_TOLERANCE} from the reference",
            res.name,
            fraction * 100_f32
        );
    } else {
        println!("[{backend_name}/{}] Output check passed", res.name);
    }
}
