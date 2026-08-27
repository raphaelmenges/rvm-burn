use crate::resolution::Resolution;
use image::ImageReader;

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
