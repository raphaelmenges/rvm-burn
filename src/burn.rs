use crate::{
    input, measure,
    measure::{ITERATIONS, WARMUP},
    model::rvmopset20::Model,
    output,
    resolution::{ACCURATE, BALANCED, FAST, Resolution},
};
use burn::{
    Tensor,
    module::{Module, ModuleMapper, Param},
    tensor::{Device, DeviceKind, FloatDType},
};
use std::time::Instant;

/// Casts every float parameter of a module to the given dtype.
struct CastMapper(FloatDType);

impl ModuleMapper for CastMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(self.0), mapper)
    }
}

fn run(device: &Device, backend_name: &str, dtype: FloatDType, res: &Resolution) {
    let model = Model::from_file(concat!(env!("OUT_DIR"), "/model/rvmopset20.bpk"), device)
        .map(&mut CastMapper(dtype));

    // Load input image.
    let chw = input::load_src(res);

    // Initial recurrent states.
    let options = (device, dtype.into());
    let mut r1i = Tensor::<4>::zeros([1, 16, res.r1_height, res.r1_width], options);
    let mut r2i = Tensor::<4>::zeros([1, 20, res.r2_height, res.r2_width], options);
    let mut r3i = Tensor::<4>::zeros([1, 40, res.r3_height, res.r3_width], options);
    let mut r4i = Tensor::<4>::zeros([1, 64, res.r4_height, res.r4_width], options);

    // Upload the constant inputs once, before any timing begins.
    let src = Tensor::<1>::from_floats(chw.as_slice(), device)
        .reshape([1, 3, res.src_height, res.src_width])
        .cast(dtype);
    let downsample_ratio = Tensor::<1>::from_floats([1_f32].as_slice(), device).cast(dtype);
    device.sync().unwrap();

    // Repeated inference.
    let mut total = std::time::Duration::ZERO;
    for i in 0..(WARMUP + ITERATIONS) {
        // Do the inference.
        let start = Instant::now();
        let (_, pha, r1o, r2o, r3o, r4o) =
            model.forward(src.clone(), r1i, r2i, r3i, r4i, downsample_ratio.clone());
        device.sync().unwrap();
        let elapsed = start.elapsed();

        // Update recurrent states.
        r1i = r1o;
        r2i = r2o;
        r3i = r3o;
        r4i = r4o;

        // Let the first runs be warm-up.
        if i < WARMUP {
            continue;
        }

        total += elapsed;

        // Save last output to disk.
        if i == WARMUP + ITERATIONS - 1 {
            let pha: Vec<f32> = pha
                .reshape([res.src_height, res.src_width])
                .cast(FloatDType::F32)
                .into_data()
                .try_into_vec::<f32>()
                .unwrap();
            output::save(&pha, res, backend_name);
            output::check(&pha, res, backend_name);
        }
    }

    measure::report(backend_name, res, total);
}

pub fn run_all() {
    let flex = Device::flex();
    run(&flex, "burn-flex", FloatDType::F32, &FAST);
    run(&flex, "burn-flex", FloatDType::F32, &BALANCED);
    run(&flex, "burn-flex", FloatDType::F32, &ACCURATE);
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    let gpu = Device::vulkan(DeviceKind::DefaultDevice);
    #[cfg(target_os = "macos")]
    let gpu = Device::metal(DeviceKind::DefaultDevice);
    run(&gpu, "burn-gpu-fp32", FloatDType::F32, &FAST);
    run(&gpu, "burn-gpu-fp32", FloatDType::F32, &BALANCED);
    run(&gpu, "burn-gpu-fp32", FloatDType::F32, &ACCURATE);
    run(&gpu, "burn-gpu-fp16", FloatDType::F16, &FAST);
    run(&gpu, "burn-gpu-fp16", FloatDType::F16, &BALANCED);
    run(&gpu, "burn-gpu-fp16", FloatDType::F16, &ACCURATE);
}
