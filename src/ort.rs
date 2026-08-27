use crate::common::{self, ACCURATE, BALANCED, FAST, ITERATIONS, Resolution, WARMUP};
#[cfg(target_os = "macos")]
use ort::ep::coreml::CoreML;
#[cfg(target_os = "windows")]
use ort::ep::directml::DirectML;
use ort::{
    ep::{ExecutionProviderDispatch, webgpu::WebGPU},
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{DynValue, Tensor},
};
use std::time::{Duration, Instant};

const MODEL_PATH: &str = "src/model/rvmopset20.onnx";
const STATE_INPUT_NAMES: [&str; 4] = ["r1i", "r2i", "r3i", "r4i"];
const STATE_OUTPUT_NAMES: [&str; 4] = ["r1o", "r2o", "r3o", "r4o"];

struct Ep {
    name: &'static str,
    /// Device on which the recurrent states are kept resident.
    device: AllocationDevice,
    /// Registration for the session builder, or None for the CPU execution
    /// provider, which is always available without registration.
    dispatch: fn() -> Option<ExecutionProviderDispatch>,
}

/// Creates a zero-filled tensor in CPU memory.
fn zero_tensor(shape: &[usize]) -> ort::Result<Tensor<f32>> {
    Tensor::from_array((shape, vec![0_f32; shape.iter().product::<usize>()]))
}

/// Creates one buffer per recurrent state on the allocator device.
fn state_set(allocator: &Allocator, shapes: &[[usize; 4]; 4]) -> ort::Result<Vec<DynValue>> {
    shapes
        .iter()
        .map(|shape| Ok(Tensor::<f32>::new(allocator, *shape)?.into_dyn()))
        .collect()
}

fn run(ep: &Ep, res: &Resolution) -> ort::Result<()> {
    let mut builder = Session::builder()?;
    if let Some(dispatch) = (ep.dispatch)() {
        builder = builder.with_execution_providers([dispatch])?;
    }
    let mut session = builder.commit_from_file(MODEL_PATH)?;

    // Allocator for the device-resident recurrent state buffers.
    let memory = MemoryInfo::new(ep.device, 0, AllocatorType::Device, MemoryType::Default)?;
    let allocator = Allocator::new(&session, memory)?;

    // Upload the constant inputs once, before any timing begins.
    let src = Tensor::from_array((
        [1_usize, 3_usize, res.src_height, res.src_width],
        common::load_src(res),
    ))?;
    let downsample_ratio = Tensor::from_array(([1_usize], vec![1_f32]))?;
    let mut binding = session.create_binding()?;
    binding.bind_input("src", &src)?;
    binding.bind_input("downsample_ratio", &downsample_ratio)?;

    // The initial recurrent states are zeros, copied to the device on bind.
    let shapes = [
        [1_usize, 16_usize, res.r1_height, res.r1_width],
        [1_usize, 20_usize, res.r2_height, res.r2_width],
        [1_usize, 40_usize, res.r3_height, res.r3_width],
        [1_usize, 64_usize, res.r4_height, res.r4_width],
    ];
    for (name, shape) in STATE_INPUT_NAMES.iter().zip(&shapes) {
        binding.bind_input(*name, &zero_tensor(shape)?)?;
    }

    // The foreground output is not consumed and stays on the device. The alpha
    // output is read back into a reused CPU buffer on every run, because a
    // real application consumes the matte each frame.
    binding.bind_output_to_device("fgr", allocator.memory_info())?;
    binding.bind_output(
        "pha",
        zero_tensor(&[1_usize, 1_usize, res.src_height, res.src_width])?,
    )?;

    // Two disjoint sets of state buffers alternate between the output and the
    // input role, so a run never reads and writes the same buffer.
    let mut out_states = state_set(&allocator, &shapes)?;
    let mut spare = state_set(&allocator, &shapes)?;

    binding.synchronize_inputs()?;

    // Repeated inference.
    let mut total = Duration::ZERO;
    let mut final_pha = None;
    for i in 0..(WARMUP + ITERATIONS) {
        // Bind this run's state output buffers.
        for (name, state) in STATE_OUTPUT_NAMES.iter().zip(out_states.drain(..)) {
            binding.bind_output(*name, state)?;
        }

        // Do the inference and take handles to the fresh state outputs. The
        // scope ends the borrow of the binding, so it can be rebound below.
        let (elapsed, fresh) = {
            let start = Instant::now();
            let mut outputs = session.run_binding(&binding)?;
            binding.synchronize_outputs()?;
            let elapsed = start.elapsed();

            // Keep the alpha matte of the last run for saving to disk.
            if i == WARMUP + ITERATIONS - 1 {
                final_pha = outputs.remove("pha");
            }

            let fresh: Vec<DynValue> = STATE_OUTPUT_NAMES
                .iter()
                .map(|name| outputs.remove(name).expect("state output is bound"))
                .collect();
            (elapsed, fresh)
        };

        // Let the first runs be warm-up.
        if i >= WARMUP {
            total += elapsed;
        }

        // Feed the fresh state outputs back as the next inputs. The previous
        // input buffers become the next run's output targets.
        for (name, state) in STATE_INPUT_NAMES.iter().zip(&fresh) {
            binding.bind_input(*name, state)?;
        }
        out_states = std::mem::replace(&mut spare, fresh);
    }

    common::report(ep.name, res, total);

    // Save the alpha matte of the last run to disk.
    let final_pha = final_pha.expect("alpha output is bound");
    let (_, pha) = final_pha.try_extract_tensor::<f32>()?;
    common::save_alpha(pha, res, ep.name);

    Ok(())
}

pub fn run_all() -> ort::Result<()> {
    #[allow(unused_mut)]
    let mut eps = vec![
        Ep {
            name: "ort-cpu",
            device: AllocationDevice::CPU,
            dispatch: || None,
        },
        Ep {
            name: "ort-webgpu",
            device: AllocationDevice::WEBGPU_BUFFER,
            dispatch: || Some(WebGPU::default().build().error_on_failure()),
        },
    ];
    #[cfg(target_os = "windows")]
    eps.push(Ep {
        name: "ort-directml",
        device: AllocationDevice::DIRECTML,
        dispatch: || Some(DirectML::default().build().error_on_failure()),
    });
    #[cfg(target_os = "macos")]
    eps.push(Ep {
        name: "ort-coreml",
        device: AllocationDevice::CPU,
        dispatch: || Some(CoreML::default().build().error_on_failure()),
    });
    for ep in &eps {
        for res in [&FAST, &BALANCED, &ACCURATE] {
            run(ep, res)?;
        }
    }
    Ok(())
}
