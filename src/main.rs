mod models;

use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::model::Model;

fn main() {
    let device = NdArrayDevice::default();

    // Create model instance and load weights from target dir default device.
    let model: Model<NdArray<f32>> = Model::default();

    // TODO: Adapt below once model can be loaded in build step.

    // Create input tensor (replace with your actual input)
    // let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 3, 224, 224], &device);

    // Perform inference
    // let output = model.forward(input);

    // println!("Model output: {:?}", output);
}
