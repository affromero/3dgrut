#include <torch/extension.h>

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

#define CHECK_CUDA(value) \
  TORCH_CHECK(value.is_cuda(), #value " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(value) \
  TORCH_CHECK(value.is_contiguous(), #value " must be contiguous")
#define CHECK_CUDA_CONTIGUOUS(value) \
  CHECK_CUDA(value);                 \
  CHECK_CONTIGUOUS(value)

namespace layered_depth {

void forward_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &depths,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    at::Tensor *chunk_transparencies,
    at::Tensor &raw_depth,
    at::Tensor &transparency,
    at::Tensor &median_depth);

void backward_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &depths,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    const at::Tensor &chunk_transparencies,
    const at::Tensor &raw_depth_gradient,
    const at::Tensor &transparency_gradient,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    at::Tensor &means2d_gradient,
    at::Tensor &conics_gradient,
    at::Tensor &opacities_gradient,
    at::Tensor &depths_gradient);

void masked_ewa_contribution_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    float transmittance_cutoff,
    at::Tensor &contribution);

std::vector<torch::Tensor> forward(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &depths,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const int64_t tile_size,
    const double alpha_cutoff,
    const int64_t maximum_chunk_count,
    const bool save_chunk_transparencies) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(depths);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(flatten_ids);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat,
              "means2d must use float32");
  TORCH_CHECK(conics.scalar_type() == at::kFloat,
              "conics must use float32");
  TORCH_CHECK(opacities.scalar_type() == at::kFloat,
              "opacities must use float32");
  TORCH_CHECK(depths.scalar_type() == at::kFloat,
              "depths must use float32");
  TORCH_CHECK(offsets.scalar_type() == at::kInt,
              "offsets must use int32");
  TORCH_CHECK(flatten_ids.scalar_type() == at::kInt,
              "flatten_ids must use int32");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}),
              "conics must have shape [N, 3]");
  TORCH_CHECK(opacities.sizes() == at::IntArrayRef({point_count}),
              "opacities must have shape [N]");
  TORCH_CHECK(depths.sizes() == at::IntArrayRef({point_count}),
              "depths must have shape [N]");
  TORCH_CHECK(offsets.dim() == 3 && offsets.size(0) == 1,
              "offsets must have shape [1, tiles_height, tiles_width]");
  TORCH_CHECK(tile_size == 16,
              "layered depth compositor requires 16-pixel tiles");
  TORCH_CHECK(width > 0 && height > 0,
              "layered depth compositor dimensions must be positive");
  TORCH_CHECK(alpha_cutoff > 0.0 && alpha_cutoff <= 1.0,
              "layered depth alpha cutoff must be in (0, 1]");
  TORCH_CHECK(!save_chunk_transparencies || maximum_chunk_count > 0,
              "maximum chunk count must be positive when saving state");
  const int64_t expected_tiles_width = (width + tile_size - 1) / tile_size;
  const int64_t expected_tiles_height = (height + tile_size - 1) / tile_size;
  TORCH_CHECK(offsets.size(1) == expected_tiles_height &&
                  offsets.size(2) == expected_tiles_width,
              "offsets do not match the render shape");
  const auto output_options = means2d.options();
  auto raw_depth = torch::zeros({1, height, width, 1}, output_options);
  auto transparency = torch::ones({1, height, width, 1}, output_options);
  auto median_depth = torch::zeros({1, height, width, 1}, output_options);
  auto chunk_transparencies = save_chunk_transparencies
      ? torch::empty({height * width, maximum_chunk_count}, output_options)
      : torch::empty({0}, output_options);
  auto *chunk_transparencies_ptr =
      save_chunk_transparencies ? &chunk_transparencies : nullptr;
  forward_launch(means2d, conics, opacities, depths, offsets, flatten_ids,
                 width, height, static_cast<float>(alpha_cutoff),
                 chunk_transparencies_ptr, raw_depth, transparency,
                 median_depth);
  return {raw_depth, transparency, median_depth, chunk_transparencies};
}

std::vector<torch::Tensor> backward(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &depths,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const torch::Tensor &chunk_transparencies,
    const torch::Tensor &raw_depth_gradient,
    const torch::Tensor &transparency_gradient,
    const int64_t width,
    const int64_t height,
    const int64_t tile_size,
    const double alpha_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(depths);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(flatten_ids);
  CHECK_CUDA_CONTIGUOUS(chunk_transparencies);
  CHECK_CUDA_CONTIGUOUS(raw_depth_gradient);
  CHECK_CUDA_CONTIGUOUS(transparency_gradient);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat &&
                  conics.scalar_type() == at::kFloat &&
                  opacities.scalar_type() == at::kFloat &&
                  depths.scalar_type() == at::kFloat,
              "layered depth parameters must use float32");
  TORCH_CHECK(offsets.scalar_type() == at::kInt &&
                  flatten_ids.scalar_type() == at::kInt,
              "layered depth tile lists must use int32");
  TORCH_CHECK(chunk_transparencies.scalar_type() == at::kFloat &&
                  raw_depth_gradient.scalar_type() == at::kFloat &&
                  transparency_gradient.scalar_type() == at::kFloat,
              "layered depth state and gradients must use float32");
  TORCH_CHECK(tile_size == 16,
              "layered depth compositor requires 16-pixel tiles");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}) &&
                  opacities.sizes() == at::IntArrayRef({point_count}) &&
                  depths.sizes() == at::IntArrayRef({point_count}),
              "layered depth parameter shapes do not agree");
  TORCH_CHECK(offsets.dim() == 3 && offsets.size(0) == 1,
              "offsets must have shape [1, tiles_height, tiles_width]");
  TORCH_CHECK(raw_depth_gradient.sizes() ==
                  at::IntArrayRef({1, height, width, 1}) &&
                  transparency_gradient.sizes() ==
                  at::IntArrayRef({1, height, width, 1}),
              "layered depth output gradients have the wrong shape");
  TORCH_CHECK(chunk_transparencies.dim() == 2 &&
                  chunk_transparencies.size(0) == height * width &&
                  chunk_transparencies.size(1) > 0,
              "layered depth saved transmittance state has the wrong shape");
  const auto output_options = means2d.options();
  auto means2d_gradient = torch::zeros_like(means2d, output_options);
  auto conics_gradient = torch::zeros_like(conics, output_options);
  auto opacities_gradient = torch::zeros_like(opacities, output_options);
  auto depths_gradient = torch::zeros_like(depths, output_options);
  backward_launch(
      means2d, conics, opacities, depths, offsets, flatten_ids,
      chunk_transparencies, raw_depth_gradient, transparency_gradient, width,
      height, static_cast<float>(alpha_cutoff), means2d_gradient,
      conics_gradient, opacities_gradient, depths_gradient);
  return {means2d_gradient, conics_gradient, opacities_gradient,
          depths_gradient};
}

torch::Tensor masked_ewa_contribution(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &pixel_mask,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const int64_t tile_size,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(pixel_mask);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(flatten_ids);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat,
              "means2d must use float32");
  TORCH_CHECK(conics.scalar_type() == at::kFloat,
              "conics must use float32");
  TORCH_CHECK(opacities.scalar_type() == at::kFloat,
              "opacities must use float32");
  TORCH_CHECK(pixel_mask.scalar_type() == at::kBool,
              "pixel_mask must use bool");
  TORCH_CHECK(offsets.scalar_type() == at::kInt,
              "offsets must use int32");
  TORCH_CHECK(flatten_ids.scalar_type() == at::kInt,
              "flatten_ids must use int32");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}),
              "conics must have shape [N, 3]");
  TORCH_CHECK(opacities.sizes() == at::IntArrayRef({point_count}),
              "opacities must have shape [N]");
  TORCH_CHECK(pixel_mask.numel() == 0 ||
                  pixel_mask.sizes() == at::IntArrayRef({height, width}),
              "pixel_mask must be empty or have shape [height, width]");
  TORCH_CHECK(offsets.dim() == 3 && offsets.size(0) == 1,
              "offsets must have shape [1, tiles_height, tiles_width]");
  TORCH_CHECK(tile_size == 16,
              "masked EWA contribution requires 16-pixel tiles");
  TORCH_CHECK(width > 0 && height > 0,
              "masked EWA contribution dimensions must be positive");
  TORCH_CHECK(alpha_cutoff > 0.0 && alpha_cutoff <= 1.0,
              "masked EWA alpha cutoff must be in (0, 1]");
  TORCH_CHECK(transmittance_cutoff >= 0.0 && transmittance_cutoff <= 1.0,
              "masked EWA transmittance cutoff must be in [0, 1]");
  const int64_t expected_tiles_width = (width + tile_size - 1) / tile_size;
  const int64_t expected_tiles_height = (height + tile_size - 1) / tile_size;
  TORCH_CHECK(offsets.size(1) == expected_tiles_height &&
                  offsets.size(2) == expected_tiles_width,
              "offsets do not match the render shape");
  TORCH_CHECK(conics.device() == means2d.device() &&
                  opacities.device() == means2d.device() &&
                  pixel_mask.device() == means2d.device() &&
                  offsets.device() == means2d.device() &&
                  flatten_ids.device() == means2d.device(),
              "masked EWA contribution tensors must share one CUDA device");
  auto contribution = torch::zeros({point_count}, means2d.options());
  masked_ewa_contribution_launch(
      means2d, conics, opacities, pixel_mask, offsets, flatten_ids, width,
      height, static_cast<float>(alpha_cutoff),
      static_cast<float>(transmittance_cutoff), contribution);
  return contribution;
}

}  // namespace layered_depth

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &layered_depth::forward);
  module.def("backward", &layered_depth::backward);
  module.def(
      "masked_ewa_contribution", &layered_depth::masked_ewa_contribution);
}
