#include <torch/extension.h>

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

#define CHECK_CUDA(value) \
  TORCH_CHECK(value.is_cuda(), #value " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(value) \
  TORCH_CHECK(value.is_contiguous(), #value " must be contiguous")
#define CHECK_CUDA_CONTIGUOUS(value) \
  CHECK_CUDA(value);                      \
  CHECK_CONTIGUOUS(value)

namespace rectangular_ewa {

void forward_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    const at::Tensor &depths,
    const at::Tensor &environment_mask,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    float transmittance_cutoff,
    at::Tensor &rgb,
    at::Tensor &all_transmittance,
    at::Tensor &raw_depth,
    at::Tensor &base_transmittance,
    at::Tensor &last_all,
    at::Tensor &last_base);

void backward_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    const at::Tensor &depths,
    const at::Tensor &environment_mask,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    const at::Tensor &all_transmittance,
    const at::Tensor &base_transmittance,
    const at::Tensor &last_all,
    const at::Tensor &last_base,
    const at::Tensor &rgb_gradient,
    const at::Tensor &raw_depth_gradient,
    const at::Tensor &base_transmittance_gradient,
    const at::Tensor &all_transmittance_gradient,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    float transmittance_cutoff,
    at::Tensor &means2d_gradient,
    at::Tensor &conics_gradient,
    at::Tensor &opacities_gradient,
    at::Tensor &colors_gradient,
    at::Tensor &depths_gradient);

void masked_contribution_launch(
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

void masked_forward_contribution_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &environment_mask,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    float transmittance_cutoff,
    at::Tensor &contribution);

void masked_visibility_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &environment_mask,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    int64_t width,
    int64_t height,
    float alpha_cutoff,
    float transmittance_cutoff,
    at::Tensor &visibility);

void check_tile_lists(
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height) {
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(flatten_ids);
  TORCH_CHECK(offsets.scalar_type() == at::kInt,
              "offsets must use int32");
  TORCH_CHECK(flatten_ids.scalar_type() == at::kInt,
              "flatten_ids must use int32");
  TORCH_CHECK(offsets.dim() == 3 && offsets.size(0) == 1,
              "offsets must have shape [1, tiles_height, tiles_width]");
  const int64_t expected_tiles_width = (width + 15) / 16;
  const int64_t expected_tiles_height = (height + 7) / 8;
  TORCH_CHECK(offsets.size(1) == expected_tiles_height &&
                  offsets.size(2) == expected_tiles_width,
              "offsets do not match the rectangular EWA render shape");
}

void check_parameters(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &colors,
    const torch::Tensor &depths,
    const torch::Tensor &environment_mask,
    const torch::Tensor &pixel_mask,
    const int64_t width,
    const int64_t height,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(colors);
  CHECK_CUDA_CONTIGUOUS(depths);
  CHECK_CUDA_CONTIGUOUS(environment_mask);
  CHECK_CUDA_CONTIGUOUS(pixel_mask);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat &&
                  conics.scalar_type() == at::kFloat &&
                  opacities.scalar_type() == at::kFloat &&
                  colors.scalar_type() == at::kFloat &&
                  depths.scalar_type() == at::kFloat,
              "rectangular EWA parameters must use float32");
  TORCH_CHECK(environment_mask.scalar_type() == at::kBool &&
                  pixel_mask.scalar_type() == at::kBool,
              "rectangular EWA masks must use bool dtype");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}) &&
                  opacities.sizes() == at::IntArrayRef({point_count}) &&
                  colors.sizes() == at::IntArrayRef({point_count, 3}) &&
                  depths.sizes() == at::IntArrayRef({point_count}) &&
                  environment_mask.sizes() ==
                      at::IntArrayRef({point_count}),
              "rectangular EWA parameter shapes do not agree");
  TORCH_CHECK(pixel_mask.numel() == 0 ||
                  pixel_mask.sizes() == at::IntArrayRef({height, width}),
              "pixel_mask must be empty or have shape [height, width]");
  TORCH_CHECK(width > 0 && height > 0,
              "rectangular EWA dimensions must be positive");
  TORCH_CHECK(alpha_cutoff > 0.0 && alpha_cutoff <= 1.0,
              "rectangular EWA alpha cutoff must be in (0, 1]");
  TORCH_CHECK(transmittance_cutoff >= 0.0 && transmittance_cutoff <= 1.0,
              "rectangular EWA transmittance cutoff must be in [0, 1]");
  TORCH_CHECK(conics.device() == means2d.device() &&
                  opacities.device() == means2d.device() &&
                  colors.device() == means2d.device() &&
                  depths.device() == means2d.device() &&
                  environment_mask.device() == means2d.device() &&
                  pixel_mask.device() == means2d.device(),
              "rectangular EWA tensors must share one CUDA device");
}

std::vector<torch::Tensor> forward(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &colors,
    const torch::Tensor &depths,
    const torch::Tensor &environment_mask,
    const torch::Tensor &pixel_mask,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  check_parameters(means2d, conics, opacities, colors, depths,
                   environment_mask, pixel_mask, width, height, alpha_cutoff,
                   transmittance_cutoff);
  check_tile_lists(offsets, flatten_ids, width, height);
  TORCH_CHECK(offsets.device() == means2d.device() &&
                  flatten_ids.device() == means2d.device(),
              "rectangular EWA tile lists must share the render device");
  const auto output_options = means2d.options();
  const auto index_options = output_options.dtype(at::kInt);
  auto rgb = torch::zeros({1, height, width, 3}, output_options);
  auto all_transmittance = torch::ones({1, height, width, 1}, output_options);
  auto raw_depth = torch::zeros({1, height, width, 1}, output_options);
  auto base_transmittance =
      torch::ones({1, height, width, 1}, output_options);
  auto last_all = torch::full({height, width}, -1, index_options);
  auto last_base = torch::full({height, width}, -1, index_options);
  forward_launch(means2d, conics, opacities, colors, depths, environment_mask,
                 pixel_mask, offsets, flatten_ids, width, height,
                 static_cast<float>(alpha_cutoff),
                 static_cast<float>(transmittance_cutoff), rgb,
                 all_transmittance, raw_depth, base_transmittance, last_all,
                 last_base);
  return {rgb, all_transmittance, raw_depth, base_transmittance, last_all,
          last_base};
}

std::vector<torch::Tensor> backward(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &colors,
    const torch::Tensor &depths,
    const torch::Tensor &environment_mask,
    const torch::Tensor &pixel_mask,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const torch::Tensor &all_transmittance,
    const torch::Tensor &base_transmittance,
    const torch::Tensor &last_all,
    const torch::Tensor &last_base,
    const torch::Tensor &rgb_gradient,
    const torch::Tensor &raw_depth_gradient,
    const torch::Tensor &base_transmittance_gradient,
    const torch::Tensor &all_transmittance_gradient,
    const int64_t width,
    const int64_t height,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  check_parameters(means2d, conics, opacities, colors, depths,
                   environment_mask, pixel_mask, width, height, alpha_cutoff,
                   transmittance_cutoff);
  check_tile_lists(offsets, flatten_ids, width, height);
  CHECK_CUDA_CONTIGUOUS(all_transmittance);
  CHECK_CUDA_CONTIGUOUS(base_transmittance);
  CHECK_CUDA_CONTIGUOUS(last_all);
  CHECK_CUDA_CONTIGUOUS(last_base);
  CHECK_CUDA_CONTIGUOUS(rgb_gradient);
  CHECK_CUDA_CONTIGUOUS(raw_depth_gradient);
  CHECK_CUDA_CONTIGUOUS(base_transmittance_gradient);
  CHECK_CUDA_CONTIGUOUS(all_transmittance_gradient);
  TORCH_CHECK(all_transmittance.scalar_type() == at::kFloat &&
                  base_transmittance.scalar_type() == at::kFloat &&
                  rgb_gradient.scalar_type() == at::kFloat &&
                  raw_depth_gradient.scalar_type() == at::kFloat &&
                  base_transmittance_gradient.scalar_type() == at::kFloat &&
                  all_transmittance_gradient.scalar_type() == at::kFloat,
              "rectangular EWA backward state must use float32");
  TORCH_CHECK(last_all.scalar_type() == at::kInt &&
                  last_base.scalar_type() == at::kInt,
              "rectangular EWA last-index state must use int32");
  const auto scalar_shape = at::IntArrayRef({1, height, width, 1});
  TORCH_CHECK(all_transmittance.sizes() == scalar_shape &&
                  base_transmittance.sizes() == scalar_shape &&
                  raw_depth_gradient.sizes() == scalar_shape &&
                  base_transmittance_gradient.sizes() == scalar_shape &&
                  all_transmittance_gradient.sizes() == scalar_shape &&
                  rgb_gradient.sizes() ==
                      at::IntArrayRef({1, height, width, 3}) &&
                  last_all.sizes() == at::IntArrayRef({height, width}) &&
                  last_base.sizes() == at::IntArrayRef({height, width}),
              "rectangular EWA backward tensors do not match the image");
  const auto output_options = means2d.options();
  auto means2d_gradient = torch::zeros_like(means2d, output_options);
  auto conics_gradient = torch::zeros_like(conics, output_options);
  auto opacities_gradient = torch::zeros_like(opacities, output_options);
  auto colors_gradient = torch::zeros_like(colors, output_options);
  auto depths_gradient = torch::zeros_like(depths, output_options);
  backward_launch(
      means2d, conics, opacities, colors, depths, environment_mask, pixel_mask,
      offsets, flatten_ids, all_transmittance, base_transmittance, last_all,
      last_base, rgb_gradient, raw_depth_gradient,
      base_transmittance_gradient, all_transmittance_gradient, width, height,
      static_cast<float>(alpha_cutoff),
      static_cast<float>(transmittance_cutoff), means2d_gradient,
      conics_gradient, opacities_gradient, colors_gradient, depths_gradient);
  return {means2d_gradient, conics_gradient, opacities_gradient,
          colors_gradient, depths_gradient};
}

torch::Tensor masked_contribution(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &pixel_mask,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(pixel_mask);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat &&
                  conics.scalar_type() == at::kFloat &&
                  opacities.scalar_type() == at::kFloat,
              "rectangular EWA contribution parameters must use float32");
  TORCH_CHECK(pixel_mask.scalar_type() == at::kBool,
              "rectangular EWA contribution mask must use bool dtype");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}) &&
                  opacities.sizes() == at::IntArrayRef({point_count}),
              "rectangular EWA contribution shapes do not agree");
  TORCH_CHECK(pixel_mask.numel() == 0 ||
                  pixel_mask.sizes() == at::IntArrayRef({height, width}),
              "pixel_mask must be empty or have shape [height, width]");
  TORCH_CHECK(width > 0 && height > 0,
              "rectangular EWA dimensions must be positive");
  TORCH_CHECK(alpha_cutoff > 0.0 && alpha_cutoff <= 1.0 &&
                  transmittance_cutoff >= 0.0 &&
                  transmittance_cutoff <= 1.0,
              "rectangular EWA cutoffs are invalid");
  check_tile_lists(offsets, flatten_ids, width, height);
  TORCH_CHECK(conics.device() == means2d.device() &&
                  opacities.device() == means2d.device() &&
                  pixel_mask.device() == means2d.device() &&
                  offsets.device() == means2d.device() &&
                  flatten_ids.device() == means2d.device(),
              "rectangular EWA contribution tensors must share one device");
  auto contribution = torch::zeros({point_count}, means2d.options());
  masked_contribution_launch(
      means2d, conics, opacities, pixel_mask, offsets, flatten_ids, width,
      height, static_cast<float>(alpha_cutoff),
      static_cast<float>(transmittance_cutoff), contribution);
  return contribution;
}

torch::Tensor masked_forward_contribution(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &environment_mask,
    const torch::Tensor &pixel_mask,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(environment_mask);
  CHECK_CUDA_CONTIGUOUS(pixel_mask);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat &&
                  conics.scalar_type() == at::kFloat &&
                  opacities.scalar_type() == at::kFloat,
              "rectangular EWA forward contribution parameters must use "
              "float32");
  TORCH_CHECK(environment_mask.scalar_type() == at::kBool &&
                  pixel_mask.scalar_type() == at::kBool,
              "rectangular EWA forward contribution masks must use bool "
              "dtype");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}) &&
                  opacities.sizes() == at::IntArrayRef({point_count}) &&
                  environment_mask.sizes() ==
                      at::IntArrayRef({point_count}),
              "rectangular EWA forward contribution shapes do not agree");
  TORCH_CHECK(pixel_mask.numel() == 0 ||
                  pixel_mask.sizes() == at::IntArrayRef({height, width}),
              "pixel_mask must be empty or have shape [height, width]");
  TORCH_CHECK(width > 0 && height > 0,
              "rectangular EWA dimensions must be positive");
  TORCH_CHECK(alpha_cutoff > 0.0 && alpha_cutoff <= 1.0 &&
                  transmittance_cutoff >= 0.0 &&
                  transmittance_cutoff <= 1.0,
              "rectangular EWA cutoffs are invalid");
  check_tile_lists(offsets, flatten_ids, width, height);
  TORCH_CHECK(conics.device() == means2d.device() &&
                  opacities.device() == means2d.device() &&
                  environment_mask.device() == means2d.device() &&
                  pixel_mask.device() == means2d.device() &&
                  offsets.device() == means2d.device() &&
                  flatten_ids.device() == means2d.device(),
              "rectangular EWA forward contribution tensors must share one "
              "device");
  auto contribution = torch::zeros({point_count}, means2d.options());
  masked_forward_contribution_launch(
      means2d, conics, opacities, environment_mask, pixel_mask, offsets,
      flatten_ids, width, height, static_cast<float>(alpha_cutoff),
      static_cast<float>(transmittance_cutoff), contribution);
  return contribution;
}

torch::Tensor masked_forward_visibility(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &environment_mask,
    const torch::Tensor &pixel_mask,
    const torch::Tensor &offsets,
    const torch::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const double alpha_cutoff,
    const double transmittance_cutoff) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means2d));
  CHECK_CUDA_CONTIGUOUS(means2d);
  CHECK_CUDA_CONTIGUOUS(conics);
  CHECK_CUDA_CONTIGUOUS(opacities);
  CHECK_CUDA_CONTIGUOUS(environment_mask);
  CHECK_CUDA_CONTIGUOUS(pixel_mask);
  TORCH_CHECK(means2d.scalar_type() == at::kFloat &&
                  conics.scalar_type() == at::kFloat &&
                  opacities.scalar_type() == at::kFloat,
              "rectangular EWA visibility parameters must use float32");
  TORCH_CHECK(environment_mask.scalar_type() == at::kBool &&
                  pixel_mask.scalar_type() == at::kBool,
              "rectangular EWA visibility masks must use bool dtype");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2,
              "means2d must have shape [N, 2]");
  const auto point_count = means2d.size(0);
  TORCH_CHECK(conics.sizes() == at::IntArrayRef({point_count, 3}) &&
                  opacities.sizes() == at::IntArrayRef({point_count}) &&
                  environment_mask.sizes() ==
                      at::IntArrayRef({point_count}),
              "rectangular EWA visibility shapes do not agree");
  TORCH_CHECK(pixel_mask.numel() == 0 ||
                  pixel_mask.sizes() == at::IntArrayRef({height, width}),
              "pixel_mask must be empty or have shape [height, width]");
  TORCH_CHECK(width > 0 && height > 0,
              "rectangular EWA dimensions must be positive");
  TORCH_CHECK(alpha_cutoff > 0.0 && alpha_cutoff <= 1.0 &&
                  transmittance_cutoff >= 0.0 &&
                  transmittance_cutoff <= 1.0,
              "rectangular EWA cutoffs are invalid");
  check_tile_lists(offsets, flatten_ids, width, height);
  TORCH_CHECK(conics.device() == means2d.device() &&
                  opacities.device() == means2d.device() &&
                  environment_mask.device() == means2d.device() &&
                  pixel_mask.device() == means2d.device() &&
                  offsets.device() == means2d.device() &&
                  flatten_ids.device() == means2d.device(),
              "rectangular EWA visibility tensors must share one device");
  auto visibility = torch::zeros(
      {point_count}, means2d.options().dtype(at::kInt));
  masked_visibility_launch(
      means2d, conics, opacities, environment_mask, pixel_mask, offsets,
      flatten_ids, width, height, static_cast<float>(alpha_cutoff),
      static_cast<float>(transmittance_cutoff), visibility);
  return visibility;
}

}  // namespace rectangular_ewa

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &rectangular_ewa::forward);
  module.def("backward", &rectangular_ewa::backward);
  module.def("masked_contribution", &rectangular_ewa::masked_contribution);
  module.def(
      "masked_forward_contribution",
      &rectangular_ewa::masked_forward_contribution);
  module.def(
      "masked_forward_visibility",
      &rectangular_ewa::masked_forward_visibility);
}
