#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace layered_depth {

constexpr int kTileSize = 16;
constexpr int kTilePixels = kTileSize * kTileSize;
constexpr float kMaximumAlpha = 0.99f;
constexpr float kTransmittanceThreshold = 1.0e-4f;

__global__ void masked_ewa_contribution_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const bool *__restrict__ pixel_mask,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    const bool has_pixel_mask,
    float *__restrict__ contribution) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileSize + local_x;
  const int pixel_x = blockIdx.x * kTileSize + local_x;
  const int pixel_y = blockIdx.y * kTileSize + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
                           ? offsets[tile_index + 1]
                           : flatten_count;

  __shared__ int shared_ids[kTilePixels];
  __shared__ float shared_means_x[kTilePixels];
  __shared__ float shared_means_y[kTilePixels];
  __shared__ float shared_conic_a[kTilePixels];
  __shared__ float shared_conic_b[kTilePixels];
  __shared__ float shared_conic_c[kTilePixels];
  __shared__ float shared_opacities[kTilePixels];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const bool active = inside &&
      (!has_pixel_mask || pixel_mask[pixel_index]);
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  float current_transmittance = 1.0f;

  for (int chunk_start = list_start; chunk_start < list_end;
       chunk_start += kTilePixels) {
    const int source_index = chunk_start + thread_index;
    if (source_index < list_end) {
      const int gaussian_id = flatten_ids[source_index];
      shared_ids[thread_index] = gaussian_id;
      shared_means_x[thread_index] = means2d[2 * gaussian_id];
      shared_means_y[thread_index] = means2d[2 * gaussian_id + 1];
      shared_conic_a[thread_index] = conics[3 * gaussian_id];
      shared_conic_b[thread_index] = conics[3 * gaussian_id + 1];
      shared_conic_c[thread_index] = conics[3 * gaussian_id + 2];
      shared_opacities[thread_index] = opacities[gaussian_id];
    }
    __syncthreads();

    if (active) {
      const int batch_size = min(kTilePixels, list_end - chunk_start);
      for (int local_gaussian = 0; local_gaussian < batch_size;
           ++local_gaussian) {
        const float delta_x = shared_means_x[local_gaussian] - sample_x;
        const float delta_y = shared_means_y[local_gaussian] - sample_y;
        const float sigma = 0.5f * (
            shared_conic_a[local_gaussian] * delta_x * delta_x +
            shared_conic_c[local_gaussian] * delta_y * delta_y) +
            shared_conic_b[local_gaussian] * delta_x * delta_y;
        const float alpha = fminf(
            kMaximumAlpha,
            shared_opacities[local_gaussian] * __expf(-sigma));
        if (sigma < 0.0f || alpha < alpha_cutoff) {
          continue;
        }
        const int gaussian_id = shared_ids[local_gaussian];
        atomicAdd(contribution + gaussian_id,
                  current_transmittance * alpha);
        current_transmittance *= 1.0f - alpha;
        if (current_transmittance < transmittance_cutoff) {
          break;
        }
      }
    }
    __syncthreads();
  }
}

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
    at::Tensor &contribution) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(tiles_width, tiles_height);
  masked_ewa_contribution_kernel<<<
      grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), pixel_mask.data_ptr<bool>(),
      offsets.data_ptr<int>(),
      flatten_ids.data_ptr<int>(), static_cast<int>(flatten_ids.numel()),
      static_cast<int>(width), static_cast<int>(height), tiles_width,
      alpha_cutoff, transmittance_cutoff, pixel_mask.numel() != 0,
      contribution.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void forward_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const float *__restrict__ depths,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    const int maximum_chunk_count,
    float *__restrict__ chunk_transparencies,
    float *__restrict__ raw_depth,
    float *__restrict__ transparency,
    float *__restrict__ median_depth) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileSize + local_x;
  const int pixel_x = blockIdx.x * kTileSize + local_x;
  const int pixel_y = blockIdx.y * kTileSize + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
                           ? offsets[tile_index + 1]
                           : flatten_count;

  __shared__ float shared_means_x[kTilePixels];
  __shared__ float shared_means_y[kTilePixels];
  __shared__ float shared_conic_a[kTilePixels];
  __shared__ float shared_conic_b[kTilePixels];
  __shared__ float shared_conic_c[kTilePixels];
  __shared__ float shared_opacities[kTilePixels];
  __shared__ float shared_depths[kTilePixels];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  float current_raw_depth = 0.0f;
  float current_transparency = 1.0f;
  float current_median_depth = 0.0f;
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;

  int chunk_index = 0;
  for (int chunk_start = list_start; chunk_start < list_end;
       chunk_start += kTilePixels, ++chunk_index) {
    const int source_index = chunk_start + thread_index;
    if (source_index < list_end) {
      const int gaussian_id = flatten_ids[source_index];
      shared_means_x[thread_index] = means2d[2 * gaussian_id];
      shared_means_y[thread_index] = means2d[2 * gaussian_id + 1];
      shared_conic_a[thread_index] = conics[3 * gaussian_id];
      shared_conic_b[thread_index] = conics[3 * gaussian_id + 1];
      shared_conic_c[thread_index] = conics[3 * gaussian_id + 2];
      shared_opacities[thread_index] = opacities[gaussian_id];
      shared_depths[thread_index] = depths[gaussian_id];
    }
    __syncthreads();

    if (inside) {
      if (chunk_transparencies != nullptr) {
        chunk_transparencies[
            pixel_index * maximum_chunk_count + chunk_index] =
            current_transparency;
      }
      const int batch_size = min(kTilePixels, list_end - chunk_start);
      float culling_transparency = current_transparency;
      double log_alpha_sum = 0.0;
      for (int local_gaussian = 0; local_gaussian < batch_size;
           ++local_gaussian) {
        const float delta_x = shared_means_x[local_gaussian] - sample_x;
        const float delta_y = shared_means_y[local_gaussian] - sample_y;
        const float sigma = 0.5f * (
            shared_conic_a[local_gaussian] * delta_x * delta_x +
            shared_conic_c[local_gaussian] * delta_y * delta_y) +
            shared_conic_b[local_gaussian] * delta_x * delta_y;
        const float selection_alpha = fminf(
            kMaximumAlpha,
            shared_opacities[local_gaussian] * __expf(-sigma));
        if (sigma < 0.0f || selection_alpha < alpha_cutoff) {
          continue;
        }
        const float next_culling_transparency =
            culling_transparency * (1.0f - selection_alpha);
        if (next_culling_transparency <= kTransmittanceThreshold) {
          break;
        }
        const float alpha = fminf(
            kMaximumAlpha,
            shared_opacities[local_gaussian] * expf(-sigma));
        if (alpha < alpha_cutoff) {
          continue;
        }
        const float log_alpha = log1pf(-alpha);
        log_alpha_sum += static_cast<double>(log_alpha);
        const double relative_after = exp(log_alpha_sum);
        const double before =
            static_cast<double>(current_transparency) * relative_after /
            static_cast<double>(1.0f - alpha);
        current_raw_depth += static_cast<float>(
            before * static_cast<double>(alpha) *
            static_cast<double>(shared_depths[local_gaussian]));
        const double after = before * static_cast<double>(1.0f - alpha);
        if (current_median_depth == 0.0f && before > 0.5 && after <= 0.5) {
          current_median_depth = shared_depths[local_gaussian];
        }
        culling_transparency = next_culling_transparency;
      }
      current_transparency *= static_cast<float>(exp(log_alpha_sum));
    }
    __syncthreads();
  }

  if (inside) {
    raw_depth[pixel_index] = current_raw_depth;
    transparency[pixel_index] = current_transparency;
    median_depth[pixel_index] = current_median_depth;
  }
}

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
    at::Tensor &median_depth) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(tiles_width, tiles_height);
  const int maximum_chunk_count = chunk_transparencies == nullptr
      ? 0
      : static_cast<int>(chunk_transparencies->size(1));
  float *chunk_transparencies_ptr = chunk_transparencies == nullptr
      ? nullptr
      : chunk_transparencies->data_ptr<float>();
  forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), depths.data_ptr<float>(),
      offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
      static_cast<int>(flatten_ids.numel()), static_cast<int>(width),
      static_cast<int>(height), tiles_width, alpha_cutoff,
      maximum_chunk_count, chunk_transparencies_ptr,
      raw_depth.data_ptr<float>(), transparency.data_ptr<float>(),
      median_depth.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void backward_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const float *__restrict__ depths,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const float *__restrict__ chunk_transparencies,
    const int maximum_chunk_count,
    const float *__restrict__ raw_depth_gradient,
    const float *__restrict__ transparency_gradient,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    float *__restrict__ means2d_gradient,
    float *__restrict__ conics_gradient,
    float *__restrict__ opacities_gradient,
    float *__restrict__ depths_gradient) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileSize + local_x;
  const int pixel_x = blockIdx.x * kTileSize + local_x;
  const int pixel_y = blockIdx.y * kTileSize + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
                           ? offsets[tile_index + 1]
                           : flatten_count;

  __shared__ int shared_ids[kTilePixels];
  __shared__ float shared_means_x[kTilePixels];
  __shared__ float shared_means_y[kTilePixels];
  __shared__ float shared_conic_a[kTilePixels];
  __shared__ float shared_conic_b[kTilePixels];
  __shared__ float shared_conic_c[kTilePixels];
  __shared__ float shared_opacities[kTilePixels];
  __shared__ float shared_depths[kTilePixels];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  const float raw_adjoint = inside ? raw_depth_gradient[pixel_index] : 0.0f;
  float transparency_adjoint =
      inside ? transparency_gradient[pixel_index] : 0.0f;
  const int chunk_count =
      (list_end - list_start + kTilePixels - 1) / kTilePixels;

  for (int chunk_index = chunk_count - 1; chunk_index >= 0; --chunk_index) {
    const int chunk_start = list_start + chunk_index * kTilePixels;
    const int source_index = chunk_start + thread_index;
    if (source_index < list_end) {
      const int gaussian_id = flatten_ids[source_index];
      shared_ids[thread_index] = gaussian_id;
      shared_means_x[thread_index] = means2d[2 * gaussian_id];
      shared_means_y[thread_index] = means2d[2 * gaussian_id + 1];
      shared_conic_a[thread_index] = conics[3 * gaussian_id];
      shared_conic_b[thread_index] = conics[3 * gaussian_id + 1];
      shared_conic_c[thread_index] = conics[3 * gaussian_id + 2];
      shared_opacities[thread_index] = opacities[gaussian_id];
      shared_depths[thread_index] = depths[gaussian_id];
    }
    __syncthreads();

    if (inside) {
      const int batch_size = min(kTilePixels, list_end - chunk_start);
      const float input_transparency = chunk_transparencies[
          pixel_index * maximum_chunk_count + chunk_index];
      float culling_transparency = input_transparency;
      double log_alpha_sum = 0.0;
      float raw_contribution = 0.0f;
      int last_included = -1;
      for (int local_gaussian = 0; local_gaussian < batch_size;
           ++local_gaussian) {
        const float delta_x = shared_means_x[local_gaussian] - sample_x;
        const float delta_y = shared_means_y[local_gaussian] - sample_y;
        const float sigma = 0.5f * (
            shared_conic_a[local_gaussian] * delta_x * delta_x +
            shared_conic_c[local_gaussian] * delta_y * delta_y) +
            shared_conic_b[local_gaussian] * delta_x * delta_y;
        const float selection_alpha = fminf(
            kMaximumAlpha,
            shared_opacities[local_gaussian] * __expf(-sigma));
        if (sigma < 0.0f || selection_alpha < alpha_cutoff) {
          continue;
        }
        const float next_culling_transparency =
            culling_transparency * (1.0f - selection_alpha);
        if (next_culling_transparency <= kTransmittanceThreshold) {
          break;
        }
        const float alpha = fminf(
            kMaximumAlpha,
            shared_opacities[local_gaussian] * expf(-sigma));
        if (alpha < alpha_cutoff) {
          continue;
        }
        log_alpha_sum += static_cast<double>(log1pf(-alpha));
        const double relative_after = exp(log_alpha_sum);
        const double before =
            static_cast<double>(input_transparency) * relative_after /
            static_cast<double>(1.0f - alpha);
        raw_contribution += static_cast<float>(
            before * static_cast<double>(alpha) *
            static_cast<double>(shared_depths[local_gaussian]));
        last_included = local_gaussian;
        culling_transparency = next_culling_transparency;
      }

      const double total_factor = exp(log_alpha_sum);
      const float chunk_factor = static_cast<float>(total_factor);
      const float output_transparency = input_transparency * chunk_factor;
      double suffix_factor = 1.0;
      double suffix_signal = 0.0;
      for (int local_gaussian = last_included; local_gaussian >= 0;
           --local_gaussian) {
        const float delta_x = shared_means_x[local_gaussian] - sample_x;
        const float delta_y = shared_means_y[local_gaussian] - sample_y;
        const float sigma = 0.5f * (
            shared_conic_a[local_gaussian] * delta_x * delta_x +
            shared_conic_c[local_gaussian] * delta_y * delta_y) +
            shared_conic_b[local_gaussian] * delta_x * delta_y;
        const float selection_alpha = fminf(
            kMaximumAlpha,
            shared_opacities[local_gaussian] * __expf(-sigma));
        if (sigma < 0.0f || selection_alpha < alpha_cutoff) {
          continue;
        }
        const float value_exponent = expf(-sigma);
        const float unclamped_alpha =
            shared_opacities[local_gaussian] * value_exponent;
        const float alpha = fminf(kMaximumAlpha, unclamped_alpha);
        if (alpha < alpha_cutoff) {
          continue;
        }
        const double one_minus_alpha = static_cast<double>(1.0f - alpha);
        suffix_factor *= one_minus_alpha;
        const double prefix = total_factor / suffix_factor;
        const double depth =
            static_cast<double>(shared_depths[local_gaussian]);
        const double alpha_double = static_cast<double>(alpha);
        const double raw_alpha_gradient =
            static_cast<double>(input_transparency) *
            (prefix * depth - suffix_signal / one_minus_alpha);
        const double transparency_alpha_gradient =
            -static_cast<double>(output_transparency) / one_minus_alpha;
        const float alpha_gradient = static_cast<float>(
            static_cast<double>(raw_adjoint) * raw_alpha_gradient +
            static_cast<double>(transparency_adjoint) *
            transparency_alpha_gradient);
        const float depth_gradient = raw_adjoint * static_cast<float>(
            static_cast<double>(input_transparency) * prefix * alpha_double);
        const int gaussian_id = shared_ids[local_gaussian];
        atomicAdd(depths_gradient + gaussian_id, depth_gradient);
        if (unclamped_alpha <= kMaximumAlpha) {
          atomicAdd(opacities_gradient + gaussian_id,
                    alpha_gradient * value_exponent);
          const float sigma_gradient = -alpha_gradient * unclamped_alpha;
          atomicAdd(means2d_gradient + 2 * gaussian_id,
                    sigma_gradient * (
                        shared_conic_a[local_gaussian] * delta_x +
                        shared_conic_b[local_gaussian] * delta_y));
          atomicAdd(means2d_gradient + 2 * gaussian_id + 1,
                    sigma_gradient * (
                        shared_conic_b[local_gaussian] * delta_x +
                        shared_conic_c[local_gaussian] * delta_y));
          atomicAdd(conics_gradient + 3 * gaussian_id,
                    sigma_gradient * 0.5f * delta_x * delta_x);
          atomicAdd(conics_gradient + 3 * gaussian_id + 1,
                    sigma_gradient * delta_x * delta_y);
          atomicAdd(conics_gradient + 3 * gaussian_id + 2,
                    sigma_gradient * 0.5f * delta_y * delta_y);
        }
        suffix_signal += prefix * alpha_double * depth;
      }
      const float normalized_signal = input_transparency == 0.0f
          ? 0.0f
          : raw_contribution / input_transparency;
      transparency_adjoint = transparency_adjoint * chunk_factor +
          raw_adjoint * normalized_signal;
    }
    __syncthreads();
  }
}

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
    at::Tensor &depths_gradient) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(tiles_width, tiles_height);
  backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), depths.data_ptr<float>(),
      offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
      static_cast<int>(flatten_ids.numel()),
      chunk_transparencies.data_ptr<float>(),
      static_cast<int>(chunk_transparencies.size(1)),
      raw_depth_gradient.data_ptr<float>(),
      transparency_gradient.data_ptr<float>(), static_cast<int>(width),
      static_cast<int>(height), tiles_width, alpha_cutoff,
      means2d_gradient.data_ptr<float>(), conics_gradient.data_ptr<float>(),
      opacities_gradient.data_ptr<float>(), depths_gradient.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace layered_depth
