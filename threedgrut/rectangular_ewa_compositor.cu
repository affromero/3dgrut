#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace rectangular_ewa {

constexpr int kTileWidth = 16;
constexpr int kTileHeight = 8;
constexpr int kThreadsPerTile = kTileWidth * kTileHeight;
constexpr int kWarpCount = kThreadsPerTile / 32;
constexpr int kGradientCount = 10;
constexpr float kMaximumAlpha = 0.99f;

struct AlphaSample {
  float alpha;
  float delta_x;
  float delta_y;
  float sigma;
  bool active;
};

__device__ AlphaSample sample_alpha(
    const float mean_x,
    const float mean_y,
    const float conic_a,
    const float conic_b,
    const float conic_c,
    const float opacity,
    const float pixel_x,
    const float pixel_y,
    const bool enabled,
    const float alpha_cutoff) {
  if (!enabled) {
    return {0.0f, 0.0f, 0.0f, 0.0f, false};
  }
  const float delta_x = mean_x - pixel_x;
  const float delta_y = mean_y - pixel_y;
  const float conic_a_delta_x = __fmul_rn(conic_a, delta_x);
  const float conic_c_delta_y = __fmul_rn(conic_c, delta_y);
  const float conic_b_delta_y = __fmul_rn(conic_b, delta_y);
  const float axial = __fmaf_rn(
      delta_y,
      conic_c_delta_y,
      __fmul_rn(delta_x, conic_a_delta_x));
  const float cross = __fmul_rn(delta_x, conic_b_delta_y);
  const float negative_sigma = __fmaf_rn(axial, -0.5f, -cross);
  const float sigma = -negative_sigma;
  const float alpha = fminf(
      kMaximumAlpha,
      opacity * expf(negative_sigma));
  return {
      alpha,
      delta_x,
      delta_y,
      sigma,
      negative_sigma <= 0.0f && alpha >= alpha_cutoff,
  };
}

__global__ void forward_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const float *__restrict__ colors,
    const float *__restrict__ depths,
    const bool *__restrict__ environment_mask,
    const bool *__restrict__ pixel_mask,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    float *__restrict__ rgb,
    float *__restrict__ all_transmittance,
    float *__restrict__ raw_depth,
    float *__restrict__ base_transmittance,
    int *__restrict__ last_all,
    int *__restrict__ last_base) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileWidth + local_x;
  const int pixel_x = blockIdx.x * kTileWidth + local_x;
  const int pixel_y = blockIdx.y * kTileHeight + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
      ? offsets[tile_index + 1]
      : flatten_count;

  __shared__ float shared_means_x[kThreadsPerTile];
  __shared__ float shared_means_y[kThreadsPerTile];
  __shared__ float shared_conic_a[kThreadsPerTile];
  __shared__ float shared_conic_b[kThreadsPerTile];
  __shared__ float shared_conic_c[kThreadsPerTile];
  __shared__ float shared_opacities[kThreadsPerTile];
  __shared__ float shared_colors_r[kThreadsPerTile];
  __shared__ float shared_colors_g[kThreadsPerTile];
  __shared__ float shared_colors_b[kThreadsPerTile];
  __shared__ float shared_depths[kThreadsPerTile];
  __shared__ bool shared_environment[kThreadsPerTile];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const bool enabled = inside &&
      (pixel_mask == nullptr || pixel_mask[pixel_index]);
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  float current_rgb_r = 0.0f;
  float current_rgb_g = 0.0f;
  float current_rgb_b = 0.0f;
  float current_all_transmittance = 1.0f;
  float current_raw_depth = 0.0f;
  float current_base_transmittance = 1.0f;
  int current_last_all = -1;
  int current_last_base = -1;
  bool base_open = enabled;

  for (int chunk_start = list_start; chunk_start < list_end;
       chunk_start += kThreadsPerTile) {
    const int source_index = chunk_start + thread_index;
    if (source_index < list_end) {
      const int gaussian_id = flatten_ids[source_index];
      shared_means_x[thread_index] = means2d[2 * gaussian_id];
      shared_means_y[thread_index] = means2d[2 * gaussian_id + 1];
      shared_conic_a[thread_index] = conics[3 * gaussian_id];
      shared_conic_b[thread_index] = conics[3 * gaussian_id + 1];
      shared_conic_c[thread_index] = conics[3 * gaussian_id + 2];
      shared_opacities[thread_index] = opacities[gaussian_id];
      shared_colors_r[thread_index] = colors[3 * gaussian_id];
      shared_colors_g[thread_index] = colors[3 * gaussian_id + 1];
      shared_colors_b[thread_index] = colors[3 * gaussian_id + 2];
      shared_depths[thread_index] = depths[gaussian_id];
      shared_environment[thread_index] = environment_mask[gaussian_id];
    }
    __syncthreads();

    const int chunk_count = min(kThreadsPerTile, list_end - chunk_start);
    for (int local = 0; local < chunk_count; ++local) {
      const AlphaSample sample = sample_alpha(
          shared_means_x[local], shared_means_y[local],
          shared_conic_a[local], shared_conic_b[local],
          shared_conic_c[local], shared_opacities[local], sample_x, sample_y,
          enabled, alpha_cutoff);
      if (base_open && sample.active) {
        const float all_weight = __fmul_rn(
            current_all_transmittance,
            sample.alpha);
        const float red_contribution = __fmul_rn(
            all_weight,
            shared_colors_r[local]);
        const float green_contribution = __fmul_rn(
            all_weight,
            shared_colors_g[local]);
        const float blue_contribution = __fmul_rn(
            all_weight,
            shared_colors_b[local]);
        current_rgb_r = __fadd_rn(current_rgb_r, red_contribution);
        current_rgb_g = __fadd_rn(current_rgb_g, green_contribution);
        current_rgb_b = __fadd_rn(current_rgb_b, blue_contribution);
        current_all_transmittance *= 1.0f - sample.alpha;
        current_last_all = chunk_start + local;
        if (!shared_environment[local]) {
          const float base_weight = __fmul_rn(
              current_base_transmittance,
              sample.alpha);
          const float depth_contribution = __fmul_rn(
              base_weight,
              shared_depths[local]);
          current_raw_depth = __fadd_rn(
              current_raw_depth,
              depth_contribution);
          current_base_transmittance *= 1.0f - sample.alpha;
          current_last_base = chunk_start + local;
          if (current_base_transmittance < transmittance_cutoff) {
            base_open = false;
          }
        }
      }
    }
    __syncthreads();
  }

  if (inside) {
    rgb[3 * pixel_index] = current_rgb_r;
    rgb[3 * pixel_index + 1] = current_rgb_g;
    rgb[3 * pixel_index + 2] = current_rgb_b;
    all_transmittance[pixel_index] = current_all_transmittance;
    raw_depth[pixel_index] = current_raw_depth;
    base_transmittance[pixel_index] = current_base_transmittance;
    last_all[pixel_index] = current_last_all;
    last_base[pixel_index] = current_last_base;
  }
}

__global__ void backward_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const float *__restrict__ colors,
    const float *__restrict__ depths,
    const bool *__restrict__ environment_mask,
    const bool *__restrict__ pixel_mask,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const float *__restrict__ all_transmittance,
    const float *__restrict__ base_transmittance,
    const int *__restrict__ last_all,
    const int *__restrict__ last_base,
    const float *__restrict__ rgb_gradient,
    const float *__restrict__ raw_depth_gradient,
    const float *__restrict__ base_transmittance_gradient,
    const float *__restrict__ all_transmittance_gradient,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    float *__restrict__ means2d_gradient,
    float *__restrict__ conics_gradient,
    float *__restrict__ opacities_gradient,
    float *__restrict__ colors_gradient,
    float *__restrict__ depths_gradient) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileWidth + local_x;
  const int lane = thread_index & 31;
  const int warp = thread_index >> 5;
  const int pixel_x = blockIdx.x * kTileWidth + local_x;
  const int pixel_y = blockIdx.y * kTileHeight + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
      ? offsets[tile_index + 1]
      : flatten_count;

  __shared__ int shared_ids[kThreadsPerTile];
  __shared__ float shared_means_x[kThreadsPerTile];
  __shared__ float shared_means_y[kThreadsPerTile];
  __shared__ float shared_conic_a[kThreadsPerTile];
  __shared__ float shared_conic_b[kThreadsPerTile];
  __shared__ float shared_conic_c[kThreadsPerTile];
  __shared__ float shared_opacities[kThreadsPerTile];
  __shared__ float shared_colors_r[kThreadsPerTile];
  __shared__ float shared_colors_g[kThreadsPerTile];
  __shared__ float shared_colors_b[kThreadsPerTile];
  __shared__ float shared_depths[kThreadsPerTile];
  __shared__ bool shared_environment[kThreadsPerTile];
  __shared__ float partials[kWarpCount][kGradientCount];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const bool enabled = inside &&
      (pixel_mask == nullptr || pixel_mask[pixel_index]);
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  const float rgb_gradient_r = enabled ? rgb_gradient[3 * pixel_index] : 0.0f;
  const float rgb_gradient_g =
      enabled ? rgb_gradient[3 * pixel_index + 1] : 0.0f;
  const float rgb_gradient_b =
      enabled ? rgb_gradient[3 * pixel_index + 2] : 0.0f;
  const float raw_depth_adjoint =
      enabled ? raw_depth_gradient[pixel_index] : 0.0f;
  const float base_transmittance_adjoint =
      enabled ? base_transmittance_gradient[pixel_index] : 0.0f;
  const float all_transmittance_adjoint =
      enabled ? all_transmittance_gradient[pixel_index] : 0.0f;
  const float final_all_transmittance =
      inside ? all_transmittance[pixel_index] : 1.0f;
  const float final_base_transmittance =
      inside ? base_transmittance[pixel_index] : 1.0f;
  const int final_last_all = inside ? last_all[pixel_index] : -1;
  const int final_last_base = inside ? last_base[pixel_index] : -1;
  float current_all_transmittance = final_all_transmittance;
  float current_base_transmittance = final_base_transmittance;
  float rgb_buffer_r = 0.0f;
  float rgb_buffer_g = 0.0f;
  float rgb_buffer_b = 0.0f;
  float depth_buffer = 0.0f;

  for (int chunk_end = list_end; chunk_end > list_start;
       chunk_end -= kThreadsPerTile) {
    const int chunk_start = max(list_start, chunk_end - kThreadsPerTile);
    const int source_index = chunk_end - 1 - thread_index;
    if (source_index >= chunk_start) {
      const int gaussian_id = flatten_ids[source_index];
      shared_ids[thread_index] = gaussian_id;
      shared_means_x[thread_index] = means2d[2 * gaussian_id];
      shared_means_y[thread_index] = means2d[2 * gaussian_id + 1];
      shared_conic_a[thread_index] = conics[3 * gaussian_id];
      shared_conic_b[thread_index] = conics[3 * gaussian_id + 1];
      shared_conic_c[thread_index] = conics[3 * gaussian_id + 2];
      shared_opacities[thread_index] = opacities[gaussian_id];
      shared_colors_r[thread_index] = colors[3 * gaussian_id];
      shared_colors_g[thread_index] = colors[3 * gaussian_id + 1];
      shared_colors_b[thread_index] = colors[3 * gaussian_id + 2];
      shared_depths[thread_index] = depths[gaussian_id];
      shared_environment[thread_index] = environment_mask[gaussian_id];
    }
    __syncthreads();

    const int chunk_count = chunk_end - chunk_start;
    for (int local = 0; local < chunk_count; ++local) {
      const int list_index = chunk_end - 1 - local;
      const bool all_enabled = enabled && list_index <= final_last_all;
      const bool base_enabled = enabled && !shared_environment[local] &&
          list_index <= final_last_base;
      const AlphaSample all_sample = sample_alpha(
          shared_means_x[local], shared_means_y[local],
          shared_conic_a[local], shared_conic_b[local],
          shared_conic_c[local], shared_opacities[local], sample_x, sample_y,
          all_enabled, alpha_cutoff);
      const AlphaSample base_sample = sample_alpha(
          shared_means_x[local], shared_means_y[local],
          shared_conic_a[local], shared_conic_b[local],
          shared_conic_c[local], shared_opacities[local], sample_x, sample_y,
          base_enabled, alpha_cutoff);
      float local_gradient[kGradientCount] = {
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      };
      float alpha_gradient = 0.0f;
      bool parameter_active = false;
      AlphaSample parameter_sample = all_sample;

      if (all_sample.active) {
        const float reciprocal = 1.0f / (1.0f - all_sample.alpha);
        current_all_transmittance *= reciprocal;
        const float contribution =
            current_all_transmittance * all_sample.alpha;
        local_gradient[6] = contribution * rgb_gradient_r;
        local_gradient[7] = contribution * rgb_gradient_g;
        local_gradient[8] = contribution * rgb_gradient_b;
        alpha_gradient +=
            (shared_colors_r[local] * current_all_transmittance -
             rgb_buffer_r * reciprocal) * rgb_gradient_r +
            (shared_colors_g[local] * current_all_transmittance -
             rgb_buffer_g * reciprocal) * rgb_gradient_g +
            (shared_colors_b[local] * current_all_transmittance -
             rgb_buffer_b * reciprocal) * rgb_gradient_b;
        alpha_gradient -= final_all_transmittance * reciprocal *
            all_transmittance_adjoint;
        rgb_buffer_r += contribution * shared_colors_r[local];
        rgb_buffer_g += contribution * shared_colors_g[local];
        rgb_buffer_b += contribution * shared_colors_b[local];
        parameter_active = true;
      }
      if (base_sample.active) {
        const float reciprocal = 1.0f / (1.0f - base_sample.alpha);
        current_base_transmittance *= reciprocal;
        const float contribution =
            current_base_transmittance * base_sample.alpha;
        local_gradient[9] = contribution * raw_depth_adjoint;
        alpha_gradient +=
            (shared_depths[local] * current_base_transmittance -
             depth_buffer * reciprocal) * raw_depth_adjoint;
        alpha_gradient -= final_base_transmittance * reciprocal *
            base_transmittance_adjoint;
        depth_buffer += contribution * shared_depths[local];
        parameter_active = true;
        parameter_sample = base_sample;
      }
      if (parameter_active && parameter_sample.alpha < kMaximumAlpha) {
        const float sigma_gradient =
            -parameter_sample.alpha * alpha_gradient;
        local_gradient[0] = sigma_gradient * (
            shared_conic_a[local] * parameter_sample.delta_x +
            shared_conic_b[local] * parameter_sample.delta_y);
        local_gradient[1] = sigma_gradient * (
            shared_conic_b[local] * parameter_sample.delta_x +
            shared_conic_c[local] * parameter_sample.delta_y);
        local_gradient[2] = 0.5f * sigma_gradient *
            parameter_sample.delta_x * parameter_sample.delta_x;
        local_gradient[3] = sigma_gradient * parameter_sample.delta_x *
            parameter_sample.delta_y;
        local_gradient[4] = 0.5f * sigma_gradient *
            parameter_sample.delta_y * parameter_sample.delta_y;
        local_gradient[5] = __expf(-parameter_sample.sigma) *
            alpha_gradient;
      }

      for (int offset = 16; offset > 0; offset /= 2) {
        for (int component = 0; component < kGradientCount; ++component) {
          local_gradient[component] += __shfl_down_sync(
              0xffffffff, local_gradient[component], offset);
        }
      }
      if (lane == 0) {
        for (int component = 0; component < kGradientCount; ++component) {
          partials[warp][component] = local_gradient[component];
        }
      }
      __syncthreads();
      if (thread_index == 0) {
        const int gaussian_id = shared_ids[local];
        for (int component = 0; component < kGradientCount; ++component) {
          const float value = partials[0][component] +
              partials[1][component] + partials[2][component] +
              partials[3][component];
          if (component < 2) {
            atomicAdd(means2d_gradient + 2 * gaussian_id + component, value);
          } else if (component < 5) {
            atomicAdd(conics_gradient + 3 * gaussian_id + component - 2,
                      value);
          } else if (component == 5) {
            atomicAdd(opacities_gradient + gaussian_id, value);
          } else if (component < 9) {
            atomicAdd(colors_gradient + 3 * gaussian_id + component - 6,
                      value);
          } else {
            atomicAdd(depths_gradient + gaussian_id, value);
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

__global__ void masked_contribution_kernel(
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
    float *__restrict__ contribution) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileWidth + local_x;
  const int lane = thread_index & 31;
  const int warp = thread_index >> 5;
  const int pixel_x = blockIdx.x * kTileWidth + local_x;
  const int pixel_y = blockIdx.y * kTileHeight + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
      ? offsets[tile_index + 1]
      : flatten_count;

  __shared__ int shared_ids[kThreadsPerTile];
  __shared__ float shared_means_x[kThreadsPerTile];
  __shared__ float shared_means_y[kThreadsPerTile];
  __shared__ float shared_conic_a[kThreadsPerTile];
  __shared__ float shared_conic_b[kThreadsPerTile];
  __shared__ float shared_conic_c[kThreadsPerTile];
  __shared__ float shared_opacities[kThreadsPerTile];
  __shared__ float partials[kWarpCount];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const bool enabled = inside &&
      (pixel_mask == nullptr || pixel_mask[pixel_index]);
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  float current_transmittance = 1.0f;
  bool open = enabled;

  for (int chunk_start = list_start; chunk_start < list_end;
       chunk_start += kThreadsPerTile) {
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

    const int chunk_count = min(kThreadsPerTile, list_end - chunk_start);
    for (int local = 0; local < chunk_count; ++local) {
      const AlphaSample sample = sample_alpha(
          shared_means_x[local], shared_means_y[local],
          shared_conic_a[local], shared_conic_b[local],
          shared_conic_c[local], shared_opacities[local], sample_x, sample_y,
          enabled, alpha_cutoff);
      float local_contribution = 0.0f;
      if (open && sample.active) {
        local_contribution = current_transmittance * sample.alpha;
        current_transmittance *= 1.0f - sample.alpha;
        if (current_transmittance < transmittance_cutoff) {
          open = false;
        }
      }
      for (int offset = 16; offset > 0; offset /= 2) {
        local_contribution += __shfl_down_sync(
            0xffffffff, local_contribution, offset);
      }
      if (lane == 0) {
        partials[warp] = local_contribution;
      }
      __syncthreads();
      if (thread_index == 0) {
        atomicAdd(contribution + shared_ids[local],
                  partials[0] + partials[1] + partials[2] + partials[3]);
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

__global__ void masked_forward_contribution_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const bool *__restrict__ environment_mask,
    const bool *__restrict__ pixel_mask,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    float *__restrict__ contribution) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileWidth + local_x;
  const int lane = thread_index & 31;
  const int warp = thread_index >> 5;
  const int pixel_x = blockIdx.x * kTileWidth + local_x;
  const int pixel_y = blockIdx.y * kTileHeight + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
      ? offsets[tile_index + 1]
      : flatten_count;

  __shared__ int shared_ids[kThreadsPerTile];
  __shared__ float shared_means_x[kThreadsPerTile];
  __shared__ float shared_means_y[kThreadsPerTile];
  __shared__ float shared_conic_a[kThreadsPerTile];
  __shared__ float shared_conic_b[kThreadsPerTile];
  __shared__ float shared_conic_c[kThreadsPerTile];
  __shared__ float shared_opacities[kThreadsPerTile];
  __shared__ bool shared_environment[kThreadsPerTile];
  __shared__ float partials[kWarpCount];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const bool enabled = inside &&
      (pixel_mask == nullptr || pixel_mask[pixel_index]);
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  float current_all_transmittance = 1.0f;
  float current_base_transmittance = 1.0f;
  bool base_open = enabled;

  for (int chunk_start = list_start; chunk_start < list_end;
       chunk_start += kThreadsPerTile) {
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
      shared_environment[thread_index] = environment_mask[gaussian_id];
    }
    __syncthreads();

    const int chunk_count = min(kThreadsPerTile, list_end - chunk_start);
    for (int local = 0; local < chunk_count; ++local) {
      const AlphaSample sample = sample_alpha(
          shared_means_x[local], shared_means_y[local],
          shared_conic_a[local], shared_conic_b[local],
          shared_conic_c[local], shared_opacities[local], sample_x, sample_y,
          enabled, alpha_cutoff);
      float local_contribution = 0.0f;
      if (base_open && sample.active) {
        local_contribution = current_all_transmittance * sample.alpha;
        current_all_transmittance *= 1.0f - sample.alpha;
        if (!shared_environment[local]) {
          current_base_transmittance *= 1.0f - sample.alpha;
          if (current_base_transmittance < transmittance_cutoff) {
            base_open = false;
          }
        }
      }
      for (int offset = 16; offset > 0; offset /= 2) {
        local_contribution += __shfl_down_sync(
            0xffffffff, local_contribution, offset);
      }
      if (lane == 0) {
        partials[warp] = local_contribution;
      }
      __syncthreads();
      if (thread_index == 0) {
        atomicAdd(contribution + shared_ids[local],
                  partials[0] + partials[1] + partials[2] + partials[3]);
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

__global__ void masked_visibility_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const bool *__restrict__ environment_mask,
    const bool *__restrict__ pixel_mask,
    const int *__restrict__ offsets,
    const int *__restrict__ flatten_ids,
    const int flatten_count,
    const int width,
    const int height,
    const int tiles_width,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    int *__restrict__ visibility) {
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int thread_index = local_y * kTileWidth + local_x;
  const int pixel_x = blockIdx.x * kTileWidth + local_x;
  const int pixel_y = blockIdx.y * kTileHeight + local_y;
  const int tile_index = blockIdx.y * tiles_width + blockIdx.x;
  const int tile_count = gridDim.x * gridDim.y;
  const int list_start = offsets[tile_index];
  const int list_end = tile_index + 1 < tile_count
      ? offsets[tile_index + 1]
      : flatten_count;

  __shared__ int shared_ids[kThreadsPerTile];
  __shared__ float shared_means_x[kThreadsPerTile];
  __shared__ float shared_means_y[kThreadsPerTile];
  __shared__ float shared_conic_a[kThreadsPerTile];
  __shared__ float shared_conic_b[kThreadsPerTile];
  __shared__ float shared_conic_c[kThreadsPerTile];
  __shared__ float shared_opacities[kThreadsPerTile];
  __shared__ bool shared_environment[kThreadsPerTile];

  const bool inside = pixel_x < width && pixel_y < height;
  const int pixel_index = pixel_y * width + pixel_x;
  const bool enabled = inside &&
      (pixel_mask == nullptr || pixel_mask[pixel_index]);
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  float current_base_transmittance = 1.0f;
  bool base_open = enabled;

  for (int chunk_start = list_start; chunk_start < list_end;
       chunk_start += kThreadsPerTile) {
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
      shared_environment[thread_index] = environment_mask[gaussian_id];
    }
    __syncthreads();

    const int chunk_count = min(kThreadsPerTile, list_end - chunk_start);
    for (int local = 0; local < chunk_count; ++local) {
      const AlphaSample sample = sample_alpha(
          shared_means_x[local], shared_means_y[local],
          shared_conic_a[local], shared_conic_b[local],
          shared_conic_c[local], shared_opacities[local], sample_x, sample_y,
          enabled, alpha_cutoff);
      if (base_open && sample.active) {
        atomicExch(visibility + shared_ids[local], 1);
        if (!shared_environment[local]) {
          current_base_transmittance *= 1.0f - sample.alpha;
          if (current_base_transmittance < transmittance_cutoff) {
            base_open = false;
          }
        }
      }
    }
    __syncthreads();
  }
}

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
    const int64_t width,
    const int64_t height,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    at::Tensor &rgb,
    at::Tensor &all_transmittance,
    at::Tensor &raw_depth,
    at::Tensor &base_transmittance,
    at::Tensor &last_all,
    at::Tensor &last_base) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileWidth, kTileHeight);
  const dim3 grid(tiles_width, tiles_height);
  const bool *pixel_mask_ptr = pixel_mask.numel() == 0
      ? nullptr
      : pixel_mask.data_ptr<bool>();
  forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), colors.data_ptr<float>(),
      depths.data_ptr<float>(), environment_mask.data_ptr<bool>(),
      pixel_mask_ptr, offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
      static_cast<int>(flatten_ids.numel()), static_cast<int>(width),
      static_cast<int>(height), tiles_width, alpha_cutoff,
      transmittance_cutoff, rgb.data_ptr<float>(),
      all_transmittance.data_ptr<float>(), raw_depth.data_ptr<float>(),
      base_transmittance.data_ptr<float>(), last_all.data_ptr<int>(),
      last_base.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const int64_t width,
    const int64_t height,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    at::Tensor &means2d_gradient,
    at::Tensor &conics_gradient,
    at::Tensor &opacities_gradient,
    at::Tensor &colors_gradient,
    at::Tensor &depths_gradient) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileWidth, kTileHeight);
  const dim3 grid(tiles_width, tiles_height);
  const bool *pixel_mask_ptr = pixel_mask.numel() == 0
      ? nullptr
      : pixel_mask.data_ptr<bool>();
  backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), colors.data_ptr<float>(),
      depths.data_ptr<float>(), environment_mask.data_ptr<bool>(),
      pixel_mask_ptr, offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
      static_cast<int>(flatten_ids.numel()),
      all_transmittance.data_ptr<float>(), base_transmittance.data_ptr<float>(),
      last_all.data_ptr<int>(), last_base.data_ptr<int>(),
      rgb_gradient.data_ptr<float>(), raw_depth_gradient.data_ptr<float>(),
      base_transmittance_gradient.data_ptr<float>(),
      all_transmittance_gradient.data_ptr<float>(), static_cast<int>(width),
      static_cast<int>(height), tiles_width, alpha_cutoff,
      means2d_gradient.data_ptr<float>(), conics_gradient.data_ptr<float>(),
      opacities_gradient.data_ptr<float>(), colors_gradient.data_ptr<float>(),
      depths_gradient.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void masked_contribution_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    at::Tensor &contribution) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileWidth, kTileHeight);
  const dim3 grid(tiles_width, tiles_height);
  const bool *pixel_mask_ptr = pixel_mask.numel() == 0
      ? nullptr
      : pixel_mask.data_ptr<bool>();
  masked_contribution_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), pixel_mask_ptr, offsets.data_ptr<int>(),
      flatten_ids.data_ptr<int>(), static_cast<int>(flatten_ids.numel()),
      static_cast<int>(width), static_cast<int>(height), tiles_width,
      alpha_cutoff, transmittance_cutoff, contribution.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void masked_forward_contribution_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &environment_mask,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    at::Tensor &contribution) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileWidth, kTileHeight);
  const dim3 grid(tiles_width, tiles_height);
  const bool *pixel_mask_ptr = pixel_mask.numel() == 0
      ? nullptr
      : pixel_mask.data_ptr<bool>();
  masked_forward_contribution_kernel<<<
      grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), environment_mask.data_ptr<bool>(),
      pixel_mask_ptr, offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
      static_cast<int>(flatten_ids.numel()), static_cast<int>(width),
      static_cast<int>(height), tiles_width, alpha_cutoff,
      transmittance_cutoff, contribution.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void masked_visibility_launch(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &environment_mask,
    const at::Tensor &pixel_mask,
    const at::Tensor &offsets,
    const at::Tensor &flatten_ids,
    const int64_t width,
    const int64_t height,
    const float alpha_cutoff,
    const float transmittance_cutoff,
    at::Tensor &visibility) {
  const int tiles_width = static_cast<int>(offsets.size(2));
  const int tiles_height = static_cast<int>(offsets.size(1));
  const dim3 block(kTileWidth, kTileHeight);
  const dim3 grid(tiles_width, tiles_height);
  const bool *pixel_mask_ptr = pixel_mask.numel() == 0
      ? nullptr
      : pixel_mask.data_ptr<bool>();
  masked_visibility_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      means2d.data_ptr<float>(), conics.data_ptr<float>(),
      opacities.data_ptr<float>(), environment_mask.data_ptr<bool>(),
      pixel_mask_ptr, offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
      static_cast<int>(flatten_ids.numel()), static_cast<int>(width),
      static_cast<int>(height), tiles_width, alpha_cutoff,
      transmittance_cutoff, visibility.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rectangular_ewa
