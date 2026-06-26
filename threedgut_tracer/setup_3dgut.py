# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import torch

from threedgrut.utils import jit


# ----------------------------------------------------------------------------
#
def setup_3dgut(conf):
    build_dir = torch.utils.cpp_extension._get_build_directory("lib3dgut_cc", verbose=True)

    include_paths = []
    prefix = os.path.dirname(__file__)
    include_paths.append(os.path.join(prefix, "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies", "fmt", "include"))
    include_paths.append(build_dir)

    # Compiler options.

    def to_cpp_bool(value):
        return "true" if value else "false"

    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha * ut_alpha * (ut_d + ut_kappa))

    radiance_sh_coeffs = (conf.render.particle_radiance_sph_degree + 1) ** 2
    gabor_enabled = bool(conf.model.get("use_gabor_carrier", False))
    siren_enabled = bool(conf.model.get("use_siren_carrier", False))
    if gabor_enabled and siren_enabled:
        raise ValueError(
            "model.use_gabor_carrier and model.use_siren_carrier are "
            "mutually exclusive."
        )
    gabor_num_terms = int(conf.model.get("gabor_num_terms", 3))
    if gabor_enabled and gabor_num_terms != 3:
        raise ValueError(
            "model.gabor_num_terms currently supports exactly 3 terms; "
            f"got {gabor_num_terms}."
        )
    gabor_coeffs = gabor_num_terms + 3 if gabor_enabled else 0
    siren_hidden_dim = int(conf.model.get("siren_hidden_dim", 6))
    if siren_enabled and siren_hidden_dim <= 0:
        raise ValueError(
            "model.siren_hidden_dim must be positive; got "
            f"{siren_hidden_dim}."
        )
    siren_bias_coeffs = (siren_hidden_dim + 2) // 3
    siren_coeffs = (
        (2 * siren_hidden_dim) + siren_bias_coeffs + siren_hidden_dim + 1
        if siren_enabled
        else 0
    )
    radiance_coeffs = radiance_sh_coeffs + gabor_coeffs + siren_coeffs
    per_ray_particle_features = gabor_enabled or siren_enabled or bool(
        conf.render.splat.get("per_ray_particle_features", False)
    )

    defines = [
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={radiance_coeffs}",
        f"-DPARTICLE_RADIANCE_NUM_SH_COEFFS={radiance_sh_coeffs}",
        f"-DPARTICLE_RADIANCE_GABOR_ENABLED={to_cpp_bool(gabor_enabled)}",
        f"-DPARTICLE_RADIANCE_GABOR_NUM_TERMS={gabor_num_terms}",
        f"-DPARTICLE_RADIANCE_GABOR_MAX_FREQUENCY={float(conf.model.get('gabor_max_frequency', 4.0))}",
        f"-DPARTICLE_RADIANCE_SIREN_ENABLED={to_cpp_bool(siren_enabled)}",
        f"-DPARTICLE_RADIANCE_SIREN_HIDDEN_DIM={siren_hidden_dim}",
        f"-DPARTICLE_RADIANCE_SIREN_OMEGA={float(conf.model.get('siren_omega', 30.0))}",
        f"-DPARTICLE_RADIANCE_SIREN_LOCAL_UV_SCALE={float(conf.model.get('siren_local_uv_scale', 3.0))}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={to_cpp_bool(conf.render.enable_normals)}",
        f"-DGAUSSIAN_PARTICLE_SURFEL={to_cpp_bool(conf.render.primitive_type=='trisurfel')}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        f"-DGAUSSIAN_ENABLE_HIT_COUNT={to_cpp_bool(conf.render.enable_hitcounts)}",
        # Specific to the 3DGUT renderer
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={conf.render.splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={conf.render.splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={to_cpp_bool(conf.render.splat.global_z_order)}",
        f"-DGAUSSIAN_PER_RAY_PARTICLE_FEATURES={to_cpp_bool(per_ray_particle_features)}",
        f"-DFINE_GRAINED_LOAD_BALANCING={to_cpp_bool(getattr(conf.render.splat, 'fine_grained_load_balancing', False))}",
        # -- Unscented Transform --
        f"-DGAUSSIAN_UT_ALPHA={ut_alpha}",
        f"-DGAUSSIAN_UT_BETA={ut_beta}",
        f"-DGAUSSIAN_UT_KAPPA={ut_kappa}",
        f"-DGAUSSIAN_UT_DELTA={ut_delta}",
        f"-DGAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR={conf.render.splat.ut_in_image_margin_factor}",
        f"-DGAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID={to_cpp_bool(conf.render.splat.ut_require_all_sigma_points_valid)}",
        # -- Culling --
        f"-DGAUSSIAN_RECT_BOUNDING={to_cpp_bool(conf.render.splat.rect_bounding)}",
        f"-DGAUSSIAN_TIGHT_OPACITY_BOUNDING={to_cpp_bool(conf.render.splat.tight_opacity_bounding)}",
        f"-DGAUSSIAN_TILE_BASED_CULLING={to_cpp_bool(conf.render.splat.tile_based_culling)}",
    ]

    cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        *defines,
    ]

    cuda_cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-use_fast_math",
        "-O3",
        *defines,
    ]

    # List of sources.
    source_files = [
        "src/splatRaster.cpp",
        "src/gutRenderer.cu",
        "src/cudaBuffer.cpp",
        "bindings.cpp",
    ]

    # Compile slang kernels
    slang_build_inc_dir = os.path.join(os.path.dirname(__file__), "include", "3dgut")
    jit.compile_slang_kernel(
        kernel_files=[f"{os.path.join(slang_build_inc_dir, 'threedgut.slang')}"],
        output_file=f"{os.path.join(build_dir, 'threedgutSlang.cuh')}",
        include_paths=[
            os.path.join(os.path.dirname(__file__), "include"),
            os.path.join(os.path.dirname(__file__), "..", "threedgrt_tracer", "include"),
        ],
        defines=defines,
    )

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    return jit.load(
        name="lib3dgut_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
    )
