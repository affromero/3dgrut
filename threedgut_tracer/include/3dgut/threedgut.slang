// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace gaussianParticle
{
    static const int KernelDegree = GAUSSIAN_PARTICLE_KERNEL_DEGREE;
    static const float MinParticleKernelDensity = GAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY;
    static const float MaxParticleAlpha = GAUSSIAN_PARTICLE_MAX_ALPHA;
    static const float MinParticleAlpha = GAUSSIAN_PARTICLE_MIN_ALPHA;
    static const bool EnableNormal = false;
    static const bool Surfel = false;
    static const bool SignedRayDepth = GAUSSIAN_SIGNED_RAY_DEPTH;
};

#include <3dgut/kernels/slang/models/gaussianParticles.slang>

namespace shRadiativeParticle
{
    static const int RadianceMaxNumSphCoefficients = PARTICLE_RADIANCE_NUM_COEFFS;
    static const int RadianceNumSphCoefficients = PARTICLE_RADIANCE_NUM_SH_COEFFS;
    static const bool GaborEnabled = PARTICLE_RADIANCE_GABOR_ENABLED;
    static const int GaborNumTerms = PARTICLE_RADIANCE_GABOR_NUM_TERMS;
    static const float GaborMaxFrequency = PARTICLE_RADIANCE_GABOR_MAX_FREQUENCY;
    static const bool HermiteEnabled = PARTICLE_RADIANCE_HERMITE_ENABLED;
    static const bool SirenEnabled = PARTICLE_RADIANCE_SIREN_ENABLED;
    static const int SirenHiddenDim = PARTICLE_RADIANCE_SIREN_HIDDEN_DIM;
    static const float SirenOmega = PARTICLE_RADIANCE_SIREN_OMEGA;
    static const float SirenLocalUvScale = PARTICLE_RADIANCE_SIREN_LOCAL_UV_SCALE;
    static const int Dim = 3;
};

#include <3dgut/kernels/slang/models/shRadiativeParticles.slang>
