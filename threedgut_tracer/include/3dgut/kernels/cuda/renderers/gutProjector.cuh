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

#pragma once

#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/renderer/renderParameters.h>

template <typename Particles, typename Params, typename UTParams>
struct GUTProjector : Params, UTParams {

    using TFeaturesVec = typename Particles::TFeaturesVec;

    struct BoundingBox2D {
        tcnn::uvec2 min;
        tcnn::uvec2 max;
    };

    static inline __device__ BoundingBox2D computeTileSpaceBBox(const tcnn::uvec2& tileGrid, const tcnn::vec2& position, const tcnn::vec2& extent) {
        return BoundingBox2D{
            {
                min(tileGrid.x, max(0, static_cast<int>(floorf((position.x - 0.5f - extent.x) / threedgut::GUTParameters::Tiling::BlockX)))),
                min(tileGrid.y, max(0, static_cast<int>(floorf((position.y - 0.5f - extent.y) / threedgut::GUTParameters::Tiling::BlockY)))),
            },
            {
                min(tileGrid.x, max(0, static_cast<int>(ceilf((position.x - 0.5f + extent.x) / threedgut::GUTParameters::Tiling::BlockX)))),
                min(tileGrid.y, max(0, static_cast<int>(ceilf((position.y - 0.5f + extent.y) / threedgut::GUTParameters::Tiling::BlockY)))),
            },
        };
    }

    // Equirectangular-only: half-open X tile range [min, max) that is allowed to
    // run negative or past gridX so a footprint straddling the +-180 deg azimuth
    // seam is binned to BOTH panorama edges. Y is NOT computed here (poles do not
    // wrap) -- callers take Y from computeTileSpaceBBox. The span is capped at one
    // full ring (gridX tiles) so a particle can never emit duplicate wrapped keys.
    // Used IDENTICALLY by the count (eval) and emit (expand) loops; both passes
    // pass the same inputs, so they get the same bounds and tile_count == tiles_emitted.
    struct EquirectXTileRange {
        int min;
        int max;
    };

    static inline __device__ EquirectXTileRange computeEquirectXTileRange(const tcnn::uvec2& tileGrid, const tcnn::vec2& position, const tcnn::vec2& extent) {
        constexpr int blockX = static_cast<int>(threedgut::GUTParameters::Tiling::BlockX);
        const int gridX      = static_cast<int>(tileGrid.x);
        int xMin             = static_cast<int>(floorf((position.x - 0.5f - extent.x) / blockX));
        int xMax             = static_cast<int>(ceilf((position.x - 0.5f + extent.x) / blockX));
        if (xMax <= xMin) {
            xMax = xMin + 1;
        }
        // Cap at one full ring so wrapped keys cannot duplicate.
        if ((xMax - xMin) > gridX) {
            xMax = xMin + gridX;
        }
        return EquirectXTileRange{xMin, xMax};
    }

    // Wrap an (possibly negative / >= gridX) tile column into [0, gridX) for key emission.
    static inline __device__ uint32_t wrapTileX(int x, int gridX) {
        const int m = x % gridX;
        return static_cast<uint32_t>(m < 0 ? m + gridX : m);
    }

    static inline __device__ uint64_t concatTileDepthKeys(uint32_t tileKey, uint32_t depthKey) {
        return (static_cast<uint64_t>(tileKey) << 32) | depthKey;
    }

    static inline __device__ float tileMinParticlePowerResponse(const tcnn::vec2& tileCoords,
                                                                const tcnn::vec4& conicOpacity,
                                                                const tcnn::vec2& meanPosition) {

        const tcnn::vec2 tileSize = tcnn::vec2(threedgut::GUTParameters::Tiling::BlockX, threedgut::GUTParameters::Tiling::BlockY);
        const tcnn::vec2 tileMin  = tileSize * tileCoords;
        const tcnn::vec2 tileMax  = tileSize + tileMin;

        const tcnn::vec2 minOffset  = tileMin - meanPosition;
        const tcnn::vec2 leftAbove  = tcnn::vec2(minOffset.x > 0.0f, minOffset.y > 0.0f);
        const tcnn::vec2 notInRange = tcnn::vec2(leftAbove.x + (meanPosition.x > tileMax.x),
                                                 leftAbove.y + (meanPosition.y > tileMax.y));

        if ((notInRange.x + notInRange.y) > 0.0f) {
            const tcnn::vec2 p    = tcnn::mix(tileMax, tileMin, leftAbove);
            const tcnn::vec2 dxy  = tcnn::copysign(tileSize, minOffset);
            const tcnn::vec2 diff = meanPosition - p;
            const tcnn::vec2 rcp  = tcnn::vec2(__frcp_rn(tileSize.x * tileSize.x * conicOpacity.x),
                                               __frcp_rn(tileSize.y * tileSize.y * conicOpacity.z));

            const float tx = notInRange.y * __saturatef((dxy.x * conicOpacity.x * diff.x + dxy.x * conicOpacity.y * diff.y) * rcp.x);
            const float ty = notInRange.x * __saturatef((dxy.y * conicOpacity.y * diff.x + dxy.y * conicOpacity.z * diff.y) * rcp.y);

            const tcnn::vec2 minPosDiff = meanPosition - tcnn::vec2(p.x + tx * dxy.x, p.y + ty * dxy.y);

            return 0.5f * (conicOpacity.x * minPosDiff.x * minPosDiff.x + conicOpacity.z * minPosDiff.y * minPosDiff.y) + conicOpacity.y * minPosDiff.x * minPosDiff.y;
        }
        // mean position is within the tile
        return 0.f;
    }

    /// Convert a projected particle to its conic/opacity representation
    static inline __device__ bool computeProjectedExtentConicOpacity(tcnn::vec3 covariance,
                                                                     float opacity,
                                                                     tcnn::vec2& extent,
                                                                     tcnn::vec4& conicOpacity,
                                                                     float& maxConicOpacityPower) {

        const tcnn::vec3 dilatedCovariance = tcnn::vec3{covariance.x + Params::CovarianceDilation, covariance.y, covariance.z + Params::CovarianceDilation};
        const float dilatedCovDet          = dilatedCovariance.x * dilatedCovariance.z - dilatedCovariance.y * dilatedCovariance.y;
        if (dilatedCovDet == 0.0f) {
            return false;
        }
        conicOpacity.slice<0, 3>() = tcnn::vec3{dilatedCovariance.z, -dilatedCovariance.y, dilatedCovariance.x} / dilatedCovDet;

        // see Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting" https://github.com/autonomousvision/mip-splatting
        if constexpr (TGUTProjectorParams::MipSplattingScaling) {
            const float covDet            = covariance.x * covariance.z - covariance.y * covariance.y;
            const float convolutionFactor = sqrtf(fmaxf(0.000025f, covDet / dilatedCovDet));
            conicOpacity.w                = opacity * convolutionFactor;
        } else {
            conicOpacity.w = opacity;
        }

        if (conicOpacity.w < Params::AlphaThreshold) {
            return false;
        }

        maxConicOpacityPower     = logf(conicOpacity.w / Params::AlphaThreshold);
        const float extentFactor = Params::TightOpacityBounding ? fminf(3.33f, sqrtf(2.0f * maxConicOpacityPower)) : 3.33f;
        const float minLambda    = 0.01f;
        const float mid          = 0.5f * (dilatedCovariance.x + dilatedCovariance.z);
        const float lambda       = mid + sqrtf(fmaxf(minLambda, mid * mid - dilatedCovDet));
        const float radius       = extentFactor * sqrtf(lambda);
        extent                   = Params::RectBounding ? min(extentFactor * sqrt(tcnn::vec2{dilatedCovariance.x, dilatedCovariance.z}), tcnn::vec2{radius}) : tcnn::vec2{radius};

        return radius > 0.f;
    }

    static inline __device__ bool unscentedParticleProjection(
        const tcnn::ivec2& resolution,
        const threedgut::TSensorModel& sensorModel,
        const tcnn::vec3& sensorWorldPosition,
        const tcnn::mat4x3& sensorMatrix,
        const threedgut::TSensorState& sensorShutterState,
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters,
        tcnn::vec3& particleSensorRay,
        float& particleProjOpacity,
        tcnn::vec2& particleProjCenter,
        tcnn::vec3& particleProjCovariance) {

        particleProjOpacity = particles.opacity(particleParameters);
        if (particleProjOpacity < Params::AlphaThreshold) {
            return false;
        }

        const tcnn::vec3& particleMean = particles.position(particleParameters);

        // Camera-to-particle vector (world frame); its norm is the equirect
        // projectPoint() radial distance and is reused below, so compute it
        // before the visibility cull.
        particleSensorRay = particleMean - sensorWorldPosition;

        const bool wideFovModel = (
            sensorModel.modelType == threedgut::TSensorModel::EquirectangularModel ||
            sensorModel.modelType == threedgut::TSensorModel::OpenCVFisheyeModel);
        if (wideFovModel) {
            // Wide-FOV cameras rely on projectPoint() for angular validity.
            // A forward-hemisphere depth cull would discard valid fisheye rim
            // samples beyond 90 deg before the KB4 maxAngle check runs.
            if (length(particleSensorRay) < 1e-6f) {
                return false;
            }
        } else if ((particleMean.x * sensorMatrix[0][2] + particleMean.y * sensorMatrix[1][2] +
                    particleMean.z * sensorMatrix[2][2] + sensorMatrix[3][2]) < Params::ParticleMinSensorZ) {
            // Perspective / rational / ftheta: forward-hemisphere cull.
            return false;
        }

        const tcnn::vec3& particleScale   = particles.scale(particleParameters);
        const tcnn::mat3 particleRotation = particles.rotation(particleParameters);

        int numValidPoints = 0;
        tcnn::vec2 projectedSigmaPoints[2 * UTParams::D + 1];

        constexpr float Lambda = UTParams::Alpha * UTParams::Alpha * (UTParams::D + UTParams::Kappa) - UTParams::D;

        if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                particleMean,
                resolution,
                sensorModel,
                sensorShutterState,
                UTParams::ImageMarginFactor,
                projectedSigmaPoints[0])) {
            numValidPoints++;
        }

#pragma unroll
        for (int i = 0; i < UTParams::D; ++i) {
            const tcnn::vec3 delta = UTParams::Delta * particleScale[i] * particleRotation[i]; ///< CHECK : column or row ?

            if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean + delta,
                    resolution,
                    sensorModel,
                    sensorShutterState,
                    UTParams::ImageMarginFactor,
                    projectedSigmaPoints[i + 1])) {
                numValidPoints++;
            }

            if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean - delta,
                    resolution,
                    sensorModel,
                    sensorShutterState,
                    UTParams::ImageMarginFactor,
                    projectedSigmaPoints[i + 1 + UTParams::D])) {
                numValidPoints++;
            }
        }

        if (numValidPoints < (2 * UTParams::D + 1)) {
            return false;
        }

        particleProjCenter = projectedSigmaPoints[0] * (Lambda / (UTParams::D + Lambda));
        constexpr float weightI = 1.f / (2.f * (UTParams::D + Lambda));
#pragma unroll
        for (int i = 1; i < 2 * UTParams::D + 1; ++i) {
            particleProjCenter += weightI * projectedSigmaPoints[i];
        }

        // Equirectangular seam handling: sigma points straddling the +-180 deg
        // azimuth seam wrap (u ~ 0 vs ~ width), which corrupts the averaged
        // center and covariance below. Unwrap each sigma point's u onto the
        // branch nearest the first point and recompute the mean; the center is
        // wrapped back into [0, width) after the covariance is built.
        const bool equirectModel =
            (sensorModel.modelType == threedgut::TSensorModel::EquirectangularModel);
        if (equirectModel) {
            const float width = static_cast<float>(resolution.x);
            const float refX  = projectedSigmaPoints[0].x;
#pragma unroll
            for (int i = 1; i < 2 * UTParams::D + 1; ++i) {
                projectedSigmaPoints[i].x -= width * roundf((projectedSigmaPoints[i].x - refX) / width);
            }
            tcnn::vec2 unwrappedCenter = projectedSigmaPoints[0] * (Lambda / (UTParams::D + Lambda));
#pragma unroll
            for (int i = 1; i < 2 * UTParams::D + 1; ++i) {
                unwrappedCenter += weightI * projectedSigmaPoints[i];
            }
            particleProjCenter = unwrappedCenter;
        }

        {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[0] - particleProjCenter;
            constexpr float weight0        = Lambda / (UTParams::D + Lambda) + (1.f - UTParams::Alpha * UTParams::Alpha + UTParams::Beta);
            particleProjCovariance         = weight0 * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                                  centeredPoint.x * centeredPoint.y,
                                                                  centeredPoint.y * centeredPoint.y);
        }
#pragma unroll
        for (int i = 0; i < 2 * UTParams::D; ++i) {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[i + 1] - particleProjCenter;
            particleProjCovariance += weightI * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                           centeredPoint.x * centeredPoint.y,
                                                           centeredPoint.y * centeredPoint.y);
        }

        if (equirectModel) {
            const float width    = static_cast<float>(resolution.x);
            particleProjCenter.x = particleProjCenter.x - width * floorf(particleProjCenter.x / width);
        }

        return true;
    }

    static inline __device__ void eval(tcnn::uvec2 tileGrid,
                                       uint32_t numParticles,
                                       tcnn::ivec2 resolution,
                                       threedgut::TSensorModel sensorModel,
                                       tcnn::vec3 sensorWorldPosition,
                                       tcnn::mat4x3 sensorViewMatrix,
                                       threedgut::TSensorState sensorShutterState,
                                       uint32_t* __restrict__ particlesTilesCountPtr,
                                       tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                       float* __restrict__ particlesGlobalDepthPtr,
                                       float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       int* __restrict__ particlesVisibilityCudaPtr,
                                       threedgut::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx >= numParticles) {
            return;
        }

        Particles particles;
        particles.initializeDensity(parameters);
        const auto particleParameters = particles.fetchDensityParameters(particleIdx);

        tcnn::vec2 particleProjCenter;
        float particleProjOpacity;
        tcnn::vec3 particleSensorRay;
        tcnn::vec3 particleProjCovariance;
        bool validProjection = false;
        {
            validProjection = unscentedParticleProjection(
                resolution,
                sensorModel,
                sensorWorldPosition,
                sensorViewMatrix,
                // FIXME : work directly in sensor space to avoid all intermediate transforms
                sensorShutterState,
                particles,
                particleParameters,
                particleSensorRay,
                particleProjOpacity,
                particleProjCenter,
                particleProjCovariance);
        }

        tcnn::vec2 particleProjExtent;
        tcnn::vec4 particleProjConicOpacity;
        float particleMaxConicOpacityPower;
        bool validConicEstimation = false;
        {
            validConicEstimation = computeProjectedExtentConicOpacity(particleProjCovariance,
                                                                      particleProjOpacity,
                                                                      particleProjExtent,
                                                                      particleProjConicOpacity,
                                                                      particleMaxConicOpacityPower);
        }

        validProjection = validProjection && validConicEstimation;

        // Visibility feeds SelectiveAdam and must match the exact render gate
        // below: a particle whose unscented projection failed is NOT visible,
        // even if its conic estimation succeeded.
        particlesVisibilityCudaPtr[particleIdx] = validProjection ? 1 : 0;

        uint32_t numValidTiles = 0;
        if (validProjection) {
            const BoundingBox2D tileBBox = computeTileSpaceBBox(tileGrid, particleProjCenter, particleProjExtent);
            if (sensorModel.modelType == threedgut::TSensorModel::EquirectangularModel) {
                // Azimuthally-wrapping X range; Y still clamps (poles do not wrap).
                // Iterate the unwrapped X span and power-test at the true (unwrapped)
                // footprint column -- identical to the emit loop in expand().
                const EquirectXTileRange xRange = computeEquirectXTileRange(tileGrid, particleProjCenter, particleProjExtent);
                if constexpr (Params::TileCulling) {
                    for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                        for (int x = xRange.min; x < xRange.max; ++x) {
                            if (tileMinParticlePowerResponse(tcnn::vec2(x, y), particleProjConicOpacity, particleProjCenter) < particleMaxConicOpacityPower) {
                                numValidTiles++;
                            }
                        }
                    }
                } else {
                    numValidTiles = static_cast<uint32_t>((xRange.max - xRange.min) * (static_cast<int>(tileBBox.max.y) - static_cast<int>(tileBBox.min.y)));
                }
            } else if constexpr (Params::TileCulling) {
                for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                    for (int x = tileBBox.min.x; x < tileBBox.max.x; ++x) {
                        if (tileMinParticlePowerResponse(tcnn::vec2(x, y), particleProjConicOpacity, particleProjCenter) < particleMaxConicOpacityPower) {
                            numValidTiles++;
                        }
                    }
                }
            } else {
                numValidTiles = (tileBBox.max.x - tileBBox.min.x) * (tileBBox.max.y - tileBBox.min.y);
            }
        }

        particlesTilesCountPtr[particleIdx] = numValidTiles;
        if (numValidTiles == 0) {
            particlesProjectedPositionPtr[particleIdx]     = tcnn::vec2::zero();
            particlesProjectedConicOpacityPtr[particleIdx] = tcnn::vec4::zero();
            particlesProjectedExtentPtr[particleIdx]       = tcnn::vec2::zero();
            particlesGlobalDepthPtr[particleIdx]           = 0.f;
            return;
        }

        const float particleSensorDistance = length(particleSensorRay);

        if constexpr (!Params::PerRayParticleFeatures) {
            particles.initializeFeatures(parameters);
            reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesPtr)[particleIdx] =
                particles.template featuresCustomFromBuffer<false>(particleIdx, particleSensorRay / particleSensorDistance);
        }

        particlesProjectedPositionPtr[particleIdx]     = particleProjCenter;
        particlesProjectedConicOpacityPtr[particleIdx] = particleProjConicOpacity;
        particlesProjectedExtentPtr[particleIdx]       = particleProjExtent;
        if constexpr (Params::GlobalZOrder) {
            if (sensorModel.modelType == threedgut::TSensorModel::EquirectangularModel) {
                // Full-sphere camera: signed sensor-frame Z would corrupt the
                // ascending unsigned-float-bits radix sort for the rear
                // hemisphere (negative Z sorts as larger and reversed). Radial
                // distance is non-negative and is the correct front-to-back
                // ordering when there is no single forward axis.
                particlesGlobalDepthPtr[particleIdx] = particleSensorDistance;
            } else {
                const tcnn::vec3& particleMean       = particles.position(particleParameters);
                particlesGlobalDepthPtr[particleIdx] = (particleMean.x * sensorViewMatrix[0][2] + particleMean.y * sensorViewMatrix[1][2] +
                                                        particleMean.z * sensorViewMatrix[2][2] + sensorViewMatrix[3][2]);
            }
        } else {
            particlesGlobalDepthPtr[particleIdx] = particleSensorDistance;
        }
    }

    static inline __device__ void expand(tcnn::uvec2 tileGrid,
                                         int numParticles,
                                         tcnn::ivec2 /*resolution*/,
                                         threedgut::TSensorModel sensorModel,
                                         threedgut::TSensorState /*sensorState*/,
                                         const uint32_t* __restrict__ particlesTilesOffsetPtr,
                                         const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                         const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                         const tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                         const float* __restrict__ particlesGlobalDepthPtr,
                                         threedgut::MemoryHandles parameters,
                                         uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                         uint32_t* __restrict__ unsortedTileParticleIdxPtr) {

        const int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (particleIdx >= numParticles) {
            return;
        }

        const tcnn::vec2 particleProjExtent = particlesProjectedExtentPtr[particleIdx];
        // check the particle projected extent x
        constexpr float eps = 1e-06f;
        if (particleProjExtent.x <= eps) {
            return;
        }

        const uint32_t depthKey             = *reinterpret_cast<const uint32_t*>(&particlesGlobalDepthPtr[particleIdx]);
        uint32_t tileOffset                 = (particleIdx == 0) ? 0 : particlesTilesOffsetPtr[particleIdx - 1];
        const tcnn::vec2 particleProjCenter = particlesProjectedPositionPtr[particleIdx];
        const BoundingBox2D tileBBox        = computeTileSpaceBBox(tileGrid, particleProjCenter, particleProjExtent);
        const bool equirectModel            = (sensorModel.modelType == threedgut::TSensorModel::EquirectangularModel);
        // Same unwrapped X range the count loop in eval() used; the iteration count
        // is identical, only the emitted KEY column is wrapped modulo gridX below.
        const EquirectXTileRange xRange     = equirectModel ? computeEquirectXTileRange(tileGrid, particleProjCenter, particleProjExtent)
                                                            : EquirectXTileRange{static_cast<int>(tileBBox.min.x), static_cast<int>(tileBBox.max.x)};
        const int gridX                     = static_cast<int>(tileGrid.x);

        if constexpr (Params::TileCulling) {

            const uint32_t maxTileOffset = particlesTilesOffsetPtr[particleIdx];

            const tcnn::vec4 conicOpacity    = particlesProjectedConicOpacityPtr[particleIdx];
            const float maxConicOpacityPower = logf(conicOpacity.w / Params::AlphaThreshold);

            if (equirectModel) {
                for (int y = tileBBox.min.y; (y < tileBBox.max.y) && (tileOffset < maxTileOffset); ++y) {
                    for (int x = xRange.min; (x < xRange.max) && (tileOffset < maxTileOffset); ++x) {
                        if (tileMinParticlePowerResponse(tcnn::vec2(x, y), conicOpacity, particleProjCenter) < maxConicOpacityPower) {
                            const uint32_t keyX                    = wrapTileX(x, gridX);
                            unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(y * tileGrid.x + keyX, depthKey);
                            unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                            tileOffset++;
                        }
                    }
                }
            } else {
                for (int y = tileBBox.min.y; (y < tileBBox.max.y) && (tileOffset < maxTileOffset); ++y) {
                    for (int x = tileBBox.min.x; (x < tileBBox.max.x) && (tileOffset < maxTileOffset); ++x) {
                        if (tileMinParticlePowerResponse(tcnn::vec2(x, y), conicOpacity, particleProjCenter) < maxConicOpacityPower) {
                            unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(y * tileGrid.x + x, depthKey);
                            unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                            tileOffset++;
                        }
                    }
                }
            }
            for (; tileOffset < maxTileOffset; ++tileOffset) {
                unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(threedgut::GUTParameters::Tiling::InvalidTileIdx,
                                                                             __float_as_uint(Params::MaxDepthValue));
                unsortedTileParticleIdxPtr[tileOffset] = threedgut::GUTParameters::InvalidParticleIdx;
            }

        } else {

            if (equirectModel) {
                for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                    for (int x = xRange.min; x < xRange.max; ++x) {
                        const uint32_t keyX                    = wrapTileX(x, gridX);
                        unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(y * tileGrid.x + keyX, depthKey);
                        unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                        tileOffset++;
                    }
                }
            } else {
                for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                    for (int x = tileBBox.min.x; x < tileBBox.max.x; ++x) {
                        unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(y * tileGrid.x + x, depthKey);
                        unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                        tileOffset++;
                    }
                }
            }
        }
    }

    static inline __device__ void
    evalBackward(tcnn::uvec2 tileGrid,
                 uint32_t numParticles,
                 tcnn::ivec2 resolution,
                 threedgut::TSensorModel sensorModel,
                 tcnn::vec3 sensorWorldPosition,
                 tcnn::mat4x3 sensorViewMatrix,
                 const uint32_t* __restrict__ particlesTilesCountPtr,
                 threedgut::MemoryHandles parameters,
                 const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                 const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                 const float* __restrict__ particlesGlobalDepthGradPtr,
                 const float* __restrict__ particlesPrecomputedFeaturesPtr,
                 const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                 threedgut::MemoryHandles parametersGradient) {
        if constexpr (!Params::PerRayParticleFeatures) {

            const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (particleIdx >= numParticles) {
                return;
            }
            if (particlesTilesCountPtr[particleIdx] == 0) {
                return;
            }

            Particles particles;
            particles.initializeDensity(parameters);
            particles.initializeDensityGradient(parametersGradient);
            const tcnn::vec3 incidentDirection = tcnn::normalize(particles.fetchPosition(particleIdx) - sensorWorldPosition);
            tcnn::vec3 incidentDirectionGrad   = tcnn::vec3(0.0f);

            particles.initializeFeatures(parameters);
            particles.initializeFeaturesGradient(parametersGradient);
            particles.featuresBwdToBuffer<true>(particleIdx,
                                                reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr)[particleIdx],
                                                incidentDirection,
                                                incidentDirectionGrad);

            particles.template densityIncidentDirectionBwdToBuffer<true>(particleIdx, sensorWorldPosition, incidentDirectionGrad);
        }
    }
};
