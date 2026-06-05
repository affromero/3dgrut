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

#include <3dgut/sensors/sensors.h>

#include <cfloat>

namespace threedgut {

// Computes 2-norm of a [x,y] vector in a numerically stable way
inline __device__ float stableNorm2(const tcnn::vec2& vec) {
    const float absX = fabsf(vec.x);
    const float absY = fabsf(vec.y);
    const float min  = fminf(absX, absY);
    const float max  = fmaxf(absX, absY);
    if (max <= 0.f) {
        return 0.f;
    }
    const float minMaxRatio = min / max;
    return max * sqrtf(1.f + minMaxRatio * minMaxRatio);
}

template <int N>
static inline __device__ float evalPolyHorner(const tcnn::vec<N>& coeffs, float x) {
    // Evaluates a N-1 degree polynomial y=f(x) using numerically stable Horner scheme.
    // With :
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    float y = coeffs[N - 1];
#pragma unroll
    for (int i = N - 2; i >= 0; --i) {
        y = x * y + coeffs[i];
    }
    return y;
}

static inline __device__ float relativeShutterTime(const TSensorModel& sensorModel,
                                                   const tcnn::ivec2& resolution,
                                                   const tcnn::vec2& position) {
    switch (sensorModel.shutterType) {
    case TSensorModel::RollingTopToBottomShutter:
        return floorf(position.y) / (resolution.y - 1.f);
    case TSensorModel::RollingLeftToRightShutter:
        return floorf(position.x) / (resolution.x - 1.f);
    case TSensorModel::RollingBottomToTopShutter:
        return (resolution.y - ceilf(position.y)) / (resolution.y - 1.f);
    case TSensorModel::RollingRightToLeftShutter:
        return (resolution.x - ceilf(position.x)) / (resolution.x - 1.f);
    default:
        return 0.5f;
    }
};

static __forceinline__ __device__ bool withinResolution(const tcnn::vec2& resolution, float tolerance, const tcnn::vec2& p) {
    const tcnn::vec2 tolMargin = resolution * tolerance;
    return (p.x > -tolMargin.x) && (p.y > -tolMargin.y) && (p.x < resolution.x + tolMargin.x) && (p.y < resolution.y + tolMargin.y);
}

static inline __device__ bool projectPoint(const OpenCVPinholeProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {

    if (position.z <= 0.f) {
        projected = tcnn::vec2::zero();
        return false;
    }

    const tcnn::vec2 uvNormalized = position.xy() / position.z;

    // computeDistortion
    const tcnn::vec2 uvSquared = uvNormalized * uvNormalized;
    const float r2             = uvSquared.x + uvSquared.y;
    const float a1             = 2.f * uvNormalized.x * uvNormalized.y;
    const float a2             = r2 + 2.f * uvSquared.x;
    const float a3             = r2 + 2.f * uvSquared.y;

    const float icD_numerator   = 1.f + r2 * (sensorParams.radialCoeffs[0] + r2 * (sensorParams.radialCoeffs[1] + r2 * sensorParams.radialCoeffs[2]));
    const float icD_denominator = 1.f + r2 * (sensorParams.radialCoeffs[3] + r2 * (sensorParams.radialCoeffs[4] + r2 * sensorParams.radialCoeffs[5]));
    const float icD             = icD_numerator / icD_denominator;

    const tcnn::vec2 delta = tcnn::vec2{
        sensorParams.tangentialCoeffs[0] * a1 + sensorParams.tangentialCoeffs[1] * a2 + r2 * (sensorParams.thinPrismCoeffs[0] + r2 * sensorParams.thinPrismCoeffs[1]),
        sensorParams.tangentialCoeffs[0] * a3 + sensorParams.tangentialCoeffs[1] * a1 + r2 * (sensorParams.thinPrismCoeffs[2] + r2 * sensorParams.thinPrismCoeffs[3])};

    // Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
    // in case radial distortion is within limits
    const tcnn::vec2 uvND = icD * uvNormalized + delta;

    constexpr float kMinRadialDist = 0.8f, kMaxRadialDist = 1.2f;
    const bool validRadial = (icD > kMinRadialDist) && (icD < kMaxRadialDist);
    if (validRadial) {
        projected = uvND * sensorParams.focalLength + sensorParams.principalPoint;
    } else {
        // If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
        // (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
        // out of the image bounds accordingly. The coordinates will be clipped to
        // viable range and direction but the exact values cannot be trusted / are still invalid
        const float roiClippingRadius = hypotf(resolution.x, resolution.y);
        projected                     = (roiClippingRadius / sqrtf(r2)) * uvNormalized + sensorParams.principalPoint;
    }

    return validRadial && withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const OpenCVFisheyeProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    float rho = stableNorm2(position);
    if (rho <= 0.f) {
        rho = FLT_EPSILON;
    }

    const float thetaFull = atan2f(rho, position.z);
    // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
    // In particular for omnidirectional cameras, this prevents points outside the FOV to be
    // wrongly projected to in-image-domain points because of badly constrained polynomials outside
    // the effective FOV (which is different to the image boundaries).
    //
    // These FOV-clamped projections will be marked as *invalid*
    const float theta = fminf(thetaFull, sensorParams.maxAngle);
    // Evaluate forward polynomial
    // (radial distances to the principal point in the normalized image domain (up to focal length scales))
    const float theta2 = theta * theta;
    const float delta =
        (theta * (evalPolyHorner<4>(sensorParams.radialCoeffs, theta2) * theta2 + 1.0f)) / rho;
    projected = sensorParams.focalLength * position.xy() * delta + sensorParams.principalPoint;

    return (theta < sensorParams.maxAngle) && withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const RationalProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    if (position.z <= 0.f) {
        projected = tcnn::vec2::zero();
        return false;
    }

    const tcnn::vec2 uvNormalized = position.xy() / position.z;
    const float x                 = uvNormalized.x;
    const float y                 = uvNormalized.y;
    const float r2                = x * x + y * y;
    const float r4                = r2 * r2;
    const float r6                = r4 * r2;
    const float numerator =
        1.f + sensorParams.numeratorCoeffs[0] * r2 + sensorParams.numeratorCoeffs[1] * r4 + sensorParams.numeratorCoeffs[2] * r6;
    float denominator =
        1.f + sensorParams.denominatorCoeffs[0] * r2 + sensorParams.denominatorCoeffs[1] * r4 + sensorParams.denominatorCoeffs[2] * r6;
    if (fabsf(denominator) <= 1e-12f) {
        denominator = 1.f;
    }
    const float radial = numerator / denominator;
    const float affine = 1.f + sensorParams.affineCoeffs[0] * r2 + sensorParams.affineCoeffs[1] * r4;
    const float p1     = sensorParams.tangentialCoeffs[0];
    const float p2     = sensorParams.tangentialCoeffs[1];

    const float xDistorted = x * radial + affine * (p2 * (r2 + 2.f * x * x) + 2.f * p1 * x * y);
    const float yDistorted = y * radial + affine * (p1 * (r2 + 2.f * y * y) + 2.f * p2 * x * y);

    const float nativeX = sensorParams.focalLength.x * xDistorted + sensorParams.skew * yDistorted + sensorParams.principalPoint.x;
    const float nativeY = sensorParams.focalLength.y * yDistorted + sensorParams.principalPoint.y;
    const tcnn::vec2 nativeProjected = {nativeX, nativeY};
    const bool withinNativeResolution = withinResolution(sensorParams.nativeResolution, tolerance, nativeProjected);

    const int rotation = ((sensorParams.imageRotationQuadrantsCw % 4) + 4) % 4;
    if (rotation == 1) {
        projected.x = static_cast<float>(resolution.x - 1) - nativeY;
        projected.y = nativeX;
    } else if (rotation == 2) {
        projected.x = static_cast<float>(resolution.x - 1) - nativeX;
        projected.y = static_cast<float>(resolution.y - 1) - nativeY;
    } else if (rotation == 3) {
        projected.x = nativeY;
        projected.y = static_cast<float>(resolution.y - 1) - nativeX;
    } else {
        projected.x = nativeX;
        projected.y = nativeY;
    }

    return withinNativeResolution && withinResolution(resolution, tolerance, projected);
}

template <int NumNewtonIterations = 3>
static inline __device__ bool projectPoint(const FThetaProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    float rho = stableNorm2(position);
    if (rho <= 0.f) {
        rho = FLT_EPSILON;
    }

    const float thetaFull = atan2f(rho, position.z);
    // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
    // In particular for omnidirectional cameras, this prevents points outside the FOV to be
    // wrongly projected to in-image-domain points because of badly constrained polynomials outside
    // the effective FOV (which is different to the image boundaries).
    //
    // These FOV-clamped projections will be marked as *invalid*
    const float theta = fminf(thetaFull, sensorParams.maxAngle);

    // Evaluate forward polynomial (depending on reference direction), giving delta = f(theta) factor
    float delta = 0.f;
    if (sensorParams.referencePoly == FThetaProjectionParameters::PolynomialType::PIXELDIST_TO_ANGLE) {
        // pixeldist-to-angle polynomial (bw) is reference, evaluate its accurate inverse via Newton-based inversion
        delta = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.angleToPixeldistPoly, theta);
        tcnn::vec<FThetaProjectionParameters::PolynomialDegree - 1> dPixeldistToAnglePoly;
#pragma unroll
        for (auto i = 1; i < FThetaProjectionParameters::PolynomialDegree; ++i) {
            dPixeldistToAnglePoly[i - 1] = i * sensorParams.pixeldistToAnglePoly[i];
        }
#pragma unroll
        for (auto i = 0; i < NumNewtonIterations; ++i) {
            const float dfdx     = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree - 1>(dPixeldistToAnglePoly, delta);
            const float residual = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.pixeldistToAnglePoly, delta) - theta;
            delta -= residual / dfdx;
        }
    } else {
        // angle-to-pixeldist polynomial (fw) is reference, evaluate it directly
        delta = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.angleToPixeldistPoly, theta);
    }

    // Apply linear term A=[c,d;e,1] to f(theta)-weighted normalized 2d vectors, relative to principal point
    projected = (delta / rho) * tcnn::vec2{sensorParams.linear_cde[0] * position.x + sensorParams.linear_cde[1] * position.y,
                                           sensorParams.linear_cde[2] * position.x + position.y};

    // FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
    // the center of the first pixel.
    projected += sensorParams.principalPoint + .5f;

    return (theta < sensorParams.maxAngle) && withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const EquirectProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    // Full-sphere equirectangular projection in the [right, down, forward]
    // camera frame (matches the dataset ray-gen and the COLMAP poses loaded
    // without a coordinate flip). Every direction maps to a valid pixel.
    constexpr float kPi = 3.14159265358979323846f;
    const float n = sqrtf(position.x * position.x + position.y * position.y + position.z * position.z);
    if (n <= 0.f) {
        projected = tcnn::vec2::zero();
        return false;
    }
    const float theta = asinf(fminf(fmaxf(-position.y / n, -1.f), 1.f)); // latitude  [-pi/2, pi/2]
    const float phi   = atan2f(-position.x, position.z);                 // longitude [-pi, pi]
    const float width  = static_cast<float>(resolution.x);
    const float height = static_cast<float>(resolution.y);
    float col = (1.f - phi / kPi) * 0.5f * width - 0.5f;
    float row = (1.f - 2.f * theta / kPi) * 0.5f * height - 0.5f;
    col       = col - width * floorf(col / width); // wrap azimuth into [0, width)
    // The floorf wrap is exact in real arithmetic but float32 rounding can
    // round col up to exactly `width` for directions at azimuth ~+-pi (e.g.
    // the seam ray of pixel column 0). Column `width` is congruent to column
    // 0 (the seam), so fold it back -- this keeps col strictly inside
    // [0, width) and is the exact inverse of create_equirect_camera's
    // pixel-center ray-gen.
    if (col >= width) {
        col -= width;
    }
    row       = fminf(fmaxf(row, 0.f), height - 1.f);
    projected = tcnn::vec2{col, row};
    // Never report a valid projection for an out-of-range pixel: after the
    // seam fold and row clamp this always holds, but assert it structurally
    // so the (col == width) class of bug can never escape this function.
    return (col >= 0.f) && (col < width) && (row >= 0.f) && (row < height);
}

static inline __device__ bool projectPoint(const TSensorModel& sensorModel,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    switch (sensorModel.modelType) {
    case TSensorModel::OpenCVPinholeModel:
        return projectPoint(sensorModel.ocvPinholeParams, resolution, position, tolerance, projected);
    case TSensorModel::OpenCVFisheyeModel:
        return projectPoint(sensorModel.ocvFisheyeParams, resolution, position, tolerance, projected);
    case TSensorModel::FThetaModel:
        return projectPoint(sensorModel.fthetaParams, resolution, position, tolerance, projected);
    case TSensorModel::RationalModel:
        return projectPoint(sensorModel.rationalParams, resolution, position, tolerance, projected);
    case TSensorModel::EquirectangularModel:
        return projectPoint(sensorModel.equirectParams, resolution, position, tolerance, projected);
    default:
        projected = tcnn::vec2::zero();
        return false;
    }
}

template <int NRollingShutterIterations>
static inline __device__ bool projectPointWithShutter(const tcnn::vec3& position,
                                                      const tcnn::ivec2& resolution,
                                                      const TSensorModel& sensorModel,
                                                      const TSensorState& sensorState,
                                                      float tolerance,
                                                      tcnn::vec2& projectedPosition) {

    const tcnn::vec3 tStart = sensorState.startPose.slice<0, 3>();
    const tcnn::quat qStart = tcnn::quat{sensorState.startPose[6], sensorState.startPose[3], sensorState.startPose[4], sensorState.startPose[5]};

    bool validProjection = projectPoint(sensorModel, resolution, tcnn::to_mat3(qStart) * position + tStart, tolerance, projectedPosition);
    if (sensorModel.shutterType == TSensorModel::GlobalShutter) {
        return validProjection;
    }

    const tcnn::vec3 tEnd = sensorState.endPose.slice<0, 3>();
    const tcnn::quat qEnd = tcnn::quat{sensorState.endPose[6], sensorState.endPose[3], sensorState.endPose[4], sensorState.endPose[5]};

    // Rolling-shutter pose interpolation must interpolate the camera CENTER,
    // not the world->camera translation. Since t = -R*C, linearly mixing t is
    // only correct when R is constant across the shutter; with rotation during
    // readout it is wrong. Interpolating the center C and applying the slerped
    // rotation is the physically-correct rigid (screw) motion. This is an
    // upstream bug that also affects pinhole/fisheye RS-with-rotation; the
    // wide-FOV equirect sweep just made it large and measurable.
    const tcnn::vec3 cStart = -(tcnn::transpose(tcnn::to_mat3(qStart)) * tStart);
    const tcnn::vec3 cEnd   = -(tcnn::transpose(tcnn::to_mat3(qEnd)) * tEnd);

    if (!validProjection) {
        validProjection = projectPoint(sensorModel, resolution, tcnn::to_mat3(qEnd) * position + tEnd, tolerance, projectedPosition);
        if (!validProjection) {
            return false;
        }
    }

    // Compute the new timestamp and project again
#pragma unroll
    for (int i = 0; i < NRollingShutterIterations; ++i) {
        const float alpha = relativeShutterTime(sensorModel, resolution, projectedPosition);
        validProjection   = projectPoint(
            sensorModel,
            resolution,
            tcnn::to_mat3(tcnn::slerp(qStart, qEnd, alpha)) * (position - tcnn::mix(cStart, cEnd, alpha)),
            tolerance,
            projectedPosition);
    }

    return validProjection;
}

} // namespace threedgut
