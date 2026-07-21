// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use it except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Spherical-harmonics radiance with an analytic positional carrier residual
// (Gabor or per-particle SIREN), selected by FEATURE_TRANSFORM_TYPE == 2.
// Same CudaDeviceExport API as shRadiativeParticles.slang for drop-in
// replacement. The carrier evaluates on the canonical-space hit position
// (canonicalPosition.xy), which per-ray feature mode already delivers to
// every entry point; carrier parameters are packed after the SH
// coefficients in the same radiance buffer, so autodiff of the fetch
// produces their gradients with no extra plumbing.

#include <3dgut/kernels/slang/common/sphericalHarmonics.slang>

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

namespace shRadiativeParticle
{

struct ParametersBuffer
{
    Ptr<vector<float, Dim>, Access.Read> _dataPtr;
    vector<float, Dim> *_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct Parameters : IDifferentiable
{
    vector<float, Dim> sphCoefficients[RadianceMaxNumSphCoefficients];
};

[BackwardDifferentiable][ForceInline]
Parameters fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                     no_diff ParametersBuffer parametersBuffer)
{
    Parameters parameters;
    const uint32_t particleOffset = particleIdx * RadianceMaxNumSphCoefficients;
    [ForceUnroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        parameters.sphCoefficients[i] = parametersBuffer._dataPtr[particleOffset + i];
    }
    return parameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline]
void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                  no_diff ParametersBuffer parametersBuffer,
                                  Parameters parametersGrad)
{
    const uint32_t particleOffset = particleIdx * RadianceMaxNumSphCoefficients;
    [ForceUnroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        const vector<float, Dim> coeffs = parametersGrad.sphCoefficients[i];
        if (parametersBuffer.exclusiveGradient) {
            [ForceUnroll] for (int j = 0; j < Dim; ++j) {
                parametersBuffer._gradPtr[particleOffset + i][j] += coeffs[j];
            }
        } else {
            [ForceUnroll] for (int j = 0; j < Dim; ++j) {
                InterlockedAdd(parametersBuffer._gradPtr[particleOffset + i][j], coeffs[j]);
            }
        }
    }
}

[BackwardDifferentiable][ForceInline]
vector<float, Dim> radianceFromParameters(float3 incidentDirection,
                                          no_diff float2 localUv,
                                          no_diff uint32_t sphDegree,
                                          Parameters parameters)
{
    vector<float, Dim> radiance =
        sphericalHarmonics.decode<Dim, RadianceMaxNumSphCoefficients>(
            sphDegree,
            parameters.sphCoefficients,
            incidentDirection);
    if (GaborEnabled)
    {
        static const float TwoPi = 6.283185307179586f;
        static const float TwoPiSquared = 19.739208802178716f;
        const int carrierOffset = RadianceNumSphCoefficients;
        const int freqXOffset = carrierOffset + GaborNumTerms;
        const int freqYOffset = freqXOffset + 1;
        const int phaseOffset = freqYOffset + 1;
        [unroll] for (int i = 0; i < GaborNumTerms; ++i)
        {
            const vector<float, Dim> amplitude =
                parameters.sphCoefficients[carrierOffset + i];
            const float fx = clamp(
                parameters.sphCoefficients[freqXOffset][i],
                -GaborMaxFrequency,
                GaborMaxFrequency);
            const float fy = clamp(
                parameters.sphCoefficients[freqYOffset][i],
                -GaborMaxFrequency,
                GaborMaxFrequency);
            const float phase = parameters.sphCoefficients[phaseOffset][i];
            const float carrierPhase =
                TwoPi * (fx * localUv.x + fy * localUv.y) + phase;
            const float dc =
                exp(-TwoPiSquared * (fx * fx + fy * fy)) * cos(phase);
            radiance += amplitude * (cos(carrierPhase) - dc);
        }
        radiance = max(radiance, vector<float, Dim>(0));
    }
    if (HermiteEnabled)
    {
        const int carrierOffset = RadianceNumSphCoefficients;
        const vector<float, Dim> slopeU =
            parameters.sphCoefficients[carrierOffset];
        const vector<float, Dim> slopeV =
            parameters.sphCoefficients[carrierOffset + 1];
        radiance += slopeU * localUv.x + slopeV * localUv.y;
        radiance = max(radiance, vector<float, Dim>(0));
    }
    if (SirenEnabled)
    {
        const int sirenOffset = RadianceNumSphCoefficients;
        const int w1Offset = sirenOffset;
        const int b1Offset = w1Offset + 2 * SirenHiddenDim;
        const int b1CoeffCount = (SirenHiddenDim + 2) / 3;
        const int w2Offset = b1Offset + b1CoeffCount;
        const int b2Offset = w2Offset + SirenHiddenDim;
        const float uvScale = max(abs(SirenLocalUvScale), 1e-6f);
        const float uvX = localUv.x / uvScale;
        const float uvY = localUv.y / uvScale;

        vector<float, Dim> residual = parameters.sphCoefficients[b2Offset];
        [unroll] for (int i = 0; i < SirenHiddenDim; ++i)
        {
            const int rowOffset = w1Offset + 2 * i;
            const vector<float, Dim> w012 =
                parameters.sphCoefficients[rowOffset];
            const vector<float, Dim> w34 =
                parameters.sphCoefficients[rowOffset + 1];
            const vector<float, Dim> bias =
                parameters.sphCoefficients[b1Offset + i / 3];
            const float hidden =
                bias[i % 3]
                + w012.x * uvX
                + w012.y * uvY
                + w012.z * incidentDirection.x
                + w34.x * incidentDirection.y
                + w34.y * incidentDirection.z;
            residual += parameters.sphCoefficients[w2Offset + i]
                * sin(SirenOmega * hidden);
        }
        radiance += residual;
        radiance = max(radiance, vector<float, Dim>(0));
    }
    return radiance;
}

[BackwardDifferentiable] [ForceInline]
vector<float, Dim> radianceFromBuffer(no_diff uint32_t particleIdx,
                                      in float3 incidentDirection,
                                      no_diff float2 localUv,
                                      no_diff uint32_t sphDegree,
                                      no_diff ParametersBuffer parametersBuffer)
{
    return radianceFromParameters(
        incidentDirection,
        localUv,
        sphDegree,
        fetchParametersFromBuffer(particleIdx, parametersBuffer));
}

[BackwardDifferentiable][ForceInline]
void integrateRadiance<let backToFront : bool>(float weight,
                                               in vector<float, Dim> radiance,
                                               inout vector<float, Dim> integratedRadiance)
{
    if (weight > 0.0f)
    {
        if (backToFront)
        {
            integratedRadiance = lerp(integratedRadiance, radiance, weight);
        }
        else
        {
            integratedRadiance += radiance * weight;
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromParameters<let backToFront : bool>(float3 incidentDirection,
                                                             no_diff float2 localUv,
                                                             no_diff uint32_t sphDegree,
                                                             float weight,
                                                             Parameters parameters,
                                                             inout vector<float, Dim> integratedRadiance)
{
    if (weight > 0.0f)
    {
        integrateRadiance<backToFront>(
            weight,
            radianceFromParameters(incidentDirection,
                                   localUv,
                                   sphDegree,
                                   parameters),
            integratedRadiance
        );
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromBuffer<let backToFront : bool>(no_diff float3 incidentDirection,
                                                         no_diff float2 localUv,
                                                         no_diff uint32_t sphDegree,
                                                         float weight,
                                                         no_diff uint32_t particleIdx,
                                                         no_diff ParametersBuffer parametersBuffer,
                                                         inout vector<float, Dim> integratedRadiance)
{
    integrateRadianceFromParameters<backToFront>(incidentDirection,
                                                 localUv,
                                                 sphDegree,
                                                 weight,
                                                 fetchParametersFromBuffer(particleIdx, parametersBuffer),
                                                 integratedRadiance);
}

} // namespace shRadiativeParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport]
inline void particleFeaturesFromBuffer(
    in uint32_t particleIdx,
    in float *radianceBufferPtr,
    in int auxParam,
    in float3 incidentDirection,
    in float3 canonicalPosition,
    out float features[shRadiativeParticle.Dim])
{
    shRadiativeParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)radianceBufferPtr;
    parametersBuffer._gradPtr = nullptr;
    parametersBuffer.exclusiveGradient = false;

    vector<float, shRadiativeParticle.Dim> featuresVec =
        shRadiativeParticle.radianceFromBuffer(
            particleIdx,
            incidentDirection,
            canonicalPosition.xy,
            auxParam,
            parametersBuffer);

    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        features[i] = featuresVec[i];
    }
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwd(in float weight,
                                         in float features[shRadiativeParticle.Dim],
                                         inout float integratedFeatures[shRadiativeParticle.Dim])
{
    vector<float, shRadiativeParticle.Dim> featuresVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        featuresVec[i] = features[i];
    }
    vector<float, shRadiativeParticle.Dim> integratedVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedVec[i] = integratedFeatures[i];
    }
    shRadiativeParticle.integrateRadiance<false>(weight, featuresVec, integratedVec);
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedFeatures[i] = integratedVec[i];
    }
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwdFromBuffer(
    in float3 incidentDirection,
    in float3 canonicalPosition,
    in float weight,
    in uint32_t particleIdx,
    in float *radianceBufferPtr,
    in int auxParam,
    inout float integratedFeatures[shRadiativeParticle.Dim])
{
    shRadiativeParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)radianceBufferPtr;
    parametersBuffer._gradPtr = nullptr;
    parametersBuffer.exclusiveGradient = false;

    vector<float, shRadiativeParticle.Dim> integratedVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedVec[i] = integratedFeatures[i];
    }
    shRadiativeParticle.integrateRadianceFromBuffer<false>(
        incidentDirection, canonicalPosition.xy, auxParam, weight, particleIdx,
        parametersBuffer, integratedVec);
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedFeatures[i] = integratedVec[i];
    }
}

[CudaDeviceExport] void particleFeaturesIntegrateBwd(
    in float alpha,
    inout float alphaGrad,
    in float features[shRadiativeParticle.Dim],
    inout float featuresGrad[shRadiativeParticle.Dim],
    inout float integratedFeatures[shRadiativeParticle.Dim],
    inout float integratedFeaturesGrad[shRadiativeParticle.Dim])
{
    if(alpha > 0.0f)
    {
        vector<float, shRadiativeParticle.Dim> featuresVec, featuresGradVec;
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            featuresVec[i] = features[i];
            featuresGradVec[i] = featuresGrad[i];
        }
        vector<float, shRadiativeParticle.Dim> integratedVec, integratedGradVec;
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedVec[i] = integratedFeatures[i];
            integratedGradVec[i] = integratedFeaturesGrad[i];
        }

        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> featuresDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(featuresVec, featuresGradVec);

        const float weight = 1.0f / (1.0f - alpha);
        integratedVec = (integratedVec - featuresVec * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedVec, integratedGradVec);

        bwd_diff(shRadiativeParticle.integrateRadiance<true>)(
            alphaDiff,
            featuresDiff,
            integratedFeaturesDiff);

        alphaGrad = alphaDiff.getDifferential();
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            featuresGrad[i] = featuresDiff.getDifferential()[i];
        }
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeaturesGrad[i] = integratedFeaturesDiff.getDifferential()[i];
        }
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeatures[i] = integratedVec[i];
        }
    }
}

// canonicalPosition feeds the carrier (no_diff): its gradient is not
// accumulated, matching the carrier's pre-2.0 semantics; the SH term is
// position-independent so canonicalPositionGrad is left untouched.
[CudaDeviceExport] void particleFeaturesIntegrateBwdToBuffer(
    in float3 incidentDirection,
    in float3 canonicalPosition,
    inout float3 canonicalPositionGrad,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    in float *featuresBufferPtr,
    in float *featuresBufferGradPtr,
    in int auxParam,
    in bool exclusiveGradient,
    in float features[shRadiativeParticle.Dim],
    inout float integratedFeatures[shRadiativeParticle.Dim],
    inout float integratedFeaturesGrad[shRadiativeParticle.Dim])
{
    if (alpha > 0.0f)
    {
        shRadiativeParticle.ParametersBuffer parametersBuffer;
        parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)featuresBufferPtr;
        parametersBuffer._gradPtr = (vector<float, 3>*)featuresBufferGradPtr;
        parametersBuffer.exclusiveGradient = exclusiveGradient;

        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);

        const float weight = 1.0f / (1.0f - alpha);
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeatures[i] = (integratedFeatures[i] - features[i] * alpha) * weight;
        }

        vector<float, shRadiativeParticle.Dim> integratedFeaturesVec;
        vector<float, shRadiativeParticle.Dim> integratedFeaturesGradVec;
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeaturesVec[i]     = integratedFeatures[i];
            integratedFeaturesGradVec[i] = integratedFeaturesGrad[i];
        }
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeaturesVec, integratedFeaturesGradVec);

        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)(
            incidentDirection,
            canonicalPosition.xy,
            auxParam,
            alphaDiff,
            particleIdx,
            parametersBuffer,
            integratedFeaturesDiff);

        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeaturesGrad[i] = integratedFeaturesDiff.getDifferential()[i];
        }
        alphaGrad += alphaDiff.getDifferential();
    }
}

// Positionless backward: only reachable on the per-particle precomputed
// features path (PerRayParticleFeatures == false), which never compiles for
// this transform type; localUv is zero so only the SH term contributes.
[CudaDeviceExport] void particleFeaturesBwdToBuffer(
    in uint32_t particleIdx,
    in float *featuresBufferPtr,
    in float *featuresBufferGradPtr,
    in int auxParam,
    in bool exclusiveGradient,
    in float featuresGrad[shRadiativeParticle.Dim],
    in float3 incidentDirection,
    inout float3 incidentDirectionGrad
)
{
    shRadiativeParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)featuresBufferPtr;
    parametersBuffer._gradPtr = (vector<float, 3>*)featuresBufferGradPtr;
    parametersBuffer.exclusiveGradient = exclusiveGradient;

    vector<float, shRadiativeParticle.Dim> featuresGradVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) featuresGradVec[i] = featuresGrad[i];

    DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, incidentDirectionGrad);

    bwd_diff(shRadiativeParticle.radianceFromBuffer)(
        particleIdx,
        incidentDirectionDiff,
        float2(0.0f, 0.0f),
        auxParam,
        parametersBuffer,
        featuresGradVec);

    incidentDirectionGrad += incidentDirectionDiff.getDifferential();
}
