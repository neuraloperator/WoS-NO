// This file implements the Boundary Value Caching algorithm for reducing variance
// of the walk-on-spheres and walk-on-stars estimators at a set of user-selected
// evaluation points via sample caching and reuse.
//
// Resources:
// - Boundary Value Caching for Walk on Spheres [2023]

#pragma once

#include <zombie/point_estimation/walk_on_stars.h>

float diffusionFreq = 0.5;
using VectorD = Eigen::Matrix<float, 3, 1>;


namespace zombie {

namespace bvc_dt {
template <typename T, size_t DIM>
float diffusion (const zombie::Vector<DIM>& x)  {
        float a = 4.0f*M_PI*diffusionFreq;
        float b = 3.0f*M_PI*diffusionFreq;
        float sinAx = std::sin(a*x(0));
        float cosAx = std::cos(a*x(0));
        float sinBy = std::sin(b*x(1));
        float cosBy = std::cos(b*x(1));
        float alpha = std::exp(-x(1)*x(1) + cosAx*sinBy);
        return alpha;
};
template <typename T, size_t DIM>
struct EvaluationPoint {
    // constructor
    EvaluationPoint(const Vector<DIM>& pt_,
                    const Vector<DIM>& normal_,
                    SampleType type_,
                    float distToAbsorbingBoundary_,
                    float distToReflectingBoundary_);

    // returns estimated solution
    T getEstimatedSolution() const;

    // returns estimated gradient
    void getEstimatedGradient(std::vector<T>& gradient) const;

    // returns estimated gradient for specified channel
    T getEstimatedGradient(int channel) const;

    // resets statistics
    void reset();

    // members
    Vector<DIM> pt;
    Vector<DIM> normal;
    SampleType type;
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;

protected:
    // members
    SampleStatistics<T, DIM> absorbingBoundaryStatistics;
    SampleStatistics<T, DIM> absorbingBoundaryNormalAlignedStatistics;
    SampleStatistics<T, DIM> reflectingBoundaryStatistics;
    SampleStatistics<T, DIM> reflectingBoundaryNormalAlignedStatistics;
    SampleStatistics<T, DIM> sourceStatistics;

    template <typename A, size_t B>
    friend class BoundaryValueCachingDelta;
};

template <typename T, size_t DIM>
class BoundaryValueCachingDelta {
public:
    // constructor
    BoundaryValueCachingDelta(const GeometricQueries<DIM>& queries_,
                         const DeltaTrack<T, DIM>& walkOnStars_);

    // solves the given PDEVar at the provided sample points
    void computeBoundaryEstimates(const PDEVar<T, DIM>& pde,
                                  const WalkSettings& walkSettings,
                                  int nWalksForSolutionEstimates,
                                  int nWalksForGradientEstimates,
                                  float robinCoeffCutoffForNormalDerivative,
                                  std::vector<SamplePoint<T, DIM>>& samplePts,
                                  bool useFiniteDifferences=false,
                                  bool runSingleThreaded=false,
                                  std::function<void(int,int)> reportProgress={}) const;

    // sets the source value at the provided sample points
    void setSourceValues(const PDEVar<T, DIM>& pde,
                         std::vector<SamplePoint<T, DIM>>& samplePts,
                         bool runSingleThreaded=false) const;

    // splats sample pt data to the input evaluation pt
    void splat(const PDEVar<T, DIM>& pde,
               const SamplePoint<T, DIM>& samplePt,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               EvaluationPoint<T, DIM>& evalPt) const;

    // splats sample pt data to the input evaluation pt
    void splat(const PDEVar<T, DIM>& pde,
               const std::vector<SamplePoint<T, DIM>>& samplePts,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               EvaluationPoint<T, DIM>& evalPt) const;

    // splats sample pt data to the input evaluation pts
    void splat(const PDEVar<T, DIM>& pde,
               const SamplePoint<T, DIM>& samplePt,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               std::vector<EvaluationPoint<T, DIM>>& evalPts,
               bool runSingleThreaded=false) const;

    // splats sample pt data to the input evaluation pts
    void splat(const PDEVar<T, DIM>& pde,
               const std::vector<SamplePoint<T, DIM>>& samplePts,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               std::vector<EvaluationPoint<T, DIM>>& evalPts,
               std::function<void(int, int)> reportProgress={}) const;

    // estimates the solution at the input evaluation pt near the boundary
    void estimateSolutionNearBoundary(const PDEVar<T, DIM>& pde,
                                      const WalkSettings& walkSettings,
                                      bool useDistanceToAbsorbingBoundary,
                                      float cutoffDistToBoundary, int nWalks,
                                      EvaluationPoint<T, DIM>& evalPt) const;

    // estimates the solution at the input evaluation pts near the boundary
    void estimateSolutionNearBoundary(const PDEVar<T, DIM>& pde,
                                      const WalkSettings& walkSettings,
                                      bool useDistanceToAbsorbingBoundary,
                                      float cutoffDistToBoundary, int nWalks,
                                      std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                      bool runSingleThreaded=false) const;

protected:
    // sets estimation data for each sample point to compute boundary estimates
    void setEstimationData(const PDEVar<T, DIM>& pde,
                           const WalkSettings& walkSettings,
                           int nWalksForSolutionEstimates,
                           int nWalksForGradientEstimates,
                           float robinCoeffCutoffForNormalDerivative,
                           bool useFiniteDifferences,
                           std::vector<int>& nWalks,
                           std::vector<SamplePoint<T, DIM>>& samplePts) const;

    // sets the estimated boundary data for each sample point
    void setEstimatedBoundaryData(const PDEVar<T, DIM>& pde,
                                  const WalkSettings& walkSettings,
                                  float robinCoeffCutoffForNormalDerivative,
                                  bool useFiniteDifferences,
                                  std::vector<SamplePoint<T, DIM>>& samplePts) const;

    // splats boundary sample data
    void splatBoundaryData(const SamplePoint<T, DIM>& samplePt,
                           const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                           float radiusClamp,
                           float kernelRegularization,
                           float robinCoeffCutoffForNormalDerivative,
                           EvaluationPoint<T, DIM>& evalPt, const PDEVar<T, DIM>& pde) const;

    // splats source sample data
    void splatSourceData(const SamplePoint<T, DIM>& samplePt,
                         const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                         float radiusClamp,
                         float kernelRegularization,
                         EvaluationPoint<T, DIM>& evalPt, const PDEVar<T, DIM>& pde) const;

    // members
    const GeometricQueries<DIM>& queries;
    const DeltaTrack<T, DIM>& walkOnStars;
};

template <typename T, size_t DIM>
class BoundaryValueCachingSolver {
public:
    // constructor
    BoundaryValueCachingSolver(const GeometricQueries<DIM>& queries_,
                               std::shared_ptr<BoundarySampler<T, DIM>> absorbingBoundarySampler_,
                               std::shared_ptr<BoundarySampler<T, DIM>> reflectingBoundarySampler_,
                               std::shared_ptr<DomainSampler<T, DIM>> domainSampler_);

    // generates boundary and domain samples
    void generateSamples(int absorbingBoundaryCacheSize,
                         int reflectingBoundaryCacheSize,
                         int domainCacheSize,
                         float normalOffsetForAbsorbingBoundary,
                         float normalOffsetForReflectingBoundary,
                         bool solveDoubleSided);

    // computes sample estimates on the boundary
    void computeSampleEstimates(const PDEVar<T, DIM>& pde,
                                const WalkSettings& walkSettings,
                                int nWalksForSolutionEstimates,
                                int nWalksForGradientEstimates,
                                float robinCoeffCutoffForNormalDerivative,
                                bool useFiniteDifferences=false,
                                bool runSingleThreaded=false,
                                std::function<void(int,int)> reportProgress={});

    // splats solution and gradient estimates into the interior
    void splat(const PDEVar<T, DIM>& pde,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               std::vector<EvaluationPoint<T, DIM>>& evalPts,
               std::function<void(int, int)> reportProgress={}) const;

    // estimates the solution at the input evaluation points near the boundary
    void estimateSolutionNearBoundary(const PDEVar<T, DIM>& pde,
                                      const WalkSettings& walkSettings,
                                      float cutoffDistToAbsorbingBoundary,
                                      float cutoffDistToReflectingBoundary,
                                      int nWalksForSolutionEstimates,
                                      std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                      bool runSingleThreaded=false) const;

    // returns the boundary and domain cache
    const std::vector<SamplePoint<T, DIM>>& getAbsorbingBoundaryCache(bool returnBoundaryNormalAligned=false) const;
    const std::vector<SamplePoint<T, DIM>>& getReflectingBoundaryCache(bool returnBoundaryNormalAligned=false) const;
    const std::vector<SamplePoint<T, DIM>>& getDomainCache() const;

protected:
    // members
    const GeometricQueries<DIM>& queries;
    std::shared_ptr<BoundarySampler<T, DIM>> absorbingBoundarySampler;
    std::shared_ptr<BoundarySampler<T, DIM>> reflectingBoundarySampler;
    std::shared_ptr<DomainSampler<T, DIM>> domainSampler;
    DeltaTrack<T, DIM> walkOnStars;
    BoundaryValueCachingDelta<T, DIM> boundaryValueCaching;
    std::vector<SamplePoint<T, DIM>> absorbingBoundaryCache;
    std::vector<SamplePoint<T, DIM>> absorbingBoundaryCacheNormalAligned;
    std::vector<SamplePoint<T, DIM>> reflectingBoundaryCache;
    std::vector<SamplePoint<T, DIM>> reflectingBoundaryCacheNormalAligned;
    std::vector<SamplePoint<T, DIM>> domainCache;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - virtual boundary creation and estimation
// - bias correction/compensation
// - Barnes-Hut acceleration for splatting

template <typename T, size_t DIM>
inline EvaluationPoint<T, DIM>::EvaluationPoint(const Vector<DIM>& pt_,
                                                const Vector<DIM>& normal_,
                                                SampleType type_,
                                                float distToAbsorbingBoundary_,
                                                float distToReflectingBoundary_):
                                                pt(pt_), normal(normal_), type(type_),
                                                distToAbsorbingBoundary(distToAbsorbingBoundary_),
                                                distToReflectingBoundary(distToReflectingBoundary_)
{

}

template <typename T, size_t DIM>
inline T EvaluationPoint<T, DIM>::getEstimatedSolution() const
{
    T solution = absorbingBoundaryStatistics.getEstimatedSolution();
    solution += absorbingBoundaryNormalAlignedStatistics.getEstimatedSolution();
    solution += reflectingBoundaryStatistics.getEstimatedSolution();
    solution += reflectingBoundaryNormalAlignedStatistics.getEstimatedSolution();
    //std::cout<<"Boundary Statistics: "<<absorbingBoundaryStatistics.getEstimatedSolution()<<std::endl;
    
    
    //std::cout<<"Point Estimated solution: "<<solution<<std::endl;
    //std::cout<<"Source Statistics: "<<sourceStatistics.getEstimatedSolution()<<std::endl;
    solution += sourceStatistics.getEstimatedSolution();
    //std::cout<<"Point Estimated solution: "<<solution<<std::endl;
    solution /= std::sqrt(diffusion<T, DIM>(pt));
    //std::cout<<"Point Estimated solution: "<<solution<<std::endl;
    return solution;
}

template <typename T, size_t DIM>
inline void EvaluationPoint<T, DIM>::getEstimatedGradient(std::vector<T>& gradient) const
{
    gradient.resize(DIM);
    for (int i = 0; i < DIM; i++) {
        gradient[i] = absorbingBoundaryStatistics.getEstimatedGradient()[i];
        gradient[i] += absorbingBoundaryNormalAlignedStatistics.getEstimatedGradient()[i];
        gradient[i] += reflectingBoundaryStatistics.getEstimatedGradient()[i];
        gradient[i] += reflectingBoundaryNormalAlignedStatistics.getEstimatedGradient()[i];
        gradient[i] += sourceStatistics.getEstimatedGradient()[i];
    }
}

template <typename T, size_t DIM>
inline T EvaluationPoint<T, DIM>::getEstimatedGradient(int channel) const
{
    T gradient = absorbingBoundaryStatistics.getEstimatedGradient()[channel];
    gradient += absorbingBoundaryNormalAlignedStatistics.getEstimatedGradient()[channel];
    gradient += reflectingBoundaryStatistics.getEstimatedGradient()[channel];
    gradient += reflectingBoundaryNormalAlignedStatistics.getEstimatedGradient()[channel];
    gradient += sourceStatistics.getEstimatedGradient()[channel];

    return gradient;
}

template <typename T, size_t DIM>
inline void EvaluationPoint<T, DIM>::reset()
{
    absorbingBoundaryStatistics.reset();
    absorbingBoundaryNormalAlignedStatistics.reset();
    reflectingBoundaryStatistics.reset();
    reflectingBoundaryNormalAlignedStatistics.reset();
    sourceStatistics.reset();
}

template <typename T, size_t DIM>
inline BoundaryValueCachingDelta<T, DIM>::BoundaryValueCachingDelta(const GeometricQueries<DIM>& queries_,
                                                          const DeltaTrack<T, DIM>& walkOnStars_):
                                                          queries(queries_), walkOnStars(walkOnStars_)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::computeBoundaryEstimates(const PDEVar<T, DIM>& pde,
                                                                   const WalkSettings& walkSettings,
                                                                   int nWalksForSolutionEstimates,
                                                                   int nWalksForGradientEstimates,
                                                                   float robinCoeffCutoffForNormalDerivative,
                                                                   std::vector<SamplePoint<T, DIM>>& samplePts,
                                                                   bool useFiniteDifferences,
                                                                   bool runSingleThreaded,
                                                                   std::function<void(int,int)> reportProgress) const
{
    // initialize estimation quantities
    std::vector<int> nWalks;
    setEstimationData(pde, walkSettings, nWalksForSolutionEstimates,
                      nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                      useFiniteDifferences, nWalks, samplePts);

    // compute estimates
    walkOnStars.solve(pde, walkSettings, nWalks, samplePts, runSingleThreaded, reportProgress);

    // set estimated boundary data
    setEstimatedBoundaryData(pde, walkSettings, robinCoeffCutoffForNormalDerivative,
                             useFiniteDifferences, samplePts);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::setSourceValues(const PDEVar<T, DIM>& pde,
                                                          std::vector<SamplePoint<T, DIM>>& samplePts,
                                                          bool runSingleThreaded) const
{
    int nSamplePoints = (int)samplePts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nSamplePoints; i++) {
            samplePts[i].contribution = pde.source(samplePts[i].pt);
            samplePts[i].estimationQuantity = EstimationQuantity::None;
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                samplePts[i].contribution = pde.source(samplePts[i].pt);
                samplePts[i].estimationQuantity = EstimationQuantity::None;
            }
        };

        tbb::blocked_range<int> range(0, nSamplePoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::splat(const PDEVar<T, DIM>& pde,
                                                const SamplePoint<T, DIM>& samplePt,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                EvaluationPoint<T, DIM>& evalPt) const
{
    // std::cout<<"Inside Splat Function Outer"<<std::endl;
    // std::cout<<"Dist to boundary"<<evalPt.distToAbsorbingBoundary<<std::endl;
    // don't evaluate if the distance to the boundary is smaller than the cutoff distance
    if (evalPt.distToAbsorbingBoundary < cutoffDistToAbsorbingBoundary ||
        evalPt.distToReflectingBoundary < cutoffDistToReflectingBoundary) return;
    //std::cout<<"TOO CLOSE DID NOT ENTER HERE"<<std::endl;
    // initialize the greens function
    std::unique_ptr<GreensFnFreeSpace<DIM>> greensFn = nullptr;
    greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.bound);
    // if (pde.absorptionCoeff > 0.0f) {
    //     greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.bound);

    // } else {
    //     greensFn = std::make_unique<HarmonicGreensFnFreeSpace<DIM>>();
    // }

    greensFn->updatePole(evalPt.pt);

    // evaluate
    if (samplePt.type == SampleType::OnAbsorbingBoundary ||
        samplePt.type == SampleType::OnReflectingBoundary) {
        splatBoundaryData(samplePt, greensFn, radiusClamp, kernelRegularization,
                          robinCoeffCutoffForNormalDerivative, evalPt, pde);

    } else {
        splatSourceData(samplePt, greensFn, radiusClamp, kernelRegularization, evalPt, pde);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::splat(const PDEVar<T, DIM>& pde,
                                                const std::vector<SamplePoint<T, DIM>>& samplePts,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                EvaluationPoint<T, DIM>& evalPt) const
{
    // std::cout<<"Inside Splat Function"<<std::endl;
    // std::cout<<"Dist to boundary"<<evalPt.distToAbsorbingBoundary<<std::endl;
    // don't evaluate if the distance to the boundary is smaller than the cutoff distance
    if (evalPt.distToAbsorbingBoundary < cutoffDistToAbsorbingBoundary ||
        evalPt.distToReflectingBoundary < cutoffDistToReflectingBoundary) return;
    //std::cout<<"Inside function"<<std::endl;
    // initialize the greens function
    std::unique_ptr<GreensFnFreeSpace<DIM>> greensFn = nullptr;
    greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.bound);
    // if (pde.absorptionCoeff > 0.0f) {
    //     greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.bound);

    // } else {
    //     greensFn = std::make_unique<HarmonicGreensFnFreeSpace<DIM>>();
    // }

    greensFn->updatePole(evalPt.pt);

    // evaluate
    for (int i = 0; i < (int)samplePts.size(); i++) {
        if (samplePts[i].type == SampleType::OnAbsorbingBoundary ||
            samplePts[i].type == SampleType::OnReflectingBoundary) {
            splatBoundaryData(samplePts[i], greensFn, radiusClamp, kernelRegularization,
                              robinCoeffCutoffForNormalDerivative, evalPt, pde);

        } else {
            splatSourceData(samplePts[i], greensFn, radiusClamp, kernelRegularization, evalPt, pde);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::splat(const PDEVar<T, DIM>& pde,
                                                const SamplePoint<T, DIM>& samplePt,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                bool runSingleThreaded) const
{
    //std::cout<<"Inside Splat Function Multiple Evalpts"<<std::endl;
    
    int nEvalPoints = (int)evalPts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nEvalPoints; i++) {
            splat(pde, samplePt, radiusClamp, kernelRegularization,
                  robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                  cutoffDistToReflectingBoundary, evalPts[i]);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                splat(pde, samplePt, radiusClamp, kernelRegularization,
                      robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                      cutoffDistToReflectingBoundary, evalPts[i]);
            }
        };

        tbb::blocked_range<int> range(0, nEvalPoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::splat(const PDEVar<T, DIM>& pde,
                                                const std::vector<SamplePoint<T, DIM>>& samplePts,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                std::function<void(int, int)> reportProgress) const
{
    const int reportGranularity = 100;
    for (int i = 0; i < (int)samplePts.size(); i++) {
        splat(pde, samplePts[i], radiusClamp, kernelRegularization,
              robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
              cutoffDistToReflectingBoundary, evalPts);

        if (reportProgress && (i + 1)%reportGranularity == 0) {
            reportProgress(reportGranularity, 0);
        }
    }

    if (reportProgress) {
        reportProgress(samplePts.size()%reportGranularity, 0);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::estimateSolutionNearBoundary(const PDEVar<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       bool useDistanceToAbsorbingBoundary,
                                                                       float cutoffDistToBoundary, int nWalks,
                                                                       EvaluationPoint<T, DIM>& evalPt) const
{
    float distToBoundary = useDistanceToAbsorbingBoundary ? evalPt.distToAbsorbingBoundary :
                                                            evalPt.distToReflectingBoundary;
    if (distToBoundary < cutoffDistToBoundary) {
        // NOTE: When the evaluation pt is on the boundary, this setup
        // evaluates the inward boundary normal aligned solution
        SamplePoint<T, DIM> samplePt(evalPt.pt, evalPt.normal, evalPt.type,
                                     EstimationQuantity::Solution, 1.0f,
                                     evalPt.distToAbsorbingBoundary,
                                     evalPt.distToReflectingBoundary);
        //std::cout<<"Going to call walk on stars"<<std::endl;
        walkOnStars.solve(pde, walkSettings, nWalks, samplePt);

        // update statistics
        evalPt.reset();
        //std::cout<<"Obtained Statistics: "<<samplePt.statistics.getEstimatedSolution()<<std::endl;

        T solutionEstimate = samplePt.statistics.getEstimatedSolution() * std::sqrt(pde.diffusion(samplePt.pt));
        evalPt.absorbingBoundaryStatistics.addSolutionEstimate(solutionEstimate);

        // if (evalPt.type == SampleType::OnAbsorbingBoundary) {
        //     std::cout<<"Inside Absorbing boundary"<<std::endl;
        //     evalPt.absorbingBoundaryStatistics.addSolutionEstimate(solutionEstimate);

        // } else if (evalPt.type == SampleType::OnReflectingBoundary) {
        //     std::cout<<"Inside Reflecting boundary"<<std::endl;
        //     evalPt.reflectingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
        // }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::estimateSolutionNearBoundary(const PDEVar<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       bool useDistanceToAbsorbingBoundary,
                                                                       float cutoffDistToBoundary, int nWalks,
                                                                       std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                                       bool runSingleThreaded) const
{
    int nEvalPoints = (int)evalPts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nEvalPoints; i++) {
            estimateSolutionNearBoundary(pde, walkSettings, useDistanceToAbsorbingBoundary,
                                         cutoffDistToBoundary, nWalks, evalPts[i]);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                estimateSolutionNearBoundary(pde, walkSettings, useDistanceToAbsorbingBoundary,
                                             cutoffDistToBoundary, nWalks, evalPts[i]);
            }
        };

        tbb::blocked_range<int> range(0, nEvalPoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::setEstimationData(const PDEVar<T, DIM>& pde,
                                                            const WalkSettings& walkSettings,
                                                            int nWalksForSolutionEstimates,
                                                            int nWalksForGradientEstimates,
                                                            float robinCoeffCutoffForNormalDerivative,
                                                            bool useFiniteDifferences,
                                                            std::vector<int>& nWalks,
                                                            std::vector<SamplePoint<T, DIM>>& samplePts) const
{
    int nSamples = (int)samplePts.size();
    nWalks.resize(nSamples, 0);
    for (int i = 0; i < nSamples; i++) {
        SamplePoint<T, DIM>& samplePt = samplePts[i];

        if (samplePt.type == SampleType::OnAbsorbingBoundary) {
            if (useFiniteDifferences) {
                samplePt.type = SampleType::InDomain;
                samplePt.estimationQuantity = EstimationQuantity::Solution;

            } else {
                Vector<DIM> normal = samplePt.normal;
                if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
                    normal *= -1.0f;
                }

                samplePt.directionForDerivative = normal;
                samplePt.estimationQuantity = EstimationQuantity::SolutionAndGradient;
            }

            nWalks[i] = nWalksForGradientEstimates;

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            if (!pde.areRobinConditionsPureNeumann) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                samplePt.robinCoeff = pde.robinCoeff(samplePt.pt, samplePt.normal,
                                                     returnBoundaryNormalAlignedValue);
            }

            if (std::fabs(samplePt.robinCoeff) > robinCoeffCutoffForNormalDerivative) {
                Vector<DIM> normal = samplePt.normal;
                if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
                    normal *= -1.0f;
                }

                nWalks[i] = nWalksForGradientEstimates;
                samplePt.directionForDerivative = normal;
                samplePt.estimationQuantity = EstimationQuantity::SolutionAndGradient;

            } else {
                nWalks[i] = nWalksForSolutionEstimates;
                samplePt.estimationQuantity = EstimationQuantity::Solution;
            }

        } else {
            std::cerr << "BoundaryValueCachingDelta::setEstimationData(): Invalid sample type!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::setEstimatedBoundaryData(const PDEVar<T, DIM>& pde,
                                                                   const WalkSettings& walkSettings,
                                                                   float robinCoeffCutoffForNormalDerivative,
                                                                   bool useFiniteDifferences,
                                                                   std::vector<SamplePoint<T, DIM>>& samplePts) const
{
    for (int i = 0; i < (int)samplePts.size(); i++) {
        SamplePoint<T, DIM>& samplePt = samplePts[i];
        samplePt.solution = samplePt.statistics.getEstimatedSolution();

        if (samplePt.type == SampleType::OnReflectingBoundary) {
            if (!walkSettings.ignoreReflectingBoundaryContribution) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                if (pde.areRobinConditionsPureNeumann) {
                    samplePt.normalDerivative = pde.robin(samplePt.pt, samplePt.normal,
                                                          returnBoundaryNormalAlignedValue);

                } else {
                    samplePt.contribution = pde.robin(samplePt.pt, samplePt.normal,
                                                      returnBoundaryNormalAlignedValue);
                    if (std::fabs(samplePt.robinCoeff) > robinCoeffCutoffForNormalDerivative) {
                        samplePt.normalDerivative = samplePt.statistics.getEstimatedDerivative();
                    }
                }
            }

        } else {
            if (useFiniteDifferences) {
                // use biased gradient estimates
                float signedDistance = queries.computeDistToAbsorbingBoundary(samplePt.pt, true);
                Vector<DIM> pt = samplePt.pt - signedDistance*samplePt.normal;

                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        signedDistance > 0.0f;
                T dirichlet = !walkSettings.ignoreAbsorbingBoundaryContribution ?
                              std::sqrt(pde.diffusion(pt))*pde.dirichlet(pt, returnBoundaryNormalAlignedValue) : T(0.0f);

                samplePt.normalDerivative = dirichlet - samplePt.solution*std::sqrt(pde.diffusion(samplePt.pt));
                samplePt.normalDerivative /= std::fabs(signedDistance);
                samplePt.type = SampleType::OnAbsorbingBoundary;

            } else {
                // use unbiased gradient estimates
                samplePt.normalDerivative = samplePt.statistics.getEstimatedDerivative();
            }
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::splatBoundaryData(const SamplePoint<T, DIM>& samplePt,
                                                            const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                                                            float radiusClamp,
                                                            float kernelRegularization,
                                                            float robinCoeffCutoffForNormalDerivative,
                                                            EvaluationPoint<T, DIM>& evalPt, const PDEVar<T, DIM> &pde) const
{
    // compute the contribution of the boundary sample
    const T& solution = pde.dirichlet(samplePt.pt, false) * std::sqrt(pde.diffusion(samplePt.pt));
    const T& normalDerivative = samplePt.normalDerivative;
    const T& robin = samplePt.contribution;
    const Vector<DIM>& pt = samplePt.pt;
    Vector<DIM> n = samplePt.normal*(samplePt.estimateBoundaryNormalAligned ? -1.0f : 1.0f);
    float pdf = samplePt.pdf;
    float robinCoeff = samplePt.robinCoeff;

    float r = std::max(radiusClamp, (pt - greensFn->x).norm());
    float G = greensFn->evaluate(r);
    //G = G * (std::sqrt(pde.diffusion(greensFn->x)/pde.diffusion(pt))) * (pde.bound - pde.transformedAbsorption(greensFn->x));
    float P = greensFn->poissonKernel(r, pt, n);
    //P = P * std::sqrt(pde.diffusion(pt));
    Vector<DIM> dG = greensFn->gradient(r, pt);
    Vector<DIM> dP = greensFn->poissonKernelGradient(r, pt, n);
    float dGNorm = dG.norm();
    float dPNorm = dP.norm();

    if (std::isinf(G) || std::isinf(P) || std::isinf(dGNorm) || std::isinf(dPNorm) ||
        std::isnan(G) || std::isnan(P) || std::isnan(dGNorm) || std::isnan(dPNorm)) {
        return;
    }

    if (kernelRegularization > 0.0f) {
        r /= kernelRegularization;
        G *= KernelRegularization<DIM>::regularizationForGreensFn(r);
        P *= KernelRegularization<DIM>::regularizationForPoissonKernel(r);
    }

    T solutionEstimate;
    T gradientEstimate[DIM];
    float alpha = evalPt.type == SampleType::OnAbsorbingBoundary ||
                  evalPt.type == SampleType::OnReflectingBoundary ?
                  2.0f : 1.0f;

    if (std::fabs(robinCoeff) > robinCoeffCutoffForNormalDerivative) {
        
        solutionEstimate = alpha*((G + P/robinCoeff)*normalDerivative - P*robin/robinCoeff)/pdf;

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*((dG[i] + dP[i]/robinCoeff)*normalDerivative - dP[i]*robin/robinCoeff)/pdf;
        }

    } else if (std::fabs(robinCoeff) > 0.0f) {
        
        solutionEstimate = alpha*(G*robin - (P + robinCoeff*G)*solution)/pdf;

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*(dG[i]*robin - (dP[i] + robinCoeff*dG[i])*solution)/pdf;
        }

    } else {
        solutionEstimate = alpha*(G*normalDerivative - P*solution)/pdf;
        //solutionEstimate /= std::sqrt(pde.diffusion(samplePt.pt));

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*(dG[i]*normalDerivative - dP[i]*solution)/pdf;
        }
    }

    // update statistics
    if (samplePt.estimateBoundaryNormalAligned) {
        evalPt.absorbingBoundaryNormalAlignedStatistics.addSolutionEstimate(solutionEstimate);
        evalPt.absorbingBoundaryNormalAlignedStatistics.addGradientEstimate(gradientEstimate);
        // if (samplePt.type == SampleType::OnAbsorbingBoundary) {
        //     evalPt.absorbingBoundaryNormalAlignedStatistics.addSolutionEstimate(solutionEstimate);
        //     evalPt.absorbingBoundaryNormalAlignedStatistics.addGradientEstimate(gradientEstimate);

        // } else if (samplePt.type == SampleType::OnReflectingBoundary) {
        //     evalPt.reflectingBoundaryNormalAlignedStatistics.addSolutionEstimate(solutionEstimate);
        //     evalPt.reflectingBoundaryNormalAlignedStatistics.addGradientEstimate(gradientEstimate);
        // }

    } else {
        evalPt.absorbingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
        evalPt.absorbingBoundaryStatistics.addGradientEstimate(gradientEstimate);
        // if (samplePt.type == SampleType::OnAbsorbingBoundary) {
        //     evalPt.absorbingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
        //     evalPt.absorbingBoundaryStatistics.addGradientEstimate(gradientEstimate);

        // } else if (samplePt.type == SampleType::OnReflectingBoundary) {
        //     evalPt.reflectingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
        //     evalPt.reflectingBoundaryStatistics.addGradientEstimate(gradientEstimate);
        // }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingDelta<T, DIM>::splatSourceData(const SamplePoint<T, DIM>& samplePt,
                                                          const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                                                          float radiusClamp,
                                                          float kernelRegularization,
                                                          EvaluationPoint<T, DIM>& evalPt, const PDEVar<T, DIM>& pde) const
{
    // compute the contribution of the source sample
    //std::cout<<"Splatting Source Data"<<std::endl;
    const T& source =  (pde.bound - pde.transformedAbsorption(samplePt.pt))*(std::sqrt(pde.diffusion(samplePt.pt))*pde.dirichlet(samplePt.pt, false)) - pde.source(samplePt.pt) / std::sqrt(pde.diffusion(samplePt.pt)) ;
    //std::cout<<"Source value: "<<source<<std::endl;
    const Vector<DIM>& pt = samplePt.pt;
    float pdf = samplePt.pdf;

    float r = std::max(radiusClamp, (pt - greensFn->x).norm());
    float G = greensFn->evaluate(r);
    //G = G * (std::sqrt(pde.diffusion(greensFn->x)/pde.diffusion(pt))) * (pde.bound - pde.transformedAbsorption(greensFn->x));
    Vector<DIM> dG = greensFn->gradient(r, pt);
    float dGNorm = dG.norm();

    if (std::isinf(G) || std::isnan(G) || std::isinf(dGNorm) || std::isnan(dGNorm)) {
        return;
    }

    if (kernelRegularization > 0.0f) {
        r /= kernelRegularization;
        G *= KernelRegularization<DIM>::regularizationForGreensFn(r);
    }

    float alpha = evalPt.type == SampleType::OnAbsorbingBoundary ||
                  evalPt.type == SampleType::OnReflectingBoundary ?
                  2.0f : 1.0f;
    T solutionEstimate = alpha*G*source/pdf;

    //solutionEstimate /= std::sqrt(pde.diffusion(samplePt.pt)*pde.diffusion(pt));

    T gradientEstimate[DIM];
    if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
    for (int i = 0; i < DIM; i++) {
        gradientEstimate[i] = alpha*dG[i]*source/pdf;
    }

    // update statistics
    evalPt.sourceStatistics.addSolutionEstimate(solutionEstimate);
    evalPt.sourceStatistics.addGradientEstimate(gradientEstimate);
}

template <typename T, size_t DIM>
inline BoundaryValueCachingSolver<T, DIM>::BoundaryValueCachingSolver(const GeometricQueries<DIM>& queries_,
                                                                      std::shared_ptr<BoundarySampler<T, DIM>> absorbingBoundarySampler_,
                                                                      std::shared_ptr<BoundarySampler<T, DIM>> reflectingBoundarySampler_,
                                                                      std::shared_ptr<DomainSampler<T, DIM>> domainSampler_):
                                                                      queries(queries_),
                                                                      absorbingBoundarySampler(absorbingBoundarySampler_),
                                                                      reflectingBoundarySampler(reflectingBoundarySampler_),
                                                                      domainSampler(domainSampler_), walkOnStars(queries),
                                                                      boundaryValueCaching(queries, walkOnStars)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::generateSamples(int absorbingBoundaryCacheSize,
                                                                int reflectingBoundaryCacheSize,
                                                                int domainCacheSize,
                                                                float normalOffsetForAbsorbingBoundary,
                                                                float normalOffsetForReflectingBoundary,
                                                                bool solveDoubleSided)
{
    absorbingBoundarySampler->generateSamples(absorbingBoundarySampler->getSampleCount(absorbingBoundaryCacheSize, false),
                                              SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                              queries, absorbingBoundaryCache, false);
    if (solveDoubleSided) {
        absorbingBoundarySampler->generateSamples(absorbingBoundarySampler->getSampleCount(absorbingBoundaryCacheSize, true),
                                                  SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                                  queries, absorbingBoundaryCacheNormalAligned, true);
    }

    reflectingBoundarySampler->generateSamples(reflectingBoundarySampler->getSampleCount(reflectingBoundaryCacheSize, false),
                                               SampleType::OnReflectingBoundary, normalOffsetForReflectingBoundary,
                                               queries, reflectingBoundaryCache, false);
    if (solveDoubleSided) {
        reflectingBoundarySampler->generateSamples(reflectingBoundarySampler->getSampleCount(reflectingBoundaryCacheSize, true),
                                                   SampleType::OnReflectingBoundary, normalOffsetForReflectingBoundary,
                                                   queries, reflectingBoundaryCacheNormalAligned, true);
    }

    domainSampler->generateSamples(domainCacheSize, queries, domainCache);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::computeSampleEstimates(const PDEVar<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       int nWalksForSolutionEstimates,
                                                                       int nWalksForGradientEstimates,
                                                                       float robinCoeffCutoffForNormalDerivative,
                                                                       bool useFiniteDifferences,
                                                                       bool runSingleThreaded,
                                                                       std::function<void(int,int)> reportProgress)
{
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  absorbingBoundaryCache, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  absorbingBoundaryCacheNormalAligned, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  reflectingBoundaryCache, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  reflectingBoundaryCacheNormalAligned, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.setSourceValues(pde, domainCache, runSingleThreaded);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::splat(const PDEVar<T, DIM>& pde,
                                                     float radiusClamp,
                                                     float kernelRegularization,
                                                     float robinCoeffCutoffForNormalDerivative,
                                                     float cutoffDistToAbsorbingBoundary,
                                                     float cutoffDistToReflectingBoundary,
                                                     std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                     std::function<void(int, int)> reportProgress) const
{
    boundaryValueCaching.splat(pde, absorbingBoundaryCache, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, absorbingBoundaryCacheNormalAligned, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, reflectingBoundaryCache, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, reflectingBoundaryCacheNormalAligned, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, domainCache, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::estimateSolutionNearBoundary(const PDEVar<T, DIM>& pde,
                                                                             const WalkSettings& walkSettings,
                                                                             float cutoffDistToAbsorbingBoundary,
                                                                             float cutoffDistToReflectingBoundary,
                                                                             int nWalksForSolutionEstimates,
                                                                             std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                                             bool runSingleThreaded) const
{
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, true, cutoffDistToAbsorbingBoundary,
                                                      nWalksForSolutionEstimates, evalPts, runSingleThreaded);
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, false, cutoffDistToReflectingBoundary,
                                                      nWalksForSolutionEstimates, evalPts, runSingleThreaded);
}

template <typename T, size_t DIM>
inline const std::vector<SamplePoint<T, DIM>>& BoundaryValueCachingSolver<T, DIM>::getAbsorbingBoundaryCache(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? absorbingBoundaryCacheNormalAligned : absorbingBoundaryCache;
}

template <typename T, size_t DIM>
inline const std::vector<SamplePoint<T, DIM>>& BoundaryValueCachingSolver<T, DIM>::getReflectingBoundaryCache(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? reflectingBoundaryCacheNormalAligned : reflectingBoundaryCache;
}

template <typename T, size_t DIM>
inline const std::vector<SamplePoint<T, DIM>>& BoundaryValueCachingSolver<T, DIM>::getDomainCache() const
{
    return domainCache;
}

} // bvc

} // zombie
