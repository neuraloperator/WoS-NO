#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <zombie/point_estimation/walk_on_stars.h>
#include <zombie/boundary_value_caching/splatter.h>
#include <zombie/utils/progress.h>
#include "grid.h"
#include "scene.h"
#include "scene_3d_varying.h"
#include "scene_3d.h"
#include "pybind11_json.h"
#include <tuple>

using json = nlohmann::json;
namespace py = pybind11;

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>> runWalkOnStars3d_var(const Scene3DVar& scene, const json& solverConfig, const json& outputConfig, std::vector<std::vector<float>> sample_points) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", true);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);
	const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

	const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
	fcpw::BoundingBox<3> bbox = scene.bbox;
	const zombie::GeometricQueries<3>& queries = scene.queries;
	const zombie::PDEVar<float, 3>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;
	//std::cout<<"RUNNING WOS ALGORITHM "<<std::endl;
	// setup solution domain
	// std::vector<zombie::SamplePoint<float, 2>> samplePts;
	// createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, pts);

	std::vector<zombie::SamplePoint<float, 3>> samplePts;
	// std::vector<std::vector<float>> sample_points;
	createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, sample_points);

	std::vector<zombie::SampleEstimationData<3>> sampleEstimationData(samplePts.size());
	for (int i = 0; i < samplePts.size(); i++) {
		sampleEstimationData[i].nWalks = nWalks;
		sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt) || solveDoubleSided ?
													 zombie::EstimationQuantity::SolutionAndGradient:   //HRISHI MODIFIED USED TO BE SolutionAndGradient
													 zombie::EstimationQuantity::None;
	}

	// initialize solver and estimate solution
	Vector3 extent = bbox.pMax - bbox.pMin;
	// float scale1 = 1;
	// float scale2 = 1;
	// if (extent.x() > extent.y()) {
	// 	scale1 = extent.x()/extent.y();
	// }
	// else {
	// 	scale2 = extent.y()/extent.x();
	// }
	// ProgressBar pb(scale1 * gridRes * scale2 * gridRes);
	ProgressBar pb(sample_points.size());
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);

	zombie::WalkOnStars<float, 3> walkOnStars(queries);
	walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, runSingleThreaded, reportProgress);
	// pb.finish();

	// saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);

	std::vector<float> solution;
	solution = getSolution(samplePts, pde, queries, solveDoubleSided, outputConfig);
	std::vector<std::vector<float>> gradient;
	gradient = getGradient(samplePts, pde, queries, solveDoubleSided, outputConfig);


	// std::vector<std::vector<float>> output;
	// output.resize(samplePts.size());
	// for (int i = 0; i < samplePts.size(); ++i) {
	// 	// output[i].resize(3);
	// 	output[i] = {samplePts[i].pt[0], samplePts[i].pt[1], solution[i]};
	// }

	// for(auto i: samplePts)
	// 	std::cout<<i.pt[0]<<" "<<i.pt[1]<<"\n";

	return std::make_tuple(sample_points, solution, gradient);
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>> runWalkOnStars3d_sampled(const Scene3D& scene, const json& solverConfig, const json& outputConfig, std::vector<std::vector<float>> sample_points) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", true);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);
	const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

	const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
	fcpw::BoundingBox<3> bbox = scene.bbox;
	const zombie::GeometricQueries<3>& queries = scene.queries;
	const zombie::PDE<float, 3>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;
	//std::cout<<"RUNNING WOS ALGORITHM "<<std::endl;
	// setup solution domain
	// std::vector<zombie::SamplePoint<float, 2>> samplePts;
	// createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, pts);

	std::vector<zombie::SamplePoint<float, 3>> samplePts;
	// std::vector<std::vector<float>> sample_points;
	createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, sample_points);

	std::vector<zombie::SampleEstimationData<3>> sampleEstimationData(samplePts.size());
	for (int i = 0; i < samplePts.size(); i++) {
		sampleEstimationData[i].nWalks = nWalks;
		sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt) || solveDoubleSided ?
													 zombie::EstimationQuantity::SolutionAndGradient:   //HRISHI MODIFIED USED TO BE SolutionAndGradient
													 zombie::EstimationQuantity::None;
	}

	// initialize solver and estimate solution
	Vector3 extent = bbox.pMax - bbox.pMin;
	// float scale1 = 1;
	// float scale2 = 1;
	// if (extent.x() > extent.y()) {
	// 	scale1 = extent.x()/extent.y();
	// }
	// else {
	// 	scale2 = extent.y()/extent.x();
	// }
	// ProgressBar pb(scale1 * gridRes * scale2 * gridRes);
	ProgressBar pb(sample_points.size());
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);

	zombie::WalkOnStars<float, 3> walkOnStars(queries);
	walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, runSingleThreaded, reportProgress);
	// pb.finish();

	// saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);

	std::vector<float> solution;
	solution = getSolution(samplePts, pde, queries, solveDoubleSided, outputConfig);
	std::vector<std::vector<float>> gradient;
	gradient = getGradient(samplePts, pde, queries, solveDoubleSided, outputConfig);

	// std::vector<std::vector<float>> output;
	// output.resize(samplePts.size());
	// for (int i = 0; i < samplePts.size(); ++i) {
	// 	// output[i].resize(3);
	// 	output[i] = {samplePts[i].pt[0], samplePts[i].pt[1], solution[i]};
	// }

	// for(auto i: samplePts)
	// 	std::cout<<i.pt[0]<<" "<<i.pt[1]<<"\n";

	return std::make_tuple(sample_points, solution, gradient);
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>> runWalkOnStars_sampled(const Scene& scene, const json& solverConfig, const json& outputConfig, std::vector<std::vector<float>> sample_points, py::object py_gino_fn = py::none()) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", true);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);
	const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

	const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
	fcpw::BoundingBox<2> bbox = scene.bbox;
	const zombie::GeometricQueries<2>& queries = scene.queries;
	const zombie::PDE<float, 2>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;
	std::function<void(zombie::WalkState<float, 2>&)> terminalCallback;
	if(!py_gino_fn.is_none()) {
		
		py::function fn = py_gino_fn.cast<py::function>();
		terminalCallback = [fn](zombie::WalkState<float, 2> & state) {
			
			auto x = state.currentPt;
			float u_val;
			try {
					py::gil_scoped_acquire acquire; // Acquire the GIL before calling Python code
					u_val = fn(x[0], x[1]).cast<float>();
			} catch (const py::error_already_set &e) {
				// Handle the exception (e.g., log it, rethrow it, etc.)
				std::cerr << "Error in Python callback: " << e.what() << std::endl;
				std::cout<<" Segmentation Fault likely in the WOS call due to multithread deadlock "<<std::endl;
				u_val = 0.0f; // Default value in case of error
				// Optionally, you can rethrow the exception if you want to propagate it
				// throw;
			}
			{

				state.terminalContribution = u_val;
				
				if (std::isnan(u_val)) {
					std::cout << "Gino function returned NaN for point: " << x[0] << ", " << x[1] << std::endl;
				}
			}
		};
	}
	else{
		terminalCallback = [](zombie::WalkState<float, 2> & state) {
			state.terminalContribution = 0.0f; // Default behavior if no callback is provided
		};
	}

	std::vector<zombie::SamplePoint<float, 2>> samplePts;
	createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, sample_points);

	std::vector<zombie::SampleEstimationData<2>> sampleEstimationData(samplePts.size());
	for (int i = 0; i < samplePts.size(); i++) {
		sampleEstimationData[i].nWalks = nWalks;
		sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt) || solveDoubleSided ?
													 zombie::EstimationQuantity::SolutionAndGradient:   //HRISHI MODIFIED USED TO BE SolutionAndGradient
													 zombie::EstimationQuantity::None;
	}

	// initialize solver and estimate solution
	Vector2 extent = bbox.pMax - bbox.pMin;
	ProgressBar pb(sample_points.size());
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);

	if(!py_gino_fn.is_none()) {
		zombie::WalkOnStars<float, 2> walkOnStars(queries, terminalCallback);
		walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, runSingleThreaded, reportProgress);
	}
	else{
		zombie::WalkOnStars<float, 2> walkOnStars(queries);
		walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, runSingleThreaded, reportProgress);
	}

	std::vector<float> solution;
	solution = getSolution(samplePts, pde, queries, solveDoubleSided, outputConfig);
	std::vector<std::vector<float>> gradient;
	gradient = getGradient(samplePts, pde, queries, solveDoubleSided, outputConfig);

	return std::make_tuple(sample_points, solution, gradient);
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>, 
std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>>,
std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>>,
std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> > runBoundaryValueCaching_sampled(const Scene& scene, const json& solverConfig, const json& outputConfig,std::vector<std::vector<float>> sample_points,
	std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> boundaryCacheIn,
	std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> domainCacheIn,
	std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> boundaryCacheNormalAlignedIn,
	bool loadCache, 
	py::object py_gino_fn = py::none()) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool useFiniteDifferencesForBoundaryDerivatives = getOptional<bool>(solverConfig, "useFiniteDifferencesForBoundaryDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);
	const bool runParallel = getOptional<bool>(solverConfig, "runParallel", true);

	const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
	const int nWalksForCachedGradientEstimates = getOptional<int>(solverConfig, "nWalksForCachedGradientEstimates", 640);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int boundaryCacheSize = getOptional<int>(solverConfig, "boundaryCacheSize", 1024);
	const int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
	const float normalOffsetForCachedDirichletSamples = getOptional<float>(solverConfig, "normalOffsetForCachedDirichletSamples", 5.0f*epsilonShell);
	const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 1e-3f);
	
	const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

	fcpw::BoundingBox<2> bbox = scene.bbox;
	const zombie::GeometricQueries<2>& queries = scene.queries;
	const zombie::PDE<float, 2>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;
	std::function<void(zombie::WalkState<float, 2>&)> terminalCallback;
	if(!py_gino_fn.is_none()) {
		py::function fn = py_gino_fn.cast<py::function>();
		terminalCallback = [fn](zombie::WalkState<float, 2> & state) {
			auto x = state.currentPt;
			float u_val;
			{
					py::gil_scoped_acquire acquire; // Acquire the GIL before calling Python code
					u_val = fn(x[0], x[1]).cast<float>();
			}
			{

				state.terminalContribution = u_val;
				if (std::isnan(u_val)) {
					std::cout << "Gino function returned NaN for point: " << x[0] << ", " << x[1] << std::endl;
				}
			}
		};
	}

	// setup solution domain
	std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
		return !queries.outsideBoundingDomain(x);
	};
	std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
		return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x);
	};
	std::function<bool(const Vector2&)> onNeumannBoundary = [&scene](const Vector2 &x) -> bool {
		return scene.onNeumannBoundary(x);
	};

	std::vector<zombie::SamplePoint<float, 2>> boundaryCache;
	std::vector<zombie::SamplePoint<float, 2>> boundaryCacheNormalAligned;
	std::vector<zombie::SamplePoint<float, 2>> domainCache;
	std::vector<zombie::EvaluationPoint<float, 2>> evalPts;
	createEvaluationGrid(evalPts, queries, bbox.pMin, bbox.pMax, gridRes, sample_points);

	// initialize solver and generate samples
	zombie::WalkOnStars<float, 2>* walkOnStarsPtr = nullptr;
	zombie::WalkOnStars<float, 2> walkOnStarsStorage = [&]() -> zombie::WalkOnStars<float, 2> {
		if (!py_gino_fn.is_none()) {
			return zombie::WalkOnStars<float, 2>(queries, terminalCallback);
		} else {
			return zombie::WalkOnStars<float, 2>(queries);
		}
	}();
	zombie::WalkOnStars<float, 2>& walkOnStars = walkOnStarsStorage;
	zombie::BoundarySampler<float, 2> boundarySampler(scene.vertices, scene.segments, queries,
													  walkOnStars, insideSolveRegionBoundarySampler,
													  onNeumannBoundary);
	zombie::DomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
												  bbox.pMin, bbox.pMax, scene.getSolveRegionVolume());

	boundarySampler.initialize(normalOffsetForCachedDirichletSamples, solveDoubleSided);
	boundarySampler.generateSamples(boundaryCacheSize, normalOffsetForCachedDirichletSamples,
									solveDoubleSided, 0.0f, boundaryCache, boundaryCacheNormalAligned);
	if (!ignoreSource) domainSampler.generateSamples(pde, domainCacheSize, domainCache);

	// estimate solution on the boundary
	//int totalWork = 2.0*(boundaryCache.size() + boundaryCacheNormalAligned.size()) + domainCacheSize;
	//ProgressBar pb(totalWork);
	ProgressBar pb(sample_points.size());
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);
	
	if(py_gino_fn.is_none()) {
		boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
										nWalksForCachedGradientEstimates, boundaryCache,
										useFiniteDifferencesForBoundaryDerivatives,
										false, reportProgress);
		boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
										nWalksForCachedGradientEstimates, boundaryCacheNormalAligned,
										useFiniteDifferencesForBoundaryDerivatives,
										false, reportProgress);
	}
	else{
		boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
										nWalksForCachedGradientEstimates, boundaryCache,
										useFiniteDifferencesForBoundaryDerivatives,
										true, reportProgress);
		boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
										nWalksForCachedGradientEstimates, boundaryCacheNormalAligned,
										useFiniteDifferencesForBoundaryDerivatives,
										true, reportProgress);
	}

	std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string> boundaryCacheData;
	std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> boundaryCacheVector;
	if(loadCache){
		zombie::SamplePoint<float, 2> samplept = boundaryCache[0];
		for (auto i: boundaryCacheIn){
			std::cout << std::get<2>(i) <<std::endl;
			if(std::get<0>(i)[0]!=0 && std::get<0>(i)[1]!=0){ 
				samplept.pt = {std::get<0>(i)[0], std::get<0>(i)[1]};
				samplept.normal = {std::get<1>(i)[0], std::get<1>(i)[1]};
				samplept.pdf = std::get<2>(i);
				samplept.dirichletDist = std::get<3>(i);
				samplept.neumannDist = std::get<4>(i);
				samplept.firstSphereRadius = std::get<5>(i);
				samplept.solution = std::get<6>(i);
				samplept.estimateBoundaryNormalAligned = std::get<8>(i);
				samplept.normalDerivative = std::get<7>(i);
				if(std::get<9>(i) == "dirichlet")
					samplept.type = zombie::SampleType::OnDirichletBoundary;
				else if(std::get<9>(i) == "neumann")
					samplept.type = zombie::SampleType::OnNeumannBoundary;
				else if(std::get<9>(i) == "domain")
					samplept.type = zombie::SampleType::InDomain;
				boundaryCache.emplace_back(samplept);
			}
		}

		for (auto i: domainCacheIn){
			std::cout << std::get<2>(i) <<std::endl;
			if(std::get<0>(i)[0]!=0 && std::get<0>(i)[1]!=0){ 
				samplept.pt = {std::get<0>(i)[0], std::get<0>(i)[1]};
				samplept.normal = {std::get<1>(i)[0], std::get<1>(i)[1]};
				samplept.pdf = std::get<2>(i);
				samplept.dirichletDist = std::get<3>(i);
				samplept.neumannDist = std::get<4>(i);
				samplept.firstSphereRadius = std::get<5>(i);
				samplept.solution = std::get<6>(i);
				samplept.estimateBoundaryNormalAligned = std::get<8>(i);
				samplept.normalDerivative = std::get<7>(i);
				if(std::get<9>(i) == "dirichlet")
					samplept.type = zombie::SampleType::OnDirichletBoundary;
				else if(std::get<9>(i) == "neumann")
					samplept.type = zombie::SampleType::OnNeumannBoundary;
				else if(std::get<9>(i) == "domain")
					samplept.type = zombie::SampleType::InDomain;
				domainCache.emplace_back(samplept);
			}
		}
		for (auto i: boundaryCacheNormalAlignedIn){
			std::cout << std::get<2>(i) <<std::endl;
			if(std::get<0>(i)[0]!=0 && std::get<0>(i)[1]!=0){ 
				samplept.pt = {std::get<0>(i)[0], std::get<0>(i)[1]};
				samplept.normal = {std::get<1>(i)[0], std::get<1>(i)[1]};
				samplept.pdf = std::get<2>(i);
				samplept.dirichletDist = std::get<3>(i);
				samplept.neumannDist = std::get<4>(i);
				samplept.firstSphereRadius = std::get<5>(i);
				samplept.solution = std::get<6>(i);
				samplept.estimateBoundaryNormalAligned = std::get<8>(i);
				samplept.normalDerivative = std::get<7>(i);
				if(std::get<9>(i) == "dirichlet")
					samplept.type = zombie::SampleType::OnDirichletBoundary;
				else if(std::get<9>(i) == "neumann")
					samplept.type = zombie::SampleType::OnNeumannBoundary;
				else if(std::get<9>(i) == "domain")
					samplept.type = zombie::SampleType::InDomain;
				boundaryCacheNormalAligned.emplace_back(samplept);
			}
		}

	}
	for (auto i: boundaryCache){
		
		if(i.type == zombie::SampleType::OnDirichletBoundary)
			boundaryCacheData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "dirichlet");
		else if(i.type == zombie::SampleType::OnNeumannBoundary)
			boundaryCacheData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "neumann");
		else if(i.type == zombie::SampleType::InDomain)
			boundaryCacheData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "domain");
		boundaryCacheVector.emplace_back(boundaryCacheData);
	}
	std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string> domainCacheData;
	std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> domainCacheVector;
	for (auto i: domainCache){
		if(i.type == zombie::SampleType::OnDirichletBoundary)
			domainCacheData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "dirichlet");
		else if(i.type == zombie::SampleType::OnNeumannBoundary)
			domainCacheData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "neumann");
		else if(i.type == zombie::SampleType::InDomain)
			domainCacheData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "domain");
		domainCacheVector.emplace_back(domainCacheData);
	}

	std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string> boundaryCacheNormalAlignedData;
	std::vector<std::tuple<std::vector<float>, std::vector<float>, float, float, float, float, float, float, bool, std::string>> boundaryCacheNormalAlignedVector;
	
	for (auto i: boundaryCacheNormalAligned){
		if(i.type == zombie::SampleType::OnDirichletBoundary)
			boundaryCacheNormalAlignedData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "dirichlet");
		else if(i.type == zombie::SampleType::OnNeumannBoundary)
			boundaryCacheNormalAlignedData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution, i.normalDerivative, i.estimateBoundaryNormalAligned, "neumann");
		else if(i.type == zombie::SampleType::InDomain)
			boundaryCacheNormalAlignedData = std::make_tuple(std::vector<float>{i.pt.x(), i.pt.y()}, std::vector<float>{i.normal.x(), i.normal.y()}, i.pdf, i.dirichletDist, i.neumannDist, i.firstSphereRadius, i.solution,i.normalDerivative, i.estimateBoundaryNormalAligned, "domain");
		boundaryCacheNormalAlignedVector.emplace_back(boundaryCacheNormalAlignedData);
	}

	// splat solution to evaluation points
	zombie::Splatter<float, 2> splatter(queries, walkOnStars);
	if(py_gino_fn.is_none()) {
	
		splatter.splat(pde, boundaryCache, radiusClampForKernels, regularizationForKernels,
					normalOffsetForCachedDirichletSamples, evalPts, false, reportProgress);
		splatter.splat(pde, boundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
					normalOffsetForCachedDirichletSamples, evalPts, false, reportProgress);
		splatter.splat(pde, domainCache, radiusClampForKernels, regularizationForKernels,
					normalOffsetForCachedDirichletSamples, evalPts, false, reportProgress);
		splatter.estimatePointwiseNearDirichletBoundary(pde, walkSettings, normalOffsetForCachedDirichletSamples,
														nWalksForCachedSolutionEstimates, evalPts, false);
	}
	else{
		splatter.splat(pde, boundaryCache, radiusClampForKernels, regularizationForKernels,
			normalOffsetForCachedDirichletSamples, evalPts, true, reportProgress);
		splatter.splat(pde, boundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
					normalOffsetForCachedDirichletSamples, evalPts, true, reportProgress);
		splatter.splat(pde, domainCache, radiusClampForKernels, regularizationForKernels,
					normalOffsetForCachedDirichletSamples, evalPts, true, reportProgress);
		splatter.estimatePointwiseNearDirichletBoundary(pde, walkSettings, normalOffsetForCachedDirichletSamples,
														nWalksForCachedSolutionEstimates, evalPts, true);
	}
	pb.finish();

	// save to file
	//saveEvaluationGrid(evalPts, pde, queries, scene.isDoubleSided, outputConfig);
	std::vector<float> solution;
	solution = getSolution_bvc(evalPts, pde, queries, solveDoubleSided, outputConfig);
	std::vector<std::vector<float>> gradient;
	gradient = getGradient_bvc(evalPts, pde, queries, solveDoubleSided, outputConfig);
	return std::make_tuple(sample_points, solution, gradient, boundaryCacheVector, boundaryCacheNormalAlignedVector, domainCacheVector);
}



void runWalkOnStars(const Scene& scene, const json& solverConfig, const json& outputConfig) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);

	const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

	fcpw::BoundingBox<2> bbox = scene.bbox;
	const zombie::GeometricQueries<2>& queries = scene.queries;
	const zombie::PDE<float, 2>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;

	// setup solution domain
	std::vector<zombie::SamplePoint<float, 2>> samplePts;
	createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes);

	std::vector<zombie::SampleEstimationData<2>> sampleEstimationData(samplePts.size());
	for (int i = 0; i < samplePts.size(); i++) {
		sampleEstimationData[i].nWalks = nWalks;
		sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt) || solveDoubleSided ?
													 zombie::EstimationQuantity::Solution:
													 zombie::EstimationQuantity::None;
	}

	// initialize solver and estimate solution
	ProgressBar pb(gridRes*gridRes);
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);

	zombie::WalkOnStars<float, 2> walkOnStars(queries);
	walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, false, reportProgress);
	pb.finish();

	// save to file
	saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);
}

void runBoundaryValueCaching(const Scene& scene, const json& solverConfig, const json& outputConfig) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool useFiniteDifferencesForBoundaryDerivatives = getOptional<bool>(solverConfig, "useFiniteDifferencesForBoundaryDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);

	const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
	const int nWalksForCachedGradientEstimates = getOptional<int>(solverConfig, "nWalksForCachedGradientEstimates", 640);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int boundaryCacheSize = getOptional<int>(solverConfig, "boundaryCacheSize", 1024);
	const int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
	const float normalOffsetForCachedDirichletSamples = getOptional<float>(solverConfig, "normalOffsetForCachedDirichletSamples", 5.0f*epsilonShell);
	const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 1e-3f);
	const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

	fcpw::BoundingBox<2> bbox = scene.bbox;
	const zombie::GeometricQueries<2>& queries = scene.queries;
	const zombie::PDE<float, 2>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;

	// setup solution domain
	std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
		return !queries.outsideBoundingDomain(x);
	};
	std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
		return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x);
	};
	std::function<bool(const Vector2&)> onNeumannBoundary = [&scene](const Vector2 &x) -> bool {
		return scene.onNeumannBoundary(x);
	};

	std::vector<zombie::SamplePoint<float, 2>> boundaryCache;
	std::vector<zombie::SamplePoint<float, 2>> boundaryCacheNormalAligned;
	std::vector<zombie::SamplePoint<float, 2>> domainCache;
	std::vector<zombie::EvaluationPoint<float, 2>> evalPts;
	createEvaluationGrid(evalPts, queries, bbox.pMin, bbox.pMax, gridRes);

	// initialize solver and generate samples
	zombie::WalkOnStars<float, 2> walkOnStars(queries);
	zombie::BoundarySampler<float, 2> boundarySampler(scene.vertices, scene.segments, queries,
													  walkOnStars, insideSolveRegionBoundarySampler,
													  onNeumannBoundary);
	zombie::DomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
												  bbox.pMin, bbox.pMax, scene.getSolveRegionVolume());

	boundarySampler.initialize(normalOffsetForCachedDirichletSamples, solveDoubleSided);
	boundarySampler.generateSamples(boundaryCacheSize, normalOffsetForCachedDirichletSamples,
									solveDoubleSided, 0.0f, boundaryCache, boundaryCacheNormalAligned);
	if (!ignoreSource) domainSampler.generateSamples(pde, domainCacheSize, domainCache);

	// estimate solution on the boundary
	int totalWork = 2.0*(boundaryCache.size() + boundaryCacheNormalAligned.size()) + domainCacheSize;
	ProgressBar pb(totalWork);
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);
	boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
									 nWalksForCachedGradientEstimates, boundaryCache,
									 useFiniteDifferencesForBoundaryDerivatives,
									 false, reportProgress);
	boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
									 nWalksForCachedGradientEstimates, boundaryCacheNormalAligned,
									 useFiniteDifferencesForBoundaryDerivatives,
									 false, reportProgress);

	// splat solution to evaluation points
	zombie::Splatter<float, 2> splatter(queries, walkOnStars);
	splatter.splat(pde, boundaryCache, radiusClampForKernels, regularizationForKernels,
				   normalOffsetForCachedDirichletSamples, evalPts, false, reportProgress);
	splatter.splat(pde, boundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
				   normalOffsetForCachedDirichletSamples, evalPts, false, reportProgress);
	splatter.splat(pde, domainCache, radiusClampForKernels, regularizationForKernels,
				   normalOffsetForCachedDirichletSamples, evalPts, false, reportProgress);
	splatter.estimatePointwiseNearDirichletBoundary(pde, walkSettings, normalOffsetForCachedDirichletSamples,
													nWalksForCachedSolutionEstimates, evalPts, false);
	pb.finish();

	// save to file
	saveEvaluationGrid(evalPts, pde, queries, scene.isDoubleSided, outputConfig);
}

int main(int argc, const char *argv[]) {
	if (argc != 2) {
		std::cerr << "must provide config filename" << std::endl;
		abort();
	}

	std::ifstream configFile(argv[1]);
	if (!configFile.is_open()) {
		std::cerr << "Error opening file: " << argv[1] << std::endl;
		return 1;
	}

	json config = json::parse(configFile);
	const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
	const json sceneConfig = getRequired<json>(config, "scene");
	const json solverConfig = getRequired<json>(config, "solver");
	const json outputConfig = getRequired<json>(config, "output");

	std::vector<float> mu1{-1.0, 1.0};
	std::vector<float> mu2{-0.0, 1.4};
	std::vector<float> beta{3.4, 2.1};
	std::vector<float> r{-0.4, -0.2, 1.0, 3.5, 3.0};

	Scene scene(sceneConfig, mu1, mu2, beta, r);
	if (solverType == "wost") {
		runWalkOnStars(scene, solverConfig, outputConfig);

	} else if (solverType == "bvc") {
		runBoundaryValueCaching(scene, solverConfig, outputConfig);
	}
}


PYBIND11_MODULE(zombie_bindings, m) {
    m.doc() = "pybind11 WoSt"; // optional module docstring
    m.def("wost", &runWalkOnStars_sampled, py::arg("scene"), py::arg("solverConfig"), py::arg("outputConfig"), py::arg("sample_points"), py::arg("py_gino_fn") = py::none(),
	"Function to run Walk on Stars with sampled points");
	m.def("wost_3dvar", &runWalkOnStars3d_var);
	m.def("bvc", &runBoundaryValueCaching_sampled, py::arg("scene"), py::arg("solverConfig"), py::arg("outputConfig"), py::arg("sample_points"), py::arg("boundaryCacheIn"),
	py::arg("domainCacheIn"), py::arg("boundaryCacheNormalAlignedIn"), py::arg("loadCache"), py::arg("py_gino_fn") = py::none(), "Function to run boundary value caching");
	
	py::class_<Scene>(m, "Scene")
        .def(py::init<json &, std::vector<float> &, std::vector<float> &, std::vector<float>&, std::vector<float> &>())
		.def(py::init<json &, std::vector<std::vector<float>> &, std::vector<float> &, std::vector<float> &, std::vector<float>&, std::vector<float> &>())
		.def(py::init<json &, std::vector<float> &, std::vector<float> &, std::vector<float>&, std::vector<float> &, std::vector<std::string> &>());

	py::class_<Scene3DVar>(m, "Scene3DVar")
        .def(py::init<json &, std::vector<float> &, std::vector<float> &, std::vector<float>&, std::vector<float> &>())
		.def(py::init<json &, std::vector<std::vector<float>> &, std::vector<float> &, std::vector<float> &, std::vector<float>&, std::vector<float> &>())
		.def(py::init<json &, std::vector<float> &, std::vector<float> &, std::vector<float>&, std::vector<float> &, std::vector<std::string> &>());
}