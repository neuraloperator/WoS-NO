// This file defines a ModelProblem class, which is used to describe a scalar-valued
// Poisson or screened Poisson PDE on a 2D domain via a boundary mesh, associated
// boundary conditions, source term, and robin and absorption coefficients.
//
// The boundary mesh is read from an OBJ file, while the input PDE data is read
// from images for the purposes of this demo. NOTE: Users may analogously define
// a ModelProblem class for 3D domains and/or vector-valued PDEs, as all functionality
// in Zombie is templated on the dimension and value type of the PDE.

#pragma once

#include <zombie/zombie.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

class ModelProblem3D {
public:
    // constructor
    ModelProblem3D(const json& config, std::string directoryPath);

    // members
    bool solveDoubleSided;
    std::vector<Vector3i> indices;
    std::vector<Vector3i> absorbingBoundaryIndices;
    std::vector<Vector3i> reflectingBoundaryIndices;
    std::vector<Vector3> positions;
    std::vector<Vector3> absorbingBoundaryPositions;
    std::vector<Vector3> reflectingBoundaryPositions;
    std::pair<Vector3, Vector3> boundingBox;
    zombie::PDE<float, 3> pde;
    zombie::GeometricQueries<3> queries;

protected:
    // loads a boundary mesh from an OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // sets up the PDE
    void setupPDE();

    // partitions the boundary mesh into absorbing and reflecting parts
    void partitionBoundaryMesh();

    // populates geometric queries for the absorbing and reflecting boundary
    void populateGeometricQueries();

    // members
    Image<1> isReflectingBoundary;
    Image<1> absorbingBoundaryValue;
    Image<1> reflectingBoundaryValue;
    //Image<1> sourceValue;
    bool domainIsWatertight;
    bool useSdfForAbsorbingBoundary;
    int sdfGridResolution;
    float robinCoeff, absorptionCoeff;
    std::vector<float> minRobinCoeffValues;
    std::vector<float> maxRobinCoeffValues;
    std::unique_ptr<zombie::SdfGrid<3>> sdfGridForAbsorbingBoundary;
    zombie::FcpwDirichletBoundaryHandler<3> absorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<3> reflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<3> reflectingRobinBoundaryHandler;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

ModelProblem3D::ModelProblem3D(const json& config, std::string directoryPath):
sdfGridForAbsorbingBoundary(nullptr)
{
    // load config settings
    std::string geometryFile = directoryPath + getRequired<std::string>(config, "geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    isReflectingBoundary = Image<1>(directoryPath + getRequired<std::string>(config, "isReflectingBoundary"));
    absorbingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "absorbingBoundaryValue"));
    reflectingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "reflectingBoundaryValue"));
    sourceValue = Image<1>(directoryPath + getRequired<std::string>(config, "sourceValue"));
    solveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    domainIsWatertight = getOptional<bool>(config, "domainIsWatertight", true);
    useSdfForAbsorbingBoundary = getOptional<bool>(config, "useSdfForAbsorbingBoundary", false);
    sdfGridResolution = getOptional<int>(config, "sdfGridResolution", 128);
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

    // load a boundary mesh from an OBJ file
    loadOBJ(geometryFile, normalize, flipOrientation);

    // setup the PDE
    setupPDE();

    // partition the boundary mesh into absorbing and reflecting boundary elements
    partitionBoundaryMesh();

    // specify the minimum and maximum Robin coefficient values for each reflecting boundary element:
    // we use a constant value for all elements in this demo, but Zombie supports variable coefficients
    minRobinCoeffValues.resize(reflectingBoundaryIndices.size(), std::fabs(robinCoeff));
    maxRobinCoeffValues.resize(reflectingBoundaryIndices.size(), std::fabs(robinCoeff));

    // populate the geometric queries for the absorbing and reflecting boundary
    populateGeometricQueries();
}

void ModelProblem3D::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation)
{
    zombie::loadBoundaryMesh<3>(filename, positions, indices);
    if (normalize) zombie::normalize<3>(positions);
    if (flipOrientation) zombie::flipOrientation<3>(indices);
    boundingBox = zombie::computeBoundingBox<3>(positions, true, 1.0);
}

void ModelProblem3D::setupPDE()
{
    Vector3 bMin = boundingBox.first;
    Vector3 bMax = boundingBox.second;
    float maxLength = (bMax - bMin).maxCoeff();

    pde.source = [this, bMin, maxLength](const Vector3& x) -> float {
        Vector3 uv = (x - bMin)/maxLength;
        return 0.0;
    };
    pde.dirichlet = [this, bMin, maxLength](const Vector3& x, bool _) -> float {
        Vector3 uv = (x - bMin)/maxLength;
        return 0.0;
    };
    pde.robin = [this, bMin, maxLength](const Vector3& x, const Vector3& n, bool _) -> float {
        Vector3 uv = (x - bMin)/maxLength;
        return 0.0;
    };
    pde.robinCoeff = [this](const Vector3& x, const Vector3& n, bool _) -> float {
        return 0.0;
    };
    pde.hasReflectingBoundaryConditions = [this, bMin, maxLength](const Vector3& x) -> bool {
        Vector3 uv = (x - bMin)/maxLength;
        return true;
    };
    pde.areRobinConditionsPureNeumann = robinCoeff == 0.0f;
    pde.areRobinCoeffsNonnegative = robinCoeff >= 0.0f;
    pde.absorptionCoeff = absorptionCoeff;
}

void ModelProblem3D::partitionBoundaryMesh()
{
    // use Zombie's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    zombie::partitionBoundaryMesh<3>(pde.hasReflectingBoundaryConditions, positions, indices,
                                     absorbingBoundaryPositions, absorbingBoundaryIndices,
                                     reflectingBoundaryPositions, reflectingBoundaryIndices);
}

void ModelProblem3D::populateGeometricQueries()
{
    // set the domain extent for geometric queries
    queries.domainIsWatertight = domainIsWatertight;
    queries.domainMin = boundingBox.first;
    queries.domainMax = boundingBox.second;

    // use an absorbing boundary handler to populate geometric queries for the absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryPositions, absorbingBoundaryIndices);
    zombie::populateGeometricQueriesForDirichletBoundary<3>(absorbingBoundaryHandler, queries);

    if (!solveDoubleSided && useSdfForAbsorbingBoundary) {
        // override distance queries to use an SDF grid. The user can also use Zombie to build
        // an SDF hierarchy for double-sided problems (ommited here for simplicity)
        sdfGridForAbsorbingBoundary = std::make_unique<zombie::SdfGrid<3>>(queries.domainMin, queries.domainMax);
        zombie::Vector3i sdfGridShape = zombie::Vector3i::Constant(sdfGridResolution);
        zombie::populateSdfGrid<3>(absorbingBoundaryHandler, *sdfGridForAbsorbingBoundary, sdfGridShape);
        zombie::populateGeometricQueriesForDirichletBoundary<zombie::SdfGrid<3>, 3>(*sdfGridForAbsorbingBoundary, queries);
    }

    // use a reflecting boundary handler to populate geometric queries for the reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette = zombie::getIgnoreCandidateSilhouetteCallback(solveDoubleSided);
    std::function<float(float)> branchTraversalWeight = zombie::getBranchTraversalWeightCallback();

    if (pde.areRobinConditionsPureNeumann) {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette);
        zombie::populateGeometricQueriesForNeumannBoundary<3>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForRobinBoundary<3>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);
    }
}
