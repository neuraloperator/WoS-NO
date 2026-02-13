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
#include <cfloat>
#include "config.h"
#include "image.h"

float laplacian(const std::vector<std::vector<float>>& img, int i, int j) {
    int height = img.size();
    int width = img[0].size();
    float h = 1.0f;  // pixel spacing

    // Handle boundary pixels with zero Neumann (zero-gradient) or simple replication
    int i_up = std::max(i - 1, 0);
    int i_down = std::min(i + 1, height - 1);
    int j_left = std::max(j - 1, 0);
    int j_right = std::min(j + 1, width - 1);

    float lap = (img[i_up][j] + img[i_down][j] + img[i][j_left] + img[i][j_right] - 4.0f * img[i][j]) / (h * h);
    return lap;
}

template<typename T, int DIM>
float compute_bound(const zombie::PDEVar<T, DIM> pde, const zombie::Vector<DIM>& bmin, const zombie::Vector<DIM>& bmax) {
    float boundMin = FLT_MAX, boundMax = -FLT_MAX;
    zombie::Vector<DIM> p;
    for(int i = 0; i < 10000; i++) {
        p = bmin.array() + zombie::sampleUniformVector<DIM>().array() * (bmax - bmin).array();
        auto v = pde.transformedAbsorption(p);
        boundMin = std::min(boundMin, v);
        boundMax = std::max(boundMax, v);
    }
    std::cout<<"BOund MAX: "<<boundMax<<std::endl;
    std::cout<<"Bound MIN: "<<boundMin<<std::endl;
    std::cout<<"Point: "<<p<<std::endl;

    return boundMax - boundMin;
}

template <typename T, size_t DIM>
class ModelProblem {
public:
    // constructor
    ModelProblem<T, DIM>(const json& config, std::string directoryPath, std::vector<std::vector<std::vector<float>>>sourceIm);
    ModelProblem<T, DIM>(const json& config, std::string directoryPath, std::vector<float>absorption, std::vector<float>diffusion, std::vector<float>dirichlet, std::vector<float>general);
    using Int32   = int32_t;
    using Vectorint = Eigen::Matrix<int, DIM, 1>;;
    using VectorD = Eigen::Matrix<T, DIM, 1>;
    using VectorDa = Eigen::Array<T, DIM, 1>;
    // members
    bool solveDoubleSided;
    std::vector<Vectorint> indices;
    std::vector<Vectorint> absorbingBoundaryIndices;
    std::vector<Vectorint> reflectingBoundaryIndices;
    std::vector<VectorD> positions;
    std::vector<VectorD> absorbingBoundaryPositions;
    std::vector<VectorD> reflectingBoundaryPositions;
    std::pair<VectorD, VectorD> boundingBox;
    zombie::PDE<float, DIM> pde;
    zombie::PDEVar<float, DIM> pdeVar;
    zombie::GeometricQueries<DIM> queries;
    int h;
	int w;
    int l;
	bool is_recursive;
	int x_start, x_end, y_start, y_end;
    std::vector<std::vector<std::vector<float>>> sourceValue;
    std::vector<float> absorption;
	std::vector<float> diffusion;
	std::vector<float> dirichlet;
	std::vector<float> general;


protected:
    // loads a boundary mesh from an OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // sets up the PDE
    void setupPDE();
    void setupPDEVar();

    // partitions the boundary mesh into absorbing and reflecting parts
    void partitionBoundaryMesh();
    void partitionBoundaryMeshVar();

    // populates geometric queries for the absorbing and reflecting boundary
    void populateGeometricQueries();
    void populateGeometricQueriesVar();

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
    std::unique_ptr<zombie::SdfGrid<DIM>> sdfGridForAbsorbingBoundary;
    zombie::FcpwDirichletBoundaryHandler<DIM> absorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<DIM> reflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<DIM> reflectingRobinBoundaryHandler;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
template <typename T, size_t DIM>
ModelProblem<T, DIM>::ModelProblem(const json& config, std::string directoryPath, std::vector<std::vector<std::vector<float>>>sourceIm):
sdfGridForAbsorbingBoundary(nullptr)
{
    // load config settings
    std::string geometryFile = directoryPath + getRequired<std::string>(config, "geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    isReflectingBoundary = Image<1>(directoryPath + getRequired<std::string>(config, "isReflectingBoundary"));
    absorbingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "absorbingBoundaryValue"));
    reflectingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "reflectingBoundaryValue"));
    //sourceValue = Image<1>(directoryPath + getRequired<std::string>(config, "sourceValue"));
    solveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    domainIsWatertight = getOptional<bool>(config, "domainIsWatertight", false);
    useSdfForAbsorbingBoundary = getOptional<bool>(config, "useSdfForAbsorbingBoundary", false);
    sdfGridResolution = getOptional<int>(config, "sdfGridResolution", 128);
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

    sourceValue = sourceIm;

	//int h = sourceValue_mat.size();
	//int w = sourceValue_mat[0].size();
	// isNeumann = std::make_shared<Image<1>>(isNeumannFile);
	//isNeumann = std::make_shared<Image<1>>(h, w, 1.0);
	//dirichletBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
	//neumannBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
	//sourceValue = std::make_shared<Image<1>>(sourceValue_mat);
	this->h = sourceValue.size();
	this->w = sourceValue[0].size();
    this->l = sourceValue[0][0].size();

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

template <typename T, size_t DIM>
ModelProblem<T, DIM>::ModelProblem(const json& config, std::string directoryPath, std::vector<float>absorption, std::vector<float>diffusion, std::vector<float>dirichlet, std::vector<float>general):
sdfGridForAbsorbingBoundary(nullptr)
{
    std::cout<<"Inside Model Problem Constructor"<<std::endl;
    // load config settings
    std::string geometryFile = directoryPath + getRequired<std::string>(config, "geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    isReflectingBoundary = Image<1>(directoryPath + getRequired<std::string>(config, "isReflectingBoundary"));
    absorbingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "absorbingBoundaryValue"));
    reflectingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "reflectingBoundaryValue"));
    //sourceValue = Image<1>(directoryPath + getRequired<std::string>(config, "sourceValue"));
    solveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    domainIsWatertight = getOptional<bool>(config, "domainIsWatertight", false);
    useSdfForAbsorbingBoundary = getOptional<bool>(config, "useSdfForAbsorbingBoundary", false);
    sdfGridResolution = getOptional<int>(config, "sdfGridResolution", 128);
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);
    std::cout<<"Defined input variables"<<std::endl;
	//int h = sourceValue_mat.size();
	//int w = sourceValue_mat[0].size();
	// isNeumann = std::make_shared<Image<1>>(isNeumannFile);
	//isNeumann = std::make_shared<Image<1>>(h, w, 1.0);
	//dirichletBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
	//neumannBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
	//sourceValue = std::make_shared<Image<1>>(sourceValue_mat);

    this->absorption = absorption;
    this->diffusion = diffusion;
    this->dirichlet = dirichlet;
    this->general = general;
    // load a boundary mesh from an OBJ file
    loadOBJ(geometryFile, normalize, flipOrientation);


    // setup the PDE
    setupPDEVar();


    // partition the boundary mesh into absorbing and reflecting boundary elements
    partitionBoundaryMeshVar();


    // specify the minimum and maximum Robin coefficient values for each reflecting boundary element:
    // we use a constant value for all elements in this demo, but Zombie supports variable coefficients
    minRobinCoeffValues.resize(reflectingBoundaryIndices.size(), std::fabs(robinCoeff));
    maxRobinCoeffValues.resize(reflectingBoundaryIndices.size(), std::fabs(robinCoeff));


    // populate the geometric queries for the absorbing and reflecting boundary
    populateGeometricQueriesVar();

}

template <typename T, size_t DIM>
void ModelProblem<T, DIM>::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation)
{
    zombie::loadBoundaryMesh<DIM>(filename, positions, indices);
    if (normalize) zombie::normalize<DIM>(positions);
    if (flipOrientation) zombie::flipOrientation<DIM>(indices);
    boundingBox = zombie::computeBoundingBox<DIM>(positions, true, 1.0);
}
template <typename T, size_t DIM>
void ModelProblem<T, DIM>::setupPDE()
{
    VectorD bMin = boundingBox.first;
    VectorD bMax = boundingBox.second;
    float maxLength = (bMax - bMin).maxCoeff();

    pde.source = [this, bMin, maxLength](const VectorD& x) -> float {
            int x_coord, y_coord, z_coord;
			x_coord = std::round(x[0]*19);
			y_coord = std::round((x[1])*19);
            z_coord = std::round((x[2])*19);
            
            // std::cout<<"X_coord: "<<x_coord<<std::endl;
            // std::cout<<"Y COORD: "<<y_coord<<std::endl;
            // std::cout<<"Z Coord: "<<z_coord<<std::endl;
			//std::cout<<"X COORD: "<<x_coord<< " Y COORD: "<<y_coord<<std::endl;
			//x_coord = std::clamp(x_coord, 0, 19);
			//y_coord = std::clamp(y_coord, 0, 19);
            //z_coord = std::clamp(z_coord, 0, 19);
			//std::cout<<"Inside Second recursion source"<<std::endl;
			//std::cout<<"X COORD: "<<x_coord<< " Y COORD: "<<y_coord<<std::endl;
			
			return sourceValue[y_coord][x_coord][z_coord];
    };
    pde.dirichlet = [this, bMin, maxLength](const VectorD& x, bool _) -> float {
        //std::cout<<"On Dirichlet"<<std::endl;
        return 0.0;
    };
    pde.robin = [this, bMin, maxLength](const VectorD& x, const VectorD& n, bool _) -> float {
        VectorD uv = (x - bMin)/maxLength;
        return 0.0;
    };
    pde.robinCoeff = [this](const VectorD& x, const VectorD& n, bool _) -> float {
        return 0.0;
    };
    pde.hasReflectingBoundaryConditions = [this, bMin, maxLength](const VectorD& x) -> bool {
        VectorD uv = (x - bMin)/maxLength;
        return false;
    };
    pde.areRobinConditionsPureNeumann = robinCoeff == 0.0f;
    pde.areRobinCoeffsNonnegative = robinCoeff >= 0.0f;
    pde.absorptionCoeff = absorptionCoeff;
}

template <typename T, size_t DIM>
void ModelProblem<T, DIM>::setupPDEVar()
{
    VectorD bMin = boundingBox.first;
    VectorD bMax = boundingBox.second;
    float maxLength = (bMax - bMin).maxCoeff();

    float absorptionMin = 10;
    float absorptionMax = 100;
    float diffusionFreq = 0.5;
    float dirichletFreq = 1.5;
    pdeVar.absorptionFunc = [this, absorptionMin, absorptionMax](const VectorD& x) -> float {
        return absorptionMin + (absorptionMax - absorptionMin) * (1.0f + 0.5f*std::sin(2.0f*M_PI*x(0))*std::cos(0.5f*M_PI*x(1)));
    };
    pdeVar.diffusionFunc = [this, diffusionFreq](const VectorD& x, 
        bool computeGradient, bool computeLaplacian, VectorD& gradient, float& laplacian) -> float {
            float a = 4.0f*M_PI*diffusionFreq;
            float b = 3.0f*M_PI*diffusionFreq;
            float sinAx = std::sin(a*x(0));
            float cosAx = std::cos(a*x(0));
            float sinBy = std::sin(b*x(1));
            float cosBy = std::cos(b*x(1));
            float alpha = std::exp(-x(1)*x(1) + cosAx*sinBy);

            if (computeGradient || computeLaplacian) {
                gradient(0) = alpha*(-sinAx*sinBy*a);
                gradient(1) = alpha*(-2.0f*x(1) + cosAx*cosBy*b);
                gradient(2) = 0.0;
            }

            if (computeLaplacian) {
                float d2Alphadx2 = gradient(0)*(-sinAx*sinBy*a) + alpha*(-cosAx*sinBy*a*a);
                float d2Alphady2 = gradient(1)*(-2.0f*x(1) + cosAx*cosBy*b) + alpha*(-2.0f - cosAx*sinBy*b*b);
                laplacian = d2Alphadx2 + d2Alphady2;
            }

            return alpha;
    };
    pdeVar.sourceFunc = [this, dirichletFreq](const VectorD& x) -> float {
        VectorD dAlpha = VectorD::Zero();
        float laplacian;
        float alpha = pdeVar.diffusionFunc(x, true, false, dAlpha, laplacian);
        float sigma = pdeVar.absorptionFunc(x);
        float a = M_PI*dirichletFreq;
        float b = 2.0f*a;
        float c = 3.0f*a;
        float sinAx = std::sin(a*x(0));
        float cosAx = std::cos(a*x(0));
        float sinBy = std::sin(b*x(1));
        float cosBy = std::cos(b*x(1));
        float sinCz = std::sin(c*x(2));
        float cosCz = std::cos(c*x(2));

        float u = sinAx*cosBy + (1.0f - cosAx)*(1.0f - sinBy) + sinCz*sinCz;
        zombie::Vector3 du = zombie::Vector3::Zero();
        du(0) = (cosAx*cosBy + sinAx*(1.0f - sinBy))*a;
        du(1) = -(sinAx*sinBy + (1.0f - cosAx)*cosBy)*b;
        float d2udx2 = (cosAx*(1.0f - sinBy) - sinAx*cosBy)*a*a;
        float d2udy2 = ((1.0f - cosAx)*sinBy - sinAx*cosBy)*b*b;
        float d2u = d2udx2 + d2udy2;
        du(2) = 2.0f*sinCz*cosCz*c;
        float d2udz2 = 2.0f*(cosCz*cosCz - sinCz*sinCz)*c*c;
        d2u += d2udz2;

        return -alpha*d2u - dAlpha.dot(du) + sigma*u;
    };
    pdeVar.dirichletFunc = [this, dirichletFreq](const VectorD& x) -> float {
        float k = M_PI*dirichletFreq;
        float sinAx = std::sin(k*x(0));
        float cosAx = std::cos(k*x(0));
        float sinBy = std::sin(2.0f*k*x(1));
        float cosBy = std::cos(2.0f*k*x(1));
        float sinCz = std::sin(3.0f*k*x(2));

        return sinAx*cosBy + (1.0f - cosAx)*(1.0f - sinBy) + sinCz*sinCz;
    };
    pdeVar.robin = [this, bMin, maxLength](const VectorD& x, const VectorD& n, bool _) -> float {
        VectorD uv = (x - bMin)/maxLength;
        return 0.0;
    };
    pdeVar.robinCoeff = [this](const VectorD& x, const VectorD& n, bool _) -> float {
        return 0.0;
    };
    pdeVar.hasReflectingBoundaryConditions = [this, bMin, maxLength](const VectorD& x) -> bool {
        VectorD uv = (x - bMin)/maxLength;
        return false;
    };
    pdeVar.bound = compute_bound(pdeVar, {-1.0, -1.0, -1.0},{1.0f, 1.000000f, 1.00000f});
    pdeVar.areRobinConditionsPureNeumann = robinCoeff == 0.0f;
    pdeVar.areRobinCoeffsNonnegative = robinCoeff >= 0.0f;
    pdeVar.absorptionCoeff = absorptionCoeff;
}


template <typename T, size_t DIM>
void ModelProblem<T, DIM>::partitionBoundaryMesh()
{
    // use Zombie's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    zombie::partitionBoundaryMesh<DIM>(pde.hasReflectingBoundaryConditions, positions, indices,
                                     absorbingBoundaryPositions, absorbingBoundaryIndices,
                                     reflectingBoundaryPositions, reflectingBoundaryIndices);
}
template <typename T, size_t DIM>
void ModelProblem<T, DIM>::partitionBoundaryMeshVar()
{
    // use Zombie's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    zombie::partitionBoundaryMesh<DIM>(pdeVar.hasReflectingBoundaryConditions, positions, indices,
                                     absorbingBoundaryPositions, absorbingBoundaryIndices,
                                     reflectingBoundaryPositions, reflectingBoundaryIndices);
}
template <typename T, size_t DIM>
void ModelProblem<T, DIM>::populateGeometricQueries()
{
    // set the domain extent for geometric queries
    queries.domainIsWatertight = domainIsWatertight;
    queries.domainMin = boundingBox.first;
    queries.domainMax = boundingBox.second;

    // use an absorbing boundary handler to populate geometric queries for the absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryPositions, absorbingBoundaryIndices);
    zombie::populateGeometricQueriesForDirichletBoundary<DIM>(absorbingBoundaryHandler, queries);

    if (!solveDoubleSided && useSdfForAbsorbingBoundary) {
        // override distance queries to use an SDF grid. The user can also use Zombie to build
        // an SDF hierarchy for double-sided problems (ommited here for simplicity)
        sdfGridForAbsorbingBoundary = std::make_unique<zombie::SdfGrid<DIM>>(queries.domainMin, queries.domainMax);
        Vectorint sdfGridShape = Vectorint::Constant(sdfGridResolution);
        zombie::populateSdfGrid<DIM>(absorbingBoundaryHandler, *sdfGridForAbsorbingBoundary, sdfGridShape);
        zombie::populateGeometricQueriesForDirichletBoundary<zombie::SdfGrid<DIM>, DIM>(*sdfGridForAbsorbingBoundary, queries);
    }

    // use a reflecting boundary handler to populate geometric queries for the reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette = zombie::getIgnoreCandidateSilhouetteCallback(solveDoubleSided);
    std::function<float(float)> branchTraversalWeight = zombie::getBranchTraversalWeightCallback();

    if (pde.areRobinConditionsPureNeumann) {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette);
        zombie::populateGeometricQueriesForNeumannBoundary<DIM>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForRobinBoundary<DIM>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);
    }
}

template <typename T, size_t DIM>
void ModelProblem<T, DIM>::populateGeometricQueriesVar()
{
    // set the domain extent for geometric queries
    queries.domainIsWatertight = domainIsWatertight;
    queries.domainMin = boundingBox.first;
    queries.domainMax = boundingBox.second;

    // use an absorbing boundary handler to populate geometric queries for the absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryPositions, absorbingBoundaryIndices);
    zombie::populateGeometricQueriesForDirichletBoundary<DIM>(absorbingBoundaryHandler, queries);

    if (!solveDoubleSided && useSdfForAbsorbingBoundary) {
        // override distance queries to use an SDF grid. The user can also use Zombie to build
        // an SDF hierarchy for double-sided problems (ommited here for simplicity)
        sdfGridForAbsorbingBoundary = std::make_unique<zombie::SdfGrid<DIM>>(queries.domainMin, queries.domainMax);
        Vectorint sdfGridShape = Vectorint::Constant(sdfGridResolution);
        zombie::populateSdfGrid<DIM>(absorbingBoundaryHandler, *sdfGridForAbsorbingBoundary, sdfGridShape);
        zombie::populateGeometricQueriesForDirichletBoundary<zombie::SdfGrid<DIM>, DIM>(*sdfGridForAbsorbingBoundary, queries);
    }

    // use a reflecting boundary handler to populate geometric queries for the reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette = zombie::getIgnoreCandidateSilhouetteCallback(solveDoubleSided);
    std::function<float(float)> branchTraversalWeight = zombie::getBranchTraversalWeightCallback();

    if (pdeVar.areRobinConditionsPureNeumann) {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette);
        zombie::populateGeometricQueriesForNeumannBoundary<DIM>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForRobinBoundary<DIM>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);
    }
}
