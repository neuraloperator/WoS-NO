#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_scene_loader.h>
#include <fstream>
#include <sstream>
#include <cfloat>
#include "config.h"
#include "image.h"



template<typename T, int DIM>
float compute_bound(const zombie::PDEVar<T, DIM> pde, const zombie::Vector<DIM>& bmin, const zombie::Vector<DIM>& bmax) {
    float boundMin = FLT_MAX, boundMax = -FLT_MAX;
    for(int i = 0; i < 10000; i++) {
        zombie::Vector<DIM> p = bmin.array() + zombie::sampleUniformVector<DIM>().array() * (bmax - bmin).array();
        auto v = pde.transformedAbsorption(p);
        boundMin = std::min(boundMin, v);
        boundMax = std::max(boundMax, v);
    }

    return boundMax - boundMin;
}


class Scene3DVar {
public:
	fcpw::BoundingBox<3> bbox;
	std::vector<Vector3> vertices;
	std::vector<std::vector<size_t>> segments;

	const bool isWatertight;
	const bool isDoubleSided;

	zombie::GeometricQueries<3> queries;
	zombie::PDEVar<float, 3> pde;
	std::vector<float> absorption;
	std::vector<float> diffusion;
	std::vector<float> dirichlet;
	std::vector<float> general;

	
	Scene3DVar(const json &config, std::vector<float>absorption, std::vector<float>diffusion, std::vector<float>dirichlet, std::vector<float>general):
		  isWatertight(getOptional<bool>(config, "isWatertight", true)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		// const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		// const std::string dirichletBoundaryValueFile = getRequired<std::string>(config, "dirichletBoundaryValue");
		// const std::string neumannBoundaryValueFile = getRequired<std::string>(config, "neumannBoundaryValue");
		const std::string sourceValueFile = getRequired<std::string>(config, "sourceValue");
		bool normalize = getOptional<bool>(config, "normalizeDomain", true);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

		//sourceValue = std::make_shared<Image<1>>(sourceValueFile);
		//int h = sourceValue->h;
		//int w = sourceValue->w;
		//isNeumann = std::make_shared<Image<1>>(h, w, 1.0);
		//dirichletBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
		//neumannBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);

		// isNeumann = std::make_shared<Image<1>>(isNeumannFile);
		// dirichletBoundaryValue = std::make_shared<Image<1>>(dirichletBoundaryValueFile);
		// neumannBoundaryValue = std::make_shared<Image<1>>(neumannBoundaryValueFile);
		this->absorption = absorption;
		this->diffusion = diffusion;
		this->dirichlet= dirichlet;
		this->general= general;

		loadOBJ(boundaryFile, normalize, flipOrientation);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	Scene3DVar(const json &config, std::vector<float>absorption, std::vector<float>diffusion, std::vector<float>dirichlet, std::vector<float>general, std::vector<std::string>geometry):
		  isWatertight(getOptional<bool>(config, "isWatertight", false)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		//const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		// const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		bool normalize = getOptional<bool>(config, "normalizeDomain", false);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", false);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

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

		//loadOBJ(geometry, normalize, flipOrientation);
		loadGeometry(geometry, normalize, flipOrientation);
		
		//std::cout<<"GEOMETRY LOADED"<<std::endl;
		separateBoundaries();
		
		populateGeometricQueries();
		
		setPDE();
		//std::cout<<"PDE SET"<<std::endl;
	}

	Scene3DVar(const json &config, std::vector<std::vector<float>> sourceValue_mat, std::vector<float>absorption, std::vector<float>diffusion, std::vector<float>dirichlet, std::vector<float>general):
		  isWatertight(getOptional<bool>(config, "isWatertight", false)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		// const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		bool normalize = getOptional<bool>(config, "normalizeDomain", false);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", false);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

		int h = sourceValue_mat.size();
		int w = sourceValue_mat[0].size();
		// isNeumann = std::make_shared<Image<1>>(isNeumannFile);
		//isNeumann = std::make_shared<Image<1>>(h, w, 1.0);
		//dirichletBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
		//neumannBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
		//sourceValue = std::make_shared<Image<1>>(sourceValue_mat);
		
		this->absorption = absorption;
		this->diffusion = diffusion;
		this->dirichlet = dirichlet;
		this->general = general;

		loadOBJ(boundaryFile, normalize, flipOrientation);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	bool onNeumannBoundary(Vector3 x) const {
		//Vector2 uv = (x - bbox.pMin) / bbox.extent().maxCoeff();
		//return isNeumann->get(uv)[0] > 0;
		return 0;
	}

	bool ignoreCandidateSilhouette(float dihedralAngle, int index) const {
		// ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
		// NOTE: for complex scenes with both open and closed meshes, the primitive index argument
		// (of an adjacent line segment/triangle in the scene) can be used to determine whether a
		// vertex/edge should be ignored as a candidate for silhouette tests.
		return isDoubleSided ? false : dihedralAngle < 1e-3f;
	}
	float getSolveRegionVolume() const {
		if (isDoubleSided) return (bbox.pMax - bbox.pMin).prod();
		float solveRegionVolume = 0.0f;
		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		if (dirichletAggregate != nullptr) solveRegionVolume += dirichletAggregate->signedVolume();
		if (neumannAggregate != nullptr) solveRegionVolume += neumannAggregate->signedVolume();
		return std::fabs(solveRegionVolume);
	}

private:

	void loadGeometry(std::vector<std::string> geometry, bool normalize, bool flipOrientation){
		//std::string line;
		for (auto &line : geometry){
			std::istringstream ss(line);
			std::string token;
			ss >> token;
			if(token == "v"){
				//std::cout<<"LINE: "<<line<<std::endl;
				float x, y, z;
				ss >> x;
				ss >> y;
				ss >> z;
				//std::cout<<"Reading Vertex: "<<x <<" Y: "<<y <<" Z:"<<z<<std::endl;
				vertices.emplace_back(Vector3(x, y, z));
			} else if (token == "l") {
				size_t i, j;
				ss >> i >> j;
				//std::cout<<"Reading Segment: "<<i <<" Y: "<<j <<std::endl;
				if (flipOrientation) {
					segments.emplace_back(std::vector<size_t>({j - 1, i - 1}));
				} else {
					segments.emplace_back(std::vector<size_t>({i - 1, j - 1}));
				}
			}
		}
		if (normalize) {
			Vector3 cm(0, 0, 0);
			for (Vector3 v : vertices) cm += v;
			cm /= vertices.size();
			float radius = 0.0f;
			for (Vector3& v : vertices) {
				v -= cm;
				radius = std::max(radius, v.norm());
			}
			for (Vector3& v : vertices) v /= radius;
		}

		bbox = zombie::computeBoundingBox(vertices, true, 1.0);
	}
	void loadOBJ(const std::string &filename, bool normalize, bool flipOrientation) {
		std::ifstream obj(filename);
		if (!obj) {
			std::cerr << "Error opening file: " << filename << std::endl;
			abort();
		}

		std::string line;
		while (std::getline(obj, line)) {
			std::istringstream ss(line);
			std::string token;
			
			ss >> token;
			if (token == "v") {
				//std::cout<<"LINE: "<<line<<std::endl;
				float x, y, z;
				ss >> x;
				ss >> y;
				ss >> z;
				//std::cout<<"Reading Vertex: "<<x <<" Y: "<<y <<std::endl;
				vertices.emplace_back(Vector3(x, y, z));
			} else if (token == "l") {
				size_t i, j;
				ss >> i >> j;
				if (flipOrientation) {
					segments.emplace_back(std::vector<size_t>({j - 1, i - 1}));
				} else {
					segments.emplace_back(std::vector<size_t>({i - 1, j - 1}));
				}
			}
		}
		obj.close();

		if (normalize) {
			Vector3 cm(0, 0, 0);
			for (Vector3 v : vertices) cm += v;
			cm /= vertices.size();
			float radius = 0.0f;
			for (Vector3& v : vertices) {
				v -= cm;
				radius = std::max(radius, v.norm());
			}
			for (Vector3& v : vertices) v /= radius;
		}

		bbox = zombie::computeBoundingBox(vertices, true, 1.0);
	}

	void separateBoundaries() {
		
		std::vector<size_t> indices(2, -1);
		size_t vDirichlet = 0, vNeumann = 0;
		std::unordered_map<size_t, size_t> dirichletIndexMap, neumannIndexMap;

		std::function<bool(float, int)> ignoreCandidateSilhouette = [this](float dihedralAngle, int index) -> bool {
			return this->ignoreCandidateSilhouette(dihedralAngle, index);
		};
		neumannSceneLoader = new zombie::FcpwSceneLoader<3>(vertices, segments,
													ignoreCandidateSilhouette, true);
		for (int i = 0; i < segments.size(); i++) {
			Vector3 pMid = 0.5f * (vertices[segments[i][0]] + vertices[segments[i][1]]); //Find the mid-point of the edges
			//std::cout<<"FINDING MIDPOINT"<<std::endl;
			if (onNeumannBoundary(pMid)) {
				for (int j = 0; j < 2; j++) {
					size_t vIndex = segments[i][j];
					if (neumannIndexMap.find(vIndex) == neumannIndexMap.end()) {
						const Vector3& p = vertices[vIndex];
						neumannVertices.emplace_back(p);
						neumannIndexMap[vIndex] = vNeumann++;
					}
					indices[j] = neumannIndexMap[vIndex];
				}
				neumannSegments.emplace_back(indices);
			} else {
				//std::cout<<"INSIDE ELSE"<<std::endl;
				for (int j = 0; j < 2; j++) {
					size_t vIndex = segments[i][j];
					if (dirichletIndexMap.find(vIndex) == dirichletIndexMap.end()) {
						const Vector3& p = vertices[vIndex];
						dirichletVertices.emplace_back(p);
						dirichletIndexMap[vIndex] = vDirichlet++;
					}
					indices[j] = dirichletIndexMap[vIndex];
				}
				dirichletSegments.emplace_back(indices);
			}
		}

		// for (auto it = dirichletVertices.begin(); it != dirichletVertices.end(); it++){
		// 	std::cout<<"VERTICES: "<<*it<<std::endl;
		// }
		// std::cout<<"VERTICES SIZE: "<<dirichletVertices.size()<<std::endl;


		dirichletSceneLoader = new zombie::FcpwSceneLoader<3>(dirichletVertices, dirichletSegments);
	}


	void populateGeometricQueries() {
		neumannSamplingTraversalWeight = [this](float r2) -> float {
			float r = std::max(std::sqrt(r2), 1e-2f);
			return std::fabs(this->harmonicGreensFn.evaluate(r));
		};

		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		zombie::populateGeometricQueries<3>(queries, bbox, dirichletAggregate, neumannAggregate,
											neumannSamplingTraversalWeight);
	}

	// void setPDE() {
	// 	// float maxLength = this->bbox.extent().maxCoeff();
	// 	Vector2 extent = this->bbox.extent();
	// 	pde.dirichlet = [this, extent](const Vector2& x) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->dirichletBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.neumann = [this, extent](const Vector2& x) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->neumannBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.dirichletDoubleSided = [this, extent](const Vector2& x, bool _) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->dirichletBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.neumannDoubleSided = [this, extent](const Vector2& x, bool _) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->neumannBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.source = [this, extent](const Vector2& x) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->sourceValue->get(uv)[0];
	// 	};
	// 	pde.absorption = absorptionCoeff;
	// }

	void setPDE() {
		// float maxLength = this->bbox.extent().maxCoeff();
		Vector3 extent = this->bbox.extent();
		float absorptionMin = 10;
        float absorptionMax = 100;
        float diffusionFreq = 0.5;
        float dirichletFreq = 1.5;

        pde.mediumType = zombie::MediumType::Absorbing;
        pde.absorptionFunc = [this, absorptionMin, absorptionMax](const zombie::Vector3& x) -> float {
            return absorptionMin + (absorptionMax - absorptionMin) * (1.0f + 0.5f*std::sin(2.0f*M_PI*x(0))*std::cos(0.5f*M_PI*x(1)));
        };
		pde.neumann = [this, extent](const Vector3& x) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->neumannBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.dirichletDoubleSided = [this, extent](const Vector3& x, bool _) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->dirichletBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.neumannDoubleSided = [this, extent](const Vector3& x, bool _) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->neumannBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.diffusionFunc = [this, diffusionFreq](const zombie::Vector3& x, 
            bool computeGradient, bool computeLaplacian, zombie::Vector3& gradient, float& laplacian) -> float {
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
            }

            if (computeLaplacian) {
                float d2Alphadx2 = gradient(0)*(-sinAx*sinBy*a) + alpha*(-cosAx*sinBy*a*a);
                float d2Alphady2 = gradient(1)*(-2.0f*x(1) + cosAx*cosBy*b) + alpha*(-2.0f - cosAx*sinBy*b*b);
                laplacian = d2Alphadx2 + d2Alphady2;
            }

            return alpha;
        };
		pde.dirichletFunc = [this, dirichletFreq](const zombie::Vector3& x) -> float {
            float k = M_PI*dirichletFreq;
			float b = 2.0f*k;
			float c = 3.0f*k;
            float sinAx = std::sin(k*x(0));
            float cosAx = std::cos(k*x(0));
            float sinBy = std::sin(b*x(1));
            float cosBy = std::cos(b*x(1));
            float sinCz = std::sin(c*x(2));

            return sinAx*cosBy + (1.0f - cosAx)*(1.0f - sinBy) + sinCz*sinCz;
        };
		pde.sourceFunc = [this, dirichletFreq](const zombie::Vector3& x) -> float {
            zombie::Vector3 dAlpha = zombie::Vector3::Zero();
            float alpha = this->pde.diffusionGradient(x, dAlpha);
            float sigma = this->pde.absorptionFunc(x);
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
		pde.absorption = absorptionCoeff;
		pde.bound = compute_bound(pde, {-1.0, -1.0, -1.0},{1.0, 1.0, 1.0});
			// returns the absorption coefficient value for the transformed problem at the input point

	}


	std::vector<Vector3> dirichletVertices;
	std::vector<Vector3> neumannVertices;

	std::vector<std::vector<size_t>> dirichletSegments;
	std::vector<std::vector<size_t>> neumannSegments;

	zombie::FcpwSceneLoader<3>* dirichletSceneLoader;
	zombie::FcpwSceneLoader<3>* neumannSceneLoader;

	std::shared_ptr<Image<1>> isNeumann;
	std::shared_ptr<Image<1>> dirichletBoundaryValue;
	std::shared_ptr<Image<1>> neumannBoundaryValue;
	std::shared_ptr<Image<1>> sourceValue;
	float absorptionCoeff;

	zombie::HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
	std::function<float(float)> neumannSamplingTraversalWeight;
};
