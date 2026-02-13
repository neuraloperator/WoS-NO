#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_scene_loader.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

class Scene3D {
public:
	fcpw::BoundingBox<3> bbox;
	std::vector<Vector3> vertices;
	std::vector<std::vector<size_t>> segments;

	const bool isWatertight;
	const bool isDoubleSided;

	zombie::GeometricQueries<3> queries;
	zombie::PDE<float, 3> pde;
	std::vector<float> mu1;
	std::vector<float> mu2;
	std::vector<float> beta;
	std::vector<float> r;

	
	Scene3D(const json &config, std::vector<float>mu1, std::vector<float>mu2, std::vector<float>beta, std::vector<float>r):
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
		this->r = r;
		this->beta = beta;
		this->mu1= mu1;
		this->mu2= mu2;

		loadOBJ(boundaryFile, normalize, flipOrientation);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	Scene3D(const json &config, std::vector<float>mu1, std::vector<float>mu2, std::vector<float>beta, std::vector<float>r, std::vector<std::string>geometry):
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
		
		this->mu1 = mu1;
		this->mu2 = mu2;
		this->beta = beta;
		this->r = r;

		//loadOBJ(boundaryFile, normalize, flipOrientation);
		loadGeometry(geometry, normalize, flipOrientation);
		//std::cout<<"GEOMETRY LOADED"<<std::endl;
		separateBoundaries();
		//std::cout<<"BOUNDARY SEPARATED"<<std::endl;
		populateGeometricQueries();
		//std::cout<<"POPULATED GEO QUERIES"<<std::endl;
		setPDE();
		//std::cout<<"PDE SET"<<std::endl;
	}

	Scene3D(const json &config, std::vector<std::vector<float>> sourceValue_mat, std::vector<float>mu1, std::vector<float>mu2, std::vector<float>beta, std::vector<float>r):
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
		
		this->mu1 = mu1;
		this->mu2 = mu2;
		this->beta = beta;
		this->r = r;

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
		//std::cout<<"DIRICHLET SCENE LOADED"<<std::endl;

		//std::cout<<"NEUMANN SCENE LOADED"<<std::endl;
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
		pde.dirichlet = [this, extent](const Vector3& x) -> float {
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->dirichletBoundaryValue->get(uv)[0];
			double theta;
			theta = atan2(x[1], x[0]);

			//float r[5] = {0.5957, 0.4741, -0.0506, -0.3519, 0.6811};
			//return this->r[0] + this->r[1]/4 * cos(theta) + this->r[2]/4 * sin(theta) + this->r[3]/4*cos(2*theta) + this->r[4]/4*sin(2*theta);
			return (x[0]*x[0]) + (x[1]*x[1]) + (x[2]*x[2]);
			//return (x[1]*x[1]) + (x[0]*x[0]);
			//return 0.6;
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
		pde.source = [this, extent](const Vector3& x) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			//Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			//float beta[2] = {-1.1116, 0.6639};
			//float mu1[2] = {-0.8498, -1.1027};
			//float mu2[2] = {0.9604, -1.4707};
			
			// float res1 = this->beta[0] * exp(-(pow((x[0] - this->mu2[0]),2) + pow((x[1] - this->mu1[0]), 2)));
			// float res2 = this->beta[1] * exp(-(pow((x[0] - this->mu2[1]),2) + pow((x[1] - this->mu1[1]), 2)));
			// return res1 + res2;
			return 6.0;
			//return 4.0;
			//return 4.0;
			//return this->sourceValue->get(uv)[0];
			//return 6.0;
		};
		pde.absorption = absorptionCoeff;
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
