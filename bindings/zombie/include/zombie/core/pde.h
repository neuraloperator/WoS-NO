#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace zombie {

template<int DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

enum class MediumType {
	Empty,
	Absorbing,
	Mixed
};

template <typename T, int DIM>
struct PDE {
	// constructor
	PDE(): absorption(0.0f), dirichlet({}), neumann({}), source({}),
		   dirichletDoubleSided({}), neumannDoubleSided({}) {}

	// members
	float absorption;
	std::function<T(const Vector<DIM>&)> dirichlet;
	std::function<T(const Vector<DIM>&)> neumann;
	std::function<T(const Vector<DIM>&)> source;
	std::function<T(const Vector<DIM>&, bool)> dirichletDoubleSided;
	std::function<T(const Vector<DIM>&, bool)> neumannDoubleSided;
};

template <typename T, int DIM>
class PDEVar {
public:
	// constructor
	PDEVar(): mediumType(MediumType::Empty), dirichletFunc({}), sourceFunc({}),
				 diffusionFunc({}), absorptionFunc({}), bound(0.0f), dirichletDoubleSided({}), neumannDoubleSided({}) {
    // nothing to do
	}

	// returns the dirichlet boundary value at the input point
	T dirichlet(const Vector<DIM>& x) const {
		return dirichletFunc(x);
	}

	// returns the source value at the input point
	T source(const Vector<DIM>& x) const {
		return sourceFunc(x);
	}

	// returns the absorption coefficient value at the input point
	float absorption;

	// returns the absorption coefficient value for the transformed problem at the input point
	float transformedAbsorption(const Vector<DIM>& x) const {
		Vector<DIM> diffusionGradient;
		float diffusionLaplacian;
		float diffusionCoeff = diffusionFunc(x, true, true, diffusionGradient, diffusionLaplacian);
		float absorptionCoeff = absorptionFunc(x);
		float absorptionTerm = absorptionCoeff/diffusionCoeff;
		float diffusionGradientTerm = 0.25f*diffusionGradient.squaredNorm()/(diffusionCoeff*diffusionCoeff);
		float diffusionLaplacianTerm = 0.5f*diffusionLaplacian/diffusionCoeff;

		return absorptionTerm + diffusionLaplacianTerm - diffusionGradientTerm;
	}

	// returns the diffusion coefficient value at the input point
	float diffusion(const Vector<DIM>& x) const {
		Vector<DIM> gradientStub;
		float laplacianStub;

		return diffusionFunc(x, false, false, gradientStub, laplacianStub);
	}

	// returns the diffusion coefficient value and its gradient at the input point
	float diffusionGradient(const Vector<DIM>& x, Vector<DIM>& gradient) const {
		float laplacianStub;

		return diffusionFunc(x, true, false, gradient, laplacianStub);
	}

	// members
	MediumType mediumType;
	std::function<T(const Vector<DIM>&)> dirichletFunc;
	std::function<T(const Vector<DIM>&)> sourceFunc;
	std::function<float(const Vector<DIM>&, bool, bool, Vector<DIM>&, float&)> diffusionFunc;
	std::function<float(const Vector<DIM>&)> absorptionFunc;
	std::function<float(const Vector<DIM>&)> solutionFunc;
	std::function<T(const Vector<DIM>&, bool)> dirichletDoubleSided;
	std::function<T(const Vector<DIM>&, bool)> neumannDoubleSided;
	std::function<T(const Vector<DIM>&)> neumann;
	float bound;
};

} // zombie


