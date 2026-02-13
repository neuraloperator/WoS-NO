// This file defines an interface for Partial Differential Equations (PDEs),
// specifically Poisson and screened Poisson equations, with Dirichlet, Neumann,
// and Robin boundary conditions. As part of the problem setup, users of Zombie
// should populate the callback functions defined by the PDE interface.

#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

namespace zombie {

template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

enum class MediumType {
	Empty,
	Absorbing,
	Mixed
};

template <typename T, size_t DIM>
struct PDE {
    // constructor
    PDE();

    // members
    float absorptionCoeff; // must be positive or equal to zero
    bool areRobinConditionsPureNeumann; // set to false if Robin coefficients are non-zero anywhere
    bool areRobinCoeffsNonnegative; // set to false if Robin coefficients are negative anywhere

    // returns source term
    std::function<T(const Vector<DIM>&)> source;

    // returns Dirichlet boundary conditions
    std::function<T(const Vector<DIM>&, bool)> dirichlet;

    // returns Robin boundary conditions and coefficients
    std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> robin; // dual purposes for Neumann conditions when Robin coeff is zero
    std::function<float(const Vector<DIM>&, const Vector<DIM>&, bool)> robinCoeff;

    // checks if the PDE has reflecting boundary conditions (Neumann or Robin) at the given point
    std::function<bool(const Vector<DIM>&)> hasReflectingBoundaryConditions;

    // check if the PDE has a non-zero robin coefficient value at the given point
    std::function<bool(const Vector<DIM>&)> hasNonZeroRobinCoeff; // set automatically
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline PDE<T, DIM>::PDE():
absorptionCoeff(0.0f),
areRobinConditionsPureNeumann(true),
areRobinCoeffsNonnegative(true),
source({}),
dirichlet({}),
robin({}),
robinCoeff({}),
hasReflectingBoundaryConditions({})
{
    hasNonZeroRobinCoeff = [this](const Vector<DIM>& x) {
        if (this->robinCoeff) {
            Vector<DIM> n = Vector<DIM>::Zero();
            n(0) = 1.0f;

            return std::fabs(this->robinCoeff(x, n, true)) > 0.0f ||
                   std::fabs(this->robinCoeff(x, n, false)) > 0.0f;
        }

        return false;
    };
}

template <typename T, int DIM>
class PDEVar {
public:
	// constructor
	PDEVar(): mediumType(MediumType::Empty), dirichletFunc({}), sourceFunc({}),
				 diffusionFunc({}), absorptionFunc({}), bound(0.0f), dirichletDoubleSided({}), neumannDoubleSided({}) {
    // nothing to do
	}

	// returns the dirichlet boundary value at the input point
	T dirichlet(const Vector<DIM>& x, bool) const {
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
	// returns Robin boundary conditions and coefficients
    std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> robin; // dual purposes for Neumann conditions when Robin coeff is zero
    std::function<float(const Vector<DIM>&, const Vector<DIM>&, bool)> robinCoeff;

    // checks if the PDE has reflecting boundary conditions (Neumann or Robin) at the given point
    std::function<bool(const Vector<DIM>&)> hasReflectingBoundaryConditions;

    // check if the PDE has a non-zero robin coefficient value at the given point
    std::function<bool(const Vector<DIM>&)> hasNonZeroRobinCoeff; // set automatically
	float bound;
	float absorptionCoeff; // must be positive or equal to zero
    bool areRobinConditionsPureNeumann; // set to false if Robin coefficients are non-zero anywhere
    bool areRobinCoeffsNonnegative; // set to false if Robin coefficients are negative anywhere
};


} // zombie
