#ifndef TRIANGULARSYSTEMSOLVERMIXED_HPP
#define TRIANGULARSYSTEMSOLVERMIXED_HPP

#include "LoadBalancer.hpp"
#include "MetricsTracker.hpp"
#include "RightHandSide.hpp"
#include "SymmetricMatrixMixed.hpp"

using namespace sycl;

/**
 * This class contains a solver for the resulting triangular systems that emerge
 * after the Cholesky decomposition has been perfomred. The systems are solved
 * using a forward and back substitution.
 */
class TriangularSystemSolverMixed {
  public:
    TriangularSystemSolverMixed(SymmetricMatrixMixed &A, void *A_gpu, RightHandSide &b,
                                queue &cpuQueue, queue &gpuQueue,
                                std::shared_ptr<LoadBalancer> loadBalancer);

    SymmetricMatrixMixed &A; /// SPD matrix A
    RightHandSide &b;        /// GPU data structure for A

    // GPU data structure
    void *A_gpu;          /// GPU data structure for A
    conf::fp_type *b_gpu; /// GPU data structure for b

    queue &cpuQueue; /// SYCL queue for the CPU device
    queue &gpuQueue; /// SYCL queue for the GPU device

    std::shared_ptr<LoadBalancer> loadBalancer; /// load balancer object
    MetricsTracker metricsTracker;              /// metrics tracker object

    double solve();
};

#endif // TRIANGULARSYSTEMSOLVERMIXED_HPP
