#ifndef MATRIXGENERATORMIXED_HPP
#define MATRIXGENERATORMIXED_HPP

#include <sycl/sycl.hpp>

#include "SymmetricMatrixMixed.hpp"

/**
 * Class that contains various methods for generating symmetric
 * positive-definite (SPD) matrices
 */
class MatrixGeneratorMixed {
  public:
    /**
     * This operation generates a mixed precision SPD kernel matrix that can be
     * used for Gaussian Processes
     *
     * @param path path to text file with data
     * @param queue CPU queue
     * @param queueGPU GPU queue
     * @return a SPD matrix
     */
    static SymmetricMatrixMixed generateSPDMatrixMixed(std::string &path, sycl::queue &queue,
                                                       sycl::queue &queueGPU);

    /**
     * This operation generates the test kernel matrix K* used required for
     * prediction with Gaussian Processes.
     *
     * @param path_train path to training data
     * @param path_test path to test data
     * @param queue CPU queue
     * @param queueGPU GPU queue
     * @param K_star the test kernel matrix
     */
    static void generateTestKernelMatrixMixed(std::string &path_train, std::string &path_test,
                                              sycl::queue &queue, sycl::queue &queueGPU,
                                              conf::fp_type *K_star);

  private:
    static void readInputVector(
        std::string &path,
        std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>>
            &dataVector,
        int N, int offset);
};

#endif // MATRIXGENERATORMIXED_HPP
