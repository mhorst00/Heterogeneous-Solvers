#ifndef RANKFINDER_HPP
#define RANKFINDER_HPP

#include "Configuration.hpp"
#include <sycl/sycl.hpp>
#include <vector>
/**
 * Class containing functions to determine rank of matrix block
 */
class RankFinder {
public:
  // Main function to compute precision types for all blocks
  static std::vector<int> compute_block_precisions(
      const std::vector<
          conf::fp_type,
          sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>>
          &trainingInputData,
      std::size_t N, std::size_t matrixBlockSize, std::size_t nRegressors,
      double verticalLengthscale = 1.0, double lengthscale = 1.0,
      double noiseVariance = 0.01);

private:
  // Compute rank via QR decomposition with Householder reflections
  // Input matrix A is in row-major format (blockSize x blockSize)
  static int compute_rank_qr(std::vector<double> &A, int m, int n,
                             double tol = 3e-2);

  // Build a single block's covariance matrix and compute its rank
  // Returns the rank of the block
  static int compute_block_rank(
      const std::vector<
          conf::fp_type,
          sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>>
          &trainingInputData,
      std::size_t i_block, std::size_t j_block, std::size_t N,
      std::size_t matrixBlockSize, std::size_t nRegressors,
      double verticalLengthscale = 1.0, double lengthscale = 1.0,
      double noiseVariance = 0.01);

  // Compute block ID using your row-major lower triangular ordering
  static int get_block_id(int i_block, int j_block, int blockCountXY);

  // Compute ranks for all blocks and return precision types
  // Returns a vector where index = blockID, value = precision type
  static std::vector<int> compute_all_block_ranks(
      const std::vector<
          conf::fp_type,
          sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::shared>>
          &trainingInputData,
      std::size_t N, std::size_t matrixBlockSize, std::size_t nRegressors,
      double verticalLengthscale = 1.0, double lengthscale = 1.0,
      double noiseVariance = 0.01);
  //
  // Determine precision type based on rank
  // Returns:   0 = double (FP64), 1 = float (FP32), 2 = half (FP16)
  static int determine_precision(int rank, int blockSize, bool isDiagonal);
};

#endif // RANKFINDER_HPP
