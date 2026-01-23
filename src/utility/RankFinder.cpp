#include "RankFinder.hpp"
#include <algorithm>
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

// Compute rank via QR decomposition with Householder reflections
// Input matrix A is in row-major format (blockSize x blockSize)
int RankFinder::compute_rank_qr(std::vector<double> &A, int m, int n,
                                double tol) {
  int min_mn = std::min(m, n);

  // Find matrix norm for relative tolerance
  double max_val = 0.0;
  for (int i = 0; i < m * n; ++i) {
    max_val = std::max(max_val, std::abs(A[i]));
  }
  double threshold = tol * max_val * std::max(m, n);

  if (max_val < tol) {
    return 0; // Zero matrix
  }

  int rank = 0;

  for (int k = 0; k < min_mn; ++k) {
    // Compute norm of column k from row k downward
    // Row-major:   A[row, col] = A[row * n + col]
    double col_norm = 0.0;
    for (int i = k; i < m; ++i) {
      double val = A[i * n + k]; // Row-major access
      col_norm += val * val;
    }
    col_norm = sycl::sqrt(col_norm);

    if (col_norm < threshold) {
      for (int i = k; i < m; ++i) {
        A[i * n + k] = 0.0;
      }
      continue;
    }

    // Compute Householder vector
    double alpha = A[k * n + k];
    double beta = (alpha >= 0) ? -col_norm : col_norm;

    A[k * n + k] = beta;

    if (std::abs(beta) > threshold) {
      ++rank;
    }

    double v_norm_sq = col_norm * col_norm - alpha * beta;

    if (v_norm_sq < 1e-30)
      continue;

    double v_k = alpha - beta;

    // Apply Householder to remaining columns
    for (int j = k + 1; j < n; ++j) {
      double dot = v_k * A[k * n + j];
      for (int i = k + 1; i < m; ++i) {
        dot += A[i * n + k] * A[i * n + j];
      }

      double scale = 2.0 * dot / v_norm_sq;
      A[k * n + j] -= scale * v_k;
      for (int i = k + 1; i < m; ++i) {
        A[i * n + j] -= scale * A[i * n + k];
      }
    }
  }

  return rank;
}

// Build a single block's covariance matrix and compute its rank
// Returns the rank of the block
int RankFinder::compute_block_rank(
    const std::vector<conf::fp_type, sycl::usm_allocator<
                                         conf::fp_type, sycl::usm::alloc::host>>
        &trainingInputData,
    std::size_t i_block, std::size_t j_block, std::size_t N,
    std::size_t matrixBlockSize, std::size_t nRegressors,
    double verticalLengthscale, double lengthscale, double noiseVariance) {
  // Determine actual block dimensions (handle boundary blocks)
  std::size_t i_start = i_block * matrixBlockSize;
  std::size_t j_start = j_block * matrixBlockSize;
  std::size_t actual_rows = std::min(matrixBlockSize, N - i_start);
  std::size_t actual_cols = std::min(matrixBlockSize, N - j_start);

  // Allocate temporary block storage (row-major)
  std::vector<double> block(actual_rows * actual_cols);

  // Fill the block with covariance values
  for (std::size_t i_local = 0; i_local < actual_rows; ++i_local) {
    for (std::size_t j_local = 0; j_local < actual_cols; ++j_local) {
      std::size_t i_global = i_start + i_local;
      std::size_t j_global = j_start + j_local;

      // Compute squared Euclidean distance
      double distance = 0.0;
      for (std::size_t k = 0; k < nRegressors; ++k) {
        double tmp = static_cast<double>(trainingInputData[i_global + k]) -
                     static_cast<double>(trainingInputData[j_global + k]);
        distance += tmp * tmp;
      }

      // RBF kernel
      double covarianceValue =
          verticalLengthscale *
          sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

      // Add noise on diagonal
      if (i_global == j_global) {
        covarianceValue += noiseVariance;
      }

      // Store in row-major order
      block[i_local * actual_cols + j_local] = covarianceValue;
    }
  }

  // Compute rank
  const int rank = compute_rank_qr(block, actual_rows, actual_cols);
  return rank;
}

// Compute block ID using your row-major lower triangular ordering
int RankFinder::get_block_id(int i_block, int j_block, int blockCountXY) {
  const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;
  const int block_j_inv = blockCountXY - (j_block + 1);
  const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;
  return i_block + referenceBlockCount - columnBlocksToRight;
}

// Compute ranks for all blocks and return precision types
// Returns a vector where index = blockID, value = precision type
std::vector<int> RankFinder::compute_all_block_ranks(
    sycl::queue &queue,
    const std::vector<conf::fp_type, sycl::usm_allocator<
                                         conf::fp_type, sycl::usm::alloc::host>>
        &trainingInputData,
    std::size_t N, std::size_t matrixBlockSize, std::size_t nRegressors,
    double verticalLengthscale, double lengthscale, double noiseVariance) {
  // Calculate block count
  std::size_t blockCountXY = (N + matrixBlockSize - 1) / matrixBlockSize;
  std::size_t totalBlocks = (blockCountXY * (blockCountXY + 1)) / 2;

  // Store ranks indexed by blockID
  std::vector<int> blockRanks(totalBlocks);

// Iterate over lower triangular blocks
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (std::size_t i_block = 0; i_block < blockCountXY; ++i_block) {
    for (std::size_t j_block = 0; j_block <= i_block; ++j_block) {
      int blockID = get_block_id(i_block, j_block, blockCountXY);

      int rank = compute_block_rank(
          trainingInputData, i_block, j_block, N, matrixBlockSize, nRegressors,
          verticalLengthscale, lengthscale, noiseVariance);

      blockRanks[blockID] = rank;
    }
  }

  return blockRanks;
}

// Determine precision type based on rank
// Returns:   8 = double (FP64), 4 = float (FP32), 2 = half (FP16)
int RankFinder::determine_precision(int rank, int blockSize, bool isDiagonal) {
  double rankRatio = static_cast<double>(rank) / blockSize;

  // Diagonal blocks need higher precision for stability
  if (isDiagonal) {
    return 8; // FP64
  }

  // Off-diagonal blocks can use lower precision for low-rank blocks
  if (rankRatio > 0.7)
    return 8; // FP64
  if (rankRatio > 0.15)
    return 4; // FP32
  return 2;   // FP16
}

// Main function to compute precision types for all blocks
std::vector<int> RankFinder::compute_block_precisions(
    sycl::queue &queue,
    const std::vector<conf::fp_type, sycl::usm_allocator<
                                         conf::fp_type, sycl::usm::alloc::host>>
        &trainingInputData,
    std::size_t N, std::size_t matrixBlockSize, std::size_t nRegressors,
    double verticalLengthscale, double lengthscale, double noiseVariance) {
  std::size_t blockCountXY = (N + matrixBlockSize - 1) / matrixBlockSize;
  std::size_t totalBlocks = (blockCountXY * (blockCountXY + 1)) / 2;

  // First compute all ranks
  std::vector<int> blockRanks = compute_all_block_ranks(
      queue, trainingInputData, N, matrixBlockSize, nRegressors,
      verticalLengthscale, lengthscale, noiseVariance);

  // Then determine precision for each block
  std::vector<int> precisionTypes(totalBlocks);

  for (std::size_t i_block = 0; i_block < blockCountXY; ++i_block) {
    for (std::size_t j_block = 0; j_block <= i_block; ++j_block) {
      int blockID = get_block_id(i_block, j_block, blockCountXY);
      bool isDiagonal = (i_block == j_block);

      precisionTypes[blockID] =
          determine_precision(blockRanks[blockID], matrixBlockSize, isDiagonal);
    }
  }

  return precisionTypes;
}
