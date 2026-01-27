#ifndef SYMMETRICMATRIXMIXED_HPP
#define SYMMETRICMATRIXMIXED_HPP

#include <sycl/sycl.hpp>
#include <vector>

#include "Configuration.hpp"

/**
 * Class that represents a mixed precision symmetric matrix that is stored in a
 * blocked manner. Symmetric values in diagonal blocks are stored redundantly.
 * Otherwise, only the lower triangle of the matrix is stored.
 *
 * The block size can be set via the constructor.
 *
 * Blocks in the lower (blocked) triangle of the matrix are enumerated from top
 * to bottom and from left to right in the matrix. Internally blocks are stored
 * after each other according to their ID. Each block itself is stored in row
 * major layout.
 *
 * Example:
 *
 * Lower triangle of the symmetric matrix divided into blocks:
 * |---+---+---|
 * | 0 |   |   |
 * |---+---+---|
 * | 1 | 3 |   |
 * |---+---+---|
 * | 2 | 4 | 5 |
 * |---+---+---|
 *
 *
 * Block order in memory:
 *
 * | 0 | 1 | 2 | 3 | 4 | 5 |
 *
 */
class SymmetricMatrixMixed {
public:
  /**
   * Constructor of the class.
   * Automatically resizes the vector matrixData to the correct size.
   *
   * @param N dimension N of the NxN symmetric matrix
   * @param blockSize the block size of the blocks the matrix is divided in for
   * storage
   * @param queue SYCL queue for allocating memory
   */
  SymmetricMatrixMixed(std::size_t N, const int blockSize, sycl::queue &queue);

  // Size N of the NxN symmetric matrix
  const std::size_t N;
  // The matrix will be partitioned in blockSize x blockSize blocks
  const int blockSize;
  // block Count in X/Y direction (if the matrix would be stored completely)
  const int blockCountXY;
  // block counts
  std::size_t blockCountFP16;
  std::size_t blockCountFP32;
  std::size_t blockCountFP64;
  // SYCL shared memory containing array of precision per block in block order
  std::vector<int, sycl::usm_allocator<int, sycl::usm::alloc::shared>>
      precisionTypes;
  // SYCL shared memory containing array of rank per block in block order
  std::vector<int, sycl::usm_allocator<int, sycl::usm::alloc::shared>>
      blockRanks;
  // SYCL shared memory containing array of byte offset until current block
  std::vector<std::size_t,
              sycl::usm_allocator<std::size_t, sycl::usm::alloc::shared>>
      blockByteOffsets;

  /// internal matrix data structure allocated as SYCL host memory
  std::vector<unsigned char,
              sycl::usm_allocator<unsigned char, sycl::usm::alloc::host>>
      matrixData;

  void allocate(size_t total_bytes);
};

#endif // SYMMETRICMATRIXMIXED_HPP
