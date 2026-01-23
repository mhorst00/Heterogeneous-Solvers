#include <sycl/sycl.hpp>

#include "Configuration.hpp"
#include "MatrixVectorOperationsMixed.hpp"

using namespace sycl;

sycl::event MatrixVectorOperationsMixed::matrixVectorBlock(
    queue &queue, void *A, const conf::fp_type *b, conf::fp_type *result,
    int *precisionTypes, std::size_t *blockByteOffsets, const int blockStart_i,
    const int blockStart_j, const int blockCount_i, const int blockCount_j,
    const int blockCountXY, const bool reset) {
  // global range corresponds to number of rows in the (sub) matrix
  const range globalRange(blockCount_i * conf::matrixBlockSize);
  const range localRange(conf::workGroupSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const bool addToPreviousEntries = !reset;

  unsigned char *A_bytes = static_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](handler &h) {
    h.parallel_for(kernelRange, [=](auto &nd_item) {
      // row i in the matrix
      const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

      // local row index in current matrix block
      const int iInBlock = i % matrixBlockSize;

      // row index of matrix block current work-item will work with
      const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

      // block count of all columns except the first one
      const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

      // block ID in the symmetric matrix
      int blockID = 0;

      // first index for block columns
      int block_j = blockStart_j;

      conf::fp_type resultValue = 0;
      if (addToPreviousEntries) {
        resultValue += result[i];
      }

      // First step: Process all matrix blocks up to the diagonal block
      // (included) or the most left block that should be processed the blocks
      // can be interpreted as they are stored in memory
      for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1);
           ++block_j) {
        // number of blocks in row to the right (if matrix would be full)
        const int block_j_inv = blockCountXY - (block_j + 1);

        // total number of blocks to the right that are stored
        const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

        // id of block in the matrix data structure for symmetric matrices
        blockID = block_i + referenceBlockCount - columnBlocksToRight;

        // Get pre-calculated byte offset and precision type from USM
        const std::size_t byte_offset = blockByteOffsets[blockID];
        const int precision_type = precisionTypes[blockID];

        // startIndex of the current block with blockID in the symmetric matrix
        // data structure
        // std::size_t blockStartIndex = static_cast<std::size_t>(blockID) *
        //                               matrixBlockSize * matrixBlockSize;
        std::size_t rowStartIndex =
            static_cast<std::size_t>(iInBlock) * matrixBlockSize;

        // go through all columns of the block and compute the matrix vector
        // product
        for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
          // Cast based on precision and access
          conf::fp_type a_value = 0;
          if (precision_type == 2) {
            sycl::half *A_fp16 =
                reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
            a_value = A_fp16[rowStartIndex + j];
          } else if (precision_type == 4) {
            float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
            a_value = A_fp32[rowStartIndex + j];
          } else if (precision_type == 8) {
            double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
            a_value = A_fp64[rowStartIndex + j];
          }

          resultValue += a_value * b[block_j * matrixBlockSize + j];
        }
      }

      // Second step: Process all matrix blocks after the diagonal block
      // the blocks have to be interpreted as transposed, since the upper
      // triangle is not stored explicitly
      for (; block_j < blockStart_j + blockCount_j; ++block_j) {
        // same block ID calculation as previously, but now block_i and block_j
        // have to be swapped due to symmetries
        const int block_i_inv = blockCountXY - (block_i + 1);
        const int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

        // id of block in the matrix data structure for symmetric matrices
        blockID = block_j + referenceBlockCount - columnBlocksToRight;

        // Get pre-calculated byte offset and precision type from USM
        const std::size_t byte_offset = blockByteOffsets[blockID];
        const int precision_type = precisionTypes[blockID];

        // go through all columns of the block and compute the matrix vector
        // product the block in storage now has to be interpreted as transposed
        // since we are working on the data of the symmetric block
        for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
          // Cast based on precision and access
          conf::fp_type a_value = 0;
          if (precision_type == 2) {
            sycl::half *A_fp16 =
                reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
            a_value = A_fp16[j * matrixBlockSize + iInBlock];
          } else if (precision_type == 4) {
            float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
            a_value = A_fp32[j * matrixBlockSize + iInBlock];
          } else if (precision_type == 8) {
            double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
            a_value = A_fp64[j * matrixBlockSize + iInBlock];
          }

          resultValue += a_value * b[block_j * matrixBlockSize + j];
        }
      }

      // store the result
      result[i] = resultValue;
    });
  });

  return event;
}

sycl::event MatrixVectorOperationsMixed::matrixVectorBlock_GPU(
    sycl::queue &queue, void *A, const conf::fp_type *b, conf::fp_type *result,
    int *precisionTypes, std::size_t *blockByteOffsets, const int blockStart_i,
    const int blockStart_j, const int blockCount_i, const int blockCount_j,
    const int blockCountXY, const bool reset) {
  // global range corresponds to number of rows in the (sub) matrix
  const range globalRange(blockCount_i * conf::matrixBlockSize);
  const range localRange(conf::workGroupSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const bool addToPreviousEntries = !reset;

  unsigned char *A_bytes = static_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](handler &h) {
    auto local_b = local_accessor<conf::fp_type, 1>(conf::workGroupSize, h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      // row i in the matrix
      const int i = nd_item.get_global_id() + blockStart_i * matrixBlockSize;

      // local row index in current matrix block
      const int iInBlock = i % matrixBlockSize;

      // row index of matrix block current work-item will work with
      const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

      // block count of all columns except the first one
      const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

      // block ID in the symmetric matrix
      int blockID = 0;

      // first index for block columns
      int block_j = blockStart_j;

      conf::fp_type resultValue = 0;
      if (addToPreviousEntries) {
        resultValue += result[i];
      }

      // First step: Process all matrix blocks up to the diagonal block
      // (included) or the most left block that should be processed the blocks
      // can be interpreted as they are stored in memory
      for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1);
           ++block_j) {
        // number of blocks in row to the right (if matrix would be full)
        const int block_j_inv = blockCountXY - (block_j + 1);

        // total number of blocks to the right that are stored
        const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

        // id of block in the matrix data structure for symmetric matrices
        blockID = block_i + referenceBlockCount - columnBlocksToRight;

        // startIndex of the current block with blockID in the symmetric matrix
        // data structure
        // std::size_t blockStartIndex = static_cast<std::size_t>(blockID) *
        //                               matrixBlockSize * matrixBlockSize;
        std::size_t rowStartIndex =
            static_cast<std::size_t>(iInBlock) * matrixBlockSize;

        // Get pre-calculated byte offset and precision type from USM
        const std::size_t byte_offset = blockByteOffsets[blockID];
        int precision_type = precisionTypes[blockID];

        // cache part of rhs b in local memory
        nd_item.barrier();
        local_b[nd_item.get_local_id()] =
            b[block_j * matrixBlockSize + nd_item.get_local_id()];
        nd_item.barrier();

        // go through all columns of the block and compute the matrix vector
        // product
        for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
          // Cast based on precision and access
          conf::fp_type a_value = 0;
          if (precision_type == 2) {
            sycl::half *A_fp16 =
                reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
            a_value = A_fp16[rowStartIndex + j];
          } else if (precision_type == 4) {
            float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
            a_value = A_fp32[rowStartIndex + j];
          } else if (precision_type == 8) {
            double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
            a_value = A_fp64[rowStartIndex + j];
          }

          resultValue += a_value * local_b[j];
        }
      }

      // Second step: Process all matrix blocks after the diagonal block
      // the blocks have to be interpreted as transposed, since the upper
      // triangle is not stored explicitly
      for (; block_j < blockStart_j + blockCount_j; ++block_j) {
        // same block ID calculation as previously, but now block_i and block_j
        // have to be swapped due to symmetries
        const int block_i_inv = blockCountXY - (block_i + 1);
        const int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

        // id of block in the matrix data structure for symmetric matrices
        blockID = block_j + referenceBlockCount - columnBlocksToRight;

        // Get pre-calculated byte offset and precision type from USM
        const std::size_t byte_offset = blockByteOffsets[blockID];
        int precision_type = precisionTypes[blockID];

        // cache part of rhs b in local memory
        nd_item.barrier();
        local_b[nd_item.get_local_id()] =
            b[block_j * matrixBlockSize + nd_item.get_local_id()];
        nd_item.barrier();

        // go through all columns of the block and compute the matrix vector
        // product the block in storage now has to be interpreted as transposed
        // since we are working on the data of the symmetric block
        for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
          // Cast based on precision and access
          conf::fp_type a_value = 0;
          if (precision_type == 2) {
            sycl::half *A_fp16 =
                reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
            a_value = A_fp16[j * matrixBlockSize + iInBlock];
          } else if (precision_type == 4) {
            float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
            a_value = A_fp32[j * matrixBlockSize + iInBlock];
          } else if (precision_type == 8) {
            double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
            a_value = A_fp64[j * matrixBlockSize + iInBlock];
          }

          resultValue += a_value * local_b[j];
        }
      }

      // store the result
      result[i] = resultValue;
    });
  });

  return event;
}

sycl::event MatrixVectorOperationsMixed::matrixVectorBlock_CPU(
    sycl::queue &queue, void *A, const conf::fp_type *b, conf::fp_type *result,
    int *precisionTypes, std::size_t *blockByteOffsets, const int blockStart_i,
    const int blockStart_j, const int blockCount_i, const int blockCount_j,
    const int blockCountXY, const bool reset) {
  // global range corresponds to number of rows in the (sub) matrix
  const std::size_t globalRange = blockCount_i * conf::matrixBlockSize;
  const auto kernelRange = range{globalRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const bool addToPreviousEntries = !reset;

  unsigned char *A_bytes = static_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](handler &h) {
    h.parallel_for(kernelRange, [=](auto &id) {
      // row i in the matrix
      const int i = id[0] + blockStart_i * matrixBlockSize;

      // local row index in current matrix block
      const int iInBlock = i % matrixBlockSize;

      // row index of matrix block current work-item will work with
      const int block_i = sycl::floor(static_cast<double>(i) / matrixBlockSize);

      // block count of all columns except the first one
      const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

      // block ID in the symmetric matrix
      int blockID = 0;

      // first index for block columns
      int block_j = blockStart_j;

      conf::fp_type resultValue = 0;
      if (addToPreviousEntries) {
        resultValue += result[i];
      }

      // First step: Process all matrix blocks up to the diagonal block
      // (included) or the most left block that should be processed the blocks
      // can be interpreted as they are stored in memory
      for (; block_j <= min(block_i, blockStart_j + blockCount_j - 1);
           ++block_j) {
        // number of blocks in row to the right (if matrix would be full)
        const int block_j_inv = blockCountXY - (block_j + 1);

        // total number of blocks to the right that are stored
        const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

        // id of block in the matrix data structure for symmetric matrices
        blockID = block_i + referenceBlockCount - columnBlocksToRight;

        // Get pre-calculated byte offset and precision type from USM
        const std::size_t byte_offset = blockByteOffsets[blockID];
        int precision_type = precisionTypes[blockID];

        std::size_t rowStartIndex =
            static_cast<std::size_t>(iInBlock) * matrixBlockSize;

        // go through all columns of the block and compute the matrix vector
        // product
#pragma clang loop vectorize(enable)
        for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
          // Cast based on precision and access
          conf::fp_type a_value = 0;
          if (precision_type == 2) {
            sycl::half *A_fp16 =
                reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
            a_value = A_fp16[rowStartIndex + j];
          } else if (precision_type == 4) {
            float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
            a_value = A_fp32[rowStartIndex + j];
          } else if (precision_type == 8) {
            double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
            a_value = A_fp64[rowStartIndex + j];
          }

          resultValue += a_value * b[block_j * matrixBlockSize + j];
        }
      }

      // Second step: Process all matrix blocks after the diagonal block
      // the blocks have to be interpreted as transposed, since the upper
      // triangle is not stored explicitly
      for (; block_j < blockStart_j + blockCount_j; ++block_j) {
        // same block ID calculation as previously, but now block_i and block_j
        // have to be swapped due to symmetries
        const int block_i_inv = blockCountXY - (block_i + 1);
        const int columnBlocksToRight = (block_i_inv * (block_i_inv + 1)) / 2;

        // id of block in the matrix data structure for symmetric matrices
        blockID = block_j + referenceBlockCount - columnBlocksToRight;

        // Get pre-calculated byte offset and precision type from USM
        const std::size_t byte_offset = blockByteOffsets[blockID];
        int precision_type = precisionTypes[blockID];

        // go through all columns of the block and compute the matrix vector
        // product the block in storage now has to be interpreted as transposed
        // since we are working on the data of the symmetric block
#pragma clang loop vectorize(enable)
        for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
          // Cast based on precision and access
          conf::fp_type a_value = 0;
          if (precision_type == 2) {
            sycl::half *A_fp16 =
                reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
            a_value = A_fp16[j * matrixBlockSize + iInBlock];
          } else if (precision_type == 4) {
            float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
            a_value = A_fp32[j * matrixBlockSize + iInBlock];
          } else if (precision_type == 8) {
            double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
            a_value = A_fp64[j * matrixBlockSize + iInBlock];
          }

          resultValue += a_value * b[block_j * matrixBlockSize + j];
        }
      }

      // store the result
      result[i] = resultValue;
    });
  });

  return event;
}

sycl::event MatrixVectorOperationsMixed::triangularSolveBlockVector(
    sycl::queue &queue, void *A, conf::fp_type *b, int *precisionTypes,
    std::size_t *blockByteOffsets, const int blockRow, const int blockID,
    const bool transposed) {
  // one work-group, one work-item per entry in b
  const range globalRange(conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) *
                                      conf::matrixBlockSize *
                                      conf::matrixBlockSize;

  const std::size_t byte_offset = blockByteOffsets[blockID];
  int precision_type = precisionTypes[blockID];
  unsigned char *A_bytes = static_cast<unsigned char *>(A);

  const std::size_t N = conf::N;

  sycl::event event;
  if (precision_type == 2) {
    sycl::half *A_fp16 = reinterpret_cast<sycl::half *>(A_bytes + byte_offset);
    event = queue.submit([&](sycl::handler &h) {
      h.parallel_for(kernelRange, [=](auto &nd_item) {
        int local_i = nd_item.get_local_id(0);

        // block in the vector b where the results are written
        const std::size_t blockStartIndex_B =
            static_cast<std::size_t>(blockRow) * matrixBlockSize;

        for (int i = 0; i < static_cast<int>(matrixBlockSize); ++i) {
          int k = i;
          if (transposed) {
            k = static_cast<int>(matrixBlockSize) - (k + 1);
          }

          // b_k = b_k/a_kk
          const conf::fp_type b_k =
              b[blockStartIndex_B + k] / A_fp16[k * matrixBlockSize + k];

          nd_item.barrier();

          if (local_i == 0 && blockStartIndex_B + k < N) {
            b[blockStartIndex_B + k] = b_k;
          }

          bool condition = (!transposed) ? local_i > k : local_i < k;

          if (condition && blockStartIndex_B + k < N) {
            // b_i = b_i - a_ik*b_k
            if (!transposed) {
              b[blockStartIndex_B + local_i] =
                  b[blockStartIndex_B + local_i] -
                  A_fp16[local_i * matrixBlockSize + k] * b_k;
            } else {
              b[blockStartIndex_B + local_i] =
                  b[blockStartIndex_B + local_i] -
                  A_fp16[k * matrixBlockSize + local_i] * b_k;
            }
          }

          nd_item.barrier();
        }
      });
    });
  } else if (precision_type == 4) {
    float *A_fp32 = reinterpret_cast<float *>(A_bytes + byte_offset);
    event = queue.submit([&](sycl::handler &h) {
      h.parallel_for(kernelRange, [=](auto &nd_item) {
        int local_i = nd_item.get_local_id(0);

        // block in the vector b where the results are written
        const std::size_t blockStartIndex_B =
            static_cast<std::size_t>(blockRow) * matrixBlockSize;

        for (int i = 0; i < static_cast<int>(matrixBlockSize); ++i) {
          int k = i;
          if (transposed) {
            k = static_cast<int>(matrixBlockSize) - (k + 1);
          }

          // b_k = b_k/a_kk
          const conf::fp_type b_k =
              b[blockStartIndex_B + k] / A_fp32[k * matrixBlockSize + k];

          nd_item.barrier();

          if (local_i == 0 && blockStartIndex_B + k < N) {
            b[blockStartIndex_B + k] = b_k;
          }

          bool condition = (!transposed) ? local_i > k : local_i < k;

          if (condition && blockStartIndex_B + k < N) {
            // b_i = b_i - a_ik*b_k
            if (!transposed) {
              b[blockStartIndex_B + local_i] =
                  b[blockStartIndex_B + local_i] -
                  A_fp32[local_i * matrixBlockSize + k] * b_k;
            } else {
              b[blockStartIndex_B + local_i] =
                  b[blockStartIndex_B + local_i] -
                  A_fp32[k * matrixBlockSize + local_i] * b_k;
            }
          }

          nd_item.barrier();
        }
      });
    });
  } else if (precision_type == 8) {
    double *A_fp64 = reinterpret_cast<double *>(A_bytes + byte_offset);
    event = queue.submit([&](sycl::handler &h) {
      h.parallel_for(kernelRange, [=](auto &nd_item) {
        int local_i = nd_item.get_local_id(0);

        // block in the vector b where the results are written
        const std::size_t blockStartIndex_B =
            static_cast<std::size_t>(blockRow) * matrixBlockSize;

        for (int i = 0; i < static_cast<int>(matrixBlockSize); ++i) {
          int k = i;
          if (transposed) {
            k = static_cast<int>(matrixBlockSize) - (k + 1);
          }

          // b_k = b_k/a_kk
          const conf::fp_type b_k =
              b[blockStartIndex_B + k] / A_fp64[k * matrixBlockSize + k];

          nd_item.barrier();

          if (local_i == 0 && blockStartIndex_B + k < N) {
            b[blockStartIndex_B + k] = b_k;
          }

          bool condition = (!transposed) ? local_i > k : local_i < k;

          if (condition && blockStartIndex_B + k < N) {
            // b_i = b_i - a_ik*b_k
            if (!transposed) {
              b[blockStartIndex_B + local_i] =
                  b[blockStartIndex_B + local_i] -
                  A_fp64[local_i * matrixBlockSize + k] * b_k;
            } else {
              b[blockStartIndex_B + local_i] =
                  b[blockStartIndex_B + local_i] -
                  A_fp64[k * matrixBlockSize + local_i] * b_k;
            }
          }

          nd_item.barrier();
        }
      });
    });
  }

  return event;
}

sycl::event MatrixVectorOperationsMixed::matrixVectorColumnUpdate(
    sycl::queue &queue, void *A, conf::fp_type *b, int *precisionTypes,
    std::size_t *blockByteOffsets, const int blockStart, const int blockCount,
    const int blockRow, const int blockID, int blockCountXY,
    const bool transposed) {
  // one work-group per block in column
  const range globalRange(blockCount * conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const int totalBlockCount = blockCountXY * (blockCountXY + 1) / 2;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(kernelRange, [=](auto &nd_item) {
      int local_i = nd_item.get_local_id(0);
      const int group_id = nd_item.get_group().get_group_id(0);

      // block in the vector b used for the update
      const std::size_t blockStartIndex_b_0 =
          static_cast<std::size_t>(blockRow) * matrixBlockSize;

      // block in the vector b where the results are written
      const std::size_t blockStartIndex_b_i =
          static_cast<std::size_t>(blockStart + group_id) * matrixBlockSize;

      // Compute the block ID for the matrix block used in the update
      int blockID_Aij;
      if (!transposed) {
        blockID_Aij = blockID + (blockStart - blockRow + group_id);
      } else {
        blockID_Aij = totalBlockCount -
                      ((blockCountXY - group_id - blockStart) *
                       (blockCountXY - group_id - blockStart + 1) / 2) +
                      blockRow - group_id - blockStart;
      }

      const std::size_t byteOffset = blockByteOffsets[blockID_Aij];
      const int precision = precisionTypes[blockID_Aij];

      conf::fp_type sum = 0.0;
      if (precision == 2) {
        sycl::half *A_typed =
            reinterpret_cast<sycl::half *>(ABytes + byteOffset);
        if (!transposed) {
          for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
            sum += static_cast<conf::fp_type>(
                       A_typed[local_i * matrixBlockSize + k]) *
                   b[blockStartIndex_b_0 + k];
          }
        } else {
          for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
            sum += static_cast<conf::fp_type>(
                       A_typed[k * matrixBlockSize + local_i]) *
                   b[blockStartIndex_b_0 + k];
          }
        }
      } else if (precision == 4) {
        float *A_typed = reinterpret_cast<float *>(ABytes + byteOffset);
        if (!transposed) {
          for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
            sum += static_cast<conf::fp_type>(
                       A_typed[local_i * matrixBlockSize + k]) *
                   b[blockStartIndex_b_0 + k];
          }
        } else {
          for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
            sum += static_cast<conf::fp_type>(
                       A_typed[k * matrixBlockSize + local_i]) *
                   b[blockStartIndex_b_0 + k];
          }
        }
      } else if (precision == 8) {
        double *A_typed = reinterpret_cast<double *>(ABytes + byteOffset);
        if (!transposed) {
          for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
            sum += static_cast<conf::fp_type>(
                       A_typed[local_i * matrixBlockSize + k]) *
                   b[blockStartIndex_b_0 + k];
          }
        } else {
          for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
            sum += static_cast<conf::fp_type>(
                       A_typed[k * matrixBlockSize + local_i]) *
                   b[blockStartIndex_b_0 + k];
          }
        }
      }
      b[blockStartIndex_b_i + local_i] -= sum;
    });
  });

  return event;
}

sycl::event MatrixVectorOperationsMixed::matrixVectorGP(
    sycl::queue &queue, void *A, conf::fp_type *b, conf::fp_type *result,
    int *precisionTypes, std::size_t *blockByteOffsets, int n, int m) {
  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(m, [=](auto &id) {
      const unsigned int i = id[0];

      conf::fp_type sum = 0.0;
      for (int k = 0; k < n; ++k) {
        // sum += A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
        //          static_cast<std::size_t>(k)] *
        //        b[k];
      }
      result[i] = sum;
    });
  });

  queue.wait();

  return event;
}
