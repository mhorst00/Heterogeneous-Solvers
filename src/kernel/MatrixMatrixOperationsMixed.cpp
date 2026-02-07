#include "MatrixMatrixOperationsMixed.hpp"
#include "Configuration.hpp"

using namespace sycl;
// Helper functions (must be device-callable)
inline conf::fp_type read_element(unsigned char *data, std::size_t offset, std::size_t idx,
                                  const int prec) {
  if (prec == 2) {
    auto *ptr = reinterpret_cast<sycl::half *>(data + offset);
    return static_cast<conf::fp_type>(ptr[idx]);
  } else if (prec == 4) {
    auto *ptr = reinterpret_cast<float *>(data + offset);
    return static_cast<conf::fp_type>(ptr[idx]);
  } else if (prec == 8) {
    auto *ptr = reinterpret_cast<double *>(data + offset);
    return static_cast<conf::fp_type>(ptr[idx]);
  }
  return conf::fp_type{0};
}

inline void write_element(unsigned char *data, std::size_t offset, std::size_t idx,
                          conf::fp_type value, const int prec) {
  if (prec == 2) {
    auto *ptr = reinterpret_cast<sycl::half *>(data + offset);
    ptr[idx] = static_cast<sycl::half>(value);
  } else if (prec == 4) {
    auto *ptr = reinterpret_cast<float *>(data + offset);
    ptr[idx] = static_cast<float>(value);
  } else if (prec == 8) {
    auto *ptr = reinterpret_cast<double *>(data + offset);
    ptr[idx] = value;
  }
}

sycl::event MatrixMatrixOperationsMixed::triangularSolve(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount) {
  // one work-group per rhs
  const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize, 1);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const std::size_t blockStartOffset_L = blockByteOffsets[blockID];
  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);
  const int blockPrec = precisionTypes[blockID];

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int group_id_i = nd_item.get_group().get_group_id(0);
      const int group_id_j = nd_item.get_group().get_group_id(1);

      // block in the matrix where the results are written
      const std::size_t block_id_B = blockID + (blockStart - blockRow) + group_id_i;
      const std::size_t blockStartOffset_B = blockByteOffsets[block_id_B];
      const int blockPrec_B = precisionTypes[block_id_B];

      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        // b_k = b_k/a_kk
        const conf::fp_type b_k =
            read_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + k,
                         blockPrec_B) /
            read_element(ABytes, blockStartOffset_L, k * matrixBlockSize + k, blockPrec);

        nd_item.barrier();

        if (local_i == 0) {
          write_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + k, b_k,
                        blockPrec_B);
        }

        if (local_i > k) {
          // b_i = b_i - a_ik*b_k
          conf::fp_type val =
              read_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + local_i,
                           blockPrec_B) -
              read_element(ABytes, blockStartOffset_L, local_i * matrixBlockSize + k, blockPrec) *
                  b_k;
          write_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + local_i, val,
                        blockPrec_B);
        }

        nd_item.barrier();
      }
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::triangularSolve_optimizedGPU(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount) {
  // one work-group per rhs
  const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize, 1);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const std::size_t blockStartOffset_L = blockByteOffsets[blockID];
  const int blockPrec_L = precisionTypes[blockID];
  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_column = local_accessor<conf::fp_type, 1>(matrixBlockSize, h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int group_id_i = nd_item.get_group().get_group_id(0);
      const int group_id_j = nd_item.get_group().get_group_id(1);

      // block in the matrix where the results are written
      const std::size_t block_id_B = blockID + (blockStart - blockRow) + group_id_i;
      const std::size_t blockStartOffset_B = blockByteOffsets[block_id_B];
      const int blockPrec_B = precisionTypes[block_id_B];

      // inverse of diagonal value in lower triangular matrix
      const conf::fp_type diagonal_ii =
          1.0 / read_element(ABytes, blockStartOffset_L, local_i * matrixBlockSize + local_i,
                             blockPrec_L);

      // current value of the position in the column that will be updated by
      // the work-item
      conf::fp_type value = read_element(ABytes, blockStartOffset_B,
                                         group_id_j * matrixBlockSize + local_i, blockPrec_B);

      // b_0 has to available for all work-items in the next iteration
      if (local_i == 0) {
        local_column[0] = value * diagonal_ii;
      }
      nd_item.barrier();

      // loop over columns in the lower triangular matrix
      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        if (local_i > k) {
          // b_i = b_i - a_ik*b_k
          value = value - read_element(ABytes, blockStartOffset_L, local_i * matrixBlockSize + k,
                                       blockPrec_L) *
                              local_column[k];
          if (local_i == k + 1) {
            // make b_{k+1} available to all work-items for the next iteration
            local_column[local_i] = value * diagonal_ii;
          }
        }

        // synchronize so that all work-items see b_{k+1} in the next iteration
        nd_item.barrier();
      }

      // store final value in global memory, also works for last entry since
      // value * diagonal_ii is recomputed
      write_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + local_i,
                    value * diagonal_ii, blockPrec_B);
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::triangularSolve_optimizedCPU(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount) {
  // one work-group per rhs
  const range globalRange(blockCount, conf::matrixBlockSize);

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  const std::size_t blockStartOffset_L = blockByteOffsets[blockID];
  const int blockPrec_L = precisionTypes[blockID];
  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(globalRange, [=](auto &id) {
      const int group_id_i = id[0];
      const int group_id_j = id[1];

      // block in the matrix where the results are written
      const std::size_t block_id_B = blockID + (blockStart - blockRow) + group_id_i;
      const std::size_t blockStartOffset_B = blockByteOffsets[block_id_B];
      const int blockPrec_B = precisionTypes[block_id_B];

      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        // b_k = b_k/a_kk
        const conf::fp_type b_k =
            read_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + k,
                         blockPrec_B) /
            read_element(ABytes, blockStartOffset_L, k * matrixBlockSize + k, blockPrec_L);

        write_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + k, b_k,
                      blockPrec_B);

#pragma clang loop vectorize(enable) unroll(enable)
        for (int j = k + 1; j < static_cast<int>(matrixBlockSize); ++j) {
          // b_i = b_i - a_ik*b_k
          conf::fp_type val =
              read_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + j,
                           blockPrec_B) -
              read_element(ABytes, blockStartOffset_L, j * matrixBlockSize + k, blockPrec_L) * b_k;
          write_element(ABytes, blockStartOffset_B, group_id_j * matrixBlockSize + j, val,
                        blockPrec_B);
        }
      }
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::symmetricMatrixMatrixDiagonal(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount,
    const int blockCountXY) {
  const int wgSize_xy = conf::workGroupSizeGEMM_xy;
  if (conf::matrixBlockSize % wgSize_xy != 0) {
    throw std::runtime_error("xy work-group dimension for matrix "
                             "multiplication must divide matrix block size");
  }

  const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

  const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
  const range localRange(wgSize_xy, wgSize_xy);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  // block count of all columns except the first one
  const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int local_j = nd_item.get_local_id(1);
      const int group_id_i = nd_item.get_group().get_group_id(0);
      const int group_id_j = nd_item.get_group().get_group_id(1);

      // block offset of current work group in column direction
      const int columnOffset = (blockStart - blockRow) + (group_id_i / wgCount_xy);

      // x/y block coordinate of the diagonal block processed by this work-group
      const int blockXYIndexDiagonal = blockRow + columnOffset;

      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = blockCountXY - (blockXYIndexDiagonal + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID_wg_diag = blockXYIndexDiagonal + referenceBlockCount - columnBlocksToRight;

      // id of the column block used in the matrix-matrix multiplication
      const int blockID_wg_col = blockID + columnOffset;

      const std::size_t block_wg_diag_offset = blockByteOffsets[blockID_wg_diag];
      const std::size_t block_wg_col_offset = blockByteOffsets[blockID_wg_col];

      const std::size_t block_wg_diag_prec = precisionTypes[blockID_wg_diag];
      const std::size_t block_wg_col_prec = precisionTypes[blockID_wg_col];

      // indices in of the current work-item in the matrix block
      const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
      const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

      const int i = internalBlockOffset_i + local_i;
      const int j = internalBlockOffset_j + local_j;

      if (i >= j) {
        // perform update for lower triangle of the diagonal
        for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
          // B_diag = B_diag - B_col * B_col^T
          conf::fp_type val = read_element(ABytes, block_wg_diag_offset, i * matrixBlockSize + j,
                                           block_wg_diag_prec) -
                              read_element(ABytes, block_wg_col_offset, i * matrixBlockSize + k,
                                           block_wg_col_prec) *
                                  read_element(ABytes, block_wg_col_offset, j * matrixBlockSize + k,
                                               block_wg_col_prec);
          write_element(ABytes, block_wg_diag_offset, i * matrixBlockSize + j, val,
                        block_wg_diag_prec);
        }
      }
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::symmetricMatrixMatrixDiagonal_optimizedGPU(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets, int blockID,
    int blockRow, int blockStart, int blockCount, int blockCountXY) {
  const int wgSize_xy = conf::workGroupSizeGEMM_xy;
  if (conf::matrixBlockSize % wgSize_xy != 0) {
    throw std::runtime_error("xy work-group dimension for matrix "
                             "multiplication must divide matrix block size");
  }

  const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

  const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);
  const range localRange(wgSize_xy, wgSize_xy);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  // block count of all columns except the first one
  const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_tile_A = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);
    auto local_tile_B = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int local_j = nd_item.get_local_id(1);
      const int group_id_i = nd_item.get_group().get_group_id(0);
      const int group_id_j = nd_item.get_group().get_group_id(1);

      // block offset of current work group in column direction
      const int columnOffset = (blockStart - blockRow) + (group_id_i / wgCount_xy);

      // x/y block coordinate of the diagonal block processed by this work-group
      const int blockXYIndexDiagonal = blockRow + columnOffset;

      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = blockCountXY - (blockXYIndexDiagonal + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID_wg_diag = blockXYIndexDiagonal + referenceBlockCount - columnBlocksToRight;

      // id of the column block used in the matrix-matrix multiplication
      const int blockID_wg_col = blockID + columnOffset;

      // start indices of blocks involved in the syrk update
      const std::size_t block_wg_diag_offset = blockByteOffsets[blockID_wg_diag];
      const std::size_t block_wg_col_offset = blockByteOffsets[blockID_wg_col];

      const std::size_t block_wg_diag_prec = precisionTypes[blockID_wg_diag];
      const std::size_t block_wg_col_prec = precisionTypes[blockID_wg_col];

      const int group_mod_count_i = (group_id_i % wgCount_xy);
      const int group_mod_count_j = (group_id_j % wgCount_xy);

      // indices in of the current work-item in the matrix block
      const int internalBlockOffset_i = group_mod_count_i * wgSize_xy;
      const int internalBlockOffset_j = group_mod_count_j * wgSize_xy;

      const int i = internalBlockOffset_i + local_i;
      const int j = internalBlockOffset_j + local_j;

      // load initial value for result
      conf::fp_type value = 0.0;
      if (i >= j) {
        value =
            read_element(ABytes, block_wg_diag_offset, i * matrixBlockSize + j, block_wg_diag_prec);
      }

      // perform update for lower triangle of the diagonal
      for (int t = 0; t < static_cast<int>(matrixBlockSize) / wgSize_xy; ++t) {
        // if tile is below diagonal or a diagonal tile, cache it in local
        // memory
        if (i >= j || group_mod_count_i == group_mod_count_j) {
          // normal block
          local_tile_A[local_i][local_j] =
              read_element(ABytes, block_wg_col_offset,
                           i * matrixBlockSize + t * wgSize_xy + local_j, block_wg_col_prec);

          // transposed block
          local_tile_B[local_i][local_j] =
              read_element(ABytes, block_wg_col_offset,
                           j * matrixBlockSize + t * wgSize_xy + local_i, block_wg_col_prec);
        }
        group_barrier(nd_item.get_group(), memory_scope::work_group);

        if (i >= j) {
          for (int k = 0; k < wgSize_xy; ++k) {
            // B_diag = B_diag - B_col * B_col^T
            value = value - local_tile_A[local_i][k] * local_tile_B[k][local_j];
          }
        }
        group_barrier(nd_item.get_group(), memory_scope::work_group);
      }
      if (i >= j) {
        write_element(ABytes, block_wg_diag_offset, i * matrixBlockSize + j, value,
                      block_wg_diag_prec);
      }
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::symmetricMatrixMatrixDiagonal_optimizedCPU(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount,
    const int blockCountXY) {
  const range globalRange(blockCount * conf::matrixBlockSize, conf::matrixBlockSize);

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  // block count of all columns except the first one
  const int referenceBlockCount = (blockCountXY * (blockCountXY - 1)) / 2;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(globalRange, [=](auto &id) {
      const int local_i = id[1];
      const int local_j = id[0];

      const int group_id = local_j / matrixBlockSize;

      // block offset of current work group in column direction
      const int columnOffset = (blockStart - blockRow) + group_id;

      // x/y block coordinate of the diagonal block processed by this work-group
      const int blockXYIndexDiagonal = blockRow + columnOffset;

      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = blockCountXY - (blockXYIndexDiagonal + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID_wg_diag = blockXYIndexDiagonal + referenceBlockCount - columnBlocksToRight;

      // id of the column block used in the matrix-matrix multiplication
      const int blockID_wg_col = blockID + columnOffset;

      // start indices of blocks involved in the syrk update
      const std::size_t block_wg_diag_offset = blockByteOffsets[blockID_wg_diag];
      const std::size_t block_wg_col_offset = blockByteOffsets[blockID_wg_col];

      const std::size_t block_wg_diag_prec = precisionTypes[blockID_wg_diag];
      const std::size_t block_wg_col_prec = precisionTypes[blockID_wg_col];

      const int i = local_i;
      const int j = local_j % matrixBlockSize;

      if (i >= j) {
        conf::fp_type value =
            read_element(ABytes, block_wg_diag_offset, i * matrixBlockSize + j, block_wg_diag_prec);
        // perform update for lower triangle of the diagonal
#pragma clang loop vectorize(enable) unroll(enable)
        for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
          // B_diag = B_diag - B_col * B_col^T
          value = value - read_element(ABytes, block_wg_col_offset, i * matrixBlockSize + k,
                                       block_wg_col_prec) *
                              read_element(ABytes, block_wg_col_offset, j * matrixBlockSize + k,
                                           block_wg_col_prec);
        }
        write_element(ABytes, block_wg_diag_offset, i * matrixBlockSize + j, value,
                      block_wg_diag_prec);
      }
    });
  });

  return event;
}

sycl::event
MatrixMatrixOperationsMixed::matrixMatrixStep(sycl::queue &queue, void *A, int *precisionTypes,
                                              std::size_t *blockByteOffsets, const int blockID,
                                              const int blockRow, const int blockStart,
                                              const int blockCount, const int blockCountXY) {
  const int wgSize_xy = conf::workGroupSizeGEMM_xy;
  if (conf::matrixBlockSize % wgSize_xy != 0) {
    throw std::runtime_error("xy work-group dimension for matrix "
                             "multiplication must divide matrix block size");
  }

  const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

  const int rowsAbove = blockStart - (blockRow + 2);
  const int rowsBelow = blockCountXY - blockStart - blockCount;

  // block Count including rows above and below that should not be processed
  const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

  const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
  const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
  const int lowerBlockCount =
      totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

  const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

  const range globalRange(wgCount_xy * wgSize_xy, wgCount * wgCount_xy * wgSize_xy);
  const range localRange(wgSize_xy, wgSize_xy);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int local_j = nd_item.get_local_id(1);
      const int group_id_i = nd_item.get_group().get_group_id(1);
      const int group_id_j = nd_item.get_group().get_group_id(0);

      // block ID of matrix blocks if one would enumerate them row by row
      const int rowBlockID = upperBlockCount + (group_id_i / wgCount_xy);

      // row ID in the lower triangle where the computation takes place
      const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

      const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

      // column ID of the matrix block int the lower triangle the current
      // work-group is associated with
      const int columnID = rowBlockID - blocksAboveCurrentRow;

      // calculation of the block ID of matrix block associated with this
      // work-group
      const int wgBlockID_A =
          blockID + blockCountXY - blockRow + columnID + 1 +
          (totalBlockCount - ((blockCountXY - blockRow - 2 - columnID) *
                              (blockCountXY - blockRow - 2 - columnID + 1) / 2)) +
          rowID - columnID;

      const int wgBlockID_B = blockID + rowID + 2;
      const int wgBlockID_C = blockID + columnID + 1;

      const std::size_t blockOffsetA = blockByteOffsets[wgBlockID_A];
      const std::size_t blockOffsetB = blockByteOffsets[wgBlockID_B];
      const std::size_t blockOffsetC = blockByteOffsets[wgBlockID_C];

      const std::size_t blockPrecA = precisionTypes[wgBlockID_A];
      const std::size_t blockPrecB = precisionTypes[wgBlockID_B];
      const std::size_t blockPrecC = precisionTypes[wgBlockID_C];

      // indices in of the current work-item in the matrix block
      const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
      const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

      const int i = internalBlockOffset_i + local_i;
      const int j = internalBlockOffset_j + local_j;

      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        // A = A - B * C^T
        conf::fp_type value =
            read_element(ABytes, blockOffsetA, i * matrixBlockSize + j, blockPrecA) -
            read_element(ABytes, blockOffsetB, i * matrixBlockSize + k, blockPrecB) *
                read_element(ABytes, blockOffsetC, j * matrixBlockSize + k, blockPrecC);
        write_element(ABytes, blockOffsetA, i * matrixBlockSize + j, value, blockPrecA);
      }
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::matrixMatrixStep_optimizedGPU(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets, int blockID,
    int blockRow, int blockStart, int blockCount, int blockCountXY) {
  const int wgSize_xy = conf::workGroupSizeGEMM_xy;
  if (conf::matrixBlockSize % wgSize_xy != 0) {
    throw std::runtime_error("xy work-group dimension for matrix "
                             "multiplication must divide matrix block size");
  }

  const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

  const int rowsAbove = blockStart - (blockRow + 2);
  const int rowsBelow = blockCountXY - blockStart - blockCount;

  // block Count including rows above and below that should not be processed
  const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

  const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
  const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
  const int lowerBlockCount =
      totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

  const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

  const range globalRange(wgCount_xy * wgSize_xy, wgCount * wgCount_xy * wgSize_xy);
  const range localRange(wgSize_xy, wgSize_xy);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_tile_B = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);
    auto local_tile_C = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy + 1), h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int local_j = nd_item.get_local_id(1);
      const int group_id_i = nd_item.get_group().get_group_id(1);
      const int group_id_j = nd_item.get_group().get_group_id(0);

      // block ID of matrix blocks if one would enumerate them row by row
      const int rowBlockID = upperBlockCount + (group_id_i / wgCount_xy);

      // row ID in the lower triangle where the computation takes place
      const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

      const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

      // column ID of the matrix block int the lower triangle the current
      // work-group is associated with
      const int columnID = rowBlockID - blocksAboveCurrentRow;

      // calculation of the block ID of matrix block associated with this
      // work-group
      const int wgBlockID_A =
          blockID + blockCountXY - blockRow + columnID + 1 +
          (totalBlockCount - ((blockCountXY - blockRow - 2 - columnID) *
                              (blockCountXY - blockRow - 2 - columnID + 1) / 2)) +
          rowID - columnID;

      const int wgBlockID_B = blockID + rowID + 2;
      const int wgBlockID_C = blockID + columnID + 1;

      const std::size_t blockOffsetA = blockByteOffsets[wgBlockID_A];
      const std::size_t blockOffsetB = blockByteOffsets[wgBlockID_B];
      const std::size_t blockOffsetC = blockByteOffsets[wgBlockID_C];

      const std::size_t blockPrecA = precisionTypes[wgBlockID_A];
      const std::size_t blockPrecB = precisionTypes[wgBlockID_B];
      const std::size_t blockPrecC = precisionTypes[wgBlockID_C];

      // indices in of the current work-item in the matrix block
      const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
      const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

      const int i = internalBlockOffset_i + local_i;
      const int j = internalBlockOffset_j + local_j;

      // i coordinate for matrix c that needs to be interpreted as transposed
      // later but is loaded non-transposed
      const int i_c = internalBlockOffset_j + local_i;

      // load initial value for result
      conf::fp_type value = read_element(ABytes, blockOffsetA, i * matrixBlockSize + j, blockPrecA);

      const std::size_t startIndexB = i * matrixBlockSize + local_j;
      const std::size_t startIndexC = i_c * matrixBlockSize + local_j;

      // perform update for lower triangle of the diagonal
      for (int t = 0; t < wgCount_xy; ++t) {
        // normal block
        local_tile_B[local_i][local_j] =
            read_element(ABytes, blockOffsetB, startIndexB + t * wgSize_xy, blockPrecB);

        // transposed block
        local_tile_C[local_i][local_j] =
            read_element(ABytes, blockOffsetC, startIndexC + t * wgSize_xy, blockPrecC);

        group_barrier(nd_item.get_group(), memory_scope::work_group);

#pragma unroll
        for (int k = 0; k < wgSize_xy; ++k) {
          // B_diag = B_diag - B_col * B_col^T
          value = value - local_tile_B[local_i][k] * local_tile_C[local_j][k];
        }
        group_barrier(nd_item.get_group(), memory_scope::work_group);
      }
      write_element(ABytes, blockOffsetA, i * matrixBlockSize + j, value, blockPrecA);
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::matrixMatrixStep_optimizedGPU2(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets, int blockID,
    int blockRow, int blockStart, int blockCount, int blockCountXY) {
  const int wgSize_xy = conf::workGroupSizeGEMM_xy;
  if (conf::matrixBlockSize % wgSize_xy != 0) {
    throw std::runtime_error("xy work-group dimension for matrix "
                             "multiplication must divide matrix block size");
  }

  const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgSize_xy;

  const int rowsAbove = blockStart - (blockRow + 2);
  const int rowsBelow = blockCountXY - blockStart - blockCount;

  // block Count including rows above and below that should not be processed
  const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

  const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
  const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
  const int lowerBlockCount =
      totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

  const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

  const range globalRange(wgCount_xy * wgSize_xy, wgCount * wgCount_xy * wgSize_xy);
  const range localRange(wgSize_xy, wgSize_xy);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_tile_B = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy), h);
    auto local_tile_C = local_accessor<conf::fp_type, 2>(sycl::range(wgSize_xy, wgSize_xy + 1), h);
    auto offset_cache = local_accessor<std::size_t, 1>(sycl::range(3), h);
    auto precision_cache = local_accessor<int, 1>(sycl::range(3), h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);
      const int local_j = nd_item.get_local_id(1);
      const int group_id_i = nd_item.get_group().get_group_id(1);
      const int group_id_j = nd_item.get_group().get_group_id(0);

      // row ID in the lower triangle where the computation takes place
      if (local_i == 0 && local_j == 0) {
        // block ID of matrix blocks if one would enumerate them row by row
        const int rowBlockID = upperBlockCount + (group_id_i / wgCount_xy);

        const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

        const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

        // column ID of the matrix block int the lower triangle the current
        // work-group is associated with
        const int columnID = rowBlockID - blocksAboveCurrentRow;

        // calculation of the block ID of matrix block associated with this
        // work-group
        const int wgBlockID_A =
            blockID + blockCountXY - blockRow + columnID + 1 +
            (totalBlockCount - ((blockCountXY - blockRow - 2 - columnID) *
                                (blockCountXY - blockRow - 2 - columnID + 1) / 2)) +
            rowID - columnID;

        const int wgBlockID_B = blockID + rowID + 2;
        const int wgBlockID_C = blockID + columnID + 1;

        offset_cache[0] = blockByteOffsets[wgBlockID_A];
        offset_cache[1] = blockByteOffsets[wgBlockID_B];
        offset_cache[2] = blockByteOffsets[wgBlockID_C];

        precision_cache[0] = precisionTypes[wgBlockID_A];
        precision_cache[1] = precisionTypes[wgBlockID_B];
        precision_cache[2] = precisionTypes[wgBlockID_C];
      }
      group_barrier(nd_item.get_group(), memory_scope::work_group);

      const std::size_t blockOffset_A = offset_cache[0];
      const std::size_t blockOffset_B = offset_cache[1];
      const std::size_t blockOffset_C = offset_cache[2];

      const int blockPrec_A = precision_cache[0];
      const int blockPrec_B = precision_cache[1];
      const int blockPrec_C = precision_cache[2];

      // indices in of the current work-item in the matrix block
      const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgSize_xy;
      const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgSize_xy;

      const int i = internalBlockOffset_i + local_i;
      const int j = internalBlockOffset_j + local_j;

      // i coordinate for matrix c that needs to be interpreted as transposed
      // later but is loaded non-transposed
      const int i_c = internalBlockOffset_j + local_i;

      // load initial value for result
      conf::fp_type value = 0;

      const std::size_t startIndexB = i * matrixBlockSize + local_j;
      const std::size_t startIndexC = i_c * matrixBlockSize + local_j;

      // perform update for lower triangle of the diagonal
      for (int t = 0; t < wgCount_xy; ++t) {
        // normal block
        local_tile_B[local_i][local_j] =
            read_element(ABytes, blockOffset_B, startIndexB + t * wgSize_xy, blockPrec_B);

        // transposed block
        local_tile_C[local_i][local_j] =
            read_element(ABytes, blockOffset_C, startIndexC + t * wgSize_xy, blockPrec_C);

        group_barrier(nd_item.get_group(), memory_scope::work_group);

#pragma unroll
        for (int k = 0; k < wgSize_xy; ++k) {
          // B_diag = B_diag - B_col * B_col^T
          value += local_tile_B[local_i][k] * local_tile_C[local_j][k];
        }
        group_barrier(nd_item.get_group(), memory_scope::work_group);
      }
      conf::fp_type tmp_val =
          read_element(ABytes, blockOffset_A, i * matrixBlockSize + j, blockPrec_A) - value;
      write_element(ABytes, blockOffset_A, i * matrixBlockSize + j, tmp_val, blockPrec_A);
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::matrixMatrixStep_optimizedGPU3(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets, int blockID,
    int blockRow, int blockStart, int blockCount, int blockCountXY) {
  const int wgSize_xy = 16;

  const int valuesPerWorkItem_xy = 4;

  const int wgBlockSize_xy = wgSize_xy * valuesPerWorkItem_xy;

  const int sharedMemBlockSize_x = 16;
  const int sharedMemBlockSize_y = wgSize_xy * valuesPerWorkItem_xy;

  const int sharedMemBlockCount = static_cast<int>(conf::matrixBlockSize) / sharedMemBlockSize_x;

  if (conf::matrixBlockSize % wgSize_xy != 0 || conf::matrixBlockSize % sharedMemBlockSize_x != 0 ||
      conf::matrixBlockSize % sharedMemBlockSize_y != 0) {
    throw std::runtime_error("xy work-group dimension for matrix multiplication must divide matrix "
                             "block size and shared memory block size");
  }

  const int wgCount_xy = static_cast<int>(conf::matrixBlockSize) / wgBlockSize_xy;

  const int rowsAbove = blockStart - (blockRow + 2);
  const int rowsBelow = blockCountXY - blockStart - blockCount;

  // block Count including rows above and below that should not be processed
  const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

  const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
  const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
  const int lowerBlockCount =
      totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

  const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

  const range globalRange(wgCount_xy * wgSize_xy, wgCount * wgCount_xy * wgSize_xy);
  const range localRange(wgSize_xy, wgSize_xy);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);
  std::cout << "matrixMatrixStep_optimizedGPU3: Starting queue part..." << std::endl;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_tile_B = local_accessor<conf::fp_type, 2>(
        sycl::range(sharedMemBlockSize_y, sharedMemBlockSize_x + 1), h);
    auto local_tile_C = local_accessor<conf::fp_type, 2>(
        sycl::range(sharedMemBlockSize_x, sharedMemBlockSize_y + 1), h);
    auto offset_cache = local_accessor<std::size_t, 1>(sycl::range(3), h);
    auto precision_cache = local_accessor<int, 1>(sycl::range(3), h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(1);
      const int local_j = nd_item.get_local_id(0);
      const int group_id_i = nd_item.get_group().get_group_id(1);
      const int group_id_j = nd_item.get_group().get_group_id(0);

      // row ID in the lower triangle where the computation takes place
      if (local_i == 0 && local_j == 0) {
        // block ID of matrix blocks if one would enumerate them row by row
        const int rowBlockID = upperBlockCount + (group_id_i / wgCount_xy);

        const int rowID = static_cast<int>((-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2);

        const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

        // column ID of the matrix block int the lower triangle the current
        // work-group is associated with
        const int columnID = rowBlockID - blocksAboveCurrentRow;

        // calculation of the block ID of matrix block associated with this
        // work-group
        const int wgBlockID_A =
            blockID + blockCountXY - blockRow + columnID + 1 +
            (totalBlockCount - ((blockCountXY - blockRow - 2 - columnID) *
                                (blockCountXY - blockRow - 2 - columnID + 1) / 2)) +
            rowID - columnID;

        const int wgBlockID_B = blockID + rowID + 2;
        const int wgBlockID_C = blockID + columnID + 1;

        offset_cache[0] = blockByteOffsets[wgBlockID_A];
        offset_cache[1] = blockByteOffsets[wgBlockID_B];
        offset_cache[2] = blockByteOffsets[wgBlockID_C];

        precision_cache[0] = precisionTypes[wgBlockID_A];
        precision_cache[1] = precisionTypes[wgBlockID_B];
        precision_cache[2] = precisionTypes[wgBlockID_C];
      }
      group_barrier(nd_item.get_group(), memory_scope::work_group);

      const std::size_t blockOffset_A = offset_cache[0];
      const std::size_t blockOffset_B = offset_cache[1];
      const std::size_t blockOffset_C = offset_cache[2];

      const int blockPrec_A = precision_cache[0];
      const int blockPrec_B = precision_cache[1];
      const int blockPrec_C = precision_cache[2];

      // indices in of the current work-item in the matrix block
      const int internalBlockOffset_i = (group_id_i % wgCount_xy) * wgBlockSize_xy;
      const int internalBlockOffset_j = (group_id_j % wgCount_xy) * wgBlockSize_xy;

      // upper left index of the valuesPerWorkItem_xy x valuesPerWorkItem_xy
      // tile of this work-item in the result matrix
      const int i = internalBlockOffset_i + local_i * valuesPerWorkItem_xy;
      const int j = internalBlockOffset_j + local_j * valuesPerWorkItem_xy;

      // load initial value for result
      conf::fp_type workItemTile[4][4];

      for (int ii = 0; ii < valuesPerWorkItem_xy; ++ii) {
        for (int jj = 0; jj < valuesPerWorkItem_xy; ++jj) {
          workItemTile[ii][jj] = conf::fp_type{0};
        }
      }

      const std::size_t startIndexB = internalBlockOffset_i * matrixBlockSize;
      const std::size_t startIndexC = internalBlockOffset_j * matrixBlockSize;

      // perform update for lower triangle of the diagonal
      for (int t = 0; t < sharedMemBlockCount; ++t) {
        // normal block
        for (int s = 0; s < valuesPerWorkItem_xy; ++s) {
          local_tile_B[local_i + s * wgSize_xy][local_j] =
              read_element(ABytes, blockOffset_B,
                           startIndexB + t * sharedMemBlockSize_x + local_i * matrixBlockSize +
                               local_j + s * wgSize_xy * matrixBlockSize,
                           blockPrec_B);
        }

        // transposed block
        for (int s = 0; s < valuesPerWorkItem_xy; ++s) {
          local_tile_C[local_i][local_j + s * wgSize_xy] =
              read_element(ABytes, blockOffset_C,
                           startIndexC + t * sharedMemBlockSize_x + local_j * matrixBlockSize +
                               local_i + s * wgSize_xy * matrixBlockSize,
                           blockPrec_C);
        }

        group_barrier(nd_item.get_group(), memory_scope::work_group);

        for (int k = 0; k < sharedMemBlockSize_x; ++k) {
          for (int ii = 0; ii < valuesPerWorkItem_xy; ++ii) {
            for (int jj = 0; jj < valuesPerWorkItem_xy; ++jj) {
              workItemTile[ii][jj] += local_tile_B[local_i * valuesPerWorkItem_xy + ii][k] *
                                      local_tile_C[k][local_j * valuesPerWorkItem_xy + jj];
            }
          }
        }

        group_barrier(nd_item.get_group(), memory_scope::work_group);
      }

      // store the result
      conf::fp_type tmp_val;
      for (int ii = 0; ii < valuesPerWorkItem_xy; ++ii) {
        for (int jj = 0; jj < valuesPerWorkItem_xy; ++jj) {
          tmp_val = read_element(ABytes, blockOffset_A, (i + ii) * matrixBlockSize + (j + jj),
                                 blockPrec_A) -
                    workItemTile[ii][jj];
          write_element(ABytes, blockOffset_A, (i + ii) * matrixBlockSize + (j + jj), tmp_val,
                        blockPrec_A);
        }
      }
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::matrixMatrixStep_optimizedCPU(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount,
    const int blockCountXY) {
  const int rowsAbove = blockStart - (blockRow + 2);
  const int rowsBelow = blockCountXY - blockStart - blockCount;

  // block Count including rows above and below that should not be processed
  const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

  const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
  const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
  const int lowerBlockCount =
      totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

  const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

  const range globalRange(wgCount * conf::matrixBlockSize, conf::matrixBlockSize);

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(globalRange, [=](auto &id) {
      const int local_i = id[1];
      const int local_j = id[0];

      const int group_id = local_j / matrixBlockSize;

      // block ID of matrix blocks if one would enumerate them row by row
      const int rowBlockID = upperBlockCount + group_id;

      // row ID in the lower triangle where the computation takes place
      const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

      const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

      // column ID of the matrix block int the lower triangle the current
      // work-group is associated with
      const int columnID = rowBlockID - blocksAboveCurrentRow;

      // calculation of the block ID of matrix block associated with this
      // work-group
      const int wgBlockID_A =
          blockID + blockCountXY - blockRow + columnID + 1 +
          (totalBlockCount - ((blockCountXY - blockRow - 2 - columnID) *
                              (blockCountXY - blockRow - 2 - columnID + 1) / 2)) +
          rowID - columnID;

      const int wgBlockID_B = blockID + rowID + 2;
      const int wgBlockID_C = blockID + columnID + 1;

      const std::size_t blockOffsetA = blockByteOffsets[wgBlockID_A];
      const std::size_t blockOffsetB = blockByteOffsets[wgBlockID_B];
      const std::size_t blockOffsetC = blockByteOffsets[wgBlockID_C];

      const std::size_t blockPrecA = precisionTypes[wgBlockID_A];
      const std::size_t blockPrecB = precisionTypes[wgBlockID_B];
      const std::size_t blockPrecC = precisionTypes[wgBlockID_C];

      const int i = local_i;
      const int j = local_j % matrixBlockSize;

      conf::fp_type value = read_element(ABytes, blockOffsetA, i * matrixBlockSize + j, blockPrecA);
#pragma clang loop vectorize(enable) unroll(enable)
      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        // A = A - B * C^T
        value = value - read_element(ABytes, blockOffsetB, i * matrixBlockSize + k, blockPrecB) *
                            read_element(ABytes, blockOffsetC, j * matrixBlockSize + k, blockPrecC);
      }

      write_element(ABytes, blockOffsetA, i * matrixBlockSize + j, value, blockPrecA);
    });
  });

  return event;
}

sycl::event MatrixMatrixOperationsMixed::matrixMatrixStep_optimizedCPU2(
    sycl::queue &queue, void *A, int *precisionTypes, std::size_t *blockByteOffsets,
    const int blockID, const int blockRow, const int blockStart, const int blockCount,
    const int blockCountXY) {
  const int rowsAbove = blockStart - (blockRow + 2);
  const int rowsBelow = blockCountXY - blockStart - blockCount;

  // block Count including rows above and below that should not be processed
  const int virtualBlockCount = blockCount + rowsAbove + rowsBelow;

  const int upperBlockCount = ((rowsAbove * (rowsAbove + 1)) / 2);
  const int totalBlockCount = (virtualBlockCount * (virtualBlockCount + 1)) / 2;
  const int lowerBlockCount =
      totalBlockCount - ((blockCount + rowsAbove) * ((blockCount + rowsAbove) + 1) / 2);

  const int wgCount = totalBlockCount - upperBlockCount - lowerBlockCount;

  const range globalRange(wgCount * conf::matrixBlockSize);

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(globalRange, [=](auto &id) {
      const int local_i = id[0];

      const int group_id = local_i / matrixBlockSize;

      // block ID of matrix blocks if one would enumerate them row by row
      const int rowBlockID = upperBlockCount + group_id;

      // row ID in the lower triangle where the computation takes place
      const int rowID = (-1.0 + sycl::sqrt(1.0 + 8.0 * rowBlockID)) / 2;

      const int blocksAboveCurrentRow = rowID * (rowID + 1) / 2;

      // column ID of the matrix block int the lower triangle the current
      // work-group is associated with
      const int columnID = rowBlockID - blocksAboveCurrentRow;

      // calculation of the block ID of matrix block associated with this
      // work-group
      const int wgBlockID_A =
          blockID + blockCountXY - blockRow + columnID + 1 +
          (totalBlockCount - ((blockCountXY - blockRow - 2 - columnID) *
                              (blockCountXY - blockRow - 2 - columnID + 1) / 2)) +
          rowID - columnID;

      const int wgBlockID_B = blockID + rowID + 2;
      const int wgBlockID_C = blockID + columnID + 1;

      const std::size_t blockOffsetA = blockByteOffsets[wgBlockID_A];
      const std::size_t blockOffsetB = blockByteOffsets[wgBlockID_B];
      const std::size_t blockOffsetC = blockByteOffsets[wgBlockID_C];

      const std::size_t blockPrecA = precisionTypes[wgBlockID_A];
      const std::size_t blockPrecB = precisionTypes[wgBlockID_B];
      const std::size_t blockPrecC = precisionTypes[wgBlockID_C];

      const int i = local_i % matrixBlockSize;

      for (int j = 0; j < static_cast<int>(matrixBlockSize); ++j) {
        conf::fp_type value = 0;
#pragma clang loop vectorize(enable) unroll(enable)
        for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
          // A = A - B * C^T
          value += read_element(ABytes, blockOffsetB, j * matrixBlockSize + k, blockPrecB) *
                   read_element(ABytes, blockOffsetC, i * matrixBlockSize + k, blockPrecC);
        }
        conf::fp_type tmp_val =
            read_element(ABytes, blockOffsetA, j * matrixBlockSize + i, blockPrecA) - value;
        write_element(ABytes, blockOffsetA, j * matrixBlockSize + i, tmp_val, blockPrecA);
      }
    });
  });

  return event;
}
