#include "MatrixOperationsMixed.hpp"
#include "Configuration.hpp"

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T>
sycl::event MatrixOperationsMixed::cholesky(sycl::queue &queue, T *A,
                                            const std::size_t blockStartOffset,
                                            const int blockRow) {
  // launch kernel with the size of exactly one work-group
  const range globalRange(conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  // Move to start of block using using offset
  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);
  T *ATyped = reinterpret_cast<T *>(ABytes + blockStartOffset);

  const std::size_t N = conf::N;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.parallel_for(kernelRange, [=](auto &nd_item) {
      int local_i = nd_item.get_local_id(0);

      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        T diagVal = ATyped[k * matrixBlockSize + k];
        if (diagVal <= T{0}) {
          diagVal = T{1e-10};
        }
        const T sqrtDiag = sycl::sqrt(diagVal);
        // a_kk = sqrt(a_kk)

        if ((blockRow * matrixBlockSize + local_i) < N) {
          // update column below diagonal value
          if (local_i > k) {
            ATyped[local_i * matrixBlockSize + k] =
                ATyped[local_i * matrixBlockSize + k] / sqrtDiag;
          }
        }

        nd_item.barrier();

        if (local_i == k) {
          ATyped[local_i * matrixBlockSize + k] = sqrtDiag;
        }

        if ((blockRow * matrixBlockSize + local_i) < N) {
          // process lower triangle right to the updated column
          for (int j = k + 1; j < static_cast<int>(matrixBlockSize); ++j) {
            if (local_i >= j) {
              const T A_ik = ATyped[local_i * matrixBlockSize + k];
              const T A_jk = ATyped[j * matrixBlockSize + k];
              ATyped[local_i * matrixBlockSize + j] =
                  ATyped[local_i * matrixBlockSize + j] - A_ik * A_jk;
            } else {
              ATyped[local_i * matrixBlockSize + j] = 0;
            }
          }
        }

        nd_item.barrier();
      }
    });
  });

  return event;
}

template <typename T>
sycl::event MatrixOperationsMixed::cholesky_GPU(sycl::queue &queue, T *A,
                                                std::size_t blockStartOffset, int blockRow) {
  // launch kernel with the size of exactly one work-group
  const range globalRange(conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  // Move to start of block using using offset
  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);
  T *ATyped = reinterpret_cast<T *>(ABytes + blockStartOffset);

  const std::size_t N = conf::N;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_column = local_accessor<T, 1>(conf::matrixBlockSize, h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int local_i = nd_item.get_local_id(0);

      for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
        T diagVal = ATyped[k * matrixBlockSize + k];
        if (diagVal <= T{0}) {
          diagVal = T{1e-10};
        }
        const T sqrtDiag = sycl::sqrt(diagVal);

        T A_ik = 0.0;

        if ((blockRow * matrixBlockSize + local_i) < N) {
          // update column below diagonal value
          if (local_i > k) {
            A_ik = ATyped[local_i * matrixBlockSize + k] / sqrtDiag;
            ATyped[local_i * matrixBlockSize + k] = A_ik;
            local_column[local_i] = A_ik; // store value of column in local memory
          }
        }

        nd_item.barrier();

        // a_kk = sqrt(a_kk)
        if (local_i == k) {
          ATyped[local_i * matrixBlockSize + k] = sqrtDiag;
        }

        if ((blockRow * matrixBlockSize + local_i) < N) {
          // process lower triangle right to the updated column
          for (int j = k + 1; j < static_cast<int>(matrixBlockSize); ++j) {
            if (local_i >= j) {
              const T A_jk = local_column[j];
              ATyped[local_i * matrixBlockSize + j] =
                  ATyped[local_i * matrixBlockSize + j] - A_ik * A_jk;
            } else {
              ATyped[local_i * matrixBlockSize + j] = 0;
            }
          }
        }

        nd_item.barrier();
      }
    });
  });

  return event;
}

template <typename T>
sycl::event MatrixOperationsMixed::cholesky_optimizedGPU(sycl::queue &queue, T *A,
                                                         std::size_t blockStartOffset,
                                                         int blockRow) {
  // launch kernel with the size of exactly one work-group
  const range globalRange(conf::matrixBlockSize * conf::matrixBlockSize);
  const range localRange(conf::matrixBlockSize);
  const auto kernelRange = nd_range{globalRange, localRange};

  const std::size_t matrixBlockSize = conf::matrixBlockSize;

  // Move to start of block using using offset
  unsigned char *ABytes = reinterpret_cast<unsigned char *>(A);
  T *ATyped = reinterpret_cast<T *>(ABytes + blockStartOffset);

  const std::size_t N = conf::N;

  sycl::event event = queue.submit([&](sycl::handler &h) {
    auto local_column = local_accessor<T, 1>(conf::matrixBlockSize, h);

    h.parallel_for(kernelRange, [=](auto &nd_item) {
      const int group_id = nd_item.get_group().get_group_id();
      const int local_i = nd_item.get_local_id(0);
      if (group_id == 0) {
        for (int k = 0; k < static_cast<int>(matrixBlockSize); ++k) {
          T A_ik = 0.0;

          if ((blockRow * matrixBlockSize + local_i) < N) {
            // update column below diagonal value
            if (local_i > k) {
              T diagVal = ATyped[k * matrixBlockSize + k];
              if (diagVal <= T{0}) {
                diagVal = T{1e-10};
              }
              const T sqrtDiag = sycl::sqrt(diagVal);
              A_ik = ATyped[local_i * matrixBlockSize + k] / sqrtDiag;
              ATyped[local_i * matrixBlockSize + k] = A_ik;
              local_column[local_i] = A_ik; // store value of column in local memory
            }
          }

          nd_item.barrier();

          if ((blockRow * matrixBlockSize + local_i) < N) {
            // process lower triangle right to the updated column
            for (int j = k + 1; j < static_cast<int>(matrixBlockSize); ++j) {
              if (local_i >= j) {
                const T A_jk = local_column[j];
                ATyped[local_i * matrixBlockSize + j] =
                    ATyped[local_i * matrixBlockSize + j] - A_ik * A_jk;
              }
            }
          }

          // a_kk = sqrt(a_kk)
          if (local_i == 0) {
            T diagVal = ATyped[k * matrixBlockSize + k];
            if (diagVal <= T{0}) {
              diagVal = T{1e-10};
            }
            const T sqrtDiag = sycl::sqrt(diagVal);
            ATyped[k * matrixBlockSize + k] = sqrtDiag;
          }

          nd_item.barrier();
        }
      } else {
        // set upper triangle to 0
        if ((blockRow * matrixBlockSize + local_i) < N) {
          for (int j = group_id; j < static_cast<int>(matrixBlockSize); ++j) {
            if (local_i < j) {
              ATyped[local_i * matrixBlockSize + j] = 0;
            }
          }
        }
      }
    });
  });

  return event;
}
// Explicit instantiations - Add types here as required
template sycl::event MatrixOperationsMixed::cholesky<sycl::half>(sycl::queue &, sycl::half *,
                                                                 size_t, int);
template sycl::event MatrixOperationsMixed::cholesky<float>(sycl::queue &, float *, size_t, int);
template sycl::event MatrixOperationsMixed::cholesky<double>(sycl::queue &, double *, size_t, int);

template sycl::event MatrixOperationsMixed::cholesky_GPU<sycl::half>(sycl::queue &, sycl::half *,
                                                                     size_t, int);
template sycl::event MatrixOperationsMixed::cholesky_GPU<float>(sycl::queue &, float *, size_t,
                                                                int);
template sycl::event MatrixOperationsMixed::cholesky_GPU<double>(sycl::queue &, double *, size_t,
                                                                 int);

template sycl::event
MatrixOperationsMixed::cholesky_optimizedGPU<sycl::half>(sycl::queue &, sycl::half *, size_t, int);
template sycl::event MatrixOperationsMixed::cholesky_optimizedGPU<float>(sycl::queue &, float *,
                                                                         size_t, int);
template sycl::event MatrixOperationsMixed::cholesky_optimizedGPU<double>(sycl::queue &, double *,
                                                                          size_t, int);
