#include "MatrixGenerator.hpp"
#include "Configuration.hpp"
#include "SymmetricMatrixMixed.hpp"
#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <vector>
using namespace sycl;

SymmetricMatrix
MatrixGenerator::generateSPDMatrixStrictDiagonalDominant(sycl::queue &queue) {
  SymmetricMatrix matrix(conf::N, conf::matrixBlockSize, queue);

  // block count of all columns except the first one
  const int referenceBlockCount =
      (matrix.blockCountXY * (matrix.blockCountXY - 1)) / 2;

  // random number generator
  std::random_device rd;
  std::mt19937 generator(123);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  for (std::size_t block_i = 0;
       block_i < static_cast<std::size_t>(matrix.blockCountXY); ++block_i) {
    for (std::size_t block_j = 0; block_j <= block_i; ++block_j) {
      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = matrix.blockCountXY - (block_j + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID = block_i + referenceBlockCount - columnBlocksToRight;

      // start index of block in matrix data structure
      const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) *
                                          conf::matrixBlockSize *
                                          conf::matrixBlockSize;

      if (block_i == block_j) {
        // Diagonal block
        for (std::size_t i = 0; i < static_cast<std::size_t>(matrix.blockSize);
             ++i) {
          for (std::size_t j = 0; j <= i; ++j) {
            if (block_i * conf::matrixBlockSize + i < conf::N &&
                block_j * conf::matrixBlockSize + j < conf::N) {
              const conf::fp_type value = distribution(generator);
              if (i == j) {
                matrix.matrixData[blockStartIndex + i * conf::matrixBlockSize +
                                  j] =
                    std::abs(value) + static_cast<conf::fp_type>(conf::N);
              } else {
                // location (i,j)
                matrix.matrixData[blockStartIndex + i * conf::matrixBlockSize +
                                  j] = value;
                // mirrored value in upper triangle (j,i)
                matrix.matrixData[blockStartIndex + j * conf::matrixBlockSize +
                                  i] = value;
              }
            }
          }
        }
      } else {
        // Non-diagonal block
        for (std::size_t i = 0; i < static_cast<std::size_t>(matrix.blockSize);
             ++i) {
          for (std::size_t j = 0;
               j < static_cast<std::size_t>(matrix.blockSize); ++j) {
            if (block_i * conf::matrixBlockSize + i < conf::N &&
                block_j * conf::matrixBlockSize + j < conf::N) {
              const conf::fp_type value = distribution(generator);
              matrix
                  .matrixData[blockStartIndex + i * conf::matrixBlockSize + j] =
                  value;
            }
          }
        }
      }
    }
  }

  return matrix;
}

SymmetricMatrix MatrixGenerator::generateSPDMatrix(std::string &path,
                                                   sycl::queue &queue,
                                                   sycl::queue &queueGPU) {
  std::cout << "-- generating SPD matrix of size " << conf::N << "x" << conf::N
            << std::endl;
  SymmetricMatrix matrix(conf::N, conf::matrixBlockSize, queueGPU);

  std::size_t nRegressors = 8;
  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>>
      trainingInput{usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
  std::size_t offset = nRegressors - 1;
  trainingInput.resize(conf::N + offset);

  conf::fp_type *matrixData = matrix.matrixData.data();
  conf::fp_type *trainingInputData = trainingInput.data();

  std::ifstream dataInputStream(path);
  std::string valueString;

  // parse input data
  std::size_t rowIndex = 0;
  while (std::getline(dataInputStream, valueString)) {
    conf::fp_type value = static_cast<conf::fp_type>(std::stod(valueString));
    trainingInput[rowIndex + offset] = value;
    rowIndex++;
    if (rowIndex == conf::N) {
      break;
    }
  }
  dataInputStream.close();

  if (rowIndex != conf::N) {
    throw std::runtime_error("Not enough data available!");
  }

  // block count of all columns except the first one
  const int referenceBlockCount =
      (matrix.blockCountXY * (matrix.blockCountXY - 1)) / 2;

  std::size_t N = conf::N;
  for (std::size_t i_block = 0;
       i_block < static_cast<std::size_t>(matrix.blockCountXY); ++i_block) {
    for (std::size_t j_block = 0; j_block <= i_block; j_block++) {
      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = matrix.blockCountXY - (j_block + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID = i_block + referenceBlockCount - columnBlocksToRight;

      // start index of block in matrix data structure
      const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) *
                                          conf::matrixBlockSize *
                                          conf::matrixBlockSize;

      // Diagonal noise for stability
      const double noiseVariance = 0.01;

      // Default values for hyperparameters
      const double verticalLengthscale = 1.0;
      const double lengthscale = 1.0;

      std::size_t matrixBlockSize = conf::matrixBlockSize;

      queue.submit([&](handler &h) {
        h.parallel_for(
            range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
              const unsigned int i_local = idx[0];
              const unsigned int j_local = idx[1];
              const unsigned long i_global =
                  matrixBlockSize * i_block + i_local;
              const unsigned long j_global =
                  matrixBlockSize * j_block + j_local;
              if (i_global >= N || j_global >= N) {
                return;
              }
              double distance = 0.0;
              for (unsigned int k = 0; k < nRegressors; k++) {
                const double tmp = trainingInputData[i_global + k] -
                                   trainingInputData[j_global + k];
                distance += tmp * tmp;
              }
              double covarianceFunction =
                  verticalLengthscale *
                  sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

              if (i_global == j_global) {
                covarianceFunction += noiseVariance;
              }
              matrixData[blockStartIndex + i_local * matrixBlockSize +
                         j_local] = covarianceFunction;
            });
      });
    }
    queue.wait();
  }
  queue.wait();

  return matrix;
}

SymmetricMatrixMixed
MatrixGenerator::generateSPDMatrixMixed(std::string &path, sycl::queue &queue,
                                        sycl::queue &queueGPU) {
  std::cout << "-- generating mixed precision SPD matrix of size " << conf::N
            << "x" << conf::N << std::endl;

  // Build and allocate matrix memory
  SymmetricMatrixMixed matrix(conf::N, conf::matrixBlockSize, queueGPU);

  int fp16_blocks = 0;
  int fp32_blocks = 0;
  int fp64_blocks = 0;

  const int boundary0 = std::ceil(matrix.blockCountXY / 2);
  const int boundary1 = std::ceil(matrix.blockCountXY / 4);

  // Build testing mixed precision vector
  std::vector<Precision> precisionVector{};
  precisionVector.resize(matrix.blockCountXY * (matrix.blockCountXY + 1) / 2);
  int continuous_index = 0;
  int distance = 0;
  for (int col = 0; col < matrix.blockCountXY; ++col) {
    for (int row = col; row < matrix.blockCountXY; ++row) {
      distance = std::abs(row - col);
      if (distance < boundary0) {
        precisionVector[continuous_index] = Precision::FP64;
        fp64_blocks++;
      } else if (distance < boundary1) {
        precisionVector[continuous_index] = Precision::FP32;
        fp32_blocks++;
      } else {
        precisionVector[continuous_index] = Precision::FP16;
        fp32_blocks++;
      }
      continuous_index++;
    }
  }

  std::cout << "FP64, FP32, FP16 blocks: (" << fp64_blocks << "," << fp32_blocks
            << "," << fp16_blocks << ")" << std::endl;

  // Prepare memory pointers for precision index data
  std::size_t *blockByteOffsets = matrix.blockByteOffsets.data();
  int *precisionTypes = matrix.precisionTypes.data();

  // Calculate mixed precision block memory byte offsets
  std::size_t cumulative_offset = 0;
  for (std::size_t i = 0; i < precisionVector.size(); ++i) {
    blockByteOffsets[i] = cumulative_offset;

    std::size_t elementByteSize = 0;
    if (precisionVector[i] == Precision::FP16) {
      elementByteSize = sizeof(sycl::half);
      precisionTypes[i] = 2;
    } else if (precisionVector[i] == Precision::FP32) {
      elementByteSize = sizeof(float);
      precisionTypes[i] = 4;
    } else if (precisionVector[i] == Precision::FP64) {
      elementByteSize = sizeof(double);
      precisionTypes[i] = 8;
    }

    cumulative_offset +=
        elementByteSize * conf::matrixBlockSize * conf::matrixBlockSize;
  }

  // Build temporary input data vector
  std::size_t nRegressors = 8;
  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>>
      trainingInput{usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
  std::size_t offset = nRegressors - 1;
  trainingInput.resize(conf::N + offset);

  // Prepare memory pointers
  unsigned char *matrixBytes =
      reinterpret_cast<unsigned char *>(matrix.matrixData.data());
  conf::fp_type *trainingInputData = trainingInput.data();

  std::ifstream dataInputStream(path);
  std::string valueString;

  // parse input data
  std::size_t rowIndex = 0;
  while (std::getline(dataInputStream, valueString)) {
    conf::fp_type value = static_cast<conf::fp_type>(std::stod(valueString));
    trainingInput[rowIndex + offset] = value;
    rowIndex++;
    if (rowIndex == conf::N) {
      break;
    }
  }
  dataInputStream.close();

  if (rowIndex != conf::N) {
    throw std::runtime_error("Not enough data available!");
  }

  // block count of all columns except the first one
  const int referenceBlockCount =
      (matrix.blockCountXY * (matrix.blockCountXY - 1)) / 2;

  std::size_t N = conf::N;
  for (std::size_t i_block = 0;
       i_block < static_cast<std::size_t>(matrix.blockCountXY); ++i_block) {
    for (std::size_t j_block = 0; j_block <= i_block; j_block++) {
      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = matrix.blockCountXY - (j_block + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID = i_block + referenceBlockCount - columnBlocksToRight;

      const std::size_t byteOffset = blockByteOffsets[blockID];

      // start index of block in matrix data structure
      const std::size_t blockStartIndex = static_cast<std::size_t>(blockID) *
                                          conf::matrixBlockSize *
                                          conf::matrixBlockSize;

      // Diagonal noise for stability
      const double noiseVariance = 0.01;

      // Default values for hyperparameters
      const double verticalLengthscale = 1.0;
      const double lengthscale = 1.0;

      std::size_t matrixBlockSize = conf::matrixBlockSize;

      if (precisionTypes[blockID] == 2) {
        queue.submit([&](handler &h) {
          h.parallel_for(
              range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
                // Add correct typing to both arrays according to block index
                sycl::half *matrixDataTyped =
                    reinterpret_cast<sycl::half *>(matrixBytes + byteOffset);
                const unsigned int i_local = idx[0];
                const unsigned int j_local = idx[1];
                const unsigned long i_global =
                    matrixBlockSize * i_block + i_local;
                const unsigned long j_global =
                    matrixBlockSize * j_block + j_local;
                if (i_global >= N || j_global >= N) {
                  return;
                }
                sycl::half distance = 0.0;
                for (unsigned int k = 0; k < nRegressors; k++) {
                  const sycl::half tmp = trainingInputData[i_global + k] -
                                         trainingInputData[j_global + k];
                  distance += tmp * tmp;
                }
                sycl::half covarianceFunction =
                    verticalLengthscale *
                    sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

                if (i_global == j_global) {
                  covarianceFunction += noiseVariance;
                }
                matrixDataTyped[i_local * matrixBlockSize + j_local] =
                    covarianceFunction;
              });
        });

      } else if (precisionTypes[blockID] == 4) {
        queue.submit([&](handler &h) {
          h.parallel_for(
              range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
                // Add correct typing to both arrays according to block index
                float *matrixDataTyped =
                    reinterpret_cast<float *>(matrixBytes + byteOffset);
                const unsigned int i_local = idx[0];
                const unsigned int j_local = idx[1];
                const unsigned long i_global =
                    matrixBlockSize * i_block + i_local;
                const unsigned long j_global =
                    matrixBlockSize * j_block + j_local;
                if (i_global >= N || j_global >= N) {
                  return;
                }
                float distance = 0.0;
                for (unsigned int k = 0; k < nRegressors; k++) {
                  const float tmp = trainingInputData[i_global + k] -
                                    trainingInputData[j_global + k];
                  distance += tmp * tmp;
                }
                float covarianceFunction =
                    verticalLengthscale *
                    sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

                if (i_global == j_global) {
                  covarianceFunction += noiseVariance;
                }
                matrixDataTyped[i_local * matrixBlockSize + j_local] =
                    covarianceFunction;
              });
        });

      } else if (precisionTypes[blockID] == 8) {
        queue.submit([&](handler &h) {
          h.parallel_for(
              range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
                // Add correct typing to both arrays according to block index
                double *matrixDataTyped =
                    reinterpret_cast<double *>(matrixBytes + byteOffset);
                const unsigned int i_local = idx[0];
                const unsigned int j_local = idx[1];
                const unsigned long i_global =
                    matrixBlockSize * i_block + i_local;
                const unsigned long j_global =
                    matrixBlockSize * j_block + j_local;
                if (i_global >= N || j_global >= N) {
                  return;
                }
                double distance = 0.0;
                for (unsigned int k = 0; k < nRegressors; k++) {
                  const double tmp = trainingInputData[i_global + k] -
                                     trainingInputData[j_global + k];
                  distance += tmp * tmp;
                }
                double covarianceFunction =
                    verticalLengthscale *
                    sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

                if (i_global == j_global) {
                  covarianceFunction += noiseVariance;
                }
                matrixDataTyped[i_local * matrixBlockSize + j_local] =
                    covarianceFunction;
              });
        });

      } else {
        throw std::runtime_error("Block precision not supported!");
      }
    }
    queue.wait();
  }
  queue.wait();

  return matrix;
}

void MatrixGenerator::generateTestKernelMatrix(std::string &path_train,
                                               std::string &path_test,
                                               sycl::queue &queue,
                                               sycl::queue &queueGPU,
                                               conf::fp_type *K_star) {
  std::cout << "-- generating Kernel matrix of size " << conf::N << "x"
            << conf::N_test << std::endl;

  constexpr std::size_t nRegressors = 8;
  constexpr std::size_t offset = nRegressors - 1;

  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>>
      trainingInput{usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
  trainingInput.resize(conf::N + offset);

  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>>
      testInput{usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
  testInput.resize(conf::N_test + offset);

  readInputVector(path_train, trainingInput, conf::N, offset);

  readInputVector(path_test, testInput, conf::N_test, offset);

  conf::fp_type *trainingInputData = trainingInput.data();
  conf::fp_type *testInputData = testInput.data();

  // Default values for hyperparameters
  constexpr double verticalLengthscale = 1.0;
  constexpr double lengthscale = 1.0;

  const std::size_t N_test = conf::N_test;
  const std::size_t N = conf::N;

  // compute transposed K_star kernel matrix
  queue.submit([&](handler &h) {
    h.parallel_for(range<2>(N_test, N), [=](id<2> idx) {
      const unsigned int i = idx[1];
      const unsigned int j = idx[0];

      double distance = 0.0;
      for (unsigned int k = 0; k < nRegressors; k++) {
        const double tmp = trainingInputData[i + k] - testInputData[j + k];
        distance += tmp * tmp;
      }
      const double covarianceFunction =
          verticalLengthscale *
          sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

      K_star[static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
             static_cast<std::size_t>(i)] = covarianceFunction;
    });
  });
  queue.wait();
}

RightHandSide MatrixGenerator::parseRHS_GP(std::string &path,
                                           sycl::queue &queue) {
  std::cout << "-- parsing data for rhs of size " << conf::N << std::endl;

  RightHandSide rhs(conf::N, conf::matrixBlockSize, queue);

  std::ifstream dataInputStream(path);
  std::string valueString;

  // parse input data
  std::size_t rowIndex = 0;
  while (std::getline(dataInputStream, valueString)) {
    conf::fp_type value = static_cast<conf::fp_type>(std::stod(valueString));
    rhs.rightHandSideData[rowIndex] = value;
    rowIndex++;
    if (rowIndex == conf::N) {
      break;
    }
  }
  dataInputStream.close();

  if (rowIndex != conf::N) {
    throw std::runtime_error("Not enough data available!");
  }

  return rhs;
}

RightHandSide MatrixGenerator::generateRHS(sycl::queue &queue) {
  RightHandSide b(conf::N, conf::matrixBlockSize, queue);

  // random number generator
  std::random_device rd;
  std::mt19937 generator(321);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  for (std::size_t i = 0; i < conf::N; ++i) {
    const conf::fp_type value = distribution(generator);
    b.rightHandSideData[i] = value;
  }
  return b;
}

void MatrixGenerator::readInputVector(
    std::string &path,
    std::vector<conf::fp_type,
                sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>>
        &dataVector,
    int N, int offset) {
  std::ifstream dataInputStream(path);
  std::string valueString;

  // parse input data
  int rowIndex = 0;
  while (std::getline(dataInputStream, valueString)) {
    conf::fp_type value = static_cast<conf::fp_type>(std::stod(valueString));
    dataVector[rowIndex + offset] = value;
    rowIndex++;
    if (rowIndex == N) {
      break;
    }
  }
  dataInputStream.close();

  if (rowIndex != N) {
    throw std::runtime_error("Not enough data available!");
  }
}
