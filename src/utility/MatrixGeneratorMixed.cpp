#include "MatrixGeneratorMixed.hpp"
#include "Configuration.hpp"
#include "RankFinder.hpp"
#include "SymmetricMatrixMixed.hpp"

using namespace sycl;

SymmetricMatrixMixed MatrixGeneratorMixed::generateSPDMatrixMixed(std::string &path,
                                                                  sycl::queue &queue,
                                                                  sycl::queue &queueGPU) {
  std::cout << "-- generating mixed precision SPD matrix of size " << conf::N << "x" << conf::N
            << std::endl;

  // Build and allocate matrix memory
  SymmetricMatrixMixed matrix(conf::N, conf::matrixBlockSize, queueGPU);

  // Build temporary input data vector
  std::size_t nRegressors = 8;
  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> trainingInput{
      usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
  std::size_t offset = nRegressors - 1;
  trainingInput.resize(conf::N + offset);

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

  std::vector<int> trainingPrecisionTypes;
  trainingPrecisionTypes.resize(matrix.blockCountXY * (matrix.blockCountXY + 1) / 2);
  if (conf::qr) {
    // Calculate block precisions based on their ranks
    std::cout << "-- computing qr decomposition to find block precisions" << std::endl;
    trainingPrecisionTypes = RankFinder::compute_block_precisions(
        queue, trainingInput, conf::N, conf::matrixBlockSize, nRegressors);
  } else if (!conf::qr) {
    // Calculate block precision based on distance
    std::size_t continuous_index = 0;
    const int remaining_precision = (conf::fp16) ? 2 : 4;
    for (std::size_t col = 0; col < static_cast<std::size_t>(matrix.blockCountXY); ++col) {
      for (std::size_t row = col; row < static_cast<std::size_t>(matrix.blockCountXY); ++row) {
        const int distance = sycl::abs(row - col);
        const double rel_distance = static_cast<double>(distance) / matrix.blockCountXY;
        if (rel_distance < 0.1) {
          trainingPrecisionTypes[continuous_index] = 8;
        } else if (rel_distance < 0.95) {
          trainingPrecisionTypes[continuous_index] = 4;
        } else {
          trainingPrecisionTypes[continuous_index] = remaining_precision;
        }
        continuous_index++;
      }
    }
  }

  // Calculate byte offsets for all blocks based on precision
  std::size_t fp16_blocks = 0;
  std::size_t fp32_blocks = 0;
  std::size_t fp64_blocks = 0;
  std::size_t cumulative_offset = 0;
  for (std::size_t i = 0; i < trainingPrecisionTypes.size(); ++i) {
    matrix.blockByteOffsets[i] = cumulative_offset;
    matrix.precisionTypes[i] = trainingPrecisionTypes[i];
    switch (trainingPrecisionTypes[i]) {
    case 2:
      fp16_blocks++;
      break;
    case 4:
      fp32_blocks++;
      break;
    case 8:
      fp64_blocks++;
      break;
    }
    cumulative_offset += trainingPrecisionTypes[i] * conf::matrixBlockSize * conf::matrixBlockSize;
  }

  // Store block counts in data structure
  matrix.blockCountFP16 = fp16_blocks;
  matrix.blockCountFP32 = fp32_blocks;
  matrix.blockCountFP64 = fp64_blocks;

  // Resize matrix vector to byte size
  const std::size_t totalByteSize =
      matrix.blockByteOffsets[matrix.blockByteOffsets.size() - 1] +
      matrix.precisionTypes[matrix.precisionTypes.size() - 1] * matrix.blockSize * matrix.blockSize;
  matrix.allocate(totalByteSize);

  const std::size_t block_elements = conf::matrixBlockSize * conf::matrixBlockSize;

  // Calculate and print size savings for mixed precision
  const std::size_t fp16_bytes = fp16_blocks * block_elements * sizeof(sycl::half);
  const std::size_t fp32_bytes = fp32_blocks * block_elements * sizeof(float);
  const std::size_t fp64_bytes = fp64_blocks * block_elements * sizeof(double);
  const std::size_t saved_bytes =
      (fp16_blocks + fp32_blocks + fp64_blocks) * block_elements * sizeof(conf::fp_type) -
      totalByteSize;

  std::cout << "-- mixed precision memory usage:\n";
  if (conf::fp16) {
    std::cout << "---- " << fp16_blocks << " FP16 blocks using " << fp16_bytes / 1024 / 1024
              << " MB\n";
  }
  std::cout << "---- " << fp32_blocks << " FP32 blocks using " << fp32_bytes / 1024 / 1024
            << " MB\n";
  std::cout << "---- " << fp64_blocks << " FP64 blocks using " << fp64_bytes / 1024 / 1024
            << " MB\n";
  std::cout << "---- mixed precision saving a total of " << saved_bytes / 1024 / 1024 << " MB\n";

  // block count of all columns except the first one
  const int referenceBlockCount = (matrix.blockCountXY * (matrix.blockCountXY - 1)) / 2;

  // Prepare memory pointers
  unsigned char *matrixBytes = matrix.matrixData.data();
  conf::fp_type *trainingInputData = trainingInput.data();

  std::size_t N = conf::N;
  for (std::size_t i_block = 0; i_block < static_cast<std::size_t>(matrix.blockCountXY);
       ++i_block) {
    for (std::size_t j_block = 0; j_block <= i_block; j_block++) {
      // number of blocks in row to the right (if matrix would be full)
      const int block_j_inv = matrix.blockCountXY - (j_block + 1);

      // total number of blocks to the right that are stored
      const int columnBlocksToRight = (block_j_inv * (block_j_inv + 1)) / 2;

      // id of block in the matrix data structure for symmetric matrices
      const int blockID = i_block + referenceBlockCount - columnBlocksToRight;

      const std::size_t byteOffset = matrix.blockByteOffsets[blockID];
      const int blockPrecision = matrix.precisionTypes[blockID];

      // Diagonal noise for stability
      const double noiseVariance = 0.01;

      // Default values for hyperparameters
      const double verticalLengthscale = 1.0;
      const double lengthscale = 1.0;

      std::size_t matrixBlockSize = conf::matrixBlockSize;

      if (blockPrecision == 2) {
        queue.submit([&](handler &h) {
          h.parallel_for(range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
            // Add correct typing to both arrays according to block
            // index
            sycl::half *matrixDataTyped = reinterpret_cast<sycl::half *>(matrixBytes + byteOffset);
            const unsigned int i_local = idx[0];
            const unsigned int j_local = idx[1];
            const unsigned long i_global = matrixBlockSize * i_block + i_local;
            const unsigned long j_global = matrixBlockSize * j_block + j_local;
            if (i_global >= N || j_global >= N) {
              return;
            }
            sycl::half distance = 0.0;
            for (unsigned int k = 0; k < nRegressors; k++) {
              const sycl::half tmp =
                  trainingInputData[i_global + k] - trainingInputData[j_global + k];
              distance += tmp * tmp;
            }
            sycl::half covarianceFunction =
                verticalLengthscale * sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

            if (i_global == j_global) {
              covarianceFunction += noiseVariance;
            }
            matrixDataTyped[i_local * matrixBlockSize + j_local] = covarianceFunction;
          });
        });

      } else if (blockPrecision == 4) {
        queue.submit([&](handler &h) {
          h.parallel_for(range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
            // Add correct typing to both arrays according to block
            // index
            float *matrixDataTyped = reinterpret_cast<float *>(matrixBytes + byteOffset);
            const unsigned int i_local = idx[0];
            const unsigned int j_local = idx[1];
            const unsigned long i_global = matrixBlockSize * i_block + i_local;
            const unsigned long j_global = matrixBlockSize * j_block + j_local;
            if (i_global >= N || j_global >= N) {
              return;
            }
            float distance = 0.0;
            for (unsigned int k = 0; k < nRegressors; k++) {
              const float tmp = trainingInputData[i_global + k] - trainingInputData[j_global + k];
              distance += tmp * tmp;
            }
            const float tmp =
                sycl::exp(-0.5f / static_cast<float>((lengthscale * lengthscale)) * distance);
            float covarianceFunction = static_cast<float>(verticalLengthscale) * tmp;

            if (i_global == j_global) {
              covarianceFunction += static_cast<float>(noiseVariance);
            }
            matrixDataTyped[i_local * matrixBlockSize + j_local] = covarianceFunction;
          });
        });

      } else if (blockPrecision == 8) {
        queue.submit([&](handler &h) {
          h.parallel_for(range<2>(matrixBlockSize, matrixBlockSize), [=](id<2> idx) {
            // Add correct typing to both arrays according to block
            // index
            double *matrixDataTyped = reinterpret_cast<double *>(matrixBytes + byteOffset);
            const unsigned int i_local = idx[0];
            const unsigned int j_local = idx[1];
            const unsigned long i_global = matrixBlockSize * i_block + i_local;
            const unsigned long j_global = matrixBlockSize * j_block + j_local;
            if (i_global >= N || j_global >= N) {
              return;
            }
            double distance = 0.0;
            for (unsigned int k = 0; k < nRegressors; k++) {
              const double tmp = trainingInputData[i_global + k] - trainingInputData[j_global + k];
              distance += tmp * tmp;
            }
            double covarianceFunction =
                verticalLengthscale * sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

            if (i_global == j_global) {
              covarianceFunction += noiseVariance;
            }
            matrixDataTyped[i_local * matrixBlockSize + j_local] = covarianceFunction;
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

void MatrixGeneratorMixed::generateTestKernelMatrixMixed(std::string &path_train,
                                                         std::string &path_test, sycl::queue &queue,
                                                         sycl::queue &queueGPU,
                                                         conf::fp_type *K_star) {
  std::cout << "-- generating Kernel matrix of size " << conf::N << "x" << conf::N_test
            << std::endl;

  constexpr std::size_t nRegressors = 8;
  constexpr std::size_t offset = nRegressors - 1;

  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> trainingInput{
      usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
  trainingInput.resize(conf::N + offset);

  std::vector<conf::fp_type, usm_allocator<conf::fp_type, usm::alloc::host>> testInput{
      usm_allocator<conf::fp_type, usm::alloc::host>(queueGPU)};
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
          verticalLengthscale * sycl::exp(-0.5 / (lengthscale * lengthscale) * distance);

      K_star[static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
             static_cast<std::size_t>(i)] = covarianceFunction;
    });
  });
  queue.wait();
}

void MatrixGeneratorMixed::readInputVector(
    std::string &path,
    std::vector<conf::fp_type, sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>>
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
