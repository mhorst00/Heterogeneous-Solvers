#include <hws/system_hardware_sampler.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "SymmetricMatrixMixed.hpp"

#include "Configuration.hpp"

SymmetricMatrixMixed::SymmetricMatrixMixed(
    const std::size_t N, const std::vector<Precision> precisionVector,
    const int blockSize, sycl::queue &queue)
    : N(N), blockSize(blockSize),
      blockCountXY(
          std::ceil(static_cast<double>(N) / static_cast<double>(blockSize))),
      precisionVector(precisionVector), byteSize(0),
      matrixData(
          sycl::usm_allocator<conf::fp_type, sycl::usm::alloc::host>(queue)) {
  // allocate memory for matrix storage
  const std::size_t blockCount = (blockCountXY * (blockCountXY + 1)) / 2.0;
  matrixData.resize(blockCount *
                    static_cast<std::size_t>(blockSize * blockSize));
}
