#include <hws/system_hardware_sampler.hpp>
#include <sycl/sycl.hpp>

#include "SymmetricMatrixMixed.hpp"

SymmetricMatrixMixed::SymmetricMatrixMixed(const std::size_t N,
                                           const int blockSize,
                                           sycl::queue &queue)
    : N(N), blockSize(blockSize),
      blockCountXY(
          std::ceil(static_cast<double>(N) / static_cast<double>(blockSize))),
      blockCountFP16(0), blockCountFP32(0), blockCountFP64(0),
      precisionTypes(sycl::usm_allocator<int, sycl::usm::alloc::shared>(queue)),
      blockRanks(sycl::usm_allocator<int, sycl::usm::alloc::shared>(queue)),
      blockByteOffsets(
          sycl::usm_allocator<std::size_t, sycl::usm::alloc::shared>(queue)),
      // allocate float, needs to be resized to actual size anyway
      matrixData(
          sycl::usm_allocator<unsigned char, sycl::usm::alloc::host>(queue)) {
  // allocate memory for matrix storage
  const std::size_t blockCount = (blockCountXY * (blockCountXY + 1)) / 2.0;
  precisionTypes.resize(blockCount);
  blockRanks.resize(blockCount);
  blockByteOffsets.resize(blockCount);
}

void SymmetricMatrixMixed::allocate(size_t total_bytes) {
  matrixData.resize(total_bytes);
  // Add sentinel for safe offset range computation
  blockByteOffsets.push_back(total_bytes);
}
