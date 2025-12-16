#include "MatrixParserMixed.hpp"
#include "Configuration.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sycl/sycl.hpp>

SymmetricMatrixMixed
MatrixParserMixed::parseSymmetricMatrix(std::string &path, sycl::queue &queue) {
  std::ifstream matrixInputStream(path);

  std::string row;

  // read first line
  std::getline(matrixInputStream, row);

  if (row[0] != '#' || row.empty()) {
    throw std::invalid_argument(
        "Invalid matrix format. First line has to be '# <N>!'");
  }

  // retrieve dimension N of the matrix form the input file
  std::size_t N = std::stoul(row.substr(2, row.size() - 2));

  std::cout << "-- Starting to parse symmetric matrix of size " << N << "x" << N
            << std::endl;
  conf::N = N;
  // create symmetric matrix
  SymmetricMatrixMixed matrix(N, conf::matrixBlockSize, queue);
  // Allocate more memory than necessary to ensure data fits
  matrix.allocate(N * N * conf::matrixBlockSize * conf::matrixBlockSize *
                  sizeof(conf::fp_type));
  const int boundary0 = std::ceil(matrix.blockCountXY) * 2;
  const int boundary1 = std::ceil(matrix.blockCountXY * 0.8);

  // Calculate mixed precision block memory byte offsets
  std::size_t continuous_index = 0;
  int distance = 0;
  std::size_t cumulative_offset = 0;
  int elementByteSize = 0;

  for (int col = 0; col < matrix.blockCountXY; ++col) {
    for (int r = col; r < matrix.blockCountXY; ++r) {
      matrix.blockByteOffsets[continuous_index] = cumulative_offset;
      elementByteSize = 0;
      distance = abs(r - col);
      if (distance < boundary0) {
        elementByteSize = sizeof(double);
        matrix.precisionTypes[continuous_index] =
            static_cast<int>(elementByteSize);
      } else if (distance < boundary1) {
        elementByteSize = sizeof(float);
        matrix.precisionTypes[continuous_index] =
            static_cast<int>(elementByteSize);
      } else {
        elementByteSize = sizeof(sycl::half);
        matrix.precisionTypes[continuous_index] =
            static_cast<int>(elementByteSize);
      }
      cumulative_offset +=
          elementByteSize * conf::matrixBlockSize * conf::matrixBlockSize;
      continuous_index++;
    }
  }

  // read file row by row
  unsigned int rowIndex = 0;
  while (std::getline(matrixInputStream, row)) {
    // process current row string and store values in matrix data structure
    processRow(row, rowIndex, matrix);
    rowIndex++;
  }

  matrixInputStream.close();
  return matrix;
}

RightHandSide MatrixParserMixed::parseRightHandSide(std::string &path,
                                                    sycl::queue &queue) {
  std::ifstream rhsInputStream(path);

  std::string row;

  // read first line
  std::getline(rhsInputStream, row);

  if (row[0] != '#' || row.empty()) {
    throw std::invalid_argument(
        "Invalid right-hand side format. First line has to be '# <N>!'");
  }

  // retrieve dimension N of the matrix form the input file
  const std::size_t N = std::stoul(row.substr(2, row.size() - 2));

  std::cout << "-- Starting to parse right-hand side of size " << N
            << std::endl;

  std::getline(rhsInputStream, row);

  auto values = getRowValuesFromString(row);

  RightHandSide b(N, conf::matrixBlockSize, queue);

  // copy values to the right-hand side vector b which is allocated as sycl host
  // memory
  for (unsigned int i = 0; i < N; i++) {
    b.rightHandSideData[i] = values[i];
  }

  return b;
}

std::vector<conf::fp_type>
MatrixParserMixed::getRowValuesFromString(const std::string &rowString) {
  std::vector<conf::fp_type> rowValues;

  int lastSplitIndex = 0;
  for (unsigned int i = 0; i < rowString.size();
       ++i) // iterate through the string
  {
    if (constexpr char delimiter = ';';
        rowString[i] ==
        delimiter) // if the delimiter is reached, split the string
    {
      std::string valueString =
          rowString.substr(lastSplitIndex, i - lastSplitIndex);

      // cast the value to conf::fp_type and store the result
      try {
        auto value = static_cast<conf::fp_type>(std::stod(valueString));
        rowValues.push_back(value);
      } catch (...) {
        throw std::invalid_argument("Error while parsing string '" +
                                    valueString + "'");
      }
      lastSplitIndex = i + 1;
    }
  }

  // parse the last value
  const std::string valueString =
      rowString.substr(lastSplitIndex, rowString.size() - lastSplitIndex);
  try {
    const auto value = static_cast<conf::fp_type>(std::stod(valueString));
    rowValues.push_back(value);
  } catch (...) {
    throw std::invalid_argument("Error while parsing string '" + valueString +
                                "'");
  }

  return rowValues;
}

void MatrixParserMixed::processRow(const std::string &row,
                                   const unsigned int rowIndex,
                                   SymmetricMatrixMixed &matrix) {
  const std::vector<conf::fp_type> rowValues = getRowValuesFromString(row);
  assert(rowValues.size() == rowIndex + 1);

  // Number of blocks in column direction in the current row
  const int columnBlockCount =
      std::ceil(static_cast<double>(rowValues.size()) /
                static_cast<double>(conf::matrixBlockSize));

  // row index divided by the block size to determine block index later
  auto rowDivBlock = std::div(rowIndex, matrix.blockSize);

  // row index of the current block
  const int rowBlockIndex = rowDivBlock.quot;

  // tracks the total number of blocks in columns left to the current column
  int blockCountLeftColumns = 0;

  // iterate over all column indices of blocks that intersect with the current
  // row
  for (int columnBlockIndex = 0; columnBlockIndex < columnBlockCount;
       ++columnBlockIndex) {
    // Index of block when enumerating all blocks column by column from top to
    // bottom
    const int blockIndex =
        blockCountLeftColumns + (rowBlockIndex - columnBlockIndex);

    // start index of block in matrix data structure
    const std::size_t blockStartOffset = matrix.blockByteOffsets[blockIndex];

    unsigned char *matrixBytes =
        reinterpret_cast<unsigned char *>(matrix.matrixData.data());

    // local row index in block
    const int value_i = rowDivBlock.rem;

    if (rowBlockIndex ==
        columnBlockIndex) // Diagonal Block, some values have to be mirrored
    {
      // for each column in block, j is a global index and corresponds to the
      // column of the whole matrix
      for (unsigned int j = columnBlockIndex * conf::matrixBlockSize;
           j < rowValues.size(); ++j) {
        // compute local column index value_j inside the current block from
        // global column index j
        const unsigned int value_j =
            j - columnBlockIndex * conf::matrixBlockSize;

        const conf::fp_type value = rowValues[j];

        if (matrix.precisionTypes[blockIndex] == 2) {
          sycl::half *matrixTyped =
              reinterpret_cast<sycl::half *>(matrixBytes + blockStartOffset);
          // location as read in file in lower triangle (i,j)
          matrixTyped[value_i * conf::matrixBlockSize + value_j] =
              static_cast<sycl::half>(value);
          // mirrored value in upper triangle (j,i)
          matrixTyped[value_j * conf::matrixBlockSize + value_i] =
              static_cast<sycl::half>(value);
        } else if (matrix.precisionTypes[blockIndex] == 4) {
          float *matrixTyped =
              reinterpret_cast<float *>(matrixBytes + blockStartOffset);
          matrixTyped[value_i * conf::matrixBlockSize + value_j] =
              static_cast<float>(value);
          matrixTyped[value_j * conf::matrixBlockSize + value_i] =
              static_cast<float>(value);
        } else if (matrix.precisionTypes[blockIndex] == 8) {
          double *matrixTyped =
              reinterpret_cast<double *>(matrixBytes + blockStartOffset);
          matrixTyped[value_i * conf::matrixBlockSize + value_j] =
              static_cast<double>(value);
          matrixTyped[value_j * conf::matrixBlockSize + value_i] =
              static_cast<double>(value);
        }
      }
    } else // normal block
    {
      // for each column in block, j is a global index and corresponds to the
      // column of the whole matrix
      for (int j = columnBlockIndex * static_cast<int>(conf::matrixBlockSize);
           j < columnBlockIndex * static_cast<int>(conf::matrixBlockSize) +
                   static_cast<int>(conf::matrixBlockSize);
           ++j) // for each column in block
      {
        // compute local column index value_j inside the current block from
        // global column index j
        const int value_j =
            j - columnBlockIndex * static_cast<int>(conf::matrixBlockSize);

        const conf::fp_type value = rowValues[j];

        // location as read in file in lower triangle (i,j)
        if (matrix.precisionTypes[blockIndex] == 2) {
          sycl::half *matrixTyped =
              reinterpret_cast<sycl::half *>(matrixBytes + blockStartOffset);
          // location as read in file in lower triangle (i,j)
          matrixTyped[value_i * conf::matrixBlockSize + value_j] =
              static_cast<sycl::half>(value);
        } else if (matrix.precisionTypes[blockIndex] == 4) {
          float *matrixTyped =
              reinterpret_cast<float *>(matrixBytes + blockStartOffset);
          matrixTyped[value_i * conf::matrixBlockSize + value_j] =
              static_cast<float>(value);
        } else if (matrix.precisionTypes[blockIndex] == 8) {
          double *matrixTyped =
              reinterpret_cast<double *>(matrixBytes + blockStartOffset);
          matrixTyped[value_i * conf::matrixBlockSize + value_j] =
              static_cast<double>(value);
        }
      }
    }

    // increment number of blocks in all columns left of the current column
    blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;
  }
}

void MatrixParserMixed::writeBlockedMatrix(const std::string &path,
                                           const SymmetricMatrixMixed &matrix) {
  std::ofstream output(path);

  output << std::setprecision(10) << std::fixed;

  const unsigned char *matrixBytes = matrix.matrixData.data();

  for (int rowIndex = 0; rowIndex < matrix.blockCountXY * matrix.blockSize;
       ++rowIndex) // for each row
  {
    if (rowIndex % conf::matrixBlockSize == 0) {
      output << std::endl;
    }
    // row index divided by the block size to determine block index later
    auto rowDivBlock = std::div(rowIndex, matrix.blockSize);
    const int rowBlockIndex = rowDivBlock.quot;

    int blockCountLeftColumns = 0;
    for (int columnBlockIndex = 0; columnBlockIndex <= rowBlockIndex;
         ++columnBlockIndex) {
      const int blockIndex =
          blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

      // start offset of block in matrix data structure
      const std::size_t blockStartOffset = matrix.blockByteOffsets[blockIndex];

      // row index in block
      const int value_i = rowDivBlock.rem;

      // for each column in block
      for (unsigned int value_j = 0; value_j < conf::matrixBlockSize;
           ++value_j) {
        conf::fp_type value;
        if (matrix.precisionTypes[blockIndex] == 2) {
          const sycl::half *matrixTyped = reinterpret_cast<const sycl::half *>(
              matrixBytes + blockStartOffset);
          // location as read in file in lower triangle (i,j)
          value = static_cast<conf::fp_type>(
              matrixTyped[value_i * conf::matrixBlockSize + value_j]);
        } else if (matrix.precisionTypes[blockIndex] == 4) {
          const float *matrixTyped =
              reinterpret_cast<const float *>(matrixBytes + blockStartOffset);
          value = static_cast<conf::fp_type>(
              matrixTyped[value_i * conf::matrixBlockSize + value_j]);
        } else if (matrix.precisionTypes[blockIndex] == 8) {
          const double *matrixTyped =
              reinterpret_cast<const double *>(matrixBytes + blockStartOffset);
          value = static_cast<conf::fp_type>(
              matrixTyped[value_i * conf::matrixBlockSize + value_j]);
        } else {
          throw std::runtime_error("Invalid block precision encountered!\n");
        }

        if (value >= 0) {
          output << " " << value << ";";
        } else {
          output << value << ";";
        }
      }

      // increment number of blocks in all columns left of the current column
      blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;

      output << "\t";
    }
    output << std::endl;
  }
  output.close();
}

void MatrixParserMixed::writeFullMatrix(const std::string &path,
                                        const SymmetricMatrixMixed &matrix) {
  std::ofstream output(path);

  output << std::setprecision(20) << std::fixed;

  for (int rowIndex = 0; rowIndex < matrix.blockCountXY * matrix.blockSize;
       ++rowIndex) // for each row
  {
    // row index divided by the block size to determine block index later
    auto rowDivBlock = std::div(rowIndex, matrix.blockSize);
    const int rowBlockIndex = rowDivBlock.quot;

    int blockIndex;
    int blockCountLeftColumns = 0;
    for (int columnBlockIndex = 0; columnBlockIndex <= rowBlockIndex;
         ++columnBlockIndex) {
      blockIndex =
          blockCountLeftColumns + (rowDivBlock.quot - columnBlockIndex);

      // start offset of block in matrix data structure
      const std::size_t blockStartOffset = matrix.blockByteOffsets[blockIndex];

      // row index in block
      const int value_i = rowDivBlock.rem;

      // for each column in block
      for (unsigned int value_j = 0; value_j < conf::matrixBlockSize;
           ++value_j) {
        conf::fp_type value;
        if (matrix.precisionTypes[blockIndex] == 2) {
          const sycl::half *matrixTyped = reinterpret_cast<const sycl::half *>(
              matrix.matrixData.data() + blockStartOffset);
          // location as read in file in lower triangle (i,j)
          value = static_cast<conf::fp_type>(
              matrixTyped[value_i * conf::matrixBlockSize + value_j]);
        } else if (matrix.precisionTypes[blockIndex] == 4) {
          const float *matrixTyped = reinterpret_cast<const float *>(
              matrix.matrixData.data() + blockStartOffset);
          value = static_cast<conf::fp_type>(
              matrixTyped[value_i * conf::matrixBlockSize + value_j]);
        } else if (matrix.precisionTypes[blockIndex] == 8) {
          const double *matrixTyped = reinterpret_cast<const double *>(
              matrix.matrixData.data() + blockStartOffset);
          value = static_cast<conf::fp_type>(
              matrixTyped[value_i * conf::matrixBlockSize + value_j]);
        } else {
          throw std::runtime_error("Invalid block precision encountered!\n");
        }

        if (value >= 0) {
          output << " " << value << ";";
        } else {
          output << value << ";";
        }
      }

      // increment number of blocks in all columns left of the current column
      blockCountLeftColumns += matrix.blockCountXY - columnBlockIndex;
    }

    int diagBlockIndex = blockIndex;
    for (int i = 0; i < matrix.blockCountXY - (rowBlockIndex + 1); ++i) {
      blockIndex = diagBlockIndex + i + 1;

      // start offset of block in matrix data structure
      const std::size_t blockStartOffset = matrix.blockByteOffsets[blockIndex];

      // row index in block
      const int value_i = rowDivBlock.rem;

      // for each column in block
      for (unsigned int value_j = 0; value_j < conf::matrixBlockSize;
           ++value_j) {
        conf::fp_type value;
        if (matrix.precisionTypes[blockIndex] == 2) {
          const sycl::half *matrixTyped = reinterpret_cast<const sycl::half *>(
              matrix.matrixData.data() + blockStartOffset);
          // location as read in file in lower triangle (i,j)
          value = static_cast<conf::fp_type>(
              matrixTyped[value_j * conf::matrixBlockSize + value_i]);
        } else if (matrix.precisionTypes[blockIndex] == 4) {
          const float *matrixTyped = reinterpret_cast<const float *>(
              matrix.matrixData.data() + blockStartOffset);
          value = static_cast<conf::fp_type>(
              matrixTyped[value_j * conf::matrixBlockSize + value_i]);
        } else if (matrix.precisionTypes[blockIndex] == 8) {
          const double *matrixTyped = reinterpret_cast<const double *>(
              matrix.matrixData.data() + blockStartOffset);
          value = static_cast<conf::fp_type>(
              matrixTyped[value_j * conf::matrixBlockSize + value_i]);
        } else {
          throw std::runtime_error("Invalid block precision encountered!\n");
        }

        if (value >= 0) {
          output << " " << value << ";";
        } else {
          output << value << ";";
        }
      }
    }

    output << std::endl;
  }
  output.close();
}
