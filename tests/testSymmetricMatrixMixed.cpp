#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include "Configuration.hpp"
#include "MatrixParserMixed.hpp"

class SymmetricMatrixTestMixed : public ::testing::Test {
  protected:
    std::string path = "../tests/testData/testMatrixSymmetric20x20.txt";
};

TEST_F(SymmetricMatrixTestMixed, parseSymmetricMatrixMixedBS4) {
    sycl::queue queue(sycl::cpu_selector_v);
    conf::matrixBlockSize = 4;
    SymmetricMatrixMixed matrix = MatrixParserMixed::parseSymmetricMatrix(path, queue);

    EXPECT_EQ(matrix.blockSize, 4);
    EXPECT_EQ(matrix.blockCountXY, 5);
    // EXPECT_EQ(matrixSize, 15 * 4 * 4);
    conf::fp_type *block0 = reinterpret_cast<conf::fp_type *>(matrix.matrixData.data());
    conf::fp_type *block5 =
        reinterpret_cast<conf::fp_type *>(matrix.matrixData.data() + matrix.blockByteOffsets[5]);
    conf::fp_type *block14 =
        reinterpret_cast<conf::fp_type *>(matrix.matrixData.data() + matrix.blockByteOffsets[14]);

    EXPECT_DOUBLE_EQ(block0[0], 2.77476983971595769773);
    EXPECT_DOUBLE_EQ(block14[15], 0.61881172156451602628);
    EXPECT_DOUBLE_EQ(block0[3], 2.45454075211619482388);
    EXPECT_DOUBLE_EQ(block5[3], 0.93814549612914632792);
    EXPECT_DOUBLE_EQ(block5[12], 0.93814549612914632792);
}

TEST_F(SymmetricMatrixTestMixed, parseSymmetricMatrixMixedBS6) {
    sycl::queue queue(sycl::cpu_selector_v);
    conf::matrixBlockSize = 6;
    SymmetricMatrixMixed matrix = MatrixParserMixed::parseSymmetricMatrix(path, queue);

    EXPECT_EQ(matrix.blockSize, 6);
    EXPECT_EQ(matrix.blockCountXY, 4);
    // EXPECT_EQ(matrix.matrixData.size(), 10 * 6 * 6);

    conf::fp_type *block0 = reinterpret_cast<conf::fp_type *>(matrix.matrixData.data());
    conf::fp_type *block3 =
        reinterpret_cast<conf::fp_type *>(matrix.matrixData.data() + matrix.blockByteOffsets[3]);
    conf::fp_type *block4 =
        reinterpret_cast<conf::fp_type *>(matrix.matrixData.data() + matrix.blockByteOffsets[4]);
    conf::fp_type *block9 =
        reinterpret_cast<conf::fp_type *>(matrix.matrixData.data() + matrix.blockByteOffsets[9]);
    EXPECT_DOUBLE_EQ(block0[0], 2.77476983971595769773);
    EXPECT_DOUBLE_EQ(block9[35], 0.0);
    EXPECT_DOUBLE_EQ(block0[5], 0.40679131158035214400);
    EXPECT_DOUBLE_EQ(block4[5], 0.42401421263563354724);
    EXPECT_DOUBLE_EQ(block4[30], 0.42401421263563354724);
    EXPECT_DOUBLE_EQ(block3[30], 0.0);
}

TEST(MatrixParserMixedTest, parseRowString) {
    std::vector<conf::fp_type> values = MatrixParserMixed::getRowValuesFromString(
        " 3.28800491977049302861e-02;-8.18219670384158145882e-03; "
        "9.21129397428693656464e-05; "
        "7.22668449547045899095e-02;-1.75117094933355359199e-02; "
        "2.43594298214563180494e-02;-3.42271744814721390338e-02; "
        "5.27051749627409447246e-03;-7.53461080610220157450e-02; "
        "7.11621899095706972327e-02;-5.05960381205778075842e-02; "
        "3.11925272356795950379e-02; 5.32773450657381797413e-01");

    std::vector<double> result = {
        3.28800491977049302861e-02,  -8.18219670384158145882e-03, 9.21129397428693656464e-05,
        7.22668449547045899095e-02,  -1.75117094933355359199e-02, 2.43594298214563180494e-02,
        -3.42271744814721390338e-02, 5.27051749627409447246e-03,  -7.53461080610220157450e-02,
        7.11621899095706972327e-02,  -5.05960381205778075842e-02, 3.11925272356795950379e-02,
        5.32773450657381797413e-01};

    EXPECT_EQ(result.size(), values.size());

    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_DOUBLE_EQ(result[i], values[i]);
    }
}

TEST(MatrixParserMixedTest, parseRowStringInvalid) {
    EXPECT_THROW(MatrixParserMixed::getRowValuesFromString("abc"), std::invalid_argument);
}
