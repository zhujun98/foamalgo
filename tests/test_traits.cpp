#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "foamalgo/traits.hpp"


namespace foam::test
{

TEST(TestReducedType, TestVector)
{
  bool value;

  value = std::is_same<typename xt::xtensor<double, 1>, ReducedVectorType<xt::xtensor<double, 2>>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<double, 1>, ReducedVectorType<xt::xtensor<double, 2>, float>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<double, 1>, ReducedVectorType<xt::xtensor<float, 2>, double>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<float, 1>, ReducedVectorType<xt::xtensor<long long, 2>, float>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<long, 1>, ReducedVectorType<xt::xtensor<long, 2>, int>>::value;
  EXPECT_TRUE(value);
}

TEST(TestReducedType, TestImage)
{
  bool value;

  value = std::is_same<typename xt::xtensor<double, 2>, ReducedImageType<xt::xtensor<double, 3>>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<double, 2>, ReducedImageType<xt::xtensor<double, 3>, float>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<double, 2>, ReducedImageType<xt::xtensor<float, 3>, double>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<float, 2>, ReducedImageType<xt::xtensor<long long, 3>, float>>::value;
  EXPECT_TRUE(value);
  value = std::is_same<typename xt::xtensor<long, 2>, ReducedImageType<xt::xtensor<long, 3>, int>>::value;
  EXPECT_TRUE(value);
}

} //foam::test