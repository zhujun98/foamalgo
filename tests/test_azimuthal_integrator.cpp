#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "foamalgo/azimuthal_integrator.hpp"

namespace foam::test
{

using ::testing::Each;
using ::testing::Eq;
using ::testing::ElementsAre;

static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

TEST(TestAzimuthalIntegrator, TestDataType)
{
  xt::xtensor<double, 2> src_double = xt::arange(1024).reshape({16, 128});
  xt::xtensor<float, 2> src_float = xt::arange(1024).reshape({16, 128});
  xt::xtensor<int16_t, 2> src_int16 = xt::arange(1024).reshape({16, 128});

  double distance = 0.2;
  double pixel1 = 1e-4;
  double pixel2 = 2e-4;
  double poni1 = -6 * pixel1;
  double poni2 = 130 * pixel2;
  double wavelength = 1e-10;
  AzimuthalIntegrator<float> itgt_float(distance, poni1, poni2, pixel1, pixel2, wavelength);
  auto ret_double = itgt_float.integrate1d(src_double, 10);
  // value_type of the result is the same as the source if the source
  // has a floating point value_type
  bool flag = std::is_same_v<typename decltype(ret_double.first)::value_type, double>;
  ASSERT_TRUE(flag);
  flag = std::is_same_v<typename decltype(ret_double.second)::value_type, double>;
  ASSERT_TRUE(flag);

  AzimuthalIntegrator<double> itgt_double(distance, poni1, poni2, pixel1, pixel2, wavelength);
  auto ret_float = itgt_double.integrate1d(src_float, 10);
  flag = std::is_same_v<typename decltype(ret_float.first)::value_type, float>;
  ASSERT_TRUE(flag);
  flag = std::is_same_v<typename decltype(ret_float.second)::value_type, float>;
  ASSERT_TRUE(flag);

  // value_type of the result is the same as the template type of the integrator
  // if the source does not have a floating point value_type
  auto ret_int16 = itgt_double.integrate1d(src_int16, 10);
  flag = std::is_same_v<typename decltype(ret_int16.first)::value_type, double>;
  ASSERT_TRUE(flag);
  flag = std::is_same_v<typename decltype(ret_int16.second)::value_type, double>;
  ASSERT_TRUE(flag);
}

TEST(TestAzimuthalIntegrator, TestIntegrator1D)
{
  xt::xtensor<float, 2> src = xt::arange(1024).reshape({16, 128});
  xt::xtensor<float, 2> src2 = src - 100;
  auto src_a = xt::xtensor<float, 3>::from_shape({4, 16, 128});
  for (size_t i = 0; i < 3; ++i) xt::view(src_a, i, xt::all(), xt::all()) = src;
  xt::view(src_a, 3, xt::all(), xt::all()) = src2;

  double distance = 0.2;
  double pixel1 = 1e-4;
  double pixel2 = 2e-4;
  double poni1 = -6 * pixel1;
  double poni2 = 130 * pixel2;
  double wavelength = 1e-10;
  AzimuthalIntegrator<float> itgt(distance, poni1, poni2, pixel1, pixel2, wavelength);

  // npt < 2
  auto ret0 = itgt.integrate1d(src, 0);
  auto ret1 = itgt.integrate1d(src, 1);
  EXPECT_EQ(ret0, ret1);

  // different min_counts
  auto ret10 = itgt.integrate1d(src, 10);
  auto ret10_cut = itgt.integrate1d(src, 10, src.size());
  EXPECT_EQ(ret10.first, ret10_cut.first);
  EXPECT_THAT(ret10_cut.second, Each(Eq(0.)));

  // test integrate an array of images
  auto ret10_a = itgt.integrate1d(src_a, 10);
  EXPECT_EQ(ret10.first, ret10_a.first);
  for (size_t i = 0; i < 3; ++i) EXPECT_EQ(ret10.second, xt::view(ret10_a.second, i, xt::all()));
  auto ret10_2 = itgt.integrate1d(src2, 10);
  EXPECT_EQ(ret10_2.second, xt::view(ret10_a.second, 3, xt::all()));

  // big npt
  itgt.integrate1d(src, 999);

  // data has a single value
  xt::xtensor<double, 2> src_single_value = xt::ones<double>({16, 128});
  itgt.integrate1d(src_single_value, 10);

  // integral source value type
  xt::xtensor<uint16_t, 2> src_int = xt::arange(1024).reshape({16, 128});
  auto ret10_uint16 = itgt.integrate1d(src, 10);
  EXPECT_EQ(ret10.first, ret10_uint16.first);
  EXPECT_THAT(ret10.second, ret10_uint16.second);

  // shape changed;
  xt::xtensor<float, 2> src_small = xt::arange(512).reshape({32, 16});
  itgt.integrate1d(src_small, 10);
  xt::xtensor<float, 2> src_big = xt::arange(2048).reshape({128, 32});
  itgt.integrate1d(src_big, 10);
}

TEST(TestConcentricRingsFinder, TestGeneral)
{
  xt::xtensor<double, 2> src = xt::ones<double>({16, 128});

  double pixel_x = 2e-4;
  double pixel_y = 1e-4;
  double cx = 128;
  double cy = -6;
  size_t min_count = 32;

  ConcentricRingsFinder<float> finder(pixel_x, pixel_y);
  finder.search(src, cx, cy, min_count);
}

} //foam::test
