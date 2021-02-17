#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"

#include "foamalgo/statistics.hpp"

namespace foam::test
{

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::NanSensitiveFloatEq;
using ::testing::FloatEq;

} //foam::test