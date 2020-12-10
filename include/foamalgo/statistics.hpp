/**
 * Distributed under the terms of the GNU General Public License v3.0.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Copyright (C) 2020, Jun Zhu. All rights reserved.
 */
#ifndef FOAM_STATISTICS_HPP
#define FOAM_STATISTICS_HPP

#include <type_traits>

#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"

#if defined(FOAM_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "traits.hpp"


namespace foam
{

} // foam


#endif //FOAM_STATISTICS_HPP
