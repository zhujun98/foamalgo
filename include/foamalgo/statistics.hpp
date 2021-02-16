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
#include "xtensor/xhistogram.hpp"

#if defined(FOAM_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "traits.hpp"

namespace xt
{

template <class E1, class E2>
inline auto histogram(E1&& data, E2 left, E2 right, std::size_t bins = 10, bool density = false)
{
  using value_type = typename std::decay_t<E1>::value_type;

  auto n = data.size();

  return detail::histogram_imp(std::forward<E1>(data),
                               histogram_bin_edges(data, left, right, bins),
                               xt::ones<value_type>({ n }),
                               density,
                               true);
}

}


namespace foam
{

} // foam


#endif //FOAM_STATISTICS_HPP
