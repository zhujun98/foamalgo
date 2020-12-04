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
#include "xtensor/xsort.hpp"

#if defined(FOAM_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "traits.hpp"
#include "utilities.hpp"


namespace foam
{
namespace detail
{

/**
 * The histogram code is a modification based on the implementation in xtensor.
 *
 * The motivation originated from the following PR:
 * https://github.com/xtensor-stack/xtensor/pull/2088
 *
 * Now the behavior of xt::histogram differs from the numpy implementation. In numpy:
 * - The dtype of count is always np.int64 if weights is not specified and density is False;
 * - The dtype of count is always np.float64 if weights is given (no matter what dtype
 *   weights has) or density is True.
 */
template <class R = double, class E1, class E2, class E3>
inline auto histogram_imp(E1&& data, E2&& bin_edges, E3&& weights, bool density, bool equal_bins)
{
  using size_type = xt::common_size_type_t<std::decay_t<E1>, std::decay_t<E2>, std::decay_t<E3>>;
  using value_type = typename std::decay_t<E3>::value_type;

  size_t n_bins = bin_edges.size() - 1;
  xt::xtensor<value_type, 1> count = xt::zeros<value_type>({ n_bins });

  if (equal_bins)
  {
    std::array<typename std::decay_t<E2>::value_type, 2> bounds = xt::minmax(bin_edges)();
    auto left = static_cast<double>(bounds[0]);
    auto right = static_cast<double>(bounds[1]);
    double norm = 1. / (right - left);
    for (size_t i = 0; i < data.size(); ++i)
    {
      auto v = static_cast<double>(data(i));
      // left and right are not bounds of data
      if ( v >= left & v < right )
      {
        auto i_bin = static_cast<size_t>(static_cast<double>(n_bins) * (v - left) * norm);
        count(i_bin) += weights(i);
      }
      else if ( v == right )
      {
        count(n_bins - 1) += weights(i);
      }
    }
  }
  else
  {
    auto sorter = xt::argsort(data);

    size_type ibin = 0;

    for (auto& idx : sorter)
    {
      while (data[idx] >= bin_edges[ibin + 1] && ibin < bin_edges.size() - 2)
      {
        ++ibin;
      }
      count[ibin] += weights[idx];
    }
  }

  xt::xtensor<R, 1> prob = xt::cast<R>(count);

  if (density)
  {
    R n = static_cast<R>(data.size());
    for (size_type i = 0; i < bin_edges.size() - 1; ++i)
    {
      prob[i] /= (static_cast<R>(bin_edges[i + 1] - bin_edges[i]) * n);
    }
  }

  return prob;
}

} //detail

/**
 * @ingroup histogram
 * @brief Compute the histogram of a set of data.
 *
 * @param data The data.
 * @param bin_edges The bin-edges. It has to be 1-dimensional and monotonic.
 * @param weights Weight factors corresponding to each data-point.
 * @param density If true the resulting integral is normalized to 1. [default: false]
 * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
 */
template <class R = double, class E1, class E2, class E3>
inline auto histogram(E1&& data, E2&& bin_edges, E3&& weights, bool density = false)
{
  return detail::histogram_imp<R>(std::forward<E1>(data),
                                  std::forward<E2>(bin_edges),
                                  std::forward<E3>(weights),
                                  density,
                                  false);
}

/**
 * @ingroup histogram
 * @brief Compute the histogram of a set of data.
 *
 * @param data The data.
 * @param bin_edges The bin-edges.
 * @param density If true the resulting integral is normalized to 1. [default: false]
 * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
 */
template <class E1, class E2>
inline auto histogram(E1&& data, E2&& bin_edges, bool density = false)
{
  using value_type = typename std::decay_t<E1>::value_type;

  auto n = data.size();

  return detail::histogram_imp(std::forward<E1>(data),
                               std::forward<E2>(bin_edges),
                               xt::ones<value_type>({ n }),
                               density,
                               false);
}

/**
 * @ingroup histogram
 * @brief Compute the histogram of a set of data.
 *
 * @param data The data.
 * @param bins The number of bins. [default: 10]
 * @param density If true the resulting integral is normalized to 1. [default: false]
 * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
 */
template <class E1>
inline auto histogram(E1&& data, std::size_t bins = 10, bool density = false)
{
  using value_type = typename std::decay_t<E1>::value_type;

  auto n = data.size();

  auto bin_edges = histogram_bin_edges(data, xt::ones<value_type>({ n }), bins);

  return detail::histogram_imp(std::forward<E1>(data),
                               std::forward<E1>(bin_edges),
                               xt::ones<value_type>({ n }),
                               density,
                               true);
}

/**
 * @ingroup histogram
 * @brief Compute the histogram of a set of data.
 *
 * @param data The data.
 * @param bins The number of bins.
 * @param weights Weight factors corresponding to each data-point.
 * @param density If true the resulting integral is normalized to 1. [default: false]
 * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
 */
template <class E1, class E2>
inline auto histogram(E1&& data, std::size_t bins, E2&& weights, bool density = false)
{
  auto bin_edges = histogram_bin_edges(data, weights, bins);

  return detail::histogram_imp(std::forward<E1>(data),
                               std::forward<E1>(bin_edges),
                               std::forward<E2>(weights),
                               density,
                               true);
}


} // foam


#endif //FOAM_STATISTICS_HPP
