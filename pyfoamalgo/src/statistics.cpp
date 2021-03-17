/**
 * Distributed under the terms of the GNU General Public License v3.0.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Copyright (C) 2020, Jun Zhu. All rights reserved.
 */
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "foamalgo/statistics.hpp"
#include "pyconfig.hpp"

namespace py = pybind11;


PYBIND11_MODULE(statistics, m)
{

  using namespace foam;

  xt::import_numpy();

  m.doc() = "A collection of statistics functions.";

#define FOAM_NAN_REDUCER_IMP(REDUCER, VALUE_TYPE, N_DIM)                                          \
  m.def(#REDUCER, [] (const xt::pytensor<VALUE_TYPE, N_DIM>& src, const std::vector<int>& axis)   \
  {                                                                                               \
    return xt::eval(xt::REDUCER<VALUE_TYPE>(src, axis));                                          \
  }, py::arg("src").noconvert(), py::arg("axis"));                                                \
  m.def(#REDUCER, [] (const xt::pytensor<VALUE_TYPE, N_DIM>& src, int axis)                       \
  {                                                                                               \
    return xt::eval(xt::REDUCER<VALUE_TYPE>(src, {axis}));                                        \
  }, py::arg("src").noconvert(), py::arg("axis"));                                                \
  m.def(#REDUCER, [] (const xt::pytensor<VALUE_TYPE, N_DIM>& src)                                 \
  {                                                                                               \
    return xt::eval(xt::REDUCER<VALUE_TYPE>(src))[0];                                             \
  }, py::arg("src").noconvert());

#define FOAM_NAN_REDUCER_ALL_DIMENSIONS(FUNCTOR, VALUE_TYPE)                                   \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 1)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 2)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 3)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 4)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 5)

#define FOAM_NAN_REDUCER(FUNCTOR)                                                              \
  FOAM_NAN_REDUCER_ALL_DIMENSIONS(FUNCTOR, float)                                              \
  FOAM_NAN_REDUCER_ALL_DIMENSIONS(FUNCTOR, double)

  FOAM_NAN_REDUCER(nansum)
  FOAM_NAN_REDUCER(nanmean)
  FOAM_NAN_REDUCER(nanstd)
  FOAM_NAN_REDUCER(nanvar)


#define FOAM_HISTOGRAM_IMP(VALUE_TYPE)                                                                \
  m.def("histogram1d", [] (const xt::pytensor<VALUE_TYPE, 1>& src,                                    \
                           VALUE_TYPE left,                                                           \
                           VALUE_TYPE right,                                                          \
                           size_t bins)                                                               \
  {                                                                                                   \
    return xt::histogram(src, bins, left, right);                                                     \
  }, py::arg("src").noconvert(), py::arg("left"), py::arg("right"), py::arg("bins"));

  FOAM_HISTOGRAM_IMP(float)
  FOAM_HISTOGRAM_IMP(double)

}
