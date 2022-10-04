################################################################################
# Distributed under the terms of the GNU General Public License v3.0.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Copyright (C) 2020, Jun Zhu. All rights reserved.
################################################################################

# dependencies
# ============

find_package(NumPy REQUIRED)
message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIRS}")

# pybind11
find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11: ${pybind11_VERSION}, ${pybind11_INCLUDE_DIRS}")

# xtensor-python
find_package(xtensor-python REQUIRED)
message(STATUS "Found xtensor-python: ${xtensor-python_VERSION}, ${xtensor-python_INCLUDE_DIRS}")

# ============
# Build flags
# ============

include(defineFoamCompilerFlags)
define_foam_compile_flags(foam_py)

# ===============
# Build libraries
# ===============

set(FOAM_MODULES
        azimuthal_integrator.cpp
        canny.cpp
        geometry.cpp
        geometry_1m.cpp
        miscellaneous.cpp
        imageproc.cpp
        smooth.cpp
        statistics.cpp
        )

set(FOAM_MODULES_REQUIRE_BLAS)

if(APPLE)
    set(target_install_rpath "@loader_path/")
else()
    set(target_install_rpath "\$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH "${target_install_rpath};${CMAKE_PREFIX_PATH}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

foreach(pathname IN LISTS FOAM_MODULES)
    get_filename_component(filename ${pathname} NAME)
    string(REPLACE ".cpp" "" modulename ${filename})
    string(REGEX REPLACE "^f_" "" modulename ${modulename})
    pybind11_add_module(${modulename} ${filename})
    target_include_directories(${modulename}
            PRIVATE
                include
                ${NUMPY_INCLUDE_DIRS}
                $<$<OR:$<BOOL:${FOAMALGO_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>
            )
    target_link_libraries(${modulename}
            PRIVATE
                foamalgo
                $<$<OR:$<BOOL:${FOAMALGO_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARY}>
                pybind11::pybind11
                xtensor-python
            )

    if (${modulename} IN_LIST FOAM_MODULES_REQUIRE_BLAS)
        target_link_libraries(${modulename} PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()

endforeach()