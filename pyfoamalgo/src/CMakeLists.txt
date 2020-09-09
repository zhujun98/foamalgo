################################################################################
# Distributed under the terms of the GNU General Public License v3.0.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Copyright (C) 2020, Jun Zhu. All rights reserved.
################################################################################

# dependencies
# ============

# pybind11
set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(thirdparties/pybind11)
find_package(pybind11 REQUIRED)

# xtensor-python
add_subdirectory(thirdparties/xtensor-python)

# ============
# Build flags
# ============

include(defineFoamCompilerFlags)
define_foam_compile_flags(foam_py)

# ===============
# Build libraries
# ===============

set(_FOAM_MODULE_FILES
    azimuthal_integrator.cpp
    canny.cpp
    geometry.cpp
    geometry_1m.cpp
    helpers.cpp
    imageproc.cpp
    smooth.cpp
    statistics.cpp
)

set(_FOAM_MODULES_NEED_BLAS)

if(APPLE)
    set(target_install_rpath "@loader_path/")
else()
    set(target_install_rpath "\$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH "${target_install_rpath};${CMAKE_PREFIX_PATH}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

foreach(pathname IN LISTS _FOAM_MODULE_FILES)
    get_filename_component(filename ${pathname} NAME)
    string(REPLACE ".cpp" "" modulename ${filename})
    string(REGEX REPLACE "^f_" "" modulename ${modulename})
    pybind11_add_module(${modulename} ${filename})
    target_include_directories(${modulename}
        PRIVATE
            include
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>)
    target_link_libraries(${modulename}
        PRIVATE
            foamalgo
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARY}>
            xtensor-python)

    if (${modulename} IN_LIST _FOAM_MODULES_NEED_BLAS)
        target_link_libraries(${modulename} PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
endforeach()