################################################################################
# Distributed under the terms of the GNU General Public License v3.0.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Jun Zhu
################################################################################

cmake_minimum_required(VERSION 3.24)  # modify "setup.py" and "environment-dev.yml" at the same time

project(foamalgo LANGUAGES CXX VERSION 0.1.2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(FOAMALGO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FOAMALGO_HEADER_DIR ${FOAMALGO_INCLUDE_DIR}/${PROJECT_NAME})

# versioning
# ==========
configure_file(foamalgo_version.in ${FOAMALGO_HEADER_DIR}/foamalgo_version.hpp @ONLY)

# options
# =======

OPTION(FOAMALGO_USE_TBB "Build ${PROJECT_NAME} (xtensor is not included) with oneTBB" ON)

OPTION(FOAMALGO_USE_XSIMD "Build ${PROJECT_NAME} (xtensor is not included) with XSIMD" ON)

OPTION(BUILD_FOAMALGO_CPP_TESTS "Build c++ unittests" OFF)

OPTION(BUILD_PYFOAMALGO "Build Python bindings" OFF)

if (NOT DEFINED XTENSOR_USE_TBB)
    set(XTENSOR_USE_TBB OFF)
endif()
if (NOT DEFINED XTENSOR_USE_XSIMD)
    set(XTENSOR_USE_XSIMD OFF)
endif()

# dependencies
# ============

set(thirdparty_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

# tbb
if(FOAMALGO_USE_TBB OR XTENSOR_USE_TBB)
    find_package(TBB 2021.5 REQUIRED)
    message(STATUS "TBB FOUND: ${TBB_VERSION}, ${TBB_LIBRARIES}")

    if(FOAMALGO_USE_TBB)
        message(STATUS "Build ${PROJECT_NAME} with oneTBB")
        add_compile_definitions(FOAMALGO_USE_TBB)
    endif()
endif()

# xsimd
if(FOAMALGO_USE_XSIMD OR XTENSOR_USE_XSIMD)
    find_package(xsimd REQUIRED)
    message(STATUS "Found xsimd: ${xsimd_VERSION}, ${xsimd_INCLUDE_DIRS}")

    if(FOAMALGO_USE_XSIMD)
        message(STATUS "Build ${PROJECT_NAME} with XSIMD")
        add_compile_definitions(FOAMALGO_USE_XSIMD)
    endif()
endif()

# xtensor
find_package(xtensor REQUIRED)
message(STATUS "Found xtensor: ${xtensor_VERSION}, ${xtensor_INCLUDE_DIRS}")

# foamalgo

set(FOAM_HEADERS
        ${FOAMALGO_HEADER_DIR}/foamalgo_config.hpp
        ${FOAMALGO_HEADER_DIR}/foamalgo_version.hpp
        ${FOAMALGO_HEADER_DIR}/geometry.hpp
        ${FOAMALGO_HEADER_DIR}/geometry_1m.hpp
        ${FOAMALGO_HEADER_DIR}/imageproc.hpp
        ${FOAMALGO_HEADER_DIR}/miscellaneous.hpp
        ${FOAMALGO_HEADER_DIR}/statistics.hpp
        ${FOAMALGO_HEADER_DIR}/traits.hpp
        ${FOAMALGO_HEADER_DIR}/utilities.hpp
        )

# check header files
foreach(header IN LISTS FOAM_HEADERS)
    if(NOT EXISTS ${header})
        message( FATAL_ERROR "${header} does not exist!" )
    endif()
endforeach()

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
        INTERFACE
            $<BUILD_INTERFACE:${FOAMALGO_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>
        )

target_link_libraries(${PROJECT_NAME}
        INTERFACE
            xtensor
            $<$<BOOL:${FOAMALGO_USE_XSIMD}>:xsimd>
        )

# ============
# Build flags
# ============

if(CMAKE_CXX_COMPILER_VERSION)
    set(FOAM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (version ${CMAKE_CXX_COMPILER_VERSION})")
else()
    set(FOAM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${FOAM_COMPILER_STR}" FOAM_COMPILER_STR)

# ============
# Unit test
# ============

if(BUILD_FOAMALGO_CPP_TESTS)
    add_subdirectory(tests)
endif()

# ============
# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${FOAM_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

configure_package_config_file(
    ${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_NAME}_VERSION
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# ============
# FOAM_PYTHON
# ============

if(BUILD_PYFOAMALGO)
    add_subdirectory(pyfoamalgo/src)
endif()
