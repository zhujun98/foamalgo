################################################################################
# Distributed under the terms of the GNU General Public License v3.0.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Copyright (C) 2020, Jun Zhu. All rights reserved.
################################################################################

cmake_minimum_required(VERSION 3.13)

project(foamalgo LANGUAGES CXX VERSION 0.0.8)

set(CMAKE_CXX_STANDARD 14)
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

OPTION(FOAM_USE_TBB "Build foamalgo (xtensor is not included) with intel TBB" ON)

OPTION(FOAM_USE_XSIMD "Build foamalgo (xtensor is not included) with XSIMD" ON)

OPTION(BUILD_FOAM_TESTS "Build c++ unit test" ON)

OPTION(BUILD_FOAM_PYTHON "Build FOAM Python bindings" OFF)

# dependencies
# ============

set(xsimd_REQUIRED_VERSION 7.4.8)
set(xtl_REQUIRED_VERSION 0.6.16)
set(xtensor_REQUIRED_VERSION 0.21.5)
set(xtensor-blas_REQUIRED_VERSION 0.17.2)

set(thirdparty_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

include(${PROJECT_SOURCE_DIR}/cmake/build_external_project.cmake)

# tbb
if(FOAM_USE_TBB OR XTENSOR_USE_TBB)
    set(TBB_REQUIRED_VERSION 2019.9)
    string(REPLACE "." "_U" TBB_REQUIRED_VERSION_STR ${TBB_REQUIRED_VERSION})

    add_compile_definitions(TBB_SUPPRESS_DEPRECATED_MESSAGES)

    find_package(TBB QUIET)
    if(TBB_FOUND AND ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} VERSION_GREATER_EQUAL TBB_REQUIRED_VERSION)
        set(TBB_LIBRARY ${TBB_LIBRARIES_RELEASE})
    else()
        message(STATUS "Downloading TBB ${TBB_REQUIRED_VERSION_STR}")

        configure_file(cmake/downloadTbb.cmake.in ${thirdparty_BINARY_DIR}/tbb-stage/CMakeLists.txt)
        setup_external_project( tbb )

        include(${thirdparty_BINARY_DIR}/tbb-src/cmake/TBBBuild.cmake)
        tbb_build(TBB_ROOT ${thirdparty_BINARY_DIR}/tbb-src
                  CONFIG_DIR TBB_DIR
                  MAKE_ARGS tbb_build_dir=${thirdparty_BINARY_DIR}/tbb-build
                            tbb_build_prefix=tbb_cmake_build)
        find_package(TBB ${TBB_REQUIRED_VERSION} EXACT CONFIG REQUIRED COMPONENTS tbb tbbmalloc)

        get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(TBB_LIBRARY TBB::tbb IMPORTED_LOCATION_RELEASE)
    endif()

    message(STATUS "TBB FOUND ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}: ${TBB_INCLUDE_DIRS}, ${TBB_LIBRARY}")

    if(FOAM_USE_TBB)
        message(STATUS "Build foamalgo with intel TBB")
        add_compile_definitions(FOAM_USE_TBB)
    endif()
endif()

# xsimd
if(FOAM_USE_XSIMD OR XTENSOR_USE_XSIMD)
    find_package(xsimd ${xsimd_REQUIRED_VERSION} QUIET)

    if(xsimd_FOUND)
        message(STATUS "Found xsimd: ${xsimd_VERSION}, ${xsimd_INCLUDE_DIRS}")
    else()
        configure_file(cmake/downloadXsimd.cmake.in thirdparty/xsimd-stage/CMakeLists.txt)
        build_external_project( xsimd )
        set(xsimd_DIR ${xsimd_BINARY_DIR})
    endif()

    if(FOAM_USE_XSIMD)
        message(STATUS "Build foamalgo with XSIMD")
        add_compile_definitions(FOAM_USE_XSIMD)
    endif()

endif()

# xtl
find_package(xtl ${xtl_REQUIRED_VERSION} QUIET)
if(xtl_FOUND)
    message(STATUS "Found xtl: ${xtl_VERSION}, ${xtl_INCLUDE_DIRS}")
else()
    configure_file(cmake/downloadXtl.cmake.in thirdparty/xtl-stage/CMakeLists.txt)
    build_external_project( xtl )
    set(xtl_DIR ${xtl_BINARY_DIR})
endif()

# xtensor
find_package(xtensor ${xtensor_REQUIRED_VERSION} QUIET)
if(xtensor_FOUND)
    message(STATUS "Found xtensor: ${xtensor_VERSION}, ${xtensor_INCLUDE_DIRS}")
else()
    configure_file(cmake/downloadXtensor.cmake.in thirdparty/xtensor-stage/CMakeLists.txt)
    build_external_project( xtensor )
    # required by xtensor-python
    set(xtensor_DIR ${xtensor_BINARY_DIR})
endif()

# xtensor-blas
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
find_package(xtensor-blas ${xtensor-blas_REQUIRED_VERSION} QUIET)
if(xtensor-blas_FOUND)
    message(STATUS "Found xtensor-blas: ${xtensor-blas_VERSION}, ${xtensor-blas_INCLUDE_DIRS}")
else()
    configure_file(cmake/downloadXtensorBlas.cmake.in thirdparty/xtensor-blas-stage/CMakeLists.txt)
    build_external_project( xtensor-blas )
    set(xtensor-blas_DIR ${xtensor-blas_BINARY_DIR})
endif()

# foamalgo

set(FOAM_HEADERS
    ${FOAMALGO_HEADER_DIR}/canny.hpp
    ${FOAMALGO_HEADER_DIR}/foamalgo_config.hpp
    ${FOAMALGO_HEADER_DIR}/foamalgo_version.hpp
    ${FOAMALGO_HEADER_DIR}/geometry.hpp
    ${FOAMALGO_HEADER_DIR}/geometry_1m.hpp
    ${FOAMALGO_HEADER_DIR}/imageproc.hpp
    ${FOAMALGO_HEADER_DIR}/miscellaneous.hpp
    ${FOAMALGO_HEADER_DIR}/smooth.hpp
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
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        xtensor
        xtensor-blas
        $<$<BOOL:${FOAM_USE_XSIMD}>:xsimd>)

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

if(BUILD_FOAM_TESTS)
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

if(BUILD_FOAM_PYTHON)
    add_subdirectory(pyfoamalgo/src)
endif()
