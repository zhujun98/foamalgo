################################################################################
# Distributed under the terms of the GNU General Public License v3.0.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Copyright (C) 2020, Jun Zhu. All rights reserved.
################################################################################

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# Download and unpack googletest at configure time
configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" -Wno-dev .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if (result)
    message(FATAL_ERROR "CMAKE step for gtest failed: ${result}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if (result)
    message(FATAL_ERROR "BUILD step for gtest failed: ${result}")
endif()

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds the following targets:
# gtest, gtest_main, gmock and gmock_main
add_subdirectory(
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL
)

set(CMAKE_WARN_DEPRECATED ON CACHE BOOL "" FORCE)

set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
set(GMOCK_INCLUDE_DIRS "${gmock_SOURCE_DIR}/include")

find_package(Threads REQUIRED)

set(FOAM_UNITTESTS
        test_tbb.cpp
        test_azimuthal_integrator.cpp
        test_canny.cpp
        test_geometry.cpp
        test_geometry_1m.cpp
        test_imageproc.cpp
        test_smooth.cpp
        test_statistics.cpp
        test_traits.cpp
        )

set(FOAM_UNITTESTS_REQUIRED_BLAS)

# Failed in compilation in macOS
if (NOT APPLE)
    list(APPEND FOAM_UNITTESTS test_blas.cpp)
    list(APPEND FOAM_UNITTESTS_REQUIRED_BLAS test_blas)
endif()

set(CMAKE_BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib")

include(defineFoamCompilerFlags)
define_foam_compile_flags(foam_test)

foreach(pathname IN LISTS FOAM_UNITTESTS)
    get_filename_component(filename ${pathname} NAME)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} main.cpp ${filename})
    target_include_directories(${targetname}
            PRIVATE
                $<$<OR:$<BOOL:${FOAMALGO_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>
                ${GTEST_INCLUDE_DIRS}
                ${GMOCK_INCLUDE_DIRS}
            )
    target_link_libraries(${targetname}
            PRIVATE
                foamalgo
                $<$<OR:$<BOOL:${FOAMALGO_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARIES}>
                gtest
                pthread
            )

    if (${targetname} IN_LIST FOAM_UNITTESTS_REQUIRED_BLAS)
        target_link_libraries(${targetname}
                PRIVATE
                    ${BLAS_LIBRARIES}
                    ${LAPACK_LIBRARIES}
                )
    endif()

    add_custom_target(f${targetname}
        COMMAND ${targetname}
        DEPENDS ${targetname} ${filename}
    )
endforeach()

add_executable(test_foamalgo main.cpp ${FOAM_UNITTESTS})

target_include_directories(test_foamalgo
        PRIVATE
            $<$<OR:$<BOOL:${FOAMALGO_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>
            ${GTEST_INCLUDE_DIRS}
            ${GMOCK_INCLUDE_DIRS}
        )

target_link_libraries(test_foamalgo
        PRIVATE
            gtest
            pthread
            foamalgo
            $<$<OR:$<BOOL:${FOAMALGO_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARIES}>
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
        )

add_custom_target(ftest
    COMMAND test_foamalgo
    DEPENDS test_foamalgo
)
