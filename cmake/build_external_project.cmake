################################################################################
# Distributed under the terms of the GNU General Public License v3.0.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Copyright (C) 2020, Jun Zhu. All rights reserved.
################################################################################

function(setup_external_project NAME)
    execute_process(
            COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${thirdparty_BINARY_DIR}/${NAME}-stage
    )
    if(result)
        message(FATAL_ERROR "CMake step for ${NAME} failed: ${result}")
    endif()

    execute_process(
            COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${thirdparty_BINARY_DIR}/${NAME}-stage
    )
    if(result)
        message(FATAL_ERROR "Build step for ${NAME} failed: ${result}")
    endif()
endfunction()

function(build_external_project NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/${NAME}-stage
    )
    if(result)
        message(FATAL_ERROR "CMake step for ${NAME} failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/${NAME}-stage
    )
    if(result)
        message(FATAL_ERROR "Build step for ${NAME} failed: ${result}")
    endif()

    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/${NAME}-src
                     ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/${NAME}-build)
endfunction()
