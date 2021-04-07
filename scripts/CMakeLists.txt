#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(TARGET "dpc_backend")

if(NOT DEFINED ENV{DPCPPROOT})
    message(FATAL_ERROR "Not set DPCPPROOT env")
else()
    set(DPCPPROOT $ENV{DPCPPROOT})
endif()

if(NOT DEFINED ENV{DPCTLROOT})
    message(FATAL_ERROR "Not set DPCTLROOT env")
else()
    set(DPCTLROOT $ENV{DPCTLROOT})
endif()

if(WIN32)
    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_C_COMPILER "clang-cl")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
    message(STATUS "Resetting Linker to: " ${CMAKE_LINKER})
    # set(WARNING_FLAGS "-Wextra -Winit-self -Wuninitialized -Wmissing-declarations")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} -Qstd=c++17 /EHsc")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG -Qstd=c++17")
elseif(UNIX)
    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_C_COMPILER "clang")
    set(SDL_FLAGS "-fstack-protector -fstack-protector-all -fpic -fPIC -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks")
    # set(WARNING_FLAGS "-Wextra -Winit-self -Wuninitialized -Wmissing-declarations -fdiagnostics-color=auto")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS} -std=c++17 -fsycl")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG -std=c++17 -fsycl")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

project("dpcpp part python onedal"
    HOMEPAGE_URL https://github.com/oneapi-src/oneDAL
)

set(PYTHON_ONEDAL "${CMAKE_CURRENT_SOURCE_DIR}/../onedal/")

file(GLOB_RECURSE sources
    ${PYTHON_ONEDAL}/**/*.cpp
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

set(USE_DPCPP "yes")
set(USE_NEW_IFACES "yes")
set(LINK_TYPE "so")
set(THREADING_TYPE "parallel")

# find_package(oneDAL REQUIRED)
find_package(Python REQUIRED COMPONENTS Development NumPy)
message("python include dir:" ${Python_INCLUDE_DIRS})
message("python libraries:" ${Python_LIBRARY_DIRS})
message("oneDAL_INCLUDE_DIRS:" ${oneDAL_INCLUDE_DIRS})
message("DPCPP_SYCL_LIBRARY:" ${DPCPP_SYCL_LIBRARY})
message("DPCPP_OPENCL_LIBRARY:" ${DPCPP_OPENCL_LIBRARY})
message("Python_LIBRARY_DIRS:" ${Python_LIBRARY_DIRS})
message("Python_LIBRARIES:" ${Python_LIBRARIES})

add_library(${TARGET}
    SHARED
    ${sources}
)

target_include_directories(${TARGET}
    PRIVATE ${PYTHON_ONEDAL}
    PRIVATE ${oneDAL_INCLUDE_DIRS}
    PRIVATE ${Python_INCLUDE_DIRS}
    PRIVATE ${Python_NumPy_INCLUDE_DIRS}
)

target_compile_definitions(${TARGET}
    PUBLIC "_DPCPP_"
    PUBLIC "ONEDAL_DATA_PARALLEL"
    PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
)

target_link_libraries(${TARGET}
    PRIVATE ${DPCPP_OPENCL_LIBRARY}
    PRIVATE ${Python_LIBRARIES}
)

# TODO: Get from findOneDAL
set(ONEDAL_LIBRARIES "onedal_core")
list(APPEND ONEDAL_LIBRARIES "onedal_thread")
list(APPEND ONEDAL_LIBRARIES "onedal_dpc")
target_link_libraries(${TARGET} PUBLIC ${ONEDAL_LIBRARIES})


if(DPCTL_ENABLE)
    target_include_directories(${TARGET} PUBLIC ${DPCTL_INCLUDE_DIR})
    target_link_directories(${TARGET} PUBLIC ${DPCTL_LIB_DIR})
    target_link_libraries(${TARGET} PUBLIC "DPCTLSyclInterface")
    target_compile_definitions(${TARGET} PUBLIC "DPCTL_ENABLE")

    # not sure but add runpath
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${DPCTL_LIB_DIR}")

    # disable stripping rpath in installation logic
    set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

install(
  TARGETS ${TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
