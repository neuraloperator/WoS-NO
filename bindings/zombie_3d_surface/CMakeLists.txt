cmake_minimum_required(VERSION 3.15...3.27)
project(zombie)

option(ZOMBIE_BUILD_DEMO "Build demo" OFF)
option(ZOMBIE_BUILD_BINDINGS "Build bindings" ON)

################################################################################
if(ZOMBIE_BUILD_BINDINGS)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.10
        REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_definitions(-DNDEBUG)
endif()

################################################################################
# add dependencies
get_directory_property(hasParent PARENT_DIRECTORY)

set(TBB_TEST          OFF CACHE BOOL " " FORCE)
set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)

if(hasParent)
    set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include PARENT_SCOPE)
    set(ZOMBIE_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include PARENT_SCOPE)
else()
    set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
    set(ZOMBIE_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include)
endif()

set(FCPW_USE_ENOKI OFF CACHE BOOL "Enable CPU vectorization" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fcpw)

list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/bessel)
list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/pcg32)
list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/demo/deps/stb)
list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/demo/deps/nlohmann)
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_TBB_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_NANOFLANN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_EIGEN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_ENOKI_INCLUDES})

################################################################################
# build library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/pybind11)
pybind11_add_module(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/demo/demo.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include> ${ZOMBIE_DEPS_INCLUDES})
target_link_libraries(${PROJECT_NAME} PUBLIC fcpw TBB::tbb)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

################################################################################
# build demo
if(ZOMBIE_BUILD_DEMO)
    add_subdirectory(demo)
endif()

# build Python bindings
if(ZOMBIE_BUILD_BINDINGS)
    add_subdirectory(deps/nanobind)
    add_subdirectory(python)
endif()
