cmake_minimum_required(VERSION 3.9)

project(AP_FEATURES LANGUAGES C)

option(AP_FEATURES_ENABLE_BENCHMARK "Build binary executable." OFF)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_FIND_FRAMEWORK LAST)

find_package(PythonInterp)

if(PYTHONINTERP_FOUND)
  # Check python version
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import platform, sys; sys.stdout.write(platform.python_version().split('.')[0])"
    OUTPUT_VARIABLE PYTHON_VERSION
  )
  message("Python version is ${PYTHON_VERSION}")

  if(${PYTHON_VERSION} LESS "3")
    # Use FindPython3
    find_package(Python3 COMPONENTS Interpreter)

    if(Python3_FOUND)
      if(NOT DEFINED AP_FEATURES_INSTALL_PYTHON_EXT_DIR)
        # Get path for platform-dependent Python modules (since we install a binary library)
        execute_process(
          COMMAND ${Python3_EXECUTABLE} -c "import sys, sysconfig; sys.stdout.write(sysconfig.get_path('platlib', vars={'platbase': '${CMAKE_INSTALL_PREFIX}'}))"
          OUTPUT_VARIABLE AP_FEATURES_INSTALL_PYTHON_EXT_DIR
        )
        set(AP_FEATURES_INSTALL_PYTHON_EXT_DIR ${AP_FEATURES_INSTALL_PYTHON_EXT_DIR}
          CACHE PATH "Python extension module installation directory.")
      endif()

      if(NOT DEFINED AP_FEATURES_INSTALL_PYTHON_MODULE_DIR)
        # Get path for pure Python modules
        execute_process(
          COMMAND ${Python3_EXECUTABLE} -c "import sys, sysconfig; sys.stdout.write(sysconfig.get_path('platlib', vars={'platbase': '${CMAKE_INSTALL_PREFIX}'}))"
          OUTPUT_VARIABLE AP_FEATURES_INSTALL_PYTHON_MODULE_DIR
        )
        set(AP_FEATURES_INSTALL_PYTHON_MODULE_DIR ${AP_FEATURES_INSTALL_PYTHON_MODULE_DIR}
          CACHE PATH "Python module installation directory.")
      endif()
    endif(Python3_FOUND)

  else()
    if(NOT DEFINED AP_FEATURES_INSTALL_PYTHON_EXT_DIR)
      # Get path for platform-dependent Python modules (since we install a binary library)
      execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys, sysconfig; sys.stdout.write(sysconfig.get_path('platlib', vars={'platbase': '${CMAKE_INSTALL_PREFIX}'}))"
        OUTPUT_VARIABLE AP_FEATURES_INSTALL_PYTHON_EXT_DIR
      )
      set(AP_FEATURES_INSTALL_PYTHON_EXT_DIR ${AP_FEATURES_INSTALL_PYTHON_EXT_DIR}
        CACHE PATH "Python extension module installation directory.")
    endif()

    if(NOT DEFINED AP_FEATURES_INSTALL_PYTHON_MODULE_DIR)
      # Get path for pure Python modules
      execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys, sysconfig; sys.stdout.write(sysconfig.get_path('platlib', vars={'platbase': '${CMAKE_INSTALL_PREFIX}'}))"
        OUTPUT_VARIABLE AP_FEATURES_INSTALL_PYTHON_MODULE_DIR
      )
      set(AP_FEATURES_INSTALL_PYTHON_MODULE_DIR ${AP_FEATURES_INSTALL_PYTHON_MODULE_DIR}
        CACHE PATH "Python module installation directory.")
    endif()
  endif()
endif(PYTHONINTERP_FOUND)

include(BuildType)

set(CMAKE_POSITION_INDEPENDENT_CODE True)

if("${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
  message(FATAL_ERROR "You cannot build in the source directory. Please run cmake from a subdirectory called 'build'")
endif()

set(CMAKE_C_FLAGS_DEBUG "-g -Og -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -g")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_library(cost_terms SHARED cost_terms.c)

if(AP_FEATURES_ENABLE_BENCHMARK)
  add_executable(cost_terms_main cost_terms.c cost_terms_main.c)
endif()

install(TARGETS cost_terms DESTINATION ${AP_FEATURES_INSTALL_PYTHON_MODULE_DIR})

if(APPLE)
  find_package(OpenMP)

  if(NOT OpenMP_FOUND)
    # libomp 15.0+ from brew is keg-only, so have to search in other locations.
    # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
    execute_process(COMMAND brew --prefix libomp
      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES omp)
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
    find_package(OpenMP)
  endif()
else()
  find_package(OpenMP)
endif()

if(OPENMP_FOUND)
  message("OpenMP was found")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
  target_link_libraries(cost_terms OpenMP::OpenMP_C)

  if(AP_FEATURES_ENABLE_BENCHMARK)
    target_link_libraries(cost_terms_main OpenMP::OpenMP_C)
  endif()
else()
  message("OpenMP was not found")
endif()
