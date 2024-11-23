##===- SingleMultiSource.cmake --------------------------------------------===##
#
# This defines functions that are used by the Kitsune-specific tests in this
# suite. They generally mimic the corresponding llvm_* functions in this suite.
#
##===----------------------------------------------------------------------===##

include(SingleMultiSource)

function (source_language source lang)
  # We treat Kokkos as its own language.
  if (${source} MATCHES ".+-kokkos[.].+$")
    set(${lang} "kokkos" PARENT_SCOPE)
  elseif (${source} MATCHES ".+[.]c$")
    set(${lang} "c" PARENT_SCOPE)
  elseif (${source} MATCHES ".+[.]cpp$" OR ${source} MATCHES ".+[.]cc$")
    set(${lang} "c++" PARENT_SCOPE)
  elseif (${source} MATCHES ".+[.]cu$")
    set(${lang} "cuda" PARENT_SCOPE)
  elseif (${source} MATCHES ".+[.]hip$")
    set(${lang} "hip" PARENT_SCOPE)
  elseif (${source} MATCHES ".+[.][Ff]$" OR
      ${source} MATCHES ".+[.][Ff]90$" OR
      ${source} MATCHES ".+[.][Ff]95$" OR
      ${source} MATCHES ".+[.][Ff]03$" OR
      ${source} MATCHES ".+[.][Ff]08$")
    set(${lang} "fortran" PARENT_SCOPE)
  else ()
    message(FATAL_ERROR "Cannot determine source language: ${source}")
  endif ()
endfunction ()

function (kitsune_singlesource_test source lang tapir_target)
  basename(base ${source})
  if (${tapir_target} STREQUAL "none")
    set(target "${base}-${lang}-lang")
  else ()
    set(target "${base}-${tapir_target}")
  endif ()
  set(test_name "${target}")

  llvm_test_executable_no_test(${target} ${source})
  llvm_test_traditional(${target})
  llvm_add_test_for_target(${target})

  set_property(TARGET ${target} PROPERTY TEST_NAME "${test_name}")
  if (${lang} STREQUAL "cuda")
    set_target_properties(${target} PROPERTIES
      CUDA_COMPILER_LAUNCHER ${NVCC})
  elseif (${lang} STREQUAL "hip")
    set_target_properties(${target} PROPERTIES
      HIP_COMPILER_LAUNCHER ${HIPCC})
  elseif (${lang} STREQUAL "c")
    target_compile_options(${target} PUBLIC
      "-ftapir=${tapir_target}" "-fno-exceptions" "${CMAKE_C_FLAGS}")
  elseif (${lang} STREQUAL "c++")
    target_compile_options(${target} PUBLIC
      "-ftapir=${tapir_target}" "-fno-exceptions" "${CMAKE_CXX_FLAGS}")
  elseif (${lang} STREQUAL "fortran")
    target_compile_options(${target} PUBLIC
      "-ftapir=${tapir_target}" "-fno-exceptions" "${CMAKE_Fortran_FLAGS}")
  else ()
    message(FATAL_ERROR "Unsupported language: ${lang} [${source}]")
  endif ()
endfunction()

# Configure the current directory as a SingleSource subdirectory - i.e. every
# C/C++/Fortran file is treated as its own test.
function(kitsune_singlesource)
  file(GLOB sources
    *.c
    *.cpp *.cc
    *.cu
    *.f *.F *.f90 *.F90 *.f03 *.F03 *.f08 *.F08
    *.hip)
  foreach(source ${sources})
    set(is_kokkos_source OFF)
    set(is_cuda_source OFF)
    set(is_hip_source OFF)

    set(lang)
    source_language(${source} lang)

    # Fortran is not yet supported. Complain loudly so we know to change this
    # and take a closer look when Fortran is supported.
    if (${lang} STREQUAL "fortran")
      message(FATAL_ERROR "Kitsune does not yet support Fortran: ${source}")
    endif ()

    if (${lang} STREQUAL "cuda" AND ${TEST_CUDA_LANG})
      kitsune_singlesource_test(${source} "cuda-lang" "none")
    elseif (${lang} STREQUAL "hip" AND ${TEST_HIP_LANG})
      kitsune_singlesource_test(${source} "hip-lang" "none")
    else ()
      if (TEST_CUDA_TARGET)
        kitsune_singlesource_test(${source} ${lang} "cuda")
      endif ()
      if (TEST_HIP_TARGET)
        kitsune_singlesource_test(${source} ${lang} "hip")
      endif()
      if (TEST_LAMBDA_TARGET)
        kitsune_singlesource_test(${source} ${lang} "lambda")
      endif ()
      if (TEST_OMPTASK_TARGET)
        kitsune_singlesource_test(${source} ${lang} "omptask")
      endif ()
      if (TEST_OPENCILK_TARGET)
        kitsune_singlesource_test(${source} ${lang} "opencilk")
      endif()
      if (TEST_OPENMP_TARGET)
        kitsune_singlesource_test(${source} ${lang} "openmp")
      endif ()
      if (TEST_QTHREADS_TARGET)
        kitsune_singlesource_test(${source} ${lang} "qthreads")
      endif ()
      if (TEST_REALM_TARGET)
        kitsune_singlesource_test(${source} ${lang} "realm")
      endif ()
      if (TEST_SERIAL_TARGET)
        kitsune_singlesource_test(${source} ${lang} "serial")
      endif ()
    endif ()
  endforeach()
endfunction()


