##===- KitsuneTestSuite.cmake ---------------------------------------------===##
#
# This defines functions that are used by the Kitsune-specific tests in this
# suite. They generally mimic the corresponding llvm_* functions in this suite.
#
##===----------------------------------------------------------------------===##

include(SingleMultiSource)

# Look at the file name to determine the source language. This is not really a
# source language in the strict sense of the term, but something for internal
# use that will help determine what to do with the file. The "not widely
# recognized as languages" languages that this will return are:
#
#    kitc       C files with kitsune-specific extensions
#    kitc++     C++ files with kitsune-specific extensions
#    kokkos     C++ files with Kokkos. These may or may not contain any
#               Kitsune-specific extensions
#
#    c          C files without kitsune-specific extensions
#    c++        C++ files without kitsune-specific extensions
#    fortran    Fortran files
#    cuda       Cuda files (those with a .cu extension)
#    hip        Hip files (those with a .hip extension)
#
function (source_language source lang)
  # We treat Kokkos as its own language.
  if (source MATCHES ".+[.]kokkos[.]cpp$" OR source MATCHES ".+[.]kokkos[.]c$")
    set(${lang} "kokkos" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.]c$" OR source MATCHES ".+[.]kit[.]c$")
    set(${lang} "kitc" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.]cpp$" OR source MATCHES ".+[.]kit[.]cc$")
    set(${lang} "kitc++" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]c$")
    set(${lang} "c" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]cpp$" OR source MATCHES ".+[.]cc$")
    set(${lang} "c++" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]cu$")
    set(${lang} "cuda" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]hip$")
    set(${lang} "hip" PARENT_SCOPE)
  elseif (source MATCHES ".+[.][Ff]$" OR
      source MATCHES ".+[.][Ff]90$" OR
      source MATCHES ".+[.][Ff]95$" OR
      source MATCHES ".+[.][Ff]03$" OR
      source MATCHES ".+[.][Ff]08$")
    set(${lang} "fortran" PARENT_SCOPE)
  else ()
    message(FATAL_ERROR "Cannot determine source language: ${source}")
  endif ()
endfunction ()

# Setup a single-source test for the given tapir target.
function (kitsune_singlesource_test source lang tapir_target)
  get_filename_component(base "${source}" NAME)
  if (tapir_target STREQUAL "none")
    set(target "${base}.nokit")
  else ()
    set(target "${base}.${tapir_target}")
  endif ()

  message(STATUS "Setting up test: ${target}")
  llvm_test_executable_no_test(${target} ${source})
  llvm_test_run()
  llvm_add_test_for_target(${target})

  if (NOT tapir_target STREQUAL "none")
    target_compile_options(${target} BEFORE PUBLIC
      "-ftapir=${tapir_target}")
    target_link_options(${target} BEFORE PUBLIC
      "-ftapir=${tapir_target}")
  endif ()

  if (lang STREQUAL "kitc++" OR lang STREQUAL "kokkos")
    target_compile_options(${target} BEFORE PUBLIC
      "-fno-exceptions")
  endif ()

  if (lang STREQUAL "kokkos")
    target_compile_options(${target} BEFORE PUBLIC
      "-fkokkos" "-fkokkos-no-init")
  endif ()
endfunction()

# Add tests for all tapir targets being tested. This should not be used by
# consumers. They should call kitsune_singlesource() instead.
function(kitsune_singlesource_all_targets source lang)
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
    set(lang)
    source_language(${source} lang)

    if (lang STREQUAL "cuda")
      if (TEST_CUDA_LANG)
        kitsune_singlesource_test(${source} ${lang} "none")
      endif ()
    elseif (lang STREQUAL "hip")
      if (TEST_HIP_LANG)
        kitsune_singlesource_test(${source} ${lang} "none")
      endif ()
    elseif (lang STREQUAL "kokkos")
      if (TEST_KOKKOS_LANG)
        kitsune_singlesource_test(${source} ${lang} "none")
      endif ()
      if (TEST_KOKKOS_MODE)
        # We only care about testing Kokkos with the GPU-centric targets
        if (TEST_CUDA_TARGET)
          kitsune_singlesource_test(${source} ${lang} "cuda")
        endif ()
        if (TEST_HIP_TARGET)
          kitsune_singlesource_test(${source} ${lang} "hip")
        endif ()
      endif ()
    elseif (lang STREQUAL "kitc")
      if (TEST_C)
        kitsune_singlesource_all_targets(${source} ${lang})
      endif ()
    elseif (lang STREQUAL "kitc++")
      if (TEST_CXX)
        kitsune_singlesource_all_targets(${source} ${lang})
      endif ()
    elseif (lang STREQUAL "fortran")
      # Fortran is not yet supported. Complain loudly so we know to change this
      # and take a closer look at everything when Fortran is supported.
      message(FATAL_ERROR "Kitsune does not yet support Fortran: ${source}")
      if (TEST_Fortran)
        kitsune_singlesource_all_targets(${source} ${lang})
      endif ()
    else ()
      message(FATAL_ERROR "Testing of file not supported: ${source} [${lang}]")
    endif ()
  endforeach()
endfunction()


