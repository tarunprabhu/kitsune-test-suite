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
# The "recognized as a language by a different name" are:
#
#    cuda-lang  Cuda files (those with a .cu extension)
#    hip-lang   Hip files (those with a .hip extension)
#
# These are done to distinguish them from the cuda and hip tapir targets.
#
# The other source languages that this will return are c, c++, and fortran
# which are all what one would expect.
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
    set(${lang} "cuda-lang" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]hip$")
    set(${lang} "hip-lang" PARENT_SCOPE)
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

function (kitsune_singlesource_test source lang tapir_target)
  # For now, we only run tests with Tapir. Anything that does not use a tapir
  # target is only needed when we also want performance comparisons, not just
  # correctness checks. That would involve dealing with other non-Kitsune
  # compilers at the same time as running Kitsune which the LLVM test suite is
  # not really setup to do. We could try and work around things, but, for now
  # at least, just do enough to get things off the ground for correctness.
  if (tapir_target STREQUAL "none")
    return ()
  elseif (NOT lang STREQUAL "kitc"
      AND NOT lang STREQUAL "kitc++"
      AND NOT lang STREQUAL "kokkos"
      AND NOT lang STREQUAL "fortran")
    # This can happen for .cu or .hip files. We don't want to deal with those
    # for now. Eventually all of this should go away.
    return ()
  endif ()

  get_filename_component(base "${source}" NAME)
  if (tapir_target STREQUAL "none")
    set(target "${base}.nokit")
  else ()
    set(target "${base}.${tapir_target}")
  endif ()
  set(test_name "${target}")

  message(STATUS "Setting up test: ${target}")
  llvm_test_executable(${target} ${source})

  if (NOT tapir_target STREQUAL "none")
    target_compile_options(${target} BEFORE PUBLIC
      "-ftapir=${tapir_target}")
    target_link_options(${target} BEFORE PUBLIC
      "-ftapir=${tapir_target}")
  endif ()

  if (lang STREQUAL "kitc++")
    target_compile_options(${target} BEFORE PUBLIC
      "-fno-exceptions")
  elseif (lang STREQUAL "kokkos")
    target_compile_options(${target} BEFORE PUBLIC
      "-fkokkos" "-fkokkos-no-init" "-fno-exceptions")
  endif ()
endfunction()

# Setup a single source test for all tapir targets that are being tested.
# This is not intended to be used directly. Consumers should use
# kitsune_singlesource()
function(kitsune_singlesource_setup source lang)
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

    # Fortran is not yet supported. Complain loudly so we know to change this
    # and take a closer look when Fortran is supported.
    if (lang STREQUAL "fortran")
      message(FATAL_ERROR "Kitsune does not yet support Fortran: ${source}")
    endif ()

    if (lang STREQUAL "cuda-lang")
      if (TEST_CUDA_LANG)
        kitsune_singlesource_test(${source} ${lang} "none")
      endif ()
    elseif (lang STREQUAL "hip-lang")
      if (TEST_HIP_LANG)
        kitsune_singlesource_test(${source} ${lang} "none")
      endif ()
    elseif (lang STREQUAL "kokkos")
      if (TEST_KOKKOS_LANG)
        # This should be tested with a regular C++ compiler, so change the
        # language to C++, otherwise, it will activate -fkokkos-mode in Kitsune.
        kitsune_singlesource_test(${source} "c++" "none")
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
        kitsune_singlesource_setup(${source} ${lang})
      endif ()
    elseif (lang STREQUAL "kitc++" AND TEST_CXX)
      if (TEST_CXX)
        kitsune_singlesource_setup(${source} ${lang})
      endif ()
    elseif (lang STREQUAL "fortran" AND TEST_Fortran)
      if (TEST_Fortran)
        kitsune_singlesource_setup(${source} ${lang})
      endif ()
    else ()
      message(FATAL_ERROR "Testing of file not supported: ${source} [${lang}]")
    endif ()
  endforeach()
endfunction()


