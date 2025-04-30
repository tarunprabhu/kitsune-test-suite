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
#    kitcxx     C++ files with kitsune-specific extensions
#    kitfort    Fortran files with kitsune-specific extensions and/or language
#               support. For instance, if it is a Fortran source file where
#               the DO CONCURRENT construct is intended to be lowered via the
#               tapir dialect instead of the standard/OpenMP dialects
#    kitkokkos  C++ files with Kokkos. These may or may not contain any
#               Kitsune-specific extensions, but these are intended to be
#               compiled by Kitsune with --kokkos
#
#    c          C files without kitsune-specific extensions
#    cxx        C++ files without kitsune-specific extensions
#    fortran    Fortran files without any special handling
#    cuda       Cuda files (those with a .cu extension)
#    hip        Hip files (those with a .hip extension)
#    kokkos     C++ files with Kokkos. These are intended to be compiled with a
#               vanilla C++ compiler (kitsune without --tapir or --kokkos will
#               work) and linked against a Kokkos installation
#
function (source_language source lang)
  if (source MATCHES ".+[.]kokkos[.]kit[.]cpp$")
    set(${lang} "kitkokkos" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.]c$" OR source MATCHES ".+[.]kit[.]c$")
    set(${lang} "kitc" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.]cpp$" OR source MATCHES ".+[.]kit[.]cc$")
    set(${lang} "kitcxx" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.][Ff]$" OR
      source MATCHES ".+[.]kit[.][Ff]90$" OR
      source MATCHES ".+[.]kit[.][Ff]95$" OR
      source MATCHES ".+[.]kit[.][Ff]03$" OR
      source MATCHES ".+[.]kit[.][Ff]08$")
    set(${lang} "kitfort" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kokkos[.]cpp$" OR
      source MATCHES ".+[.]kokkos[.]cc$")
    set(${lang} "kokkos" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]c$")
    set(${lang} "c" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]cpp$" OR source MATCHES ".+[.]cc$")
    set(${lang} "cxx" PARENT_SCOPE)
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

# Register the target as a test.
#
#     target         The cmake target
#     tapir_target   The tapir target. A value of "none" is a special case. It
#                    will be treated as "not to be built with any tapir target"
#     cmdargs        A list of command line arguments to be passed when running
#                    the test
#
function (register_test target tapir_target cmdargs)
  llvm_test_executable_no_test(${target} ${source})
  llvm_test_run(WORKDIR "%S" "${cmdargs}")

  # timeit adds --append-exitstatus to the test output. We expect that the tests
  # will perform their own verification and return 0 on success, non-zero on
  # failure. Since we don't support Windows, we should have grep.
  llvm_test_verify(${GREP} -E "\"^exit 0$\"" %o)
  llvm_add_test_for_target(${target})

  # The include directory in Kitsune/ contains headers for timings and, perhaps,
  # other things. The timing is only really needed for the benchmarks, but we
  # might as well tell the compiler to always look in that directory. It is
  # unlikely that anything there will collide with something that is used by the
  # tests.
  target_include_directories(${target} PUBLIC
    ${CMAKE_SOURCE_DIR}/Kitsune/include)

  if (NOT tapir_target STREQUAL "none")
    set(tapir_flags "--tapir=${tapir_target}")
    if (tapir_target STREQUAL "cuda" AND NOT KITSUNE_CUDA_ARCH STREQUAL "")
      list(APPEND tapir_flags "--tapir-cuda-arch=${KITSUNE_CUDA_ARCH}")
    endif ()
    if (tapir_target STREQUAL "hip" AND NOT KITSUNE_HIP_ARCH STREQUAL "")
      list(APPEND tapir_flags "--tapir-hip-arch=${KITSUNE_HIP_ARCH}")
    endif ()

    # We need to set the tapir flags on the link options, otherwise the runtime
    # libraries (kitrt, opencilk etc.) will not be linked in correctly.
    target_compile_options(${target} BEFORE PUBLIC "${tapir_flags}")
    target_link_options(${target} BEFORE PUBLIC "${tapir_flags}" "${KITSUNE_LINKER_FLAGS}")
  endif ()
endfunction ()

# "Copy" the data files to the current build directory. For now, this does not
# actually copy, but simply creates a symlink to the files which are all assumed
# to be in the current source directory.
function (setup_data data)
  # These should really be a POST_BUILD command for some target, but it's not
  # clear that it's worth the trouble since each source file could match to a
  # number of cmake targets depending on which tapir targets have been enabled.
  # There is also no single target that is guaranteed to always be built. So
  # just do this at configure time instead.
  foreach (file IN LISTS data)
    file(CREATE_LINK
      ${CMAKE_CURRENT_SOURCE_DIR}/${file}
      ${CMAKE_CURRENT_BINARY_DIR}/${file}
      SYMBOLIC)
  endforeach ()
endfunction()

# Setup a single-source test for the given tapir target.
#
#     source         The absolute path to the source file
#     lang           The source language
#     tapir_target   The tapir target. A value of "none" is a special case. It
#                    will be treated as "not to be built with any tapir target"
#     cmdargs        A list of command line arguments to be passed when running
#                    the test
#
function (kit_singlesource_test source lang tapir_target cmdargs)
  get_filename_component(base "${source}" NAME_WE)
  if (tapir_target STREQUAL "none")
    if (lang STREQUAL "cuda")
      set(target ${base}-nvcc)
    elseif (lang STREQUAL "hip")
      set(target ${base}-hipcc)
    elseif (lang STREQUAL "kokkos-nvidia" OR
        lang STREQUAL "kokkos-amd")
      set(target ${base}-${lang})
    else ()
      message(FATAL_ERROR "Unsupported language '${lang}' for '${source}'")
    endif ()
  elseif (lang STREQUAL "kitkokkos")
    set(target "${base}-kokkos-${tapir_target}")
  elseif (lang STREQUAL "kitc")
    set(target "${base}-c-${tapir_target}")
  elseif (lang STREQUAL "kitcxx")
    set(target "${base}-cxx-${tapir_target}")
  elseif (lang STREQUAL "kitfort")
    set(target "${base}-fortran-${tapir_target}")
  else ()
    message(FATAL_ERROR "Language '${lang}' not handled for '${source}'")
  endif ()

  register_test("${target}" "${tapir_target}" "${cmdargs}")

  # If this is a target that is compiled with --tapir=, set the additional
  # flags that were provided at configure time.
  if (lang STREQUAL "kitc")
    target_compile_options(${target} PUBLIC "${KITSUNE_C_FLAGS}")
  elseif (lang STREQUAL "kitcxx" OR lang STREQUAL "kitkokkos")
    target_compile_options(${target} PUBLIC ${KITSUNE_CXX_FLAGS})
  elseif (lang STREQUAL "kitfort")
    target_compile_options(${target} PUBLIC "${KITSUNE_Fortran_FLAGS}")
  endif ()

  # Since we do not support cross compiling, or portability across GPUs, just
  # compile the vanilla cuda code for the current GPU. If this is not done, it
  # will try to JIT the code which we don't want because it becomes a less fair
  # comparison.
  if (lang STREQUAL "cuda" AND tapir_target STREQUAL "none")
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "native")
  endif ()

  # cmake uses the compiler when linking. By passing -fkokkos to it, we ensure
  # that Kitsune correctly links the Kokkos libraries.
  if (lang STREQUAL "kitkokkos")
    target_compile_options(${target} BEFORE PUBLIC -fkokkos -fkokkos-no-init)
    target_link_options(${target} BEFORE PUBLIC -fkokkos)
  endif ()

  if (lang STREQUAL "kokkos-nvidia")
    target_include_directories(${target} BEFORE PUBLIC
      ${KOKKOS_CUDA_PREFIX}/include)
    target_link_libraries(${target} PUBLIC
      ${KOKKOS_CUDA_PREFIX}/lib/libkokkoscore.a)
  elseif (lang STREQUAL "kokkos-amd")
    set_source_files_properties(${source} PROPERTIES
      LANGUAGE HIP)
    target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__)
    target_include_directories(${target} BEFORE PUBLIC
      /opt/rocm/include
      ${KOKKOS_HIP_PREFIX}/include)
    target_link_libraries(${target} PUBLIC ${KOKKOS_HIP_PREFIX}/lib/libkokkoscore.a)
  endif ()
endfunction()

# Add tests for all tapir targets being tested. This should not be used by
# consumers. They should call kitsune_singlesource() instead.
#
# ARGUMENTS
#
#     source     The absolute path to the source file
#     lang       The source language
#     cmdargs    A list of command line arguments to be passed when running the
#                test
#     data       A list of files containing that will be used by the test. A
#                symlink to these will be created in the build directory. In
#                general, the LLVM test suite is intended to be built without
#                any dependence on the source directory, but we don't really
#                care about this.
#
function(kit_singlesource_all_targets source lang cmdargs data)
  if (TEST_CUDA_TARGET)
    kit_singlesource_test(${source} ${lang} "cuda" "${cmdargs}" "${data}")
  endif ()
  if (TEST_HIP_TARGET)
    kit_singlesource_test(${source} ${lang} "hip" "${cmdargs}" "${data}")
  endif()
  if (TEST_LAMBDA_TARGET)
    kit_singlesource_test(${source} ${lang} "lambda" "${cmdargs}" "${data}")
  endif ()
  if (TEST_OMPTASK_TARGET)
    kit_singlesource_test(${source} ${lang} "omptask" "${cmdargs}" "${data}")
  endif ()
  if (TEST_OPENCILK_TARGET)
    kit_singlesource_test(${source} ${lang} "opencilk" "${cmdargs}" "${data}")
  endif()
  if (TEST_OPENMP_TARGET)
    kit_singlesource_test(${source} ${lang} "openmp" "${cmdargs}" "${data}")
  endif ()
  if (TEST_QTHREADS_TARGET)
    kit_singlesource_test(${source} ${lang} "qthreads" "${cmdargs}" "${data}")
  endif ()
  if (TEST_REALM_TARGET)
    kit_singlesource_test(${source} ${lang} "realm" "${cmdargs}" "${data}")
  endif ()
  if (TEST_SERIAL_TARGET)
    kit_singlesource_test(${source} ${lang} "serial" "${cmdargs}" "${data}")
  endif ()
  setup_data("${data}")
endfunction()

# Configure the current directory as a SingleSource subdirectory - i.e. every
# C/C++/Fortran file is treated as its own test.
#
# ARGUMENTS
#
#     <none>
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    The list of command line arguments to be passed when running
#                the test
#
#     DATA       A list of files containing data that will be used by the
#                test. These will be copied to the build directory
#
function(kitsune_singlesource)
  cmake_parse_arguments(KIT "" "" "CMDARGS;DATA" ${ARGN})
  set(cmdargs "${KIT_CMDARGS}")
  set(data "${KIT_DATA}")

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
      if (TEST_VANILLA_CUDA)
        kit_singlesource_test(${source} ${lang} "none" "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "hip")
      if (TEST_VANILLA_HIP)
        kit_singlesource_test(${source} ${lang} "none" "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "kokkos")
      if (TEST_VANILLA_KOKKOS_CUDA)
        kit_singlesource_test(${source} "${lang}-nvidia" "none" "${cmdargs}" "${data}")
      endif ()
      if (TEST_VANILLA_KOKKOS_HIP)
        kit_singlesource_test(${source} "${lang}-amd" "none" "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "kitc")
      if (TEST_C)
        kit_singlesource_all_targets(${source} ${lang} "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "kitcxx")
      if (TEST_CXX)
        kit_singlesource_all_targets(${source} ${lang} "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "fortran")
      # Fortran is not yet supported. Complain loudly so we know to change this
      # and take a closer look at everything when Fortran is supported.
      message(FATAL_ERROR "Kitsune does not yet support Fortran: ${source}")
      if (TEST_Fortran)
        kit_singlesource_all_targets(${source} ${lang} "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "kitkokkos")
      if (TEST_KOKKOS_MODE)
        # Kokkos is only tested with the GPU-centric tapir targets because those
        # are the only ones that we really care about as far as Kokkos support
        # goes.
        if (TEST_CUDA_TARGET)
          kit_singlesource_test(${source} ${lang} "cuda" "${cmdargs}" "${data}")
        endif ()
        if (TEST_HIP_TARGET)
          kit_singlesource_test(${source} ${lang} "hip" "${cmdargs}" "${data}")
        endif ()
      endif ()
    else ()
      message(FATAL_ERROR
        "Unsupported source file '${source}'. Detected language '${lang}'. "
        "See Kitsune/README.md for source file name requirements")
    endif ()
  endforeach()
endfunction()

# Create a target with the given name. This should never be called directly. It
# is only intended to be used by kitsune_multisource(). This will add the
# tapir target flag to the compile and link options of the created target. If
# the KOKKOS option argument is provided, it will also add the correct kokkos
# flags to the target.
#
# ARGUMENTS
#
#     base          The base name of the target
#     type          Must be one of "EXECUTABLE", "SHARED", or "STATIC"
#     tapir_target  The tapir target. This must be a target that is a valid
#                   argument of the --tapir flag
#     kokkos        If ON, kokkos mode should be enabled on the target
#     cmdargs       A list of command line arguments to be passed when running
#                   the test
#     data          A list of files containing that will be used by the test.
#                   These will be copied into the build directory.
#
# RETURN
#
#     out           The list to which to append the target that was created
#
macro(make_target base type tapir_target kokkos out)
  set(target)
  if (kokkos)
    set(target "${base}-kokkos-${tapir_target}")
  else ()
    set(target "${base}-unknown-${tapir_target}")
  endif ()

  if (type STREQUAL "EXECUTABLE")
    # The executable target will be created by register_test().
    register_test("${target}" "${tapir_target}" "${cmdargs}" "${data}")
  elseif (type STREQUAL "SHARED")
    add_library(${target} SHARED)
  elseif (type STREQUAL "STATIC")
    add_library(${target} STATIC)
  endif ()

  target_include_directories(${target} PUBLIC
    ${CMAKE_SOURCE_DIR}/Kitsune/include)

  set(tapir_flags "--tapir=${tapir_target}")
  if (${tapir_target} STREQUAL "cuda" AND NOT KITSUNE_CUDA_ARCH STREQUAL "")
    list(APPEND tapir_flags "--tapir-cuda-arch=${KITSUNE_CUDA_ARCH}")
  endif ()
  if (${tapir_target} STREQUAL "hip" AND NOT KITSUNE_HIP_ARCH STREQUAL "")
    list(APPEND tapir_flags "--tapir-hip-arch=${KITSUNE_HIP_ARCH}")
  endif ()

  target_compile_options(${target} BEFORE PUBLIC -flto ${tapir_flags})
  target_link_options(${target} PUBLIC -flto ${tapir_flags} ${KITSUNE_LINKER_FLAGS})

  if (kokkos)
    target_compile_options(${target} BEFORE PUBLIC -fkokkos -fkokkos-no-init)
    target_link_options(${target} BEFORE PUBLIC -fkokkkos)
  endif ()

  list(APPEND ${out} ${target})
endmacro ()

# Setup executables for a multi-source test. Unlike the single-source tests, we
# will never try to be clever and determine exactly how to build such tests.
# They could be arbitrarily complex (since we could potentially dump a sizable
# application there). Instead, we expect a that a CMakeLists.txt file is
# provided with each multi-source test that does most of the work. However, we
# do need to provide different flags depending on the tapir targets and other
# other frontend options (such as Kokkos mode) that we wish to test.
#
# This function will take a base name and using that, create a cmake target for
# each tapir target being tested. The type of the cmake target must be provided
# and should be one of EXECUTABLE, STATIC and SHARED. The first of these creates
# and executable target while the remaining create a static and shared library
# respectively. This function will add the some compiler and linker options to
# the target. A list of these targets will be returned.
#
# It is expected that the CMakeLists.txt file for the multi-source test will
# loop over the returned targets. This is not ideal since the loop will be
# repeated in each multi-source test, but it'll have to do for now.
#
# If the KOKKOS option argument is provided, only the GPU-centric tapir targets
# (if any are enabled) will be created.
#
# ARGUMENTS
#
#     base       The base name of the target to be created
#
# OPTION ARGUMENTS
#
#     <type>     Exactly one of EXECUTABLE, SHARED, or STATIC is required
#
#     KOKKOS     If provided, indicates that the multi-source test contains
#                Kokkos constructs that should be processed with Kitsune's
#                Kokkos-mode (if enabled).
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    The list of command line arguments to be passed when running
#                the test
#
#     DATA       A list of files containing data that will be used by the
#                test. These will be copied to the build directory
#
# OUTPUT
#
#     out        The list of targets that were created
#
function (kitsune_multisource base out)
  cmake_parse_arguments(KIT "EXECUTABLE;SHARED;STATIC;KOKKOS" "" "CMDARGS;DATA" ${ARGN})
  set(cmdargs "${KIT_CMDARGS}")
  set(data "${KIT_DATA}")

  set(type)
  if (KIT_EXECUTABLE)
    set(type "EXECUTABLE")
  elseif (KIT_SHARED)
    set(type "SHARED")
  elseif (KIT_STATIC)
    set(type "STATIC")
  else ()
    message(FATAL_ERROR "Unknown multisource target type")
  endif ()

  set(targets)
  if (KIT_KOKKOS)
    if (TEST_CUDA_TARGET)
      make_target(${base} ${type} "cuda" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_HIP_TARGET)
      make_target(${base} ${type} "hip" ${KIT_KOKKOS} targets)
    endif()
  else ()
    if (TEST_CUDA_TARGET)
      make_target(${base} ${type} "cuda" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_HIP_TARGET)
      make_target(${base} ${type} "hip" ${KIT_KOKKOS} targets)
    endif()
    if (TEST_LAMBDA_TARGET)
      make_target(${base} ${type} "lambda" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_OMPTASK_TARGET)
      make_target(${base} ${type} "omptask" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_OPENCILK_TARGET)
      make_target(${base} ${type} "opencilk" ${KIT_KOKKOS} targets)
    endif()
    if (TEST_OPENMP_TARGET)
      make_target(${base} ${type} "openmp" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_QTHREADS_TARGET)
      make_target(${base} ${type} "qthreads" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_REALM_TARGET)
      make_target(${base} ${type} "realm" ${KIT_KOKKOS} targets)
    endif ()
    if (TEST_SERIAL_TARGET)
      make_target(${base} ${type} "serial" ${KIT_KOKKOS} targets)
    endif ()
  endif ()

  setup_data("${data}")
  set(${out} ${targets} PARENT_SCOPE)
endfunction ()
