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
#
#    kitcxx     C++ files with kitsune-specific extensions
#
#    kitfort    Fortran files with kitsune-specific extensions and/or language
#               support. For instance, if it is a Fortran source file where
#               the DO CONCURRENT construct is intended to be lowered via the
#               tapir dialect instead of the standard/OpenMP dialects
#
#    kitkokkos  C++ files with Kokkos. These may or may not contain any
#               Kitsune-specific extensions, but these are intended to be
#               compiled by Kitsune with --kokkos
#
#    c          C files without kitsune-specific extensions
#
#    cxx        C++ files without kitsune-specific extensions
#
#    fortran    Fortran files without any special handling
#
#    cuda       Cuda files (those with a .cu extension)
#
#    hip        Hip files (those with a .hip extension)
#
#    kokkos     C++ files with Kokkos. These are intended to be compiled with a
#               vanilla C++ compiler (kitsune without --tapir or --kokkos will
#               work) and linked against a Kokkos installation
#
function (source_language SOURCE LANG)
  if ("${SOURCE}" MATCHES ".+[.]kokkos[.]kit[.]cpp$")
    set(${LANG} "kitkokkos" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]kit[.]c$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.]c$")
    set(${LANG} "kitc" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]kit[.]cpp$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.]cc$")
    set(${LANG} "kitcxx" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]kit[.][Ff]$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.][Ff][Pp][Pp]$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.][Ff]90$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.][Ff]95$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.][Ff]03$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.][Ff]08$" OR
      "${SOURCE}" MATCHES ".+[.]kit[.][Ff]18$")
    set(${LANG} "kitfort" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]kokkos[.]cpp$" OR
      "${SOURCE}" MATCHES ".+[.]kokkos[.]cc$")
    set(${LANG} "kokkos" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]c$")
    set(${LANG} "c" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]cpp$" OR
      "${SOURCE}" MATCHES ".+[.]cc$")
    set(${LANG} "cxx" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.][Ff]$" OR
      "${SOURCE}" MATCHES ".+[.][Ff][Pp][Pp]$" OR
      "${SOURCE}" MATCHES ".+[.][Ff]90$" OR
      "${SOURCE}" MATCHES ".+[.][Ff]95$" OR
      "${SOURCE}" MATCHES ".+[.][Ff]03$" OR
      "${SOURCE}" MATCHES ".+[.][Ff]08$" OR
      "${SOURCE}" MATCHES ".+[.][Ff]18$")
    set(${LANG} "fortran" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]cu$")
    set(${LANG} "cuda" PARENT_SCOPE)
  elseif ("${SOURCE}" MATCHES ".+[.]hip$")
    set(${LANG} "hip" PARENT_SCOPE)
  else ()
    message(FATAL_ERROR "Cannot determine source language for '${SOURCE}'")
  endif ()
endfunction ()

# "Copy" the data files to the current build directory. For now, this does not
# actually copy, but simply creates a symlink to the files which are all assumed
# to be in the current source directory.
function (setup_data DATA)
  # These should really be a POST_BUILD command for some target, but it's not
  # clear that it's worth the trouble since each source file could match to a
  # number of cmake targets depending on which tapir targets have been enabled.
  # There is also no single target that is guaranteed to always be built. So
  # just do this at configure time instead.
  foreach (FILE ${DATA})
    file(CREATE_LINK
      ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
      ${CMAKE_CURRENT_BINARY_DIR}/${FILE}
      SYMBOLIC)
  endforeach ()
endfunction()

# Register the target as a test.
#
#     TARGET     The cmake target
#
#     TT         The tapir target. A value of "none" is a special case. It will
#                be treated as "not to be built with any tapir target"
#
#     CMDARGS    A list of command line arguments to be passed when running the
#                test
#
function (register_test TARGET TT CMDARGS)
  llvm_test_executable_no_test(${TARGET})

  if ("${TT}" STREQUAL "pthreads")
    # FIXME: This only works on shells that allow environment variables to be
    # set on the same line as the executable. But this is a temporary workaround
    # anyway. It is only here because nested tapir loops are painfully slow with
    # the pthreads tapir target.
    llvm_test_run(WORKDIR "%S"
      EXECUTABLE "KIT_NUM_THREADS=2 %S/${TARGET}"
      "${CMDARGS}")
  else ()
    llvm_test_run(WORKDIR "%S" "${CMDARGS}")
  endif ()

  # timeit adds --append-exitstatus to the test output. We expect that the tests
  # will perform their own verification and return 0 on success, non-zero on
  # failure. Since we don't support Windows, we should have grep.
  llvm_test_verify(${GREP} -E "\"^exit 0$\"" %o)
  llvm_add_test_for_target(${TARGET})

  # The include directory in Kitsune/ contains headers for timings and, perhaps,
  # other things. The timing is only really needed for the benchmarks, but we
  # might as well tell the compiler to always look in that directory. It is
  # unlikely that anything there will collide with something that is used by the
  # tests.
  target_include_directories(${TARGET} PUBLIC
    ${CMAKE_SOURCE_DIR}/Kitsune/include)

  if (NOT "${TT}" STREQUAL "none")
    set(TTOPTS "--tapir=${TT}")

    # We need to set the tapir flags on the link options, otherwise the runtime
    # libraries (kitrt, opencilk etc.) will not be linked in correctly.
    target_compile_options(${TARGET} BEFORE PUBLIC "${TTOPTS}")
    target_link_options(${TARGET} BEFORE PUBLIC
      "${TTOPTS}" "${KITSUNE_LINKER_FLAGS}")
  endif ()
endfunction ()

# Setup a single-source test for the given tapir target.
#
#     SOURCE     The absolute path to the source file
#
#     LANG       The source language
#
#     TT         The tapir target. A value of "none" is a special case. It will
#                be treated as "not to be built with any tapir target"
#
#     CMDARGS    A list of command line arguments to be passed when running the
#                test
#
function (kit_singlesource_test SOURCE LANG TT CMDARGS)
  get_filename_component(BASE "${SOURCE}" NAME_WE)
  if ("${TT}" STREQUAL "none")
    if ("${LANG}" STREQUAL "cuda")
      set(TARGET "${BASE}-nvcc")
    elseif ("${LANG}" STREQUAL "hip")
      set(TARGET "${BASE}-hipcc")
    elseif ("${LANG}" STREQUAL "kokkos-nvidia" OR
        "${LANG}" STREQUAL "kokkos-amd")
      set(TARGET ${BASE}-${LANG})
    elseif ("${LANG}" STREQUAL "c")
      set(TARGET "${BASE}-${LANG}-std")
    elseif ("${LANG}" STREQUAL "cxx")
      set(TARGET "${BASE}-${LANG}-std")
    elseif ("${LANG}" STREQUAL "fortran")
      set(TARGET "${BASE}-${LANG}-std")
    else ()
      message(FATAL_ERROR "Unsupported language '${LANG}' for '${SOURCE}'")
    endif ()
  elseif ("${LANG}" STREQUAL "kitkokkos")
    set(TARGET "${BASE}-kokkos-${TT}")
  elseif ("${LANG}" STREQUAL "kitc")
    set(TARGET "${BASE}-c-${TT}")
  elseif ("${LANG}" STREQUAL "kitcxx")
    set(TARGET "${BASE}-cxx-${TT}")
  elseif ("${LANG}" STREQUAL "kitfort")
    set(TARGET "${BASE}-fortran-${TT}")
  else ()
    message(FATAL_ERROR "Language '${LANG}' not handled for '${SOURCE}'")
  endif ()

  register_test("${TARGET}" "${TT}" "${CMDARGS}")

  # The target will have been created in register_test, but without any sources.
  target_sources(${TARGET} PUBLIC ${SOURCE})

  # If this is a target that is compiled with --tapir=, set the additional
  # flags that were provided at configure time.
  if ("${LANG}" STREQUAL "kitc")
    target_compile_options(${TARGET} PUBLIC "${KITSUNE_C_FLAGS}")
  elseif ("${LANG}" STREQUAL "kitcxx" OR "${LANG}" STREQUAL "kitkokkos")
    target_compile_options(${TARGET} PUBLIC ${KITSUNE_CXX_FLAGS})
  elseif ("${LANG}" STREQUAL "kitfort")
    target_compile_options(${TARGET} PUBLIC "${KITSUNE_Fortran_FLAGS}")
  endif ()

  # Since we do not support cross compiling, or portability across GPUs, just
  # compile the vanilla cuda code for the current GPU. If this is not done, it
  # will try to JIT the code which we don't want because it becomes a less fair
  # comparison.
  if ("${TT}" STREQUAL "none")
    if ("${LANG}" STREQUAL "cuda" OR
        "${LANG}" STREQUAL "kokkos-nvidia")
      set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "native")
    endif ()
  endif ()

  # cmake uses the compiler when linking. By passing -fkokkos to it, we ensure
  # that Kitsune correctly links the Kokkos libraries.
  if ("${LANG}" STREQUAL "kitkokkos")
    target_compile_options(${TARGET} BEFORE PUBLIC -fkokkos -fkokkos-no-init)
    target_link_options(${TARGET} BEFORE PUBLIC -fkokkos)
  endif ()

  if ("${LANG}" STREQUAL "kokkos-nvidia")
    # We need to force the language of the file to be "cuda" because the Kokkos
    # headers expect that a cuda compiler is being used.
    set_source_files_properties(${SOURCE} PROPERTIES
      LANGUAGE CUDA)

    target_compile_options(${TARGET} BEFORE PUBLIC --extended-lambda)

    target_include_directories(${TARGET} BEFORE PUBLIC
      ${KOKKOS_CUDA_PREFIX}/include)

    # In addition to Kokkos, we also need to link in cuda because clang won't
    # do that automatically. I am not sure why.
    target_link_libraries(${TARGET} PUBLIC
      ${LIBKOKKOSCORE_CUDA} ${LIBCUDART} ${LIBCUDA})
  elseif ("${LANG}" STREQUAL "kokkos-amd")
    # We need to force the language of the file to be "hip" because the Kokkos
    # headers expect that a cuda compiler is being used.
    set_source_files_properties(${SOURCE} PROPERTIES
      LANGUAGE HIP)

    target_compile_definitions(${TARGET} PUBLIC __HIP_PLATFORM_AMD__)

    target_include_directories(${TARGET} BEFORE PUBLIC
      ${KOKKOS_HIP_PREFIX}/include)

    target_link_libraries(${TARGET} PUBLIC
      ${LIBKOKKOSCORE_HIP})
  endif ()
endfunction()

# Add tests for all tapir targets being tested. This should not be used by
# consumers. They should call kitsune_singlesource() instead.
#
# ARGUMENTS
#
#     SOURCE     The absolute path to the source file
#
#     LANG       The source language
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    A list of command line arguments to be passed to the test
#
#     EXCLUDE    The list of tapir targets to exclude i.e. the test(s) will not
#                be compiled with the tapir targets in the list
#
#     ONLY       The list of tapir targets used to compile the test(s). If a
#                tapir target is in the list but has not been enabled, it will
#                be ignored
#
function(kit_singlesource_all_targets SOURCE LANG)
  cmake_parse_arguments(KIT "" "" "CMDARGS;EXCLUDE;ONLY" ${ARGN})

  if (KIT_ONLY)
    foreach (TT ${KIT_ONLY})
      if (TEST_${TT}_TARGET)
        kit_singlesource_test(${SOURCE} ${LANG} ${TT} "${KIT_CMDARGS}")
      endif ()
    endforeach ()
  else ()
    # We skip the custom tapir target because we do not have a "default" custom
    # tapir target to test with. At some point, it may be good to have one so
    # we can be sure that it works end-to-end. For now, we have to rely on the
    # core tests to test enough that we can be reasonably certain that it works.
    foreach (TT ${TEST_TAPIR_TARGETS})
      if (TEST_${TT}_TARGET AND
          NOT "${TT}" STREQUAL "custom" AND
          NOT "${TT}" IN_LIST KIT_EXCLUDE)
        kit_singlesource_test(${SOURCE} ${LANG} ${TT} "${KIT_CMDARGS}")
      endif ()
    endforeach ()
  endif ()
endfunction()

# Configure the current directory as a single-source directory - i.e. every
# C/C++/Fortran file is treated as its own test. Depending on the source
# language of the file, it may be compiled with all tapir targets that have been
# enabled. The EXCLUDE and ONLY keyword arguments can be used to control which
# tapir targets are used to compile the tests.
#
# ARGUMENTS
#
#     <none>
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    The list of command line arguments to be passed when running
#                the test(s)
#
#     DATA       A list of files containing data that will be used by the
#                test(s). These will be copied to the build directory
#
#     EXCLUDE    The list of tapir targets to exclude i.e. the test(s) will not
#                be compiled with the tapir targets in the list
#
#     ONLY       The list of tapir targets used to compile the test(s). If a
#                tapir target is in the list but has not been enabled, it will
#                be ignored
#
function(kitsune_singlesource)
  cmake_parse_arguments(KIT "" "" "CMDARGS;DATA;EXCLUDE;ONLY" ${ARGN})

  if (KIT_EXCLUDE AND KIT_ONLY)
    message(FATAL_ERROR "Only one of EXCLUDE and ONLY may be provided to kitsune_singlesource()")
  endif ()

  # If any input files are required, set them up.
  setup_data("${KIT_DATA}")

  file(GLOB SOURCES
    *.c
    *.cpp *.cc
    *.cu
    *.f *.F *.f90 *.F90 *.f03 *.F03 *.f08 *.F08 *.fpp *.FPP
    *.hip)
  foreach(SOURCE ${SOURCES})
    set(LANG)
    source_language(${SOURCE} LANG)

    if ("${LANG}" STREQUAL "cuda")
      if (TEST_VANILLA_cuda)
        kit_singlesource_test("${SOURCE}" ${LANG} none "${KIT_CMDARGS}")
      endif ()
    elseif ("${LANG}" STREQUAL "hip")
      if (TEST_VANILLA_hip)
        kit_singlesource_test("${SOURCE}" ${LANG} none "${KIT_CMDARGS}")
      endif ()
    elseif ("${LANG}" STREQUAL "kokkos")
      if (TEST_VANILLA_KOKKOS_cuda)
        kit_singlesource_test("${SOURCE}" "${LANG}-nvidia" none "${KIT_CMDARGS}")
      endif ()
      if (TEST_VANILLA_KOKKOS_hip)
        kit_singlesource_test("${SOURCE}" "${LANG}-amd" none "${KIT_CMDARGS}")
      endif ()
    elseif ("${LANG}" STREQUAL "kitc")
      if (TEST_C)
        kit_singlesource_all_targets(${SOURCE} ${LANG}
          CMDARGS "${KIT_CMDARGS}"
          EXCLUDE "${KIT_EXCLUDE}"
          ONLY "${KIT_ONLY}")
      endif ()
    elseif ("${LANG}" STREQUAL "kitcxx")
      if (TEST_CXX)
        kit_singlesource_all_targets("${SOURCE}" ${LANG}
          CMDARGS "${KIT_CMDARGS}"
          EXCLUDE "${KIT_EXCLUDE}"
          ONLY "${KIT_ONLY}")
      endif ()
    elseif ("${LANG}" STREQUAL "kitfort")
      # Fortran is not yet supported. Complain loudly so we know to change this
      # and take a closer look at everything when Fortran is supported.
      message(FATAL_ERROR "Kitsune does not yet support Fortran: ${SOURCE}")
      if (TEST_Fortran)
        kit_singlesource_all_targets("${SOURCE}" ${LANG}
          CMDARGS "${KIT_CMDARGS}"
          EXCLUDE "${KIT_EXCLUDE}"
          ONLY "${KIT_ONLY}")
      endif ()
    elseif ("${LANG}" STREQUAL "kitkokkos")
      if (TEST_KOKKOS_MODE)
        # Kokkos is only tested with the GPU-centric tapir targets because those
        # are the only ones that we really care about as far as Kokkos support
        # goes. But, we allow overriding it if needed
        set(TTS cuda hip)
        if (KIT_ONLY)
          set(TTS ${KIT_ONLY})
        endif ()
        foreach (TT IN LISTS TTS)
          if (TEST_${TT}_TARGET)
            kit_singlesource_test("${SOURCE}" ${LANG} ${TT} "${KIT_CMDARGS}")
          endif ()
        endforeach ()
      endif ()
    elseif ("${LANG}" STREQUAL "c")
      if (KITSUNE_BUILD_VANILLA_C)
        kit_singlesource_test("${SOURCE}" ${LANG} none "${KIT_CMDARGS}")
      endif ()
    elseif ("${LANG}" STREQUAL "cxx")
      if (KITSUNE_BUILD_VANILLA_CXX)
        kit_singlesource_test("${SOURCE}" ${LANG} none "${KIT_CMDARGS}")
      endif ()
    elseif ("${LANG}" STREQUAL "fortran")
      if (KITSUNE_BUILD_VANILLA_fortran)
        kit_singlesource_test("${SOURCE}" ${LANG} none "${KIT_CMDARGS}")
      endif ()
    else ()
      message(FATAL_ERROR
        "Unsupported source file '${SOURCE}'. Detected language '${LANG}'. "
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
#     BASE       The base name of the target
#
#     TYPE       Must be one of "EXECUTABLE", "SHARED", or "STATIC"
#
#     TT         The tapir target. This must be a valid value for the --tapir
#                option
#
#     KOKKOS     If ON, kokkos mode should be enabled on the target
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    A list of command line arguments to be passed when running the
#                test
#
# RETURN
#
#     OUT        The list to which to append the target that was created
#
function(make_target BASE TYPE TT KOKKOS OUT)
  cmake_parse_arguments(KIT "" "" "CMDARGS" ${ARGN})

  set(TARGET)
  if (${KOKKOS})
    set(TARGET "${BASE}-kitkokkos-${TT}")
  else ()
    set(TARGET "${BASE}-kitlang-${TT}")
  endif ()

  if ("${TYPE}" STREQUAL "EXECUTABLE")
    # The executable target will be created by register_test().
    register_test(${TARGET} ${TT} "${KIT_CMDARGS}")
  elseif ("${TYPE}" STREQUAL "SHARED")
    add_library(${TARGET} SHARED)
  elseif ("${TYPE}" STREQUAL "STATIC")
    add_library(${TARGET} STATIC)
  endif ()

  target_include_directories(${TARGET} PUBLIC
    ${CMAKE_SOURCE_DIR}/Kitsune/include)

  set(TTOPTS "--tapir=${TT}")
  target_compile_options(${TARGET} BEFORE PUBLIC -flto ${TTOPTS})
  target_link_options(${TARGET} PUBLIC
    -flto ${TTOPTS} ${KITSUNE_LINKER_FLAGS})

  if (${KOKKOS})
    target_compile_options(${TARGET} BEFORE PUBLIC -fkokkos -fkokkos-no-init)
    target_link_options(${TARGET} BEFORE PUBLIC -fkokkkos)
  endif ()

  set(${OUT} ${TARGET} PARENT_SCOPE)
endfunction ()

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
#     BASE       The base name of the target to be created
#
# OPTION ARGUMENTS
#
#     <type>     Exactly one of EXECUTABLE, SHARED, or STATIC is required
#
#     KOKKOS     If provided, indicates that the multi-source test contains
#                Kokkos constructs that should be processed with Kitsune's
#                Kokkos-mode (if enabled)
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    The list of command line arguments to be passed when running
#                the test
#
#     DATA       A list of files containing data that will be used by the
#                test. These will be copied to the build directory
#
#     EXCLUDE    The list of tapir targets to exclude i.e. the test(s) will not
#                be compiled with the tapir targets in the list
#
#     ONLY       The list of tapir targets used to compile the test(s). If a
#                tapir target is in the list but has not been enabled, it will
#                be ignored
#
# OUTPUT
#
#     OUT        The list of targets that were created
#
function (kitsune_multisource BASE OUT)
  cmake_parse_arguments(KIT
    "EXECUTABLE;SHARED;STATIC;KOKKOS"
    ""
    "CMDARGS;DATA;EXCLUDE;ONLY"
    ${ARGN})

  if (KIT_EXCLUDE AND KIT_ONLY)
    message(FATAL_ERROR "Only one of EXCLUDE and ONLY may be provided to kitsune_multisource()")
  endif ()

  set(TYPE)
  if (KIT_EXECUTABLE)
    set(TYPE "EXECUTABLE")
  elseif (KIT_SHARED)
    set(TYPE "SHARED")
  elseif (KIT_STATIC)
    set(TYPE "STATIC")
  else ()
    message(FATAL_ERROR "Unknown multisource target type")
  endif ()

  set(TARGET)
  if (KIT_ONLY)
    foreach (TT ${KIT_ONLY})
      make_target("${BASE}" ${TYPE} ${TT} ${KIT_KOKKOS} TARGET
        CMDARGS "${KIT_CMDARGS}")
      list(APPEND TARGETS "${TARGET}")
    endforeach ()
  elseif (KIT_KOKKOS)
    foreach (TT IN ITEMS cuda hip)
      if (TEST_${TT}_TARGET AND NOT "${TT}" IN_LIST KIT_EXCLUDE)
        make_target("${BASE}" ${TYPE} ${TT} ${KIT_KOKKOS} TARGET
          CMDARGS "${KIT_CMDARGS}")
        list(APPEND TARGETS "${TARGET}")
      endif ()
    endforeach ()
  else ()
    foreach (TT IN LISTS TEST_TAPIR_TARGETS)
      if (TEST_${TT}_TARGET AND
          NOT "${TT}" STREQUAL "custom" AND
          NOT "${TT}" IN_LIST KIT_EXCLUDE)
        make_target("${BASE}" ${TYPE} ${TT} ${KIT_KOKKOS} TARGET
          CMDARGS "${KIT_CMDARGS}")
        list(APPEND TARGETS "${TARGET}")
      endif ()
    endforeach ()
  endif ()

  setup_data("${KIT_DATA}")
  set(${OUT} ${TARGETS} PARENT_SCOPE)
endfunction ()
