include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # samples    8
  # width      160
  # height     100
  set(base "raytracer.small")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j 8 160 100
    DATA ${refc} ${refg})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # samples    128
  # width      1280
  # height     1024
  set(base "raytracer.medium")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j 128 1280 1024
    DATA ${refc} ${refg})
else()
  kitsune_singlesource()
endif()
