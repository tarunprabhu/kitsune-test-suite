include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # samples    16
  # width      320
  # height     200
  set(base "raytracer.small")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j 16 320 200
    DATA ${refc} ${refg})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # samples    128
  # width      1280
  # height     1024
  set(base "raytracer.medium")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j 16 320 200
    DATA ${refc} ${refg})
else()
  kitsune_singlesource()
endif()
