include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_BENCHMARK)
  # samples    128
  # width      1280
  # height     1024
  kitsune_singlesource(
    CMDARGS -j 128 1280 1024)
else()
  # samples    8
  # width      160
  # height     100
  set(base "raytracer.ref")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -c ${refc} -g ${refg} 8 160 100
    DATA ${refc} ${refg})
endif()
