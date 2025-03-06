include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_BENCHMARK)
  # iterations = 2000
  # rows       = 6400
  # cols       = 6400
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  kitsune_singlesource(
    CMDARGS -j 2000 6400 6400 0 127 0 127 0.5)
else()
  # iterations = 100
  # rows       = 640
  # cols       = 640
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  set(base "srad.ref")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -c ${refc} -g ${refg} 100 640 640 0 127 0 127 0.5
    DATA ${refc} ${refg})
endif()
