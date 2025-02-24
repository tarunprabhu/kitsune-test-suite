include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # iterations = 200
  # rows       = 640
  # cols       = 640
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  set(base "srad.small")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j -c ${refc} -g ${refg} 200 640 640 0 127 0 127 0.5
    DATA ${refc} ${refg})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # iterations = 2000
  # rows       = 6400
  # cols       = 6400
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  set(base "srad.medium")
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j -c ${refc} -g ${refg} 2000 6400 6400 0 127 0 127 0.5
    DATA ${refc} ${refg})
else()
  kitsune_singlesource()
endif()
