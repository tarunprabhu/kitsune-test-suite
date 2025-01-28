include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # samples    64
  # width      320
  # height     200
  # checkFile  raytracer.small.ppm
  set(check raytracer.small.ppm)
  kitsune_singlesource(CMDARGS 16 320 200 ${check} DATA ${check})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # samples    128
  # width      1280
  # height     1024
  # checkFile  raytracer.medium.ppm
  set(check raytracer.medium.ppm)
  kitsune_singlesource(CMDARGS 128 1280 1024 ${check} DATA ${check})
else()
  kitsune_singlesource()
endif()
