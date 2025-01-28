include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # check-file = srad.small.dat
  # rows       = 640
  # cols       = 640
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  # iterations = 200
  set(check "srad.small.dat")
  kitsune_singlesource(
    CMDARGS ${check} 640 640 0 127 0 127 0.5 200
    DATA ${check})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # check-file = srad.medium.dat
  # rows       = 6400
  # cols       = 6400
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  # iterations = 2000
  set(check "srad.medium.dat")
  kitsune_singlesource(
    CMDARGS ${check} 6400 6400 0 127 0 127 0.5 2000
    DATA ${check})
else()
  kitsune_singlesource()
endif()
