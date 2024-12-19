include(KitsuneTestSuite)

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "default")
  # rows       = 6400
  # cols       = 6400
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  # iterations = 2000
  kitsune_singlesource(CMDARGS "6400;6400;0;127;0;127;0.5;2000")
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # rows       = 640
  # cols       = 640
  # y1         = 0
  # y2         = 127
  # x1         = 0
  # x2         = 127
  # lambda     = 0.5
  # iterations = 200
  kitsune_singlesource(CMDARGS "640;640;0;127;0;127;0.5;200")
else()
  kitsune_singlesource()
endif()
