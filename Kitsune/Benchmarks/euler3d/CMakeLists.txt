include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(input fvcorr.domn.193K)
if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # file          fvcorr.domn.193k
  # iterations    100
  kitsune_singlesource(CMDARGS ${input} 100 DATA ${input})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # file          fvcorr.domn.193k
  # iterations    4000
  kitsune_singlesource(CMDARGS ${input} 4000 DATA ${input})
else()
  kitsune_singlesource()
endif()
