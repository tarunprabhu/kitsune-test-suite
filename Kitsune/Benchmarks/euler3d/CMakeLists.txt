include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(input fvcorr.domn.193K)
if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # iterations  100
  # in-file     fvcorr.domn.193k
  # check-file  euler3d.small.dat
  set(check euler3d.small.dat)
  kitsune_singlesource(CMDARGS 100 ${input} ${check} DATA ${input} ${check})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # iterations  4000
  # in-file     fvcorr.domn.193k
  # check-file  euler3d.medium.dat
  kitsune_singlesource(CMDARGS 4000 ${input} ${check} DATA ${input} ${check})
else()
  kitsune_singlesource()
endif()
