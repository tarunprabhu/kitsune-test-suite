include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(input fvcorr.domn.193K)
if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # domain      fvcorr.domn.193k
  # iterations  50
  set(base euler3d.small)
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j -c ${refc} -g ${refg} ${input} 50
    DATA ${input} ${refc} ${refg})
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  # domain      fvcorr.domn.193k
  # iterations  4000
  set(base euler3d.medium)
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -j -c ${refc} -g ${refg} ${input} 4000
    DATA ${input} ${refc} ${refg})
else()
  kitsune_singlesource()
endif()
