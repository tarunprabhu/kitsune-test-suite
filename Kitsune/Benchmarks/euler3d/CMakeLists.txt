include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(input fvcorr.domn.193K)
if (KITSUNE_BENCHMARK)
  # domain      fvcorr.domn.193k
  # iterations  4000
  kitsune_singlesource(
    CMDARGS -j ${input} 4000
    DATA ${input})
else()
  # domain      fvcorr.domn.193k
  # iterations  50
  set(base euler3d.ref)
  set(refc "${base}.cpu.dat")
  set(refg "${base}.gpu.dat")
  kitsune_singlesource(
    CMDARGS -c ${refc} -g ${refg} ${input} 50
    DATA ${input} ${refc} ${refg})
endif()
