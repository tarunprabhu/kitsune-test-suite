include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(input fvcorr.domn.193K)

# domain      fvcorr.domn.193k
# iterations  4000
kitsune_singlesource(
    CMDARGS -j ${input} 4000
    DATA ${input})
