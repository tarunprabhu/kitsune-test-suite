include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# domain      fvcorr.domn.193k
# iterations  50

set(input fvcorr.domn.193K)
set(ref "euler3d.ref.dat")
kitsune_singlesource(
    CMDARGS -c ${ref} ${input} 50
    DATA ${input} ${ref})
