include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(input fvcorr.domn.193K)
set(ref euler3d.ref.dat)
set(cmdargs ${input} 20)

# Don't test this with pthreads because it is even slower than serial.
# FIXME: If the performance of the pthreads tapir target is ever improved,
# remove the EXCLUDE argument from here.
kitsune_singlesource(
    CMDARGS -c ${ref} ${cmdargs}
    DATA ${input} ${ref}
    EXCLUDE pthreads)
