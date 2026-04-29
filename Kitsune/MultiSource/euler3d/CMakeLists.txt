include(KitsuneTestSuite)

set(input fvcorr.domn.193K)
set(ref euler3d.ref.dat)
set(cmdargs ${input} 20)

# Don't test this with pthreads because it is even slower than serial.
# FIXME: If the performance of the pthreads tapir target is ever improved,
# remove the EXCLUDE argument from here.
kitsune_multisource(euler3d-multifile targets EXECUTABLE
  CMDARGS -c ${ref} ${cmdargs}
  DATA ${input} ${ref}
  EXCLUDE pthreads)

foreach (target IN LISTS targets)
  target_sources(${target} PUBLIC
    compute_flux.cpp
    compute_step_factor.cpp
    init.cpp
    main.cpp
    time_step.cpp
    utils.cpp)
  target_compile_options(${target} PUBLIC ${KITSUNE_CXX_FLAGS})
endforeach ()
