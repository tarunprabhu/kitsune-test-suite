include(KitsuneTestSuite)

set(input fvcorr.domn.193K)
set(ref euler3d.ref.dat)

kitsune_multisource(euler3d-multifile targets EXECUTABLE
  CMDARGS -c ${ref} ${input} 50
  DATA ${input} ${ref})

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
