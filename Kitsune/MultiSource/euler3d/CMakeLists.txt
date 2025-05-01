include(KitsuneTestSuite)

set(input fvcorr.domn.193K)
set(base euler3d.ref)
set(refc "${base}.cpu.dat")
set(refg "${base}.gpu.dat")

kitsune_multisource(euler3d-multifile targets EXECUTABLE
  CMDARGS -c ${refc} -g ${refg} ${input} 50
  DATA ${input} ${refc} ${refg})

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
