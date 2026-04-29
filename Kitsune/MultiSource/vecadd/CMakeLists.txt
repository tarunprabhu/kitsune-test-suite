include(KitsuneTestSuite)

kitsune_multisource(vecadd-multifile targets EXECUTABLE
  CMDARGS 2048)

foreach (target IN LISTS targets)
  target_sources(${target} PUBLIC
    add.cpp
    main.cpp)
  target_compile_options(${target} PUBLIC ${KITSUNE_CXX_FLAGS})
endforeach ()
