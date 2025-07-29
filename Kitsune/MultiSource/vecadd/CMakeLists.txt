include(KitsuneTestSuite)

kitsune_multisource(vecadd-multifile targets EXECUTABLE LTO
  CMDARGS 1024)
kitsune_multisource(vecadd-multifile targets EXECUTABLE
  CMDARGS 1024)

foreach (target IN LISTS targets)
  target_sources(${target} PUBLIC
    add.cpp
    main.cpp)
  target_compile_options(${target} PUBLIC ${KITSUNE_CXX_FLAGS})
endforeach ()
