include(KitsuneTestSuite)

kitsune_multisource(vecadd-multifile targets EXECUTABLE
  CMDARGS 1024)

foreach (target IN LISTS targets)
  target_sources(${target} PUBLIC
    add.cpp
    main.cpp)
endforeach ()
