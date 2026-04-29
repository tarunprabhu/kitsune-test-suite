include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# samples    8
# width      160
# height     100
set(ref "raytracer.ref.dat")
kitsune_singlesource(
    CMDARGS -c ${ref} 8 160 100
    DATA ${ref})
