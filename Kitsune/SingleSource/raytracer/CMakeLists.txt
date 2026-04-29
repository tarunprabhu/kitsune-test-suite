include(KitsuneTestSuite)

# samples    8
# width      160
# height     100
set(ref "raytracer.ref.dat")
kitsune_singlesource(
    CMDARGS -c ${ref} 8 160 100
    DATA ${ref})
