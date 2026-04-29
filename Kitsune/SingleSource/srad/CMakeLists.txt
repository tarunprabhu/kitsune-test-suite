include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# iterations = 100
# rows       = 320
# cols       = 320
# y1         = 0
# y2         = 127
# x1         = 0
# x2         = 127
# lambda     = 0.5
set(ref "srad.ref.dat")
kitsune_singlesource(
    CMDARGS -c ${ref} 100 320 320 0 127 0 127 0.5
    DATA ${ref})
