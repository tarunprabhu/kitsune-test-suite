include(KitsuneTestSuite)

add_subdirectories()

# This contains a lit.local.cfg since files may be added directly to this
# directory. Single-source tests are not required to be in subdirectories.
file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

kitsune_singlesource()
