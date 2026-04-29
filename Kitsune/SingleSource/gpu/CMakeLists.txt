include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# The files in this directory test behavior that is only relevant when compiling
# for a GPU. We could build them for the other tapir targets as well, but it
# will only increase testing time and resource usage for limited returns.
kitsune_singlesource(ONLY cuda hip)
