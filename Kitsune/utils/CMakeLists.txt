# Copy the kit-combine script into the build directory because the run-tests
# script will call it and we should not have a dependence on the source
# directory.

configure_file(kit-combine ${CMAKE_CURRENT_BINARY_DIR}/kit-combine COPYONLY)

# Copy the kit-plot script into the build directory because the test suite
# generates reports in the build directory, so it makes sense to keep the
# plotting util with the reports
configure_file(kit-plot ${CMAKE_CURRENT_BINARY_DIR}/kit-plot COPYONLY)
