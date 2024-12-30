# Copy the kit-combine script into the build directory because the run-tests
# script will call it and we should not have a dependence on the source
# directory.

configure_file(kit-combine ${CMAKE_CURRENT_BINARY_DIR}/kit-combine COPYONLY)
