include(KitsuneTestSuite)

# If a new directory is added, cmake may need to be re-run before it is picked
# up.
file(GLOB dirs LIST_DIRECTORIES ON CONFIGURE_DEPENDS *)
foreach (dir IN LISTS dirs)
  if (IS_DIRECTORY "${dir}")
    add_subdirectory("${dir}")
  endif ()
endforeach ()

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

kitsune_singlesource()
