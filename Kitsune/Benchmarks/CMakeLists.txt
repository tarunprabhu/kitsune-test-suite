# If a new directory is added, cmake may need to be re-run before it is picked
# up.
file(GLOB dirs LIST_DIRECTORIES ON CONFIGURE_DEPENDS *)

foreach (dir IN LISTS dirs)
  # FIXME: This is temporary while I get everything else working and correctly
  # passing tests on the GPU targets
  if (dir MATCHES "raytracer$")
    continue()
  endif()
  if (IS_DIRECTORY "${dir}")
    add_subdirectory("${dir}")
  endif ()
endforeach ()
