# Currently, not all examples here are correctly setup (or even correct).
# Exclude what we know does not work because we do want the globbing, but this
# list should eventually go away
set(exclude intersect)

# If a new directory is added, cmake may need to be re-run before it is picked
# up.
file(GLOB dirs LIST_DIRECTORIES ON CONFIGURE_DEPENDS *)

foreach (dir IN LISTS dirs)
  if (IS_DIRECTORY "${dir}")
    get_filename_component(basename "${dir}" NAME)
    if (NOT basename IN_LIST exclude)
      add_subdirectory("${dir}")
    endif ()
  endif ()
endforeach ()
