# # If a new directory is added, cmake may need to be re-run before it is picked
# # up.
# file(GLOB dirs LIST_DIRECTORIES ON CONFIGURE_DEPENDS *)

# foreach (dir IN LISTS dirs)
#   if (IS_DIRECTORY "${dir}")
#     add_subdirectory("${dir}")
#   endif ()
# endforeach ()

add_subdirectory(vecadd)
add_subdirectory(copy)
add_subdirectory(saxpy)
add_subdirectory(srad)
# add_subdirectory(euler3d)
add_subdirectory(raytracer)
