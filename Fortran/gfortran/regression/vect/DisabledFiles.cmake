
#===------------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# There are currently no unsupported files.
set(UNSUPPORTED_FILES "")

# There are currently no unimplemented files.
set(UNIMPLEMENTED_FILES "")

# There are currently no skipped files.
set(SKIPPED_FILES "")

# These tests fail when they are expected to pass.
file(GLOB FAILING_FILES CONFIGURE_DEPENDS
  # These tests fail to compile when compilation is expected to succeed.
  vect-8-epilogue.F90

  # The cause of failure of this test needs to be investigated
  pr49955.f
)
