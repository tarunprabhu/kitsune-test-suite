# Kitsune-specific tests #

This directory contains end-to-end tests for Kitsune. They are intended to test
for both correctness and (some) performance regressions. The sections on
[building](#Building) and [running](#Running) the test suite should be
sufficient to just use the test suite.

The section for [developers](#Developer Guide) contains important information
about the test suite itself, how it is organized and the contents. The
organization of the Kitsune tests is somewhat idiosyncratic and relies on file
names following a strict pattern. Please read that section carefully before
adding/modifying this test suite.

## Building ##

The most straightforward way of building the LLVM test suite to run _only_ the
Kitsune-specific tests would be something like this:

```
$ cmake -DCMAKE_C_COMPILER=/path/to/kitcc \
    -DCMAKE_CXX_COMPILER=/path/to/kit++ \
    -DTEST_SUITE_SUBDIRS=Kitsune \
    /path/to/kitsune-test-suite
```

This will test all the tapir targets that kitsune has been built with. This may
may not always be desirable. For instance, if both the `cuda` and `hip` backends
have been built, this will attempt to run tests on both NVIDIA and AMD GPU's. In
most cases, only one of these will be present on the test machine. Also, this
will test the `serial` tapir target (which is always built) which, for some
tests, can be very slow. There are a number of ways to test only a subset of the
tapir targets.

`-DKITSUNE_SKIP_TAPIR_TARGETS=<targets>` can be used to test all tapir targets
that have been built _except_ those in `<targets>`. `<targets>` is a
semicolon-separated list.

`-DKITSUNE_TEST_TAPIR_TARGETS=<targets>` can be used to test _only_ those
tapir targets that have been built _and_ are present in `<targets>`.
`<targets>` is a semicolon-separated list.

In general, only one of `-DKITSUNE_SKIP_TAPIR_TARGETS` and
`-DKITSUNE_TEST_TAPIR_TARGETS` should be specified. Providing both is allowed,
but it is unlikely that any good will come of this.

By default, all the frontends that Kitsune was built with are tested. Currently,
frontends for C (`kitcc`) and C++ (`kit++`) are available. A Fortran frontend
(`kitfc`) is under development and should also be available in the future. The
test suite can be restricted to testing only some of the built frontends.

`-DKITSUNE_SKIP_FRONTENDS=<frontends>` can be used to test all frontends that
have been built _except_ those in `<frontends>`. `<frontends>` is a
semicolon-separated list.

`-DKITSUNE_TEST_FRONTENDS=<frontends>` can be used to test _only_ those
frontends that have been built _and_ are present in `<frontends>`. `<frontends>`
is a semciolon-separated list.

`-DKITSUNE_TEST_INPUT_SIZE` can be used to change the input sizes to some of the
tests, particularly those in `Benchmarks/`. The accepted values are `"small"`
and `"medium"`. `"small"` is the default and ensures that the tests run
relatively quickly. If `-DKITUSUNE_BENCHMARK=ON`, the problem size is
automatically set to `"medium"` since that is more useful for performance
comparisons.

`-DKITSUNE_TEST_KOKKOS_MODE` can be used to enable or disable Kitsune's Kokkos
mode where it recognizes some Kokkos constructs and generates custom code for
them. By default, this is set to `ON` and will test Kokkos mode if it has been
enabled in Kitsune. When set to `OFF`, Kokkos mode will not be tested.

`-DKITSUNE_BENCHMARK=ON` is required to enable the "benchmarking mode" of the
test suite. The `Benchmarks/` directory contains tests intended for performance
comparisons with, for instance, equivalent cuda (`.cu`) implementations. This
typically requires `nvcc` or `hipcc` to be available on the system. Turning this
on will also run the Kokkos tests in vanilla mode (in addition to Kitsune's
`-fkokkos-mode`). This will use the Kokkos backend that Kitsune was built with.
This is typically one with GPU support. _[[TODO: Explain the Kokkos details \
better]]_

## Running ##

Running the tests in the test suite requires LLVM's `lit` utility. This can be
found in the `bin/` directory within Kitsune's build directory,
`/path/to/kitsune/build/bin/llvm-lit`. This utility is not installed, even if
`-DLLVM_INSTALL_UTILS=ON` was set when building Kitsune. Another option is to
install `lit`[^1]. Some Linux distributions such as Arch and Debian package this
together with the LLVM package. Otherwise it can be obtained from PyPi, for
example, using `pip`

```
$ pip install --user lit
```

Once installed, please ensure that `lit` is in `$PATH`. A script to run the
Kitsune tests is available in the Kitsune subdirectory of the _build_
directory. In addition to running the tests, it also runs a post-processing
script that collects useful statistics into a report file in
`Kitsune/report.json`. It can be run from the build directory as shown

```
$ ./Kitsune/run-tests
```

The `run-tests` script will produce a report, `Kitsune/report.json`, containing
statistics collected at both compile and run-time. The report collects data from
both the raw report produced by the test suite in `Kitsune/report-ts.json` and
the output produced by the tests themselves. Alternatively, the tests can
also be run manually by calling `lit` directly (note that this will not produce
any report).

```
$ lit /path/to/kitsune-test-suite/build/Kitsune
```

`lit` supports some other options that may be useful. Use `lit --help` to see
a full list.


[^1]: LLVM's lit utility is written in Python, and as such, is somewhat
independent of LLVM itself. As long as one doesn't obtain a very old version of
lit, it should work even if installed separately.

## Developer Guide ##

This section contains notes on the organization of the Kitsune-specific tests
and information on how to add new tests.

## Organization ##

All tests should be within one of the three top-level directories, `Benchmarks`,
`SingleSource` and `MultiSource`.

    - *`SingleSource`* contains single-source tests that are only used for
      correctness checks and are never used for performance comparisons

    - *`MultiSource`* contains multi-source tests that are, currently, only used
      for correctness checks. We may support using these for performance
      comaprisons at some point, but it is not currently planned

### Benchmarks/ ###

The `Benchmarks/` directory is intended for code that can be used for both
correctness checks and performance comparisons. All the tests *must* be
single-source.

`Benchmarks/` consists of subdirectories, with one subdirectory for each
benchmark. Each benchmark subdirectory consists of multiple source files.
Typically, these will be C/C++/Fortran with Kitsune-specific extensions/
annotations (`.kit.*`), Cuda (`.cu`) and Hip (`.hip`) implementations of the
exact same program. Standard C++ and Fortran implementations may also be
present. In addition, there may be "included-source" files (`.inc.*`) which
are typically `#include`'ed directly into the other source files[^1].

The trailing extensions on the test file names are significant. Files containing
Kitsune-specific extensions/annotations *must* end with
`.kit.c`, `.kit.cpp`, or `.kit.f90` for C, C++ and Fortran respectively. Only
files with this name will be compiled with the various tapir targets being
tested. Cuda implementations must end with `.cu` and hip implementations must
end with `.hip`. Tests using Kokkos must end with `.kokkos.cpp` if they are
intended to be used to test Kitsune's `-fkokkos` mode. These tests can only
contain Kokkos' `parallel_for` construct. Kokkos views are not currently
supported.

Each subdirectory within `Benchmarks/` must contain a `CMakeLists.txt` file.
The following is an example of a `CMakeLists.txt` file for a benchmark that
is self-contained.
```
include(KitsuneTestSuite)

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  kitsune_singlesource(CMDARGS 8192)
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  kitsune_singlesource(CMDARGS 268435456)
else()
  kitsune_singlesource()
endif()
```
In this case, the test is given a different set of command-line arguments
depending on the input size specified at configure time. It is recommended for
a test to be able to support at least the `small` and `medium` input sizes.
If tests take too long to execute, the correctness checks may take too long on
certain tapir targets (e.g. the `serial` tapir target), while tests that run
for a very short time are not useful for performance comparisons because they
are very susceptible to system noise.

The following is an example of a `CMakeLists.txt` file for a benchmark that
compares its output against a reference output.
```
include(KitsuneTestSuite)
file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(check_file_base "refout")
if (KITSUNE_TEST_INPUT_SIZE STREQUAL "${check_file_base}")
  kitsune_singlesource(
    CMDARGS 640 640
    REFOUT "${check_file_base}.small")
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  kitsune_singlesource(
    CMDARGS ${check} 6400 6400
    REFOUT "${check_file_base}.medium")
else()
  kitsune_singlesource()
endif()
```

In order to add a new benchmark, create a subdirectory within `Benchmarks/`
and add all the equivalent files to it while making sure that they have the
correct extensions. It is not strictly necessary to have
multiple implementations --- that is mostly useful for performance
comparisons. However, if the test is entirely self-contained, i.e. it does not
require external resources such as files to check its output, and multiple
implementations are not desired, it may be more appropriate to add it to the
[`SingleSource/`](#SingleSource) directory instead.

[^1]: Yes, this is rather terrible, but this is not really intended to be a
    showcase of good software engineering practices. Besides, the tests here
    are also intended to be convenient for developers to use during
    experimentation (not just as part of a regression-testing process) and
    having to compile multiple files by hand is not as convenient.

### SingleSource/ ###

The `SingleSource/` directory contains single-source tests that are only
intended for correctness checks. All files in the directory are expected to
contain Kitsune-specific extensions/annotations, and, therefore, must be
named `.kit.*`.

Each file must check its own results and return 0 on success and non-zero on
failure. The files are not allowed to accept any command-line arguments and
must be able to check their results without any external resources (e.g. files).
There are currently no plans to remove this limitation, but it may be possible
to do so.

To add a test, simply add a file to this directory with the correct extension
and reconfigure[^2] the test suite. The newly added file should be picked up
automatically.

[^2]: It may be necessary to delete the build directory and reconfigure if
cmake does not automatically pick up any newly added files/directories.
