# Kitsune-specific tests #

This directory contains end-to-end tests for Kitsune. They are intended to test
for both correctness and (some) performance regressions. The sections on
[building](#Building) and [running](#Running) the test suite should be
sufficient to use the test suite.

The section for [developers](#Developer Guide) contains important information
about the test suite itself, how it is organized and the contents. The
organization of the Kitsune tests is somewhat idiosyncratic and relies on file
names following a strict pattern. Please read that section carefully before
adding/modifying this test suite.

## Building ##

The following is the most straightforward way to build this test suite. By
default, only the tests in the `Kitsune/` subdirectory will be run.

```
$ cmake -DCMAKE_C_COMPILER=/path/to/kitcc \
    -DCMAKE_CXX_COMPILER=/path/to/kit++ \
    /path/to/kitsune-test-suite
```

This will test all tapir targets that have been enabled in the Kitsune build. If
a tapir target cannot be used on the system on which the test suite is built, it
will not be tested. In practice this only happens with the GPU-centric tapir
targets. For instance, the `cuda` tapir target will not be tested if the system
does not contain an NVIDIA GPU.

Some options are provided to control the set of tapir targets that are tested.

`-DKITSUNE_SKIP_TAPIR_TARGETS=<targets>` can be used to test all tapir targets
that have been built _except_ those in `<targets>`. `<targets>` is a
semicolon-separated list.

`-DKITSUNE_TEST_TAPIR_TARGETS=<targets>` can be used to test _only_ those
tapir targets that have been built _and_ are present in `<targets>`.
`<targets>` is a semicolon-separated list.

Exactly one of `-DKITSUNE_SKIP_TAPIR_TARGETS` and `-DKITSUNE_TEST_TAPIR_TARGETS`
may be specified. Providing both will result in a configure-time error.

By default, all frontends enabled in the Kitsune build are tested. Currently,
frontends for C (`kitcc`) and C++ (`kit++`) are available. A Fortran frontend
(`kitfc`) is under development and should also be available in the future.

Some options are provided to control the set of frontends that are tested.

`-DKITSUNE_SKIP_FRONTENDS=<frontends>` can be used to test all frontends that
have been built _except_ those in `<frontends>`. `<frontends>` is a
semicolon-separated list.

`-DKITSUNE_TEST_FRONTENDS=<frontends>` can be used to test _only_ those
frontends that have been built _and_ are present in `<frontends>`. `<frontends>`
is a semciolon-separated list.

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

[Running](#Running) the tests in the test suite requires LLVM's `lit` utility.
This can be found in the `bin/` directory within Kitsune's build directory,
`/path/to/kitsune/build/bin/llvm-lit`. This utility is not installed, even if
`-DLLVM_INSTALL_UTILS=ON` was set when building Kitsune. Another option is to
install `lit` separately[^1]. Note that the separately installed `lit`
executable is not exactly the same as `llvm-lit` that is bundled together with
LLVM. It usually works, but using `llvm-lit` is the safer option.

To ensure that the expected `lit` executable is used, use
`-DTEST_SUITE_LIT=<path/to/lit>`.

## Running ##

There are two ways to run the tests.

The standard approach is to simply invoke the `check` target. If the `ninja`
generator was used when configuring the test suite (recommended), this can be
done as follows:

```
$ ninja check
```

An alternative is to use the `run-tests` script.In addition to running the
tests, this also runs a post-processing pass that collects useful statistics
into a concise report file. It can be run from the `<build>` directory.

```
$ ./Kitsune/run-tests
```

The `run-tests` script will produce a report, `Kitsune/report.json`, containing
statistics collected at both compile and run-time. The report collects data from
both the raw report produced by the test suite in `Kitsune/report-ts.json` and
the output produced by the tests themselves. This is most useful when
`-DKITSUNE_BENCHMARK=ON` is specified when configuring this test suite.

Yet another approach is to invoke `lit` directly.

```
$ lit -j 1 <build>/Kitsune
```

Here `<build>` is the path to the build directory. Note that `-j 1` (or the
equivalent `--threads=1` or `--workers=1`) is required. This forces `lit` to use
a single worker i.e. all tests are run sequentially.

`lit` supports several other options that may be useful. Use `lit --help` to see
a full list.

[^1]: Some Linux distributions such as Arch and Debian package this together
with the LLVM package. Otherwise it can be obtained from PyPi, for example,
using `pip` _[[`pip install --user lit`]]_.

## Developer Guide ##

This section contains notes on the organization of the Kitsune-specific tests
and information on how to add new tests.

### Organization ###

All tests should be within one of the three top-level directories, `Benchmarks`,
`SingleSource` and `MultiSource`.

    - *`Benchmarks`* contains tests that are used for both correctness checks
      and performance comparisons

    - *`SingleSource`* contains single-source tests that are only used for
      correctness checks and are never used for performance comparisons

    - *`MultiSource`* contains multi-source tests that are, currently, only used
      for correctness checks. We may support using these for performance
      comaprisons at some point, but it is not currently planned

### Benchmarks/ ###

The `Benchmarks/` directory is intended for code that can be used for both
correctness checks and performance comparisons.

`Benchmarks/` consists of subdirectories, with one subdirectory for each
benchmark. Each benchmark subdirectory consists of multiple source files. Each
file contains a different implementation of the same benchmark, just using a
different language/framework. Each of these implementations must, obviously, be
"equivalent" to ensure that their performance can be compared. In addition,
there may be "included-source" files (`.inc.*`) which are typically
`#include`'ed directly into the other source files[^1].

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

if (KITSUNE_BENCHMARK)
  kitsune_singlesource(CMDARGS 268435456)
else ()
  kitsune_singlesource(CMDARGS 8192)
endif()
```
In this case, the test is given a different set of command-line arguments
depending on whether or not `KITSUNE_BENCHMARK` is set.

In order to add a new benchmark, create a subdirectory within `Benchmarks/`.
Add the required implementations to it while making sure that they have the
correct extensions. It is not strictly necessary to have multiple
implementations, but a benchmark is, obviously, less useful without them.

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
