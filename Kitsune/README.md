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
This is typically one with GPU support. _[[TODO: Explain the Kokkos details
better]]_

`-DKITSUNE_C_FLAGS`, `-DKITSUNE_CXX_FLAGS`, and `-DKITSUNE_Fortran_FLAGS` can be
used to pass options to Kitsune's C, C++, and Fortran frontends. While the
standard `cmake` variables `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS`, and
`CMAKE_Fortran_FLAGS` could also be used for this, the flags would then also be
used to compile other test-suite utilities such as [fpcmp](tools/fpcmp.c). This
may not be desirable, so the use of the `KITSUNE_*` variables is strongly
recommended in this case.

[Running](#Running) the tests in the test suite requires LLVM's `lit` utility.
This can be found in the `bin/` directory within Kitsune's build directory.
This utility is not installed, even if `-DLLVM_INSTALL_UTILS=ON` is set when
building Kitsune. Another option is to install `lit` separately[^1]. Note that
the separately installed `lit` executable is not exactly the same as `llvm-lit`
that is bundled together with LLVM. It usually works, but using `llvm-lit` is
the safer option.

To ensure that the expected `lit` executable is used, use
`-DTEST_SUITE_LIT=<path/to/lit>`. `-DTEST_SUITE_LIT_FLAGS` can be used to pass
additional options to `lit`. Thie can be set to `-s` to produce less verbose
terminal output when running the test suite.

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

[^1]: Some Linux distributions such as Arch and Debian package this together with the LLVM package. Otherwise it can be obtained from PyPi, for example, using `pip`. One way to do this may be `pip install --user lit`.

## Developer Guide ##

This section contains notes on the organization of the Kitsune-specific tests
and information on how to add new tests.

### File names ###

The tests in this test-suite are either single-source or multi-source. The
trailing extensions on the single-source test file names are significant. These
are used to determine the correct way to build the files and their use is
_required_.

Files containing Kitsune-specific extensions/annotations *must* end with
`.kit.c`, `.kit.cpp`, or `.kit.f90` for C, C++ and Fortran respectively. Only
files with this name will be compiled with the various tapir targets being
tested.

Tests using Kokkos must end with `.kokkos.kit.cpp` if they are intended to be
used to test Kitsune's `-fkokkos` mode. Currently, these are only built with
the GPU-centric tapir targets, `cuda`, and `hip`. These tests should only
contain supported Kokkos' constructs. At the time of writing, this is only
`kokkos::parallel_for` and a limited set of execution policies. `kokkos::View`s
are not yet supported and will likely never be supported.

Cuda-language implementations for NVIDIA GPU's must end with `.cu`. HIP-language
implementations for AMD GPU's must end with `.hip`.

Files ending with `.kokkos.cpp` (note the absence of `.kit` here) are treated
as "vanilla" Kokkos and not compiled with any tapir target. These may contain
any Kokkos construct. These, too, are linked against an appropriate Kokkos
GPU backend.

```{note}
These restrictions are only for single-source tests. Multi-source tests may
use the "standard" extensions for source files.
```

### Organization ###

All tests should be within one of the three top-level directories, `Benchmarks`,
`SingleSource` and `MultiSource`.

  - *`Benchmarks`* contains tests that are used to monitor for performance
    regressions and to carry out performance measurements. This directory is
    only entered when `-DKITSUNE_BENCHMARK=ON` is provided at configure-time.
    The [dedicated section](#Benchmarks) contains more information about these.

  - *`SingleSource`* contains single-source tests that are only used for
    correctness checks and are never used for performance comparisons. For
    more information, see [here](#SingleSource).

  - *`MultiSource`* contains multi-source tests that are, currently, only used
    for correctness checks. We may support using these for performance
    comaprisons at some point, but it is not currently planned.

### Benchmarks ###

The `Benchmarks/` directory is intended for code that can be used for both
correctness checks and performance comparisons.

`Benchmarks/` consists of subdirectories, with one subdirectory for each
benchmark. Each of these subdirectories contain multiple source files. Each
file is a different implementation of the benchmark, using different
languages/frameworks. Each of these implementations must, obviously, be
"equivalent" to ensure that their performance can be compared. In addition,
there may be "included-source" files (`.inc.*`) which are typically
`#include`'ed directly into the other source files[^2].

Each subdirectory within `Benchmarks/` must contain a `CMakeLists.txt` file.

In order to add a new benchmark, create a subdirectory within `Benchmarks/`.
Add the required implementations to it while making sure that they have the
correct extensions. It is not strictly necessary to have multiple
implementations, but a benchmark is less useful without them.

[^2]: Yes, this is rather terrible, but this is not really intended to be a showcase of good software engineering practices. Besides, the tests here are also intended to be convenient for developers to use during experimentation (not just as part of a regression-testing process) and having to compile multiple files by hand is not as convenient.

### SingleSource/ ###

The `SingleSource/` directory contains single-source tests that are only
intended for correctness checks. To add a test, simply add a file to this
directory with the correct extension and reconfigure[^3] the test suite. These
tests must not take any command line arguments, must not use any external
resources such as files and must check their own output, returning a
system-specific code indicating success or failure. On all systems that Kitsune
supports, returning 0 indicates success, non-zero indicates failure.

Tests that require command-line arguments, or external resources such as
input files, or reference outputs must be added to a subdirectory of
`SingleSource/`. All such subdirectories must have a `CMakeLists.txt` file.
A minimal `CMakeLists.txt` for such directories is shown below.

```cmake
include(KitsuneTestSuite)

kitsune_singlesource(CMDARGS <args>)
```

Here, `<args>` is a space separated list of command-line arguments that are
passed to all single-source tests in the directory. Note that the same set of
arguments are passed to _all_ tests in the directory.

The `DATA` keyword argument must be passed to `kitsune_singlesource` to ensure
that any files required by the test are available at test-time. The example
below shows how this might be used:

```cmake
set(input1 "input1.txt")
set(input2 "input2.dat")
set(ref "expected.dat")
kitsune_singlesource(
    CMDARGS -c ${ref} ${input1} ${input2}
    DATA ${ref} ${input1} ${input2})
```

Here, `input1.txt`, `input2.txt`, and `expected.dat` are all present in the
same directory as the tests. The test suite will copy these files to (or
symlink to them from) the directory that will eventually contain the test
executables. The tests themselves will be run from the same directory, so
simply passing the file name to a suitable `open` function will work as
expected. For example, in C, one could even do the following:

```c
FILE *fp = fopen("input1.txt", "r");
```

The tests can also be configured to only be built with specific tapir targets
as shown below.

```cmake
kitsune_singlesource(ONLY cuda hip)
```

Here, the tests will only built with the `cuda` and `hip` tapir targets. If
the `ONLY` keyword argument is not used, the tests will be built with all
enabled tapir targets.

The `EXCLUDE` keyword argument can be used to prevent a test from being built
with one or more tapir targets.

```cmake
kitsune_singlesource(EXCLUDE pthreads serial)
```

Here, the tests will not be built with the `pthreads` and `serial` tapir
targets.

```{warning}
Providing both `EXCLUDE` and `ONLY` to `kitsune_singlesource` will result in a
configure-time error.
```

[^3]: It may be necessary to delete the build directory and reconfigure if cmake does not automatically pick up any newly added files/directories.


### MultiSource ###

The `MultiSource/` directory contains tests consisting of multiple source files.
Each subdirectory of `MultiSource/` contains a single test. Unlike for the
single-source tests, there is no automatic way to build these. A minimal
example of a `CMakeLists.txt` file to build these is as shown below:

```cmake
include(KitsuneTestSuite)

set(execs)
kitsune_multisource(<target-base-name> execs EXECUTABLE)

foreach (exec IN LISTS execs)
  target_sources(${exec} PUBLIC
    <source-files>)
  target_compile_options(${target} PUBLIC ${KITSUNE_CXX_FLAGS})
endforeach ()
```

Here, we first create a variable named `execs`. This is a list that will be
populated by `kitsune_multisource` with the `CMake` targets for the executables
that will be created for each tapir target. For instance, if
`<target-base-name>` is `vecadd`, and the tapir targets enabled in Kitsune are
`cuda`, `openmp`, and `serial`, `kitsune_multisource` will set
`execs` to something like the following

```cmake
vecadd-multifile-kitlang-cuda;vecadd-multifile-kitlang-openmp;vecadd-multifile-kitlang-serial
```

We then loop over `execs` and add the source files and any required compiler
options explicitly to each target. While we do not do so here, any preprocessor
and linker options may have to be added here. This loop is required since there
is no other way to know which source files are needed, and the test suite does
not know how this test is to be compiled. In the example above, we have added
`KITSUNE_CXX_FLAGS` because, presumably, the test consists of exclusively C++
source files. But if the source is in other languages, we may have to pass a
different set of flags.

Since there are too many different ways in which multi-file tests can be built,
we currently do not provide any convenience functions, even for the simple
cases.
