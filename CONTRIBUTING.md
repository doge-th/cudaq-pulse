# Contributing to cudaq-pulse

Everyone taking part in the project follows the
[Code of Conduct](CODE_OF_CONDUCT.md). [SUPPORT.md](SUPPORT.md) says where to
ask questions and report bugs; report security vulnerabilities as described
in [SECURITY.md](SECURITY.md), not in an issue.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it provides Python, CMake, Ninja, and
  every Python tool below)
- A C++20 compiler: GCC or Clang on Linux, the Xcode command-line tools on
  macOS. CI builds with GCC 13 (Ubuntu 24.04), the manylinux_2_28 toolchain,
  the Apple Clang of macOS 15 and the oldest compilers supported, GCC 11 and
  Clang 14 (Ubuntu 22.04)
- For GPU work only: Linux, an NVIDIA GPU and driver, and the CUDA 12.9
  toolkit (or no toolkit at all; see [GPU builds](#gpu-builds))

You do not need CUDA-Q, an LLVM build, or Docker.

## Set up

```bash
git clone https://github.com/NVIDIA/cudaq-pulse.git
cd cudaq-pulse

uv sync                                  # .venv with build, test, lint, docs tools
python3 tools/fetch-mlir-toolchain.py    # pinned MLIR 22.1.8 into .cache/mlir/
```

`uv sync` installs tooling only (`[tool.uv] package = false`); the package is
built by CMake, below. The fetch script downloads the toolchain archive for
your platform, verifies its SHA-256 against `tools/mlir-toolchain.lock.json`,
and unpacks it. It refuses archives it cannot verify. Configuring runs it
for you if you skip this step; on a platform without a pinned archive, pass
`-DMLIR_DIR=<prefix>/lib/cmake/mlir` for your own MLIR 22.1.8 build.

## Build and test

```bash
source .venv/bin/activate

cmake --preset dev                         # build/dev, with the tests
cmake --build --preset check               # build, then run check-pulse
```

`CMakePresets.json` has a configure preset for each build in this guide:
`dev`, `docs`, `gpu`, `free-threaded`, `asan` and `tsan`, each in
`build/<preset>` and using `.venv/bin/python`; `cmake --list-presets` lists
them. Add `-D` options after `--preset`, and keep personal presets in an
untracked `CMakeUserPresets.json`.

`check-pulse` builds everything and runs the lit/FileCheck tests in `test/`,
the non-GPU pytest suite in `tests/`, and `check-pulse-runtime`: the
cuDensityMat runtime built against a host fake of cuDensityMat
(`tests/runtime/fake_cudensitymat/`), so its callback table and integrators
are tested without a GPU. The staged package lives in
`build/dev/python`; to use it directly:

```bash
export PYTHONPATH="$PWD/build/dev/python"
export PATH="$PWD/build/dev/bin:$PATH"     # cudaq-pulse-opt
python examples/01_single_qubit_rabi.py
```

Python sources are staged as symlinks, so edits to `core/frontend/` take
effect without rebuilding; C++ changes need `cmake --build build/dev`, and a
new or renamed Python file needs a configure (`cmake build/dev`).

### Run a subset

`test/` holds the lit and FileCheck tests of the dialects and passes;
`tests/` holds the pytest suite. `check-pulse` runs `check-pulse-mlir` (lit),
`check-pulse-python` (pytest without the GPU tests) and
`check-pulse-runtime` (the cuDensityMat host test); each is a target of its
own. For a faster loop, run the tools directly:

```bash
export PYTHONPATH="$PWD/build/dev/python"
pytest tests/test_compile.py -k metrics    # one file, filtered by name
pytest tests -m "not gpu"                  # what check-pulse-python runs
pytest tests/physics -m physics --run-nightly   # the physics suite, slow checks too
pytest tests/test_perf_guard.py --run-benchmarks
lit -sv build/dev/test/Dialect/Pulse        # one lit directory
```

The markers are `gpu` (needs a GPU runtime; selecting one without a GPU
fails instead of skipping), `physics` (everything in `tests/physics`),
`nightly` (slow physics checks, skipped without `--run-nightly`),
`benchmark` (timing guards, skipped without `--run-benchmarks`),
`scaling` (benchmarks that compare two sizes in one run) and `slow` (tests
that spend seconds in native code or child processes, which the sanitizer job
leaves out). `tests/conftest.py`
sets `CUDAQ_PULSE_QUANTUM_SYSTEM=ladder:8192` for the whole session, and
`CUDAQ_PULSE_VERIFY=1` unless it is set already. Tests run with
the default unroll threshold; a test of a rolled loop of a few iterations
passes `unroll_threshold=` to `pulse.compile` or `@pulse.kernel`.

Tests that need pyright, mypy, matplotlib or a C compiler skip without
them, except when `CI` is set: there they fail, unless the job names the
tool in `CUDAQ_PULSE_OPTIONAL_TOOLS` (`tests/_tools.py`). Rewrite the IR
goldens with `CUDAQ_PULSE_UPDATE_GOLDENS=1`; a run that changes them fails
and names the changed cases, and the next run, without the variable, passes.

`CUDAQ_PULSE_VERIFY=1` turns on the verifications that only catch a bug of
the compiler itself, as MLIR's verify-each does: the module `run_passes()`
leaves, before scheduling verifies it, and each specialization rescheduled
from a verified module, which is otherwise printed without verifying it; a
module noted as verified that does not verify then fails `mlir()`. A user's
compile skips them (unset or `0`); any other value makes compiling raise
`ValueError`. The test suite, `check-pulse`, `ctest` and every CI job that
runs pytest have it on through `tests/conftest.py`; run a subset with
`CUDAQ_PULSE_VERIFY=0 pytest ...` to test what users run. The perf gate
measures with it off. It is a development aid, not part of the stable
interface.

`CUDAQ_PULSE_GPU_OPERATORS` chooses how `qop-to-cudm` builds the GPU
backend's operator, as its `operators` option: `fused`, the default, sums
the static terms and dissipators into a few dense operators and stores the
single-diagonal operators of modes of four or more levels by their diagonal;
`banded` only does the latter; `dense` lowers every product to its own term
of dense operators, as before either. Use it to tell whether a
difference in a GPU simulation comes from the lowering; any other value fails
the lowering. Like `CUDAQ_PULSE_VERIFY` it is a development aid, not part of
the stable interface.

The property tests use hypothesis profiles (`tests/property_settings.py`),
chosen with `HYPOTHESIS_PROFILE`: `ci`, the default, is derandomized so every
run checks the same examples; `dev` draws new examples; `nightly` draws ten
times as many. A failure prints a `@reproduce_failure` line; a nightly
failure also gives its seed in the job summary, replayed with
`HYPOTHESIS_PROFILE=nightly pytest <test> --hypothesis-seed=<seed>`.

## Writing passes

Compiler passes are C++ MLIR passes in `core/mlir/transforms/`. To try one
without changing the tree, build it as a plugin and load it with
`cudaq-pulse-opt --load-pass-plugin`; checks over a compiled kernel's
schedule can be plain Python. [`examples/custom_passes/`](examples/custom_passes)
has a minimal example of each, and
[docs/user_guide/custom_passes.rst](docs/user_guide/custom_passes.rst)
walks through them. `check-pulse` builds the example plugin and runs it in
`test/Plugins/`.

## Lint and type-check

```bash
pre-commit install                       # once; runs the hooks on each commit
pre-commit run --all-files

MYPYPATH=build/dev/python mypy -p cudaq_pulse
```

The hooks run the ruff linter and formatter, clang-format, whitespace and
YAML/TOML checks, and a license-header check
(`tools/check-license-headers.py`). New source files start with these two
lines, in the file's comment syntax (after a shebang line, if any):

```text
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

mypy runs in strict mode against the staged package, because it needs the
type stub that the build generates for the native extension.

## Documentation

```bash
cmake --preset docs
cmake --build --preset docs                # pulse-docs; HTML in build/docs/docs/html
```

The dialect reference pages under `architecture/dialects/` are generated by
`mlir-tblgen -gen-dialect-doc` from the ODS files, so `pulse-docs` builds the
site from a copy of `docs/` with those pages added; running `sphinx-build` on
`docs/` directly leaves them out. Describe a new op with `description` in its
`.td` definition.

## Sanitizers

`-DCUDAQ_PULSE_SANITIZERS="address;undefined"` (or `thread`) instruments
cudaq-pulse's own C++; the prebuilt MLIR toolchain stays uninstrumented. The
`sanitizers` workflow builds both variants on Linux. The address and
undefined-behavior build runs `check-pulse-mlir`, `check-pulse-runtime` and
the Python suite; the thread build runs `check-pulse-runtime`, whose GPU
runtime host test evolves runs concurrently. Under address and undefined
behavior, `check-pulse-mlir` and `check-pulse-runtime` block a merge. The
Python suite there and the thread build are informational until the MLIR
toolchain is built with the sanitizers too; it leaves out the tests marked
`slow` and stops after 40 minutes. For the Python suite, preload the runtimes
into the uninstrumented interpreter; tests that run a tool other than Python
or cudaq-pulse-opt (pyright, git, a C compiler) remove them from its
environment with `tool_env` (`tests/_tools.py`):

```bash
LD_PRELOAD="$(gcc -print-file-name=libasan.so):$(gcc -print-file-name=libubsan.so)" \
ASAN_OPTIONS=detect_leaks=0 PYTHONPATH=build/san/python python -m pytest tests -m "not gpu and not slow"
```

## Benchmarks

### The perf gate

The `perf-gate` CI job blocks a pull request on measurements that repeat
exactly from run to run, not on wall-clock time. It runs the cases in
`benchmarks/perf_gate/cases.py`: compiles of small and large kernels,
including kernels over lists of qudits and kernels with classical calls,
plus the first compile in a process, binding and specializing new values,
`submit()`, the controller and emulator exports, and the setup of a
simulation. It compares each case with `benchmarks/baselines/perf_gate.json`
on two measures:

- **Instructions** executed per repetition, counted by Valgrind's Callgrind
  on Linux x86-64. A case fails at more than 0.4% above its baseline. It
  also fails at more than 1% below, so that a speedup is recorded in the
  baseline instead of hiding a later regression. A failure says whether it
  is a regression, a gain or layout only (below).
- **Program sizes**: op count, packed program bytes and export bytes. Any
  change fails. These are the same on every platform, and
  `tests/test_perf_gate.py` checks them in the regular test suite too.

```bash
PYTHONPATH=build/dev/python python tools/perf-gate.py      # compare
PYTHONPATH=build/dev/python python tools/perf-gate.py --update
```

Without Valgrind (macOS, or Linux on another architecture), the tool
compares and updates only the program sizes. The instruction counts depend
on the toolchain, so they are taken in CI's environment, where the image,
its packages, uv and Python are pinned. Once a change that moves them on
purpose is on its pull request's branch in this repository, a manual run of
CI on that branch with `perf_baseline` runs only the perf gate, with five
identical runs, and its `perf-gate-baseline` artifact is the new baseline:

```bash
gh workflow run ci.yml --ref <branch> -f perf_baseline=true
gh run list -w ci.yml -b <branch> -e workflow_dispatch -L 1    # its <run-id>
gh run download <run-id> -R <owner>/<repo> -n perf-gate-baseline -D benchmarks/baselines
```

Check the run's summary first (no banner saying the counts aren't valid or
that identical runs differed). Then commit `perf_gate.json` alone on top of
the branch, which must still be the commit the run measured (the file's
`source.commit`), and check that the pull request's run passes.
`python tools/perf-gate.py --help` lists the same steps.

**Bootstrap mode.** A baseline that says `"bootstrap": true` and holds no
instruction counts is in bootstrap mode. The job still measures every case
twice and reports the counts and their spread, but they don't fail the run;
program sizes block as always. The summary says so in a banner, together
with the `gh run download` command for that run. Taking a run's
measurements as the baseline removes the field, and `"bootstrap": true`
next to any instruction count is an
error (in the job and in `tests/test_perf_gate.py`), so the gate can't stay
in bootstrap mode once it has counts. Bootstrap mode also can't reach a
release: the field fails the perf gate on `main`, and `tools/release-meta.py`
refuses a release tag while it is there. Without the field, a case with no
count fails. If the two runs of a bootstrap differ at all for some case, the
summary flags it: give that case its own `tolerance` in
`benchmarks/perf_gate/cases.py` before taking the baseline. A new case
added later needs its count taken the same way, on its pull request's branch.

**Counts that can't be right fail in every mode**, bootstrap included: a
case without a count, a count of zero, a compile under 100,000 instructions
per repetition, or a heavier case (a large compile, the first compile)
measuring less than twice a lighter one. Such a run's counts are left out of
its `perf-gate-baseline` artifact. The `perf-gate-callgrind-dumps` artifact
holds each run's raw Callgrind dumps and Valgrind log for inspection
(`callgrind_annotate` reads them); locally, `--dumps DIR` keeps them.

The
summary lists every case with its baseline, its count in this run, the
change, and the spread between two identical runs made in the same job. A
case whose own numbers move more than the default allows declares its own
`tolerance` (a fraction, below 0.5%) in `benchmarks/perf_gate/cases.py`;
the baseline records it, and fails if it doesn't match. The baseline also
records the environment its counts were measured in and the run they came
from. Changing a pin in the job (image digest, `UBUNTU_SNAPSHOT`, uv or
Python version, MLIR toolchain) changes the counts, and counts from another
environment fail the gate, so take a new baseline as above on the change's
pull request branch.

**Regressions, gains and layout only.** The job also counts each case's
Python-level work per repetition (bytecodes executed, Python calls and C
calls, with `sys.monitoring`), natively and apart from the Callgrind runs,
and the baseline records it. It never gates, but every failure on
instructions says its kind, and the summary counts them by kind:

- **regression**: above the limit;
- **gain**: more than 1% below; take a new baseline to keep it;
- **layout only**: the same Python work as the baseline's (the summary's
  "Python work" column says `same`) and a change of at most 2% either way.
  The instructions moved below Python, most likely in the layout of the
  interpreter's caches and memory: for example, one new class at import
  renumbers the interpreter's type version tags and changes its method
  cache's collisions. Judge it (compare the Callgrind dumps with the
  baseline run's) before taking a new baseline.

A regression or a gain lists each counter of Python work that changed, or,
with the same work, says the change is in native code. Every kind fails. A
baseline without Python work tells only regressions from gains.

Wall-clock times are informational only: shared runners are too noisy to
block on. The test job still reports `tests/test_perf_guard.py`'s
baseline ratios as a warning, and fails on its `scaling` tests, which
compare two sizes of one program in one process. Release-time wall-clock
comparisons stay a manual A/B run on a quiet machine.

### Other benchmarks

```bash
PYTHONPATH=build/dev/python python benchmarks/compiler/compile_time.py \
  --baseline benchmarks/baselines/compile_time.json
```

`compile_time.py` times the canonical workloads and checks their op counts
against `compile_time.json`. `tests/classical/test_no_call_identity.py`
checks the same op counts on every platform and in the wheels, and
`tests/test_perf_gate.py` checks that they agree with `perf_gate.json`. If a
change alters them on purpose, regenerate the file with `--write-baseline`
as well. Timing is only gated when you pass `--gate FACTOR`.

`benchmarks/compiler/mlir_pipeline_bench.py` breaks compile time down by
stage and by pass and reports IR sizes; `benchmarks/simulation/qec_scaling.py`
shows how compile time grows with surface-code distance, and
`benchmarks/compiler/controller_export.py` the time, bytes and peak memory
of checking and exporting large programs for a controller, in format /1
and /2. CI runs these at a small size to keep them working.
`benchmarks/compiler/specialize_bench.py` compares a call with a new float
value against a fresh compile and fails if the call is not at least ten
times cheaper; the nightly workflow runs it.

## GPU builds

GPU support is off by default and, when requested, mandatory: configuration
fails rather than silently producing a compiler-only build.

```bash
uv pip install -r pyproject.toml --extra cu12
SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

cmake --preset gpu -DCUDAQ_PULSE_CUDA12_ROOT="/usr/local/cuda-12.9;$SITE"
cmake --build build/gpu --target check-pulse
cmake --build build/gpu --target check-pulse-gpu
```

The first command installs the `[cu12]` extra's dependencies (not the package
itself) into `.venv`: the same libraries `pip install "cudaq-pulse[cu12]"`
gives users, including the cuDensityMat headers the build compiles against.
`check-pulse-gpu` runs the cuDensityMat linkage smoke test and the numerical
GPU suite, and fails if no device is usable.

The build produces one runtime library per CUDA major,
`libcudm-runtime-cu12.so` and `libcudm-runtime-cu13.so`, from
`CUDAQ_PULSE_CUDA12_ROOT` and `CUDAQ_PULSE_CUDA13_ROOT`. Set either or both.
Each is a list of directories that together provide that major's CUDA runtime
(`cuda_runtime_api.h`, `crt/host_defines.h`, `libcudart.so.NN`) and
cuDensityMat (`cudensitymat.h`, `libcudensitymat.so.0`), as toolkit-style
`include/` and `lib/` prefixes or as site-packages trees. Above, the toolkit
supplies the CUDA headers and the `[cu12]` wheels supply cuDensityMat.

Without a toolkit, `crt/host_defines.h` comes from one more wheel
(`nvidia-cuda-nvcc-cu12` for CUDA 12, `nvidia-cuda-crt` for CUDA 13), and the
site-packages directory alone is the root; `tools/ci/manylinux-before-all.sh`
does this for the release wheels. For CUDA 13, use `--extra cu13` in a
separate environment (the two extras cannot share one) and pass
`-DCUDAQ_PULSE_CUDA13_ROOT`. At run time the loader picks the runtime matching
the installed extra; `CUDAQ_PULSE_CUDA=12|13` forces one.

## Build a wheel

```bash
pip install ./packaging/cudaq-pulse-tools .
# or: uv build --wheel packaging/cudaq-pulse-tools && uv build --wheel
```

There are two distributions. `cudaq-pulse-tools`
(`packaging/cudaq-pulse-tools`) holds the command-line tools, for now the
`cudaq-pulse-opt` MLIR driver alone, a `py3-none` wheel per platform that
serves every Python; `cudaq-pulse` is the Python package and its extension,
and requires `cudaq-pulse-tools` at exactly its own version, so
`pip install .` alone finds no driver to install. Both are built from this tree and take their version
from the same git tags: `CUDAQ_PULSE_BUILD_OPT=OFF` and
`CUDAQ_PULSE_BUILD_PYTHON=OFF` select the half each one builds. Both use the
same fetched toolchain. `cudaq-pulse-tools` has no sdist. Wheels are
compiler-only unless `CUDAQ_PULSE_ENABLE_GPU=ON` is set in the environment
(as the release build does on Linux; see `[tool.cibuildwheel]` in
`pyproject.toml`).

Every binary a wheel ships writes a linker map to `<build dir>/link-maps/`,
and installing fails if the wheel's `THIRD_PARTY_NOTICES` (the root one for
`cudaq-pulse`, `packaging/cudaq-pulse-tools/THIRD_PARTY_NOTICES` for the
driver) no longer lists the third-party code those maps show linked. After a
change that links something new (an MLIR library, a toolchain bump),
regenerate them and commit the result, with a license text in `LICENSES/`
(and a link to it in `packaging/cudaq-pulse-tools/LICENSES/` if the driver
needs it) for any new component:

```bash
python tools/third-party-notices.py write --notices THIRD_PARTY_NOTICES \
    --map build/dev/link-maps/_cudaq_pulse_native.map
python tools/third-party-notices.py write \
    --notices packaging/cudaq-pulse-tools/THIRD_PARTY_NOTICES \
    --map build/dev/link-maps/cudaq-pulse-opt.map
```

The tool rejects archives it does not know; teach it new ones in its
component tables. `tools/check-wheel-contents.py` checks the built wheels
for the license files, the notices (and any library a wheel repair grafted),
and for CUDA or GPL/LGPL libraries.

The abi3 wheels build the extension against the limited Python API
(`Py_LIMITED_API`), which a development build never sets, so a macro such
as `PyList_SET_ITEM` builds everywhere but in the wheels. Check a change to
`core/mlir/bindings` against it with
`python tools/check-limited-api.py build/dev`, which CI's test job runs.

## The MLIR toolchain

Configuring needs the pinned MLIR 22.1.8 toolchain:

1. `python3 tools/fetch-mlir-toolchain.py` downloads the archive for your
   platform from the `mlir-toolchain-22.1.8` release, checks its SHA-256
   against `tools/mlir-toolchain.lock.json` and unpacks it into
   `.cache/mlir/`. Configuring runs it when the toolchain is missing.
2. On a platform without a pinned archive, build one with
   `tools/build-mlir-toolchain.sh --out-dir DIR` (it downloads the LLVM
   22.1.8 sources), unpack the archive it writes, and configure with
   `-DMLIR_DIR=<prefix>/lib/cmake/mlir -DLLVM_DIR=<prefix>/lib/cmake/llvm`.
   Another MLIR 22.1.8 build works the same way.

`tools/build-mlir-toolchain.sh` builds the slim, static LLVM/MLIR 22.1.8
distribution that cudaq-pulse links against. The `mlir-toolchain` workflow
runs it for every platform. Pushing a tag named after `release_tag` in
`tools/mlir-toolchain.lock.json` (for example `mlir-toolchain-22.1.8`) builds
the archives, uploads them to that release, and opens a pull request to
`main` from a `toolchain-checksums/<tag>` branch that records their
checksums in the lock file. A pull request the workflow opens starts no CI
(GitHub runs no workflows for its token's pushes), so re-run the checks on
it, or push a commit to its branch, before merging. A manual run with
`publish` enabled uploads too, but leaves committing the printed lock file
to you.

## Continuous integration

Workflows use third-party actions pinned by full commit SHA, with the release
in a trailing comment, and container images pinned by digest;
`tools/check-workflow-pins.py` (a pre-commit hook) rejects tags, and
Dependabot proposes weekly updates for actions and `uv.lock`, each a week
after its release. [zizmor](https://docs.zizmor.sh/) audits the workflows
(a pre-commit hook, and online in CI), and every job sets
`timeout-minutes` (`tests/test_release_tools.py` checks it).

- `ci.yml`: ruff and pre-commit; build, lit, pytest, mypy, and the benchmark
  baseline on Linux x86_64 (Python 3.12 to 3.15), Linux aarch64 (3.12 and
  3.15), and macOS (3.12 to 3.14), and the lit and pytest suites on
  free-threaded CPython 3.14t with the GIL off (configured with
  `-DCUDAQ_PULSE_FREE_THREADED=ON`, which declares the extension safe
  without the GIL, as the free-threaded wheels do). `old-compilers` builds
  and runs lit with GCC 11 and Clang 14 on Ubuntu 22.04. The `gpu` job
  builds with GPU support and runs the GPU tests (see [GPU CI](#gpu-ci)).
- `wheels.yml`: cibuildwheel for the `cudaq-pulse-tools` wheel (`py3-none`)
  first, then the `cudaq-pulse` wheels for `cp312-abi3` (3.12 and newer) and
  `cp314-cp314t` (free-threaded 3.14, which has no stable ABI), on
  manylinux_2_28 (x86_64, aarch64, with the CUDA 12 and CUDA 13 runtimes) and
  macOS arm64 (compiler-only), plus an sdist. Each wheel is size-gated and
  `tools/check-wheel-contents.py` checks both layouts. It then installs
  both distributions from the built wheels alone (no index) in clean
  containers without a CUDA toolkit, and the 3.14t wheels on the runners
  with the GIL required to stay off; on the runners it also loads the
  example pass plugin into the installed `cudaq-pulse-opt`
  (`tools/ci/opt-plugin-test.sh`). `gpu-install-test`
  runs the GPU tests from the wheels (see [GPU CI](#gpu-ci)) and gates the
  release (see [Releases](#releases)). Once everything has passed, the
  `dist` job checks the complete set of files with `tools/check-dist.py` and
  uploads it as the run's one artifact for upload,
  `cudaq-pulse-<version>-dist` (see [Releases](#releases)); the per-platform
  `intermediate-*` artifacts only feed the run's own jobs.
- `gpu-tests.yml`: the GPU tests, called by `ci.yml` and `wheels.yml`.
- `docs.yml`: Sphinx with warnings as errors; deploys `main` to GitHub Pages
  once Pages takes its source from GitHub Actions, and warns until then.
- `property-nightly.yml`: the property tests (every test file that imports
  hypothesis) with random examples, the slow physics checks (`nightly`),
  the perf guard and check-pulse on macOS with Python 3.15, every night and
  on manual dispatch, with the seed in the job summary and the example
  database kept in the Actions cache. GitHub runs scheduled workflows from the default
  branch, so the schedule tests `main`; a manual dispatch tests the branch it
  is dispatched on.

### GPU CI

The GPU tests run on NVIDIA's shared GitHub Actions GPU runners, through
`gpu-tests.yml`. Each test runs in a container that gets the machine's GPU
and driver and nothing else. `tools/ci/gpu-matrix.py` lists the rows,
following what the GPU code varies on:

- `wheels.yml` (`gpu-install-test`) installs the run's wheels with one CUDA
  extra per row, in images that have no CUDA toolkit
  (`install-test.sh --driver-only` fails if one is present), and runs the
  doctor and `pytest -m gpu`. On every push to `main` it runs `[cu12]`
  and `[cu13]` on x86_64 with the newest driver. Release tags and manual
  runs with `gpu` also run both extras on aarch64 and on x86_64 with the
  oldest driver, plus `[cu13]` on the newest Python (the abi3 wheel). On
  the oldest driver, an extra whose minimum driver (CUDA 12.0 for `[cu12]`,
  CUDA 13.0 for `[cu13]`) is newer than the installed one must be refused
  by the doctor with that reason; on any other row an old driver fails the
  test. The free-threaded wheels have no row yet.
- `ci.yml` (`gpu`) builds from source with `CUDAQ_PULSE_ENABLE_GPU=ON` in
  the CUDA 12.9 development image with the `[cu12]` libraries, and runs
  `check-pulse-gpu`, on every push to `main`, on vetted pull
  requests (below) and on manual runs with `gpu`.

The `GPU_RUNNERS` repository variable switches GPU CI on, and names a runner
for each role, as a JSON object:

```json
{"x86_64": "<label>", "x86_64-min-driver": "<label>", "aarch64": "<label>"}
```

`x86_64` has the newest driver, `x86_64-min-driver` the oldest. A value may be
a JSON list of labels, and a bare label means `x86_64` alone. A row whose
role is missing is skipped and listed in the job summary. While
`GPU_RUNNERS` is unset, no GPU job runs or waits for a runner, and
`gpu-gate` requires the manual attestation (see
[GPU verification](#gpu-verification)).

**Pull requests.** The `pull_request` event never runs a GPU job, since a
pull request from a fork controls the workflow files it runs with. Instead,
the [copy-pr-bot](https://github.com/apps/copy-pr-bot) GitHub App
(configured in `.github/copy-pr-bot.yaml`) pushes a copy of the pull request
to a `pull-request/<N>` branch of this repository, and `ci.yml` runs the
`gpu` job there alone. Commits the app already trusts are copied on their
own; for any other, a maintainer first reviews the commit, workflow changes
included, and approves it with the app's `/ok to test` comment. Every new
commit needs approval again. The GPU jobs use no secrets and a
read-only token. The pull request shows the result as the `ci` checks of
its head commit.

**Switching it on** (repository administrators):

1. NVIDIA's runner administrators grant this repository the GPU runners.
2. Set `GPU_RUNNERS` to their labels (Settings, *Secrets and variables*,
   *Actions*, *Variables*).
3. Install the copy-pr-bot app on the repository, for pull requests.
4. Run `ci` and `wheels` manually with `gpu` on `main`, and check that
   every row ran on a GPU and passed.

Deleting `GPU_RUNNERS` switches everything off again. Uninstalling the app
stops the pull-request copies.

To run the wheels' GPU tests on a machine with an NVIDIA GPU, put the wheels
in `wheelhouse/` and run `bash tools/ci/install-test.sh gpu-cu12` (or
`gpu-cu13`, or `gpu` for both).

## Versioning and stability

Versions follow [PEP 440](https://peps.python.org/pep-0440/) in the shape of
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`, with
pre-releases `X.Y.Z{a,b,rc}N` and post releases `X.Y.Z.postN`. Both
distributions take the version from the `v*` git tags through setuptools-scm:
a build of a tag is exactly the tag's version, and any other build is a dev
release, such as `0.1.1.dev3` three commits after `v0.1.0` or `0.1.0rc2.dev1`
one commit after `v0.1.0rc1` (`0.1.1.dev0` for a tag with local changes).
Versions have no local segment (`+g<commit>`), which package indexes reject.

- **What is covered.** The public API is the names in the `__all__` of
  `cudaq_pulse` and of its documented submodules (`cudaq_pulse.ops`,
  `cudaq_pulse.compiler`, `cudaq_pulse.quantum_systems`,
  `cudaq_pulse.quantum_systems.transmon_model`, `cudaq_pulse.runtime`,
  `cudaq_pulse.runtime.simulate`, `cudaq_pulse.viz`, `cudaq_pulse.controller`,
  `cudaq_pulse.backends`, `cudaq_pulse.classical`,
  `cudaq_pulse.classical.wire`, `cudaq_pulse.classical.emulation`,
  `cudaq_pulse.classical.calibration`, `cudaq_pulse.classical.loop` and
  `cudaq_pulse.quantum_systems.drift`), their
  documented behavior, the `cudaq_pulse.backends` entry-point group, the
  controller export format (whose version is `cudaq_pulse.controller.FORMAT`),
  the documented environment variables, and the `cudaq-pulse-opt` and
  `python -m cudaq_pulse.doctor` command lines. Modules and names that start
  with an underscore, the native extension, the C++ headers, and the textual
  form of the MLIR dialects are internal and may change in any release.
  Numerical results are covered by the tests' tolerances, not bit for bit.
- **While the major version is 0** (the research preview), a minor release
  may break the public API; a patch release fixes bugs and does not break it
  on purpose. From 1.0, only a major release breaks it.
- **Deprecation.** Where it is practical, a name or behavior that is going
  away warns with `DeprecationWarning` for at least one minor release before
  it is removed. Every break and deprecation is listed under **Changed**,
  **Deprecated** or **Removed** in `CHANGELOG.md`.
- **Python versions.** CPython 3.12 and newer. A new CPython is supported
  from its first release, and dropping one happens only in a minor release.
- **Post releases** (`X.Y.Z.postN`) fix packaging or metadata only, never
  code; they need their own dated section in `CHANGELOG.md`.

### Withdrawing a broken release

Release assets are write-once and tags are never moved or deleted, so a
broken release is withdrawn, not replaced:

1. Open an issue that describes the problem and who is affected.
2. Put a warning at the top of the GitHub release's notes that links the
   issue and names the release to use instead. Keep the assets: deleting
   them breaks pinned installs and reproducibility.
3. Release the fix as a new version: `X.Y.(Z+1)` for a code fix, or
   `X.Y.Z.postN` for a packaging-only fix, with a `CHANGELOG.md` entry that
   names the withdrawn release.
4. Also yank the broken version on PyPI
   ([PEP 592](https://peps.python.org/pep-0592/)) with the issue link as
   the reason. Yanking hides it from new resolutions but keeps exact pins
   working; never delete files from PyPI.

A vulnerability is handled through [SECURITY.md](SECURITY.md) first, and the
withdrawal follows the coordinated disclosure.

## Releases

Changes reach `main` only through pull requests, and releases are tagged from
`main`. CI runs on every pull request and every push to `main`, but only a
`v*` tag publishes.
A release has two distributions, each with a wheel per platform:
`cudaq-pulse-tools`, with the MLIR driver, uploaded first, and `cudaq-pulse`,
which requires it at the same version, with the sdist. An index needs both, in
that order. A release is published as a GitHub release first. Once index
publishing is enabled (below), the same wheels then go to TestPyPI for a
pre-release (`vX.Y.Z{a,b,rc}N`) or to PyPI for a final release; until then,
releases are on GitHub only. The sdist is only on the GitHub release.

Every run of `wheels.yml` whose builds and install tests pass uploads one
artifact with exactly the files a release publishes,
`cudaq-pulse-<version>-dist` (for example `cudaq-pulse-0.1.0rc1-dist` for
the tag `v0.1.0rc1`, `cudaq-pulse-0.1.1.dev3-dist` for a push to a branch):

```text
dist/          the cudaq-pulse wheels (cp312-abi3 and cp314-cp314t) and the
               cudaq-pulse-tools wheel (py3-none) for each of
               manylinux_2_28 x86_64, manylinux_2_28 aarch64 and macOS arm64,
               and the cudaq-pulse sdist: ten files of one version
SHA256SUMS     their SHA-256 digests (`sha256sum --check SHA256SUMS`)
MANIFEST.json  each file's distribution, version, tags, size and digest
```

The `dist` job writes it with `tools/check-dist.py`, which fails the run if
a file is missing or unexpected, if the files do not all have the run's
version (a release version on a tag, a dev version otherwise, never a local
version), if one is over 90 MiB, lacks its license files or fails
`twine check --strict`. The release jobs check the same artifact against
its `SHA256SUMS`: the GitHub release gets exactly its `dist/`, and TestPyPI
or PyPI exactly its wheels.

**Rehearse a release** without a tag: run `wheels` manually on a branch
(*Run workflow*) with `version` set, for example `0.1.0rc1`. Every file is
built as that version and the run's artifact is
`cudaq-pulse-0.1.0rc1-rehearsal-dist`; nothing is published. Check a
downloaded bundle locally with
`python tools/check-dist.py --version 0.1.0rc1 --release --verify-sums <dir>`
(with `twine` installed).

**Upload by hand** if trusted publishing (below) is not used: download the
`cudaq-pulse-<version>-dist` artifact of the tag's run that published the
GitHub release (or the same files from the release's assets), unpack it,
check it, and upload the wheels of `dist/`, `cudaq-pulse-tools` first (the
sdist stays on the GitHub release):

```bash
sha256sum --check SHA256SUMS
twine upload dist/cudaq_pulse_tools-*.whl
twine upload dist/cudaq_pulse-*.whl
```

Add `--repository testpypi` for TestPyPI.

1. Tag a release candidate, for example `v0.1.0rc1`. `wheels.yml` builds and
   tests the wheels and the sdist, publishes them as a GitHub pre-release and,
   with index publishing enabled, uploads the wheels to TestPyPI.
2. Date the version's section in `CHANGELOG.md` (`## [0.1.0] - YYYY-MM-DD`)
   and update `version` and `date-released` in `CITATION.cff` in a pull request    to `main`. The next release's section,
   `## [X.Y.Z] - Unreleased`, starts with the first change after the tag.
3. Tag `v0.1.0` on `main`; with index publishing enabled, its wheels go to
   PyPI.

Release notes come from `CHANGELOG.md`, not from the commit history:
`tools/release-notes.py` prints the section for the version, and a
pre-release uses the section of the release it precedes. Add user-visible
changes to the top section, the next release's `## [X.Y.Z] - Unreleased`,
in the pull request that makes them. It is the only undated section, so
everything in it reaches the release notes.

Before anything is released, the `release-meta` job checks the tag with
`tools/release-meta.py`: it must name a version `X.Y.Z`, `X.Y.Z.postN` or
`X.Y.Z{a,b,rc}N` that the setuptools-scm `tag_regex` accepts, a final
version must be on `main`, `CHANGELOG.md` must have a dated, non-empty
section for it, and the release must not exist yet. The release is created as a
draft, the bundle is checked against the tag, every file gets a
signed build-provenance attestation (check one with
`gh attestation verify <file> --repo NVIDIA/cudaq-pulse`), the files, their
`SHA256SUMS` and `MANIFEST.json` are uploaded once (never replaced), and only
then is the release published. The MLIR toolchain assets are write-once too.

The `testpypi` and `pypi` jobs upload the same checked wheels with trusted
publishing: the index accepts a short-lived token that GitHub issues to this
repository's `wheels.yml` running in the `testpypi` or `pypi` environment,
so the repository holds no index API token. Each of the two project names
needs a trusted publisher on each index for that workflow and environment,
and the environments' required reviewers approve every upload. GitHub
creates an environment that does not exist yet without any protection, so
the two jobs also need the repository variable `PYPI_PUBLISHING` set to
`true`, which is set only after the trusted publishers and the protected
environments exist; without it, a release is GitHub-only. The jobs run only
for a `v*` tag of NVIDIA/cudaq-pulse, after `github-release` has published
it: pushes to `main` and manual runs on a branch never reach them. A
re-run of a failed upload skips the files already on the index.

### GPU verification

No release is published without GPU verification of the tagged commit; the
`gpu-gate` job in `wheels.yml` enforces it, and it fails closed:

- **With GPU runners** (`GPU_RUNNERS` set; see [GPU CI](#gpu-ci)), the
  tag's run executes every `gpu-install-test` row and publishes only if they
  all pass. A failed GPU run cannot be overridden: fix the problem and tag a
  new version.
- **Without GPU runners**, the tag's run builds and tests everything else
  and then stops at `gpu-gate`. To release it:
  1. Run the GPU test suite on the tagged commit on a machine with an
     NVIDIA GPU: build it as in [GPU builds](#gpu-builds) and run
     `check-pulse` and `check-pulse-gpu`, or unpack the tag run's
     `cudaq-pulse-<version>-dist` artifact into `bundle/` in a checkout of
     the tag and run
     `bash tools/ci/install-test.sh --wheelhouse bundle/dist gpu`. Every test
     must pass, with no GPU test skipped.
  2. Run the `wheels` workflow manually on the tag (*Run workflow*, *Use
     workflow from*: the tag) with `gpu-verified-sha` set to the full commit
     SHA the tag points to (`git rev-parse vX.Y.Z^{commit}`). This is the
     maintainer's attestation that step 1 passed on that commit; the run's
     summary records who made it.

  The attestation counts only on a manual run, only when it names the exact
  commit being released, and only when `gpu-install-test` did not run.

## Developer Certificate of Origin

Every commit in a pull request must be signed off. Signing off means adding
a `Signed-off-by:` line with your name and email address at the end of the
commit message:

```text
Signed-off-by: Jane Doe <jane.doe@example.com>
```

With it you certify that you can contribute the change under the project's
open source license, as the Developer Certificate of Origin 1.1 states:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Use your real name and the email address you commit with. `git commit -s`
(`--signoff`) adds the line from `user.name` and `user.email`:

```bash
git config user.name "Jane Doe"
git config user.email jane.doe@example.com
git commit -s -m "Describe the change"
```

To sign off commits you already made, amend the last one, or rebase the
branch onto the one your pull request targets (`<base>`) with every commit
signed off, then update the pull request:

```bash
git commit --amend -s --no-edit         # the last commit
git rebase --signoff <base>             # every commit since <base>
git push --force-with-lease
```
