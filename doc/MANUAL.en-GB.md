# User manual

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Contents: [installation](#installation), [local build](#local-build), [deployment](#deployment),
[chat](#chat), [monitor](#monitor), [models and tools](#models), [choosing models](#settings-models), [model page](#model-page), [supported models](#supported-models), [API](#api),
[request queue](#request-queue), [security limits](#security-limits), [Vulkan](#vulkan), [ROCm](#rocm), [performance](#performance), [removed compatibility](#removed-compatibility),
[maintenance](#maintenance), [licence](#license).

<a id="installation"></a>
## Installation and first launch

Install as root on the production host. Debian/Ubuntu accepts the installer
directly on standard input:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

The installer deploys in a single step: it compiles `mgs`, installs it in
`/usr/local/bin/mgs` and configures the inference and web services
([deployment](#deployment)). No neighbouring files need to be downloaded first. The entry script downloads the
complete `main` source archive from [this repository](https://github.com/nipegun/moe-gguf-server/), checks
it and synchronises a managed copy in `/opt/moe-gguf-server-source`. Downloads
and build caches stay under the project's `_/temp/`. Curl and CA certificates
are needed to fetch the entry script; missing bootstrap dependencies are then
installed automatically.

Pass build and web options with:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com --cpu --jobs 4
```

The managed copy always lives in `/opt/moe-gguf-server-source`, owned by root with
mode 700. The installer only adopts it while it is empty or reuses it when it bears
its source-origin marker; unrelated contents are rejected. A lock prevents concurrent installation into
that copy. Subsequent `curl | bash` runs synchronise source files, including files
removed from `main`, while preserving `_/` and build caches. API keys, models and
certificates live outside that source tree and are kept. A failed
download, build or option check does not install a new executable; rerun the same command to
retry. No Git operations are required.

A standalone downloaded script uses the same remote mode. Running
`bash deploy/install-update-reinstall-debian.sh` from a complete local checkout
builds that checkout without downloading sources. Arguments after `bash -s --`
are forwarded to the build installer. `--help` requires neither root nor a source
download. `--no-system-deps` also disables bootstrap package installation: curl,
CA certificates, tar, gzip, rsync, util-linux and coreutils must already exist.
Installer subprocesses cannot consume the script from the pipe; all phases share
one `/root/webapp-install.log` output stream.

The web options are checked before the build, so a mistake does not surface after
a long compilation: `--domain NAME` (by default the output of `hostname -f`, or of
`hostname` or `uname -n` when that is not a valid name), `--certificate PATH` and
`--private-key PATH` (always together) and `--proxy-ip IP` (see
[deployment](#deployment)). `/opt/moe-gguf-server-source/deploy/configure-web.sh`
remains available to change the web configuration later without rebuilding.

Alpine still uses a complete local checkout and
`bash deploy/install-update-reinstall-alpine.sh`, with Bash installed first; it
only builds for the CPU. On Debian, choose the backend:

- CUDA (`--cuda`, the default) for NVIDIA GPUs. The CUDA Toolkit and the NVIDIA
  driver must already be installed; the installer adds the NCCL packages that
  match that CUDA version (see below). `--no-nccl` disables NCCL explicitly.
- ROCm (`--rocm`) for AMD GPUs supported by ROCm; `--rocm-targets LIST` chooses the
  GPU targets. ROCm must already be installed ([ROCm](#rocm)).
- Vulkan (`--vulkan`) for AMD, Intel or NVIDIA GPUs ([Vulkan](#vulkan)).
- CPU only (`--cpu`), for machines without a GPU.

`--no-system-deps` leaves package management to the administrator.

The local UI needs Node.js 20.19+ on 20.x or 22.12+. `--no-ui` omits it. Other
options are `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST` and `--no-openssl`. `CUDACXX` selects the CUDA compiler
and `HIPCXX` the ROCm clang compiler. Build and staging directories must be distinct,
non-overlapping descendants of the project's `_/temp/` directory.

The installer adds matching NCCL packages and NVIDIA's repository when needed,
but never installs or upgrades CUDA Toolkit, ROCm or GPU drivers. It links the internal
libraries statically and installs `/usr/local/bin/mgs`. Neither
installer uses or installs sudo.

On Debian/Ubuntu, `libnccl2` and `libnccl-dev` are installed together at the same
exact version selected for the detected CUDA release. This APT command uses
`--allow-downgrades` so a newer installed NCCL package does not block installation
of the selected version. The flag is only passed to the NCCL installation command.

If the distribution ships an older Node.js, install a supported version first.
The UI is always compiled locally; the build cannot download prebuilt upstream
assets, which lack this project's translations and API routes. A failed UI build
stops installation.

From a complete local checkout, choose a CPU build or list installation options:

```bash
bash deploy/install-update-reinstall-debian.sh --domain ai.example.com --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

On Alpine, install Bash with `apk add --no-cache bash` before running its installer.

Run a single model with:

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

No other option is needed: `mgs` chooses the settings that give the most inference speed on the machine it runs
on. Automatic fitting places the model on the GPU and, when system memory allows, keeps the MoE experts in RAM so
that the free VRAM becomes the [MoE expert cache](#expert-cache); it also chooses the
[micro-batch and the context size](#context-size). The number of CPU threads follows the processor
([CPU threads](#threads)), and the MTP layers found in the GGUF are activated and kept in VRAM
([MTP](#mtp-vram)). Any option given explicitly (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`,
`--override-tensor`…) prevails over the automatic choice. The startup log shows what was chosen.

Run `mgs` with your own user: it never needs root. Started as root, it logs the
warning `mgs is running as root: run it with an unprivileged user instead` (the
local shell MCP server would give the model a root shell) and keeps starting.

Open the HTTPS URL printed at startup, normally `https://127.0.0.1:11443/`.
Loading a large model may take time. The web
interface and API documentation remain available while model requests report
loading. `--help` lists the inherited engine options. GGUF files, GPU memory,
context length and model-specific projectors must match the intended workload.

The server code enables HTTPS by default for every build and installation method.
It binds the first available HTTPS port from 11443, then an HTTP redirect port
from 11080, skipping the chosen HTTPS port. HTTP responds with a permanent 308
redirect to the actual HTTPS port, preserving the requested host, path and query.
The log prints the selected URLs; port conflicts advance each sequence independently.
`--port N` fixes the main port and fails if occupied; `--port 0` lets the OS select
it. `--http-port N` changes the start of the redirect port search.
The corresponding environment variables are `MGS_ARG_PORT` and `MGS_ARG_HTTP_PORT`.

Up to version 0.1.34 the search stopped at the second port: an `mgs` started while 11443 was in use (for example by another `mgs`) exited with `couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 11444` instead of taking the next free port.

Without certificate files, the executable creates a self-signed certificate and its
private key the first time it starts, without requiring an external `openssl`
command, and saves them in `~/.mgs/certificates/` (`fullchain.pem` and
`privkey.pem`, the key readable only by you). Every later run, and every new build,
reuses them, so the browser keeps its trust exception and needs no reload; a new
pair is only created when those files are missing, do not match or the certificate
has expired (after 365 days). The certificate covers `localhost`, the loopback
addresses, the host given with `--host`, and the machine name and network addresses
it had when it was created; to stop the browser warning altogether, import
`~/.mgs/certificates/fullchain.pem` as trusted, and delete both files to create a
new one. If that folder cannot be written, each run generates its own certificate in
memory. Supply both `--ssl-cert-file /path/fullchain.pem` and `--ssl-key-file
/path/privkey.pem` to use a certificate trusted by your clients. A missing or invalid pair stops
startup. Builds made with `--no-openssl` cannot serve HTTPS and require explicit
`--http` (or `MGS_ARG_HTTPS=false`); plain HTTP starts its port search at 11080.
Router children use explicit HTTP on private loopback ports. Unix sockets and
Vertex AI's `AIP_MODE=PREDICTION` retain their HTTP transport.

CMake configuration lives in `build/CMakeLists.txt`. Both installation and local
build scripts include `build/` in their source copy and select it automatically.
The installers reset generated CMake configuration if an existing cache points
to another source directory. Local builds always start in a fresh directory under
`/tmp/` and remove it on exit. For direct CMake use, run
`cmake -S build -B _/temp/cmake` from the project root; the usual installer and
local build commands remain the same. HTTPS uses the system OpenSSL by default
(`MGS_OPENSSL=ON`); in a direct CMake build, `-DMGS_BUILD_BORINGSSL=ON` or
`-DMGS_BUILD_LIBRESSL=ON` downloads and builds BoringSSL or LibreSSL instead, and
`-DMGS_OPENSSL=OFF` without either builds the server without HTTPS.

<a id="local-build"></a>
## Local build for manual use

`build/build-local.sh` compiles the project on your own computer and installs a
self-contained copy for your user that you launch by hand. It installs no services, starts
nothing in the background, never uses root or sudo and never installs packages.
Run it as your regular user from any folder:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

The one-liner downloads the `main` source archive without Git into the private
build folder under `/tmp`. The archive and extracted sources are removed with
the other temporary build files on exit. A standalone copy of the script works
the same way. From a complete local project copy, build your local sources with:

```bash
bash build/build-local.sh
```

Running this script by its absolute path also uses the complete project that
contains it. The `Using local sources: ...` message confirms which folder is
being compiled, including its interface translations. If the run ends with
`Error: Compilation failed.`, no new executable has been installed; an existing
installation keeps its previous interface. Read the first compiler error above
`ninja: build stopped`, correct its cause and rerun the same script. Once the
build and installation succeed, restart `mgs` and reload the web interface.

Errors such as `std::vector` having no member named `mPushBack` or
`no declaration matches` in `subproc.cpp` indicate inconsistent source names.
Use the corrected source files and rerun the build. The interface build also
requires its configuration imports to match the names exported by the source
modules; a missing export is reported by the npm/Vite step.

If SvelteKit reports `Page options are ignored when router.type === 'hash'`
and names `fLoad`, the chat routes must export the framework's exact `load`
name. The corrected routes keep API-key validation. The build runs
`svelte-kit sync` before generating assets and compiling the interface, and
stops if synchronisation fails. A failure at `Provisioning UI assets` still
prevents installation even when the C++ libraries have already compiled.

To pass options through the one-liner, use `bash -s --`; for example, a CPU build:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` shows the options without downloading the source archive or compiling.

The result is installed for your user under the `~/.local` prefix:

| path | content |
| --- | --- |
| `~/.local/bin/mgs` | The program you run: a small static launcher that finds `../lib/mgs/` relative to itself and starts the server through the bundled loader. |
| `~/.local/lib/mgs/` | The server binary (`mgs`, with the web interface and API documentation embedded) and every shared library it needs: the glibc dynamic loader and C library, the C++ runtime, OpenMP, OpenSSL, the glibc name-service modules and, in CUDA builds, the CUDA runtime, cuBLAS and NCCL; in ROCm builds, the ROCm libraries (HIP, hipBLAS, rocBLAS, hipBLASLt…) and the GPU kernel folders of rocBLAS and hipBLASLt (`rocblas/library`, `hipblaslt/library`). |
| `~/.local/share/doc/mgs/licenses/` | The project's PolyForm NonCommercial 1.0.0 licence (`LICENSE`) and the third-party licence notices copied from the source tree's `build/licenses/` (llama.cpp, FreeToken, Swagger UI, cpp-httplib and nlohmann/json). |

Only the user who runs the script can use this installation. Everything is resolved relative to where the launcher
is: it runs `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` from its prefix, and the server and its libraries search
`lib/mgs/` before anything else, including `LD_LIBRARY_PATH`. The only library still taken from the system is the
NVIDIA driver (`libcuda.so.1` and the `libnvidia-*` libraries it loads), because it must match the running kernel
module; with `--vulkan`, the Vulkan driver of the system ([Vulkan](#vulkan)); with `--rocm`, only the amdgpu kernel
driver (`/dev/kfd`, `/dev/dri`) is used from the system, and the launcher points rocBLAS and hipBLASLt to the kernel
folders bundled in `lib/mgs/` ([ROCm](#rocm)). At the end the script lists, through the bundled loader, the libraries the server would load, fails if any
comes from outside `lib/mgs/`, and runs `--version` through the launcher.

Debian adds `~/.local/bin` to the `PATH` at login once that folder exists (through `~/.profile`), so after logging in
again you can run `mgs` by name; the script prints a reminder while it is not in the `PATH`. Launch it whenever you
want and stop it with Ctrl+C:

```bash
mgs --model /path/model.gguf
```

The full path works too: `~/.local/bin/mgs --model /path/model.gguf`.

Open `https://127.0.0.1:11443/` for the chat and `https://127.0.0.1:11443/api/doc/` for
the API documentation. Every option in this manual applies. Run `mgs` without `--model` to start the
multi-model router and choose the models folder in Settings → Models
([choosing and loading models](#settings-models)).

Requirements, which you install yourself as root because the script only checks
them: `cmake`, a C/C++ compiler and the static C library (`build-essential`, which
pulls in `libc6-dev`), `patchelf`, `libssl-dev` (or
`--no-openssl`), Node.js 20.19+/22.12+ with npm (or `--no-ui`) and the toolkit of the
chosen backend: for CUDA (the default), an existing CUDA Toolkit and NCCL with its
development files (`libnccl2` and `libnccl-dev`, or `--no-nccl`); for `--rocm`, an
existing ROCm ([ROCm](#rocm)); for `--vulkan`, the Vulkan packages ([Vulkan](#vulkan));
`--cpu` needs none. `nvcc` is taken from `CUDACXX`, the `PATH` or
`/usr/local/cuda/bin/nvcc`. Ninja is used when installed.
Basic utilities include `tar`, `coreutils` and `util-linux`. Remote builds also
require `curl`, `ca-certificates` and `gzip`; no dependencies are installed automatically.

Options: `--prefix PATH` (another installation prefix), `--jobs N`, `--cuda`, `--rocm`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--rocm-targets LIST`, `--no-ui`, `--no-openssl`, `--no-nccl`
and `--help`. `CUDACXX` selects the CUDA compiler and `HIPCXX` the ROCm clang compiler.

Good to know:

- The build is optimised for this computer's CPU and for the GPUs present while
  compiling. Rebuild on each computer instead of copying the installation.
- The installation is relocatable as a prefix: move `bin/mgs` and `lib/mgs/` together, keeping their `bin/` and
  `lib/` levels. Always start `bin/mgs`; `lib/mgs/mgs` is internal and must not be run directly, because it would
  bypass the bundled loader.
- Rerunning the script updates the installation. It replaces only `bin/mgs`, `lib/mgs/` and
  `share/doc/mgs/licenses/`. A copy that is already running keeps working; restart it to use the new build.
- The script refuses a prefix where `bin/mgs`, `lib/mgs/` or that licenses folder already exist unless it created
  them; it leaves a hidden `.moe-gguf-server-bundle` marker in `lib/mgs/`.
- `--prefix PATH` installs under another prefix that your user can write to; it cannot be `/` or a folder inside the
  project.
- Staged sources, build files, caches, the assembled installation and the log use a
  private `/tmp/moe-gguf-server-build.XXXXXXXX/` folder. After copying the result
  to the prefix, the script removes that folder. It also cleans up on errors
  and handled interruptions. Each run compiles from scratch, without writing into
  the source tree or `/opt/`. Local builds do not accept `--build-dir`.
- The prefix is locked while copying and replacing completed artefacts,
  preventing simultaneous updates of the same installation.
- A CUDA build needs the NVIDIA driver to start; use `--rocm` or `--vulkan` on computers with another GPU and
  `--cpu` on computers without a GPU.

<a id="vulkan"></a>
## Vulkan (AMD, Intel and NVIDIA GPUs)

Besides CUDA (the default) and CPU, both `build/build-local.sh` and the Debian installer build the Vulkan backend
with `--vulkan`, for GPUs without CUDA such as AMD Radeon cards. Install the build dependencies as root first:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Then, as your regular user:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

The Debian installer takes the same option and installs those Vulkan packages by itself:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` shows the GPU, for example `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- The local installation bundles the Vulkan loader (`libvulkan.so.1`) like any other library, but the Vulkan driver
  comes from the system: the loader reads the drivers registered in `/usr/share/vulkan/icd.d/` (Mesa RADV for AMD,
  ANV for Intel, the NVIDIA driver) and these load their own system libraries, because they must match the kernel
  and the system. Build on the Debian release that will run the binary.
- The user that runs `mgs` needs the GPU render node (`/dev/dri/renderD*`). A user logged into the desktop already
  has it; other users need the `render` group (and `video`). `configure-web.sh` adds the service account to both.
- MTP in VRAM, the MoE expert cache, the expert prefetch, `--fit` and the Monitor work as with CUDA. On AMD the
  Monitor reads usage, VRAM and temperature from sysfs.
- `mgs` only counts the real VRAM (the device-local memory heaps) as GPU memory, never GTT. GTT is the part of the
  system RAM (normally half of it) that an AMD or Intel GPU can map: it is not extra memory, and the GPU reads it
  over PCIe, many times slower than its VRAM. The MTP layers, the dense weights, the KV cache and the expert cache are
  allocated in VRAM; the experts that do not fit stay in normal RAM, where the CPU computes them.
- An integrated GPU (the graphics of an AMD or Intel processor) has no VRAM of its own: every memory heap it reports
  is system RAM. `mgs` counts that memory and the RAM as one budget and does not create the expert cache on it; see
  [integrated GPUs](#integrated-gpu).
- `mgs` checks where the kernel really keeps that memory (`/proc/self/fdinfo`, published by amdgpu, i915 and xe). If
  another program fills the VRAM and the driver moves part of it to GTT, the log and the Monitor say so, and the
  Monitor shows the real location (VRAM and GTT) of every discrete AMD or Intel GPU (an integrated GPU has no VRAM
  of its own, so it is not checked). The expert cache is allocated with a
  lower memory priority (`VK_EXT_memory_priority`, used whenever the GPU offers it, with no variable to turn it on),
  so the driver moves it before the MTP layers, the weights or the KV cache. Without resizable BAR, device buffers
  also skip the small VRAM window the CPU can map (256 MiB on an RX 580), which the desktop shares and the driver
  empties into GTT first, and that window does not count as free memory for them.
- Only CUDA can pin the model pages that stay in RAM, so with Vulkan the experts reach the GPU through pageable
  memory. `mgs` copies them through a small staging ring and queues the GPU copy without waiting for it, so the GPU
  and the CPU keep working while experts are uploaded: on an RTX 4060 Ti with Qwen3.6-35B-A3B Q8_0 this raised the
  prompt from 455 to 515 tokens/s and generation from 20.6 to 23.2 tokens/s. `--load-mode none` copies the experts
  into pinned host memory instead; before the staging ring, on the same card with the Q4_K model, it raised expert
  uploads from 2.7 to 13.1 GB/s and generation from 27.5 to 36.6 tokens/s. On
  AMD and Intel that pinned memory counts against GTT: when the experts do not fit in it, loading fails with
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), as the 18.9 GB of experts of that
  model did with 16.8 GB of GTT on an RX 580. Use it only when the experts fit in GTT and in the free RAM.
- With an NVIDIA GPU through Vulkan, and with BF16 experts on any GPU, the expert products first convert their
  input in a work buffer. That buffer used to take 512 KiB per prompt token, so prompts of a few thousand tokens
  failed with `ErrorOutOfDeviceMemory` once the expert cache had filled the VRAM, unless `--moe-cache-reserve` was
  raised. It now takes a few KiB per token (32 MiB for a 4096-token micro-batch of Qwen3.6-35B-A3B), well within
  the default reserve. On an RTX 4060 Ti with Qwen3.6-35B-A3B Q8_0 this also made the prompt faster, from 515 to
  549 tokens/s.
- Size the model to the machine: the whole GGUF should fit in the free RAM plus VRAM, minus the KV cache and the
  compute buffers. With 8 GB of VRAM and 16 GB of RAM that means about 18–20 GB of weights; larger files are read
  from disk again and again and generation becomes very slow. The 21.6 GB Q4_K above reached 7.8 tokens/s on an
  RX 580 with 15 GB of RAM and 15.9 tokens/s once the machine had 31 GB (10.0 with `--spec-type none`).

<a id="rocm"></a>
## ROCm (AMD GPUs)

Both `build/build-local.sh` and the Debian installer build the ROCm (HIP) backend with `--rocm`, for AMD GPUs that
the installed ROCm release supports. Neither script installs ROCm: install ROCm 6.1 or later yourself as root, with
HIP, hipBLAS and rocBLAS and their development files. For the local build, also install the common build
dependencies as root (the Debian installer installs them by itself):

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Then, as your regular user:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

The Debian installer takes the same option:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --rocm
```

- The scripts find ROCm through `hipconfig` (in the `PATH` or `/opt/rocm/bin/hipconfig`) and use the clang compiler
  it names; `HIPCXX` selects another one. A failed configuration names what is missing: HIP 6.1 or later with the
  hipBLAS and rocBLAS development files.
- The GPU targets are those of the GPUs present while compiling, read with `amdgpu-arch` or `rocminfo`. Without a
  detected GPU, or to build for others, pass `--rocm-targets` with a comma-separated list, for example
  `--rocm-targets gfx1100,gfx1030`.
- Current ROCm releases do not support GCN GPUs up to Polaris (`gfx6xx` to `gfx8xx`, such as the Radeon RX 580,
  `gfx803`): the scripts warn about those targets; use `--vulkan` for them.
- The local installation copies every ROCm library (HIP, hipBLAS, rocBLAS, hipBLASLt…) to `lib/mgs/`, together with
  the GPU kernel folders that rocBLAS and hipBLASLt load at run time (`rocblas/library`, `hipblaslt/library`); the
  launcher points both libraries to those copies (`ROCBLAS_TENSILE_LIBPATH`, `HIPBLASLT_TENSILE_LIBPATH`), so they
  never read the kernels of another ROCm version. Only the amdgpu kernel driver comes from the system. The kernel
  folders take gigabytes.
- The `/usr/local/bin/mgs` of the Debian installer loads the ROCm libraries of the system, through the run path
  recorded while compiling; when some cannot be found, the installer stops and asks to add the `lib` folder of ROCm
  to `/etc/ld.so.conf.d/` and run `ldconfig`.
- The user that runs `mgs` needs `/dev/kfd` and `/dev/dri`, that is, the `render` and `video` groups; the installer
  adds the service account to both.

<a id="deployment"></a>
## Production deployment

The installer ([installation](#installation)) builds, installs and configures the
web services in one run:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com
```

Without `--domain`, the domain is the output of `hostname -f` (or of `hostname` or
`uname -n` when that is not a valid name). The domain must resolve to the production
host. To change the web configuration later without rebuilding, run as root, from the
source directory (`/opt/moe-gguf-server-source` for remote installs):

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

`configure-web.sh` takes the same web options as the installer (`--domain`,
`--certificate`, `--private-key`, `--proxy-ip`), plus `--binary PATH` for a nondefault
executable location, `--no-system-deps` and `--check`, which only validates the
options. Every run rewrites the whole configuration from the options it receives:
rerunning it without `--proxy-ip` removes the proxy, and without `--domain` the
domain returns to that of `hostname -f`.

The `moe-gguf-server-web`
service runs a dedicated Apache instance as `moe-gguf-server`, without root.
Every startup selects the first available HTTP port starting at 11080 and HTTPS
port starting at 11443. Each sequence advances independently up to 65535, without
assigning the same port to both protocols. It inspects IPv4/IPv6 TCP listeners
and retries if Apache detects a port becoming occupied after selection. Other
configuration failures stop startup instead of being mistaken for port conflicts.

HTTP permanently redirects to the selected HTTPS port, preserving path and query.
The redirect is not cached. The inference service (`moe-gguf-server`, also running as
`moe-gguf-server` and started with `--http`) listens on no TCP port: it listens on the
Unix socket `/opt/moe-gguf-server/_/temp/web/mgs.sock`, which only that account can
open, and Apache forwards the requests to it. Before listening, `mgs` deletes a socket
file left by a server that is no longer running, but never the socket of a server that
still answers. Installation requires root;
neither service needs to run as root. No HAProxy is configured and no shared
Apache sites are imported or modified.

Stopping the inference service (`systemctl stop moe-gguf-server` or `systemctl restart moe-gguf-server`), like Ctrl+C or a `SIGTERM` when you run `mgs` yourself, ends it in order within a couple of seconds, also in the middle of a long prompt: requests in progress get a 503 error (`The server is shutting down.`) or, when streamed, their stream ends. A second Ctrl+C or `SIGTERM` ends the process at once, and while the model is loading the first one already does.

Actual URLs are published after both listeners have opened:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

For example, HTTP 11081 and HTTPS 11445 redirect to
`https://ai.example.com:11445/`. The file is removed on shutdown and regenerated
at startup; a restart may choose different ports. Clients, firewalls and any
external proxy must use the published values. The port is part of the
[browser origin](https://developer.mozilla.org/en-US/docs/Glossary/Origin), so a
different HTTPS port has separate local history/settings. Existing conversations
remain in their original origin; export/import can transfer them.

The system account has no interactive password. Models are stored in
`/opt/moe-gguf-server/models`, and the program's configuration lives in the home of the
service account, `/opt/moe-gguf-server/.mgs/config/`: the API key (`api.key`), the web
domain (`web-domain`), the trusted proxy (`web-proxy-ip`, only with `--proxy-ip`), the
Apache template (`apache2.conf.in`) and the settings of the
interface (`ui-settings.json`), all owned by `moe-gguf-server` with mode 0600.
The installation output is recorded in `/root/webapp-install.log`, and
`/root/webapp-credentials.txt` holds every credential the deployment creates: the
service account (`moe-gguf-server`, without a password and with login disabled) and
the API key. Both files are mode 600. Reconfiguration preserves existing
keys, models and certificates.

Use your own TLS certificate with:

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

The certificate lives in `/opt/moe-gguf-server/certificates/` (`fullchain.pem` and
`privkey.pem`, owned by the `moe-gguf-server` account, the key readable only by it).
Without supplied files, the script keeps the certificate already there or, the first
time, creates a self-signed one, which every later reinstall reuses, so browsers
keep their trust exception. Trust it on your client or supply a certificate your
clients already trust. Supplied files are copied to that folder; certificate renewal
must refresh those copies and restart `moe-gguf-server-web`. The script does not obtain or renew Let's Encrypt certificates.
Use the same command after renewing source certificates.

When a reverse proxy (HAProxy, nginx…) stands in front of Apache, give its address
with `--proxy-ip` (IPv4 or IPv6), to the installer or to `configure-web.sh`:

```bash
bash deploy/configure-web.sh --domain ai.example.com --proxy-ip 10.7.10.1
```

Apache then loads `mod_remoteip`. On HTTPS it requires the PROXY protocol header
only from that address (`RemoteIPProxyProtocol On`, with every other address listed in
`RemoteIPProxyProtocolExceptions`), so the other machines keep connecting directly; on
HTTP it trusts the `X-Forwarded-For` header of that address (`RemoteIPHeader`,
`RemoteIPTrustedProxy`). The access logs record the real client address. Without
`--proxy-ip`, no proxy is configured.

The service uses router mode. Put readable GGUF models in
`/opt/moe-gguf-server/models`, or use model management in the interface. For
specialised command-line options, use a systemd override on Debian/Ubuntu or the
OpenRC service configuration on Alpine. Keep the inference service on its Unix
socket. The web configurator regenerates its managed service file when rerun.

Restart the web service to select ports again:

```bash
systemctl restart moe-gguf-server-web
```

On Alpine use `rc-service moe-gguf-server-web restart`. Startup logs are in
`journalctl -u moe-gguf-server-web` on Debian or `/var/log/moe-gguf-server-web.log`
on Alpine. Apache diagnostics are in `/opt/moe-gguf-server/_/temp/web/startup.log`
and `error.log`; domain access logs remain in `/var/www/DOMAIN-logs/`. The standalone
template is `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. Do not edit the generated
configuration in `_/temp/web/`; it is rebuilt on every startup.

<a id="chat"></a>
## Using the interface

1. Open the server URL and enter its API key when prompted, or in Settings → General.
2. Select a model. In router mode, load an available model if none is loaded; while there is none, the model
   selector links to Settings → Models, where you choose the folder of your models
   ([choosing and loading models](#settings-models)).
3. Type a message and send it. Enter sends by default; Shift+Enter inserts a line
   break. These shortcuts can be changed in settings.
4. Add files with the attachment action. Images, audio and video require compatible
   model modalities. PDF files can be extracted as text or converted to images;
   non-vision models use the text path.
5. Stop generation, skip reasoning when supported, edit a message, regenerate a
   response, or fork a conversation using the message actions.

The three lines at the top of the sidebar expand and collapse it. It shows **New chat**, **Monitor**, **Model** and
**Settings**, followed by the conversations: **Pinned**, **Recent chats** (the five you worked with last; a chat you
send a message to again moves up there) and **Chats** (the rest, newest first). Every chat has a speech bubble with a
number from 01 to 99, given when it is created and kept for its whole life (a number freed by deleting a chat goes to
the next new one), so that it can be told apart while the sidebar is collapsed, when only the icons of the menu and
the bubbles are shown. Chats can be pinned, renamed, deleted and selected in bulk. The search field next to the three
lines (Ctrl+K, or ⌘K on a Mac, expands the sidebar and focuses it) finds, without regard to case or accents, the chats by their title and by the content of their messages,
the pages and their parts, every setting, the models, the tools and the MCP servers. Click a result, or press Enter
for the first one, to open its chat or its page, where what it found is highlighted for a moment.
Conversations are kept by the server in `~/.mgs/chats/` of the user that runs `mgs` (in the deployed service,
`/opt/moe-gguf-server/.mgs/chats/`), one file per chat named after the date it was created, in UTC
(`2026-11-03@14-59-59.723.jsonl`), so every browser that opens the interface sees the same chats, and an open page
shows within a few seconds the chats that another browser creates, changes or deletes. The settings are kept by the server in
`~/.mgs/config/ui-settings.json` of the user that runs `mgs` (in the deployed service,
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), so every browser that opens the
interface shares them: generation parameters, display options, theme, language, MCP
servers, favourite models, disabled and always-allowed tools, the default reasoning
effort and the models folder of Settings → Models. Only the API key stays in each browser, because it authenticates
the request that loads the rest, so enter it once in every browser. Import/export is available in
settings. Exports can contain attachments and, if explicitly selected, sensitive
API keys or MCP authorization headers.

Choose the language in Settings → General. The 15 options are ordered `de-DE`,
`en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`,
`ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; the default is `en-US`.
Brazilian and European Portuguese have separate translations. Hebrew uses a
right-to-left interface; chat input and plain user messages follow their text's
direction, while code and terminal output are displayed left to right.
Changing language saves the choice in the server's settings and
reloads the interface; the server publishes it in `/api/config.js`, so every browser
starts in that language. Save other pending settings first. The `lang` URL query
parameter overrides it for that page, and keeps the choice when it cannot be saved.
For example, append `?lang=ja-JP` to the interface URL to select Japanese.

Settings → General → **Always reply in the interface language** is off by
default. Enable it and save the settings to instruct the model to use the
interface's language and regional variety for every new reply, including in
existing conversations, regenerated replies, continuations, turns after tool
calls and model-generated titles. For `es-AR`, the instruction explicitly asks
for Argentine voseo: “Seguí estos pasos”, “tenés” and “podés”; `es-ES` asks for
Spanish usage from Spain. British/American English and Brazilian/European
Portuguese also receive distinct regional instructions.

The instruction accompanies every new inference request and KV-cache prefill;
it is combined with the outgoing system message without editing your saved
system prompt or conversation. Changing the interface language affects new
requests after the reload; a response already being generated keeps the
instruction it received. Disabling the setting stops adding it to subsequent
requests. Existing answers are not translated. Code, commands, names, quotes
and explicitly requested translations retain their required language; their
surrounding explanations follow the interface language. The model's ability
to follow the instruction determines consistency; this is not an automatic
translation or a guarantee of perfect compliance. External API clients are
unaffected by this browser setting.

Choose System, Light or Dark for the theme. Like the language, the server publishes
the saved theme in `/api/config.js`, so every page opens in it before it is drawn,
without keeping anything in the browser; with System, the interface follows the light
or dark mode of the operating system at once, also while it is open. Generation settings include context
usage, temperature, samplers, penalties and agentic turn limits. Some capabilities
depend on server flags or the selected model.

After every response the interface sends the conversation again, when the server is
idle, so that it is already encoded in the KV cache while you read and the next turn
starts sooner. It is on by default; turn off Settings → Developer → **Pre-fill KV
cache after response** to stop it.

While the server is resolving another request, a message you send waits in the
[queue](#request-queue), and its reply shows a notice instead of the progress text until it
starts: “Waiting in the queue: another request sent directly to the API is being resolved
first (requests ahead: N).” while a request that a program sent directly to the API is
running, or otherwise “Waiting in the queue: another request from the web interface is being
resolved first (requests ahead: N).” (another tab or browser, a generated title or the KV
cache pre-fill). N counts the requests running and waiting ahead of yours. The notice goes
away as soon as your request starts.

<a id="monitor"></a>
## Monitor and status bar

**Monitor** sits under **New chat** in the sidebar (address `#/monitor`). While it is open it refreshes every
two seconds and shows:

- **Usage**: first one card per GPU of the machine (NVIDIA, AMD or Intel, even if no model uses it), titled with its
  label in bold and its name (for example **gpu-1**: NVIDIA GeForce RTX 4060 Ti; with several GPUs, gpu-1, gpu-2 and
  so on), with its usage, memory-controller (bandwidth) usage, VRAM in use, temperature and power draw against its
  limit; then one card per processor, titled the same way with its model (for example **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; with several processors, cpu-1, cpu-2 and so on), with its usage, one small vertical bar per
  logical CPU of that processor and the system load average. Every usage bar is green and fills in red as the usage
  grows; a bar whose value is unknown stays grey. NVIDIA GPUs are read through the driver's NVML library and AMD
  GPUs through `gpu_busy_percent` in sysfs; any other GPU (Intel, or NVIDIA without NVML) shows no usage, and its
  name and memory come from the inference backend once a model uses it. On a wide screen the cards go two per row,
  each half of the width like the memory bars; with an odd number of cards, the last row holds three.
- **Memory**: one bar per GPU (VRAM) and one for the RAM; a GPU without memory figures, such as an Intel GPU that no
  model uses yet, has no bar. Every bar is split by purpose, each purpose always with the same colour, and its
  legend lists every segment with its size and its share of the total, one per line and in the same order as in the
  bar. Hover a segment to see its size.

  | Segment | Contents |
  | --- | --- |
  | MTP / draft weights | Weights of the MTP (nextn) layers of the model, of a separate MTP GGUF passed with `--spec-draft-model`, or of a draft model. |
  | Expert weights | Routed expert tensors of the model. |
  | Dense weights | Every other weight: attention, shared experts, embeddings and output. |
  | KV cache | Context memory of the model (KV cache or recurrent state). |
  | MTP / draft KV cache | Context memory of the MTP or draft context. |
  | Compute buffers | Temporary buffers of the computation graphs. |
  | Expert cache and prefetch | MoE expert cache (pool, identifiers and staging) and the expert prefetch slots. |
  | Prompt cache and checkpoints | Prompts saved with `--cache-ram` and the context checkpoints of the slots (RAM). |
  | Other | Used memory that none of the above explains: other processes, the desktop, driver contexts, the multimodal projector. |
  | Free | The rest of the device or of the RAM. |

  Weights mapped from the model file count as RAM even when the kernel reports that page cache as available, so the
  segments can add up to more than the memory "in use"; the bar then shows no "Other" and less free memory. The
  expert cache appears after the first request, when it is created. In router mode the bars include every loaded
  model, and the Monitor never loads or wakes a model.

The **status bar** at the bottom of every page shows first the usage of every GPU and then that of every physical
processor, each one in its own bar, with the same labels as the Monitor cards: `gpu-1`, `gpu-2`... and `cpu-1`,
`cpu-2`... The bars have the same size, refresh every two seconds and, like those of the Monitor, are green and fill
in red as the usage grows; the hover text of each one gives the name of the device. Without a GPU it shows an empty
`GPU` bar, and when the bars do not fit on a narrow screen the status bar scrolls sideways. Click it to open the
Monitor. It keeps working while the model loads. It is shown by default; turn off Settings → Display → **Show status
bar** to hide it.

<a id="models"></a>
## Models, tools and MCP

`mgs` only loads MoE models. When it starts, it reads the header of the GGUF passed with `--model` (and of the one passed with `--spec-draft-model`); if it is a dense model, such as GPT-2 or Phi-2, `mgs` logs an error like this one and exits with status 1 before loading anything:

```text
srv    mgs_server: the model '/path/to/model.gguf' is not a Mixture of Experts (MoE) model
srv    mgs_server: mgs only loads MoE models; dense models are not supported
srv    mgs_server: exiting due to a dense model
```

In router mode only the process of that model exits: `/api/v1/models` marks it as failed and the other models keep working. Dense decoder implementations are not compiled, but the dense layers inside a MoE model and an explicitly supplied, compatible MTP-only head (`--spec-draft-model`) are retained.

<a id="settings-models"></a>
### Choosing and loading models in Settings → Models

Started without `--model`, `mgs` runs in router mode: it loads no model by itself and lists the models it finds, so
you can load them from the interface. Settings → Models has two parts:

- **Models folder.** Press **Choose folder**: a browser of the folders of the computer where `mgs` runs opens at the
  current models folder (or at your home folder). Click a folder to open it, use the arrow to go up and the house to
  go to your home folder, or type a path in its field and press Enter (a leading `~` is your home folder); **Show
  hidden folders** adds the folders whose name starts with a dot, and the browser counts the `.gguf` files of the
  folder it shows. Press **Choose** and the absolute path of that folder fills the field and is used at once. The
  folder may hold one `.gguf` file per model, or one subfolder per model with the parts of a split model
  (`NAME-00001-of-0000N.gguf`) and its multimodal projector (`mmproj*.gguf`). `mgs` checks that it can read the
  folder, saves it with the other settings and scans it at once; every later start scans the same folder. To read it
  again after you add or remove files, choose the same folder again. Dense GGUF files, separate MTP heads among them,
  are left out of the list, because `mgs` only loads MoE models. Choosing another folder unloads the running models of
  the previous one. The models downloaded to `~/.cache/mgs/` are listed as well. When `mgs` was started with
  `--models-dir`, that folder prevails: **Choose folder** is disabled and the page says so.
- **Available models.** Every model with its status (not loaded, loading with its percentage, loaded, sleeping or
  failed to load), its file and its size on disk, and a button to load or unload it. Loading a model here also selects
  it in the chat.

What the loaded models are and how they were loaded is on the [Model page](#model-page), which the end of Settings →
Models links to. The model selector of the chat lists the same models and loads the one you pick; while the list is
empty, it links to Settings → Models. Started with `--model`, Settings → Models only says that the model cannot be
changed without restarting `mgs`.

Only one model is loaded at a time: loading another one, here, from the selector of the chat or through the API,
unloads the one that was loaded (`--models-max N` allows N at once). A new chat uses the model loaded at that moment.
A chat keeps its own model, the one of its last answer: when you send a message to it while another model is loaded,
that one is unloaded and the model of the chat is loaded first. If the model of the chat is no longer where it was
(it left the list, or it is now another file), a dialog says so and lets you cancel the message, which stays in the
field, or send it with the loaded model.

The folder is kept in `~/.mgs/config/ui-settings.json` with the other settings of the interface, as the section
`modelsFolder`. `mgs` writes it when you choose the folder and reads it on every start:

```json
{
  "modelsFolder": "/home/user/models"
}
```

The deployed service (`deploy/`) starts `mgs` with `--models-dir /opt/moe-gguf-server/models`, so that is the folder
it lists.

<a id="model-page"></a>
### Model page

**Model**, in the sidebar after **Monitor** (address `#/model`), shows the loaded model in two boxes: **Model file**,
with what its GGUF file holds (description, architecture, file, size on disk, parameters, layers, MTP layers,
training context and multimodal projector), and **Load parameters**, with how it was loaded: the parameters that
`mgs` chose by itself or that you passed (context size, parallel slots and context per slot, batch and micro-batch,
generation and batch threads, flash attention, KV cache types, `--fit`, MTP, speculative decoding with its maximum
number of draft tokens, MoE expert cache and devices) and, in router mode, the command line of its process. Every state that is on (flash attention, `--fit`, MTP, speculative decoding and the MoE expert cache) is
shown in green. Flash attention says whether `auto` turned it on; MTP shows how many draft tokens it asks for in
every step, or that it is still measuring them (0 to 3), or that drafting does not pay off on this computer; and the
MoE expert cache says whether it was created, with its automatic or requested size: the first request after loading
the model decides it, so until then it says so, and it stays off on an integrated GPU or when every expert is
already in VRAM. These live states come from the [Monitor](#monitor) report, which also has the memory that each
device uses, by purpose. While no model is loaded, the page links to Settings → Models.

<a id="supported-models"></a>
### Supported model families

| Family | GGUF architecture | Notes |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP when the GGUF carries it. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP when the GGUF carries it. Its per-layer embedding table (about 29 GB) is read from the model file on demand, so keep the file on a fast disk. Long contexts are computed with dense attention under the sparse selection: correct, but without the speedup of sparse kernels. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; images with its `glm5v` mmproj. |
| Kimi-K3 | `kimi-k3` | The published GGUFs carry no MTP. |
| DeepSeek-V4.1-Flash | `deepseek41` | Loads both published layouts: antirez's DwarfStar files (such as `antirez/deepseek-v4.1-flash-gguf`) and the files of the llama.cpp ecosystem. Its Engram table (about 188 GiB inside the file) is never loaded: every token reads its rows from the model file, so the file must stay available while `mgs` runs, preferably on an SSD, and Linux or another POSIX system is required. Files without a chat template use a built-in one, and `reasoning_effort` accepts a number from 1 to 100. No MTP and no images. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |
| Flint | `flint` | The published GGUFs carry no MTP. |

Gemma 4 and the DeepSeek models older than V4.1 are no longer supported (see [removed compatibility](#removed-compatibility)). Neither are the other MoE architectures inherited from llama.cpp, such as gpt-oss, Qwen3-MoE, Qwen3-Next, GLM-4.x, Granite 4, MiniMax-M2 or Kimi-Linear.

With several selected GPUs, MTP checks all of them before falling back to RAM. Its memory estimate includes the output head. Integrated MTP buffers load before the other weights; an external MTP GGUF passed with `--spec-draft-model` loads before the main model, also with `--fit off`, and turns MTP on by itself (see [automatic MTP](#performance)); an MTP that is only in another GGUF is never used unless it is passed that way. The current explicit device selection still determines which GPUs are available.

Use `--reasoning on` or `--reasoning off` to control thinking. When thinking is off (with that option, with the chat's reasoning switch, or in a request with `enable_thinking: false` in `chat_template_kwargs` or `reasoning_effort: "none"`), `mgs` also closes any reasoning block as soon as the model opens it, so models whose chat template ignores that setting, such as Flint, answer without reasoning too. Passing `enable_thinking` through `--chat-template-kwargs` is rejected. Use the documented hyphenated option names; underscore spellings are rejected. Pass list values as a comma-separated value instead of repeating an option. Only `--spec-type` remains repeatable to compose speculative modes.

The model selector shows available, loaded and favourite models. Model details
include context size, modalities, quantization and the chat template. Router mode
manages separate child processes for loaded models; requests identify the model
through the `model` field or query parameter.

MCP servers are configured in Settings → MCP servers, by URL with optional authorization headers. You can
inspect connection logs, available prompts, resources and tools. The optional
server CORS proxy must be explicitly enabled using the engine's corresponding
flag; it only returns inert content and refuses to be opened as a page (see
[security limits](#security-limits)). Browser restrictions on HTTP resources within HTTPS pages still apply.
When two enabled MCP servers offer a tool with the same name, neither copy is offered to the model and the
interface shows a warning: disable all but one of those servers to use the tool.
The server's built-in tools and the browser JavaScript sandbox are optional.
Tool approvals and the agentic turn limit remain available in the interface. The turn limit
(Settings → Agentic → **Agentic turns**, the maximum number of tool execution cycles
before stopping, to avoid infinite loops) is 50 by default.

Streams can reconnect using the saved conversation identity and cursor while the
server retains the stream. A stopped server or expired stream cannot provide a
guaranteed replay. The interface reports a failed reconnection rather than
silently creating a second generation request.

### Local shell MCP server

Turn on **Local shell MCP server** in Settings → MCP servers, or start `mgs` with `--local-shell-mcp-server`, to let
the models of the web chat use the terminal of the machine where `mgs` runs:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- The switch works at once, without restarting `mgs`, and is saved in the section `localShellMcpServer` of
  `~/.mgs/config/ui-settings.json`, so later starts turn it on again (`POST /api/mcp/shell/enabled` with
  `{"enabled": true}` does the same). With `--local-shell-mcp-server` it stays on and the switch is disabled.
  Without an API key, only a browser on the computer where `mgs` runs can use the switch, so nobody else on the
  network can turn on a shell there; with an API key, anyone who has the key can.

- At startup `mgs` checks which shells are installed and offers one tool for each of them: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) and `psh.run` (PowerShell, `pwsh`). The log lists the shells it found; a shell
  that is not installed gets no tool.
- The interface detects it, adds it to the MCP servers as **local shell mcp server** and enables it. You can disable it there,
  or for a single conversation, and your choice is kept. When the server is turned off, the entry disappears.
- Every tool takes `command` and, optionally, `cwd` (working directory; by default the home directory of the user
  that runs `mgs`) and `timeout` (seconds; 60 by default, 3600 at most). The answer is a JSON object with
  `exit_code`, `stdout` and `stderr`, plus `signal`, `timed_out`, `cancelled`, `timeout`, `seconds`, `shell` and
  `cwd`; long outputs keep only their beginning and their end.
- Each command runs in a new process without a terminal and with empty input: `cd` and variables do not carry over
  to the next call, and interactive programs (editors, questions) cannot be used. When the timeout expires or the
  request is cancelled, every process the command started is killed. Processes left in the background must
  redirect their output.
- The interface asks for permission before every call, unless you choose to always allow the tool.
- `sudo` works without a terminal (Linux): when it needs a password, a window opens in the centre of the page,
  on whatever page you are, with the prompt of sudo (`[sudo] password for user:`), the tool and the command that
  asks, a password field and, under it, the **Accept** and **Cancel** buttons. The cursor is already in the
  password field, so you can type the password right away; **Enter** accepts and **Escape** cancels; a click
  outside the window does not close it. The password only goes to sudo on the machine where `mgs`
  runs: it is not saved in the conversation, it is not written to the logs and the model never sees it. While the
  window waits, the timeout of the command stops counting; if you press **Cancel**, or nobody answers within 5
  minutes, sudo fails. sudo asks again in every call, because each one is a new process, and up to three times when
  the password is wrong; the window then warns that the previous password may be wrong. Commands run in a session
  of their own, without the terminal where `mgs` was started, so sudo never asks for the password in that terminal.
  Only the web interface of `mgs` can answer: with other MCP clients, sudo fails as before for lack of a terminal.
  The deployed service runs as the `moe-gguf-server` account, which normally has neither a password nor sudo
  permission, so this is meant for `mgs` run with your own user (for example, the `~/.local` build). When `mgs`
  has no `DISPLAY` variable, the commands receive `DISPLAY=:mgs-askpass`, because sudo only uses its password
  helper without a terminal when `DISPLAY` exists; programs that try to open a window fail at once.
- In the chat, every call is shown with the name of its tool, such as `bash.run`. Expanded, it shows up to three
  grey boxes in a monospace font: **stdin** with the bare command, **stdout** with the output and **stderr** with
  the error text. stdout only appears when the command printed something and stderr only when there was an error;
  the exit code goes next to their title, for example `stdout (code 2):` (next to stdin when there is no output).
  The stdin box has a white border (dark grey on the light theme), the stdout box a green one and the stderr box
  a red one, with every shell tool (`bash.run`, `zsh.run`, `sh.run` and `psh.run`).
- Turn on **Settings → Display → Expand local shell commands automatically** to open every call on its own: each
  box appears as soon as its content is known, stdin while the model writes the command and stdout and stderr when
  the command ends.
- Every call to a shell tool leaves its own file in `~/.mgs/local-shell-mcp-server/` (in the home of the user that
  runs `mgs`; in the deployed service, `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), named after the local
  date and time of the call down to the millisecond, for example `y2026m10d12h23m59s59ms731.jsonl`; `-2`, `-3`… are
  added when several calls start in the same millisecond. It is a JSONL file, one JSON object per line: first the command (`"event": "command"`,
  with the tool, the shell, the working directory and the timeout), written before it runs; then the whole output,
  without cuts, as it arrives, one line per piece (`"event": "stdout"` or `"stderr"`, with the text in `data`) in
  the order the command wrote it; and last the result (`"event": "result"`: exit code, signal, whether it timed out
  or was cancelled, the seconds it took and the bytes of stdout and stderr) or the reason it could not start
  (`"event": "error"`). A call rejected before running (an unknown tool, no command, a working directory that does not
  exist, an invalid timeout) also leaves its file, with the command as it was asked and the error. The chat keeps only the
  beginning and the end of very long outputs, but the file has everything, and it already holds the output if `mgs`
  stops while the command runs. To read the whole stdout of a call, run `jq -j 'select(.event == "stdout") | .data'
  FILE`. JSON only holds text, so bytes that are not UTF-8 (binary output) are replaced with `�`; a command that
  prints gigabytes leaves a file of that size. Only that user can read the folder and the files, because the output
  of a command may hold secrets; nothing deletes them, so remove the old ones when you no longer need them.

In the deployed service, add `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` to the `[Service]` section of a systemd
override (`systemctl edit moe-gguf-server`) and restart `moe-gguf-server`; on Alpine, export the same variable in
the OpenRC service configuration.

**Security.** Commands run with the permissions of the user that runs `mgs` (in the deployed service, the
`moe-gguf-server` account, never `root`). Anyone who can open the interface or call the API can run them, so enable
the option only in trusted environments and protect the server with an API key. Requests from pages of other sites
are rejected (the default CORS origin is localhost) and the log records every command with its exit code. Read
the command shown in the password window before typing a sudo password: the password authorises that command. The
server only hands it to a password helper whose parent process is sudo running as root for your user, so a command
cannot start the helper by itself to read the password.

<a id="api"></a>
## HTTP API

Examples use 11443; substitute the HTTPS port from `ports.json` if different.

The default API root is `/api`. The UI remains at `/`. The root `/v1/...` routes
from the original server are no longer registered. OpenAI-compatible clients must
use `https://ai.example.com:11443/api/v1` as their base URL: the OpenAI routes (`/api/v1/models`,
`/api/v1/chat/completions`, `/api/v1/responses`, `/api/v1/audio/transcriptions` and their
`input_tokens` variants) exist only under `/api/v1`, and the health check only at `/api/health`.
`/api/rerank` and `/api/v1/rerank` both remain for TEI/Jina clients, and the native routes
(`/api/completions`, `/api/embeddings`…) are different handlers from their `/api/v1` namesakes.
Request/response field names
retain upstream compatibility; the C/C++ engine interfaces keep their current functions,
but not the ones that were marked deprecated or had no use (see [removed compatibility](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Set `MGS_API_KEY` to the installed key in your own shell (only `curl` uses it; `mgs` reads its key from
`--api-key`, `MGS_ARG_API_KEY` or `--api-key-file`). Native operations,
embeddings, tokenisation, reranking, model management and resumable streams are
documented in the bundled Swagger UI at `/api/doc/`; its OpenAPI JSON is at
`/api/doc/openapi.json`, generated by the server on each request without a static JSON file. The inventory reflects the routes registered by the
running process, including router and optional compatibility routes. Disabled
features may return 403. A missing/invalid key returns 401 and model loading
returns 503. `/api/health` and `/api/v1/models` keep public access for `GET` and `HEAD`; see
[security limits](#security-limits) for the routes that always need the key.

`--api-prefix /api/custom` relocates inference routes below `/api/custom`.
The root UI receives that prefix from `/api/config.js`; docs remain at `/api/doc/`.
The documentation namespace is reserved and cannot be used as an inference
prefix. Router children use `/api` internally. The server validates prefixes;
values outside `/api/` and paths with trailing slashes are rejected.

The Monitor uses `GET /api/monitor` (full report), `GET /api/monitor/memory` (memory snapshots of the loaded models)
and `GET /api/monitor/usage` (CPU and GPU usage only, also available while the model loads). They read counters and
snapshots published by the inference loop, never the task queue, so polling them neither wakes a sleeping model nor
resets its idle timer. Every model of the report carries `expert_cache` (with `active` and `pending`) and, with MTP,
`mtp`: `draft_n` (the draft tokens asked for in every step, 0 when drafting did not pay off, -1 while they are
measured), `draft_n_max` and `measuring`. Their fields are described in Swagger.

The interface reads its settings with `GET /api/ui/settings` (`{"settings": DOCUMENT}`, or
`{"settings": null}` before the first save) and saves them with `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, which replaces `~/.mgs/config/ui-settings.json` whole, with mode 0600). Both require
the API key and also answer while the model loads.

The chats use `GET /api/chats` (the conversations and their `version`), `GET /api/chats/version`,
`GET /api/chats/search?q=WORDS&limit=N` (the newest message of the user or the assistant of every chat that holds every
word, without regard to case or accents) and `GET`, `POST` and `DELETE /api/chats/{id_chat}`. Every file starts with
the conversation and follows with one message per line; the files keep the dates in the native format
`2026-11-03@14-59-59.723` (UTC) and the API in milliseconds since the epoch. A chat saved without a valid number gets
the lowest free one. They require the API key and also answer while the model loads.

In router mode, `GET /api/models/folder` returns the models folder: `folder`, `source` (`cli` for `--models-dir`,
`settings` for the section `modelsFolder` of `~/.mgs/config/ui-settings.json`, or `none`), `exists`, `model_count`,
`settings_file` and `cache_dir`. `POST /api/models/folder` with `{"folder": "/absolute/path"}` (a leading `~` is the
home folder; `""` removes it) checks that the folder can be read, saves it in that section, rescans the models and
returns the same object; it answers 400 when the folder was set with `--models-dir` or cannot be read.
`GET /api/models/folder/browse?path=/absolute/path` feeds the folder browser: it returns `path`, `parent` (empty at
`/`), `home`, `folders` (the subfolders sorted by name, each with `name`, `path` and `hidden`) and `gguf_files` (how
many `.gguf` files are right inside the folder); without `path` it lists the models folder, else the home folder, and it
answers 400 when the path is not a folder that can be read. All three require the API key. In router mode every entry of
`GET /api/v1/models` carries `path`, `file_size` (bytes, every part of a split model) and `mmproj`, and a running model
adds `meta` and `load`, the configuration it was loaded with (`architecture`, `description`, `n_layer`, `n_layer_mtp`,
`n_ctx`, `n_ctx_slot`, `n_parallel`, `n_batch`, `n_ubatch`, `n_threads`, `n_threads_batch`, `flash_attn`,
`flash_attn_enabled` (whether the context uses it, also with `auto`), `cache_type_k`, `cache_type_v`, `fit`, `mtp`,
`speculative`, `draft_n_max`, `moe_cache_mib`, `devices` and `mmproj`).
With `--model`, the only entry also carries `load`.

<a id="request-queue"></a>
### Request queue and parallel requests

By default `mgs` resolves one request at a time (`-np 1`) and gives it the whole context, so every request can
use all the context tokens, whether it comes from the web interface or directly from the API. The requests that
arrive meanwhile wait in a single queue and are resolved in the order they arrived: requests of the interface do
not jump ahead of API requests, nor the other way round, and a request that has to wait again keeps its place.
The API behaves the same whatever the interface settings are.

To resolve several requests at the same time, start `mgs` with `-np N` (`--parallel N`, `MGS_ARG_PARALLEL`):
up to N requests run in parallel and the context is split among them, so each one gets a share (`n_ctx_slot` in
the `initializing` line of the log); with `--kv-unified` they share a single KV cache of the whole context
instead. Requests beyond N still wait in the queue.

The web interface adds `"mgs_client": "webui"` to its requests. While one of its streamed requests waits, the
stream begins with events such as

```text
data: {"mgs_queue":{"ahead":2,"running_api":1,"running_webui":0,"waiting_api":0,"waiting_webui":1}}
```

(`ahead`: requests running or waiting ahead of it; `running_api` and `running_webui`: how many of them are running
now and came directly to the API or from the web interface; `waiting_api` and `waiting_webui`: the same for the
requests waiting ahead of it in the queue), and a last one with every count at 0 when it starts. Clients that do not
send `mgs_client` never receive these events, so OpenAI-compatible clients get the usual stream. Swagger
describes the field with the other completion parameters.

<a id="security-limits"></a>
### Security limits

These limits protect the server and the interface from hostile requests, files and models. They are generous
enough for normal use: what exceeds them fails with a clear error instead of stopping the server.

- **API key.** The health and model list routes (`/api/health` and `/api/v1/models`) are public only for `GET`
  and `HEAD`. Reloading the model list (`GET /api/v1/models?reload=1`) and adding or deleting models in router
  mode (`POST` and `DELETE /api/models`) need the key whenever one is configured.
- **Pages of other sites.** By default (`--cors-origins` not set, which means `localhost`) only the pages of
  `localhost` and of this same server may call the API: the server answers 403 to any request other than `GET`,
  `HEAD` or `OPTIONS` whose `Origin` header names another site, and the browser does not let other sites read
  its answers. The interface (also behind Apache) and clients that send no `Origin`, such as curl or the SDKs,
  are not affected. To let the pages of other sites call the API, list their origins in `--cors-origins` (`*`
  admits any site: then set an API key). `POST /api/tools` only accepts `Content-Type: application/json` (415
  otherwise).
- **DNS rebinding.** With plain HTTP (`--http`) on a TCP port and no API key, the server answers 403
  (`Host not allowed`) to every request whose `Host` header is not an IP address, `localhost` (or a name ending
  in `.localhost`), the name of the machine (also with `.local`) or the name given with `--host`, so a page of
  another site that points its own name at this machine cannot use the server. HTTPS, an API key, the Unix
  socket of the deployment and Vertex AI (`AIP_MODE=PREDICTION`) are not affected. To reach a plain HTTP server
  through another name, set an API key.
- **CORS proxy.** `/api/cors-proxy` refuses requests that a browser would open as a page, frame or worker
  (403) and returns every answer as inert content: it cannot run scripts or set cookies on the server's address.
- **Router listings.** In router mode, `status.args` and `status.preset` of `/api/v1/models` show `<redacted>`
  instead of the values of `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  and `--mcp-servers-json`; the log does the same. The model processes still receive the real values.
- **Vertex AI.** With `AIP_MODE=PREDICTION`, the `@requestFormat` of each instance only reaches the inference
  routes (completions, chat completions, responses, messages, infill, embeddings, rerank, tokenize, detokenize,
  apply-template and token counting); any other route, the prediction route itself included, answers an error
  for that instance.
- **Model downloads.** The Hugging Face token (`--hf-token`, `HF_TOKEN`) only goes to the Hugging Face endpoint
  (`MGS_MODEL_ENDPOINT`, `https://huggingface.co/` by default); a model, projector or draft model downloaded
  from any other URL is requested without it.
- **Slow clients.** While the server receives a request, each read waits 30 s at most (`--timeout` still governs
  writes, so long answers are not cut), and at most 256 accepted connections wait for a free thread.
- **Request bodies.** JSON bodies nested deeper than 1024 arrays or objects are rejected with 400. JSON schemas
  (`response_format`, `json_schema`) may nest up to 256 levels; empty `pattern` values and quantifiers with
  nothing to repeat are rejected.
- **Remote media.** URLs in `image_url`, `input_audio` and `input_video` may only point to public internet
  addresses. Loopback, private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10), link-local
  addresses (including the cloud metadata address 169.254.169.254), IPv6 unique local addresses and other
  non-public addresses are rejected with 400, also when a redirect leads to them. To allow them, for example to
  read images from a server in your network, start `mgs` with `--media-allow-private-urls`
  (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); do it only when every API client is trusted. Each download has 10 s to
  connect, 30 s in total, up to 5 redirects and 10 MiB, and the server runs at most 8 downloads at once. All the
  media of one request together may hold at most 256 items and 512 MiB, and its remote downloads have 120 s in
  total; files read with `--media-path` count towards the same limits and are checked before they are read.
- **Decoded media.** Images up to 32768 pixels per side and 100 megapixels, audio up to one hour, video frames up
  to 8192 pixels per side and 2 GiB of decoded frames per video, and at most 2 GiB of decoded images, audio and
  video frames per request (the frames of all its videos included).
- **Projectors.** A multimodal projector whose metadata does not fit its graph (for example a patch merge factor
  below 2 in a projector that merges patches) fails to load with an error instead of stopping the server.
- **Server tools.** `read_file` with a line range reads the file in blocks and keeps only those lines, scanning
  at most 256 MiB of the file; without `end_line`, files over 16 KB are refused.
- **Chat templates of the model.** `range()` produces at most one million elements, built strings stay below
  256 MiB, `indent()` and `tojson()` indentation below 256 spaces and nesting below 1024 statements; a template
  that exceeds them fails with a template error.
- **Interface.** Links in messages keep only the http, https, mailto and tel schemes (or relative addresses), and
  image attachments must be embedded images. Conversation import accepts files up to 512 MB, ZIP archives with up
  to 10 000 entries, 256 MB per JSONL file and 1 GB extracted in total. A file with malformed records, or with a
  message that belongs to another conversation, is rejected whole; a conversation whose messages already exist,
  or whose messages link to each other in a loop, is skipped and reported. SVG images written by the model stay
  inside their own box and cannot cover the rest of the page. Listing the resources or resource templates of an
  MCP server stops after 100 pages, 10 000 items, 60 s or a repeated cursor. An MCP resource is always read,
  cached and attached from the server it was chosen from, even when another server offers the same URI. An
  **always allow** approval of an MCP tool applies while its server keeps the same address, transport, proxy use
  and headers: if any of them changes, the interface asks again.

The installation commands still download the current `main` branch over HTTPS; they do not check a signature or
a fixed revision.

<a id="performance"></a>
## Memory and performance

Host memory registration and expert prefetch are enabled by default. Prefetch
overlaps expert transfers with computation, particularly for large prompts.
Results depend on the model, memory capacity and RAM/PCIe bandwidth.

For models with CPU-offloaded experts, `--cpu-moe` or `--n-cpu-moe N` can reduce
VRAM use. Host registration pins mapped weight pages for faster host-to-device
copies. Pinned memory cannot be paged out, so leave enough RAM for the operating
system. This registration path is POSIX-specific; Windows retains its earlier
loading behaviour. `--load-mode none` is recommended only when host registration fails
wholly or partly, not as a normal requirement.

`mgs` takes into account the memory limit of its cgroup (a systemd service with
`MemoryMax`, a container) as well as the free RAM. Automatic fitting keeps the
MoE experts in RAM only when they fit within that limit, and otherwise places
whole expert layers on the GPU. When the experts stay in RAM, the prompt cache
(`--cache-ram`, 8 GiB by default) takes at most the memory left after them and
the margin, which the log reports as `prompt cache limited to N MiB`;
`--cache-ram` sets it explicitly. Only the model pages that stay in RAM are pinned, and only as many as leave a margin (4 GiB or 10% of the RAM) of the memory still available; beyond that the log shows `pinning X of Y GiB of mapped weights` and the rest uses pageable memory. In a
30 GB cgroup with the 37.6 GB Qwen3.6-35B-A3B Q8_0, version 0.1.17 hung while
loading until the system killed it; now it starts in a few minutes and
generates. Up to version 0.1.22, with whole expert layers on the GPU, `mgs` also pinned the experts already copied to the GPU, so on a desktop with little free RAM and no swap the whole system could stop responding while the model loaded.

When the model files are larger than the memory still available (or than the limit of the cgroup it runs in), `mgs` does not read them whole while loading, which would only evict their own first pages and the page cache of every other program: it reads what goes to the GPU and what it pins, and the rest when it is first used. With Qwen3.8-Flash-Next (113 GB) in 40 GiB of memory and a SATA disk, the load went from more than 5 minutes to about a minute and a half.

Expert prefetch defaults to three VRAM slots. Each slot must fit the largest
transferred expert tensor. Values 2–8 in `GGML_SCHED_PREFETCH_EXPERTS` explicitly
choose a slot count. If memory allocation fails, at least two allocated slots are
kept when possible; otherwise regular transfers are used. Prefetch starts only
when the batch has at least twice as many expert selections as experts. Its pool
belongs to the first GPU that activates it; other GPUs keep selective copies.
Larger batches can help prefill but require more memory. On an [integrated GPU](#integrated-gpu) there is no prefetch: filling a slot would be a copy from RAM to RAM by the CPU, and the slots would take RAM that the page cache of the experts needs, so the experts in use are copied as needed instead (with Qwen3.8-Flash-Next on a Radeon 860M the prompt went from 15 to 33 tokens/s and generation from 4.3 to 5.0 tokens/s).

Each slot takes the size the GPU allocates for the largest expert tensor, including the padding that CUDA adds to rows whose length is not a multiple of 512 elements, and leaves `--moe-cache-reserve` (1024 MiB by default) free for the memory that the computation takes while it runs; when a slot does not fit, prefetch continues with the slots that fit, if there are at least two, or copies the experts in use instead. Up to version 0.1.31 the slots left that padding out, so Qwen3.8-Flash-Next, whose experts have 640-element rows, could stop at the first long prompt with `CUDA error: an illegal memory access was encountered` (or `an internal operation failed`) whenever the GPU had room for two slots, for example with MTP on an RTX 4060 Ti. With Qwen3.8-Flash-Next on the RTX 4060 Ti, MTP of 2 tokens now processes the prompt at 12 tokens/s and generates at 1.31 tokens/s.

For diagnostics, set `GGML_CUDA_REGISTER_HOST=0`, `GGML_SCHED_PREFETCH_EXPERTS=0` or
`GGML_VK_STAGING_RING=0` (Vulkan uploads through the staging ring) to disable the corresponding optimisation.
Do not set those variables during normal operation unless intentionally opting out.

Automatic MTP examines GGUF metadata and tensor names before weight allocation,
including every shard. For the MTP integrated in `--model`, it requires a supported
architecture, a model trunk and all integrated heads. A separate MTP GGUF passed with
`--spec-draft-model` also activates it when it declares complete MTP heads, without a
trunk; a draft model without MTP heads keeps the speculative type you choose. An MTP
that is only in another GGUF is never looked for, and an unsupported architecture does
not activate it. The default draft limit is two tokens: with MTP, two generated as fast as three on an RTX
4060 Ti and on an integrated Radeon 860M, and a rejected draft wastes less verification work. Any
explicit `--spec-type`
selection takes precedence and turns automatic selection off; `--spec-type none`
disables MTP. `--spec-draft-n-max` changes the limit. MTP affects generation,
not prefill or time to first token directly. Improvement depends on acceptance
rate and hardware; increasing the limit does not guarantee more speed.

Whether MTP speeds generation up depends on the machine: each step checks the drafted tokens with the whole model, which costs little when the experts are in the GPU and a lot when most of them come from RAM or disk. When MTP is enabled automatically, `mgs` therefore measures it during the first answers: after a short warm-up it takes turns step by step with 0 (no draft), 1, 2 and 3 draft tokens until each has generated 64 tokens, about 500 tokens in all, keeps the fastest when drafting is more than 5% faster than no draft, and logs it in a line such as `MTP drafts 2 tokens from now on, the fastest on this machine`. The result is saved in `~/.mgs/tuning/mtp.json` per model, machine and version of `mgs`, and later starts use it directly; when drafting was not worth it, the log says `MTP drafting is not worth it on this machine`, and from the next start MTP is not loaded at all, which also leaves its VRAM to the experts, the context and the micro-batch. Removing the file (or its entry) measures again. `--spec-draft-n-max N` fixes the number of draft tokens and turns the measurement off, `--spec-type draft-mtp` keeps MTP on, and a separate MTP GGUF passed with `--spec-draft-model` is always used. On the Radeon 860M, Qwen3.8-Flash-Next generated 5.12 / 7.09 / 7.34 / 6.97 tokens/s with 0 / 1 / 2 / 3 draft tokens and keeps 2; on the RTX 4060 Ti it generated 1.81 / 1.79 / 1.30 / 1.59, so it runs without MTP from the next start, with 23-24 tokens/s of prompt and 1.69-1.70 of generation instead of 12 and 1.4, while Qwen3.6-35B-A3B generated 26.38 / 33.92 / 36.21 / 35.41 there and keeps 2.

<a id="mtp-vram"></a>
MTP weights always stay in GPU memory when the GPU can hold them: the MTP layers grafted onto the model, a
separate MTP GGUF passed with `--spec-draft-model`, and the output head they use. The options that move weights to
system memory (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, their `--spec-draft-*` equivalents and automatic
fitting) do not apply to them, and they stay on the GPU even with a low `--gpu-layers`. Automatic fitting places the
rest of the model around them, and the [MoE expert cache](#expert-cache) uses the memory left after that. When an
MTP layer does not fit in the free memory of any selected GPU, the log shows a warning (`no selected GPU has enough free
memory for MTP layer 0; using system memory`) and only that layer stays in system memory; a tensor whose type the GPU
cannot hold also stays there. On an [integrated GPU](#integrated-gpu), the free memory counted for MTP is at most what
the system RAM can still give. With a separate MTP GGUF, the MTP layers grafted onto the main model are not loaded, so
they do not take memory twice; with `--spec-type none`, no MTP weight is loaded. The load log confirms the placement
with lines such as `MTP layer 0 assigned to CUDA0 (856.36 MiB)`, visible with `-lv 4`.

<a id="expert-cache"></a>
### MoE expert cache

When MoE experts stay in system memory and the rest of the layer runs on a GPU, the server keeps the most
recently used experts of all layers in a cache that takes the free VRAM;
with several GPUs, every GPU caches the layers assigned to it in its own VRAM. For every generated token (and any
batch below 32 tokens), the experts found in the cache run on the GPU, a share of the missing ones is uploaded
to the cache and the rest runs on the CPU at the same time; the share follows the measured PCIe and CPU
bandwidths. Long prompts keep the expert prefetch, which now copies the cached experts inside the GPU and only
uploads the others. The output matches the run without cache up to rounding. The GPU and the CPU round differently and the experts
each one computes depend on the cache contents, so a long greedy generation can drift to a different but equally
valid text after some tokens, from one run to the next; the cache of the official llama.cpp behaves the same way.

The cache is enabled by default and is created by the first request, once the model, the MTP context and the
multimodal projector are loaded. Without `--gpu-layers`, `--n-cpu-moe` or `--override-tensor`, automatic
fitting keeps every expert in system memory when it fits with a margin (4 GiB or 10% of the RAM) and leaves the free VRAM to the cache (with several GPUs, it spreads the layers over
them by their free memory so every GPU has a cache); otherwise it places whole expert layers on the GPU as before and the cache uses what
remains. Explicit placement options are respected: `--cpu-moe` gives the cache the most VRAM and
`--n-cpu-moe N` keeps some static layers on the GPU. The experts kept in RAM are pinned as long as the memory still available keeps that margin; the rest stays pageable, so the system keeps responding when RAM is short.

Every layer keeps its own equal share of the cache, so a burst of misses in some layers never empties the
cache of the others, and within every share the experts used again are protected from those used only once:
an expert uploaded to the cache enters a probation part and moves to a protected part (at most half of the
share) when it is used again, and new experts replace the probation part first. When only part of the missing
experts can be uploaded, the ones used most recently and often go first. With Qwen3.6-35B-A3B Q8_0 on an RTX
4060 Ti, generation went from 37.4 to 38.1 tokens/s (from 45.2 to 47.1 with MTP) and the cache hits from 72.3
to 74.1%. The shares also keep the cache useful when a step brings more experts than the whole cache can
hold, as with MTP on GPUs with less VRAM, where a single list shared by every layer kept almost nothing.

<a id="integrated-gpu"></a>
On an integrated GPU (the graphics of an AMD or Intel processor through Vulkan, or NVIDIA Jetson and DGX Spark
through CUDA) the memory the GPU reports is part of the system RAM. Automatic fitting counts both as a single
budget: the GPU may use at most what the RAM can still give after the buffers that stay on the CPU. If the whole
model fits there (reducing the context if needed, as on a discrete GPU), everything runs on the GPU; otherwise
whole expert layers go to the GPU up to that limit and the CPU computes the rest. There is no expert cache on an
integrated GPU, not even with `--cpu-moe` or `--n-cpu-moe`, because it would copy experts from the RAM to the same
RAM and keep them twice. The log shows the budget in a line such as
`Vulkan0 is an integrated GPU that shares the system memory: N MiB usable` and the decision in `no MoE expert cache on an integrated GPU`. With the model mapped (`--load-mode mmap`, the default), the weights that stay on the CPU when every layer is on the GPU (the token and per-layer embedding tables, read row by row) do not count against that budget, because only the rows read stay in memory: Qwen3.8-Flash-Next, whose per-layer embeddings take 28 GB, gets all the memory the GPU reports instead of what is left after them. On such a GPU `mgs` also restarts itself once at startup with `OMP_WAIT_POLICY=passive`, unless the variable is already set, so that the CPU threads sleep instead of spinning while the GPU works: they share the power of the processor with it, and with spinning threads the GPU of a Radeon 860M ran at about half its clock. The log shows it in the line `the OpenMP threads of the CPU wait passively`; `OMP_WAIT_POLICY=active` keeps the spinning threads.

<a id="context-size"></a>
When the model does not fit in VRAM and `-c` is not given, automatic fitting also chooses the context size: the
KV cache takes at most a quarter of the VRAM left after the dense weights, up to the training context of the
model, and the experts (the cache or whole expert layers) keep the rest. Previously such models always got the
minimum of 4096 tokens shared by every slot, which cut long answers short (`truncated = 1` in the log). The
log shows the result in a line such as `context size raised from 4096 to N tokens` and in `n_ctx_slot` of the
`initializing` line. `-c N` sets the context explicitly (more context leaves less VRAM to the cache and lowers
generation speed: 35.8 tokens/s with 4096 tokens and 33.9 with 122624 on the reference machine) and
`--fit-ctx N` raises the minimum. The MTP context takes the same size as the model context instead of reserving
VRAM for the whole training context. When whole expert layers fill the GPU because the experts do not stay in RAM for the cache (always on an [integrated GPU](#integrated-gpu)), the KV cache takes at most a sixteenth instead, because every expert layer left out of the GPU is read from the RAM, or from the disk when the model is larger than the RAM, on every token: Qwen3.8-Flash-Next on a Radeon 860M with 46 GB of RAM gets 37376 tokens and generates 5.3 tokens/s, where with a quarter it got 137216 tokens and 4.9 tokens/s.

Automatic fitting counts the memory of the MTP context for the context size it chooses, not for the training context of the model: up to version 0.1.32 it counted 1300 MiB of VRAM for it with Qwen3.8-Flash-Next on an RTX 4060 Ti, where it takes about 160 MiB, and chose a shorter context for the model. With Qwen3.8-Flash-Next and MTP of 2 tokens on the RTX 4060 Ti, the fit now counts 130 MiB plus 4.57 KiB per context token for the MTP context and gives the model a context of 7936 tokens instead of 6912.

Before the context, automatic fitting chooses the micro-batch (`-ub`), the number of prompt tokens processed in
one pass. With the experts in RAM, every pass uploads the experts that are not in the cache to the GPU, so fewer,
larger passes process long prompts much faster, while generation does not change. Unless `-b` or `-ub` is given,
it takes the largest of 4096, 2048 and 1024 tokens whose extra buffers fit in an eighth of the VRAM left after the
dense weights, and raises the batch (`-b`) with it. On the reference machine a 2514-token prompt went from 214 to
755 tokens/s with 4096, at the same generation speed; the larger buffers shorten the default context (about 90000
tokens without MTP and 61000 with MTP on 16 GB). The log shows the choice in a line such as `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

When whole expert layers fill the GPU, they leave below the margin less memory than the next expert tensor needs, and nothing else uses it; automatic fitting then raises the micro-batch with that memory, so no expert leaves the GPU. The log shows it in a line such as `micro-batch size raised from 2048 to 4096 tokens (batch 4096) with the device memory that the whole expert layers leave unused`. Whether it is enough depends on the free VRAM: with Qwen3.8-Flash-Next and MTP on an RTX 4060 Ti it was not with the default margin, and it was with `--fit-target 800`. In that run the prompt was processed at 21-22 tokens/s instead of 12, with generation at 1.41-1.46 tokens/s (1.39 before).

With an explicit placement (`--gpu-layers`, `--tensor-split`, `--override-tensor`, `--cpu-moe` or `--n-cpu-moe`),
automatic fitting keeps it as given and only fits around it the context and the micro-batch that you did not set,
instead of stopping; the log says `the device placement was set explicitly`.

| Option | Variable | Effect |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_ARG_MOE_CACHE` (`0` or `false` disables) | Enable or disable the cache (enabled by default). |
| `--moe-cache-size MiB` | `MGS_ARG_MOE_CACHE_SIZE` | Fixed total cache size, split among the GPUs by the experts each one caches; `-1` (default) uses the free VRAM of every GPU. |
| `--moe-cache-reserve MiB` | `MGS_ARG_MOE_CACHE_RESERVE` | VRAM left free on every GPU besides the cache and the prefetch slots (default 1024). Increase it when other programs use the GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnostics: fixed share (0–1) of missing experts uploaded instead of the measured balance. |
| — | `MGS_MOE_CACHE_STATS` | Diagnostics: log the cache counters every N decoding steps (visible with `-lv 4`). |
| — | `MGS_MOE_CACHE_PROTECTED` | Diagnostics: share (0–1) of the part of every layer kept for experts used again (default 0.5); `0` gives a plain least-recently-used order per layer. |

`GET /api/expert-cache` returns whether the cache exists (`active`) or the next request still has to create it or
apply a new size (`pending`), the size, cached layers and slots, hits and misses, uploads, CPU experts, the
share of prefetch bytes served from the cache and the measured bandwidths (with several GPUs,
sizes and counters add up and the upload bandwidth is their mean). `POST /api/expert-cache` with
`{"size_mib": N}` changes the size without restarting (`-1` automatic, `0` frees the cache); the next request
applies it and starts with an empty cache:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

With `--metrics`, `/api/metrics` adds `mgs:moe_cache_*` counters and gauges (steps, hits, misses, uploads,
CPU experts, prefetch bytes, size, slots and bandwidths).

Reference measurement on an RTX 4060 Ti 16 GB with Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 CPU threads), compared with
`--n-cpu-moe 28` without cache: generation 26.9 → 35.7 tokens/s (+33%), prompt processing 767–778 →
718–730 tokens/s (−6%), and with MTP enabled 33.6 → 47.0 tokens/s (+40%). On the same machine, the official
llama.cpp with its own cache (`-cmoe --moe-cache-mib 10240`) reached 26.7 tokens/s, and 27.8 with MTP. Gains depend on the share of experts
that fits in VRAM, the routing locality of the model and the PCIe and RAM bandwidths.

<a id="threads"></a>
### CPU threads

The CPU computes the experts that are not in the cache. That work is limited by the memory bandwidth and
saturates with few threads, while using every logical CPU makes the threads compete with the one that drives
the GPU. Without `-t`, `mgs` uses the cores useful for maths, but never more than half of the logical CPUs it may
run on, except on a CPU of up to 8 logical CPUs without SMT, which uses all its cores: with so few cores the CPU is
the limit (on a 4-core i7-7700K with an RX 580, where the CPU computes 40 of the 41 expert layers, 4 threads
generated 17% faster than 2). On a machine with SMT (two threads per core) the cap is the number of physical cores,
as before; in a larger virtual machine that does not show SMT it is half of the virtual CPUs. On the reference machine (16 virtual
cores with a desktop session, Qwen3.6-35B-A3B Q8_0, without MTP), 4 to 12 threads generated at the same speed
(32.9–33.9 tokens/s) and 16 threads at 28.7 tokens/s. The startup log shows the choice in a line such as
`using 8 CPU threads for generation and 8 for batches`; `-t` and `-tb` set them explicitly.

<a id="kv-cache-type"></a>
### KV cache type

Unless `-ctk` (`--cache-type-k`) or `-ctv` (`--cache-type-v`) is given, `mgs` gives the KV cache the precision of the model's weights, judged by the type that holds most of its expert weights: unquantized models (F16, BF16, F32) keep an `f16` cache, models with more than 5 bits per weight (Q5_K_M, Q6_K, Q8_0) get `q8_0`, and the rest (Q4_K_M, IQ4_XS, MXFP4, Q3_K, Q2_K…) get `q4_0`, the two quantized types that flash attention reads on every backend. The rule is the same on every backend (CPU, CUDA, ROCm and Vulkan). A quantized cache needs flash attention, so the cache stays `f16` when flash attention is disabled (`-fa off`) or cannot run with that type on the GPU. The startup log shows the choice in a line such as `KV cache type chosen from the weights of the model: q8_0`, and the Model page shows the types in use.

A `q8_0` cache takes about half the memory of an `f16` one and a `q4_0` cache about 28%, so automatic fitting gives much more context in the same memory, or leaves that memory to the expert cache, at the cost of a slightly less precise attention and a few percent of speed. Reading a quantized cache takes some extra work on every backend, and CUDA (NVIDIA GPUs) and ROCm (AMD GPUs with a `--rocm` build), which share the same flash attention kernels, read it directly only when very few tokens are processed at a time (plain generation, not the prompt or the MTP verification of several tokens) with attention heads of 64, 128 or 256 values, and convert it to `f16` at every step otherwise. On a Radeon 860M (Vulkan) with Qwen3.6-35B-A3B Q6_K and without MTP, `q8_0` generated 3.8% fewer tokens per second than `f16` after a short prompt and 2.7% fewer after an 8060-token prompt, and processed that prompt 1.3% more slowly. Giving only one of the two options leaves the other at `f16`, and `-ctk f16 -ctv f16` restores the `f16` cache when the memory is enough and those few percent matter more than the context. The MTP context and the draft model of `--spec-draft-model` follow the same rule; `--spec-draft-type-k` and `--spec-draft-type-v` set their types.

<a id="checkpoints"></a>
### Context checkpoints

Models with recurrent or sliding-window layers (for example Qwen3.5/3.6/3.8 hybrids or Nemotron-H)
cannot roll their state back, so when a request changes an earlier part of the conversation the
server resumes from the newest checkpoint before the change. Checkpoints are placed at the start of user,
assistant and tool messages (where coding agents remove old reasoning or trim old tool results) and at least
every `--checkpoint-min-step` tokens (default 1024, previously 8192) without extra prompt passes. When
`--ctx-checkpoints` (32) is reached, the checkpoint closest to its neighbours is removed, so the others stay
spread over the conversation. In the reference agent scenario (14k tokens, eight tool results), trimming a late
tool result went from 19.3 s to 5.5 s and an early one from 18.0 s to 13.6 s. Each checkpoint keeps the
recurrent or sliding-window state in host memory (63 MiB for Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Updated options and diagnostics

Keep the default `--load-mode mmap` when host registration succeeds. Use `--load-mode none` only as a fallback when registration fails. `--load-mode dio` now stages reads through at most 64 MiB of extra memory; that is not a cap on total model RAM.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` was removed because it had no runtime effect. The old loading flags (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) and the `GGML_CUDA_FA_ALL_QUANTS` build flag are no longer accepted; see [removed compatibility](#removed-compatibility) for their replacements. For a manual CMake build, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` adds that K/V vector-kernel combination; comma-separated or quoted semicolon-separated lists are accepted. The default list is `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 is always included. `all` increases compilation work. Missing vector-kernel combinations use an F16 conversion fallback and may be slower.

MTP separates graphs with/without outputs, preserves original token order across concurrent sequences, and uses actual positions after images.

Automatic fitting counts the nextn (MTP) layers, so the first layer no longer stays on the CPU when MTP is disabled. MTP weights now always stay in GPU memory; see [MTP placement](#mtp-vram). The default `--checkpoint-min-step` is now 1024 instead of 8192; see [context checkpoints](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` and `--moe-cache-reserve` control the [MoE expert cache](#expert-cache). Automatic fitting no longer leaves models larger than the VRAM with a 4096-token context and now also chooses the micro-batch; see [context size](#context-size). The default number of CPU threads is now at most half of the logical CPUs; see [CPU threads](#threads). `-np` now defaults to one slot with the whole context, instead of the automatic mode with four slots sharing a unified KV cache, and `--kv-unified` is off by default; see [request queue](#request-queue). On an integrated GPU, automatic fitting now counts the GPU memory and the RAM as one budget and no longer creates the MoE expert cache; see [integrated GPUs](#integrated-gpu). Since 0.1.19, automatic fitting no longer stops when the placement is given explicitly: it keeps that placement and fits the context and the micro-batch around it; see [context size](#context-size).

With `--metrics`, authenticated `GET /api/metrics` includes `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total`, and `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Position indices start at zero; that series appears after a speculative request completes. Compare deltas of accepted/drafted counters when evaluating acceptance, and measure latency separately; acceptance alone does not prove a speedup. In router mode, select the model with `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, off by default) lets the media URLs of requests reach loopback, private and other non-public addresses; see [security limits](#security-limits).

Conversation exports read the persisted full message tree and metadata. If server tools return 403, new messages stop repeating that failed lookup; reopening the tools panel retries it. First-visit server defaults preserve settings already changed by the user, including the API key.

[Implementation and upstream sources](CODE.en-GB.md#upstream-review).

<a id="removed-compatibility"></a>
## Removed compatibility

Backward-compatibility shims have been removed. Old names are no longer
translated or warned about: the server rejects removed command-line options as
invalid arguments, ignores removed environment variables and answers 404 for
removed routes. Update scripts, service overrides, presets and clients to the
replacements below.

| Removed | Use instead |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (or `mmap+mlock` to combine both) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Nothing; it had no effect. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Old draft-model aliases such as `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` or `--draft-p-min` | The matching `--spec-draft-*` option, for example `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` or `--spec-draft-p-min`. `mgs --help` lists all of them. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` or `--spec-ngram-mod-n-max` / `--spec-draft-n-min` or `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | The `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` and `--spec-ngram-*-min-hits` option of the chosen n-gram mode |
| Environment variables, CMake options and C API symbols with the former upstream prefix, including the aliases that earlier versions had already dropped | The current name with the `MGS_` prefix (`mgs_` in the C API), for example `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` instead of a `NO_` variant, `MGS_ARG_LOAD_MODE` instead of the memory-mapping, locking and Direct I/O variables, `MGS_ARG_SPEC_DRAFT_N_MAX` and `MGS_ARG_SPEC_DRAFT_N_MIN` for the draft limits, or `MGS_BUILD_UI`; for the backend options, the matching `GGML_*` option, for example `GGML_CUDA` or `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `HF_HOME` and `MGS_CACHE` | Nothing; every download, also from Hugging Face, goes to `~/.cache/mgs/` (under `XDG_CACHE_HOME` when it is set). |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` or `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| `reasoning_format` value `deepseek-legacy` (request field and `--reasoning-format`) | `deepseek` or `auto`; reasoning is returned in `reasoning_content` |
| `reasoning_budget_end_tag` request field | `reasoning_budget_end_tags` (array of strings) |
| `reasoning_in_content` in `generation_settings` | Nothing; the field was removed. |
| Download of a prebuilt upstream interface from a Hugging Face bucket (the `USE_PREBUILT_UI` and `UI_HF_BUCKET` CMake options and the `HF_UI_VERSION` environment variable) | Nothing; the interface is always built from `frontend/` (`MGS_BUILD_UI=ON`) or left out. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Build number and commit: the CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` variables (also in `mgs-config.cmake`) and the `b0-unknown` string of `build_info`, `system_fingerprint` and the `User-Agent` | `MGS_VERSION`, set in `backend/cmake/build-info.cmake`; those fields now carry the version, such as `0.1.19` |
| Meta Llama models (the `llama`, `llama4` and `llama-embed` GGUF architectures and the Llama 4 vision projector) and the built-in `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` and `llama4` chat templates | Nothing; use one of the supported MoE models. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Installer option `--no-sudo` | Nothing; the installers always run as root without sudo. |
| Functions and fields that the C APIs had marked deprecated: in `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, the `mgs_token_*` aliases (for example `mgs_token_bos`, `mgs_token_get_text` or `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, the `defrag_thold` field of `mgs_context_params` and `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; in `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` and `ggml_upscale_ext`; in `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` and the `image_marker` field | The current function: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` instead of `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` and `media_marker`. Warm-up runs are done by the caller, and `defrag_thold` had no effect. |
| Gemma 4: the `gemma4` and `gemma4-assistant` architectures, the `gemma4v`, `gemma4uv`, `gemma4a` and `gemma4ua` projectors, the `peg-gemma4` chat format, the `gemma4` tokeniser and pre-tokeniser, and the `--vision-gemma-4b-default` and `--vision-gemma-12b-default` presets | Nothing; use one of the [supported families](#supported-models). |
| DeepSeek models older than V4.1: the `deepseek`, `deepseek2` (also the Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 and Youtu files converted with it), `deepseek2-ocr` (with the `deepseekocr` and `deepseekocr2` projectors), `deepseek32`, `mistral4` and `deepseek4` architectures, the `deepseek-llm` and `deepseek-coder` pre-tokenisers, the DeepSeek-V4 DSpark draft backbone and the detection of the DeepSeek-V3.2 chat template (DSML with `function_calls`) | DeepSeek-V4.1-Flash (`deepseek41`). |
| The other MoE architectures inherited from llama.cpp: `grok`, `nomic-bert-moe`, `qwen2moe`, `qwen3moe`, `qwen3next`, `qwen3vlmoe`, `phimoe`, `minicpm`, `jamba`, `cohere2moe`, `dbrx`, `olmoe`, `arctic`, `glm4moe`, `exaone-moe`, `granitemoe`, `granitehybrid`, `bailingmoe`, `bailingmoe2`, `dots1`, `afmoe`, `laguna`, `ernie4_5-moe`, `hunyuan-moe`, `hy_v3`, `gpt-oss`, `lfm2moe`, `smallthinker`, `llada-moe`, `grovemoe`, `minimax-m2`, `rnd1`, `mistral3`, `step35`, `kimi-linear` and `mellum`, and the chat formats of Ministral 3, gpt-oss, Functionary v3.2, Kimi-K2, Cohere2-MoE, LFM2, GigaChat3 and MiniCPM5 | Nothing; use one of the [supported families](#supported-models). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) and the built-in templates of the non-Jinja engine (`--chat-template` with a name such as `deepseek3`, `chatglm4` or `mistral-v7-tekken`) | Nothing; chat templates always go through Jinja. `--chat-template chatml` still selects ChatML; for another template pass its Jinja source with `--chat-template` or `--chat-template-file`. Requests with `tools` no longer need any flag. |
| `-hfr` | `-hf` |
| Second long names of an option: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe` (router preset keys follow the same names) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, named after their options |
| `thinking_budget_tokens` request field | `reasoning_budget_tokens` (Anthropic clients keep sending `thinking.budget_tokens`) |
| `content` in `POST /api/embeddings` and `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, the `mgs_chat_message` struct and `MGS_FILE_MAGIC_GGLA` in `mgs.h`; `mtmd_image_tokens_get_id` in `mtmd.h` | Nothing; chat formatting is done by the Jinja templates of `common/chat.h`, and the others had no use. |
| `MGS_API_KEY`, `MGS_ARG_BATCH`, `MGS_ARG_UBATCH`, `MGS_ARG_N_PREDICT`, `MGS_ARG_N_PARALLEL`, `MGS_ARG_N_GPU_LAYERS`, `MGS_ARG_EMBEDDINGS`, `MGS_ARG_RERANKING`, `MGS_ARG_LOG_VERBOSITY`, `MGS_ARG_ROPE_SCALING_TYPE`, `MGS_ARG_SPEC_DRAFT_HF_REPO`, `MGS_ARG_STATIC_PATH`, `MGS_ARG_ENDPOINT_METRICS`, `MGS_ARG_ENDPOINT_PROPS`, `MGS_ARG_ENDPOINT_SLOTS` | `MGS_ARG_API_KEY`, `MGS_ARG_BATCH_SIZE`, `MGS_ARG_UBATCH_SIZE`, `MGS_ARG_PREDICT`, `MGS_ARG_PARALLEL`, `MGS_ARG_GPU_LAYERS`, `MGS_ARG_EMBEDDING`, `MGS_ARG_RERANK`, `MGS_ARG_VERBOSITY`, `MGS_ARG_ROPE_SCALING`, `MGS_ARG_SPEC_DRAFT_HF`, `MGS_ARG_PATH`, `MGS_ARG_METRICS`, `MGS_ARG_PROPS`, `MGS_ARG_SLOTS`: every variable is `MGS_ARG_` followed by the long name of its option in capitals, with `_` instead of `-` |
| `MGS_MOE_CACHE`, `MGS_MOE_CACHE_MIB`, `MGS_MOE_CACHE_RESERVE_MIB` | `--moe-cache`/`--no-moe-cache`, `--moe-cache-size` and `--moe-cache-reserve`, or `MGS_ARG_MOE_CACHE`, `MGS_ARG_MOE_CACHE_SIZE` and `MGS_ARG_MOE_CACHE_RESERVE` |
| `MTMD_BACKEND_DEVICE`, `MTMD_DEBUG_EMBEDDINGS` and the CMake option `MTMD_VIDEO` | `MGS_MTMD_BACKEND_DEVICE`, `MGS_MTMD_DEBUG_EMBEDDINGS` and `MGS_MTMD_VIDEO` |
| `--spec-mtp-auto`, `--no-spec-mtp-auto` (`MGS_ARG_SPEC_MTP_AUTO`) | Nothing: MTP turns on by itself; `--spec-type none` disables it and any other `--spec-type` replaces it. |
| `--spec-default` | `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, the values it set |
| The model presets `--tts-oute-default`, `--embd-gemma-default`, `--fim-qwen-1.5b-default`, `--fim-qwen-3b-default`, `--fim-qwen-7b-default`, `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec`, `--fim-qwen-30b-default`, `--gpt-oss-20b-default` and `--gpt-oss-120b-default` | Nothing; pass the model with `--model` or `-hf`. |
| The options of the command-line tools that `mgs` does not include, and their variables, such as `-p`/`--prompt`, `-f`/`--file`, `-sys`/`--system-prompt`, `-i`/`--interactive`, `-cnv`/`--conversation`, `--hellaswag`, `--winogrande`, `--kl-divergence`, `--save-all-logits`, `--ppl-stride`, `-o`/`--output`, `--chunk`, `--image`, `--audio`, `--video`, `-pps`, `-tgs`, `-npp`, `--diffusion-steps`, `-lr`/`--learning-rate`, `-mv`/`--model-vocoder`, `-hfv`/`--hf-repo-v`, `--tts-speaker-file`, `-fitp`/`--fit-print`, `--show-timings`, `--prompt-cache`, `--tensor-filter`, `-gan`/`--grp-attn-n` and the `--mtp`, `--dflash` and `--eagle3` download options | Nothing; the server never used them. |
| The `draft-eagle3`, `draft-dflash` and `draft-dspark` values of `--spec-type`, with the Eagle3, DFlash and DSpark draft models | `draft-mtp`, `draft-simple` or one of the n-gram types |
| The ggml backends other than CPU, CUDA, Vulkan and ROCm (HIP): BLAS, CANN, ET, Hexagon, Metal, MUSA, OpenCL, OpenVINO, RPC, SYCL, virtgpu, WebGPU, zDNN and ZenDNN, with their CMake options (such as `GGML_BLAS`, `GGML_METAL`, `GGML_RPC` or `GGML_SYCL`) and `--rpc` | CPU, CUDA (`--cuda`), ROCm (`--rocm`) or Vulkan (`--vulkan`) |
| In `mgs.h`, `mgs_get_logits`, `mgs_get_embeddings`, `mgs_encode`, `mgs_set_causal_attn`, `mgs_model_has_encoder`, `mgs_model_has_decoder`, `mgs_model_decoder_start_token`, `mgs_model_is_recurrent`, `mgs_get_ctx_other` and the `ctx_other` field of `mgs_context_params`; in `mgs-ext.h`, `mgs_set_embeddings_layer_inp`, `mgs_get_embeddings_layer_inp`, `mgs_model_target_layer_ids` and `mgs_model_target_layer_ids_n`; in `mtmd.h`, `mtmd_decode_use_non_causal`, `mtmd_image_tokens_get_n_tokens` and `mtmd_image_tokens_get_n_pos` | `mgs_get_logits_ith`, `mgs_get_embeddings_ith` or `mgs_get_embeddings_seq`, `mgs_decode` (which also runs models without a KV cache), `mtmd_input_chunk_get_n_tokens` and `mtmd_input_chunk_get_n_pos`; the others served encoder-decoder, embedding-only or dense models, or the shared KV cache between contexts, and have no replacement. |
| The Gemma 3 multimodal projectors (`gemma3`, `gemma3nv`, `gemma3na` and MobileNetV5) | Nothing; use the projector of one of the [supported families](#supported-models). |
| The OpenAI routes without `/v1` (`GET /api/models`, `POST /api/chat/completions`, `POST /api/responses`, `POST /api/audio/transcriptions`, `POST /api/chat/completions/input_tokens`, `POST /api/responses/input_tokens`) and `GET /api/v1/health` | The same route under `/api/v1` (`GET /api/v1/models`, `POST /api/v1/chat/completions`…) and `GET /api/health`. `POST /api/models` and `DELETE /api/models` of router mode stay. |
| The `ahead_api` and `ahead_webui` fields of the `mgs_queue` events | `running_api`, `running_webui`, `waiting_api` and `waiting_webui` ([request queue](#request-queue)) |
| Installer variables `MGS_INSTALL_SOURCE_DIR`, `MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`, `MGS_BUILD_JOBS` and `MGS_BUILD_BACKEND`, and the `--build-ui` option of the installers and of `build/build-local.sh` | `--build-dir`, `--prefix`, `--jobs` and `--cuda`, `--rocm`, `--vulkan` or `--cpu`; the sources always go to `/opt/moe-gguf-server-source` and the interface is always built unless `--no-ui` is given. |
| Configuring the web services as a separate step after the installation | The installer configures them in the same run (`--domain`, `--certificate`, `--private-key`, `--proxy-ip`); `configure-web.sh` remains for reconfiguring without rebuilding. |
| The inference service on `127.0.0.1:8080` (`--port 8080`) behind Apache | The Unix socket `/opt/moe-gguf-server/_/temp/web/mgs.sock` ([deployment](#deployment)) |

CMake reports removed `-D` variables as unused instead of applying them, so check
the configuration output of manual builds.

The web interface keeps its settings (`~/.mgs/config/ui-settings.json`) and its chats (`~/.mgs/chats/`) on the server,
and in the browser only its API key (`MgsUi.apiKey` in `localStorage`). It does not read or migrate the
settings that older versions kept in `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, the favourite models, the tool lists, the reasoning effort,
the dismissed MCP recommendations, the language and the colour mode of the former theme
library), so the settings start from
their defaults and the API key has to be entered again once. Nor does it read or
migrate data saved by older versions under other names, the standalone `theme`
key, inline reasoning and tool-call markers in old messages, pasted-text
attachments of the old `context` type, the old list format of dismissed MCP
recommendations or the server's former `webui_settings` field. Nor does it read the conversations that older versions kept in the browser (the `MgsUi` IndexedDB database):
to keep them, export them from the older version before upgrading and import the export afterwards, which saves
them on the server. Conversation import accepts only the JSONL and ZIP exports, within the limits listed
in [security limits](#security-limits); the old JSON export format is rejected. The old settings stay in the browser until they are deleted: to
remove them, open the browser console (F12) on the interface page and run
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
once for every address and port the interface was opened with.

Every download is cached in `~/.cache/mgs/` (under `XDG_CACHE_HOME` when it is set):
files downloaded with `-mu`/`--model-url` and, keeping their own structure
(`~/.cache/mgs/hub/models--<organization>--<repo>/`), Hugging Face downloads. No
variable changes that folder. The files that older versions saved in another cache
folder, including the Hugging Face cache (`~/.cache/huggingface/hub`), are downloaded again.

### Old model files

Compatibility code for the following old conversions was also removed.
Re-convert the original model, or download a current GGUF, to use them.

| Old file | Behaviour now |
| --- | --- |
| Grok-1 converted before August 2025 (without `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` or `grok.attn_logit_softcapping`) | Does not load: missing key. |
| Kimi-Linear with the unsplit `blk.N.attn_kv_b` tensor | Does not load: `attn_k_b`/`attn_v_b` are missing. That path already failed while building the graph. |
| Qwen3-Next files with only the merged `ssm_in` tensor | Does not load: missing tensor `blk.0.attn_qkv.weight`. |
| Files without `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) or `full_attention_interval` (Qwen3-Next), and Cohere2-MoE files with a single SWA period instead of the per-layer array | Does not load: missing key. |
| BPE vocabularies without `tokenizer.ggml.pre` or with the pre-tokeniser names `llama3`, `llama-v3`, `jina-es`, `jina-de` or `llada-moe`, and Kimi-K2 tokenisers without merges | Does not load: missing key, unknown pre-tokeniser type or missing merges. |
| GGUF version 1 or 2 files | Does not load: only GGUF version 3 is accepted. |
| mmproj files of the old one-off scripts: projector types `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x to 4.5, MiniCPM-o 2.6) and `adapter` (GLM-Edge-V), and files without a projector type | The projector does not load: unknown or missing projector type. |
| mmproj files of old conversions with `ffn_up` and `ffn_down` swapped | No longer corrected; the projector fails. |
| Kimi-K2.5 mmproj without `image_min_pixels`/`image_max_pixels` and Step3-VL without `preproc_image_size` | The projector does not load: missing key or tensor. |
| Files that only carry the old `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` and `tokenizer.ggml.middle_token_id` keys instead of the `tokenizer.ggml.fim_*_token_id` ones | Loads; the infill tokens are recognised by their text (for example `<PRE>` or `<\|fim_prefix\|>`), and infill does not work if they are not found. |

Fallbacks that current conversions or the most downloaded published GGUFs of the
supported families still depend on were kept, checked against their headers on
Hugging Face: the default `add_bos` of the `llama-bpe`, Tekken and similar
pre-tokenisers; the recurrent-layer layout of Qwen3.6 derived from
`full_attention_interval` (the most downloaded Qwen3.6 GGUFs have no
`recurrent_layers`); and the built-in GLM-5.2 indexer layout for GLM-5.2/5.3
files without `attention.indexer.types`.

<a id="maintenance"></a>
## Maintenance and troubleshooting

- Check which version is installed with `mgs --version` (it prints, for example, `version: 0.1.57`); the local build
  runs it at the end and CMake prints `mgs version: 0.1.57` while configuring. The interface shows the same version
  in Settings → Display → **Show build version information** (bottom-right corner) and under **Build Info** in
  the model information dialog. Every change to the project raises the last number: 0.1.0, 0.1.1, 0.1.2, etc.
- Repeat `curl | bash` to update from `main`, or the appropriate local installation command to rebuild and replace the binary.
  Pass the same web options again (`--domain`, `--proxy-ip`…), because every run rewrites the web configuration. The
  installer restarts both services at the end; to restart the inference service by hand, run
  `systemctl restart moe-gguf-server` on Debian, or `rc-service moe-gguf-server restart` on Alpine.
- Read `/root/webapp-install.log`, `journalctl -u moe-gguf-server` on Debian,
  or `/var/log/moe-gguf-server.log` on Alpine. Apache logs live under
  `/var/www/DOMAIN-logs/`.
- If CUDA compilation is unavailable, install the Toolkit externally or use
  `--rocm`, `--vulkan` or `--cpu`. If NCCL cannot be found, check repository availability and CUDA version,
  or explicitly choose `--no-nccl`. If ROCm cannot be found, install it with its development files or set
  `HIPCXX` ([ROCm](#rocm)).
- If UI compilation fails, check Node.js and npm versions, dependency downloads
  and the log. Do not substitute an upstream prebuilt interface.
- For deploy builds, use a fresh directory inside `_/temp/` after source layout changes.
  A project copy on SSHFS may prohibit executables; use a local disk on production.
- For the local build, rerun the one-liner above to download current `main`, or
  `bash build/build-local.sh` to use your local sources, and restart the copy you
  launched by hand. Read the terminal output if it fails or reports libraries
  outside `lib/`. The temporary `build.log` is deleted with the build directory.
- API 404 errors in older clients usually require changing the base URL to
  `/api/v1`. The installed API key belongs in the Authorization header.

See [CODE.en-GB.md](CODE.en-GB.md) for the implementation map and extension points.

<a id="license"></a>
## Licence

`mgs` is distributed under the [PolyForm NonCommercial License 1.0.0](../LICENSE). You can use,
modify and share it for any NonCommercial purpose: personal use, study, research, hobby projects,
and use by charitable organisations, educational institutions, public research organisations and
government institutions. Commercial use, such as a company using it for its business, needs a
separate licence from nipegun.

The portions that come from llama.cpp (MIT, The ggml authors), FreeToken (Apache 2.0) and the
bundled libraries keep their own licences; their notices are in `build/licenses/` and the local
build installs them, with `LICENSE`, in `~/.local/share/doc/mgs/licenses/`.
