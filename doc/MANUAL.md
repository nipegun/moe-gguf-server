# User manual

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Contents: [installation](#installation), [local build](#local-build), [deployment](#deployment),
[chat](#chat), [monitor](#monitor), [models and tools](#models), [supported models](#supported-models), [API](#api),
[request queue](#request-queue), [security limits](#security-limits), [Vulkan](#vulkan), [performance](#performance), [removed compatibility](#removed-compatibility),
[maintenance](#maintenance), [license](#license).

<a id="installation"></a>
## Installation and first launch

Install as root on the production host. Debian/Ubuntu accepts the installer
directly on standard input:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

No neighboring files need to be downloaded first. The entry script downloads the
complete `main` source archive from [this repository](https://github.com/nipegun/moe-gguf-server/), checks
it and synchronizes a managed copy in `/opt/moe-gguf-server-source`. Downloads
and build caches stay under the project's `_/temp/`. Curl and CA certificates
are needed to fetch the entry script; missing bootstrap dependencies are then
installed automatically.

Choose a different source directory and pass build options with:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

The remote directory must be absolute and on a local disk. The installer only
adopts an empty directory or reuses a copy bearing its source-origin marker;
unrelated directories are rejected. A lock prevents concurrent installation into
that copy. Subsequent `curl | bash` runs synchronize source files, including files
removed from `main`, while preserving `_/` and build caches. API keys, models and
deployed services live outside that source tree and are not changed. A failed
download or build does not install a new executable; rerun the same command to
retry. No Git operations are required.

A standalone downloaded script uses the same remote mode. Running
`bash deploy/install-update-reinstall-debian.sh` from a complete local checkout
builds that checkout without downloading sources. Arguments after `bash -s --`
are forwarded to the build installer. `--help` requires neither root nor a source
download. `--no-system-deps` also disables bootstrap package installation: curl,
CA certificates, tar, gzip, rsync, util-linux and coreutils must already exist.
Installer subprocesses cannot consume the script from the pipe; all phases share
one `/root/webapp-install.log` output stream.

Web configuration remains a separate step using
`/opt/moe-gguf-server-source/deploy/configure-web.sh` or its equivalent in the
chosen source directory.

Alpine still uses a complete local checkout and
`bash deploy/install-update-reinstall-alpine.sh`, with Bash installed first.
Debian defaults to CUDA/NCCL; Alpine defaults to CPU and does not support the
CUDA/NCCL installation path. Use `--cpu` for a Debian CPU build or `--vulkan` for AMD, Intel or NVIDIA GPUs
through Vulkan ([Vulkan](#vulkan)). CUDA Toolkit and NVIDIA drivers must already be installed. `--no-nccl` disables NCCL explicitly;
`--no-system-deps` leaves package management to the administrator.

The local UI needs Node.js 20.19+ on 20.x or 22.12+. `--no-ui` omits it. Other
options are `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST`, `--no-openssl` and `--build-ui`.
`MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`,
`MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND` and `CUDACXX` provide corresponding
environment overrides. Build and staging directories must be distinct,
non-overlapping descendants of the project's `_/temp/` directory.

The installer adds matching NCCL packages and NVIDIA's repository when needed,
but never installs or upgrades CUDA Toolkit or GPU drivers. It links the internal
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
bash deploy/install-update-reinstall-debian.sh --cpu
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
stops if synchronization fails. A failure at `Provisioning UI assets` still
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
| `~/.local/lib/mgs/` | The server binary (`mgs`, with the web interface and API documentation embedded) and every shared library it needs: the glibc dynamic loader and C library, the C++ runtime, OpenMP, OpenSSL, the glibc name-service modules and, in CUDA builds, the CUDA runtime, cuBLAS and NCCL. |
| `~/.local/share/doc/mgs/licenses/` | The project's PolyForm NonCommercial 1.0.0 license (`LICENSE`) and the third-party license notices copied from the source tree's `build/licenses/` (llama.cpp, FreeToken, Swagger UI, cpp-httplib and nlohmann/json). |

Only the user who runs the script can use this installation. Everything is resolved relative to where the launcher
is: it runs `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` from its prefix, and the server and its libraries search
`lib/mgs/` before anything else, including `LD_LIBRARY_PATH`. The only library still taken from the system is the
NVIDIA driver (`libcuda.so.1` and the `libnvidia-*` libraries it loads), because it must match the running kernel
module; with `--vulkan`, the Vulkan driver of the system ([Vulkan](#vulkan)). At the end the script lists, through the bundled loader, the libraries the server would load, fails if any
comes from outside `lib/mgs/`, and runs `--version` through the launcher.

Debian adds `~/.local/bin` to the `PATH` at login once that folder exists (through `~/.profile`), so after logging in
again you can run `mgs` by name; the script prints a reminder while it is not in the `PATH`. Launch it whenever you
want and stop it with Ctrl+C:

```bash
mgs --model /path/model.gguf
```

The full path works too: `~/.local/bin/mgs --model /path/model.gguf`.

Open `https://127.0.0.1:11443/` for the chat and `https://127.0.0.1:11443/api/doc/` for
the API documentation. Every option in this manual applies; for example,
`--models-dir` enables the multi-model router.

Requirements, which you install yourself as root because the script only checks
them: `cmake`, a C/C++ compiler and the static C library (`build-essential`, which
pulls in `libc6-dev`), `patchelf`, `libssl-dev` (or
`--no-openssl`), Node.js 20.19+/22.12+ with npm (or `--no-ui`), an existing CUDA
Toolkit (or `--cpu`) and NCCL with its development files (`libnccl2` and
`libnccl-dev`, or `--no-nccl`). `nvcc` is taken from `CUDACXX`, the `PATH` or
`/usr/local/cuda/bin/nvcc`. Ninja is used when installed.
Basic utilities include `tar`, `coreutils` and `util-linux`. Remote builds also
require `curl`, `ca-certificates` and `gzip`; no dependencies are installed automatically.

Options: `--prefix PATH` (another installation prefix), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`
and `--help`.

Good to know:

- The build is optimized for this computer's CPU and for the GPUs present while
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
- The prefix is locked while copying and replacing completed artifacts,
  preventing simultaneous updates of the same installation.
- A CUDA build needs the NVIDIA driver to start; use `--vulkan` on computers with another GPU and `--cpu` on
  computers without a GPU.

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
- `mgs` checks where the kernel really keeps that memory (`/proc/self/fdinfo`, published by amdgpu, i915 and xe). If
  another program fills the VRAM and the driver moves part of it to GTT, the log and the Monitor say so, and the
  Monitor shows the real location (VRAM and GTT) of every AMD or Intel GPU. The expert cache is allocated with a
  lower memory priority (`VK_EXT_memory_priority`), so the driver moves it before the MTP layers, the weights or the
  KV cache. Without resizable BAR, device buffers also skip the small VRAM window the CPU can map (256 MiB on an
  RX 580), which the desktop shares and the driver empties into GTT first.
- Only CUDA can pin the model pages that stay in RAM, so with Vulkan the experts reach the GPU through pageable
  memory. `--load-mode none` copies them into pinned host memory instead, which on an RTX 4060 Ti through Vulkan
  (Qwen3.6-35B-A3B Q4_K) raised expert uploads from 2.7 to 13.1 GB/s and generation from 27.5 to 36.6 tokens/s. On
  AMD and Intel that pinned memory counts against GTT: when the experts do not fit in it, loading fails with
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), as the 18.9 GB of experts of that
  model did with 16.8 GB of GTT on an RX 580. Use it only when the experts fit in GTT and in the free RAM.
- Size the model to the machine: the whole GGUF should fit in the free RAM plus VRAM, minus the KV cache and the
  compute buffers. With 8 GB of VRAM and 16 GB of RAM that means about 18–20 GB of weights; larger files are read
  from disk again and again and generation becomes very slow. The 21.6 GB Q4_K above reached 7.8 tokens/s on an
  RX 580 with 15 GB of RAM and 15.9 tokens/s once the machine had 31 GB (10.0 with `--spec-type none`).

<a id="deployment"></a>
## Production deployment

From the source directory (`/opt/moe-gguf-server-source` for remote installs),
after installing the binary, run:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

The domain must resolve to the production host. The `moe-gguf-server-web`
service runs a dedicated Apache instance as `moe-gguf-server`, without root.
Every startup selects the first available HTTP port starting at 11080 and HTTPS
port starting at 11443. Each sequence advances independently up to 65535, without
assigning the same port to both protocols. It inspects IPv4/IPv6 TCP listeners
and retries if Apache detects a port becoming occupied after selection. Other
configuration failures stop startup instead of being mistaken for port conflicts.

HTTP permanently redirects to the selected HTTPS port, preserving path and query.
The redirect is not cached. Apache forwards requests to inference on
`127.0.0.1:8080`, explicitly started with `--http --port 8080` and also running as
`moe-gguf-server`. Installation requires root;
neither service needs to run as root. No HAProxy is configured and no shared
Apache sites are imported or modified.

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
domain (`web-domain`), the Apache template (`apache2.conf.in`) and the settings of the
interface (`ui-settings.json`), all owned by `moe-gguf-server` with mode 0600.
Credentials and installation output are recorded in `/root/webapp-credentials.txt`
and `/root/webapp-install.log`, both mode 600. Reconfiguration preserves existing
keys, models and certificates. The TLS private key is root-owned, group-readable
by moe-gguf-server (mode 640), allowing the unprivileged web process to read it.

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
Use the same command after renewing source certificates. The script's
`--binary` option supports a nondefault executable location.

The service uses router mode. Put readable GGUF models in
`/opt/moe-gguf-server/models`, or use model management in the interface. For
specialized command-line options, use a systemd override on Debian/Ubuntu or the
OpenRC service configuration on Alpine. Keep the backend bound to loopback.
The web configurator regenerates its managed service file when rerun.

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
2. Select a model. In router mode, load an available model if none is loaded.
3. Type a message and send it. Enter sends by default; Shift+Enter inserts a line
   break. These shortcuts can be changed in settings.
4. Add files with the attachment action. Images, audio and video require compatible
   model modalities. PDF files can be extracted as text or converted to images;
   non-vision models use the text path.
5. Stop generation, skip reasoning when supported, edit a message, regenerate a
   response, or fork a conversation using the message actions.

The sidebar supports searching, pinning, renaming, deleting and bulk selection.
Conversations are stored in the browser using IndexedDB; export them before clearing
browser storage or changing browser profiles. The settings are kept by the server in
`~/.mgs/config/ui-settings.json` of the user that runs `mgs` (in the deployed service,
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), so every browser that opens the
interface shares them: generation parameters, display options, theme, language, MCP
servers, favorite models, disabled and always-allowed tools and the default reasoning
effort. Only the API key stays in each browser, because it authenticates the request
that loads the rest, so enter it once in every browser. Import/export is available in
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

Choose System, Light or Dark for the theme. Generation settings include context
usage, temperature, samplers, penalties and agentic turn limits. Some capabilities
depend on server flags or the selected model.

After every response the interface sends the conversation again, when the server is
idle, so that it is already encoded in the KV cache while you read and the next turn
starts sooner. It is on by default; turn off Settings → Developer → **Pre-fill KV
cache after response** to stop it.

While the server is resolving another request, a message you send waits in the
[queue](#request-queue), and its reply shows a notice instead of the progress text until it
starts: “Waiting in the queue: another request sent directly to the API is being resolved
first (requests ahead: N).” when a program uses the API at the same time, or the same notice
for a request of the web interface itself (another tab or browser, a generated title or the KV
cache pre-fill). The notice goes away as soon as your request starts.

<a id="monitor"></a>
## Monitor and status bar

**Monitor** sits under **Settings** in the sidebar (address `#/monitor`). While it is open it refreshes every
two seconds and shows:

- **Usage**: first one card per GPU of the machine (NVIDIA, AMD or Intel, even if no model uses it), titled with its
  label in bold and its name (for example **gpu-1**: NVIDIA GeForce RTX 4060 Ti; with several GPUs, gpu-1, gpu-2 and
  so on), with its usage, memory-controller (bandwidth) usage, VRAM in use, temperature and power draw against its
  limit; then one card per processor, titled the same way with its model (for example **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; with several processors, cpu-1, cpu-2 and so on), with its usage, one small vertical bar per
  logical CPU of that processor and the system load average. Every usage bar is green and fills in red as the usage
  grows; a bar whose value is unknown stays grey. NVIDIA GPUs are read through the driver's NVML library and AMD
  GPUs through `gpu_busy_percent` in sysfs; any other GPU (Intel, or NVIDIA without NVML) shows no usage, and its
  name and memory come from the inference backend once a model uses it.
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

`mgs` accepts MoE main models only. Dense models such as GPT-2 or Phi-2 are rejected before their weights are loaded; their decoder implementations are not compiled. Dense layers inside a MoE model and an explicitly supplied, compatible MTP-only head are retained.

<a id="supported-models"></a>
### Supported model families

| Family | GGUF architecture | Notes |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP when the GGUF carries it. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP when the GGUF carries it. Its per-layer embedding table (about 29 GB) is read from the model file on demand, so keep the file on a fast disk. Long contexts are computed with dense attention under the sparse selection: correct, but without the speedup of sparse kernels. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; images with its `glm5v` mmproj. |
| Kimi-K3 | `kimi-k3` | The published GGUFs carry no MTP. With Vulkan its residual mixing runs on the CPU. |
| DeepSeek-V4.1-Flash | `deepseek41` | Loads both published layouts: antirez's DwarfStar files (such as `antirez/deepseek-v4.1-flash-gguf`) and the files of the llama.cpp ecosystem. Its Engram table (about 188 GiB inside the file) is never loaded: every token reads its rows from the model file, so the file must stay available while `mgs` runs, preferably on an SSD, and Linux or another POSIX system is required. Files without a chat template use a built-in one, and `reasoning_effort` accepts a number from 1 to 100. No MTP and no images. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

Gemma 4 and the DeepSeek models older than V4.1 are no longer supported (see [removed compatibility](#removed-compatibility)).

With several selected GPUs, MTP checks all of them before falling back to RAM. Its memory estimate includes the output head. Integrated MTP buffers load before the other weights; an external MTP GGUF passed with `--spec-draft-model` loads before the main model, also with `--fit off`. The current explicit device selection still determines which GPUs are available.

Use `--reasoning on` or `--reasoning off` to control thinking. Passing `enable_thinking` through `--chat-template-kwargs` is rejected. Use the documented hyphenated option names; underscore spellings are rejected. Pass list values as a comma-separated value instead of repeating an option. Only `--spec-type` remains repeatable to compose speculative modes.

The model selector shows available, loaded and favorite models. Model details
include context size, modalities, quantization and the chat template. Router mode
manages separate child processes for loaded models; requests identify the model
through the `model` field or query parameter.

MCP servers are configured by URL with optional authorization headers. You can
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

Start `mgs` with `--local-shell-mcp-server` to let the models of the web chat use the terminal of the machine where
`mgs` runs:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- At startup `mgs` checks which shells are installed and offers one tool for each of them: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) and `psh.run` (PowerShell, `pwsh`). The log lists the shells it found; a shell
  that is not installed gets no tool.
- The interface detects it, adds it to the MCP servers as **local shell mcp server** and enables it. You can disable it there,
  or for a single conversation, and your choice is kept. When `mgs` starts without the option, the entry
  disappears.
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
  (`"event": "error"`). A call rejected before running (no command, a working directory that does not exist, an
  invalid timeout) also leaves its file, with the command as it was asked and the error. The chat keeps only the
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
are rejected, the default CORS origin becomes localhost and the log records every command with its exit code. Read
the command shown in the password window before typing a sudo password: the password authorizes that command. The
server only hands it to a password helper whose parent process is sudo running as root for your user, so a command
cannot start the helper by itself to read the password.

<a id="api"></a>
## HTTP API

Examples use 11443; substitute the HTTPS port from `ports.json` if different.

The default API root is `/api`. The UI remains at `/`. The root `/v1/...` routes
from the original server are no longer registered. OpenAI-compatible clients must
use `https://ai.example.com:11443/api/v1` as their base URL. Request/response field names
retain upstream compatibility; the C/C++ engine interfaces keep their current functions,
but not the ones that were marked deprecated (see [removed compatibility](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Set `MGS_API_KEY` to the installed key in your own shell. Native operations,
embeddings, tokenization, reranking, model management and resumable streams are
documented in the bundled Swagger UI at `/api/doc/`; its OpenAPI JSON is at
`/api/doc/openapi.json`, generated by the server on each request without a static JSON file. The inventory reflects the routes registered by the
running process, including router and optional compatibility routes. Disabled
features may return 403. A missing/invalid key returns 401 and model loading
returns 503. Health and model list routes keep public access for `GET` and `HEAD`; see
[security limits](#security-limits) for the routes that always need the key.

`--api-prefix /api/custom` relocates inference routes below `/api/custom`.
The root UI receives that prefix from `/api/config.js`; docs remain at `/api/doc/`.
The documentation namespace is reserved and cannot be used as an inference
prefix. Router children use `/api` internally. The server validates prefixes;
values outside `/api/` and paths with trailing slashes are rejected.

The Monitor uses `GET /api/monitor` (full report), `GET /api/monitor/memory` (memory snapshots of the loaded models)
and `GET /api/monitor/usage` (CPU and GPU usage only, also available while the model loads). They read counters and
snapshots published by the inference loop, never the task queue, so polling them neither wakes a sleeping model nor
resets its idle timer. Their fields are described in Swagger.

The interface reads its settings with `GET /api/ui/settings` (`{"settings": DOCUMENT}`, or
`{"settings": null}` before the first save) and saves them with `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, which replaces `~/.mgs/config/ui-settings.json` whole, with mode 0600). Both require
the API key and also answer while the model loads.

<a id="request-queue"></a>
### Request queue and parallel requests

By default `mgs` resolves one request at a time (`-np 1`) and gives it the whole context, so every request can
use all the context tokens, whether it comes from the web interface or directly from the API. The requests that
arrive meanwhile wait in a single queue and are resolved in the order they arrived: requests of the interface do
not jump ahead of API requests, nor the other way round. The API behaves the same whatever the interface
settings are.

To resolve several requests at the same time, start `mgs` with `-np N` (`--parallel N`, `MGS_ARG_N_PARALLEL`):
up to N requests run in parallel and the context is split among them, so each one gets a share (`n_ctx_slot` in
the `initializing` line of the log); with `--kv-unified` they share a single KV cache of the whole context
instead. Requests beyond N still wait in the queue.

The web interface adds `"mgs_client": "webui"` to its requests. While one of its streamed requests waits, the
stream begins with events such as

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: requests running or waiting ahead of it; `ahead_api` and `ahead_webui`: how many of them came directly
to the API or from the web interface), and a last one with every count at 0 when it starts. Clients that do not
send `mgs_client` never receive these events, so OpenAI-compatible clients get the usual stream. Swagger
describes the field with the other completion parameters.

<a id="security-limits"></a>
### Security limits

These limits protect the server and the interface from hostile requests, files and models. They are generous
enough for normal use: what exceeds them fails with a clear error instead of stopping the server.

- **API key.** Health and model list routes are public only for `GET` and `HEAD`. Reloading the model list
  (`GET /api/models?reload=1`) and adding or deleting models in router mode (`POST` and `DELETE /api/models`)
  need the key whenever one is configured.
- **Pages of other sites.** When server tools or MCP servers are enabled (`--tools`, `--agent`,
  `--mcp-servers-config`, `--mcp-servers-json` or `--local-shell-mcp-server`) and `--cors-origins` is not set,
  the server answers 403 to any request other than `GET`, `HEAD` or `OPTIONS` whose `Origin` header names
  another site. The interface (also behind Apache) and clients that send no `Origin`, such as curl or the SDKs,
  are not affected. `POST /api/tools` only accepts `Content-Type: application/json` (415 otherwise).
- **CORS proxy.** `/api/cors-proxy` refuses requests that a browser would open as a page, frame or worker
  (403) and returns every answer as inert content: it cannot run scripts or set cookies on the server's address.
- **Router listings.** In router mode, `status.args` and `status.preset` of `/api/models` show `<redacted>`
  instead of the values of `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  and `--mcp-servers-json`; the log does the same. The model processes still receive the real values.
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
  connect, 30 s in total, up to 5 redirects and 10 MiB, and the server runs at most 8 downloads at once.
- **Decoded media.** Images up to 32768 pixels per side and 100 megapixels, audio up to one hour, video frames up
  to 8192 pixels per side and 2 GiB of decoded frames per video, and at most 2 GiB of decoded images and audio per
  request.
- **Chat templates of the model.** `range()` produces at most one million elements, built strings stay below
  256 MiB, `indent()` and `tojson()` indentation below 256 spaces and nesting below 1024 statements; a template
  that exceeds them fails with a template error.
- **RPC.** In a build with the RPC backend, the RPC server checks the frame sizes and tensor bounds sent by its
  peers and closes only the offending connection. Expose it only to trusted machines: it still serves one client
  at a time.
- **Interface.** Links in messages keep only the http, https, mailto and tel schemes (or relative addresses), and
  image attachments must be embedded images. Conversation import accepts files up to 512 MB, ZIP archives with up
  to 10 000 entries, 256 MB per JSONL file and 1 GB extracted in total. A file with malformed records, or with a
  message that belongs to another conversation, is rejected whole; a conversation whose messages already exist
  is skipped and reported.

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
loading behavior. `--load-mode none` is recommended only when host registration fails
wholly or partly, not as a normal requirement.

Expert prefetch defaults to three VRAM slots. Each slot must fit the largest
transferred expert tensor. Values 2–8 in `GGML_SCHED_PREFETCH_EXPERTS` explicitly
choose a slot count. If memory allocation fails, at least two allocated slots are
kept when possible; otherwise regular transfers are used. Prefetch starts only
when the batch has at least twice as many expert selections as experts. Its pool
belongs to the first GPU that activates it; other GPUs keep selective copies.
Larger batches can help prefill but require more memory.

For diagnostics, set `GGML_CUDA_REGISTER_HOST=0` or
`GGML_SCHED_PREFETCH_EXPERTS=0` to disable the corresponding optimization.
Do not set those variables during normal operation unless intentionally opting out.

Automatic MTP examines GGUF metadata and tensor names before weight allocation,
including every shard. It requires a supported architecture, a model trunk and
all integrated heads. An external head or unsupported architecture does not
activate it. The default draft limit is three tokens. Explicit `--spec-type`
selection takes precedence; `--no-spec-mtp-auto` and `--spec-type none` disable
automatic selection. `--spec-draft-n-max` changes the limit. MTP affects generation,
not prefill or time to first token directly. Improvement depends on acceptance
rate and hardware; increasing the limit does not guarantee more speed.

<a id="mtp-vram"></a>
MTP weights always stay in GPU memory when the GPU can hold them: the MTP layers grafted onto the model, a
separate MTP GGUF passed with `--spec-draft-model`, and the output head they use. The options that move weights to
system memory (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, their `--spec-draft-*` equivalents and automatic
fitting) do not apply to them, and they stay on the GPU even with a low `--gpu-layers`. Automatic fitting places the
rest of the model around them, and the [MoE expert cache](#expert-cache) uses the memory left after that. If the MTP
layers do not fit in the free GPU memory with a 512 MiB margin, the log shows a warning and they are placed like any
other layer. With a separate MTP GGUF, the MTP layers grafted onto the main model are not loaded, so they do not take
memory twice; with `--spec-type none`, no MTP weight is loaded. The load log confirms the placement with a line such
as `keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)`, visible with `-lv 4`.

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
`--n-cpu-moe N` keeps some static layers on the GPU. The experts kept in RAM are pinned, so the RAM must hold
all of them plus the operating system.

<a id="context-size"></a>
When the model does not fit in VRAM and `-c` is not given, automatic fitting also chooses the context size: the
KV cache takes at most a quarter of the VRAM left after the dense weights, up to the training context of the
model, and the experts (the cache or whole expert layers) keep the rest. Previously such models always got the
minimum of 4096 tokens shared by every slot, which cut long answers short (`truncated = 1` in the log). The
log shows the result in a line such as `context size raised from 4096 to N tokens` and in `n_ctx_slot` of the
`initializing` line. `-c N` sets the context explicitly (more context leaves less VRAM to the cache and lowers
generation speed: 35.8 tokens/s with 4096 tokens and 33.9 with 122624 on the reference machine) and
`--fit-ctx N` raises the minimum. The MTP context takes the same size as the model context instead of reserving
VRAM for the whole training context.

Before the context, automatic fitting chooses the micro-batch (`-ub`), the number of prompt tokens processed in
one pass. With the experts in RAM, every pass uploads the experts that are not in the cache to the GPU, so fewer,
larger passes process long prompts much faster, while generation does not change. Unless `-b` or `-ub` is given,
it takes the largest of 4096, 2048 and 1024 tokens whose extra buffers fit in an eighth of the VRAM left after the
dense weights, and raises the batch (`-b`) with it. On the reference machine a 2514-token prompt went from 214 to
755 tokens/s with 4096, at the same generation speed; the larger buffers shorten the default context (about 90000
tokens without MTP and 61000 with MTP on 16 GB). The log shows the choice in a line such as `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

| Option | Variable | Effect |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE` (`0` disables) | Enable or disable the cache (enabled by default). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | Fixed total cache size, split among the GPUs by the experts each one caches; `-1` (default) uses the free VRAM of every GPU. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | VRAM left free on every GPU besides the cache and the prefetch slots (default 1024). Increase it when other programs use the GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnostics: fixed share (0–1) of missing experts uploaded instead of the measured balance. |
| — | `MGS_MOE_CACHE_STATS` | Diagnostics: log the cache counters every N decoding steps (visible with `-lv 4`). |

`GET /api/expert-cache` returns the size, cached layers and slots, hits and misses, uploads, CPU experts, the
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
the GPU. Without `-t`, `mgs` uses the cores useful for math, but never more than half of the logical CPUs it may
run on. On a machine with SMT (two threads per core) that is the number of physical cores, as before; in a
virtual machine that does not show SMT it is half of the virtual CPUs. On the reference machine (16 virtual
cores with a desktop session, Qwen3.6-35B-A3B Q8_0, without MTP), 4 to 12 threads generated at the same speed
(32.9–33.9 tokens/s) and 16 threads at 28.7 tokens/s. The startup log shows the choice in a line such as
`using 8 CPU threads for generation and 8 for batches`; `-t` and `-tb` set them explicitly.

<a id="checkpoints"></a>
### Context checkpoints

Models with recurrent or sliding-window layers (for example Qwen3.5/3.6/3.8 hybrids, Nemotron-H or Granite 4)
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

Complete single-head GLM-4.5-Air (`glm4moe`) GGUFs can now activate automatic MTP. Other existing eligibility checks and explicit-option precedence still apply. MTP separates graphs with/without outputs, preserves original token order across concurrent sequences, and uses actual positions after images.

Automatic fitting counts the nextn (MTP) layers, so the first layer no longer stays on the CPU when MTP is disabled. MTP weights now always stay in GPU memory; see [MTP placement](#mtp-vram). The default `--checkpoint-min-step` is now 1024 instead of 8192; see [context checkpoints](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` and `--moe-cache-reserve` control the [MoE expert cache](#expert-cache). Automatic fitting no longer leaves models larger than the VRAM with a 4096-token context and now also chooses the micro-batch; see [context size](#context-size). The default number of CPU threads is now at most half of the logical CPUs; see [CPU threads](#threads). `-np` now defaults to one slot with the whole context, instead of the automatic mode with four slots sharing a unified KV cache, and `--kv-unified` is off by default; see [request queue](#request-queue).

With `--metrics`, authenticated `GET /api/metrics` includes `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total`, and `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Position indices start at zero; that series appears after a speculative request completes. Compare deltas of accepted/drafted counters when evaluating acceptance, and measure latency separately; acceptance alone does not prove a speedup. In router mode, select the model with `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, off by default) lets the media URLs of requests reach loopback, private and other non-public addresses; see [security limits](#security-limits).

Conversation exports read the persisted full message tree and metadata. If server tools return 403, new messages stop repeating that failed lookup; reopening the tools panel retries it. First-visit server defaults preserve settings already changed by the user, including the API key.

[Implementation and upstream sources](CODE.md#upstream-review).

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
| Environment variables, CMake options and C API symbols with the former upstream prefix, including the aliases that earlier versions had already dropped | The current name with the `MGS_` prefix (`mgs_` in the C API), for example `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` instead of a `NO_` variant, `MGS_ARG_LOAD_MODE` instead of the memory-mapping, locking and Direct I/O variables, `MGS_ARG_SPEC_DRAFT_N_MAX` and `MGS_ARG_SPEC_DRAFT_N_MIN` for the draft limits, `MGS_CACHE` or `MGS_BUILD_UI`; for the backend options, the matching `GGML_*` option, for example `GGML_CUDA` or `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` or `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` or `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| `reasoning_format` value `deepseek-legacy` (request field and `--reasoning-format`) | `deepseek` or `auto`; reasoning is returned in `reasoning_content` |
| `reasoning_budget_end_tag` request field | `reasoning_budget_end_tags` (array of strings) |
| `reasoning_in_content` in `generation_settings` | Nothing; the field was removed. |
| Download of a prebuilt upstream interface from a Hugging Face bucket (the `USE_PREBUILT_UI` and `UI_HF_BUCKET` CMake options and the `HF_UI_VERSION` environment variable) | Nothing; the interface is always built from `frontend/` (`MGS_BUILD_UI=ON`) or left out. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Build number and commit: the CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` variables (also in `mgs-config.cmake`) and the `b0-unknown` string of `build_info`, `system_fingerprint` and the `User-Agent` | `MGS_VERSION`, set in `backend/cmake/build-info.cmake`; those fields now carry the version, such as `0.1.0` |
| Meta Llama models (the `llama`, `llama4` and `llama-embed` GGUF architectures and the Llama 4 vision projector) and the built-in `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` and `llama4` chat templates | Nothing; use one of the supported MoE models. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Installer option `--no-sudo` | Nothing; the installers always run as root without sudo. |
| Functions and fields that the C APIs had marked deprecated: in `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, the `mgs_token_*` aliases (for example `mgs_token_bos`, `mgs_token_get_text` or `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, the `defrag_thold` field of `mgs_context_params` and `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; in `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` and `ggml_upscale_ext`; in `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` and the `image_marker` field | The current function: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` instead of `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` and `media_marker`. Warm-up runs are done by the caller, and `defrag_thold` had no effect. |
| Gemma 4: the `gemma4` and `gemma4-assistant` architectures, the `gemma4v`, `gemma4uv`, `gemma4a` and `gemma4ua` projectors, the `peg-gemma4` chat format, the `gemma4` tokenizer and pre-tokenizer, and the `--vision-gemma-4b-default` and `--vision-gemma-12b-default` presets | Nothing; use one of the [supported families](#supported-models). |
| DeepSeek models older than V4.1: the `deepseek`, `deepseek2` (also the Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 and Youtu files converted with it), `deepseek2-ocr` (with the `deepseekocr` and `deepseekocr2` projectors), `deepseek32`, `mistral4` and `deepseek4` architectures, the `deepseek-llm` and `deepseek-coder` pre-tokenizers and the DeepSeek-V4 DSpark draft backbone | DeepSeek-V4.1-Flash (`deepseek41`). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) and the built-in templates of the non-Jinja engine (`--chat-template` with a name such as `deepseek3`, `chatglm4` or `mistral-v7-tekken`) | Nothing; chat templates always go through Jinja. `--chat-template chatml` still selects ChatML; for another template pass its Jinja source with `--chat-template` or `--chat-template-file`. Requests with `tools` no longer need any flag. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| Second long names of an option: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk` (router preset keys follow the same names) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, named after their options |
| `thinking_budget_tokens` request field | `reasoning_budget_tokens` (Anthropic clients keep sending `thinking.budget_tokens`) |
| `content` in `POST /api/embeddings` and `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, the `mgs_chat_message` struct and `MGS_FILE_MAGIC_GGLA` in `mgs.h`; `mtmd_image_tokens_get_id` in `mtmd.h` | Nothing; chat formatting is done by the Jinja templates of `common/chat.h`, and the others had no use. |

CMake reports removed `-D` variables as unused instead of applying them, so check
the configuration output of manual builds.

The web interface keeps its settings on the server (`~/.mgs/config/ui-settings.json`),
and in the browser only its API key (`MgsUi.apiKey` in `localStorage`) and its
conversations (the `MgsUi` IndexedDB database). It does not read or migrate the
settings that older versions kept in `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, the favorite models, the tool lists, the reasoning effort,
the dismissed MCP recommendations and the language), so the settings start from
their defaults and the API key has to be entered again once. Nor does it read or
migrate data saved by older versions under other names, the standalone `theme`
key, inline reasoning and tool-call markers in old messages, pasted-text
attachments of the old `context` type, the old list format of dismissed MCP
recommendations or the server's former `webui_settings` field. To keep the
conversations of an older version, export them from it before upgrading and import
the export afterwards. Conversation import accepts only the JSONL and ZIP exports, within the limits listed
in [security limits](#security-limits); the old JSON export format is rejected. The old settings stay in the browser until they are deleted: to
remove them, open the browser console (F12) on the interface page and run
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
once for every address and port the interface was opened with.

Files downloaded with `-mu`/`--model-url` are cached in `~/.cache/mgs/`, so the
ones that older versions saved in their former cache folder are downloaded again.
Hugging Face downloads keep using the Hugging Face cache (`~/.cache/huggingface/hub`,
or the folder in `MGS_CACHE`, `HF_HUB_CACHE` or `HF_HOME`).

### Old model files

Compatibility code for the following old conversions was also removed.
Re-convert the original model, or download a current GGUF, to use them.

| Old file | Behaviour now |
| --- | --- |
| Grok-1 converted before August 2025 (without `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` or `grok.attn_logit_softcapping`) | Does not load: missing key. |
| Kimi-Linear with the unsplit `blk.N.attn_kv_b` tensor | Does not load: `attn_k_b`/`attn_v_b` are missing. That path already failed while building the graph. |
| Qwen3-Next files with only the merged `ssm_in` tensor | Does not load: missing tensor `blk.0.attn_qkv.weight`. |
| Files without `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) or `full_attention_interval` (Qwen3-Next), and Cohere2-MoE files with a single SWA period instead of the per-layer array | Does not load: missing key. |
| BPE vocabularies without `tokenizer.ggml.pre` or with the pre-tokenizer names `llama3`, `llama-v3`, `jina-es`, `jina-de` or `llada-moe`, and Kimi-K2 tokenizers without merges | Does not load: missing key, unknown pre-tokenizer type or missing merges. |
| GGUF version 1 or 2 files | Does not load: only GGUF version 3 is accepted. |
| mmproj files of the old one-off scripts: projector types `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x to 4.5, MiniCPM-o 2.6) and `adapter` (GLM-Edge-V), and files without a projector type | The projector does not load: unknown or missing projector type. |
| mmproj files of old conversions with `ffn_up` and `ffn_down` swapped | No longer corrected; the projector fails. |
| Kimi-K2.5 mmproj without `image_min_pixels`/`image_max_pixels` and Step3-VL without `preproc_image_size` | The projector does not load: missing key or tensor. |
| Files that only carry the old `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` and `tokenizer.ggml.middle_token_id` keys instead of the `tokenizer.ggml.fim_*_token_id` ones | Loads; the infill tokens are recognized by their text (for example `<PRE>` or `<\|fim_prefix\|>`), and infill does not work if they are not found. |

Fallbacks that current conversions or the most downloaded published GGUFs of the
supported families still depend on were kept, checked against their headers on
Hugging Face: the default `add_bos` of the `llama-bpe`, Tekken and similar
pre-tokenizers; the recurrent-layer layout of Qwen3.6 derived from
`full_attention_interval` (the most downloaded Qwen3.6 GGUFs have no
`recurrent_layers`); and the built-in GLM-5.2 indexer layout for GLM-5.2/5.3
files without `attention.indexer.types`.

<a id="maintenance"></a>
## Maintenance and troubleshooting

- Check which version is installed with `mgs --version` (it prints, for example, `version: 0.1.0`); the local build
  runs it at the end and CMake prints `mgs version: 0.1.0` while configuring. The interface shows the same version
  in Settings → Display → **Show build version information** (bottom-right corner) and under **Build Info** in
  the model information dialog. Every change to the project raises the last number: 0.1.0, 0.1.1, 0.1.2, etc.
- Repeat `curl | bash` to update from `main`, or the appropriate local installation command to rebuild and replace the binary.
  Restart the service afterward: `systemctl restart moe-gguf-server` on Debian,
  or `rc-service moe-gguf-server restart` on Alpine.
- Read `/root/webapp-install.log`, `journalctl -u moe-gguf-server` on Debian,
  or `/var/log/moe-gguf-server.log` on Alpine. Apache logs live under
  `/var/www/DOMAIN-logs/`.
- If CUDA compilation is unavailable, install the Toolkit externally or use
  `--cpu`. If NCCL cannot be found, check repository availability and CUDA version,
  or explicitly choose `--no-nccl`.
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

No tests, deployment runs, GPU benchmarks or automated mobile browser checks are
performed by the coding assistant under this project's rules. Static syntax
checks are not a guarantee of runtime behavior. See [CODE.md](CODE.md) for the
implementation map and extension points.

<a id="license"></a>
## License

`mgs` is distributed under the [PolyForm NonCommercial License 1.0.0](../LICENSE). You can use,
modify and share it for any noncommercial purpose: personal use, study, research, hobby projects,
and use by charitable organizations, educational institutions, public research organizations and
government institutions. Commercial use, such as a company using it for its business, needs a
separate license from nipegun.

The license covers only nipegun's contributions. The portions that come from llama.cpp (MIT, The
ggml authors), FreeToken (Apache 2.0) and the bundled libraries keep their own licenses; their
notices are in `build/licenses/` and the local build installs them in
`~/.local/share/doc/mgs/licenses/`.
