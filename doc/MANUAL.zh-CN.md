# 用户手册

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

目录：[安装](#installation)、[本地构建](#local-build)、[部署](#deployment)、[聊天](#chat)、[监控](#monitor)、[模型与工具](#models)、[支持的模型](#supported-models)、[API](#api)、[请求队列](#request-queue)、[安全限制](#security-limits)、[Vulkan](#vulkan)、[性能](#performance)、[已移除的兼容性](#removed-compatibility)、[维护](#maintenance)、[许可证](#license)。

<a id="installation"></a>
## 安装与首次启动

请在生产主机上以 root 身份安装。Debian/Ubuntu 可以直接通过标准输入接收安装程序：

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

无需事先下载任何相邻文件。入口脚本会从[此仓库](https://github.com/nipegun/moe-gguf-server/)下载完整的 `main` 源码归档，对其进行校验，并在 `/opt/moe-gguf-server-source` 中同步一份受管理的副本。下载内容和构建缓存都保存在项目的 `_/temp/` 下。获取入口脚本需要 curl 和 CA 证书；缺失的引导依赖项随后会自动安装。

如需选择其他源码目录并传递构建选项，请使用：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

远程模式使用的目录必须是绝对路径，并且位于本地磁盘上。安装程序只会接管空目录，或者复用带有其源码来源标记的副本；无关的目录会被拒绝。锁机制可防止同时向该副本执行安装。之后的 `curl | bash` 运行会同步源文件（包括已从 `main` 中删除的文件），同时保留 `_/` 和构建缓存。API 密钥、模型和已部署的服务位于该源码树之外，不会被更改。下载或构建失败时不会安装新的可执行文件；重新运行同一条命令即可重试。整个过程无需任何 Git 操作。

单独下载的脚本使用同样的远程模式。在完整的本地源码副本中运行 `bash deploy/install-update-reinstall-debian.sh`，会直接构建该副本，而不下载源码。`bash -s --` 之后的参数会转发给构建安装程序。`--help` 既不需要 root 权限，也不需要下载源码。`--no-system-deps` 还会禁用引导软件包的安装：此时 curl、CA 证书、tar、gzip、rsync、util-linux 和 coreutils 必须已经存在。安装程序的子进程无法从管道中读走脚本；所有阶段共用同一个 `/root/webapp-install.log` 输出流。

Web 配置仍是一个单独的步骤，需使用 `/opt/moe-gguf-server-source/deploy/configure-web.sh` 或所选源码目录中的对应脚本来完成。

Alpine 仍然使用完整的本地源码副本和 `bash deploy/install-update-reinstall-alpine.sh`，并且需要先安装 Bash。Debian 默认使用 CUDA/NCCL；Alpine 默认使用 CPU，且不支持 CUDA/NCCL 安装路径。在 Debian 上，使用 `--cpu` 进行 CPU 构建，或使用 `--vulkan` 通过 Vulkan 支持 AMD、Intel 或 NVIDIA GPU（[Vulkan](#vulkan)）。CUDA Toolkit 和 NVIDIA 驱动程序必须事先安装好。`--no-nccl` 会显式禁用 NCCL；`--no-system-deps` 则把软件包管理交给管理员。

本地 UI 需要 Node.js 20.19+（20.x 系列）或 22.12+。`--no-ui` 会省略 UI。其他选项有 `--jobs N`、`--prefix PATH`、`--build-dir PATH`、`--cuda-architectures LIST`、`--no-openssl` 和 `--build-ui`。`MGS_BUILD_DIR`、`MGS_SOURCE_STAGE_DIR`、`MGS_INSTALL_PREFIX`、`MGS_BUILD_JOBS`、`MGS_BUILD_BACKEND` 和 `CUDACXX` 提供相应的环境变量覆盖。构建目录和暂存目录必须是项目 `_/temp/` 目录下互不相同、互不重叠的子目录。

安装程序会在需要时添加匹配的 NCCL 软件包和 NVIDIA 软件源，但从不安装或升级 CUDA Toolkit 或 GPU 驱动程序。它以静态方式链接内部库，并安装 `/usr/local/bin/mgs`。两个安装程序都不会使用或安装 sudo。

在 Debian/Ubuntu 上，`libnccl2` 和 `libnccl-dev` 会一起安装，且版本完全相同，即针对检测到的 CUDA 版本所选定的版本。该 APT 命令使用 `--allow-downgrades`，以免已安装的较新 NCCL 软件包阻碍所选版本的安装。该标志只会传递给 NCCL 的安装命令。

如果发行版自带的 Node.js 版本较旧，请先安装受支持的版本。UI 始终在本地编译；构建过程不能下载上游预构建的资源，因为这些资源缺少本项目的翻译和 API 路由。UI 构建失败会中止安装。

在完整的本地源码副本中，可以选择 CPU 构建，或列出安装选项：

```bash
bash deploy/install-update-reinstall-debian.sh --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

在 Alpine 上，运行其安装程序之前，请先用 `apk add --no-cache bash` 安装 Bash。

运行单个模型：

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

无需其他任何选项：`mgs` 会在其运行的机器上选择推理速度最快的设置。自动适配会将模型放到 GPU 上，并在系统内存允许时把 MoE 专家保留在 RAM 中，使空闲的 VRAM 成为 [MoE 专家缓存](#expert-cache)；它还会选择[微批次和上下文大小](#context-size)。CPU 线程数取决于处理器（[CPU 线程](#threads)），GGUF 中找到的 MTP 层会被激活并保留在 VRAM 中（[MTP](#mtp-vram)）。任何显式给出的选项（`-t`、`-b`、`-ub`、`-c`、`--gpu-layers`、`--n-cpu-moe`、`--override-tensor`……）都优先于自动选择。启动日志会显示所选的配置。

打开启动时打印的 HTTPS URL，通常是 `https://127.0.0.1:11443/`。加载大型模型可能需要一些时间。在模型请求报告正在加载期间，Web 界面和 API 文档仍然可用。`--help` 会列出继承而来的引擎选项。GGUF 文件、GPU 内存、上下文长度和模型专用的投影器必须与预期的工作负载相匹配。

无论采用哪种构建和安装方式，服务器代码默认都会启用 HTTPS。它会从 11443 开始绑定第一个可用的 HTTPS 端口，然后从 11080 开始绑定一个 HTTP 重定向端口，并跳过已选定的 HTTPS 端口。HTTP 会以永久性的 308 重定向指向实际的 HTTPS 端口，并保留所请求的主机、路径和查询字符串。日志会打印所选的 URL；发生端口冲突时，两个序列各自独立地向后递增。`--port N` 会固定主端口，若该端口已被占用则启动失败；`--port 0` 让操作系统自行选择端口。`--http-port N` 会改变重定向端口搜索的起始值。对应的环境变量是 `MGS_ARG_PORT` 和 `MGS_ARG_HTTP_PORT`。

在没有证书文件的情况下，可执行文件会在首次启动时创建一个自签名证书及其私钥（无需外部 `openssl` 命令），并将它们保存在 `~/.mgs/certificates/` 中（`fullchain.pem` 和 `privkey.pem`，其中私钥仅您本人可读）。之后的每次运行以及每次新构建都会复用它们，因此浏览器会保留其信任例外，无需重新加载；只有当这些文件缺失、彼此不匹配或证书已过期（365 天后）时，才会创建新的一对。该证书涵盖 `localhost`、环回地址、通过 `--host` 指定的主机，以及创建证书时本机的名称和网络地址；若要彻底消除浏览器警告，请将 `~/.mgs/certificates/fullchain.pem` 导入为受信任的证书；删除这两个文件即可创建新证书。如果该文件夹无法写入，每次运行都会在内存中生成自己的证书。若要使用受客户端信任的证书，请同时提供 `--ssl-cert-file /path/fullchain.pem` 和 `--ssl-key-file /path/privkey.pem`。缺少证书对或证书对无效都会中止启动。使用 `--no-openssl` 构建的版本无法提供 HTTPS，必须显式指定 `--http`（或 `MGS_ARG_HTTPS=false`）；纯 HTTP 从 11080 开始搜索端口。路由器的子进程在私有环回端口上显式使用 HTTP。Unix 套接字和 Vertex AI 的 `AIP_MODE=PREDICTION` 保留其 HTTP 传输方式。

CMake 配置位于 `build/CMakeLists.txt`。安装脚本和本地构建脚本都会把 `build/` 包含在其源码副本中，并自动选用它。如果现有缓存指向另一个源码目录，安装程序会重置已生成的 CMake 配置。本地构建总是在 `/tmp/` 下的一个全新目录中开始，并在退出时将其删除。若要直接使用 CMake，请在项目根目录下运行 `cmake -S build -B _/temp/cmake`；常规的安装程序和本地构建命令保持不变。HTTPS 默认使用系统的 OpenSSL（`MGS_OPENSSL=ON`）；在直接使用 CMake 的构建中，`-DMGS_BUILD_BORINGSSL=ON` 或 `-DMGS_BUILD_LIBRESSL=ON` 会改为下载并构建 BoringSSL 或 LibreSSL，而在不使用这两者的情况下指定 `-DMGS_OPENSSL=OFF`，则会构建不带 HTTPS 的服务器。

<a id="local-build"></a>
## 供手动使用的本地构建

`build/build-local.sh` 会在您自己的计算机上编译本项目，并为您的用户安装一份自包含的副本，由您手动启动。它不安装任何服务，不在后台启动任何程序，从不使用 root 或 sudo，也从不安装软件包。请以普通用户身份在任意文件夹中运行它：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

这条单行命令会在不使用 Git 的情况下，将 `main` 源码归档下载到 `/tmp` 下的私有构建文件夹中。退出时，该归档和解压出的源码会与其他临时构建文件一起被删除。单独保存的脚本副本也以同样的方式工作。在完整的本地项目副本中，可以用以下命令构建本地源码：

```bash
bash build/build-local.sh
```

通过绝对路径运行此脚本时，同样会使用包含该脚本的完整项目。`Using local sources: ...` 消息会确认正在编译的是哪个文件夹（包括其中的界面翻译）。如果运行以 `Error: Compilation failed.` 结束，则表示没有安装新的可执行文件；已有的安装会保留之前的界面。请阅读 `ninja: build stopped` 上方的第一条编译器错误，排除其原因后重新运行同一脚本。构建和安装成功后，请重启 `mgs` 并重新加载 Web 界面。

诸如 `std::vector` 没有名为 `mPushBack` 的成员，或 `subproc.cpp` 中出现 `no declaration matches` 之类的错误，表明源码中的名称不一致。请使用已修正的源文件，然后重新运行构建。界面构建还要求其配置导入与源模块导出的名称相匹配；缺失的导出会由 npm/Vite 步骤报告。

如果 SvelteKit 报告 `Page options are ignored when router.type === 'hash'` 并提到 `fLoad`，则说明聊天路由必须导出框架所要求的确切名称 `load`。修正后的路由保留了 API 密钥校验。构建过程会在生成资源和编译界面之前运行 `svelte-kit sync`，若同步失败则停止。即使 C++ 库已经编译完成，`Provisioning UI assets` 阶段的失败仍会阻止安装。

若要通过单行命令传递选项，请使用 `bash -s --`；例如，进行 CPU 构建：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` 会显示选项，而不会下载源码归档，也不会进行编译。

结果会以 `~/.local` 为前缀，为您的用户安装：

| 路径 | 内容 |
| --- | --- |
| `~/.local/bin/mgs` | 您运行的程序：一个小型静态启动器，它会相对于自身的位置找到 `../lib/mgs/`，并通过捆绑的加载器启动服务器。 |
| `~/.local/lib/mgs/` | 服务器二进制文件（`mgs`，内嵌 Web 界面和 API 文档）及其所需的全部共享库：glibc 动态加载器和 C 库、C++ 运行时、OpenMP、OpenSSL、glibc 名称服务模块，在 CUDA 构建中还包括 CUDA 运行时、cuBLAS 和 NCCL。 |
| `~/.local/share/doc/mgs/licenses/` | 本项目的 PolyForm NonCommercial 1.0.0 许可证（`LICENSE`），以及从源码树的 `build/licenses/` 复制过来的第三方许可声明（llama.cpp、FreeToken、Swagger UI、cpp-httplib 和 nlohmann/json）。 |

只有运行该脚本的用户才能使用此安装。一切都相对于启动器所在的位置来解析：它从其前缀运行 `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs`，服务器及其库会最先搜索 `lib/mgs/`，然后才搜索其他位置（包括 `LD_LIBRARY_PATH`）。唯一仍从系统获取的库是 NVIDIA 驱动程序（`libcuda.so.1` 及其加载的 `libnvidia-*` 库），因为它必须与正在运行的内核模块相匹配；使用 `--vulkan` 时，还有系统的 Vulkan 驱动程序（[Vulkan](#vulkan)）。最后，脚本会通过捆绑的加载器列出服务器将会加载的库，只要其中有任何一个来自 `lib/mgs/` 之外就报错退出，并通过启动器运行 `--version`。

Debian 会在 `~/.local/bin` 文件夹存在后，于登录时将其添加到 `PATH` 中（通过 `~/.profile`），因此重新登录后，您就可以直接按名称运行 `mgs`；在它尚未加入 `PATH` 时，脚本会打印一条提醒。您可以随时启动它，并用 Ctrl+C 停止它：

```bash
mgs --model /path/model.gguf
```

也可以使用完整路径：`~/.local/bin/mgs --model /path/model.gguf`。

打开 `https://127.0.0.1:11443/` 使用聊天，打开 `https://127.0.0.1:11443/api/doc/` 查看 API 文档。本手册中的所有选项均适用；例如，`--models-dir` 会启用多模型路由器。

要求（需要您自己以 root 身份安装，因为脚本只会检查它们）：`cmake`、C/C++ 编译器和静态 C 库（`build-essential`，它会引入 `libc6-dev`）、`patchelf`、`libssl-dev`（或 `--no-openssl`）、带 npm 的 Node.js 20.19+/22.12+（或 `--no-ui`）、已安装的 CUDA Toolkit（或 `--cpu`），以及 NCCL 及其开发文件（`libnccl2` 和 `libnccl-dev`，或 `--no-nccl`）。`nvcc` 取自 `CUDACXX`、`PATH` 或 `/usr/local/cuda/bin/nvcc`。如果已安装 Ninja，则会使用它。基本工具包括 `tar`、`coreutils` 和 `util-linux`。远程构建还需要 `curl`、`ca-certificates` 和 `gzip`；不会自动安装任何依赖项。

选项：`--prefix PATH`（另一个安装前缀）、`--jobs N`、`--cuda`、`--vulkan`、`--cpu`、`--cuda-architectures LIST`、`--build-ui`、`--no-ui`、`--no-openssl`、`--no-nccl` 和 `--help`。

须知：

- 构建会针对本机的 CPU 以及编译时存在的 GPU 进行优化。请在每台计算机上分别重新构建，而不要复制安装。
- 该安装作为前缀是可迁移的：将 `bin/mgs` 和 `lib/mgs/` 一起移动，并保持它们的 `bin/` 和 `lib/` 层级。请始终启动 `bin/mgs`；`lib/mgs/mgs` 是内部文件，不得直接运行，因为那样会绕过捆绑的加载器。
- 重新运行脚本会更新安装。它只替换 `bin/mgs`、`lib/mgs/` 和 `share/doc/mgs/licenses/`。已在运行的副本会继续工作；重启它即可使用新的构建。
- 如果前缀中已经存在 `bin/mgs`、`lib/mgs/` 或该许可证文件夹，且它们并非由脚本创建，脚本会拒绝使用该前缀；它会在 `lib/mgs/` 中留下一个隐藏的 `.moe-gguf-server-bundle` 标记。
- `--prefix PATH` 会安装到您的用户可写入的另一个前缀下；该前缀不能是 `/`，也不能是项目内的文件夹。
- 暂存的源码、构建文件、缓存、组装好的安装以及日志都使用一个私有的 `/tmp/moe-gguf-server-build.XXXXXXXX/` 文件夹。将结果复制到前缀后，脚本会删除该文件夹。出错时以及发生已处理的中断时，它也会进行清理。每次运行都从头开始编译，不会写入源码树或 `/opt/`。本地构建不接受 `--build-dir`。
- 在复制和替换已完成的构建产物期间，前缀会被锁定，以防止同时更新同一安装。
- CUDA 构建需要 NVIDIA 驱动程序才能启动；在配备其他 GPU 的计算机上请使用 `--vulkan`，在没有 GPU 的计算机上请使用 `--cpu`。

<a id="vulkan"></a>
## Vulkan（AMD、Intel 和 NVIDIA GPU）

除了 CUDA（默认）和 CPU 之外，`build/build-local.sh` 和 Debian 安装程序都可以通过 `--vulkan` 构建 Vulkan 后端，用于不支持 CUDA 的 GPU，例如 AMD Radeon 显卡。请先以 root 身份安装构建依赖项：

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

然后以普通用户身份运行：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

Debian 安装程序接受同样的选项，并会自行安装这些 Vulkan 软件包：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` 会显示 GPU，例如 `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`。

- 本地安装会像捆绑其他库一样捆绑 Vulkan 加载器（`libvulkan.so.1`），但 Vulkan 驱动程序来自系统：加载器会读取在 `/usr/share/vulkan/icd.d/` 中注册的驱动程序（AMD 的 Mesa RADV、Intel 的 ANV、NVIDIA 驱动程序），而这些驱动程序会加载各自的系统库，因为它们必须与内核和系统相匹配。请在将要运行该二进制文件的 Debian 版本上进行构建。
- 运行 `mgs` 的用户需要访问 GPU 渲染节点（`/dev/dri/renderD*`）。已登录桌面的用户已经拥有该权限；其他用户需要加入 `render` 组（以及 `video` 组）。`configure-web.sh` 会将服务账户加入这两个组。
- VRAM 中的 MTP、MoE 专家缓存、专家预取、`--fit` 和监控页面的工作方式与 CUDA 相同。在 AMD 上，监控页面从 sysfs 读取使用率、VRAM 和温度。
- `mgs` 只将真实的 VRAM（设备本地内存堆）计为 GPU 内存，从不计入 GTT。GTT 是 AMD 或 Intel GPU 可以映射的那部分系统 RAM（通常为系统 RAM 的一半）：它并不是额外的内存，而且 GPU 要通过 PCIe 读取它，速度比其 VRAM 慢许多倍。MTP 层、稠密权重、KV 缓存和专家缓存都分配在 VRAM 中；放不下的专家保留在普通 RAM 中，由 CPU 进行计算。
- `mgs` 会检查内核实际将这些内存保存在何处（`/proc/self/fdinfo`，由 amdgpu、i915 和 xe 提供）。如果另一个程序占满了 VRAM，导致驱动程序将其中一部分移到 GTT，日志和监控页面都会指出这一点，并且监控页面会显示每个 AMD 或 Intel GPU 的实际内存位置（VRAM 和 GTT）。专家缓存以较低的内存优先级（`VK_EXT_memory_priority`）分配，因此驱动程序会先移动它，然后才移动 MTP 层、权重或 KV 缓存。在没有可调整大小的 BAR（resizable BAR）时，设备缓冲区还会避开 CPU 可以映射的那一小块 VRAM 窗口（在 RX 580 上为 256 MiB），因为桌面会共享这块窗口，而且驱动程序会最先将它清空到 GTT。
- 只有 CUDA 能够锁定保留在 RAM 中的模型页面，因此在 Vulkan 下，专家要通过可分页内存到达 GPU。`--load-mode none` 会改为将它们复制到锁页主机内存中；在通过 Vulkan 运行的 RTX 4060 Ti 上（Qwen3.6-35B-A3B Q4_K），这使专家上传速度从 2.7 GB/s 提高到 13.1 GB/s，生成速度从 27.5 tokens/s 提高到 36.6 tokens/s。在 AMD 和 Intel 上，这部分锁页内存会计入 GTT：当专家放不进 GTT 时，加载会失败并报出 `radv/amdgpu: Not enough memory for command submission`（`ErrorDeviceLost`），该模型 18.9 GB 的专家在拥有 16.8 GB GTT 的 RX 580 上就出现了这种情况。请仅在专家能够同时放进 GTT 和空闲 RAM 时使用该选项。
- 根据机器来选择模型大小：整个 GGUF 应能放进空闲 RAM 与 VRAM 的总和中，还要减去 KV 缓存和计算缓冲区。对于 8 GB VRAM 和 16 GB RAM，这意味着大约 18–20 GB 的权重；更大的文件会被一次又一次地从磁盘重新读取，生成会变得非常缓慢。上面那个 21.6 GB 的 Q4_K 在配备 15 GB RAM 的 RX 580 上达到 7.8 tokens/s，而在机器拥有 31 GB RAM 后达到 15.9 tokens/s（使用 `--spec-type none` 时为 10.0）。

<a id="deployment"></a>
## 生产部署

在源码目录中（远程安装时为 `/opt/moe-gguf-server-source`），安装二进制文件之后，运行：

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

该域名必须解析到生产主机。`moe-gguf-server-web` 服务以 `moe-gguf-server` 身份运行一个专用的 Apache 实例，无需 root。每次启动时，它都会从 11080 开始选择第一个可用的 HTTP 端口，并从 11443 开始选择第一个可用的 HTTPS 端口。两个序列各自独立地递增，最高到 65535，且不会把同一个端口同时分配给两种协议。它会检查 IPv4/IPv6 TCP 监听端口，如果 Apache 在选定端口后检测到该端口已被占用，就会重试。其他配置错误会中止启动，而不会被误认为是端口冲突。

HTTP 会永久重定向到所选的 HTTPS 端口，并保留路径和查询字符串。该重定向不会被缓存。Apache 将请求转发给位于 `127.0.0.1:8080` 的推理服务，该服务通过 `--http --port 8080` 显式启动，同样以 `moe-gguf-server` 身份运行。安装需要 root；这两个服务都无需以 root 身份运行。不会配置 HAProxy，也不会导入或修改任何共享的 Apache 站点。

两个监听器都打开后，才会发布实际的 URL：

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

例如，HTTP 11081 和 HTTPS 11445 会重定向到 `https://ai.example.com:11445/`。该文件在关闭时删除，在启动时重新生成；重启后可能会选择不同的端口。客户端、防火墙以及任何外部代理都必须使用所发布的值。端口是[浏览器源（origin）](https://developer.mozilla.org/en-US/docs/Glossary/Origin)的一部分，因此不同的 HTTPS 端口拥有各自独立的本地历史记录/设置。已有的对话会保留在其原来的源中；可以通过导出/导入来转移它们。

系统账户没有交互式密码。模型存放在 `/opt/moe-gguf-server/models` 中，程序的配置位于服务账户的主目录 `/opt/moe-gguf-server/.mgs/config/` 中：API 密钥（`api.key`）、Web 域名（`web-domain`）、Apache 模板（`apache2.conf.in`）和界面设置（`ui-settings.json`），全部归 `moe-gguf-server` 所有，权限模式为 0600。凭据和安装输出记录在 `/root/webapp-credentials.txt` 和 `/root/webapp-install.log` 中，两者的权限模式均为 600。重新配置会保留现有的密钥、模型和证书。TLS 私钥归 root 所有，moe-gguf-server 组可读（模式 640），从而让无特权的 Web 进程能够读取它。

使用您自己的 TLS 证书：

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

证书位于 `/opt/moe-gguf-server/certificates/`（`fullchain.pem` 和 `privkey.pem`，归 `moe-gguf-server` 账户所有，私钥仅该账户可读）。如果没有提供文件，脚本会保留已有的证书，或者在首次运行时创建一个自签名证书，之后的每次重新安装都会复用它，因此浏览器会保留其信任例外。请在客户端上信任该证书，或者提供一个您的客户端已经信任的证书。提供的文件会被复制到该文件夹中；证书续期时必须刷新这些副本并重启 `moe-gguf-server-web`。该脚本不会获取或续期 Let's Encrypt 证书。续期源证书后，请使用相同的命令。脚本的 `--binary` 选项支持使用非默认位置的可执行文件。

该服务使用路由器模式。请将可读取的 GGUF 模型放在 `/opt/moe-gguf-server/models` 中，或者使用界面中的模型管理功能。如需专门的命令行选项，在 Debian/Ubuntu 上请使用 systemd 覆盖配置（override），在 Alpine 上请使用 OpenRC 服务配置。请让后端始终绑定在环回地址上。重新运行时，Web 配置程序会重新生成其管理的服务文件。

重启 Web 服务即可重新选择端口：

```bash
systemctl restart moe-gguf-server-web
```

在 Alpine 上请使用 `rc-service moe-gguf-server-web restart`。启动日志在 Debian 上通过 `journalctl -u moe-gguf-server-web` 查看，在 Alpine 上位于 `/var/log/moe-gguf-server-web.log`。Apache 诊断信息位于 `/opt/moe-gguf-server/_/temp/web/startup.log` 和 `error.log`；域名访问日志仍位于 `/var/www/DOMAIN-logs/`。独立模板为 `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`。请不要编辑 `_/temp/web/` 中生成的配置；每次启动时都会重新生成它。

<a id="chat"></a>
## 使用界面

1. 打开服务器 URL，在提示时输入其 API 密钥，或在“设置 → 常规”中输入。
2. 选择一个模型。在路由器模式下，如果尚未加载任何模型，请加载一个可用的模型。
3. 输入消息并发送。默认情况下按 Enter 发送；按 Shift+Enter 插入换行。这些快捷键可以在设置中更改。
4. 使用附件操作添加文件。图像、音频和视频需要模型支持相应的模态。PDF 文件可以提取为文本，也可以转换为图像；不支持视觉的模型使用文本方式。
5. 通过消息操作，可以停止生成、在支持时跳过思考、编辑消息、重新生成回复或为对话创建分支。

侧边栏支持搜索、置顶、重命名、删除和批量选择。对话使用 IndexedDB 存储在浏览器中；在清除浏览器存储或更换浏览器配置文件之前，请先将其导出。设置由服务器保存在运行 `mgs` 的用户的 `~/.mgs/config/ui-settings.json` 中（在已部署的服务中为 `/opt/moe-gguf-server/.mgs/config/ui-settings.json`），因此所有打开该界面的浏览器都共享这些设置：生成参数、显示选项、主题、语言、MCP 服务器、收藏的模型、已禁用和始终允许的工具，以及默认推理强度。只有 API 密钥保留在各个浏览器中，因为它用于对加载其余设置的请求进行身份验证，所以需要在每个浏览器中各输入一次。导入/导出功能位于设置中。导出内容可以包含附件，并且在明确选择时，还可以包含敏感的 API 密钥或 MCP 授权标头。

在“设置 → 常规”中选择语言。15 个选项依次为 `de-DE`、`en-GB`、`en-US`、`es-AR`、`es-ES`、`fr-FR`、`he-IL`、`hi-IN`、`it-IT`、`ja-JP`、`ko-KR`、`pt-BR`、`pt-PT`、`ru-RU`、`zh-CN`；默认为 `en-US`。巴西葡萄牙语和欧洲葡萄牙语有各自独立的翻译。希伯来语使用从右到左的界面；聊天输入框和纯文本用户消息遵循其文本的方向，而代码和终端输出则从左到右显示。更改语言会将所选语言保存到服务器的设置中，并重新加载界面；服务器会在 `/api/config.js` 中发布该设置，因此每个浏览器都会以该语言启动。请先保存其他尚未保存的设置。URL 查询参数 `lang` 会针对该页面覆盖这一设置，并在无法保存时保留所选语言。例如，在界面 URL 后附加 `?lang=ja-JP` 即可选择日语。

“设置 → 常规”中的**始终使用界面语言回答**默认处于关闭状态。启用它并保存设置后，模型在每次新的回复中都会被要求使用界面的语言和地区变体，包括在已有对话、重新生成的回复、续写、工具调用之后的轮次以及模型生成的标题中。对于 `es-AR`，该指令会明确要求使用阿根廷的 voseo 用法：“Seguí estos pasos”“tenés”和“podés”；`es-ES` 则要求使用西班牙本土的用法。英式/美式英语以及巴西/欧洲葡萄牙语也会收到各自不同的地区指令。

该指令会随每个新的推理请求和 KV 缓存预填充一起发送；它会与发出的系统消息合并，而不会编辑您保存的系统提示词或对话。更改界面语言会影响重新加载之后的新请求；正在生成的回复会保留它已收到的指令。禁用该设置后，后续请求将不再添加该指令。已有的回答不会被翻译。代码、命令、名称、引文以及明确要求的翻译会保留其所需的语言；围绕它们的说明则遵循界面语言。一致性取决于模型遵循指令的能力；这并不是自动翻译，也不保证完全遵循。外部 API 客户端不受此浏览器设置的影响。

主题可选择“跟随系统”“浅色”或“深色”。生成设置包括上下文使用情况、温度、采样器、惩罚以及智能体轮次限制。某些功能取决于服务器标志或所选模型。

每次回复之后，界面都会在服务器空闲时再次发送对话，以便在您阅读时对话已在 KV 缓存中完成编码，下一轮也能更快开始。该功能默认开启；若要停止，请关闭“设置 → 开发”中的**回复后预填充 KV 缓存**。

当服务器正在处理另一个请求时，您发送的消息会在[队列](#request-queue)中等待；在开始处理之前，其回复中会显示一条通知，而不是进度文本：如果有程序同时在使用 API，通知为“排队等待中：正在优先处理另一个直接发送到 API 的请求（前面的请求：N）。”；如果是 Web 界面自身的请求（另一个标签页或浏览器、生成的标题或 KV 缓存预填充），则显示针对 Web 界面请求的同类通知。一旦您的请求开始处理，该通知就会消失。

<a id="monitor"></a>
## 监控与状态栏

**监控**位于侧边栏中的**设置**下方（地址为 `#/monitor`）。该页面打开期间每两秒刷新一次，并显示：

- **使用率**：首先，机器上的每个 GPU（NVIDIA、AMD 或 Intel，即使没有任何模型使用它）各有一张卡片，标题为加粗的标签及其名称（例如 **gpu-1**: NVIDIA GeForce RTX 4060 Ti；有多个 GPU 时依次为 gpu-1、gpu-2 等），显示其使用率、内存控制器（带宽）使用率、已用 VRAM、温度以及功耗与其上限的对比；然后，每个处理器各有一张卡片，标题格式相同，附带其型号（例如 **cpu-1**: AMD Ryzen 9 5950X 16-Core Processor；有多个处理器时依次为 cpu-1、cpu-2 等），显示其使用率、该处理器每个逻辑 CPU 对应的一根小竖条以及系统平均负载。所有使用率条都是绿色的，并随着使用率的增长逐渐被红色填满；数值未知的条保持灰色。NVIDIA GPU 通过驱动程序的 NVML 库读取，AMD GPU 通过 sysfs 中的 `gpu_busy_percent` 读取；其他任何 GPU（Intel，或没有 NVML 的 NVIDIA）都不显示使用率，其名称和内存信息会在有模型使用它之后由推理后端提供。
- **内存**：每个 GPU（VRAM）各有一个条，RAM 也有一个条；没有内存数据的 GPU（例如尚无模型使用的 Intel GPU）没有条。每个条按用途划分，每种用途始终使用相同的颜色，其图例列出每个分段及其大小和占总量的比例，每行一项，顺序与条中的顺序相同。将鼠标悬停在某个分段上即可查看其大小。

  | 分段 | 内容 |
  | --- | --- |
  | MTP / 草稿模型权重 | 模型的 MTP（nextn）层的权重、通过 `--spec-draft-model` 传入的单独 MTP GGUF 的权重，或草稿模型的权重。 |
  | 专家权重 | 模型的路由专家张量。 |
  | 稠密权重 | 其他所有权重：注意力、共享专家、嵌入和输出。 |
  | KV 缓存 | 模型的上下文内存（KV 缓存或循环状态）。 |
  | MTP / 草稿模型 KV 缓存 | MTP 上下文或草稿模型上下文的上下文内存。 |
  | 计算缓冲区 | 计算图的临时缓冲区。 |
  | 专家缓存与预取 | MoE 专家缓存（池、标识符和暂存区）以及专家预取槽位。 |
  | 提示词缓存与检查点 | 通过 `--cache-ram` 保存的提示词以及各槽位的上下文检查点（RAM）。 |
  | 其他 | 以上各项均无法解释的已用内存：其他进程、桌面、驱动程序上下文、多模态投影器。 |
  | 空闲 | 设备或 RAM 的其余部分。 |

  从模型文件映射的权重即使在内核将该页缓存报告为可用时也计为 RAM，因此各分段之和可能超过“已用”内存；此时条中不显示“其他”，空闲内存也会显得更少。专家缓存在第一个请求时创建，因此在那之后才会出现。在路由器模式下，这些条包含所有已加载的模型，而且监控页面从不加载或唤醒模型。

每个页面底部的**状态栏**先显示每个 GPU 的使用率，再显示每个物理处理器的使用率，每个设备各占一个条，标签与监控卡片相同：`gpu-1`、`gpu-2`……以及 `cpu-1`、`cpu-2`……。这些条大小相同，每两秒刷新一次，并且与监控页面上的条一样，都是绿色的，并随着使用率的增长逐渐被红色填满；每个条的悬停文本会给出设备名称。没有 GPU 时，会显示一个空的 `GPU` 条；当这些条在窄屏上放不下时，状态栏可以横向滚动。点击状态栏即可打开监控页面。在模型加载期间它仍会继续工作。它默认显示；若要隐藏，请关闭“设置 → 显示”中的**显示状态栏**。

<a id="models"></a>
## 模型、工具与 MCP

`mgs` 只接受 MoE 主模型。诸如 GPT-2 或 Phi-2 之类的稠密模型会在加载其权重之前被拒绝；它们的解码器实现不会被编译。MoE 模型内部的稠密层，以及显式提供的、兼容的纯 MTP 头部，均会保留。

<a id="supported-models"></a>
### 支持的模型系列

| 系列 | GGUF 架构 | 说明 |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | 当 GGUF 包含 MTP 时支持 MTP。 |
| Qwen3.8-Flash-Next | `qwen4exp` | 当 GGUF 包含 MTP 时支持 MTP。其逐层嵌入表（约 29 GB）会按需从模型文件中读取，因此请将该文件放在高速磁盘上。长上下文在稀疏选择之下使用稠密注意力进行计算：结果正确，但没有稀疏内核带来的加速。 |
| GLM-5.3 | `glm-dsa` | MTP。 |
| GLM-5.3-Flash | `glm5-next` | MTP；配合其 `glm5v` mmproj 可处理图像。 |
| Kimi-K3 | `kimi-k3` | 已发布的 GGUF 不包含 MTP。使用 Vulkan 时，其残差混合在 CPU 上运行。 |
| DeepSeek-V4.1-Flash | `deepseek41` | 可加载两种已发布的布局：antirez 的 DwarfStar 文件（例如 `antirez/deepseek-v4.1-flash-gguf`）和 llama.cpp 生态系统的文件。其 Engram 表（在文件中约占 188 GiB）从不加载：每个 token 都从模型文件中读取其所需的行，因此在 `mgs` 运行期间该文件必须始终可用（最好放在 SSD 上），并且需要 Linux 或其他 POSIX 系统。没有聊天模板的文件会使用内置模板，`reasoning_effort` 接受 1 到 100 之间的数字。不支持 MTP，也不支持图像。 |
| MiniMax-M3、Nemotron-3、MiMo-V2.6 | `minimax-m3`、`nemotron_h_moe`、`mimo2` | |

Gemma 4 以及早于 V4.1 的 DeepSeek 模型已不再受支持（参见[已移除的兼容性](#removed-compatibility)）。

选择了多个 GPU 时，MTP 会先检查所有这些 GPU，然后才回退到 RAM。其内存估算包括输出头。集成的 MTP 缓冲区会先于其他权重加载；通过 `--spec-draft-model` 传入的外部 MTP GGUF 会先于主模型加载，即使使用 `--fit off` 也是如此。当前显式的设备选择仍然决定哪些 GPU 可用。

使用 `--reasoning on` 或 `--reasoning off` 来控制思考。通过 `--chat-template-kwargs` 传递 `enable_thinking` 会被拒绝。请使用文档中记载的带连字符的选项名称；带下划线的写法会被拒绝。列表值应以逗号分隔的形式传递，而不是重复同一个选项。只有 `--spec-type` 仍可重复使用，以组合多种推测解码模式。

模型选择器会显示可用的、已加载的和收藏的模型。模型详情包括上下文大小、模态、量化方式和聊天模板。路由器模式会为已加载的模型管理各自独立的子进程；请求通过 `model` 字段或查询参数来指定模型。

MCP 服务器通过 URL 进行配置，并可选择附带授权标头。您可以查看连接日志、可用的提示词、资源和工具。可选的服务器 CORS 代理必须使用引擎的相应标志显式启用；它只返回惰性内容，并拒绝作为页面被打开（参见[安全限制](#security-limits)）。浏览器对 HTTPS 页面中 HTTP 资源的限制仍然适用。当两个已启用的 MCP 服务器提供同名工具时，两份工具都不会提供给模型，界面会显示一条警告：要使用该工具，请只保留其中一个服务器处于启用状态。服务器的内置工具和浏览器 JavaScript 沙箱都是可选的。工具审批和智能体轮次限制仍可在界面中使用。轮次限制（设置 → 智能体 → **智能体轮次**，即停止前工具执行循环的最大次数，用于避免无限循环）默认为 50。

在服务器保留流式响应期间，流可以使用已保存的对话标识和游标重新连接。服务器已停止或流已过期时，无法保证重放。界面会报告重新连接失败，而不会悄悄地创建第二个生成请求。

### 本地 shell MCP 服务器

使用 `--local-shell-mcp-server` 启动 `mgs`，即可让 Web 聊天中的模型使用运行 `mgs` 的机器上的终端：

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- 启动时，`mgs` 会检查已安装了哪些 shell，并为每个 shell 提供一个工具：`bash.run`（Bash）、`zsh.run`（Zsh）、`sh.run`（sh）和 `psh.run`（PowerShell，`pwsh`）。日志会列出它找到的 shell；未安装的 shell 不会获得工具。
- 界面会检测到该服务器，将它以 **local shell mcp server** 的名称添加到 MCP 服务器中并启用它。您可以在那里禁用它，也可以只针对单个对话禁用它，您的选择会被保留。如果 `mgs` 启动时未带该选项，该条目就会消失。
- 每个工具都接受 `command`，以及可选的 `cwd`（工作目录；默认为运行 `mgs` 的用户的主目录）和 `timeout`（秒；默认为 60，最大为 3600）。返回结果是一个 JSON 对象，包含 `exit_code`、`stdout` 和 `stderr`，以及 `signal`、`timed_out`、`cancelled`、`timeout`、`seconds`、`shell` 和 `cwd`；过长的输出只保留开头和结尾部分。
- 每条命令都在一个新进程中运行，没有终端，输入为空：`cd` 和变量不会延续到下一次调用，也无法使用交互式程序（编辑器、提问）。超时到期或请求被取消时，该命令启动的所有进程都会被终止。留在后台运行的进程必须重定向其输出。
- 除非您选择始终允许该工具，否则界面会在每次调用前请求权限。
- `sudo` 可在没有终端的情况下使用（Linux）：当它需要密码时，无论您位于哪个页面，页面中央都会打开一个窗口，显示 sudo 的提示（`[sudo] password for user:`）、发出请求的工具和命令、一个密码字段，以及该字段下方的**接受**和**取消**按钮。光标已位于密码字段中，因此您可以立即输入密码；按 **Enter** 接受，按 **Escape** 取消；在窗口外点击不会关闭它。密码只会发送给运行 `mgs` 的机器上的 sudo：它不会保存在对话中，不会写入日志，模型也永远看不到它。在窗口等待期间，命令的超时计时会暂停；如果您按下**取消**，或者 5 分钟内无人应答，sudo 就会失败。由于每次调用都是一个新进程，sudo 在每次调用中都会再次询问；密码错误时最多询问三次，此时窗口会提示之前的密码可能有误。命令在各自独立的会话中运行，不使用启动 `mgs` 的那个终端，因此 sudo 永远不会在该终端中询问密码。只有 `mgs` 的 Web 界面能够应答：使用其他 MCP 客户端时，sudo 会像以前一样因缺少终端而失败。已部署的服务以 `moe-gguf-server` 账户运行，该账户通常既没有密码也没有 sudo 权限，因此此功能适用于以您自己的用户运行的 `mgs`（例如 `~/.local` 构建）。当 `mgs` 没有 `DISPLAY` 变量时，命令会收到 `DISPLAY=:mgs-askpass`，因为 sudo 只有在 `DISPLAY` 存在时才会在没有终端的情况下使用其密码辅助程序；尝试打开窗口的程序会立即失败。
- 在聊天中，每次调用都会显示其工具的名称，例如 `bash.run`。展开后，它会以等宽字体显示最多三个灰色框：**stdin** 显示纯命令，**stdout** 显示输出，**stderr** 显示错误文本。只有在命令打印了内容时才会出现 stdout，只有在出错时才会出现 stderr；退出码显示在它们的标题旁边，例如 `stdout (代码 2):`（没有输出时显示在 stdin 旁边）。stdin 框带有白色边框（浅色主题下为深灰色），stdout 框为绿色边框，stderr 框为红色边框，所有 shell 工具（`bash.run`、`zsh.run`、`sh.run` 和 `psh.run`）均是如此。
- 打开**设置 → 显示 → 自动展开本地 shell 命令**，即可让每次调用自动展开：每个框的内容一旦确定就会立即出现，stdin 在模型编写命令时出现，stdout 和 stderr 在命令结束时出现。
- 每次调用 shell 工具都会在 `~/.mgs/local-shell-mcp-server/` 中留下一个单独的文件（位于运行 `mgs` 的用户的主目录中；在已部署的服务中为 `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`），文件以调用时的本地日期和时间命名，精确到毫秒，例如 `y2026m10d12h23m59s59ms731.jsonl`；如果多次调用在同一毫秒内开始，则会追加 `-2`、`-3`……。这是一个 JSONL 文件，每行一个 JSON 对象：首先是命令（`"event": "command"`，包含工具、shell、工作目录和超时），在命令运行前写入；然后是完整的输出，不作任何截断，随着输出到达而写入，每个片段一行（`"event": "stdout"` 或 `"stderr"`，文本位于 `data` 中），顺序与命令写出的顺序一致；最后是结果（`"event": "result"`：退出码、信号、是否超时或被取消、耗时秒数以及 stdout 和 stderr 的字节数），或者命令无法启动的原因（`"event": "error"`）。在运行之前就被拒绝的调用（没有命令、工作目录不存在、超时无效）也会留下文件，其中包含所请求的命令和错误。聊天中只保留超长输出的开头和结尾，但文件中包含全部内容；如果 `mgs` 在命令运行期间停止，文件中也已经保存了输出。若要读取某次调用的完整 stdout，请运行 `jq -j 'select(.event == "stdout") | .data'
  FILE`。JSON 只能保存文本，因此非 UTF-8 的字节（二进制输出）会被替换为 `�`；打印数 GB 内容的命令会留下同样大小的文件。只有该用户才能读取该文件夹和这些文件，因为命令的输出可能包含机密信息；没有任何机制会删除它们，因此在不再需要时，请自行删除旧文件。

在已部署的服务中，请将 `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` 添加到 systemd 覆盖配置（`systemctl edit moe-gguf-server`）的 `[Service]` 部分，然后重启 `moe-gguf-server`；在 Alpine 上，请在 OpenRC 服务配置中导出同一个变量。

**安全性。** 命令以运行 `mgs` 的用户的权限运行（在已部署的服务中为 `moe-gguf-server` 账户，绝不是 `root`）。任何能够打开界面或调用 API 的人都可以运行这些命令，因此请仅在受信任的环境中启用该选项，并使用 API 密钥保护服务器。来自其他网站页面的请求会被拒绝，默认的 CORS 源变为 localhost，日志会记录每条命令及其退出码。在输入 sudo 密码之前，请阅读密码窗口中显示的命令：该密码授权的正是那条命令。服务器只会把密码交给这样的密码辅助程序：其父进程是以 root 身份、为您的用户运行的 sudo。因此，命令无法自行启动该辅助程序来读取密码。

<a id="api"></a>
## HTTP API

示例使用端口 11443；如果端口不同，请替换为 `ports.json` 中的 HTTPS 端口。

默认的 API 根路径是 `/api`。UI 仍位于 `/`。原始服务器的根路径 `/v1/...` 路由已不再注册。与 OpenAI 兼容的客户端必须使用 `https://ai.example.com:11443/api/v1` 作为其基础 URL。请求/响应的字段名称保持与上游兼容；C/C++ 引擎接口保留其当前的函数，但不再保留那些曾被标记为已弃用的函数（参见[已移除的兼容性](#removed-compatibility)）。

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

请在您自己的 shell 中将 `MGS_API_KEY` 设置为已安装的密钥。原生操作、嵌入、分词、重排序、模型管理和可恢复的流式响应都记录在 `/api/doc/` 处捆绑的 Swagger UI 中；其 OpenAPI JSON 位于 `/api/doc/openapi.json`，由服务器在每次请求时生成，无需静态 JSON 文件。该清单反映正在运行的进程所注册的路由，包括路由器路由和可选的兼容性路由。已禁用的功能可能会返回 403。缺少密钥或密钥无效时返回 401，模型加载期间返回 503。健康检查和模型列表路由对 `GET` 和 `HEAD` 保持公开访问；始终需要密钥的路由请参见[安全限制](#security-limits)。

`--api-prefix /api/custom` 会将推理路由移到 `/api/custom` 之下。根路径的 UI 从 `/api/config.js` 获取该前缀；文档仍位于 `/api/doc/`。文档命名空间是保留的，不能用作推理前缀。路由器的子进程在内部使用 `/api`。服务器会校验前缀；`/api/` 之外的值以及带有末尾斜杠的路径会被拒绝。

监控页面使用 `GET /api/monitor`（完整报告）、`GET /api/monitor/memory`（已加载模型的内存快照）和 `GET /api/monitor/usage`（仅 CPU 和 GPU 使用率，在模型加载期间也可用）。它们读取的是推理循环发布的计数器和快照，从不读取任务队列，因此轮询它们既不会唤醒处于休眠状态的模型，也不会重置其空闲计时器。它们的字段在 Swagger 中有说明。

界面通过 `GET /api/ui/settings` 读取其设置（`{"settings": DOCUMENT}`，或在首次保存之前返回 `{"settings": null}`），并通过 `POST /api/ui/settings` 保存设置（`{"settings": DOCUMENT}`，它会以模式 0600 整体替换 `~/.mgs/config/ui-settings.json`）。两者都需要 API 密钥，并且在模型加载期间也会响应。

<a id="request-queue"></a>
### 请求队列与并行请求

默认情况下，`mgs` 一次只处理一个请求（`-np 1`），并将整个上下文分配给它，因此无论请求来自 Web 界面还是直接来自 API，每个请求都可以使用全部上下文 token。在此期间到达的请求会在单一队列中等待，并按到达顺序依次处理：界面的请求不会插到 API 请求之前，反之亦然。无论界面如何设置，API 的行为都相同。

若要同时处理多个请求，请使用 `-np N`（`--parallel N`、`MGS_ARG_N_PARALLEL`）启动 `mgs`：最多 N 个请求并行运行，上下文在它们之间分配，因此每个请求获得其中一份（见日志中 `initializing` 行的 `n_ctx_slot`）；使用 `--kv-unified` 时，它们改为共享一个覆盖整个上下文的 KV 缓存。超出 N 个的请求仍会在队列中等待。

Web 界面会在其请求中添加 `"mgs_client": "webui"`。当它的某个流式请求处于等待状态时，流的开头是如下事件

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

（`ahead`：在它之前正在运行或等待的请求数；`ahead_api` 和 `ahead_webui`：其中直接发送到 API 的请求数和来自 Web 界面的请求数），请求开始时还会发送最后一个所有计数均为 0 的事件。不发送 `mgs_client` 的客户端永远不会收到这些事件，因此与 OpenAI 兼容的客户端会得到通常的流。Swagger 将该字段与其他补全参数一起进行了说明。

<a id="security-limits"></a>
### 安全限制

这些限制可保护服务器和界面免受恶意请求、文件和模型的侵害。它们足够宽松，可满足正常使用：超出限制的操作会以清晰的错误失败，而不会导致服务器停止。

- **API 密钥。** 健康检查和模型列表路由仅对 `GET` 和 `HEAD` 公开。只要配置了密钥，重新加载模型列表（`GET /api/models?reload=1`）以及在路由器模式下添加或删除模型（`POST` 和 `DELETE /api/models`）都需要该密钥。
- **其他网站的页面。** 当启用了服务器工具或 MCP 服务器（`--tools`、`--agent`、`--mcp-servers-config`、`--mcp-servers-json` 或 `--local-shell-mcp-server`）且未设置 `--cors-origins` 时，对于 `Origin` 标头指向其他网站的任何非 `GET`、`HEAD` 或 `OPTIONS` 请求，服务器都会返回 403。界面（包括位于 Apache 之后的界面）以及不发送 `Origin` 的客户端（例如 curl 或各种 SDK）不受影响。`POST /api/tools` 只接受 `Content-Type: application/json`（否则返回 415）。
- **CORS 代理。** `/api/cors-proxy` 会拒绝浏览器将作为页面、框架或 worker 打开的请求（403），并将每个响应都作为惰性内容返回：它无法在服务器的地址上运行脚本或设置 Cookie。
- **路由器列表。** 在路由器模式下，`/api/models` 的 `status.args` 和 `status.preset` 会显示 `<redacted>`，而不是 `--hf-token`、`--api-key`、`--api-key-file`、`--ssl-key-file`、`--mcp-servers-config` 和 `--mcp-servers-json` 的值；日志也是如此。模型进程仍会收到真实的值。
- **慢速客户端。** 服务器接收请求时，每次读取最多等待 30 秒（写入仍由 `--timeout` 控制，因此较长的回答不会被截断），并且最多有 256 个已接受的连接等待空闲线程。
- **请求正文。** 嵌套超过 1024 层数组或对象的 JSON 正文会以 400 被拒绝。JSON 模式（`response_format`、`json_schema`）最多可嵌套 256 层；空的 `pattern` 值以及没有可重复内容的量词会被拒绝。
- **远程媒体。** `image_url`、`input_audio` 和 `input_video` 中的 URL 只能指向公共互联网地址。环回地址、私有网络（10.0.0.0/8、172.16.0.0/12、192.168.0.0/16、100.64.0.0/10）、链路本地地址（包括云元数据地址 169.254.169.254）、IPv6 唯一本地地址以及其他非公共地址都会以 400 被拒绝，即使是通过重定向到达这些地址也是如此。若要允许它们（例如从您网络中的服务器读取图像），请使用 `--media-allow-private-urls`（`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`）启动 `mgs`；请仅在所有 API 客户端都受信任时这样做。每次下载的连接时间不超过 10 秒，总时间不超过 30 秒，最多 5 次重定向和 10 MiB，并且服务器最多同时进行 8 个下载。
- **解码后的媒体。** 图像每边最多 32768 像素且不超过 1 亿像素，音频最长一小时，视频帧每边最多 8192 像素且每个视频的解码帧最多 2 GiB，每个请求的解码图像和音频合计最多 2 GiB。
- **模型的聊天模板。** `range()` 最多生成一百万个元素，构建的字符串保持在 256 MiB 以下，`indent()` 和 `tojson()` 的缩进少于 256 个空格，嵌套少于 1024 条语句；超出这些限制的模板会以模板错误失败。
- **RPC。** 在带有 RPC 后端的构建中，RPC 服务器会检查对端发送的帧大小和张量边界，并且只关闭违规的连接。请只将其暴露给受信任的机器：它仍然一次只服务一个客户端。
- **界面。** 消息中的链接只保留 http、https、mailto 和 tel 协议（或相对地址），图像附件必须是嵌入的图像。对话导入接受最大 512 MB 的文件、最多 10 000 个条目的 ZIP 归档、每个 JSONL 文件最大 256 MB，解压后总计最大 1 GB。包含格式错误的记录，或包含属于其他对话的消息的文件，会被整体拒绝；消息已存在的对话会被跳过并报告。

安装命令仍会通过 HTTPS 下载当前的 `main` 分支；它们不会校验签名，也不会固定到某个修订版本。

<a id="performance"></a>
## 内存与性能

主机内存注册和专家预取默认处于启用状态。预取会让专家传输与计算重叠进行，对大型提示词尤其有效。效果取决于模型、内存容量以及 RAM/PCIe 带宽。

对于专家卸载到 CPU 的模型，`--cpu-moe` 或 `--n-cpu-moe N` 可以减少 VRAM 用量。主机注册会锁定已映射的权重页面，以加快主机到设备的复制。锁页内存无法被换出，因此请为操作系统留出足够的 RAM。这一注册路径是 POSIX 专用的；Windows 保留其原有的加载行为。仅当主机注册完全或部分失败时，才建议使用 `--load-mode none`，而不应将其作为常规要求。

专家预取默认使用三个 VRAM 槽位。每个槽位都必须能容纳所传输的最大专家张量。`GGML_SCHED_PREFETCH_EXPERTS` 中的值 2–8 可显式选择槽位数量。如果内存分配失败，会尽可能保留至少两个已分配的槽位；否则使用常规传输。只有当批次中的专家选择次数至少是专家数量的两倍时，预取才会启动。其池属于第一个激活它的 GPU；其他 GPU 保留选择性副本。更大的批次有助于预填充，但需要更多内存。

如需诊断，请设置 `GGML_CUDA_REGISTER_HOST=0` 或 `GGML_SCHED_PREFETCH_EXPERTS=0` 来禁用相应的优化。除非有意停用这些优化，否则在正常运行时请不要设置这些变量。

自动 MTP 会在分配权重之前检查 GGUF 元数据和张量名称（包括每个分片）。它要求架构受支持、具备模型主干以及所有集成的头部。外部头部或不受支持的架构不会激活它。默认的草稿上限为三个 token。显式的 `--spec-type` 选择优先；`--no-spec-mtp-auto` 和 `--spec-type none` 会禁用自动选择。`--spec-draft-n-max` 可更改该上限。MTP 影响的是生成，并不直接影响预填充或首个 token 的延迟。提升幅度取决于接受率和硬件；提高上限并不保证速度更快。

<a id="mtp-vram"></a>
只要 GPU 能够容纳，MTP 权重就始终保留在 GPU 内存中：包括嫁接到模型上的 MTP 层、通过 `--spec-draft-model` 传入的单独 MTP GGUF，以及它们所使用的输出头。将权重移到系统内存的选项（`--cpu-moe`、`--n-cpu-moe`、`--override-tensor`、它们对应的 `--spec-draft-*` 选项以及自动适配）对它们不适用，即使 `--gpu-layers` 设置得很低，它们也会留在 GPU 上。自动适配会围绕它们放置模型的其余部分，而 [MoE 专家缓存](#expert-cache)则使用此后剩余的内存。如果在保留 512 MiB 余量的情况下 MTP 层无法放入空闲的 GPU 内存，日志会显示一条警告，并像其他任何层一样放置它们。使用单独的 MTP GGUF 时，嫁接到主模型上的 MTP 层不会被加载，因此不会重复占用内存；使用 `--spec-type none` 时，不会加载任何 MTP 权重。加载日志会通过类似 `keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)` 的一行来确认放置位置，使用 `-lv 4` 时可见。

<a id="expert-cache"></a>
### MoE 专家缓存

当 MoE 专家保留在系统内存中、而层的其余部分在 GPU 上运行时，服务器会将所有层中最近使用的专家保存在一个占用空闲 VRAM 的缓存中；有多个 GPU 时，每个 GPU 都会在自己的 VRAM 中缓存分配给它的层。对于每个生成的 token（以及任何少于 32 个 token 的批次），在缓存中找到的专家在 GPU 上运行，缺失专家中的一部分会被上传到缓存，其余部分则同时在 CPU 上运行；这一比例取决于测得的 PCIe 和 CPU 带宽。长提示词仍使用专家预取，此时预取会在 GPU 内部复制已缓存的专家，只上传其余专家。除舍入误差外，输出结果与不使用缓存时的运行结果一致。GPU 和 CPU 的舍入方式不同，而且各自计算哪些专家取决于缓存内容，因此在不同的运行之间，较长的贪婪生成可能会在若干 token 之后偏离为另一段不同但同样有效的文本；官方 llama.cpp 的缓存也有同样的表现。

该缓存默认启用，并在模型、MTP 上下文和多模态投影器加载完成后，由第一个请求创建。在未指定 `--gpu-layers`、`--n-cpu-moe` 或 `--override-tensor` 时，如果所有专家都能在保留余量（4 GiB 或 RAM 的 10%）的情况下放入系统内存，自动适配就会将它们全部保留在系统内存中，并把空闲的 VRAM 留给缓存（有多个 GPU 时，它会按照各 GPU 的空闲内存将层分布到这些 GPU 上，使每个 GPU 都有缓存）；否则，它会像以前一样将完整的专家层放到 GPU 上，缓存则使用剩余的内存。显式的放置选项会得到遵守：`--cpu-moe` 会给缓存留出最多的 VRAM，而 `--n-cpu-moe N` 会将一些静态层保留在 GPU 上。保留在 RAM 中的专家会被锁页，因此 RAM 必须能够容纳所有这些专家以及操作系统。

<a id="context-size"></a>
当模型无法放入 VRAM 且未指定 `-c` 时，自动适配还会选择上下文大小：KV 缓存最多占用稠密权重之后剩余 VRAM 的四分之一，上限为模型的训练上下文，其余部分留给专家（缓存或完整的专家层）。以前，此类模型总是只能得到由所有槽位共享的最低 4096 个 token，导致较长的回答被截断（日志中的 `truncated = 1`）。日志会在类似 `context size raised from 4096 to N tokens` 的一行以及 `initializing` 行的 `n_ctx_slot` 中显示结果。`-c N` 可显式设置上下文（更大的上下文会给缓存留下更少的 VRAM，并降低生成速度：在参考机器上，4096 个 token 时为 35.8 tokens/s，122624 个 token 时为 33.9 tokens/s），`--fit-ctx N` 则可提高最小值。MTP 上下文采用与模型上下文相同的大小，而不是为整个训练上下文预留 VRAM。

在确定上下文之前，自动适配会先选择微批次（`-ub`），即一次处理的提示词 token 数。当专家位于 RAM 中时，每一次处理都会将不在缓存中的专家上传到 GPU，因此用次数更少、规模更大的处理可以大大加快长提示词的处理速度，而生成速度不变。除非指定了 `-b` 或 `-ub`，否则它会在 4096、2048 和 1024 个 token 中选取额外缓冲区能放入稠密权重之后剩余 VRAM 八分之一的最大值，并随之提高批次大小（`-b`）。在参考机器上，使用 4096 时，一个 2514 个 token 的提示词的处理速度从 214 tokens/s 提高到了 755 tokens/s，生成速度保持不变；更大的缓冲区会缩短默认上下文（在 16 GB 上，不使用 MTP 时约为 90000 个 token，使用 MTP 时约为 61000 个 token）。日志会在类似 `micro-batch size
raised from 512 to 4096 tokens (batch 4096)` 的一行中显示所做的选择。

| 选项 | 变量 | 作用 |
| --- | --- | --- |
| `--moe-cache`、`--no-moe-cache` | `MGS_MOE_CACHE`（`0` 表示禁用） | 启用或禁用缓存（默认启用）。 |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | 固定的缓存总大小，按各 GPU 所缓存的专家在 GPU 之间分配；`-1`（默认）使用每个 GPU 的空闲 VRAM。 |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | 除缓存和预取槽位之外，每个 GPU 上保留的空闲 VRAM（默认 1024）。当其他程序也使用 GPU 时，请增大该值。 |
| — | `MGS_MOE_CACHE_FILL_RATIO` | 诊断用：以固定比例（0–1）上传缺失的专家，而不是使用测得的平衡比例。 |
| — | `MGS_MOE_CACHE_STATS` | 诊断用：每 N 个解码步骤将缓存计数器写入日志（使用 `-lv 4` 时可见）。 |

`GET /api/expert-cache` 会返回大小、已缓存的层和槽位、命中和未命中次数、上传次数、CPU 专家数、从缓存提供的预取字节所占比例以及测得的带宽（有多个 GPU 时，大小和计数器会累加，上传带宽取其平均值）。使用 `{"size_mib": N}` 调用 `POST /api/expert-cache` 可在不重启的情况下更改大小（`-1` 表示自动，`0` 表示释放缓存）；下一个请求会应用该设置，并以空缓存开始：

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

使用 `--metrics` 时，`/api/metrics` 会添加 `mgs:moe_cache_*` 计数器和仪表（步骤数、命中、未命中、上传、CPU 专家、预取字节、大小、槽位和带宽）。

在 RTX 4060 Ti 16 GB 上使用 Qwen3.6-35B-A3B Q8_0（35 GiB，8 个 CPU 线程）进行的参考测量，与不使用缓存的 `--n-cpu-moe 28` 相比：生成速度 26.9 → 35.7 tokens/s（+33%），提示词处理速度 767–778 → 718–730 tokens/s（−6%），启用 MTP 时为 33.6 → 47.0 tokens/s（+40%）。在同一台机器上，官方 llama.cpp 使用其自身的缓存（`-cmoe --moe-cache-mib 10240`）达到 26.7 tokens/s，启用 MTP 时为 27.8 tokens/s。收益取决于能够放入 VRAM 的专家所占的比例、模型的路由局部性以及 PCIe 和 RAM 带宽。

<a id="threads"></a>
### CPU 线程

CPU 负责计算不在缓存中的专家。这项工作受内存带宽限制，用少量线程就会达到饱和，而使用所有逻辑 CPU 则会让这些线程与驱动 GPU 的线程相互争抢。在未指定 `-t` 时，`mgs` 会使用对数学运算有用的核心，但绝不会超过其可运行的逻辑 CPU 数量的一半。在支持 SMT（每个核心两个线程）的机器上，这就是物理核心数，与以前相同；在不显示 SMT 的虚拟机中，则为虚拟 CPU 数量的一半。在参考机器上（16 个虚拟核心，运行桌面会话，Qwen3.6-35B-A3B Q8_0，不使用 MTP），4 到 12 个线程的生成速度相同（32.9–33.9 tokens/s），16 个线程则为 28.7 tokens/s。启动日志会在类似 `using 8 CPU threads for generation and 8 for batches` 的一行中显示所做的选择；`-t` 和 `-tb` 可显式设置它们。

<a id="checkpoints"></a>
### 上下文检查点

具有循环层或滑动窗口层的模型（例如 Qwen3.5/3.6/3.8 混合模型、Nemotron-H 或 Granite 4）无法回滚其状态，因此当请求更改了对话中较早的部分时，服务器会从更改之前最新的检查点恢复。检查点设置在用户、助手和工具消息的开头（编码智能体会在这些位置删除旧的推理内容或裁剪旧的工具结果），并且至少每 `--checkpoint-min-step` 个 token（默认 1024，以前为 8192）设置一个，无需额外的提示词处理。当达到 `--ctx-checkpoints`（32）时，会删除与相邻检查点距离最近的那个检查点，从而使其余检查点继续分布在整个对话中。在参考智能体场景中（14k 个 token，八个工具结果），裁剪一个靠后的工具结果的耗时从 19.3 秒降至 5.5 秒，裁剪一个靠前的工具结果的耗时从 18.0 秒降至 13.6 秒。每个检查点都会在主机内存中保存循环状态或滑动窗口状态（Qwen3.6-35B-A3B 为 63 MiB）。

<a id="upstream-options"></a>
### 已更新的选项与诊断

主机注册成功时，请保持默认的 `--load-mode mmap`。仅在注册失败时才将 `--load-mode none` 作为后备方案使用。`--load-mode dio` 现在通过最多 64 MiB 的额外内存来暂存读取的数据；这并不是对模型 RAM 总用量的上限。

`GGML_CUDA_PEER_MAX_BATCH_SIZE` 已被移除，因为它在运行时没有任何效果。旧的加载标志（`--mmap`、`--no-mmap`、`--mlock`、`--direct-io`、`--no-direct-io`）和 `GGML_CUDA_FA_ALL_QUANTS` 构建标志已不再被接受；有关它们的替代项，请参见[已移除的兼容性](#removed-compatibility)。对于手动 CMake 构建，`-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` 会添加该 K/V 向量内核组合；可以使用逗号分隔的列表，或用引号括起的分号分隔列表。默认列表为 `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`；F16 始终包含在内。`all` 会增加编译工作量。缺失的向量内核组合会使用 F16 转换后备方案，速度可能较慢。

完整的单头 GLM-4.5-Air（`glm4moe`）GGUF 现在可以激活自动 MTP。其他现有的资格检查以及显式选项的优先规则仍然适用。MTP 会将有输出和无输出的计算图分开，在并发序列之间保留原始的 token 顺序，并在图像之后使用实际位置。

自动适配会计入 nextn（MTP）层，因此在禁用 MTP 时，第一层不再留在 CPU 上。MTP 权重现在始终保留在 GPU 内存中；参见 [MTP 放置](#mtp-vram)。默认的 `--checkpoint-min-step` 现在为 1024，而不是 8192；参见[上下文检查点](#checkpoints)。`--moe-cache`、`--no-moe-cache`、`--moe-cache-size` 和 `--moe-cache-reserve` 用于控制 [MoE 专家缓存](#expert-cache)。自动适配不再让大于 VRAM 的模型只使用 4096 个 token 的上下文，现在还会选择微批次；参见[上下文大小](#context-size)。CPU 线程的默认数量现在最多为逻辑 CPU 数量的一半；参见 [CPU 线程](#threads)。`-np` 现在默认为一个使用整个上下文的槽位，而不再是四个槽位共享统一 KV 缓存的自动模式，并且 `--kv-unified` 默认关闭；参见[请求队列](#request-queue)。

使用 `--metrics` 时，经过身份验证的 `GET /api/metrics` 包括 `mgs:spec_decode_num_draft_tokens_total`、`mgs:spec_decode_num_accepted_tokens_total`、`mgs:spec_decode_num_drafts_total` 和 `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`。位置索引从零开始；该序列会在推测请求完成后出现。评估接受率时，请比较已接受/已起草计数器的增量，并单独测量延迟；仅凭接受率并不能证明速度有所提升。在路由器模式下，请使用 `?model=MODEL_ID` 选择模型。

`--media-allow-private-urls`（`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`，默认关闭）允许请求中的媒体 URL 访问环回地址、私有地址和其他非公共地址；参见[安全限制](#security-limits)。

对话导出会读取持久化的完整消息树和元数据。如果服务器工具返回 403，新消息将不再重复那次失败的查询；重新打开工具面板会重试。首次访问时应用的服务器默认值会保留用户已更改的设置，包括 API 密钥。

[实现与上游来源](CODE.zh-CN.md#upstream-review)。

<a id="removed-compatibility"></a>
## 已移除的兼容性

向后兼容的适配代码已被移除。旧名称不再被转换，也不再发出警告：服务器会将已移除的命令行选项视为无效参数而拒绝，忽略已移除的环境变量，并对已移除的路由返回 404。请将脚本、服务覆盖配置、预设和客户端更新为下表中的替代项。

| 已移除 | 替代项 |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock`（或用 `mmap+mlock` 同时启用两者） |
| `-dio`、`--direct-io` / `-ndio`、`--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`、`--defrag-thold` | 无；它从未产生任何效果。 |
| `--swa-checkpoints` | `-ctxcp`、`--ctx-checkpoints` |
| `--webui`、`--no-webui`、`--webui-config`、`--webui-config-file`、`--webui-mcp-proxy`、`--no-webui-mcp-proxy` | `--ui`、`--no-ui`、`--ui-config`、`--ui-config-file`、`--ui-mcp-proxy`、`--no-ui-mcp-proxy` |
| 旧的草稿模型别名，例如 `-md`/`--model-draft`、`-ngld`/`--gpu-layers-draft`、`-td`/`--threads-draft`、`-devd`/`--device-draft`、`-ctkd`/`--cache-type-k-draft`、`-hfd`/`--hf-repo-draft` 或 `--draft-p-min` | 对应的 `--spec-draft-*` 选项，例如 `--spec-draft-model`、`--spec-draft-ngl`、`--spec-draft-threads`、`--spec-draft-device`、`--spec-draft-type-k`、`--spec-draft-hf` 或 `--spec-draft-p-min`。`mgs --help` 会列出所有这些选项。 |
| `--draft`、`--draft-n`、`--draft-max` / `--draft-min`、`--draft-n-min` | `--spec-draft-n-max` 或 `--spec-ngram-mod-n-max` / `--spec-draft-n-min` 或 `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`、`--spec-ngram-size-m`、`--spec-ngram-min-hits` | 所选 n-gram 模式对应的 `--spec-ngram-*-size-n`、`--spec-ngram-*-size-m` 和 `--spec-ngram-*-min-hits` 选项 |
| 带有以前上游前缀的环境变量、CMake 选项和 C API 符号，包括早期版本已经去除的别名 | 带有 `MGS_` 前缀的当前名称（C API 中为 `mgs_`），例如 `MGS_ARG_PORT`；用 `MGS_ARG_<NAME>=false` 代替 `NO_` 变体；用 `MGS_ARG_LOAD_MODE` 代替内存映射、锁定和 Direct I/O 变量；用 `MGS_ARG_SPEC_DRAFT_N_MAX` 和 `MGS_ARG_SPEC_DRAFT_N_MIN` 设置草稿上限；`MGS_CACHE` 或 `MGS_BUILD_UI`；对于后端选项，则使用对应的 `GGML_*` 选项，例如 `GGML_CUDA` 或 `GGML_NATIVE` |
| `HF_ENDPOINT`、`MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` 或 `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` 或 `POST /api/v1/embeddings` |
| `POST /api/reranking`、`POST /api/v1/reranking` | `POST /api/rerank`、`POST /api/v1/rerank` |
| `reasoning_format` 的值 `deepseek-legacy`（请求字段和 `--reasoning-format`） | `deepseek` 或 `auto`；推理内容在 `reasoning_content` 中返回 |
| `reasoning_budget_end_tag` 请求字段 | `reasoning_budget_end_tags`（字符串数组） |
| `generation_settings` 中的 `reasoning_in_content` | 无；该字段已被移除。 |
| 从 Hugging Face 存储桶下载上游预构建的界面（`USE_PREBUILT_UI` 和 `UI_HF_BUCKET` CMake 选项以及 `HF_UI_VERSION` 环境变量） | 无；界面始终从 `frontend/` 构建（`MGS_BUILD_UI=ON`），或者被省略。 |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| 构建编号和提交：CMake 的 `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` 变量（也包括 `mgs-config.cmake` 中的变量），以及 `build_info`、`system_fingerprint` 和 `User-Agent` 中的 `b0-unknown` 字符串 | 在 `backend/cmake/build-info.cmake` 中设置的 `MGS_VERSION`；这些字段现在携带版本号，例如 `0.1.0` |
| Meta Llama 模型（`llama`、`llama4` 和 `llama-embed` GGUF 架构以及 Llama 4 视觉投影器）以及内置的 `llama2`、`llama2-sys`、`llama2-sys-bos`、`llama2-sys-strip`、`llama3` 和 `llama4` 聊天模板 | 无；请使用受支持的 MoE 模型之一。 |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| 安装程序选项 `--no-sudo` | 无；安装程序始终以 root 身份运行，不使用 sudo。 |
| C API 中曾被标记为已弃用的函数和字段：`mgs.h` 中的 `mgs_load_model_from_file`、`mgs_free_model`、`mgs_new_context_with_model`、`mgs_n_ctx_train`、`mgs_n_embd`、`mgs_n_layer`、`mgs_n_head`、`mgs_n_vocab`、`mgs_get_state_size`、`mgs_copy_state_data`、`mgs_set_state_data`、`mgs_load_session_file`、`mgs_save_session_file`、`mgs_token_*` 别名（例如 `mgs_token_bos`、`mgs_token_get_text` 或 `mgs_token_fim_pre`）、`mgs_add_bos_token`、`mgs_add_eos_token`、`mgs_vocab_cls`、`mgs_set_warmup`、`mgs_sampler_init_grammar_lazy`、`mgs_context_params` 的 `defrag_thold` 字段以及 `MGS_STATE_SEQ_FLAGS_SWA_ONLY`；`ggml.h` 中的 `ggml_type_sizef`、`ggml_add1`、`ggml_add1_inplace`、`ggml_rope_custom`、`ggml_rope_custom_inplace` 和 `ggml_upscale_ext`；`mtmd.h` 中的 `mtmd_encode`、`mtmd_image_tokens_get_nx`、`mtmd_image_tokens_get_ny` 以及 `image_marker` 字段 | 当前的函数：`mgs_model_load_from_file`、`mgs_model_free`、`mgs_init_from_model`、`mgs_model_n_*`、`mgs_vocab_n_tokens`、`mgs_state_*`、`mgs_vocab_*`（用 `mgs_vocab_bos` 代替 `mgs_vocab_cls`）、`mgs_vocab_get_add_bos`、`mgs_vocab_get_add_eos`、`mgs_sampler_init_grammar_lazy_patterns`、`MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`、`ggml_row_size`、`ggml_add`、`ggml_rope_ext`、`ggml_interpolate`、`mtmd_encode_chunk`、`mtmd_image_tokens_get_decoder_pos` 和 `media_marker`。预热运行由调用方执行，而 `defrag_thold` 从未产生任何效果。 |
| Gemma 4：`gemma4` 和 `gemma4-assistant` 架构，`gemma4v`、`gemma4uv`、`gemma4a` 和 `gemma4ua` 投影器，`peg-gemma4` 聊天格式，`gemma4` 分词器和预分词器，以及 `--vision-gemma-4b-default` 和 `--vision-gemma-12b-default` 预设 | 无；请使用[受支持的系列](#supported-models)之一。 |
| 早于 V4.1 的 DeepSeek 模型：`deepseek`、`deepseek2`（也包括用它转换的 Kimi-K2.x、GLM-4.7-Flash、Moonlight、GigaChat3、Kanana-2、Mistral-Large-3 和 Youtu 文件）、`deepseek2-ocr`（以及 `deepseekocr` 和 `deepseekocr2` 投影器）、`deepseek32`、`mistral4` 和 `deepseek4` 架构，`deepseek-llm` 和 `deepseek-coder` 预分词器，以及 DeepSeek-V4 DSpark 草稿主干 | DeepSeek-V4.1-Flash（`deepseek41`）。 |
| `--jinja`、`--no-jinja`（`MGS_ARG_JINJA`）以及非 Jinja 引擎的内置模板（带有 `deepseek3`、`chatglm4` 或 `mistral-v7-tekken` 等名称的 `--chat-template`） | 无；聊天模板始终经由 Jinja 处理。`--chat-template chatml` 仍然选择 ChatML；如需其他模板，请通过 `--chat-template` 或 `--chat-template-file` 传入其 Jinja 源码。带有 `tools` 的请求不再需要任何标志。 |
| `-hfr`、`-hfrv` | `-hf`、`-hfv` |
| 选项的第二个长名称：`--usage`、`--n-predict`、`--sampling-seq`、`--temperature`、`--top-n-sigma`、`--typical-p`、`--no-mmproj-auto`、`--n-gpu-layers`、`--embeddings`、`--reranking`、`--log-verbose`、`--log-verbosity`、`--spec-draft-ncmoe`、`--kl-divergence-base`、`--output-file`、`--from-chunk` | `--help`、`--predict`、`--sampler-seq`、`--temp`、`--top-nsigma`、`--typical`、`--no-mmproj`、`--gpu-layers`、`--embedding`、`--rerank`、`--verbose`、`--verbosity`、`--spec-draft-n-cpu-moe`、`--save-all-logits`、`--output`、`--chunk`（路由器预设的键遵循相同的名称） |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`、`MGS_ARG_FIT_ESTIMATE`、`MGS_ARG_THINK`、`MGS_ARG_THINK_BUDGET`、`MGS_ARG_THINK_BUDGET_MESSAGE`、`MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`、`MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`、`MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`、`MGS_ARG_FIT_PRINT`、`MGS_ARG_REASONING_FORMAT`、`MGS_ARG_REASONING_BUDGET`、`MGS_ARG_REASONING_BUDGET_MESSAGE`、`MGS_ARG_SPEC_DRAFT_TYPE_K`、`MGS_ARG_SPEC_DRAFT_TYPE_V`、`MGS_ARG_SPEC_DRAFT_NGL`，均以其对应的选项命名 |
| `thinking_budget_tokens` 请求字段 | `reasoning_budget_tokens`（Anthropic 客户端仍继续发送 `thinking.budget_tokens`） |
| `POST /api/embeddings` 和 `POST /api/v1/embeddings` 中的 `content` | `input` |
| `mgs.h` 中的 `mgs_chat_apply_template`、`mgs_chat_builtin_templates`、`mgs_chat_message` 结构体和 `MGS_FILE_MAGIC_GGLA`；`mtmd.h` 中的 `mtmd_image_tokens_get_id` | 无；聊天格式化由 `common/chat.h` 的 Jinja 模板完成，其他几项则没有用途。 |

CMake 会将已移除的 `-D` 变量报告为未使用，而不会应用它们，因此请检查手动构建的配置输出。

Web 界面将其设置保存在服务器上（`~/.mgs/config/ui-settings.json`），在浏览器中只保存其 API 密钥（`localStorage` 中的 `MgsUi.apiKey`）和对话（`MgsUi` IndexedDB 数据库）。它不会读取或迁移旧版本保存在 `localStorage` 中的设置（`MgsUi.config`、`MgsUi.userOverrides`、收藏的模型、工具列表、推理强度、已关闭的 MCP 推荐以及语言），因此设置会从默认值开始，API 密钥也需要重新输入一次。它也不会读取或迁移旧版本以其他名称保存的数据、独立的 `theme` 键、旧消息中的内联推理和工具调用标记、旧 `context` 类型的粘贴文本附件、已关闭 MCP 推荐的旧列表格式，或服务器以前的 `webui_settings` 字段。若要保留旧版本的对话，请在升级之前从旧版本中导出，并在升级之后导入该导出文件。对话导入只接受 JSONL 和 ZIP 导出文件，且须在[安全限制](#security-limits)中列出的限制范围之内；旧的 JSON 导出格式会被拒绝。旧设置会一直保留在浏览器中，直到被删除为止：若要删除它们，请在界面页面上打开浏览器控制台（F12），并针对打开过该界面的每个地址和端口各运行一次 `for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`。

使用 `-mu`/`--model-url` 下载的文件缓存在 `~/.cache/mgs/` 中，因此旧版本保存在其以前缓存文件夹中的文件会被重新下载。Hugging Face 下载继续使用 Hugging Face 缓存（`~/.cache/huggingface/hub`，或 `MGS_CACHE`、`HF_HUB_CACHE` 或 `HF_HOME` 中指定的文件夹）。

### 旧的模型文件

针对以下旧转换的兼容代码也已被移除。若要使用这些模型，请重新转换原始模型，或下载当前的 GGUF。

| 旧文件 | 现在的行为 |
| --- | --- |
| 2025 年 8 月之前转换的 Grok-1（不含 `grok.logit_scale`、`grok.embedding_scale`、`grok.attention.output_scale` 或 `grok.attn_logit_softcapping`） | 无法加载：缺少键。 |
| 带有未拆分的 `blk.N.attn_kv_b` 张量的 Kimi-Linear | 无法加载：缺少 `attn_k_b`/`attn_v_b`。该路径此前在构建计算图时就已经失败。 |
| 只有合并后的 `ssm_in` 张量的 Qwen3-Next 文件 | 无法加载：缺少张量 `blk.0.attn_qkv.weight`。 |
| 不含 `expert_gating_func`（`glm-dsa`、`glm4-moe`、`hy-v3`、`laguna`）、`rope.scaling.finetuned`（Granite 4.0-H）或 `full_attention_interval`（Qwen3-Next）的文件，以及使用单个 SWA 周期而非逐层数组的 Cohere2-MoE 文件 | 无法加载：缺少键。 |
| 不含 `tokenizer.ggml.pre`，或使用预分词器名称 `llama3`、`llama-v3`、`jina-es`、`jina-de` 或 `llada-moe` 的 BPE 词表，以及不含合并规则的 Kimi-K2 分词器 | 无法加载：缺少键、预分词器类型未知或缺少合并规则。 |
| GGUF 版本 1 或 2 的文件 | 无法加载：只接受 GGUF 版本 3。 |
| 旧的一次性脚本生成的 mmproj 文件：投影器类型 `mlp`、`mlp_norm`、`ldp`、`ldpv2`、`resampler`（MiniCPM-V 2.x 至 4.5、MiniCPM-o 2.6）和 `adapter`（GLM-Edge-V），以及没有投影器类型的文件 | 投影器无法加载：投影器类型未知或缺失。 |
| `ffn_up` 和 `ffn_down` 被互换的旧转换 mmproj 文件 | 不再进行纠正；投影器会失败。 |
| 不含 `image_min_pixels`/`image_max_pixels` 的 Kimi-K2.5 mmproj，以及不含 `preproc_image_size` 的 Step3-VL | 投影器无法加载：缺少键或张量。 |
| 只携带旧的 `tokenizer.ggml.prefix_token_id`、`tokenizer.ggml.suffix_token_id` 和 `tokenizer.ggml.middle_token_id` 键，而不携带 `tokenizer.ggml.fim_*_token_id` 键的文件 | 可以加载；填充（infill）token 会通过其文本被识别（例如 `<PRE>` 或 `<\|fim_prefix\|>`），如果找不到这些 token，填充功能将无法工作。 |

当前的转换或受支持系列中下载量最高的已发布 GGUF 仍然依赖的后备机制均已保留，并已对照它们在 Hugging Face 上的文件头进行了核实：`llama-bpe`、Tekken 及类似预分词器的默认 `add_bos`；根据 `full_attention_interval` 推导出的 Qwen3.6 循环层布局（下载量最高的 Qwen3.6 GGUF 不含 `recurrent_layers`）；以及针对不含 `attention.indexer.types` 的 GLM-5.2/5.3 文件的内置 GLM-5.2 索引器布局。

<a id="maintenance"></a>
## 维护与故障排除

- 使用 `mgs --version` 查看已安装的版本（例如，它会打印 `version: 0.1.0`）；本地构建会在结束时运行该命令，CMake 也会在配置时打印 `mgs version: 0.1.0`。界面会通过“设置 → 显示”中的**显示构建版本信息**（显示在右下角）以及模型信息对话框中的**构建信息**显示相同的版本。项目的每次更改都会使最后一个数字加一：0.1.0、0.1.1、0.1.2 等。
- 重复执行 `curl | bash` 即可从 `main` 更新，或者使用相应的本地安装命令重新构建并替换二进制文件。之后请重启服务：在 Debian 上使用 `systemctl restart moe-gguf-server`，在 Alpine 上使用 `rc-service moe-gguf-server restart`。
- 请查看 `/root/webapp-install.log`，在 Debian 上查看 `journalctl -u moe-gguf-server`，在 Alpine 上查看 `/var/log/moe-gguf-server.log`。Apache 日志位于 `/var/www/DOMAIN-logs/` 下。
- 如果无法进行 CUDA 编译，请在外部安装 Toolkit，或使用 `--cpu`。如果找不到 NCCL，请检查软件源是否可用以及 CUDA 版本，或者显式选择 `--no-nccl`。
- 如果 UI 编译失败，请检查 Node.js 和 npm 的版本、依赖项的下载情况以及日志。不要用上游预构建的界面来替代。
- 对于部署构建，在源码布局发生变化后，请在 `_/temp/` 中使用一个全新的目录。位于 SSHFS 上的项目副本可能禁止运行可执行文件；在生产环境中请使用本地磁盘。
- 对于本地构建，请重新运行上面的单行命令以下载当前的 `main`，或运行 `bash build/build-local.sh` 以使用您的本地源码，然后重启您手动启动的副本。如果构建失败或报告了 `lib/` 之外的库，请阅读终端输出。临时的 `build.log` 会随构建目录一起被删除。
- 旧客户端中出现的 API 404 错误通常需要将基础 URL 改为 `/api/v1`。已安装的 API 密钥应放在 Authorization 标头中。

根据本项目的规则，编码助手不会执行任何测试、部署运行、GPU 基准测试或自动化的移动浏览器检查。静态语法检查并不能保证运行时的行为。有关实现地图和扩展点，请参见 [CODE.zh-CN.md](CODE.zh-CN.md)。

<a id="license"></a>
## 许可证

`mgs` 依据 [PolyForm NonCommercial License 1.0.0](../LICENSE) 分发。您可以出于任何非商业目的使用、修改和分享它：个人使用、学习、研究、业余项目，以及由慈善组织、教育机构、公共研究机构和政府机构使用。商业用途（例如公司将其用于自身业务）需要另行获得 nipegun 的许可。

该许可证仅涵盖 nipegun 的贡献。来自 llama.cpp（MIT，The ggml authors）、FreeToken（Apache 2.0）以及捆绑库的部分保留其各自的许可证；它们的声明位于 `build/licenses/` 中，本地构建会将其安装到 `~/.local/share/doc/mgs/licenses/`。
