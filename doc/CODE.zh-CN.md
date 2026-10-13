# 技术指南

[Deutsch (Deutschland)](CODE.de-DE.md) · [English (United Kingdom)](CODE.en-GB.md) · [English (United States)](CODE.md) · [Español (Argentina)](CODE.es-AR.md) · [Español (España)](CODE.es-ES.md) · [Français (France)](CODE.fr-FR.md) · [עברית (ישראל)](CODE.he-IL.md) · [हिन्दी (भारत)](CODE.hi-IN.md) · [Italiano (Italia)](CODE.it-IT.md) · [日本語 (日本)](CODE.ja-JP.md) · [한국어(대한민국)](CODE.ko-KR.md) · [Português (Brasil)](CODE.pt-BR.md) · [Português (Portugal)](CODE.pt-PT.md) · [Русский (Россия)](CODE.ru-RU.md) · [中文（中国）](CODE.zh-CN.md)

[架构与设计决策](#architecture) · [模块图](#modules) · [关键符号索引](#symbols) · [主要调用路径](#flows) · [入口点与路由](#routes) · [影响分析](#impact) · [扩展点](#extensions) · [上游审查——2026-10-02](#upstream-review)

<a id="architecture"></a>
## 架构与设计决策

**MoE 模型加载与 MTP 优先级。** 引擎在 `backend/src/CMakeLists.txt` 中编译一份显式列出的源文件清单。仅适用于稠密模型的解码器实现已被移除。共享的计算图归属于使用它们的 MoE 实现（包括 Phi-MoE、Granite、Nemotron、Ernie 和 Nomic）；MoE 模型内部的稠密块仍然是必需的。`mgs_model_create` 会在加载权重之前检查专家数量，并拒绝稠密的主模型。显式作为 MTP 加载、仅包含预测头的 GGUF 是例外：稠密的预测头可以为 MoE 目标模型服务。Qwen35 只保留其 MTP 计算图。

MTP 的放置会考虑每一块选中的 GPU，计入后端的分配大小和对齐要求，并包含输出头（或与之绑定的 token 嵌入）。它首先寻找一块能够容纳所有预测头的 GPU；如果找不到，就把最大的几组分布到多块 GPU 上。只有当没有任何选中的 GPU 能容纳某一组时，该组才会退回到 RAM。独立的 `dMtpContexts` 会先于主干上下文分配并上传；张量覆盖无法替换这一放置。外部 MTP 权重会在目标模型之前预加载（使用 `--fit off` 时也是如此），并移交给推测解码的初始化过程，而不会加载第二份副本。fit 估算会为它们的上下文/计算内存预留空间，但不会把它们已经分配的权重重复计算。

**源代码约定与 CLI 校验。** 项目自有的标识符使用 CamelCase，并带有项目规则规定的函数、方法、参数和值前缀。公共 C API、GGUF 名称、协议字段、第三方名称以及框架要求的回调/组件名称保持其既定约定。C++ 格式化使用两个空格缩进。命名规范同样适用于任何 Linux 构建都不会编译的预处理器分支（Windows、macOS、调试宏、`MGS_LLGUIDANCE`、`#if 0`）：编译器无法在这些分支中发现旧名称，因此任何重命名都必须在这些分支中手动检查。`--chat-template-kwargs` 会拒绝 `vEnableThinking`；请改用 `--reasoning on` 或 `--reasoning off`。带下划线的选项拼写不会被改写，重复的选项会被拒绝，但明确允许重复的 `--spec-type` 除外。`Github-Repo-Push.sh` 使用 `set -euo pipefail`、`fCleanup`、其 EXIT trap 以及经过显式检查的 `fMain`；任何一个发布步骤失败都会终止该脚本。

`frontend/` 包含 UI；`backend/` 包含引擎、服务器和依赖项；`deploy/` 包含安装程序、服务和资源脚本；`build/` 包含本地自包含构建脚本及其可重定位的启动器；`doc/` 包含文档；`_/` 包含本地工作材料。导入的依赖项保持原样。

Svelte 界面位于 `frontend/`，C++ 引擎/服务器位于 `backend/`，安装相关内容位于 `deploy/`。引擎保留了 `src/`、`common/`、`include/`、`ggml/`、`vendor/` 和 `tools/` 之间的相对关系。`build/CMakeLists.txt` 中的 CMake 入口点委托给后端；UI 库从 `frontend/` 构建并嵌入可执行文件。CMake 目标包括 `mgs-engine`（引擎库，`backend/src/`）、`mgs-common-base` 和 `mgs-common`（`backend/common/`）、`mtmd`、`server-context` 和 `mgs-server-impl`（`backend/tools/`）、`mgs-ui`（嵌入式界面）以及可执行文件目标 `mgs`；构建输出和安装都使用 `mgs` 这一名称。

**命名。** 从 llama.cpp 继承的代码使用本项目自己的名称：`backend/include/mgs.h` 和 `mgs-cpp.h` 的公共 C API 使用 `mgs_` 和 `MGS_` 前缀，引擎源文件为 `backend/src/mgs*.{h,cpp}`，服务器入口函数为 `mgs_server`，环境变量为 `MGS_*`（`MGS_ARG_*`、`MGS_API_KEY`、`MGS_CACHE`、`MGS_MOE_CACHE*`、`MGS_SERVER_*`），CMake 选项为 `MGS_*`，Prometheus 指标为 `mgs:*`，CORS 代理请求头前缀为 `x-mgs-proxy-header-`，浏览器存储使用 `MgsUi`，下载缓存为 `~/.cache/mgs/`，tinyBLAS CPU 内核为 `ggml-cpu/tinyblas/` 中的 `GGML_TINYBLAS`。其余的 `llama` 拼写都是有意保留的：llama.cpp 的许可证声明（`LICENSE`、`build/licenses/LICENSE-llama.cpp`）和 README 的标语；对上游项目本身的提及（来源、比较），且不链接到其代码仓库；GGUF 文件内部存储的值（SentencePiece 分词器模型 `llama` 以及 `llama-bpe` 和 `llama4` 预分词器，连同其 `cMgsVocabPreTypeLlama3`/`LLAMA4` 标识符和 `fUnicodeRegexSplitCustomLlama3`）；注释中第三方模型和代码的名称（CodeLlama、Llama-3_1-Nemotron、LLaDA、Ultravox 模型以及 YaRN 参考实现 `LlamaYaRNScaledRotaryEmbedding.py`）；tinyBLAS 源代码中的一条参考文献；virtgpu 后端所用的 virglrenderer API remoting 的 `APIR_LLAMA_CPP_*` 变量；以及“llamada”之类的西班牙语单词。Meta 的 Llama 架构（`llama`、`llama4`、`llama-embed`）、其内置聊天模板以及 Llama 4 视觉投影器均已移除。

HTTP 注册会将 API 前缀应用于每一条已发布的路由。处理函数接收的是不带前缀的规范路径；路由器在转发给模型子进程时会加上 `/api`。公共身份验证白名单使用完整的外部路径。UI 保持在 `/`；`/api/config.js` 会在应用模块加载之前发布该前缀。Swagger 和 OpenAPI 位于 `/api/doc/` 下，模型加载期间也可访问。路由/方法注册表提供实时的路由清单；补全参数来自 `server-schema`。

JSON 语言目录保存了 15 个语言区域的界面翻译：`de-DE`、`en-GB`、`en-US`、`es-AR`、`es-ES`、`fr-FR`、`he-IL`、`hi-IN`、`it-IT`、`ja-JP`、`ko-KR`、`pt-BR`、`pt-PT`、`ru-RU` 和 `zh-CN`。`aLanguages` 和 `dCatalogs` 按语言区域代码的字母顺序注册它们；`en-US` 仍是默认语言和消息的回退语言。每个目录都包含全部 15 种语言的本地化名称。`fTranslate` 会插入具名值，而不会修改对话数据。新的翻译在删除/导入消息中使用计数标签，而不是插入英语的复数后缀。切换语言时，会将其保存到服务器的设置文档中并重新加载界面，使在模块导入时计算的标签也随之改变；`/api/config.js` 会在模块加载之前发布已保存的语言，URL 中的 `?lang=` 则会覆盖它。根布局设置 HTML 的语言和方向（仅希伯来语为 `rtl`）；导航使用逻辑定位，聊天输入框和用户的纯文本使用 `dir="auto"`，代码/终端文本保持从左到右。框架契约、协议字段和导入的符号保留其原始名称；项目自有的函数遵循规定的命名前缀。

**文档语言。** README、MANUAL 和 CODE 文档提供与界面相同的 15 个语言区域版本：`README.md`（en-US）位于根目录，其翻译为 `doc/README.<locale>.md`；手册和本指南为 `doc/MANUAL.md` 和 `doc/CODE.md`（en-US），以及 `doc/MANUAL.<locale>.md` 和 `doc/CODE.<locale>.md`。每个版本都以一个语言栏开头，按语言区域代码的字母顺序链接所有其他版本。各语言的标题都带有相同的 `<a id>` 锚点，因此文档之间的链接（例如 `MANUAL.<locale>.md#local-build`）在所有语言中都有效。代码块、命令、路径、选项和标识符在每种翻译中都完全相同；只有正文会改变。希伯来语版本是纯 Markdown：GitHub 和 Gitea 会检测每个段落的方向，因此正文从右到左阅读，而代码块保持从左到右。

安装需要 root 权限，会显式处理命令失败，并且只在 `_/temp/` 下暂存文件。UI 在本地构建，而不是下载缺少本项目修改的上游界面。部署会创建无特权的推理服务和 Web 服务。其专用的 Apache 实例在每次启动时从 11080 开始搜索 HTTP 端口、从 11443 开始搜索 HTTPS 端口；推理服务始终位于 loopback:8080。不配置 HAProxy。程序的配置位于 `/opt/moe-gguf-server/.mgs/config/`，即服务账户的 `~/.mgs/config/`，归该账户所有，模式为 0600：`api.key`、`web-domain` 和 `apache2.conf.in`（Apache 的 `ServerRoot`）由 `configure-web.sh` 写入，`ui-settings.json` 由 `mgs` 写入；只有系统集成部分（systemd 或 OpenRC 单元）保留在 `/etc` 下。对话保存在浏览器中；后端没有用户数据库。

服务器本身默认启用 HTTPS：`common_params` 从 11443 开始，`server_http_context` 绑定第一个空闲端口，随后从 11080 开始绑定 HTTP。第二个监听器返回 308，重定向到所选的 HTTPS 端口，保留主机、路径和查询字符串，并带有 `Cache-Control: no-store`。`--port` 固定主端口（0 表示交由操作系统分配）；`--http-port` 更改 HTTP 搜索的起始端口。只有“地址已被占用”的冲突才会让搜索继续往后；其他错误会终止启动。`server-tls.cpp` 加载用户提供的 PEM 密钥对；如果没有提供，则加载 `~/.mgs/certificates/` 中的自签名身份（`fPersistentServerIdentity`）：只要 `fullchain.pem` 和 `privkey.pem` 相互匹配且证书未过期，就重复使用它们，否则生成一张 ECDSA P-256 证书（有效期 365 天，SAN 包含 localhost、回环地址、配置的主机、机器名及其各网络接口的地址），并通过临时文件和重命名保存（文件夹 0700，密钥 0600），因此每次运行和每次构建都提供同一张证书，浏览器也会保留其信任例外。不会调用任何外部命令；如果该文件夹无法写入，本次运行就使用内存中的身份。没有 OpenSSL 支持时，HTTPS 会明确报错失败。`--http` 用于有意选择纯 HTTP，供路由器子进程和 8080 上的 Apache 后端使用；Unix 套接字和 Vertex AI 仍使用 HTTP。析构函数会停止两个监听器并等待（join）它们结束。这些决策都在 C++ 中实现，独立于安装程序和启动器。

`build/build-local.sh` 面向在工作站上的手动使用：它不需要 root，不安装任何软件包或服务，也不会留下任何正在运行的进程。它会复制完整的本地源代码树；在通过 `curl | bash` 调用或作为独立脚本运行时，则不借助 Git 下载 GitHub 的 `main` 归档。随后它在 `/tmp/moe-gguf-server-build.XXXXXXXX/` 中构建 `mgs` 目标，并在某个前缀下安装一份可重定位、自包含的副本（默认为 `~/.local`，仅供当前用户使用；`--prefix` 可以更改它）：`bin/mgs`、`lib/mgs/` 和 `share/doc/mgs/licenses/`。`lib/mgs/` 包含服务器二进制文件、其传递性的 `ldd` 依赖闭包、glibc 动态加载器以及 glibc 通过 `dlopen` 打开的 NSS 模块。内核只认可绝对路径的 `PT_INTERP`，因此 `bin/mgs` 是一个静态启动器（`build/mgs-launcher.c`）：它读取 `/proc/self/exe`，从 `bin/` 向上找到前缀，然后带上其余参数执行 `lib/mgs/<loader> lib/mgs/mgs`；前缀中的任何内容都不依赖其绝对位置。服务器二进制文件和捆绑的库都带有 `DT_RPATH` `$ORIGIN`，之所以有意选择它，是因为它的优先级高于 `LD_LIBRARY_PATH`。由于此时服务器的进程映像是加载器，启动器会导出 `MGS_EXECUTABLE`，并由 `fGetServerExecPath` 返回它，从而让路由器子进程通过启动器重新进入。NVIDIA 驱动库（`libcuda`、`libnvidia-*`）从不捆绑，因为它们必须与内核模块匹配。`lib/mgs/` 内的一个标记文件可防止替换并非由该脚本创建的安装。

每次本地运行都会使用一个由 `mktemp -d /tmp/moe-gguf-server-build.XXXXXXXX` 新建的私有目录。下载的归档、其成员列表、复制或解压出的源代码、CMake 输出、npm/Node/CUDA 缓存、临时文件、日志以及组装好的 `bundle/` 都留在那里。自动编译器缓存已禁用（`GGML_CCACHE=OFF`、`CCACHE_DISABLE=1`），因此构建结束后不会残留缓存服务器。产物准备就绪后，脚本使用 `flock` 锁定前缀目录本身，将完成的文件复制到其最终位置旁边的临时条目中，然后在同一文件系统上将它们重命名到位。`fCleanup` 会等待日志写入进程结束，并在成功、失败以及已处理的 INT/TERM/HUP 信号时删除私有工作区。不会向源代码树或 `/opt/` 写入任何构建文件；本地构建器没有 `--build-dir` 选项。

入口脚本的路径在调用 `fMain` 之前通过 `${BASH_SOURCE[0]:-}` 获取，因此即使启用了 `set -u`，从 stdin 执行也能正常工作。选项在查找源代码之前解析；`--help` 既不需要源代码树，也不需要下载源代码。`fHasProjectSources` 会检查构建入口点、启动器、前端清单文件和许可证目录。完整的本地副本会按照现有的排除规则进行暂存。否则，`fDownloadSources` 要求系统中已有 curl/gzip，通过 HTTPS 下载并在失败时重试，校验归档中的路径，并解压到私有的 `source/` 目录，然后再检查所需的文件。`fMain` 运行时其 stdin 重定向自 `/dev/null`，因此构建工具无法读走通过管道传入的脚本。远程选项使用 `bash -s --`；从不安装软件包。


Debian 入口点是独立的：它会检测完整的本地检出副本，否则从 GitHub 下载 `main`。远程模式使用 `/opt/moe-gguf-server-source`（可由 `MGS_INSTALL_SOURCE_DIR` 覆盖），检查源代码来源标记，并在同步和编译的整个过程中持有锁。`rsync --checksum --no-times --delete` 会删除过时的源文件，保留 `_/`，并为已更改的文件赋予新的时间戳。辅助脚本在一个单独的 Bash 中运行，其 stdin 已关闭，并且不会复制其父进程的日志流。

不保留任何向后兼容层。以下内容均已移除：已弃用的 CLI 别名和已删除选项的占位提示、重复的长选项名称（每个选项只保留一个短名称和一个长名称）、`MGS_ARG_NO_*` 环境变量回退以及以旧选项命名的环境变量、旧版 HTTP 路由别名（`/completion`、`/embedding`、`/reranking`）和重复的请求字段（`thinking_budget_tokens`、嵌入路由中的 `content`）、`deepseek-legacy` 推理格式、非 Jinja 模板引擎（`--jinja`/`--no-jinja`、`mgs-chat.cpp` 的内置 C++ 模板以及 `mgs_chat_apply_template`：每个聊天模板都经由 Jinja 处理，`chatml` 指代内置的 ChatML 源）、旧的 CMake 变量名、浏览器数据迁移以及安装程序中的兼容垫片，因此每个选项、路由、变量和存储值都只有一个当前名称。由于本项目从未部署过，公共的 mgs/GGML/mtmd C API 也不保留任何已弃用的符号：只保留其当前函数（已移除的函数列在 [MANUAL.zh-CN.md](MANUAL.zh-CN.md#removed-compatibility) 中）。例外都是有意为之的：通过回退路径加载 GGUF 文件——受支持家族的当前转换版本或下载量最高的已发布 GGUF 仍依赖这些回退路径，这一点已对照它们在 Hugging Face 上的文件头核实（由 `full_attention_interval` 推导出的 Qwen3.6 循环层布局，以及为不带 `attention.indexer.types` 的 GLM-5.2/5.3 文件内置的 GLM-5.2 索引器布局）；工具链和操作系统的版本保护；以及与外部协议（OpenAI、Anthropic、TEI/Jina、GCP Vertex、MCP）和外部对等方（例如 ggml-virtgpu 的 virglrenderer 客户机）的互操作性。针对旧文件的兼容垫片已被移除：Grok-1、MiniCPM、Kimi-Linear 未拆分的 `attn_kv_b`（非 MLA）路径、Qwen3-Next 的 `ssm_in` 路径、缺失的 `expert_gating_func`、`rope.scaling.finetuned` 和 `full_attention_interval` 键、Cohere2-MoE 的标量 SWA 周期、缺失的预分词器和 Kimi-K2 合并规则、`llama3`、`llama-v3`、`jina-es`、`jina-de` 和 `llada-moe` 预分词器名称、`rope.scale_linear`、早期的 Gemma 2、GGUF 版本 1 和 2、旧的 `tokenizer.ggml.prefix/suffix/middle_token_id` 键，以及 `mtmd` 中由旧的一次性脚本生成的投影器（`mlp`、`mlp_norm`、`ldp`、`ldpv2`、`resampler`、`adapter`、没有投影器类型的文件）、旧版 surgery 转换中交换过的 FFN，以及缺失的 Kimi-K2.5 和 Step3-VL 键的默认值。此外，根据用户的决定，若干完整的架构也被移除：Gemma 4 以及所有早于 V4.1 的 DeepSeek 架构（参见下文的受支持家族）。

从项目根目录使用 `cmake -S build -B _/temp/cmake` 手动配置。构建脚本会将 `build/` 与源代码一起暂存，并选择 `source/build` 作为 CMake 源目录。部署脚本将生成的文件保存在 `_/temp/` 下，如果缓存的配置指向另一个源目录，就重置 `CMakeCache.txt` 和 `CMakeFiles/`。本地构建始终在 `/tmp/` 下使用全新的一次性工作区。cpp-httplib 的 TLS 库在 `backend/vendor/cpp-httplib/CMakeLists.txt` 中根据 `backend/CMakeLists.txt` 声明的选项选定：`MGS_BUILD_BORINGSSL` 和 `MGS_BUILD_LIBRESSL`（默认均关闭）会下载并构建相应的库，并按此顺序优先于 `MGS_OPENSSL`（默认开启，使用系统的 OpenSSL）；三者都未启用时，服务器不支持 HTTPS。

**版本。** 项目版本只在一处定义，即 `backend/cmake/build-info.cmake` 中的 `MGS_VERSION`（从 `0.1.0` 开始），对项目的每一次更改都会将其补丁号加一（0.1.1、0.1.2……）。本地构建和 Debian 安装程序配置的是同一棵 CMake 树，因此两者生成相同的版本：CMake 会打印 `mgs version: …`，`build-info.cpp` 通过 `mgs_version()` 返回它，它还用于设置共享库的文件版本（soname 保持为 `.so.0`）、`mgs.pc`、`mgs-config.cmake` 以及界面的 `build.json`。本项目不使用 Git，因此没有构建号或提交：`--version`、`/api/props` 的 `build_info` 属性、OpenAI 的 `system_fingerprint`、shell MCP 的 `serverInfo.version` 以及下载时的 `User-Agent`（`mgs/0.1.0`）报告的都是这一版本。当 `build.json` 中记录的是另一个版本时，`deploy/scripts/ui-assets.cmake` 会重新构建界面，因此版本变更无需改动界面源代码，也能体现在复用的 `_/temp/` 构建树中。

格式化配置按文件夹限定作用范围：`backend/`、`frontend/` 和 `build/` 中有 `.clang-format`；这些文件夹以及 `deploy/` 和 `tests/` 中有 `.editorconfig`。`build/licenses/` 中的许可证声明继承 `build/.editorconfig`。请保持等效的设置同步。项目根目录和 `doc/` 中的文件需要在编辑器中设置 UTF-8、LF 和两个空格缩进。

`doc/` 文件夹包含供阅读的文档。Swagger 页面位于 `frontend/api.html`，由 CMake 嵌入服务器中。后端在每次请求时根据已注册的路由生成 `/api/doc/openapi.json`；不存在静态的 `openapi.json` 源文件。

本分叉保留了 llama.cpp 的服务器、多模态支持和 GGML 后端，并针对拆分在 GPU 内存与系统 RAM 之间的权重进行了 MoE 优化。主机内存注册和专家预取源自 thecodacus 的分叉中的 `fable5/host-register` 和 `fable5/prefetch-experts` 分支。预取槽位保存 gate/up/down 专家张量，并让其传输与计算重叠进行。自动 MTP 在选择 `draft-mtp` 之前会检查 `nextn_predict_layers`、模型主干以及所有集成的预测头；这些检查涵盖每一个 GGUF 分片。调优方法和限制请参阅[手册](MANUAL.zh-CN.md#performance)。

**MoE 专家缓存。** 对于专家保留在系统内存中的每一层，`backend/src/mgs-expert-cache.{h,cpp}` 会把其最近使用的专家保存在设备池中：每块 GPU 一个池，由分配给该 GPU 的各层共享（同一 GPU 上逐专家张量类型和形状都相同的每组层各有一个 LRU 列表）。一个槽位保存一个专家的全部逐专家张量：权重、偏置和逐专家缩放因子，gate/up 投影既可以是分开的，也可以是合并的。对于低于算子卸载阈值（32 个 token，`GGML_OP_OFFLOAD_MIN_BATCH`）的微批次，`mBuildMoeFfn` 会构建两次专家链：一个通过槽位标识符读取池的设备分支，以及一个读取主机权重的 CPU 分支。首先运行一个 CPU 分区操作（`mPartition`）：它统计命中数，按带宽均衡上传一部分未命中的专家（约 m·B_upload/(B_upload+B_cpu)，至少一个），并在设备分支之前将槽位标识符和上传操作排入设备流；然后 CPU 分支同时计算其余未命中的专家。由另一个分支处理的专家在 CPU 上获得标识符 -1（CPU 的 `mul_mat_id`/`add_id` 内核会跳过它们并写入零），在设备上获得一个空槽位（位置 k 使用空槽位 k，因此同一 token 内槽位不会重复，这是 CUDA MMQ 辅助函数的要求），因此两个分支之和在舍入误差范围内等于常规执行的结果。只有当前一个拆分复制过输入、或其后端能够访问当前缓冲区类型时，调度器才会将无输入的拆分与前一个拆分同步；这既保留了上游 #26040 的修复，又让两个分支能够重叠执行。上传带宽在创建时用真实的专家测量，CPU 带宽则根据 CPU 分支的耗时在线估算。较大的微批次仍使用专家预取：`ggml_backend_sched_set_weight_upload` 让缓存以设备到设备的方式将其中的专家复制到预取槽位，只有其余专家从主机内存上传。缓存使用的是分配上下文之后剩余的设备内存，再扣除 `--moe-cache-reserve` 和预取槽位。它在每个上下文和投影器都加载完成后由第一个请求创建（通用的预热流程会推迟其创建），可以在运行时调整大小（`mgs_set_expert_cache_size`、`POST /api/expert-cache`），并通过 `mgs_get_expert_cache_info`、`/api/expert-cache` 和 `/api/metrics` 报告其计数器。设置通过 `MGS_MOE_CACHE*` 环境变量传递（由 `--moe-cache*` 选项设置），从而保持公共 API 不变。在启用缓存且 RAM 充足时，`--fit` 会将所有专家保留在系统内存中，把空闲的设备内存留给缓存，因为对解码而言，LRU 缓存比整层的静态专家更有效；在有多块 GPU 时，它会按内存目标将仅含稠密权重的层分布到各 GPU 上，使每块 GPU 缓存其自身各层的专家。上传带宽按 GPU 分别测量，请求的大小会按各 GPU 所缓存的主机专家在 GPU 之间分配；在流水线并行下缓存保持关闭，因为流水线并行会同时有多个微批次在途。`--fit` 还会计入 nextn（MTP）层，加载器会把这些层算作其卸载的层；否则在 MTP 关闭时，第一层会留在 CPU 上。

**使用 `--fit` 时的上下文大小。** `fCommonParamsFitImpl` 的第 2 步在缩减用户未设置的上下文时，就好像所有权重都必须放进设备内存一样，因此比设备内存更大的模型以前总是会降到 `--fit-ctx`（4096 个 token），尽管它的专家随后仍留在系统内存中，导致推理回答被截断。一旦稠密权重能够放下，`fRaiseContext` 就会分别在缩减后的上下文和训练上下文下测量仅放置稠密权重的布局，进行线性插值并提高上下文（向下取整到 256 的倍数，并再测量一次加以校验），直到 KV 缓存以及随其增长的缓冲区最多占用稠密权重之后剩余设备内存的四分之一（`cContextShare`）。它在选定所有专家都位于 RAM 中（专家缓存）的放置方案之后、整层专家填满设备之前运行，因此专家始终至少保有这部分内存的四分之三。显式指定的 `--ctx-size` 永远不会被更改，`--fit-ctx` 仍是最小值，并且当 KV 缓存位于系统内存中时不做任何更改。草稿上下文采用目标模型的上下文大小（`mLoadModel` 中的 `vParamsDft.n_ctx = n_ctx`）：以前在上下文大小为 0 时，它会按模型的训练上下文分配自己的 KV 缓存，从而挤占专家缓存的设备内存。

**自动配置。** `mgs --model FILE` 必须自行为本机选出最快的设置；显式指定的选项始终优先。当专家位于系统内存中时，提示词的每个微批次都要通过 PCIe 流式传输不在设备上的专家（在 RTX 4060 Ti 上运行 Qwen3.6-35B-A3B Q8_0 时，每个微批次约 2 秒），因此提示词处理速度几乎随微批次大小线性增长，而生成速度与之无关：一个 2514 token 的提示词，微批次为 512 时为 215 token/s，2048 时为 408，4096 时为 737–755，生成速度不变。在提高上下文之前，`fRaiseBatch` 会在选定的放置方案下依次尝试 4096、2048 和 1024 token 的微批次，并保留额外内存不超过稠密权重之后剩余设备内存八分之一（`cBatchShare`）的最大者；逻辑批次也随之增大。它只在未给出 `-b`/`-ub`（或其对应的环境变量）时运行（`common_params::vBatchExplicit`，以 `pAutoBatch` 传入）。`common_init_result` 会将得到的 `n_batch` 写回参数中，因为草稿上下文和 MTP 上下文会复制这些参数，并把目标模型的每个批次接收到一个按其自身 `n_batch` 分配的缓冲区中；它们的微批次保持默认值。默认的 CPU 线程数（`fCommonDefaultThreads`，在未指定 `-t` 时由 `fPostprocessCpuParams` 应用）是可用于数学计算的核心数，上限为进程亲和性掩码中逻辑 CPU 数量的一半：MoE 推理中由 CPU 承担的部分受内存带宽限制，而在看不到 SMT 的虚拟机上使用全部逻辑 CPU，会让工作线程与驱动 GPU 的线程相互争抢（16 个 vCPU：4 到 12 个线程之间的差异在噪声范围内，16 个线程则慢 13%）。在 SMT 可见时，该上限等于物理核心数，因此物理机保持以前的默认值。`common_init_result` 会在日志中记录线程数；`--fit` 会记录其选定的微批次和上下文。

**MTP 放置。** MTP 权重驻留在设备内存中，优先于其他所有放置规则：包括嫁接到模型上的 nextn 层，以及通过 `--spec-draft-model` 传入的纯 MTP GGUF 的每一层。即使 `n_gpu_layers` 会把这些层和输出层留在 CPU 上，`mgs_model_base::mLoadTensors` 也会将它们分配给某块 GPU，并且 `mgs_model_loader::mCreateTensor` 会忽略那些把它们放到主机内存的覆盖设置（`--fit` 的部分层、`--cpu-moe`、`--n-cpu-moe`、`--override-tensor` 及其对应的 `--spec-draft-*` 选项）。token 嵌入与主干的嵌入一样留在 CPU 上，因为按行查找放到 GPU 上毫无收益。当 MTP 层在保留 512 MiB 余量后无法放入设备的空闲内存时，就采用常规放置。`--fit` 通过加载模型但不分配内存来进行测量，因此它能看到位于设备上的 MTP 权重，并围绕它们安排其余部分；专家缓存则使用剩下的内存。使用单独的 MTP GGUF 时，`fCommonModelParamsToMgs` 会让嫁接在目标模型上的 MTP 层保持未加载状态，而不是保留一份用不到的副本。

**受支持的家族。** 必须正常工作的 MoE 家族有 Qwen3.6（`qwen35moe`）、Qwen3.8-Flash-Next（`qwen4exp`）、GLM-5.3（`glm-dsa`）、GLM-5.3-Flash（`glm5-next`）、Kimi-K3（`kimi-k3`）、DeepSeek-V4.1-Flash（`deepseek41`）、MiniMax-M3（`minimax-m3`）、Nemotron-3（`nemotron_h_moe`）和 MiMo-V2.6（`mimo2`）。根据用户的决定，Gemma 4（`gemma4`、`gemma4-assistant`，`gemma4v`、`gemma4uv`、`gemma4a` 和 `gemma4ua` 投影器以及 `peg-gemma4` 聊天格式）和所有早于 V4.1 的 DeepSeek 架构均已移除：`deepseek`、`deepseek2`（Kimi-K2.x、GLM-4.7-Flash、Moonlight、GigaChat3、Kanana-2、Mistral-Large-3 和 Youtu 也曾使用它）、带有 `deepseekocr` 投影器的 `deepseek2-ocr`、`deepseek32`、`mistral4` 和 `deepseek4`。它们的 GGUF 会被当作未知架构而拒绝。

**Kimi-K3（`models/kimi-k3.cpp`）。** KDA 线性注意力层（采用有界 sigmoid 衰减的满秩门控 `ssm_g`、共享的 `fBuildGdnL2Norm` 以及 `mBuildRecurrentAttn` 的逐通道门控）与 MLA 层交替排列，后者采用吸收后的 `k_b`/`v_b` 形式，并带有 sigmoid 输出门控。由 `ggml_dsv4_hc_pre` 构建的跨层残差混合会读取 `attn_res_score`、`ffn_res_score` 和 `output_res_score`。其 MoE 是潜空间式的：路由器以完整宽度工作，而 896 个专家以潜在宽度工作（`ffn_routed_down` → 专家 → `ffn_routed_norm` → `ffn_routed_up`），`situ` 激活函数（`cLlmFfnSitu`）在 `mBuildMoeFfn` 中处理；对该模型而言 `MGS_MAX_EXPERTS` 为 1024。循环状态支持回滚（`n_rs_seq`）。已发布的 GGUF 不带 MTP。聊天模板使用 `<|open|>`/`<|sep|>`/`<|close|>`，由 `fCommonChatParamsInitKimiK3` 解析；分词器沿用 `kimi-k2` 预分词器。

**GLM-5.3-Flash（`models/glm5-next.cpp`）和 `glm5v`。** KDA 层和仅含 nope 部分的 MLA 层，配合 k-pool DSA 索引器（`indexer_kpool_gate`、`indexer_kpool_ape`）、使用融合 hc 算子的超连接流，以及使用 `ggml_swiglu_clamp`（一个新算子，支持 CPU、CUDA 和 Vulkan，能在解码期间保留 CUDA 的矩阵-向量乘 + GLU 融合）的 DeepSeek 风格 MoE。NextN 块是一个 DSA 层：只需 `--model` 即可启用 MTP，其权重采用 VRAM 优先的放置方式。内存实现为 `mgs_memory_hybrid_idx`。`glm5v` 投影器复用 glm4v 的计算图，并带有可选的卷积后归一化、带截断的 SwiGLU、自己的预处理器以及 `<|begin_of_image|>`/`<|end_of_image|>` 标记。`glm4` 和 `glm5` 预分词器使用带 `ignore_merges` 的 ChatGLM4。

**Qwen3.8-Flash-Next（`models/qwen4exp.cpp`）。** 低秩超连接混合器与合并（`ggml_dsv4_hc_pre_gated`、采用恒等混合的 `ggml_dsv4_hc_post`）；n-gram 哈希的逐层嵌入，其表 `per_layer_token_embd`（约 29 GB）保持内存映射，按需读取并带有行预取；循环缓存中的膨胀卷积历史（`cache_ple_r_l`）；带回滚的门控 delta-net 层；QSA 块稀疏注意力层（k-pool 索引器缓存、`ggml_lightning_indexer` 评分、叠加在 KQ 掩码之上的 top-k 选择掩码，由于上游的稀疏内核尚未移植，因而使用稠密 flash attention 计算）；带门控共享专家的 MoE、IMROPE，以及一个拥有自己的超连接混合器和草稿内存的 MTP 块。`mgs_memory_hybrid_idx`（`mgs-memory-hybrid-idx.{h,cpp}`）为 qwen4exp 和 glm5-next 组合了 KV 缓存、循环缓存和 k-pool 索引器缓存；KV 单元会记录 token id（`vTok`）以供 n-gram 查找，`seq_pos` 是（位置，单元）的集合。

**DeepSeek-V4.1-Flash（`models/deepseek41.cpp`、`mgs-kv-cache-dsv41.{h,cpp}`）。** 有两种已发布的布局声明了此架构：antirez 的 DwarfStar 文件（键名取自 Hugging Face 配置，例如 `deepseek41.hidden_size`，一个 `deepseek41.config` JSON 字符串，以及一张由 E4M3 值和 E8M0 缩放组成的 I8 Engram 表，每行 264 字节），以及 llama.cpp 生态系统的文件（标准键，Engram 表采用 ggml 类型）。`fMgsDeepseek41AdaptMetadata` 会在创建模型之前将第一种布局转换为标准键（评分函数 → 门控枚举，标量 `swiglu_limit` → 逐层数组，40 或 43 个压缩比，Engram 偏移量为素数的累加和），加载器也接受 Engram 投影的两种名称。层的角色（KV 来源、索引键所有者、top-k 来源）由实际存在的张量决定。Engram 表（约 188 GiB）使用 `cTensorExternal` 创建：它会被计入，但从不分配，预取在它之前停止，每个 token 的行通过私有文件描述符上的 `pread` 读取（`mEngramReadRows`，解码 E4M3+E8M0，或通过 `to_float` 反量化）。缓存为每种压缩比保留一组（压缩比 2 使用门控池化，压缩比 1 从第 20 层起使用全长流），只在键所有者上保留索引键缓存，读取层则直接寻址其来源层的存储；Engram 从原始缓存单元中获取之前的 token。计算图使用滞后的超连接混合（one-hot 的首次复制、与最后一个 FFN 混合一起进行的最终折叠、没有 HC 头）、没有 q 头归一化、在 RoPE 之前从潜在向量得到的索引键、top-k 复用和两级候选块池、位于第 1 层和第 14 层之前的 Engram，以及路由专家和共享专家中的 `ggml_swiglu_clamp`。当 GGUF 不带聊天模板时，使用 `DEEPSEEK41_TEMPLATE_SRC`（一段 Jinja 源代码）；DSML 标签带有一个前导空格，服务器会将 `reasoning_effort`（1–100）传给模板。不支持 MTP/DSpark，也不支持视觉。`dsv4_*` 和 `lightning_indexer` 这两类 ggml 算子以及 `vDsv4HcMult`/`aDsv4CompressRatios` 超参数予以保留，因为 Kimi-K3、GLM-5.3-Flash 和 Qwen3.8 会用到它们；在 Vulkan 和 Metal 上，这些算子在 CPU 上运行。

**上下文检查点。** 内存无法回滚的模型（含循环层或 SWA 层）会从检查点恢复。检查点在以下时机创建：在开启一条用户、助手或工具消息的批次开头（即智能体编辑上下文的边界）、在提示词接近末尾处，以及在上一个检查点已落后 `--checkpoint-min-step` 个 token（默认 1024）之后的任意批次。消息边界只会切分最后一个微批次至少已满 3/4 的批次，因为每多一个微批次，都要再次流式传输被卸载的专家。达到 `--ctx-checkpoints` 时，会删除与相邻检查点距离最近的那个检查点，使剩余的检查点在上下文中保持分散。

**监控与状态栏。** `GET /api/monitor` 报告 CPU 使用率（全局、按物理处理器以及按逻辑 CPU，基于 `/proc/stat` 的差值；每个逻辑 CPU 按编号匹配，并按 sysfs 的 `physical_package_id` 分组，每个处理器的型号取自它在 `/proc/cpuinfo` 中的信息块）、RAM（`/proc/meminfo`）以及每块 GPU（先通过 `dlopen("libnvidia-ml.so.1")` 使用 NVML，然后是 NVML 未列出的 GPU 驱动程序的每张 DRM 卡：amdgpu 提供来自 sysfs 的 `gpu_busy_percent` 和 VRAM，i915、xe、nouveau 或 nvidia 则没有使用率，一旦有模型使用它们，就用其后端的名称和内存信息补全；仅用于显示输出的适配器不包括在内，UI 也不会为没有内存数据的 GPU 绘制 VRAM 条），此外还有已加载模型按用途划分的内存。内存信息从不经过任务队列：每个提交的任务都会重置 `vTimeLastTask`，因此一个每两秒轮询一次的页面会让模型永远保持唤醒，而且 `mCreateResponse` 会等待休眠中的模型醒来。取而代之的是，主循环会发布一份快照（`mRefreshMonitorSnapshot`）：在 `mLoadModel` 之后发布，在槽位工作期间最多每秒发布一次，在槽位空闲后则总会发布一次（请求结束时会保存其提示词）；HTTP 线程只在互斥锁保护下复制这份快照，进入休眠时则发布 `{"state": "sleeping"}`。快照组合了 `mgs_get_memory_breakdown`（按缓冲区类型划分的模型、上下文和计算内存）、`mgs_model_get_weight_breakdown`（MTP 和路由专家的权重，由于需要遍历每个张量，只在加载后计算一次）以及 `mgs_get_expert_memory_breakdown`（专家缓存缓冲区和调度器的预取槽位，常规明细看不到这些）。目标模型上的 MTP 上下文与其共享权重，因此只计入它的 KV 缓存和计算缓冲区；单独的 MTP GGUF 计为 MTP 权重，任何其他草稿模型计为草稿权重。主机缓冲区类型（包括 GPU 的锁页主机内存）归入 RAM 条，设备缓冲区类型归入其所属的 GPU，并通过 PCI 总线 ID（小写，去掉 CUDA 虚拟设备的 `-vN` 后缀）与 NVML 匹配。`fMonitorReport` 会添加“other”（任何分段都无法解释的已用内存：其他进程、驱动程序上下文、投影器）以及剩余的空闲内存。路由器父进程从不初始化 GPU 后端，因为那会创建它自己的 CUDA 上下文：它通过 NVML 读取 GPU 信息，并向每个已就绪的子进程请求 `/api/monitor/memory`（超时 1 秒），而正在加载或休眠的子进程只报告其状态。`GET /api/monitor/usage`（只有使用率：全局 CPU、每个物理处理器（不含其逻辑 CPU）以及每块 GPU）在模型加载期间也允许访问。并发的轮询方会在 750 毫秒内共享 CPU 和 GPU 读数；超过 10 秒的 CPU 基准会被一个新的 250 毫秒采样区间取代。在 UI 中，`cMonitorStore` 只在有订阅者时轮询，并在标签页隐藏时暂停；“监控”页面打开时，状态栏会复用其报告。该页面只绘制使用率和内存：报告中各模型的条目（状态、常驻内存、专家缓存）保留在 API 中，供其他客户端使用。页面卡片的标题是加粗的设备标签（`gpu-N` 和 `cpu-N`，按报告顺序，每块 GPU 和每个物理处理器各一张卡片），后接设备名称；每张处理器卡片都显示系统平均负载，每个使用率计量条（`MonitorMeter`、每核心的条和状态栏）都是一条绿色的条（`--monitor-meter-free`），按使用率填充红色（`--monitor-meter-used`），与内存条的分类调色板无关。状态栏固定在底部：`.has-status-bar` 设置 `--status-bar-height`，聊天表单、侧边栏、粘性页脚和 `body` 的内边距都会加上这个值，因此隐藏状态栏时（“设置 → 显示”），所有偏移都与以前相同。内存条使用八种经过验证的分类颜色，外加一种中性的“其他”，并按固定顺序排列（`aMonitorSegmentGroups`），使相关的服务器分段共用同一种颜色（MTP 与草稿权重、专家缓存与预取）。“思考”菜单按内容确定其条目的尺寸（`w-max`，名称使用 `whitespace-nowrap`）：在全局的 `min-width: 0` 和 `overflow-wrap: anywhere` 作用下，紧挨着提示文字的 `flex-1` 名称以前会被逐字母拆开。

**Vulkan 后端。** `build/build-local.sh --vulkan` 和 Debian 安装程序（`--vulkan`，也可以通过 `MGS_BUILD_BACKEND=vulkan`）会在关闭 CUDA 的情况下配置 `-DGGML_VULKAN=ON`；着色器在构建时用 `glslc` 编译，CMake 需要 Vulkan 和 SPIR-V 头文件（`libvulkan-dev`、`spirv-headers`）。安装程序会安装这些软件包以及 `mesa-vulkan-drivers`；本地构建只检查 `glslc` 是否存在，并在配置失败时给出一条列出所需软件包的提示。本地捆绑包通过常规的 `ldd` 闭包复制 Vulkan 加载器 `libvulkan.so.1`，而驱动程序（ICD）由加载器从 `/usr/share/vulkan/icd.d/` 打开，并加载其自身的系统库（RADV 会引入 LLVM、libdrm 以及 X/Wayland 客户端库）；已在 `lib/mgs/` 中的 soname 会被共用，这就是捆绑包必须在运行它的同一 Debian 版本上构建的原因。Vulkan 设备报告 `async` 和 `events` 能力，因此专家预取和专家缓存无需改动即可运行；其 `device_id` 是 PCI 总线 ID，因此监控功能能将它与 NVML 或 amdgpu 的 sysfs 卡匹配（此时该卡的通用名称会被后端描述取代）。Vulkan 不导出 `ggml_backend_register_host_buffer`，因此映射的专家页面仍是可分页的：在 RTX 4060 Ti 上通过 Vulkan（Qwen3.6-35B-A3B Q4_K）测得，使用 mmap 时专家上传速度达到 2.7 GB/s，使用 `--load-mode none`（锁页主机缓冲区）时达到 13.1 GB/s，生成速度分别为 27.5 和 36.6 token/s；加载器现在会提示没有任何后端能够锁定映射内存，而不再归咎于 CUDA。渲染节点需要 `render` 组，`configure-web.sh` 会将该组授予服务账户（还有 `video` 组；在 Alpine 上以幂等方式添加）。

**Vulkan 下的真实 VRAM。** 在独立 GPU 上，`ggml-vulkan` 只统计和分配 device-local 内存（VRAM）；GTT 是 AMD 和 Intel GPU 能够映射的那部分系统 RAM（通常为一半），永远不会被算作 GPU 内存，因为 GPU 要通过 PCIe 读取它。三项保护措施确保模型位于真实的 VRAM 中。在没有 Resizable BAR 时，CPU 可见的 device-local 堆只是一个很小的窗口（RX 580 上为 256 MiB），还要与桌面共享，而 `ggml-vulkan` 以前会为每个设备缓冲区优先尝试这个堆：较小的缓冲区（约 99 MiB，其中包括 MTP 的 KV 缓存）落到了那里，内核又把其中 67 MiB 驱逐到了 GTT。现在，当该堆小于最大 device-local 堆的一半时，会自动设置 `disable_host_visible_vidmem`，使设备缓冲区进入大堆（在 RX 580 上：驱逐量为 0，窗口中仅占用 872 KiB）。专家缓存池使用第二种 Vulkan 缓冲区类型，其 `VK_EXT_memory_priority` 为 0.5（导出为 `ggml_backend_dev_get_cache_buffer_type`，由 `fPoolBufferType` 选用），因此在 VRAM 紧张时，驱动程序会先驱逐缓存，然后才是保持优先级 1.0 的 MTP 层、权重和 KV 缓存；异步复制的检查接受同一 GPU 的任何设备缓冲区类型（`ggml_backend_vk_buffer_is_device_of`）。最后，每个推理进程都会读取 `/proc/self/fdinfo`（`fMonitorDrmResidency`：amdgpu、i915 和 xe 发布的 `drm-resident-*`/`drm-memory-*` 和 `amd-evicted-vram`），将其附加到自己的监控快照中，当其 GPU 分配中有超过 64 MiB（或 2%）不在 VRAM 中时，在日志中警告一次（`mCheckVramResidency`），监控页面则会显示真实位置和一条警告。`--load-mode none` 不会被自动选择：在 AMD 上，锁页主机内存会计入 GTT，18.9 GB 的专家配合 16.8 GB 的 GTT 曾导致 RADV 失败，并报告“Not enough memory for command submission”（`ErrorDeviceLost`）。

**本地 shell MCP 服务器。** `--local-shell-mcp-server` 会启用 `server_shell_mcp`（`backend/tools/server/server-shell-mcp.cpp`），这是一个在 `mgs` 内部用 C++ 编写的 MCP Streamable HTTP 服务器，因此自包含捆绑包无需 Python 或额外的软件包。它不是一个独立的监听器：它以 `<api prefix>/mcp/shell` 的形式注册在与 API 相同的服务器上，因此 Web UI 可以从任何机器通过相同的源、TLS 证书、API 密钥和 Apache 代理访问它，而 `127.0.0.1` 上的端口只能从服务器本机访问。启动时，`mStart` 会在 `PATH` 中的绝对路径条目和常用目录中查找 `bash`、`zsh`、`sh` 和 `pwsh`/`powershell`，并且只公开解释器存在且可执行的工具（`bash.run`、`zsh.run`、`sh.run`、`psh.run`）。服务器不保存任何状态：每个 POST 一条 JSON-RPC 消息，以 JSON 响应，没有 SSE 流（GET 返回 405），也没有会话。`fRunCommand` 使用 `posix_spawn` 在一个新会话中启动解释器（拥有自己的进程组，没有服务器的控制终端，因此 sudo 永远不会在启动 `mgs` 的终端上提示输入密码），以 `/dev/null` 作为输入，使用默认的信号处理（服务器自身会忽略 SIGPIPE），并为 stdout 和 stderr 使用独立的管道，通过 `poll` 读取；超时（默认 60 秒，最长 3600 秒）或连接关闭会终止整个进程组，shell 退出后还会继续读取待处理的输出 250 毫秒，因此后台进程无法拖住调用。每个流都保留其开头和结尾（stdout 为 8+8 KiB，stderr 为 4+4 KiB）。每次调用 shell 工具都会留下自己的 JSONL 文件（`server_shell_mcp_call_log`），即使其参数被拒绝也是如此；该文件位于运行 `mgs` 的用户的 `~/.mgs/local-shell-mcp-server/` 中，以调用时的本地时间（精确到毫秒）命名（`y2026m10d12h23m59s59ms731.jsonl`，使用 `O_EXCL` 创建，因此同一毫秒内的调用会得到 `-2`、`-3`……）：首先是运行前写入的一行 `command`（工具、shell、cwd、超时、命令），然后每次读取写入一行 `stdout` 或 `stderr`，按到达顺序记录全部输出（`fRunCommand` 将每一段输出都交给 `pOutput`，`mOutput` 则会暂存被两次读取拆开的 UTF-8 字符），最后是一行 `result`（退出码、信号、是否超时、是否取消、秒数以及每个流的字节数）或一行 `error`（被拒绝的调用有记录所请求内容的 `command` 行，以及 `error` 行）。输出从不保存在内存中，因此该文件包含工具结果中被截掉的全部内容，并且即使 `mgs` 在运行期间停止，文件也会保留下来。文件夹以 0700 模式创建，文件以 0600 模式创建，因为输出可能包含机密信息；无法写入的文件只会在日志中留下一条警告。`mToolsCallResult` 以一个 JSON 对象作答（`vExitCode`，被终止的 shell 无法回收时为 null，以及 `stdout`、`stderr`、`signal`、`vTimedOut`、`cancelled`、`timeout`、`seconds`、`shell`、`cwd`），同时作为 `structuredContent` 和文本块返回，因为 UI 只保留工具结果的文本：模型读取的是同一个 JSON，而聊天界面将其拆分为多个框，无需猜测一个流在哪里结束。带有 `Origin` 头的请求必须来自本服务器的页面（`fOriginAllowed`，与 `Host`/`X-Forwarded-Host` 比较；为开发代理也接受回环地址的页面），并且必须使用 `Content-Type: application/json`，这可以阻止跨站请求；不带 `Origin` 的客户端则依靠 API 密钥。与其他智能体功能一样，该选项会把默认 CORS 源限制为 localhost，并添加一条启动警告。路由器子进程永远不会收到该选项，并且在模型加载期间该路由也会响应。`/api/props` 发布 `shell_mcp_server`；`cMcpStore.mSyncShellServer`（从根布局调用）会添加 `mgs-shell` 条目（名为 `local shell mcp server`，与服务器的 `serverInfo.title` 一致），首次添加时启用它，使其 URL 始终指向当前源并保持其名称，并在服务器不再提供它时将其移除；`MCPService.mCreateTransport` 会将 UI 的 API 密钥发送给同源的 MCP 服务器。工具调用仍要经过 UI 的权限提示。在聊天中，`ChatMessageToolCallBlock` 会将 `bash.run`、`zsh.run`、`sh.run` 和 `psh.run` 调用（`fIsLocalShellTool`：位于 `aShellMcpToolNames` 中、且不属于其他任何 MCP 服务器的名称）交给 `ChatMessageToolCallBlockLocalShell`，后者以工具名称作为标题，并显示与其他工具的“输入”框相同的灰色等宽字体框：stdin 中只有命令本身，在参数流式传输期间从已收到的部分参数中解析；stdout 和 stderr 只在有内容时显示，标题中带有退出码（`stdout (代码 2):`，没有输出时则为 `stdin (代码 0):`）；stdin 框为白色边框（浅色主题下为深灰色），stdout 框为绿色边框，stderr 框为红色边框，四种工具都一样（`SyntaxHighlightedCode` 的 `borderColor`，配合 `--local-shell-stdin-border`、`--local-shell-stdout-border` 和 `--local-shell-stderr-border`）。不是该 JSON 对象的结果（参数错误、权限被拒绝、请求失败）会显示在 stderr 框中。`expandLocalShellCommands` 设置（“显示”）会让 `ChatMessageAgenticContent.fGetDefaultExpanded` 展开这些调用，因此各个框会在其内容可知时随即出现。

**Web UI 设置。** 界面的所有设置都保存在一个 JSON 文档中，服务器将其保存在运行 `mgs` 的用户的 `~/.mgs/config/ui-settings.json` 中（`backend/tools/server/server-ui-settings.cpp`：文件夹 0700，文件 0600，在互斥锁保护下先写入临时文件，再重命名覆盖旧文件），因此所有浏览器共享这些设置。`GET /api/ui/settings` 返回 `{"settings": DOCUMENT}`（首次保存之前为 `null`），`POST /api/ui/settings` 则整体替换该文档；两者都需要 API 密钥，在所有模式下都会注册（路由器父进程自行提供它们），并且在模型加载期间也会响应。`/api/config.js` 会将已保存的语言发布为 `globalThis.cUiLanguage`（仅限有效的语言标签），因为 `i18n.ts` 在模块导入时就确定了语言。在界面中，`cUiSettingsStore`（`frontend/src/lib/stores/ui-settings.svelte.ts`）会在根布局挂载时加载该文档，并将每个部分交给注册它的 store：`config` 和 `userOverrides`（设置 store）、`favoriteModels`（模型）、`disabledToolKeys`（工具）、`alwaysAllowedTools`（权限）、`reasoningEffortDefault`（对话）以及 `dismissedRecommendedMcpServers`（添加 MCP 服务器对话框）；`language` 由 `fSetLanguage` 写入，它会等保存完成后再重新加载。`mSet` 在最后一次更改 300 毫秒后保存整个文档，仍在等待的更改会在页面卸载时通过 `keepalive` 发送。文档到达之前不会保存任何内容，因此不完整的状态永远不会覆盖该文件；加载失败时每 5 秒重试一次，但 401/403 除外，此时会等待 API 密钥。服务器默认值的同步和 shell MCP 条目都会等待 `/api/props` 和该文档；文档不存在即表示首次访问，此时会应用一次 `--ui-config` 默认值。只有 API 密钥保留在浏览器中（`MgsUi.apiKey`），因为它用于验证加载其余设置的请求；对话保留在 IndexedDB 中，流恢复和构建版本的键则保留在 `localStorage` 中。

**请求队列。** 服务器默认只有一个槽位（`-np 1`，`common_params::vNParallel`；上游那种由四个槽位共享统一 KV 缓存的自动模式已被移除，因此 `--kv-unified` 默认也是关闭的）：每个请求都获得完整的上下文，无论它来自 Web UI 还是直接来自 API，其他请求则在 `server_queue` 的延迟队列中等待，并按到达顺序执行（槽位释放后，第一个延迟任务会被移到任务队列的最前面，`mPopDeferredTask`）。`-np N` 最多并行运行 N 个请求，并在它们之间分配上下文。API 不依赖于界面或其设置文档；两者唯一的联系是 Web UI 会用 `"mgs_client": "webui"` 标记其请求（这是补全 schema 中的一个 `field_str`，用于设置 `task_params::vFromWebui`），从而让服务器知道每个正在运行和正在等待的请求的来源。当一个任务找不到空闲槽位时，`mDeferTask` 会统计繁忙的槽位（`mCountRunning`）以及排在它前面的延迟任务（`server_queue::mDeferredTasks`），并按 API 和 Web UI 分别计数；如果该请求要求接收通知（`vQueueNotice`：来自 Web UI、采用流式传输，且格式不是 Anthropic Messages 或 OpenAI Responses——这两者的事件流语法是固定的），就发送一个 `server_task_result_queue`，它以 `data: {"mgs_queue": {"ahead": N, "ahead_api": A, "ahead_webui": W}}` 的形式到达流中。流式处理函数会等收到第一个结果之后才发送响应头，因此该事件也会立即开启响应。当排队的任务获得槽位时，它会收到最后一条所有计数都为 0 的通知，`mNotifyQueuePositions` 则会更新仍在等待的任务。不发送 `mgs_client` 的客户端永远不会收到这些事件，因此它们的 OpenAI 兼容流保持不变。在 UI 中，`ChatService.mHandleStreamResponse` 将这些事件传给 `onQueue`，并在收到第一个常规数据块时清除该状态；`cChatStore` 按对话保存该状态（`mGetQueueStatus`），`ChatMessageAssistantProcessingInfo` 会在其他任何进度文本之前显示 `fQueueStatusText`：如果前面有任何请求直接来自 API，就显示 API 版本的文本，否则显示 Web UI 版本，两者都带有排在前面的请求数。UI 自行发送的请求（响应后的 KV 缓存预填充、生成的标题）也带有相同的标记，同样需要排队等候。

**sudo 密码提示。** 在 Linux 上，如果存在 sudo，`mStart` 还会创建 `server_shell_mcp_askpass`：一个抽象 Unix 套接字（没有需要保护或清理的文件），由其专用线程提供服务，每个连接一个线程。`_meta` 中带有 `mgs/callId` 的 `tools/call`（UI 在 `cMcpStore.mExecuteTool` 中为每次调用生成一个）会连同一个随机令牌一起注册，其命令会获得 `SUDO_ASKPASS`（即 `mgs` 可执行文件本身：由启动器发布的 `MGS_EXECUTABLE`，或 `/proc/self/exe`）、`MGS_LOCAL_SHELL_ASKPASS`（套接字名称和令牌），以及在服务器没有该变量时的 `DISPLAY=:mgs-askpass`，因为只有当 `DISPLAY` 存在时，sudo 才会在没有终端的情况下改用 askpass 辅助程序。sudo 会运行 `mgs <prompt>`；`fServerShellMcpAskpassMain` 是 `mgs_server` 调用的第一个函数，它会识别这种情况（该变量、单个参数以及 sudo 父进程），并通过套接字发送令牌和提示。`mServe` 会用 `SO_PEERCRED` 检查辅助程序属于同一用户，并用 `fParentIsSudo` 检查其父进程是 sudo（名称为 `sudo`，真实 uid 为本进程的真实 uid，有效 uid 或保存的 uid 为 root：无特权进程无法读取 setuid 进程的可执行文件，因此这是可用的最强检查），因此命令无法自己用伪造的提示启动该辅助程序来窃取密码。随后，提示会等待（最多 5 分钟），同时 `fRunCommand` 会把其截止时间顺延等待所花的时间（`mPaused`）；`DialogLocalShellAskpass` 在根布局中只挂载一次，因此可以在任何页面上打开，它会针对每个正在运行的调用（`cMcpStore.mActiveShellCalls`，由 `mExecuteTool` 填入工具和命令）每 500 毫秒轮询一次 `GET /api/mcp/shell/askpass?call=`，并在一个居中的对话框中显示第一个提示（ShadCN 的 `Dialog`，不带关闭按钮：提示一出现，密码字段就获得焦点（`fFocusPassword`，即该字段的一个 attachment，因为 `onOpenAutoFocus` 在其 ref 绑定之前就已运行），“接受”或 Enter（`fHandlePasswordKeydown`）作答，“取消”或 Escape 取消，点击对话框外部不起作用），其中包括 sudo 的提示、工具和命令，并在输入密码后 sudo 再次询问时显示一条通知；`POST /api/mcp/shell/askpass` 递交密码或取消操作，辅助程序将其写入 sudo 读取的管道，各份副本随即被擦除。密码永远不会进入工具结果、对话或日志；错误（已取消、未应答、命令已结束）会以 `mgs: ...` 的形式写入命令的 stderr。其他 MCP 客户端不发送调用 ID，因此它们的命令不会获得 askpass 变量，sudo 会像以前一样因缺少终端而失败。
**安全加固。** 0.1.9 版处理了一次外部静态安全审查的发现（共 21 项：已修复 20 项；有关安装程序的一项按设计保留，见下文）。
- **HTTP（`server-http.cpp`）：** 公共路径只有在 GET/HEAD 请求时才跳过 API 密钥，并且在存在密钥时，`GET /models?reload=...` 需要密钥。在工具和 MCP 服务器所启用的 `localhost` CORS 模式下，方法不是 GET、HEAD 或 OPTIONS 的请求，如果其 `Origin` 既不是回环地址、也不是本服务器（`Host` 或 `X-Forwarded-Host`；Apache 通过 `ProxyPreserveHost On` 保留 `Host`），会在路由之前收到 403；不带 `Origin` 的请求（curl、SDK）则可以通过。`POST /tools` 需要 `Content-Type: application/json`（否则返回 415），因此跨站表单若不经过会被 CORS 拒绝的预检请求，就无法访问它。接收请求期间，每次读取最多等待 30 秒（`cHttpRequestReadTimeoutSec`）；写入仍由 `--timeout` 控制，因此长时间的生成不会被中断。两个线程池各自最多排队 256 个已接受的连接（`cHttpMaxQueuedConnections`），并且 `mPost`/`mDel` 会在任何处理函数解析之前，拒绝嵌套深度超过 1024 层数组或对象的请求体（`fJsonNestingExceeds`、`cHttpMaxRequestJsonDepth`），因为 nlohmann 会以递归方式复制和序列化文档。
- **路由器（`server-models.cpp`）：** `fRedactArgs` 和 `fRedactPreset` 会在 `status.args`、`status.preset` 和启动子进程的日志中，将机密选项（`sSecretOptionEnvs`：HF 令牌、API 密钥和密钥文件、SSL 密钥、MCP 服务器配置）的值替换为 `<redacted>`；子进程仍会收到真实值。`mProxyRequest` 在转发给子进程时会丢弃 `Origin`，因为路由器已经检查过它。
- **CORS 代理（`server-cors-proxy.h`）：** 导航和嵌入请求（`Sec-Fetch-Mode` 为 navigate，`Sec-Fetch-Dest` 为 document、frame、object 或 worker）会收到 403（`fProxyIsNavigation`），并且每个经过代理的响应都会被无害化处理（`fProxyMakeInert`）：加上 sandbox CSP、`nosniff`、`Content-Disposition: attachment` 和同源 CORP，并去除上游的 cookie、CSP、CORS、HSTS、重定向、link 和 service-worker 响应头。`Content-Type`、`mcp-session-id` 和 `WWW-Authenticate` 予以保留，供 UI 的 MCP 客户端使用。
- **远程媒体（`download.{h,cpp}`、`server-common.cpp`）：** `fCommonRemoteGetContent` 会应用连接超时和总截止时间（`set_max_timeout` 加上在正文接收器中的检查），并手动跟随重定向。对于 `image_url`、`input_audio` 和 `input_video`（`vPublicHostsOnly`），主机只解析一次，并且每个地址都必须是公网地址：不允许回环、RFC 1918、CGNAT、链路本地（包括 169.254.169.254）、文档用途、组播或保留地址段，IPv6 中只允许 2000::/3，映射地址、NAT64 和 6to4 地址则按其对应的 IPv4 地址判断。连接被固定到已检查的地址（`set_hostname_addr_map`），每次重定向都会再次检查；`--media-allow-private-urls` 会取消地址检查。媒体下载的限制为：每次连接、读取或写入 10 秒，总计 30 秒，5 次重定向，10 MiB，同时最多 8 个（`media_download_slot`）；当输入是请求缓冲区时，`ffprobe` 和 `ffmpeg` 只从管道读取。
- **解码（`mtmd-helper.cpp`、`server-common.cpp`）：** 图像每边最多 32768 像素、总计最多 1 亿像素，在解码前通过 `stbi_info_from_memory` 检查；音频最长一小时，在分配采样数据之前检查；视频帧每边最多 8192 像素，每个视频的 RGB 帧最多 2 GiB；每个请求解码后的图像和音频总量最多 2 GiB（`fProcessMtmdPrompt`）。
- **解析器：** GBNF 转换器（`json-schema-to-grammar.cpp`）会拒绝空模式以及没有可重复内容的量词；它将 schema 嵌套、嵌套访问和 `$ref` 链限制为 256（`cMaxSchemaDepth`，由 `fJsonExceedsDepth` 以迭代方式测量），将模式中的分组限制为 128 个，排除的属性名称限制为 1024 字节，`type` 列表限制为 7 项，并用循环代替递归的 `std::regex` 来清理规则名称。Jinja 运行时（`jinja/value.h`）将 `range()` 限制为一百万个元素，构建的字符串限制为 256 MiB 和 1 Mi 个片段，`indent()`/`tojson()` 的缩进限制为 256，语句执行的嵌套限制为 1024，解析的嵌套限制为 256。`gguf.cpp` 在溢出检查中会跳过零维度，而不是除以它们；空张量仍然有效。
- **RPC（`ggml-rpc/`）：** 变长的 `recv_msg` 采用按命令设定的上限（SET_TENSOR：消息头加上该连接中最大的存活缓冲区；GRAPH_COMPUTE：`INT32_MAX`，即客户端所能发送的最大值），并捕获 `std::length_error`。对端发送的张量边界使用防溢出的辅助函数（`rpc_checked_nbytes`、`rpc_tensor_region_ok`）检查，并以返回错误代替 `GGML_ASSERT`；任何异常都只会关闭该连接，释放缓冲区时会丢弃使用它的已存储计算图。RDMA 接收从不复制超过所请求的数据量，其余部分保留在 `rx_pending` 中；队列对不授予任何远程访问权限。
- **Web UI：** Markdown 渲染失败时显示转义后的文本。协议不是 http、https、mailto 或 tel（也不是相对链接）的链接会失去其 `href`，图片附件必须是 `data:image/...;base64`，图片损坏的提示使用 DOM 节点构建（`fCreateImageErrorFallback`）。对话导入在解压之前会检查文件（512 MB）、ZIP 条目（10 000 个）、每个 JSONL（256 MB）以及解压后的总量（1 GB），验证每一条记录，要求每条消息的 `convId` 与其所属对话一致，并使用 `bulkAdd` 写入，跳过消息 ID 已存在的对话。每个工具名称对应唯一的身份（`mResolveEnabledEntry`），同时用于模型、审批和执行，执行会发往确切的那个服务器（`mExecuteTool(pToolCall, pServerId, pSignal)`）；由两个已启用的 MCP 服务器同时提供的名称会被扣下并发出警告，“始终允许以下服务器的所有工具”则按服务器 ID 生效。
- **Web UI 依赖项（`frontend/package.json`，0.1.10）：** 被 `npm audit` 标记且会进入浏览器包的软件包已升级到修复版本：DOMPurify 3.4.17、mermaid 11.17.2 和 `@modelcontextprotocol/sdk` 1.32.1。SvelteKit 2.70.3 和 `@vite-pwa/assets-generator` 1.0.4（sharp 0.35.5，需要 Node.js 20.9 或更高版本）也已更新，`npm audit fix` 在各自的版本范围内刷新了构建期的传递依赖包；直接依赖的版本保持精确锁定。`npm audit` 仍然列出的问题只涉及构建工具（通过 `@tailwindcss/typography` 引入的 `postcss-selector-parser`、Windows 上的 esbuild 开发服务器）或 KaTeX，后者的修复版本（0.18.2）超出了 `rehype-katex` 和 mermaid 所接受的 `^0.16` 范围；该 KaTeX 安全公告需要事先存在原型污染才能被利用。
- **未处理：** 将 `curl | bash` 安装程序固定到不可变的修订版本或签名（项目规则要求从 `main` 安装）、整个请求头块的绝对截止时间（cpp-httplib 对每次读取分别应用超时）、RPC 读取截止时间和并发客户端、RPC 服务器中迭代式的 `create_node`、Jinja 模板的全局 CPU 和内存预算，以及 `mgs-grammar.cpp` 对请求中 `grammar` 的递归。

<a id="modules"></a>
## 模块图

| 模块 | 路径 | 职责 | 依赖 | 使用方 |
| --- | --- | --- | --- | --- |
| MoE 模型清单 | `backend/src/CMakeLists.txt; backend/src/models/models.h; backend/src/mgs-model.cpp` | 编译 MoE 计算图及其所需的头部/共享块；拒绝稠密主模型。 | GGUF 元数据、模型加载器 | 模型初始化 |
| 构建 | `build/CMakeLists.txt; backend/CMakeLists.txt` | 组合引擎、服务器和本地 UI；输出 `mgs`，同时保留上游 target。 | CMake、C/C++ 工具链 | deploy/install-common.sh; build/build-local.sh |
| 版本 | `backend/cmake/build-info.cmake; backend/common/build-info.{h,cpp.in}; deploy/scripts/ui-assets.cmake; frontend/scripts/vite-plugin-build-info.ts` | 只在一处定义 `MGS_VERSION`，并将其传递给二进制文件（`mgs_version()`）、库文件版本、CMake 包以及界面的 `build.json`。 | CMake | `--version`、`/api/props`、OpenAI 响应、shell MCP、下载、UI |
| 原生 TLS | `backend/tools/server/server-tls.cpp; backend/tools/server/server-http.cpp` | 自动证书或 PEM 证书、默认 HTTPS，以及使用空闲端口的 HTTP 重定向。 | OpenSSL、cpp-httplib | 公开可执行文件和路由器 |
| HTTP | `backend/tools/server/server-http.cpp` | 注册带前缀的 API，对请求进行身份验证（公开路径仅限 GET/HEAD），在 `localhost` CORS 模式下拒绝跨站写入，限制请求读取、连接队列以及请求体的 JSON 嵌套深度，并在根路径提供 UI 资源。 | cpp-httplib、mgs-ui | server.cpp |
| 远程媒体 | `backend/common/download.{h,cpp}; backend/tools/server/server-common.cpp` | 在总时限内获取 `image_url`、`input_audio` 和 `input_video` 的 URL，只访问经一次解析后即固定下来的公网地址（除非使用 `--media-allow-private-urls`），逐一检查每次重定向，单个上限 10 MiB，最多同时进行 8 个下载。 | cpp-httplib、getaddrinfo | 推理路由 |
| 媒体解码限制 | `backend/tools/mtmd/mtmd-helper.cpp; backend/tools/server/server-common.cpp` | 解码前检查图像尺寸，分配前检查音频时长，检查视频帧尺寸和帧预算，以及每个请求解码后的总量。 | stb_image、miniaudio、ffmpeg | mtmd、server |
| CORS 代理 | `backend/tools/server/server-cors-proxy.h` | 为 UI 获取 MCP 端点，拒绝导航请求并返回惰性响应。 | http.h、server-models | UI MCP 客户端 |
| 推理 | `backend/tools/server/server-context.cpp` | 管理槽位、推测解码指标和解码；在统计 token 之前唤醒。 | mgs-common、mgs-engine、mtmd | HTTP、路由器 |
| 请求队列 | `backend/tools/server/server-{context,queue,task,schema}.{cpp,h}; frontend/src/lib/services/chat.service.ts; frontend/src/lib/stores/chat.svelte.ts; frontend/src/lib/utils/progress.ts` | 默认按到达顺序一次只运行一个请求（`-np 1`）；告知流式 Web UI 请求（`mgs_client`）在它之前有多少个 API 请求和 Web UI 请求正在运行或等待，并在聊天中显示出来。 | server_queue、槽位、补全 schema、15 个语言目录 | 聊天、API 客户端 |
| Kimi-K3 | `backend/src/models/kimi-k3.cpp` | KDA + MLA 混合架构、跨层残差混合以及潜空间 MoE。 | mgs-graph（KDA、MoE）、ggml `dsv4_hc_pre` | 模型计算图 |
| GLM-5.3-Flash 和 glm5v | `backend/src/models/glm5-next.cpp; backend/tools/mtmd/models/glm4v.cpp` | 带 k-pool 索引器和 MTP 的 KDA + MLA；glm5v 视觉投影器。 | mgs-memory-hybrid-idx、ggml `swiglu_clamp` | 模型计算图、mtmd |
| Qwen3.8-Flash-Next | `backend/src/models/qwen4exp.cpp` | 超连接（hyper-connections）、n-gram 逐层嵌入、门控 delta-net、QSA 注意力和 MTP。 | mgs-memory-hybrid-idx、ggml `dsv4_hc_*`、`lightning_indexer` | 模型计算图 |
| 混合索引内存 | `backend/src/mgs-memory-hybrid-idx.{h,cpp}` | KV 缓存、循环状态缓存和 k-pool 索引器缓存。 | mgs-kv-cache、mgs-memory-recurrent | qwen4exp、glm5-next |
| DeepSeek-V4.1 | `backend/src/models/deepseek41.cpp; backend/src/mgs-kv-cache-dsv41.{h,cpp}` | 两种 GGUF 布局、外部 Engram 表、共享的压缩 KV 和候选池。 | mgs-model-loader（`cTensorExternal`）、ggml `dsv4_hc_*`、`lightning_indexer`、`swiglu_clamp` | 模型计算图 |
| 路由器 | `backend/tools/server/server-models.cpp` | 管理模型子进程，在父进程保留基于 API 密钥文件的身份验证，并以不带 `Origin` 的方式转发规范的 API 请求；在模型列表和日志中隐去机密选项；当启动器设置了 `MGS_EXECUTABLE` 时重新执行它。 | 子进程、HTTP、流 | server.cpp |
| 子进程 | `backend/common/subproc.{h,cpp}` | 通过 `mCreate`、`mAlive`、流访问器、`mCloseStdin`、`mTerminate` 和 `mJoin` 管理子进程句柄；两个构建分支都与头文件保持一致。在调用方和禁用构建时的桩代码中保留供应商的 `subprocess_option_*` 名称。 | sheredom/subprocess.h、std::vector | 路由器、MCP stdio、服务器工具 |
| 视频输入 | `backend/tools/mtmd/mtmd-helper.cpp` | 通过 ffprobe/ffmpeg 子进程解码文件或缓冲区。传递声明的参数以及公开的 `ffmpeg_bin_dir`/`timestamp_interval_ms` 字段；使用 `push_back` 构建命令向量。 | mtmd、子进程、ffmpeg/ffprobe | 多模态输入 |
| 采样缓冲区 | `backend/common/sampling.cpp; backend/src/mgs-sampler.cpp` | 保留环形缓冲区自身的 `mPushBack` 方法；它转换为 `std::vector` 时使用标准的 `push_back` 方法。 | std::vector | 通用采样器、引擎采样器 |
| 前端构建 | `frontend/package.json; frontend/vite.config.ts; frontend/pwa-assets*.config.ts; frontend/scripts/vite-plugin-*.ts` | 在生成 PWA 资源和用 Vite 编译之前运行 `svelte-kit sync`；向上传递同步错误。直接导入当前的 `dSveltekitPwaOptions`、`cBuildConfig`、PWA 常量以及 `cTab`/`cNewline`。 | frontend/src/lib/constants、SvelteKit、Vite、PWA 资源生成器 | npm run build、mgs-ui |
| 聊天路由 | `frontend/src/routes/(chat)/+page.ts; frontend/src/routes/(chat)/chat/[id]/+page.ts` | 导出框架要求的确切名称 `load`（哈希路由器支持该名称），并在渲染聊天之前通过 `/api/props` 验证 API 密钥。 | SvelteKit PageLoad、api-key-validation | 新对话和已有对话 |
| 视觉注意力 | `backend/tools/mtmd/models/granite4-vision.cpp` | 将归一化后的输入张量与 Q/K/V 投影变量分开。Granite 自注意力的三个投影及其残差都使用同一个归一化后的查询输入。 | GGML、clip 计算图 | 视觉投影器 |
| 流 | `backend/tools/server/server-stream.cpp` | 保留生成会话，并从游标处恢复。 | server-task、server-queue | 聊天、路由器 |
| API 文档 | `backend/tools/server/server-api-doc.cpp; frontend/api.html` | 根据路由注册表和补全 schema 构建 OpenAPI；嵌入 Swagger UI。 | server-schema、vendor/swagger-ui | /api/doc/ |
| Shell MCP 服务器 | `backend/tools/server/server-shell-mcp.{h,cpp}` | 检测已安装的 shell，并在 `/api/mcp/shell` 以 MCP Streamable HTTP 方式提供 `bash.run`、`zsh.run`、`sh.run` 和 `psh.run`；在独立的会话中带超时运行命令；将每次调用记录到 `~/.mgs/local-shell-mcp-server/` 下各自的 JSONL 文件中。 | server-http、server-common、POSIX spawn | server.cpp；UI `cMcpStore` |
| sudo 密码提示 | `backend/tools/server/server-shell-mcp.{h,cpp}; backend/tools/server/server.cpp` | 将 `mgs` 作为 SUDO_ASKPASS 运行，只接受父进程为 sudo 的辅助程序，挂起每个提示直到 UI 作答，并在此期间暂停该命令的超时计时。 | Linux 抽象 Unix 套接字、/proc | shell MCP 服务器、UI 密码对话框 |
| UI shell MCP 条目 | `frontend/src/lib/stores/mcp.svelte.ts; frontend/src/routes/+layout.svelte; frontend/src/lib/services/mcp.service.ts` | 将 `/api/props` 中的 `shell_mcp_server` 映射为名为 `local shell mcp server` 的 `mgs-shell` MCP 服务器，并对同源 MCP 请求进行身份验证。 | serverStore、settingsStore | MCP 设置和聊天 |
| UI 本地 shell 调用 | `frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockLocalShell.svelte; frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts; frontend/src/lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte` | 将 shell MCP 工具的调用显示为带退出码的 stdin、stdout 和 stderr 框；通过 `expandLocalShellCommands` 自动展开这些调用。 | mcpStore、settingsStore、SyntaxHighlightedCode | 聊天 |
| UI 密码对话框 | `frontend/src/lib/services/local-shell-askpass.service.ts; frontend/src/lib/components/app/dialogs/DialogLocalShellAskpass.svelte; frontend/src/lib/stores/mcp.svelte.ts; frontend/src/routes/+layout.svelte` | 在每个 shell 调用的 _meta 中发送调用 ID，轮询正在运行的调用中待处理的 sudo 提示，并将其显示在居中的对话框中，由该对话框提交密码或取消请求。 | apiFetch、mcpStore、ShadCN Dialog | 根布局 |
| 参数和 MTP | `backend/common/arg.cpp; backend/common/common.cpp` | 解析选项（包括 `--load-mode`），并在自动启用 MTP 之前检查 GGUF，包括单头的 glm4moe。使用单独的 MTP GGUF 时，嫁接在目标模型上的 MTP 层不会被加载。 | GGUF、mgs-engine | 服务器、模型初始化 |
| 模型加载 | `backend/src/mgs-model-loader.cpp; backend/src/mgs-model.cpp; backend/src/mgs-mmap.cpp` | 加载权重，保留/注册映射，并将 Direct I/O 暂存区限制在 64 MiB。MTP 权重和输出层保留在设备内存中，并忽略指向主机内存的覆盖设置。 | GGML 后端注册 | mgs-model |
| 专家预取 | `backend/ggml/src/ggml-backend.cpp` | 借助有界环形队列让专家传输与计算重叠，同步无输入的 split，并跳过空 ID。预取上传可以由上传回调（专家缓存）提供；无输入的 split 只在复制了输入之后或使用共享缓冲区类型时才需要等待。 | GGML 设备和缓冲区 | 计算图调度器 |
| 前端 | `frontend/src/routes; frontend/src/lib` | 渲染聊天/设置页面并导出对话。根据处理指示器的位置推导其边距；通过 getter 暴露当前的文件上传错误。ModelsStore 直接导入身份验证头，避免工具函数 barrel 导致的 chunk 循环依赖。头部脱敏让默认脱敏集合与额外脱敏集合保持相互独立。 | Svelte、浏览器存储、API | 浏览器 |
| 本地化 | `frontend/src/lib/i18n.ts; frontend/src/lib/locales; frontend/src/lib/components/app/settings/LanguageSelector.svelte` | 按字母顺序注册 15 种语言区域，解析 JSON 消息，插入变量值，并将所选语言保存到服务器的设置文档中。 | 15 个 JSON 语言目录 | UI 组件、配置和根布局 |
| 回复语言 | `frontend/src/lib/utils/response-language.ts; frontend/src/lib/services/chat.service.ts; frontend/src/lib/constants/settings-{keys,registry}.ts; frontend/src/lib/locales` | 持久化需要主动开启的 `respondInInterfaceLanguage` 设置（默认为 false）；在每个发出的补全请求和 KV 预填充中加入当前语言区域的地区化指令。 | fGetLanguage、fTranslate、SettingsStore、15 个语言目录 | 普通聊天、重新生成、继续生成、工具轮次、生成的标题 |
| 文本方向 | `frontend/src/routes/+layout.svelte; frontend/src/app.css; frontend/src/lib/components/app/navigation/SidebarNavigation; frontend/src/lib/components/app/settings/SettingsChatDesktopSidebar.svelte; frontend/src/lib/components/app/chat/ChatForm/ChatFormTextarea.svelte; frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserBubble.svelte` | 设置 HTML 的语言/方向，按逻辑方向定位导航，跟随聊天文本的方向，并将代码隔离为 LTR。 | `fGetLanguage`、CSS 逻辑属性、HTML `dir` | 所有页面、聊天和代码预览 |
| 部署 | `deploy/install-common.sh; deploy/configure-web.sh` | 安装二进制文件，并配置 Apache 和系统服务。 | apt/apk、OpenSSL、systemd/OpenRC | 生产主机 |
| Debian 引导 | `deploy/install-update-reinstall-debian.sh` | 为 curl-to-Bash 方式下载/同步源码，加锁并将编译委托出去。 | curl、tar、rsync、flock、install-common.sh | 本地/远程 Debian 安装 |
| Web 启动 | `deploy/start-web.sh; deploy/apache2.conf.in` | 选择 HTTP/HTTPS 端口，以非 root 身份启动 Apache，遇到绑定冲突时重试，并发布实际生效的 URL。 | Bash、Apache、iproute2、flock | moe-gguf-server-web |
| 项目许可证 | `LICENSE` | 适用于 nipegun 所作贡献的 PolyForm NonCommercial License 1.0.0，以其 `Required Notice:` 行开头；前言保留了 llama.cpp 的原始声明（The ggml authors），并将每个第三方部分指向其在 `build/licenses/` 中的许可证。以 `LICENSE` 的名称复制到生成包的 licenses/ 文件夹中。 | build/licenses/ | GitHub、build/build-local.sh |
| 许可证声明 | `build/licenses/` | 存放第三方许可证声明（`LICENSE-llama.cpp`，即 ggml 作者的 MIT 许可证；`LICENSE-FreeToken`；`LICENSE-swagger-ui` 及 `NOTICE-swagger-ui`；`LICENSE-cpp-httplib` 和 `LICENSE-jsonhpp`），并将它们复制到生成包的 licenses/ 文件夹中。 | 第三方声明 | LICENSE、build/build-local.sh |
| 本地构建 | `build/build-local.sh; build/mgs-launcher.c` | 选择本地源码或通过 HTTPS 下载 main 分支，在私有的 /tmp/ 工作区中编译并组装可重定位的安装，默认安装到 `~/.local` 下（`bin/mgs`、`lib/mgs/`），并在退出时清理；静态启动器加载捆绑的库，NVIDIA 驱动除外。 | CMake、ldd、patchelf、静态 libc、CUDA Toolkit、Node.js；下载时需要 curl/gzip | 工作站手动使用、本地文件或 curl-to-Bash |
| Vulkan 后端 | `backend/ggml/src/ggml-vulkan; build/build-local.sh; deploy/install-common.sh; deploy/configure-web.sh` | 通过 `--vulkan` 构建 Vulkan 后端（本地构建和 Debian 安装程序均支持），安装其构建所需的软件包和 Mesa 驱动，捆绑加载器但从系统获取驱动，并将服务账户加入 `render` 和 `video` 组。 | glslc、libvulkan-dev、spirv-headers、Vulkan 驱动 | 不使用 CUDA 的 AMD、Intel 和 NVIDIA GPU |
| MoE 内核 | `backend/ggml/src/ggml-{cuda,cpu,vulkan,sycl}` | 路由专家，调优 tile/融合，并处理注意力尾部。CPU 的 `mul_mat_id`/`add_id` 会跳过负的专家标识符。 | GGML、设备编译器 | 推理计算图 |
| MTP 执行 | `backend/src/mgs-{context,batch,model}.cpp; backend/src/models/glm4-moe.cpp; backend/common/{sampling,speculative,fit}.cpp` | 缓存独立的计算图 arena，保持行顺序，并管理草稿头和接受情况。 | mgs-engine、GGML | 服务器槽位 |
| 草稿指标 | `backend/tools/server/server-{context,task}.{cpp,h}` | 汇总草稿/接受计数器，发布 Prometheus 指标。 | 槽位、任务队列 | /api/metrics |
| UI 持久化 | `frontend/src/lib/stores/{conversations,tools,permissions,models,settings}.svelte.ts` | 导出持久化的对话树（IndexedDB），并将工具可用性、权限、收藏和用户设置保存在服务器的设置文档中。 | IndexedDB、UI 设置 | 聊天、设置 |
| 对话导入 | `frontend/src/lib/stores/conversations.svelte.ts; frontend/src/lib/services/database.service.ts; frontend/src/lib/constants/conversation-import.ts` | 在解压之前限制文件和 ZIP 条目的大小，校验记录，并添加新对话而不覆盖已有消息。 | fflate、IndexedDB（Dexie） | 导入对话框 |
| UI 链接和图像安全 | `frontend/src/lib/utils/safe-url.ts; frontend/src/lib/utils/image-error-fallback.ts; frontend/src/lib/components/app/content/MarkdownContent/plugins/rehype/{enhance-links,resolve-attachment-images}.ts; frontend/src/lib/components/app/content/MarkdownContent/MarkdownContent.svelte` | 只保留安全的链接协议和 `data:image` 附件，用 DOM 节点构建图像损坏提示，并在 Markdown 渲染失败时显示转义后的文本。 | rehype、DOM | 消息渲染 |
| UI 工具标识 | `frontend/src/lib/stores/{tools,mcp,agentic}.svelte.ts` | 为每个工具名解析出唯一一个已启用的条目，用于模型、审批以及在该确切 MCP 服务器上的执行；不提供同时由两个服务器提供的名称。 | MCP 客户端、权限 | 智能体循环 |
| UI 设置 | `backend/tools/server/server-ui-settings.{h,cpp}; frontend/src/lib/stores/ui-settings.svelte.ts; frontend/src/lib/services/ui-settings.service.ts` | 将界面的设置文档保存在 `~/.mgs/config/ui-settings.json` 中，提供 `GET`/`POST /api/ui/settings`，在 `/api/config.js` 中发布已保存的语言，并将每个部分交给负责它的 store。 | std::filesystem、POSIX 文件 API、API fetch | 设置、模型、工具、权限、对话、i18n、根布局 |
| MoE 专家缓存 | `backend/src/mgs-expert-cache.{h,cpp}` | 在每个 GPU 上用 LRU 池保存最近使用过的主机端专家，在设备与 CPU 之间拆分较小的微批次，从池中提供预取上传，并报告其状态。 | GGML 后端和调度器、mgs-model | mgs-context、mgs-graph、common、server |
| 混合 MoE 计算图 | `backend/src/mgs-graph.cpp` | 构建一次专家链；对于已缓存的层，则围绕 CPU 分区构建两次（设备池和主机权重），并将两个分支相加。 | mgs-expert-cache、GGML | 模型计算图 |
| 自动适配 | `backend/common/fit.cpp` | 将各层适配到设备内存；RAM 允许时把所有专家保留在系统内存中供缓存使用（有多个 GPU 时按内存大小把各层分布到各 GPU 上）；在扣除稠密权重后剩余的设备内存中，提高微批次（最多 4096，占用不超过八分之一）和缩减后的上下文（不超过四分之一），并计入 nextn 层。 | mgs-ext、/proc/meminfo | 通用初始化 |
| 上下文检查点 | `backend/tools/server/server-context.cpp; backend/common/chat.h` | 在消息边界和批次起点创建检查点；淘汰与相邻检查点距离最近的那一个。 | 消息跨度、mgs 状态 API | 服务器槽位 |
| 监控（服务器） | `backend/tools/server/server-monitor.{h,cpp}; backend/tools/server/server-context.cpp; backend/tools/server/server-models.cpp` | 读取 CPU、RAM 和 GPU 使用率（procfs、NVML、sysfs），发布每个推理循环的内存快照，汇总路由器各子进程的数据，并提供 `/api/monitor*`。 | mgs-ext 内存 API、libnvidia-ml（可选，dlopen） | HTTP、UI 监控页面和状态栏 |
| 内存统计 | `backend/src/mgs-model.cpp; backend/src/mgs-context.cpp; backend/src/mgs-expert-cache.cpp; backend/ggml/src/ggml-backend.cpp` | 按缓冲区类型将模型权重划分为 MTP 和路由专家；报告专家缓存缓冲区和预取槽。 | GGML 缓冲区和调度器 | 监控（服务器） |
| 监控（UI） | `frontend/src/lib/components/app/monitor; frontend/src/lib/stores/monitor.svelte.ts; frontend/src/lib/services/monitor.service.ts; frontend/src/lib/constants/monitor.ts; frontend/src/routes/monitor` | 监控页面（使用率卡片和带图例的堆叠内存条）以及带 CPU 和 GPU 指示条的固定状态栏；使用率指示条为绿色，并随使用率以红色填充。 | `/api/monitor`、`/api/monitor/usage`、设置 | 根布局、侧边栏 |


<a id="symbols"></a>
## 关键符号索引

| 符号 | 文件:行号 | 用途 |
| --- | --- | --- |
| `mgs_model_create` | [backend/src/mgs-model.cpp:157](../backend/src/mgs-model.cpp#L157) | 在构造模型之前，验证专家数量或显式的纯 MTP 头。 |
| `main` | [backend/tools/server/main.cpp:3](../backend/tools/server/main.cpp#L3) | `mgs` 的 CLI 入口点。 |
| `MGS_VERSION` | [backend/cmake/build-info.cmake:3](../backend/cmake/build-info.cmake#L3) | 项目版本；项目的每次修改都要将其补丁号加一。 |
| `mgs_version` | [backend/common/build-info.cpp.in:9](../backend/common/build-info.cpp.in#L9) | 为 `--version`、`/api/props`、`system_fingerprint`、shell MCP 和 `User-Agent` 返回项目版本。 |
| `mgs_server` | [backend/tools/server/server.cpp:89](../backend/tools/server/server.cpp#L89) | 初始化进程并注册路由。 |
| `server_http_context::mInit` | [backend/tools/server/server-http.cpp:234](../backend/tools/server/server-http.cpp#L234) | 配置 HTTPS 和 HTTP 重定向，验证 API 前缀并安装中间件。 |
| `server_http_context::mStart` | [backend/tools/server/server-http.cpp:654](../backend/tools/server/server-http.cpp#L654) | 绑定两个不同的端口并启动两个监听器。 |
| `fBindServerPort` | [backend/tools/server/server-http.cpp:88](../backend/tools/server/server-http.cpp#L88) | 仅在端口冲突时重试，同时保持所选的监听套接字处于绑定状态。 |
| `fCreateHttpsServer` | [backend/tools/server/server-tls.cpp:302](../backend/tools/server/server-tls.cpp#L302) | 加载 PEM 证书，或加载 `~/.mgs/certificates/` 中的自签名身份（首次运行时创建并保存）。 |
| `fPersistentServerIdentity` | [backend/tools/server/server-tls.cpp:277](../backend/tools/server/server-tls.cpp#L277) | 在 `~/.mgs/certificates/fullchain.pem` 和 `privkey.pem` 有效期间重复使用它们；否则生成一个自签名身份并保存（0700/0600）。 |
| `server_http_context::mGet` | [backend/tools/server/server-http.cpp:856](../backend/tools/server/server-http.cpp#L856) | 注册一个 GET 路由并记录其文档元数据。 |
| `vMiddlewareValidateApiKey` | [backend/tools/server/server-http.cpp:402](../backend/tools/server/server-http.cpp#L402) | 只有公开路径上的 GET/HEAD 请求才跳过 API 密钥检查；带非空 `reload` 的 `GET /models` 仍需要密钥。 |
| `fOriginIsSameHost` | [backend/tools/server/server-http.cpp:187](../backend/tools/server/server-http.cpp#L187) | 接受通过 `Host` 或 `X-Forwarded-Host` 指向本服务器的 `Origin`（忽略默认端口）；在 `localhost` CORS 模式下，预路由处理器会拒绝其他跨站写入。 |
| `fJsonNestingExceeds` | [backend/tools/server/server-http.cpp:34](../backend/tools/server/server-http.cpp#L34) | 将请求体作为 JSON 文本扫描而不解析，并报告超过限制的数组/对象嵌套深度。 |
| `fRejectDeepRequestBody` | [backend/tools/server/server-http.cpp:800](../backend/tools/server/server-http.cpp#L800) | 在处理器运行之前，对嵌套深度超过 `cHttpMaxRequestJsonDepth`（1024）的 POST 和 DELETE 请求体返回 400。 |
| `fRedactArgs` | [backend/tools/server/server-models.cpp:173](../backend/tools/server/server-models.cpp#L173) | 在列表和日志中将机密选项（`sSecretOptionEnvs`，包括每种写法以及 `--opt=value` 形式）的值替换为 `<redacted>`。 |
| `fRedactPreset` | [backend/tools/server/server-models.cpp:161](../backend/tools/server/server-models.cpp#L161) | 为 `status.preset` 复制一份预设，并将其中的机密选项替换为 `<redacted>`。 |
| `fProxyIsNavigation` | [backend/tools/server/server-cors-proxy.h:34](../backend/tools/server/server-cors-proxy.h#L34) | 拒绝浏览器会将其作为文档、frame、object 或 worker 渲染的代理请求（`Sec-Fetch-Mode`、`Sec-Fetch-Dest`）。 |
| `fProxyMakeInert` | [backend/tools/server/server-cors-proxy.h:44](../backend/tools/server/server-cors-proxy.h#L44) | 去除上游的活动头部，并在代理返回的响应上强制设置沙箱 CSP、`nosniff`、`attachment` 和同源 CORP。 |
| `fCommonRemoteGetContent` | [backend/common/download.cpp:603](../backend/common/download.cpp#L603) | 带连接时限和总时限下载，手动处理重定向；启用 `vPublicHostsOnly` 时，只访问解析后固定下来的公网目标。 |
| `fRemoteResolveHost` | [backend/common/download.cpp:503](../backend/common/download.cpp#L503) | 在剩余时限内对主机只解析一次，用于地址检查和地址固定。 |
| `fRemoteIpv4IsPublic` | [backend/common/download.cpp:438](../backend/common/download.cpp#L438) | 拒绝环回、私有、CGNAT、链路本地、文档、测试、组播、保留和广播 IPv4 地址。 |
| `fRemoteIpv6IsPublic` | [backend/common/download.cpp:469](../backend/common/download.cpp#L469) | 只接受 Teredo 和文档地址之外的 2000::/3；映射地址、兼容地址、NAT64 和 6to4 地址按其中的 IPv4 地址判断。 |
| `fHandleMedia` | [backend/tools/server/server-common.cpp:872](../backend/tools/server/server-common.cpp#L872) | 在 `media_download_slot` 下读取 data URI、`--media-path` 文件或远程 URL（除非使用 `--media-allow-private-urls`，否则仅限公网地址）。 |
| `fJsonExceedsDepth` | [backend/common/json-schema-to-grammar.cpp:305](../backend/common/json-schema-to-grammar.cpp#L305) | 以迭代方式测量 schema 的嵌套深度，使转换器能够在递归之前拒绝它（`cMaxSchemaDepth`）。 |
| `recv_msg` | [backend/ggml/src/ggml-rpc/ggml-rpc.cpp:275](../backend/ggml/src/ggml-rpc/ggml-rpc.cpp#L275) | 只有在声明的大小不超过其命令的上限时，才接收可变长度的 RPC 帧。 |
| `rpc_server::deserialize_tensor` | [backend/ggml/src/ggml-rpc/ggml-rpc.cpp:1141](../backend/ggml/src/ggml-rpc/ggml-rpc.cpp#L1141) | 重建对端的张量；当其数据范围溢出或超出其缓冲区时返回 `nullptr`。 |
| `socket_t::impl::rdma_recv` | [backend/ggml/src/ggml-rpc/transport.cpp:449](../backend/ggml/src/ggml-rpc/transport.cpp#L449) | 每个 RDMA 完成事件最多只复制所请求的字节数，其余部分保留在 `rx_pending` 中供下一次读取。 |
| `fRegisterApiDocumentation` | [backend/tools/server/server-api-doc.cpp:384](../backend/tools/server/server-api-doc.cpp#L384) | 提供 Swagger 查看器、捆绑的资源和实时生成的 OpenAPI。 |
| `server_shell_mcp::mStart` | [backend/tools/server/server-shell-mcp.cpp:1151](../backend/tools/server/server-shell-mcp.cpp#L1151) | 检测 bash、zsh、sh 和 pwsh/powershell；记录主目录、用户和主机名，供说明文本使用。 |
| `server_shell_mcp::mHandlePost` | [backend/tools/server/server-shell-mcp.cpp:1464](../backend/tools/server/server-shell-mcp.cpp#L1464) | 检查 Origin 和 Content-Type，解析一条 JSON-RPC 消息并响应 initialize、ping、tools/list 和 tools/call；对通知返回 202。 |
| `server_shell_mcp::mToolsCallResult` | [backend/tools/server/server-shell-mcp.cpp:1311](../backend/tools/server/server-shell-mcp.cpp#L1311) | 验证 command、cwd 和 timeout，运行命令，并以 `structuredContent` 和文本两种形式返回包含 exit_code、stdout 和 stderr 的 JSON 对象。 |
| `fRunCommand` | [backend/tools/server/server-shell-mcp.cpp:474](../backend/tools/server/server-shell-mcp.cpp#L474) | 在没有控制终端的新会话中执行 `posix_spawn`（`POSIX_SPAWN_SETSID`，同时也是新的进程组），对两个管道执行 `poll`，每次读取的内容也交给 `pOutput`（调用日志），超时或连接断开时杀死整个进程组。 |
| `server_shell_mcp_call_log` | [backend/tools/server/server-shell-mcp.cpp:340](../backend/tools/server/server-shell-mcp.cpp#L340) | 每次调用在 `~/.mgs/local-shell-mcp-server/` 中对应一个 JSONL 文件（`y2026m10d12h23m59s59ms731.jsonl`，模式 0600）：运行前写入 `command` 行，输出到达时以 `stdout`/`stderr` 行写入全部输出，最后写入 `result` 或 `error`。 |
| `fOriginAllowed` | [backend/tools/server/server-shell-mcp.cpp:140](../backend/tools/server/server-shell-mcp.cpp#L140) | 接受不带 Origin 的请求、来自本服务器源的请求，或从环回地址发往环回地址的请求。 |
| `cMcpStore.mSyncShellServer` | [frontend/src/lib/stores/mcp.svelte.ts:598](../frontend/src/lib/stores/mcp.svelte.ts#L598) | 根据 `/api/props` 添加、更新（URL 和名称）或移除 `mgs-shell` 服务器条目。 |
| `fIsLocalShellTool` | [frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts:25](../frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts#L25) | 对 `bash.run`、`zsh.run`、`sh.run` 和 `psh.run` 返回 true，除非该工具名属于另一个 MCP 服务器。 |
| `fParseLocalShellResult` | [frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts:44](../frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts#L44) | 读取 shell 调用的 JSON 对象（exit_code、stdout、stderr、timed_out、cancelled）；对其他任何结果返回 null。 |
| `ChatMessageToolCallBlockLocalShell` | [frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockLocalShell.svelte:1](../frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockLocalShell.svelte#L1) | 以工具名作为标题；显示带退出码的 stdin、stdout 和 stderr 框。 |
| `server_shell_mcp_askpass::mServe` | [backend/tools/server/server-shell-mcp.cpp:1002](../backend/tools/server/server-shell-mcp.cpp#L1002) | 检查 `SO_PEERCRED` 和 sudo 父进程，发布提示，等待 UI 作答（最多 5 分钟），然后回复密码或原因。 |
| `fParentIsSudo` | [backend/tools/server/server-shell-mcp.cpp:735](../backend/tools/server/server-shell-mcp.cpp#L735) | 接受这样的进程：其父进程名为 sudo，真实 uid 与我们相同，且有效 uid 或保存的 uid 为 root。 |
| `fServerShellMcpAskpassMain` | [backend/tools/server/server-shell-mcp.cpp:1094](../backend/tools/server/server-shell-mcp.cpp#L1094) | sudo 将 `mgs` 作为 SUDO_ASKPASS 运行时的辅助模式：发送提示，并将密码写入 sudo 的管道。 |
| `server_shell_mcp::mHandleAskpassGet` | [backend/tools/server/server-shell-mcp.cpp:1525](../backend/tools/server/server-shell-mcp.cpp#L1525) | 返回某个调用中待处理的提示（`?call=`）。 |
| `server_shell_mcp::mHandleAskpassPost` | [backend/tools/server/server-shell-mcp.cpp:1536](../backend/tools/server/server-shell-mcp.cpp#L1536) | 检查 Origin 和 Content-Type，并递交密码（一行，最多 4096 字节）或取消请求。 |
| `fUiSettingsMakeGetHandler` | [backend/tools/server/server-ui-settings.cpp:167](../backend/tools/server/server-ui-settings.cpp#L167) | `GET /api/ui/settings`：返回已保存的文档；首次保存之前返回 `{"settings": null}`。 |
| `fUiSettingsMakePostHandler` | [backend/tools/server/server-ui-settings.cpp:181](../backend/tools/server/server-ui-settings.cpp#L181) | `POST /api/ui/settings`：验证 `{"settings": {...}}`，并以原子方式替换 `~/.mgs/config/ui-settings.json`（0600）。 |
| `fUiSettingsLanguage` | [backend/tools/server/server-ui-settings.cpp:142](../backend/tools/server/server-ui-settings.cpp#L142) | 已保存的界面语言；仅当它是有效的语言标签时才由 `/api/config.js` 发布。 |
| `UiSettingsStore::mLoad` | [frontend/src/lib/stores/ui-settings.svelte.ts:82](../frontend/src/lib/stores/ui-settings.svelte.ts#L82) | 加载一次服务器的设置文档，并运行每个已注册的分区加载器；除非缺少 API 密钥，否则失败时会重试。 |
| `UiSettingsStore::mSet` | [frontend/src/lib/stores/ui-settings.svelte.ts:72](../frontend/src/lib/stores/ui-settings.svelte.ts#L72) | 修改一个分区，并在 300 ms 后保存整个文档；文档加载完成之前忽略修改。 |
| `UiSettingsStore::mFlush` | [frontend/src/lib/stores/ui-settings.svelte.ts:96](../frontend/src/lib/stores/ui-settings.svelte.ts#L96) | 立即保存待处理的修改并等待其完成（在因语言切换而重新加载之前使用）。 |
| `cMcpStore.mActiveShellCalls` | [frontend/src/lib/stores/mcp.svelte.ts:1233](../frontend/src/lib/stores/mcp.svelte.ts#L1233) | 本地 shell MCP 服务器上正在运行的调用（在 `_meta` 中发送的 ID、工具和命令），由 `mExecuteTool` 设置。 |
| `DialogLocalShellAskpass` | [frontend/src/lib/components/app/dialogs/DialogLocalShellAskpass.svelte:1](../frontend/src/lib/components/app/dialogs/DialogLocalShellAskpass.svelte#L1) | 根布局中的居中对话框：轮询正在运行的 shell 调用的 sudo 提示，并通过“接受”按钮或 Enter 键（`fHandlePasswordKeydown`）作答，或通过“取消”按钮或 Escape 键取消；提示一出现，密码字段就会获得焦点（`fFocusPassword`）。 |
| `LocalShellAskpassService` | [frontend/src/lib/services/local-shell-askpass.service.ts:16](../frontend/src/lib/services/local-shell-askpass.service.ts#L16) | 基于 `/api/mcp/shell/askpass` 的 `mPending`、`mAnswer` 和 `mCancel`。 |
| `MCPService.mCreateTransport` | [frontend/src/lib/services/mcp.service.ts:414](../frontend/src/lib/services/mcp.service.ts#L414) | 构建 MCP 传输层；为同源服务器和 CORS 代理加上 UI 的 API 密钥。 |
| `fCompletionProperties` | [backend/tools/server/server-api-doc.cpp:32](../backend/tools/server/server-api-doc.cpp#L32) | 从 server-schema 中公开补全参数及其说明。 |
| `server_model_meta::mUpdateArgs` | [backend/tools/server/server-models.cpp:241](../backend/tools/server/server-models.cpp#L241) | 强制子进程使用显式 HTTP、私有端口和 /api 前缀。 |
| `common_subproc` | [backend/common/subproc.h:21](../backend/common/subproc.h#L21) | 持有 `vProc`，并在启用和未启用 `MGS_SUBPROCESS` 时提供一致的方法；参数/环境变量向量以 null 结尾，关闭 stdin 时会在 join/析构之前分离供应商的 `stdin_file`。 |
| `fCommonParamsPrintInfo` | [backend/common/common.cpp:448](../backend/common/common.cpp#L448) | 记录构建和系统信息；两个预处理器分支都声明 `vBuildType`，定义了 `NDEBUG`（Release）时为空，否则为 ` (debug)`。 |
| `fCommonMaybeEnableEmbeddedMtp` | [backend/common/common.cpp:1539](../backend/common/common.cpp#L1539) | 在尊重显式选项的前提下应用自动推测解码。 |
| `fDetectEmbeddedMtp` | [backend/common/common.cpp:1422](../backend/common/common.cpp#L1422) | 检查所有 GGUF 分片，查找受支持的完整 MTP 头。 |
| `fCommonModelParamsToMgs` | [backend/common/common.cpp:1933](../backend/common/common.cpp#L1933) | 将通用参数转换为模型参数；只有在没有单独的 MTP GGUF 替代时，才加载嫁接的 MTP 层。 |
| `mgs_model_base::mLoadTensors` | [backend/src/mgs-model.cpp:1224](../backend/src/mgs-model.cpp#L1224) | 将各层分配到设备，并在放得下时把 MTP 层和输出层保留在 GPU 上。 |
| `mgs_model_loader::mCreateTensor` | [backend/src/mgs-model-loader.cpp:1183](../backend/src/mgs-model-loader.cpp#L1183) | 为每个权重选择缓冲区类型，应用覆盖设置，但会把 MTP 权重移到主机内存的覆盖设置除外。 |
| `mgs_mmap::mRegisterHost` | [backend/src/mgs-mmap.cpp:639](../backend/src/mgs-mmap.cpp#L639) | 锁定已映射的页面，并保留其注销回调。 |
| `fInitializeExpertPrefetch` | [backend/ggml/src/ggml-backend.cpp:1650](../backend/ggml/src/ggml-backend.cpp#L1650) | 分配/复用专家槽，并在内存紧张时回退。 |
| `fApiUrl` | [frontend/src/lib/api-url.ts:8](../frontend/src/lib/api-url.ts#L8) | 依据服务器发布的 API 前缀解析 UI 请求。 |
| `aLanguages` | [frontend/src/lib/i18n.ts:17](../frontend/src/lib/i18n.ts#L17) | 按字母顺序排列的受支持语言区域代码；由它派生出 `Language` 类型和选择器选项。 |
| `fApplyResponseLanguage` | [frontend/src/lib/utils/response-language.ts:6](../frontend/src/lib/utils/response-language.ts#L6) | 启用时，将本地化指令追加到第一条系统消息（字符串或内容片段）的副本中，必要时则在开头插入一条系统消息；不改动已存储的历史记录和调用方的对象。 |
| `fGetLanguage` | [frontend/src/lib/i18n.ts:70](../frontend/src/lib/i18n.ts#L70) | 返回通过 URL/存储选定的语言区域，否则返回 `en-US`；布局用它来设置 HTML 的语言和方向。 |
| `fTranslate` | [frontend/src/lib/i18n.ts:74](../frontend/src/lib/i18n.ts#L74) | 解析消息并插入具名的变量值。 |
| `fSetLanguage` | [frontend/src/lib/i18n.ts:81](../frontend/src/lib/i18n.ts#L81) | 将语言持久化到存储/URL 中，并重新加载 UI。 |
| `fMain (installer)` | [deploy/install-common.sh:171](../deploy/install-common.sh#L171) | 验证选项，安装依赖，暂存源码（包括 build/），刷新不匹配的 CMake 元数据，并从 source/build 构建。 |
| `fInstallNccl` | [deploy/install-common.sh:118](../deploy/install-common.sh#L118) | 根据检测到的 CUDA 版本选择 NCCL，并以完全相同的版本安装 libnccl2/libnccl-dev，在这条 APT 命令中允许降级。 |
| `fMain (Debian)` | [deploy/install-update-reinstall-debian.sh:168](../deploy/install-update-reinstall-debian.sh#L168) | 选择本地/远程模式，并协调 Debian 安装。 |
| `fDownloadSource (Debian)` | [deploy/install-update-reinstall-debian.sh:114](../deploy/install-update-reinstall-debian.sh#L114) | 验证并同步源码归档，同时保留缓存。 |
| `fLockSourceDirectory (Debian)` | [deploy/install-update-reinstall-debian.sh:91](../deploy/install-update-reinstall-debian.sh#L91) | 防止多个安装同时使用同一份源码检出。 |
| `fRunLocalInstaller (Debian)` | [deploy/install-update-reinstall-debian.sh:150](../deploy/install-update-reinstall-debian.sh#L150) | 在另一个 Bash 中运行辅助脚本，隔离函数/trap，并关闭 stdin。 |
| `fMain (web deployment)` | [deploy/configure-web.sh:40](../deploy/configure-web.sh#L40) | 创建凭据、证书、虚拟主机和服务文件。 |
| `fMain (local build)` | [build/build-local.sh:261](../build/build-local.sh#L261) | 选择本地或远程源码，创建一次性的 /tmp/ 工作区，从 source/build 编译，组装捆绑包，锁定输出目录，复制结果并进行验证。 |
| `fHasProjectSources` | [build/build-local.sh:96](../build/build-local.sh#L96) | 检查所需的源文件和许可证目录，用于选择本地源码和验证下载的归档。 |
| `fDownloadSources` | [build/build-local.sh:107](../build/build-local.sh#L107) | 通过 HTTPS 下载 main 分支，验证归档中的路径，在私有的 /tmp/ 工作区内解压源码，并检查所需的项目文件。 |
| `fCleanup (local build)` | [build/build-local.sh:16](../build/build-local.sh#L16) | 关闭日志写入进程并等待其结束，删除目标位置中不完整的条目和私有的 /tmp/ 工作区，并保留退出状态。 |
| `fBundleDependencies` | [build/build-local.sh:167](../build/build-local.sh#L167) | 将一个对象的 `ldd` 依赖闭包复制到 `lib/` 中，NVIDIA 驱动库除外。 |
| `fBundleNameServiceModules` | [build/build-local.sh:184](../build/build-local.sh#L184) | 捆绑 glibc 的 NSS 模块及其依赖，用于主机和用户查询。 |
| `fPinLibrarySearchPaths` | [build/build-local.sh:196](../build/build-local.sh#L196) | 将捆绑文件中的 `RUNPATH` 条目转换为 `DT_RPATH` `$ORIGIN`。 |
| `fReplaceDirectory` | [build/build-local.sh:211](../build/build-local.sh#L211) | 将新组装好的文件夹替换到位，失败时恢复原来的文件夹。 |
| `fVerifyBundle` | [build/build-local.sh:228](../build/build-local.sh#L228) | 以启动器运行它的方式，通过捆绑的加载器列出依赖，并拒绝在 `lib/` 之外解析到的库。 |
| `main (launcher)` | [build/mgs-launcher.c:27](../build/mgs-launcher.c#L27) | 根据 `bin/` 推导出自身的前缀，导出 `MGS_EXECUTABLE`，并执行 `lib/mgs/<loader> lib/mgs/mgs`。 |
| `fGetServerExecPath` | [backend/tools/server/server-models.cpp:76](../backend/tools/server/server-models.cpp#L76) | 为路由器子进程选择可执行文件，优先使用 `MGS_EXECUTABLE`。 |
| `fReadOccupiedPorts` | [deploy/start-web.sh:40](../deploy/start-web.sh#L40) | 读取 IPv4/IPv6 监听器，而不停止其他服务。 |
| `fFindFreePort` | [deploy/start-web.sh:52](../deploy/start-web.sh#L52) | 查找下一个空闲端口，排除已保留的 HTTPS 端口和绑定失败的端口。 |
| `fOwnsListener` | [deploy/start-web.sh:65](../deploy/start-web.sh#L65) | 验证监听器属于 Apache 子进程。 |
| `fPublishPorts` | [deploy/start-web.sh:72](../deploy/start-web.sh#L72) | 绑定之后以原子方式发布端口、URL 和 PID。 |
| `fMain` | [deploy/start-web.sh:82](../deploy/start-web.sh#L82) | 监管 Apache 并转发关闭信号。 |
| `ggml_backend_sched_compute_splits` | [backend/ggml/src/ggml-backend.cpp:1714](../backend/ggml/src/ggml-backend.cpp#L1714) | 同步各个 split，并分派选择性复制/预取的专家副本。 |
| `ggml_cuda_mul_mat_id_needs_sync` | [backend/ggml/src/ggml-cuda/ggml-cuda.cu:1870](../backend/ggml/src/ggml-cuda/ggml-cuda.cu#L1870) | 在异步专家内核上保持计算图捕获。 |
| `ggml_cuda_fattn_vec_instances` | [backend/ggml/cmake/common.cmake:53](../backend/ggml/cmake/common.cmake#L53) | 为 CUDA、HIP 和 MUSA 选择要编译的 K/V 量化组合。 |
| `mgs_context::mGetGfResPrev` | [backend/src/mgs-context.cpp:2365](../backend/src/mgs-context.cpp#L2365) | 根据是否有输出选择 arena，并延迟创建它。 |
| `mgs_context::mOutputReorder` | [backend/src/mgs-context.cpp:2257](../backend/src/mgs-context.cpp#L2257) | 为未屏蔽的 NextN 行和层输入行恢复原始的 token 顺序。 |
| `mgs_model_glm4_moe::graph_mtp` | [backend/src/models/glm4-moe.cpp:140](../backend/src/models/glm4-moe.cpp#L140) | 构建单头的 GLM MTP 解码器。 |
| `fCommonSamplerSampleAndAcceptN` | [backend/common/sampling.cpp:655](../backend/common/sampling.cpp#L655) | 在第一个非末尾的 EOG 处结束推测接受。 |
| `server_routes::mHandleCountTokens` | [backend/tools/server/server-context.cpp:5730](../backend/tools/server/server-context.cpp#L5730) | 只有在响应唤醒服务器之后才使用模型指针。 |
| `ConversationsStore::mGetConversationsForExport` | [frontend/src/lib/stores/conversations.svelte.ts:542](../frontend/src/lib/stores/conversations.svelte.ts#L542) | 按所选顺序读取持久化的元数据和完整的消息树。 |
| `ConversationsStore::mParseImportFile` | [frontend/src/lib/stores/conversations.svelte.ts:1061](../frontend/src/lib/stores/conversations.svelte.ts#L1061) | 先检查文件大小，然后在解压之前用 `unzipSync` 过滤 ZIP 条目（只接受 `.jsonl`，并检查条目数和声明的大小）。 |
| `ConversationsStore::mParseSessionsJsonl` | [frontend/src/lib/stores/conversations.svelte.ts:986](../frontend/src/lib/stores/conversations.svelte.ts#L986) | 验证每个导入的对话和消息，并要求每个 `convId` 与其所属对话一致。 |
| `DatabaseService.mImportConversations` | [frontend/src/lib/services/database.service.ts:574](../frontend/src/lib/services/database.service.ts#L574) | 在一个事务中用 `bulkAdd` 添加每个对话及其消息；若某条消息的 ID 已存在，则整体跳过该对话。 |
| `fCreateImageErrorFallback` | [frontend/src/lib/utils/image-error-fallback.ts:10](../frontend/src/lib/utils/image-error-fallback.ts#L10) | 用 DOM 节点和 `textContent` 构建图像损坏提示，只为 http(s) URL 生成链接。 |
| `fIsSafeLinkUrl` | [frontend/src/lib/utils/safe-url.ts:50](../frontend/src/lib/utils/safe-url.ts#L50) | 允许相对链接以及 http、https、mailto 和 tel 协议；`fIsDataImageUrl` 只接受 `data:image/...;base64`。 |
| `ToolsStore::mResolveEnabledEntry` | [frontend/src/lib/stores/tools.svelte.ts:320](../frontend/src/lib/stores/tools.svelte.ts#L320) | 为一个工具名解析出唯一一个已启用的条目，用于公开、审批和执行；同时由两个服务器提供的 MCP 名称解析不到任何条目（`mConflictingMcpToolNames`）。 |
| `MCPStore::mExecuteTool` | [frontend/src/lib/stores/mcp.svelte.ts:1175](../frontend/src/lib/stores/mcp.svelte.ts#L1175) | 在确认该服务器确实声明了此工具之后，在获得批准的那个确切服务器 ID 上运行该工具。 |
| `mgs_expert_cache::mFCreate` | [backend/src/mgs-expert-cache.cpp:181](../backend/src/mgs-expert-cache.cpp#L181) | 选择专家位于主机内存的 GPU 层，确定每个 GPU 上池的大小并分配一个池，测量每个 GPU 的上传带宽。 |
| `mgs_expert_cache::mPartition` | [backend/src/mgs-expert-cache.cpp:838](../backend/src/mgs-expert-cache.cpp#L838) | 将一个微批次拆分为命中、上传和 CPU 专家；将标识符和上传任务加入队列。 |
| `mgs_expert_cache::mUploadWeights` | [backend/src/mgs-expert-cache.cpp:701](../backend/src/mgs-expert-cache.cpp#L701) | 填充一个预取槽：已缓存的专家从设备复制到设备，其余专家从主机内存上传。 |
| `mgs_expert_cache::mFillInfo` | [backend/src/mgs-expert-cache.cpp:972](../backend/src/mgs-expert-cache.cpp#L972) | 报告大小、内容和计数器。 |
| `llm_graph_context::mBuildMoeFfn` | [backend/src/mgs-graph.cpp:1500](../backend/src/mgs-graph.cpp#L1500) | 构建常规的专家链，或混合的设备分支与 CPU 分支。 |
| `ggml_backend_sched_set_weight_upload` | [backend/ggml/src/ggml-backend.cpp:2197](../backend/ggml/src/ggml-backend.cpp#L2197) | 安装专家预取所用的上传回调。 |
| `mgs_set_expert_cache_size` | [backend/src/mgs-context.cpp:4065](../backend/src/mgs-context.cpp#L4065) | 请求新的缓存大小，在下一次解码时生效。 |
| `mgs_get_expert_cache_info` | [backend/src/mgs-context.cpp:4063](../backend/src/mgs-context.cpp#L4063) | 返回缓存状态，供监控使用。 |
| `fCommonMoeCacheSize` | [backend/common/common.cpp:1067](../backend/common/common.cpp#L1067) | 根据选项和环境变量解析所配置的缓存大小。 |
| `fHostAvailableMemory` | [backend/common/fit.cpp:185](../backend/common/fit.cpp#L185) | 读取可供保留在 RAM 中的专家使用的系统内存。 |
| `fRaiseContext`（`fCommonParamsFitImpl` 中的 lambda） | [backend/common/fit.cpp:619](../backend/common/fit.cpp#L619) | 提高被 `--fit` 缩减的上下文，直到 KV 缓存最多占用扣除稠密权重后剩余设备内存的四分之一。 |
| `fRaiseBatch`（`fCommonParamsFitImpl` 中的 lambda） | [backend/common/fit.cpp:678](../backend/common/fit.cpp#L678) | 当专家保留在系统内存中且未指定 `-b`/`-ub` 时，提高微批次和逻辑批次（4096、2048 或 1024）。 |
| `fCommonDefaultThreads` | [backend/common/common.cpp:252](../backend/common/common.cpp#L252) | 默认 CPU 线程数：可用于数学运算的核心数，最多为亲和性掩码中逻辑 CPU 数量的一半。 |
| `server_routes::vPostExpertCache` | [backend/tools/server/server-context.cpp:5464](../backend/tools/server/server-context.cpp#L5464) | 验证 `pSizeMib`，并将缓存大小调整加入队列。 |
| `fExpertCacheJson` | [backend/tools/server/server-context.cpp:872](../backend/tools/server/server-context.cpp#L872) | 将缓存状态转换为 `/expert-cache` 和 `/metrics` 所用的格式。 |
| `mCreateCheckpoint` | [backend/tools/server/server-context.cpp:2575](../backend/tools/server/server-context.cpp#L2575) | 存储一个检查点，并淘汰与相邻检查点距离最近的那一个。 |
| `common_chat_msg_spans::mIsTurnStart` | [backend/common/chat.h:178](../backend/common/chat.h#L178) | 检测用户、助手和工具消息的起始位置。 |
| `mgs_model_get_weight_breakdown` | [backend/src/mgs-model.cpp:2755](../backend/src/mgs-model.cpp#L2755) | 按缓冲区类型将模型已加载的权重划分为 MTP（nextn）层和路由专家（`*exps.*`）。 |
| `mgs_get_expert_memory_breakdown` | [backend/src/mgs-context.cpp:4086](../backend/src/mgs-context.cpp#L4086) | 按缓冲区类型统计专家缓存缓冲区和预取槽占用的内存。 |
| `ggml_backend_sched_get_prefetch_size` | [backend/ggml/src/ggml-backend.cpp:2204](../backend/ggml/src/ggml-backend.cpp#L2204) | 调度器中专家预取槽的字节数和缓冲区类型。 |
| `mgs_expert_cache::mAddMemory` | [backend/src/mgs-expert-cache.cpp:995](../backend/src/mgs-expert-cache.cpp#L995) | 按缓冲区类型累加缓存的池、槽标识符和暂存缓冲区。 |
| `server_context_impl::mGetMonitorSnapshot` | [backend/tools/server/server-context.cpp:920](../backend/tools/server/server-context.cpp#L920) | 为 HTTP 线程复制已发布的内存快照，而不触及上下文或任务队列。 |
| `server_context_impl::mBuildMonitorSnapshot` | [backend/tools/server/server-context.cpp:1002](../backend/tools/server/server-context.cpp#L1002) | 构建目标上下文和草稿上下文在每个 GPU 上及 RAM 中的内存段，外加提示词缓存和检查点。 |
| `server_context_impl::mRefreshMonitorSnapshot` | [backend/tools/server/server-context.cpp:1132](../backend/tools/server/server-context.cpp#L1132) | 加载完成后发布快照；槽位工作期间最多每秒发布一次，空闲时则每次都发布。 |
| `fMonitorReport` | [backend/tools/server/server-monitor.cpp:698](../backend/tools/server/server-monitor.cpp#L698) | 将 CPU、RAM 和 GPU 读数与快照合并：每个 GPU（按 PCI ID）和 RAM 中的内存段、“other”以及空闲内存。 |
| `fMonitorUsageJson` | [backend/tools/server/server-monitor.cpp:641](../backend/tools/server/server-monitor.cpp#L641) | 供状态栏使用的 CPU 使用率（全局及每个物理处理器）和 GPU 使用率。 |
| `fCpuJson` | [backend/tools/server/server-monitor.cpp:183](../backend/tools/server/server-monitor.cpp#L183) | 根据 `/proc/stat` 的增量计算 CPU 使用率，包括全局和每个物理处理器的使用率（`processors`：型号、使用率以及带编号的逻辑 CPU），结果共享 750 ms；超过 10 s 的基准会被替换为 250 ms 的采样间隔。 |
| `fNvml` | [backend/tools/server/server-monitor.cpp:444](../backend/tools/server/server-monitor.cpp#L444) | 用 `dlopen` 打开一次 `libnvidia-ml.so.1`，并解析监控所用的 NVML 调用。 |
| `fCollectMonitorSnapshots` | [backend/tools/server/server-models.cpp:1593](../backend/tools/server/server-models.cpp#L1593) | 向每个已就绪的路由器子进程请求 `/api/monitor/memory`；正在加载和处于休眠状态的子进程只报告其状态。 |
| `MonitorStore::mSubscribeReport` | [frontend/src/lib/stores/monitor.svelte.ts:39](../frontend/src/lib/stores/monitor.svelte.ts#L39) | 带引用计数地每 2 s 轮询一次 `/monitor`，标签页隐藏时暂停；同时为状态栏提供数据。 |
| `MonitorStore::mSubscribeUsage` | [frontend/src/lib/stores/monitor.svelte.ts:24](../frontend/src/lib/stores/monitor.svelte.ts#L24) | 在没有轮询报告时，带引用计数地每 2 s 轮询一次 `/monitor/usage`。 |
| `aMonitorSegmentGroups` | [frontend/src/lib/constants/monitor.ts:23](../frontend/src/lib/constants/monitor.ts#L23) | 内存条各段的固定顺序、标签和颜色变量。 |
| `MonitorScreen` | [frontend/src/lib/components/app/monitor/MonitorScreen.svelte:1](../frontend/src/lib/components/app/monitor/MonitorScreen.svelte#L1) | 监控页面：使用率卡片（先是 GPU，标为 `gpu-N`；然后每个物理处理器一张卡片，标为 `cpu-N`；每个标签以粗体显示在设备名称之前）和内存条，其图例按内存条中的顺序每行显示一个段（`MonitorMemoryBar`）。 |
| `StatusBar` | [frontend/src/lib/components/app/monitor/StatusBar.svelte:1](../frontend/src/lib/components/app/monitor/StatusBar.svelte#L1) | 固定在底部的状态栏，带有大小相同的指示条：先每个 GPU 一个（`gpu-1`、`gpu-2`……），再每个物理处理器一个（`cpu-1`、`cpu-2`……），标签与监控页面的卡片相同；放不下时可横向滚动；点击可打开监控页面。 |
| `fMonitorDrmResidency` | [backend/tools/server/server-monitor.cpp:347](../backend/tools/server/server-monitor.cpp#L347) | 从 `/proc/self/fdinfo` 中读取内核 DRM 驱动按 PCI ID 将进程的 GPU 内存放在何处（VRAM、GTT、被逐出的 VRAM）。 |
| `server_context_impl::mCheckVramResidency` | [backend/tools/server/server-context.cpp:1155](../backend/tools/server/server-context.cpp#L1155) | 当模型在 GPU 上分配的内存有一部分位于 VRAM 之外时发出一次警告，在其回到 VRAM 时再发出一次。 |
| `fPoolBufferType` | [backend/src/mgs-expert-cache.cpp:135](../backend/src/mgs-expert-cache.cpp#L135) | 当后端导出了逐出优先级较低的缓冲区类型时，将其用于专家缓存池。 |
| `ggml_backend_vk_device_get_cache_buffer_type` | [backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp:20805](../backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L20805) | `VK_EXT_memory_priority` 为 0.5 的 Vulkan 设备内存，以 `ggml_backend_dev_get_cache_buffer_type` 的形式导出。 |
| `ggml_backend_vk_buffer_is_device_of` | [backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp:18281](../backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L18281) | 在异步复制中接受后端 GPU 的任何设备缓冲区类型（默认类型和缓存类型）。 |
| `server_context_impl::mDeferTask` | [backend/tools/server/server-context.cpp:2685](../backend/tools/server/server-context.cpp#L2685) | 将找不到空闲槽位的任务加入队列，并发送它的第一条排队通知（按来源统计前面的忙碌槽位和等待中的任务）。 |
| `server_context_impl::mNotifyQueuePositions` | [backend/tools/server/server-context.cpp:2672](../backend/tools/server/server-context.cpp#L2672) | 每当有任务开始或被重新排队后，向每个请求了通知的等待任务发送其当前位置。 |
| `server_queue::mDeferredTasks` | [backend/tools/server/server-queue.h:93](../backend/tools/server/server-queue.h#L93) | 被推迟任务的 ID、索引、来源和通知标志，按到达顺序排列。 |
| `server_task_result_queue` | [backend/tools/server/server-task.h:476](../backend/tools/server/server-task.h#L476) | 序列化为 `{"mgs_queue": {"ahead", "ahead_api", "ahead_webui"}}` 的排队通知；它永远不会结束流。 |
| `ChatService.mHandleStreamResponse` | [frontend/src/lib/services/chat.service.ts:754](../frontend/src/lib/services/chat.service.ts#L754) | 解析补全的 SSE 流；将 `mgs_queue` 事件交给 `onQueue`，并在收到第一个常规数据块时清除它们。 |
| `ChatStore::mGetQueueStatus` | [frontend/src/lib/stores/chat.svelte.ts:585](../frontend/src/lib/stores/chat.svelte.ts#L585) | 对话的请求等待槽位期间的排队状态，否则为 null。 |
| `fQueueStatusText` | [frontend/src/lib/utils/progress.ts:15](../frontend/src/lib/utils/progress.ts#L15) | 聊天中的排队通知：带前面请求数量的 API 版本或 Web UI 版本；请求无需等待时为 null。 |
| `mgs_model_deepseek41` | [backend/src/models/models.h:258](../backend/src/models/models.h#L258) | DeepSeek-V4.1-Flash 模型：两种 GGUF 布局、外部 Engram 表、共享的压缩 KV 和候选池。 |
| `fMgsDeepseek41AdaptMetadata` | [backend/src/models/deepseek41.cpp:140](../backend/src/models/deepseek41.cpp#L140) | 在创建模型之前，将 `deepseek41` GGUF 中的 DwarfStar 键转换为标准键。 |
| `mgs_model_deepseek41::mEngramReadRows` | [backend/src/models/deepseek41.cpp:380](../backend/src/models/deepseek41.cpp#L380) | 用 `pread` 读取 Engram 行，并解码 E4M3+E8M0 或将其反量化为 F32。 |
| `mgs_kv_cache_dsv41` | [backend/src/mgs-kv-cache-dsv41.h:87](../backend/src/mgs-kv-cache-dsv41.h#L87) | DeepSeek-V4.1 缓存：原始 SWA 缓存，外加每种压缩比一个压缩组，由读取层共享。 |
| `mgs_model_loader::cTensorExternal` | [backend/src/mgs-model-loader.h:73](../backend/src/mgs-model-loader.h#L73) | 张量标志：计入统计但从不分配；预取在遇到第一个此类张量之前停止。 |
| `mgs_model_glm5_next` | [backend/src/models/models.h:344](../backend/src/models/models.h#L344) | GLM-5.3-Flash 模型：KDA + 仅 nope 的 MLA，带 k-pool DSA 索引器、超连接和 MTP。 |
| `mgs_model_qwen4exp` | [backend/src/models/models.h:749](../backend/src/models/models.h#L749) | Qwen3.8-Flash-Next 模型：低秩超连接、n-gram 逐层嵌入、门控 delta-net、QSA 和 MTP。 |
| `mgs_memory_hybrid_idx` | [backend/src/mgs-memory-hybrid-idx.h:20](../backend/src/mgs-memory-hybrid-idx.h#L20) | qwen4exp 和 glm5-next 的 KV 缓存、循环状态缓存和 k-pool 索引器缓存。 |
| `mgs_model_kimi_k3` | [backend/src/models/models.h:855](../backend/src/models/models.h#L855) | Kimi-K3 模型：KDA + MLA 混合架构、跨层残差混合以及拥有 896 个专家的潜空间 MoE。 |
| `fCommonChatParamsInitKimiK3` | [backend/common/chat.cpp:1838](../backend/common/chat.cpp#L1838) | Kimi-K3 聊天格式（`<|open|>`/`<|sep|>`/`<|close|>`）：思考、回复和工具部分。 |
| `DEEPSEEK41_TEMPLATE_SRC` | [backend/common/chat.cpp:642](../backend/common/chat.cpp#L642) | 当 `deepseek41` GGUF 不带模板时使用的内置 Jinja 模板。 |
| `ggml_swiglu_clamp` | [backend/ggml/include/ggml.h:1346](../backend/ggml/include/ggml.h#L1346) | 对输入进行截断的 SwiGLU（CPU、带融合矩阵-向量路径的 CUDA、Vulkan）。 |
| `ggml_dsv4_hc_pre_gated` | [backend/ggml/include/ggml.h:2572](../backend/ggml/include/ggml.h#L2572) | 带逐元素 sigmoid 门控的超连接预混合（qwen4exp；CPU 和 CUDA）。 |


<a id="flows"></a>
## 主要调用路径

1. **安装：** Debian 文件/stdin 入口 → 验证 → 日志和锁 → 本地检出或 GitHub `main` 归档 → 源码验证/同步 → 关闭了 stdin 的子 Bash → `install-common.sh:fMain` → 依赖 → 在稳定的 `_/temp/source` 中暂存（包括 `build/`） → 使用 `-S _/temp/source/build` 运行 CMake → 安装 `bin/mgs`。Alpine 直接从本地检出调用辅助脚本。
2. **部署：** `configure-web.sh` → 服务账户 → API 密钥/证书 → 独立模板和服务 → `start-web.sh` → 已占用的监听器 → 选择 11080+/11443+ → Apache 绑定 → 确认两个监听器的归属 → `ports.json`。真正的绑定冲突会回到端口选择；其他故障则会停止服务。
3. **聊天：** 组件 → `ChatService.mSendMessage` → 规范化的消息和自定义 JSON → `fApplyResponseLanguage`（使用当前的 `respondInInterfaceLanguage` 标志和 UI 语言区域） → `fApiUrl` → HTTP 注册 → 身份验证/就绪检查 → `server_routes` → 任务队列/槽位 → mgs/GGML 求值 → JSON 或 SSE → 浏览器存储和渲染。`mGenerateTitle` 和每个智能体轮次都使用同一条路径；`mPreEncode` 在准备 KV 缓存之前应用相同的系统指令。
4. **路由器：** 外部处理器 → 规范的内部路径 → 模型选择 → 子进程 `/api` 请求 → 响应或保留的流。查找、恢复和取消遵循相同的约定。
5. **MTP：** 通用初始化 → `fCommonMaybeEnableEmbeddedMtp` → `fDetectEmbeddedMtp` → 所有 GGUF 分片的头部/张量 → 架构/主干/头部检查 → 除非被显式覆盖，否则选择 `draft-mtp` → 模型加载：`mgs_model_base::mLoadTensors` 将 MTP 层和输出层放到 GPU 上 → `mgs_model_loader::mCreateTensor` 对它们跳过指向主机内存的覆盖设置。
6. **MoE：** 映射的权重 → 主机注册 → 计算图拆分 → `fInitializeExpertPrefetch` → 通过独立后端传输 → 就绪/空闲事件 → 计算。内存紧张时会减少槽数量或恢复常规复制。
7. **语言：** URL `lang`（优先）/存储 → 15 个语言目录之一（默认 `en-US`） → `fTranslate` → 经 Svelte 转义的文本；根布局 → HTML `lang`/`dir` → 按逻辑方向排列的导航和双向聊天文本；选择器 → 持久化 → 重新加载。
8. **更新后的推理：** 微批次的原始索引 → 相互独立的有输出/无输出 arena → CPU/CUDA/HIP/Vulkan/SYCL 内核 → 恢复 NextN 行顺序 → 一直接受到 EOG → 已完成槽位的推测解码指标。
9. **导出/工具：** 所选 ID → 读取持久化的对话/消息 → JSONL/ZIP；工具 HTTP 403 → 保留禁用状态 → 在面板中显式重试。
10. **本地构建：** 从文件或 stdin 运行 `build/build-local.sh` → 选项 → 本地/远程选择 → 在 `/tmp/moe-gguf-server-build.XXXXXXXX/` 中创建私有工作区和日志 → 检查工具链、CUDA 和 Node.js → 本地复制，或通过 HTTPS 下载归档、验证并解压到 `source/`（包括 `build/`） → 在 `cmake/` 中使用 `-S source/build` 运行 CMake → `mgs` → 静态启动器 → 加载器、`ldd` 依赖闭包和 NSS 模块放入 `bundle/lib/mgs/` → 服务器二进制文件和 `DT_RPATH` `$ORIGIN` → `bundle/lib/mgs/` 中的标记文件 → `build/licenses/` 和根目录的 `LICENSE` 放入 `bundle/licenses/` → 前缀锁 → 将完成的产物复制到各自的目标位置旁边 → 以原子方式替换 `lib/mgs/`、`bin/mgs` 和 `share/doc/mgs/licenses/` → 用 `<loader> --list` 验证 → 通过启动器运行 `--version` → `fCleanup` 删除私有工作区。运行时：启动器 → `/proc/self/exe` → `MGS_EXECUTABLE` → 根据 `bin/` 得到前缀 → `execv(lib/mgs/<loader>, lib/mgs/mgs, arguments)` → 路由器子进程通过启动器重新进入。


11. **原生 Web 启动：** `common_params` / CLI → `server_http_context::mInit` → `fCreateHttpsServer`（PEM，或 `fPersistentServerIdentity`：加载或创建 `~/.mgs/certificates/`） → `start` → 为 HTTPS 调用 `fBindServerPort` → 另一个不同的 HTTP 端口 → 两个监听线程 → 在日志中输出实际 URL → 停止并 join 两个线程。
12. **MoE 专家缓存：** 加载后的第一个请求 → `mgs_context::mDecode` → `mgs_expert_cache::mFCreate` → 符合条件的层和组 → 每个 GPU 的预算 → 每个 GPU 上的池和标识符缓冲区 → 逐个 GPU 探测带宽 → 使用混合计算图和上传回调执行 `mSchedReserve`。少于 32 个 token 的微批次：在设备上运行路由器 → 在 CPU 上运行 `mPartition`（复制选择结果、LRU 命中、均衡上传、标识符） → 基于池的设备分支 → 同时运行基于主机权重的 CPU 分支 → `mMeasureCpu` → 在设备上合并 → 路由器权重。较大的微批次：预取 → `mUploadWeights`（已缓存的专家在设备间复制，其余专家从主机上传） → 计算。`POST /api/expert-cache` → 任务队列 → `mgs_set_expert_cache_size` → 下一次解码时重建缓存和计算图。
13. **上下文检查点：** 填充提示词批次 → 在至少 3/4 满的微批次中出现消息起点，或遇到最后一条用户消息 → 切分批次 → 在批次起点创建检查点（消息起点、提示词末尾，或距上一个检查点已达 `--checkpoint-min-step`） → `mCreateCheckpoint`（淘汰与相邻检查点距离最近的那个） → 之后在位置 p 处出现分歧的请求会恢复 p 之前最新的检查点 → 只处理其后的部分。
14. **自动配置（适配、批次、上下文、线程）：** 参数解析 → `fPostprocessCpuParams` → 未指定 `-t` 时调用 `fCommonDefaultThreads` → `mLoadModel` 将测得的草稿/MTP 上下文计入设备余量 → `common_init_result` → `fCommonFitParams`（`pAutoBatch` = 未指定 `-b`/`-ub`） → `fCommonParamsFitImpl`：用初始参数测量 → 模型放不下时，第 2 步将未设置的上下文缩减到 `--fit-ctx` → 从后往前处理纯稠密层 → 所有专家放在 RAM 中（专家缓存）或整个专家层 → `fRaiseBatch` → `fRaiseContext`（两者都在整个专家层之前） → 将 `n_batch` 写回参数 → 线程日志 → 加载模型和目标上下文 → 使用目标的 `n_ctx` 和 `n_batch` 创建草稿上下文 → 专家缓存在第一个请求时占用剩余的设备内存。
15. **监控：** `mLoadModel` → `mgs_model_get_weight_breakdown`（一次） → `mRefreshMonitorSnapshot`（加载后；槽位工作时最多每秒 1 次，空闲时每次都发布） → `mBuildMonitorSnapshot`（目标上下文和草稿上下文的 `mgs_get_memory_breakdown` 与 `mgs_get_expert_memory_breakdown`、提示词缓存和检查点、每个 GPU 的 `ggml_backend_dev_get_props`） → 互斥锁。`GET /api/monitor` → `vGetMonitor`（路由器：`vGetRouterMonitor` → `fCollectMonitorSnapshots` → 子进程的 `/api/monitor/memory`） → `fMonitorReport`（`fCpuJson`、`fRamJson`、`fGpusJson` → NVML/sysfs，每个 GPU（按 PCI ID）和 RAM 中的内存段、“other”、空闲） → `cMonitorStore` → `MonitorScreen`。状态栏：`StatusBar` → `cMonitorStore.mSubscribeUsage` → `GET /api/monitor/usage` → 每 2 s 调用一次 `fMonitorUsageJson`。
16. **Vulkan 构建：** `--vulkan` → `vBuildBackend=vulkan` → （安装程序：`libvulkan-dev spirv-headers glslc mesa-vulkan-drivers`） → 检查 `glslc` → 使用 `-DGGML_VULKAN=ON -DGGML_CUDA=OFF` 运行 CMake（失败时会列出所需的软件包） → 着色器和 `mgs` → 本地构建：`libvulkan.so.1` 放入 `lib/mgs/` → 运行时：加载器 → 来自 `/usr/share/vulkan/icd.d/` 的 ICD → `Vulkan0` 设备 → `--fit`、专家缓存和预取的工作方式与 CUDA 相同。
17. **本地 shell MCP 服务器：** `--local-shell-mcp-server` → `vShellMcpServer`（默认 localhost CORS，不传给路由器子进程） → `mgs_server` → `server_shell_mcp::mStart`（检测 shell） → 注册 `POST/GET/DELETE /mcp/shell` 并发出警告 → `/api/props` 中的 `shell_mcp_server` → 根布局 → `cMcpStore.mSyncShellServer` → `mgs-shell` 条目 → 健康检查和 `MCPService.connect`（StreamableHTTP，同源 API 密钥） → `initialize`/`tools/list` → 模型调用工具 → 权限提示 → `tools/call` → `mHandlePost` → `fOriginAllowed` → `mToolsCallResult` → `server_shell_mcp_call_log::mOpen` 和 `command` 行 → `fRunCommand`（`posix_spawn`、`poll`、杀死进程组；每次读取 → `pOutput` → `mOutput` → `stdout`/`stderr` 行） → `mFlushOutput` → `result` 或 `error` 行 → 以文本和 `structuredContent` 两种形式返回的 JSON 对象（`vExitCode`、`stdout`、`stderr`……） → 工具消息 → `ChatMessageToolCallBlock` → `fIsLocalShellTool` → `ChatMessageToolCallBlockLocalShell`（`fParseLocalShellCommand`、`fParseLocalShellResult`） → stdin、stdout 和 stderr 框，启用 `expandLocalShellCommands` 时默认展开。
18. **sudo 密码提示：** `cMcpStore.mExecuteTool`（`_meta` 中的 `mgs/callId`、`dShellCalls`） → `mToolsCallResult` → `server_shell_mcp_askpass::mRegister` 和 `mEnvironment`（`SUDO_ASKPASS`、`MGS_LOCAL_SHELL_ASKPASS`、`DISPLAY`） → `fRunCommand` → 没有终端的 sudo → `mgs <prompt>` → `mgs_server` → `fServerShellMcpAskpassMain` → 抽象套接字 → `mServe`（`SO_PEERCRED`、`fParentIsSudo`、令牌） → 待处理的提示和暂停的时限（`mPaused`） → `DialogLocalShellAskpass`（`mActiveShellCalls`，每 500 ms 调用一次 `LocalShellAskpassService.mPending`） → 居中的密码对话框 → `mAnswer`/`mCancel` → `mHandleAskpassPost` → `server_shell_mcp_askpass::mAnswer` → 回复辅助程序 → 将密码写入 sudo 读取的管道（或在 stderr 上输出 `mgs: ...`） → 命令结束时调用 `mUnregister`。
19. **请求队列：** `ChatService.mSendMessage`（`mgs_client: "webui"`、`stream`）或 API 客户端 → `server_routes` → `fEvalMgsCmplSchema`（`vFromWebui`） → `vQueueNotice` → 任务队列 → `mProcessSingleTask` → `mGetAvailableSlot` → 没有空闲槽位 → `mDeferTask`（`mCountRunning`、`server_queue::mDeferredTasks`） → `server_queue::mDefer` → `server_task_result_queue` → 第一个流结果 → `data: {"mgs_queue": ...}` → `mHandleStreamResponse` → `onQueue` → `ChatStore::mSetQueueStatus` → `fQueueStatusText` → `ChatMessageAssistantProcessingInfo`。槽位释放 → `vCallbackOnRelease` → `mPopDeferredTask`（将第一个被推迟的任务移到最前面） → `mProcessSingleTask` → `mLaunchSlotWithTask` → 数量为 0 的通知 → `mNotifyQueuePositions` → 第一个 token → `onQueue(null)`。
20. **DeepSeek-V4.1 加载和 Engram：** `mgs_model_loader` → `fMgsDeepseek41AdaptMetadata`（DwarfStar 键 → 标准键） → `mgs_model_deepseek41` 的超参数和张量（Engram 表带有 `cTensorExternal` 标志，预取在它们之前停止） → 层角色 → `mgs_kv_cache_dsv41` → 每个 ubatch 调用 `llm_graph_input_dsv41_engram::mSetInput`（对 token 历史计算 n-gram 哈希 → 通过 `pread` 执行 `mEngramReadRows` → F32 输入） → 在第 1 层和第 14 层之前执行 `mBuildEngram` → 在共享压缩 KV 上计算注意力，使用索引键和候选池 → 使用 `ggml_swiglu_clamp` 的 MoE。
21. **远程媒体和解码：** 请求部分（`image_url`、`input_audio`、`input_video`） → `fHandleMedia` → data URI、`--media-path` 文件或 `media_download_slot`（同时最多 8 个） → `fCommonRemoteGetContent`（除非使用 `--media-allow-private-urls`，否则启用 `vPublicHostsOnly`） → `fRemoteResolveHost` → 对每个地址调用 `fRemoteIpv4IsPublic`/`fRemoteIpv6IsPublic` → `set_hostname_addr_map` → 在 30 s 和 10 MiB 的限制内执行 GET → 每次重定向都重新检查（最多 5 次） → `fProcessMtmdPrompt`（每个请求解码总量上限 2 GiB） → 解码前调用 `stbi_info_from_memory`，分配前检查音频时长，检查视频帧尺寸和预算 → mtmd 分块。被拒绝的地址会返回 400，并指出相关的选项。

<a id="routes"></a>
## 入口点与路由

| 路由/URL/命令 | 处理器 | 文件 |
| --- | --- | --- |
| `mgs` | `main → mgs_server` | `backend/tools/server/main.cpp` |
| `bash build/build-local.sh` | `fMain (local build)` | `build/build-local.sh` |
| `curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh` &#124; `bash -s -- [options]` | `fMain (local build) → fDownloadSources` | `build/build-local.sh` |
| `<prefix>/bin/mgs` | `main (launcher) → main → mgs_server` | `build/mgs-launcher.c` |
| `/` | 内嵌的 UI 资源 | `backend/tools/server/server-http.cpp` |
| `/api/config.js` | 运行时 API 前缀和已保存的界面语言 | `backend/tools/server/server-http.cpp` |
| `/api/doc/`、`/api/doc/openapi.json` | `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp` |
| `POST /api/models` | `vModelsRoutes->vPostRouterModels` | `backend/tools/server/server.cpp` |
| `POST /api/models/load` | `vModelsRoutes->vPostRouterModelsLoad` | `backend/tools/server/server.cpp` |
| `POST /api/models/unload` | `vModelsRoutes->vPostRouterModelsUnload` | `backend/tools/server/server.cpp` |
| `GET /api/models/sse` | `vModelsRoutes->vGetRouterModelsSse` | `backend/tools/server/server.cpp` |
| `DELETE /api/models` | `vModelsRoutes->vDelRouterModels` | `backend/tools/server/server.cpp` |
| `GET /api/health` | `routes.vGetHealth` | `backend/tools/server/server.cpp` |
| `GET /api/v1/health` | `routes.vGetHealth` | `backend/tools/server/server.cpp` |
| `GET /api/metrics` | `routes.vGetMetrics` | `backend/tools/server/server.cpp` |
| `GET /api/props` | `routes.vGetProps` | `backend/tools/server/server.cpp` |
| `POST /api/props` | `routes.vPostProps` | `backend/tools/server/server.cpp` |
| `GET /api/models` | `routes.vGetModels`（公开；`reload` 需要 API 密钥） | `backend/tools/server/server.cpp` |
| `GET /api/v1/models` | `routes.vGetModels`（公开；`reload` 需要 API 密钥） | `backend/tools/server/server.cpp` |
| `POST /api/completions` | `routes.vPostCompletions` | `backend/tools/server/server.cpp` |
| `POST /api/v1/completions` | `routes.vPostCompletionsOai` | `backend/tools/server/server.cpp` |
| `POST /api/chat/completions` | `routes.vPostChatCompletions` | `backend/tools/server/server.cpp` |
| `POST /api/v1/chat/completions` | `routes.vPostChatCompletions` | `backend/tools/server/server.cpp` |
| `POST /api/v1/chat/completions/control` | `routes.vPostControl` | `backend/tools/server/server.cpp` |
| `POST /api/v1/responses` | `routes.vPostResponsesOai` | `backend/tools/server/server.cpp` |
| `POST /api/responses` | `routes.vPostResponsesOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/audio/transcriptions` | `routes.vPostTranscriptionsOai` | `backend/tools/server/server.cpp` |
| `POST /api/audio/transcriptions` | `routes.vPostTranscriptionsOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/messages` | `routes.vPostAnthropicMessages` | `backend/tools/server/server.cpp` |
| `POST /api/infill` | `routes.vPostInfill` | `backend/tools/server/server.cpp` |
| `POST /api/embeddings` | `routes.vPostEmbeddings` | `backend/tools/server/server.cpp` |
| `POST /api/v1/embeddings` | `routes.vPostEmbeddingsOai` | `backend/tools/server/server.cpp` |
| `POST /api/rerank` | `routes.vPostRerank` | `backend/tools/server/server.cpp` |
| `POST /api/v1/rerank` | `routes.vPostRerank` | `backend/tools/server/server.cpp` |
| `POST /api/tokenize` | `routes.vPostTokenize` | `backend/tools/server/server.cpp` |
| `POST /api/detokenize` | `routes.vPostDetokenize` | `backend/tools/server/server.cpp` |
| `POST /api/apply-template` | `routes.vPostApplyTemplate` | `backend/tools/server/server.cpp` |
| `POST /api/chat/completions/input_tokens` | `routes.vPostChatCompletionsTok` | `backend/tools/server/server.cpp` |
| `POST /api/v1/chat/completions/input_tokens` | `routes.vPostChatCompletionsTok` | `backend/tools/server/server.cpp` |
| `POST /api/responses/input_tokens` | `routes.vPostResponsesTokOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/responses/input_tokens` | `routes.vPostResponsesTokOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/messages/count_tokens` | `routes.vPostAnthropicCountTokens` | `backend/tools/server/server.cpp` |
| `GET /api/lora-adapters` | `routes.vGetLoraAdapters` | `backend/tools/server/server.cpp` |
| `POST /api/lora-adapters` | `routes.vPostLoraAdapters` | `backend/tools/server/server.cpp` |
| `GET /api/expert-cache` | `routes.vGetExpertCache` | `backend/tools/server/server.cpp` |
| `POST /api/expert-cache` | `routes.vPostExpertCache` | `backend/tools/server/server.cpp` |
| `GET /api/monitor` | `routes.vGetMonitor`（路由器：`vModelsRoutes->vGetRouterMonitor`） | `backend/tools/server/server.cpp` |
| `GET /api/monitor/memory` | `routes.vGetMonitorMemory`（路由器：`vModelsRoutes->vGetRouterMonitorMemory`） | `backend/tools/server/server.cpp` |
| `GET /api/monitor/usage` | `routes.vGetMonitorUsage`（加载期间也可用） | `backend/tools/server/server.cpp` |
| `GET /api/ui/settings` | `fUiSettingsMakeGetHandler`（加载期间也可用） | `backend/tools/server/server.cpp` |
| `POST /api/ui/settings` | `fUiSettingsMakePostHandler`（加载期间也可用） | `backend/tools/server/server.cpp` |
| `#/monitor`（UI） | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |
| `GET /api/slots` | `routes.vGetSlots` | `backend/tools/server/server.cpp` |
| `POST /api/slots/:id_slot` | `routes.vPostSlots` | `backend/tools/server/server.cpp` |
| `GET /api/v1/stream` | `vStreamGetH` | `backend/tools/server/server.cpp` |
| `POST /api/v1/streams/lookup` | `vStreamsLookupH` | `backend/tools/server/server.cpp` |
| `DELETE /api/v1/stream` | `vStreamDeleteH` | `backend/tools/server/server.cpp` |
| `GET /api/cors-proxy` | `vProxyHandlerGet`（不允许导航；返回惰性响应） | `backend/tools/server/server.cpp` |
| `POST /api/cors-proxy` | `vProxyHandlerPost`（不允许导航；返回惰性响应） | `backend/tools/server/server.cpp` |
| `GET /api/tools` | `tools.vHandleGet` | `backend/tools/server/server.cpp` |
| `POST /api/tools` | `tools.vHandlePost`（仅限 `application/json`） | `backend/tools/server/server.cpp` |
| `mgs --local-shell-mcp-server` | `vShellMcpServer → server_shell_mcp::mStart` | `backend/common/arg.cpp; backend/tools/server/server.cpp` |
| `POST /api/mcp/shell` | `vShellMcp.mHandlePost`（加载期间也可用） | `backend/tools/server/server.cpp` |
| `GET /api/mcp/shell`、`DELETE /api/mcp/shell` | `vShellMcp.mHandleUnsupported`（405） | `backend/tools/server/server.cpp` |
| `GET /api/mcp/shell/askpass?call=<id>` | `vShellMcp.mHandleAskpassGet`（加载期间也可用） | `backend/tools/server/server.cpp` |
| `POST /api/mcp/shell/askpass` | `vShellMcp.mHandleAskpassPost`（加载期间也可用） | `backend/tools/server/server.cpp` |
| 由 sudo 作为 `SUDO_ASKPASS` 运行的 `mgs <prompt>` | `fServerShellMcpAskpassMain` | `backend/tools/server/server.cpp; backend/tools/server/server-shell-mcp.cpp` |
| `AIP_HEALTH_ROUTE`、`AIP_PREDICT_ROUTE` | `mRegisterGcpCompat` | `backend/tools/server/server-http.cpp` |
| `HTTP 11080+ (executable)` | 308 重定向到所选的 HTTPS 端口 | `backend/tools/server/server-http.cpp` |
| `HTTPS 11443+ (executable)` | 基于 TLS 的原生 UI 和 API | `backend/tools/server/server-http.cpp; server-tls.cpp` |
| `HTTP 11080+ (Apache)` | 永久重定向到所选的 HTTPS 端口 | `deploy/apache2.conf.in` |
| `HTTPS 11443+ (Apache)` | ProxyPass → 127.0.0.1:8080 | `deploy/apache2.conf.in` |

<a id="impact"></a>
## 影响分析

- **回复语言：** 在自定义 JSON 之后应用该指令，这样即使替换了 `messages` 也无法绕过已启用的设置。每个新请求都读取当前的标志，而不是把语言区域指令保存在对话中。补全和预填充必须使用同一个辅助函数，以保持 KV 前缀一致。流重连会复用已在运行的生成过程，无法更改其语言指令。让全部 15 个语言目录中的 `responseLanguage.*` 条目保持一致；对阿根廷/西班牙的西班牙语、英国/美国英语以及巴西/欧洲葡萄牙语，地区化指引都是显式给出的。模型是否遵循提示词仍取决于模型本身；外部 API 调用方不受影响。

- **远程安装：** URL、GitHub 归档布局以及入口/辅助脚本协议的变化都会影响 `curl | bash`。保持帮助选项一致，并保持仓库来源固定不变。标记文件和锁保护受管理的目录树；不要在其中存放需要在同步后保留下来的本地修改。
- **原生 TLS：** 直接连接的客户端必须使用 HTTPS，并信任该证书或提供已知的 PEM 密钥对。对 Apache 后端和路由器子进程需要显式指定 `--http`；子进程不会继承公开的 TLS 变量。显式指定的主端口永远不会被悄悄更改。不带 OpenSSL 的构建必须显式使用 HTTP。
- **Web 端口：** 影响重定向、公开地址、防火墙以及浏览器存储的源。只有当两个监听器都属于已启动的 Apache 进程之后，才会发布 `ports.json`。排他锁可防止多个启动器并发运行。在搜索 HTTP 端口之前先预留 HTTPS 端口，这样相互重叠的端口序列永远不会共用同一个端口。端口耗尽时会产生明确的错误。
- **前缀和中间件：** 影响每个 HTTP 客户端、身份验证、路由器、流、前端和 Swagger。`server-http`、`server-models`、`api-url` 及其文档需要一起修改。
- **请求加固：** 跨站检查依赖 `Host` 或 `X-Forwarded-Host` 指明公开地址：如果反向代理改写了 `Host` 却不发送 `X-Forwarded-Host`，那么在启用工具或 MCP 服务器时，UI 的写入请求会以 403 失败（`deploy/apache2.conf.in` 保持 `ProxyPreserveHost On`）。提高 `cHttpRequestReadTimeoutSec` 或去掉队列上限，会重新引入慢速客户端耗尽资源的问题；将 `cHttpMaxRequestJsonDepth` 降到 `cMaxSchemaDepth` 的约两倍以下，则会拒绝转换器本可接受的 JSON schema。
- **机密选项：** 保存凭据的选项必须列入 `sSecretOptionEnvs`，否则其值会出现在公开的路由器列表和子进程启动日志中。
- **远程媒体策略：** `common_remote_params` 的默认值对 Hugging Face 和 Docker 清单保持以前的行为；只有媒体路径会设置 `vPublicHostsOnly`。自动跟随重定向，或在检查之后按名称连接，都会重新引入 DNS 重绑定和基于重定向的 SSRF。解码限制是 `mtmd-helper.cpp` 和 `server-common.cpp` 中的通用常量；降低这些限制会拒绝合法的大型媒体。
- **UI 工具标识：** 审批、权限和执行必须使用 `mResolveEnabledEntry` 返回的条目及其服务器 ID；如果重新改为按裸名称解析（在 UI 中取第一个匹配项，在 `MCPStore` 中取最后写入的那个），那么对一个服务器的审批就可能运行另一个服务器的工具，包括本地 shell。
- **补全 schema：** 影响推理验证、默认值和 OpenAPI 文档。扩展必须保留协议字段名。
- **语言目录/选择器：** 影响所有视图以及在模块导入期间初始化的标签。在全部 15 个语言目录中保持键、语言名称和数据参数一致；如果某个翻译使用计数标签，可以省略英文的复数后缀参数。保留 HTML 标签和标题模板令牌，例如 `{{USER}}` 和 `{{ASSISTANT}}`。新的语言区域要同时在 `aLanguages` 和 `dCatalogs` 中注册。希伯来语还会影响导航方向；代码必须保持 LTR。不要翻译协议标识符或用户数据。
- **注册/预取：** 影响 RAM/VRAM 使用、GPU 事件以及映射/缓冲区的生命周期。已注册的页面或槽仍在使用时，绝不能释放它们。
- **MTP：** 影响在分配权重之前进行的解码器选择。保持显式选项的优先级，并检查每个 GGUF 分片。MTP 权重会忽略指向主机内存的覆盖设置，并在 `--fit` 放置主干之前、在确定专家缓存大小之前占用设备内存，因此修改 `mLoadTensors` 中 MTP 层的分配方式，会改变每个带 MTP 的模型的 VRAM 使用量。
- **CMake/安装程序：** 影响每个构建平台。不要复用旧布局留下的缓存，也不要掩盖本地 UI 构建失败。
- **版本：** `MGS_VERSION` 用于命名共享库文件（`libmgs-engine.so.0.1.0`；soname 保持为 `.so.0`），并会出现在 `/api/props`、`system_fingerprint`、shell MCP 的 `serverInfo.version`、下载时的 `User-Agent` 以及 UI 的 `build.json` 中。保持 `MAJOR.MINOR.PATCH` 形式：UI 构建会拒绝 `[A-Za-z0-9._-]` 之外的字符。
- **Vulkan：** 捆绑包与它所加载的系统驱动共享 C/C++ 运行时，因此在更旧或更新的 Debian 版本上构建的捆绑包可能无法加载 ICD。基于锁页内存的专家上传只在 CUDA 下存在；对 `--load-mode` 默认值或主机缓冲区类型的任何修改，都会改变 Vulkan 的传输速度。不属于 `render` 组的服务账户无法打开 GPU。设备缓冲区必须保留在 device-local 内存中；提供 MoE 推理服务时，绝不要启用 `GGML_VK_ALLOW_SYSMEM_FALLBACK` 或 `GGML_VK_PREFER_HOST_MEMORY`。新增的类似缓存的设备分配应使用缓存缓冲区类型，`ggml-vulkan` 中任何新的异步路径都必须接受该 GPU 的每一种设备缓冲区类型。
- **本地捆绑包：** 新链接的库会通过 `ldd` 自动纳入。修改 `OUTPUT_NAME`、路由器的重新执行路径（`fGetServerExecPath`、`MGS_EXECUTABLE`）、加载器的显式执行模式或库的搜索顺序，都会影响 `build/build-local.sh` 和启动器。直接运行 `lib/mgs/mgs` 会绕过捆绑的加载器，不受支持。驱动库必须继续排除在外。
- **MoE 专家缓存：** 影响设备内存（池占用上下文之后剩余的内存；`--moe-cache-reserve` 为其他进程和延迟分配的缓冲区留出空间）、调度器中 split 的同步、CPU 的 `mul_mat_id`/`add_id` 内核（负标识符）以及每种 MoE 架构的计算图（`mBuildMoeFfn`）。用于专家乘积的设备内核必须接受在同一 token 内互不相同的槽标识符。修改 `--fit` 的放置方式会改变所需的 RAM：当 `MemAvailable` 不足时，“所有专家都放在 RAM 中”会回退为整个专家层。有多个 GPU 时，每个 GPU 都为自己的层持有一个池；层的划分（`--fit`、`--tensor-split`）决定每个 GPU 缓存哪些专家。
- **批次大小和线程：** `fRaiseBatch` 中的 `cBatchShare` 在设备内存与提示词处理速度之间进行权衡；微批次在上下文之前确定，因此更大的微批次也会缩短默认上下文（16 GB 上的 Qwen3.6-35B-A3B Q8_0：取 512 时为 122880 个 token，取 4096 时为 89856 个）。服务器按目标的 `mgs_n_batch`/`mgs_n_ubatch` 切分提示词并放置上下文检查点，因此它们的位置会随所选的大小而变化。任何让草稿或 MTP 上下文接收超过其自身 `n_batch` 的批次的修改，都会使其批次缓冲区溢出。`fCommonDefaultThreads` 会改变 CPU 在每个专家位于 RAM 中的 MoE 层中承担的份额；修改上限之前，请在虚拟机和裸机上使用 `--spec-type none` 测量生成速度。
- **`--fit` 中的上下文大小：** `fRaiseContext` 中的 `cContextShare` 在上下文与专家之间进行权衡：它允许的每一个字节的 KV 缓存，都要从专家缓存池或放在设备上的整个专家层中扣除，因此占比越大，生成速度越低；占比越小，默认上下文越短。草稿上下文跟随目标上下文，因此修改 `mLoadModel` 确定其大小的方式，会改变每个带 MTP 或草稿模型的模型留给专家缓存的设备内存。
- **上下文检查点：** 间隔和放置方式会影响主机内存（每个槽位最多 `--ctx-checkpoints` 个状态）和预填充的批处理；每多切分一次，就要对卸载出去的专家多遍历一次。
- **已移除的兼容性：** 客户端、路由器预设、服务覆盖配置和构建脚本都必须使用当前名称；已移除的别名现在作为选项会因未知而失败，作为环境变量会被忽略，作为路由会返回 404。重新引入某个名称时，需要将其添加到 `common/arg.cpp`、`server.cpp` 或 CMake 中，并在 CODE 和 MANUAL 的各语言版本中记录。只有在 README 的概述或安装命令发生变化时才更新 README。
- **监控：** 快照在主循环中构建，主循环是唯一可以访问上下文的线程；在 `mgs_get_memory_breakdown` 之外分配的缓冲区（新的缓存或池）会显示为“other”，直到某个统计调用报告它为止。绝不要让监控经过任务队列：那样会重置空闲计时器并唤醒休眠的模型。路由器的汇总依赖子进程约定（`/api` 前缀，无 API 密钥）。新增的固定定位或底部锚定的 UI 元素必须加上 `--status-bar-height`。
- **Shell MCP 服务器：** 任何能够使用 API（或打开 UI）的人都能以服务用户的身份运行命令，因此它只适用于可信环境，并且应与 API 密钥配合使用。修改路由、`shell_mcp_server` 属性或 `mgs-shell` 标识符，会影响 UI 同步和已保存的设置；修改 Origin 或 Content-Type 检查，会重新引入跨站命令执行；修改 `fRunCommand`，会影响进程组清理、超时和输出限制；修改 `mToolsCallResult` 的 JSON 字段，会影响模型读取的内容、`fParseLocalShellResult`（届时它会把整个结果显示为 stderr 文本）以及调用日志中的 `result` 行；修改 `cCallLogFolder`、文件名或 `server_shell_mcp_call_log` 的 0700/0600 模式，会改变调用日志的位置或把命令输出暴露给其他用户。
- **Web UI 设置：** 该文档由所有浏览器共享，以最后一次保存为准；每次保存都会发送所有分区，因此如果某个 store 把设置保存在 `cUiSettingsStore` 之外，这些设置在其他浏览器上就会丢失。在文档加载之前保存会用不完整的状态覆盖文件，`mSet` 会阻止这种情况。重命名某个分区或配置键后，旧值将不再被读取（没有迁移）。该文件可能包含 MCP 头部，因此模式为 0600；任何拥有 API 密钥的人都可以通过 API 读取它。
- **sudo 密码提示：** 密码会经过浏览器、API（HTTPS 和 API 密钥）以及抽象套接字；修改 `fParentIsSudo`、`SO_PEERCRED` 检查或令牌查找，会让命令能够通过伪造的提示读取密码；修改 `fServerShellMcpAskpassMain` 或它在 `mgs_server` 开头的位置，会破坏辅助程序，或使命令中普通运行的 `mgs` 也表现得像辅助程序；修改 `cAskpassDisplay` 或设置 `DISPLAY` 的条件，会使 sudo 不再使用该辅助程序；启动命令时若不为其创建独立的会话（`POSIX_SPAWN_SETSID`），sudo 就会改为在启动 `mgs` 的终端上提示；从根布局中卸载 `DialogLocalShellAskpass`，会使这些提示在过期之前一直无人应答；修改 `mgs/callId` 会影响 `cMcpStore.mExecuteTool` 和 UI 的轮询；记录 `/mcp/shell/askpass` 的请求体会把密码写入日志。
- **请求队列：** 默认只有一个槽位，每个请求都能使用完整的上下文；提高 `-np` 的默认值（或重新引入自动模式）会把上下文分给多个槽位，并缩短长回答。`mgs_queue` 事件只会发给标有 `mgs_client: "webui"` 的请求：把它们发给其他客户端，或放进 Anthropic Messages 或 OpenAI Responses 流中，会破坏这些客户端的解析器。每个流式处理器都必须能接受作为第一个结果或后续结果出现的 `server_task_result_queue`，并且每条新的推迟路径都必须经过 `mDeferTask`，这样等待中的请求才能保持各自的位置。重新排队的任务会进入推迟队列的末尾，与上游的做法一致。
- **已移除的架构：** Gemma 4 以及所有早于 V4.1 的 DeepSeek 架构都已移除；若要重新引入其中之一，就需要恢复其架构条目、表、计算图、词表和聊天相关部分，并将其重新加入 `RulesForThisProjectOnly.md`。以 `deepseek2` 转换的其他模型的 GGUF（Kimi-K2.x、GLM-4.7-Flash 等）已无法再加载。
- **共享的 DSV4 组件：** `ggml_dsv4_hc_pre`、`ggml_dsv4_hc_pre_gated`、`ggml_dsv4_hc_comb`、`ggml_dsv4_hc_post`、`ggml_lightning_indexer`、`vDsv4HcMult`、`aDsv4CompressRatios` 以及融合的 hc cparams 同时服务于 Kimi-K3、GLM-5.3-Flash、Qwen3.8 和 DeepSeek-V4.1；在这些地方所做的修改会影响这四者。它们没有 Vulkan 或 Metal 内核，因此在这些后端上会改在 CPU 上运行。
- **混合索引内存：** 由 qwen4exp 和 glm5-next 共享；KV 单元存储 token ID，`seq_pos` 保存（位置，单元）对，n-gram 嵌入和状态恢复都依赖于此。
- **外部张量：** `cTensorExternal` 张量从不分配，预取在遇到第一个此类张量时停止；遍历张量数据的代码（量化、保存、内存报告）必须跳过它们。在整个运行期间，读取 Engram 行都需要对模型文件执行 POSIX `pread`。

项目特定规则禁止编写或运行测试。审查是静态的，无法确认运行时、GPU、远程安装或移动端溢出方面的正确性。`tests/` 已预留，但其中没有可执行的测试。


<a id="extensions"></a>
## 扩展点

- **端点：** 实现其处理器，并在 `server.cpp` 中用 `get`、`post` 或 `del` 注册。注册时会加上前缀，并将其纳入清单。用参数、响应和示例扩展 `fRouteDescription` 和 `fOperation`。对于路由器操作，还要检查身份验证和转发。
- **UI 文本：** 在全部 15 个 JSON 语言目录中添加一个稳定的键，并通过 `fTranslate` 解析它，变量值使用 `{p0}`、`{p1}` 等占位符。帮助 HTML 必须保持可信；不要把用户内容作为未经净化的 HTML 插入。要添加一种语言区域，需要在 `i18n.ts` 中导入/注册其完整的 JSON，保持 `aLanguages` 按字母顺序排列，在每个语言目录中添加其 `language.<locale>` 标签，并添加其 `README`、`MANUAL` 和 `CODE` 翻译，同时在每个语言版本的语言栏中加入其链接。
- **CLI 选项：** 扩展 `common/arg.cpp` 及其参数结构，保持公开约定不变，记录其行为，并更新受影响的符号/模块/流程行。保存凭据的选项还要加入 `sSecretOptionEnvs`（`server-models.cpp`）。
- **安装：** 扩展 `fMain` 和 `fShowHelp`，在修改系统之前进行验证，向上传递每一个失败，并更新 CODE 和 MANUAL 的各语言版本；如果安装命令发生变化，还要更新 README 的各语言版本。
- **本地捆绑包：** 若要把另一个系统库排除在 `lib/mgs/` 之外，请扩展 `fIsDriverLibrary`，它同时也决定 `fVerifyBundle` 的行为。在临时的 `bundle/` 目录下组装新的辅助文件，将完成的产物复制到前缀中各自目标位置的旁边，然后使用 `fReplaceDirectory` 替换目录；将新安装的条目加入 `fMain` 中的归属检查。
- **逐专家张量：** 在 `mgs_expert_role` 中添加一个角色，在 `fLayerExpertTensors` 和 `mBuildMoeFfn` 的 `vTensors` 集合中映射它，并将池张量传给设备分支；之后，池、上传和预取的复用机制就会涵盖它。
- **构建后端：** 将该选项添加到两个脚本中（`build-local.sh` 和 `install-common.sh`，以及 Debian 入口脚本的透传列表），将其软件包加入 `fInstallSystemDependencies`，将其 `-DGGML_*` 标志加入配置参数，并将其消息加入摘要；二进制文件直接链接的驱动库要加入 `fIsDriverLibrary`。
- **新版本：** 项目的每次修改都要将 `backend/cmake/build-info.cmake` 中 `MGS_VERSION` 的补丁号加一（0.1.1、0.1.2……）。二进制文件、库和界面都从这里获取版本，因此无需修改其他任何地方。
- **内存段：** 在 `mBuildMonitorSnapshot`（`fAdd`）、`aMonitorSegments` 和 `MonitorSegmentKey` 中添加该键，然后将其放入 `aMonitorSegmentGroups` 的某个组中（八种分类颜色都已在使用：应并入某个组，而不是添加第九种颜色），并在 MANUAL 的内存段表格和 `fRouteDescription` 中记录它。
- **Shell 工具：** 在 `server_shell_mcp::mStart` 的 `lCandidates` 中添加一个候选项（工具名、标题、可执行文件名以及放在命令之前的参数）；检测、`tools/list`、`tools/call` 和日志无需进一步修改即可识别它。将该工具名加入 `aShellMcpToolNames`，使聊天以 stdin、stdout 和 stderr 框显示它，并更新 `fRouteDescription` 中的路由描述、`--local-shell-mcp-server` 的帮助、`expandLocalShellCommands` 的帮助文本以及 MANUAL。
- **由界面保存的设置：** 普通设置放在 `settings-registry.ts` 中，并随 `config` 分区一起保存。新的分区需要在 `UiSettingsDocument` 中添加一个字段，在负责它的 store 中用 `cUiSettingsStore.mRegister` 注册一个加载器，并在每次修改时调用 `cUiSettingsStore.mSet`；在 `server-api-doc.cpp` 的 `/ui/settings` 描述中记录它。绝不要把设置保存在 `localStorage` 中。
- **面向其他客户端的排队通知：** 想要接收 `mgs_queue` 事件的客户端需要发送 `"mgs_client": "webui"` 和 `"stream": true`；其他来源值则需要在 `task_params` 中添加一个字段，在 `fMakeMgsCmplSchema` 的 `mgs_client` 字段中添加其处理逻辑，在 `mCountRunning`、`mDeferTask`、`mNotifyQueuePositions` 和 `server_task_result_queue` 中添加其自己的计数器，并在 `fQueueStatusText` 和 15 个语言目录中添加翻译后的文本。

实现参考：[Svelte 编译器](https://svelte.dev/docs/svelte/svelte-compiler)、[OpenAPI 服务器（servers）](https://swagger.io/docs/specification/v3_0/api-host-and-base-path/) 和 [Apache mod_proxy](https://httpd.apache.org/docs/2.4/mod/mod_proxy.html)。Swagger UI 5.30.2 以 vendored 方式随附在 `backend/vendor/swagger-ui/` 中，并附有其许可证声明。



[Apache 的 Listen 指令与重启行为](https://httpd.apache.org/docs/2.4/bind.html)

[GitHub 源代码归档](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives)

<a id="upstream-review"></a>
## 上游审查——2026-10-02

共筛查了 1,103 条上游历史记录（含 8 月 4 日）；其中 1,081 条的日期在 8 月 5 日至 10 月 2 日之间，截止于 **a8c9a4e7ccba**（08:18:51 UTC）。在不使用 Git 的情况下使用了源代码归档/API。用于比较的快照 **5788b510a1e3** 来自 8 月 4 日 09:12 UTC；它只是一个参考，并非经过证实的本地分叉的 SHA。这是对 38 个提交或其适用部分的选择性集成，而不是完整升级到上游的最新版本。现有的公开 mgs/GGML 头文件、本地主机注册、专家预取、API 前缀、翻译和部署布局都得以保留。

本地提交/文件清单：`_/upstream/upstream-review-2026-10-02.json`（不包含在发布的源码中）。

| 领域 | 已应用的修改 | 上游提交 |
| --- | --- | --- |
| 调度器 | 无输入 split 的同步、更少的人为 split、空专家 ID。 | #26040, #28387, #28739 |
| CUDA 计算图 | 为异步 MoE 路径保留计算图；为有输出和无输出的情况分别使用独立的 MTP arena。 | #26802, #28549 |
| CUDA 内核 | 针对 10 个选中专家的快速路由；小批次 GLU/top-k 融合；修复 warp 竞争。 | #27978, #27621, #28475 |
| CUDA 调优 | 依赖量化类型的 MMVQ/MMQ 阈值、Pascal/Volta 调优、无分支的 Q4_K/Q5_K 以及 Spark L2 预取。 | #26079, #26264, #28912, #29753, #26705 |
| HIP / Vulkan / SYCL | 在 RDNA/Vulkan 上按专家确定 tile 大小，减少 Vulkan 的无效工作，并在 SYCL 上分派 IQ MoE。 | #28552, #28935, #29182, #25483, #28476 |
| MTP 内存 | 只适配已加载的头，正确跳过融合张量，过滤 KV 层，并隔离草稿嵌入设置。 | #26605, #29014, #28630, #26352 |
| MTP 正确性 | 保持批次行顺序，在 EOG 处停止，跳过不活跃的草稿，并使用多模态位置。 | #29019, #29638, #27404, #28715 |
| GLM-4.5-Air | 可执行的 MTP 计算图，以及对完整单头 glm4moe GGUF 的自动检测。 | #26534 |
| CPU 注意力 | 向量化的 F16 转换，以及分块处理的注意力尾部，并正确处理填充部分的 softcap 掩码。 | #26947, #29423 |
| Direct I/O | 将额外的暂存分配限制在 64 MiB，而不是复制整个张量。 | #29749 |
| 服务器 | 在统计 token 之前安全唤醒；在父进程保留路由器基于 API 密钥文件的身份验证；公开草稿指标。 | #29309, #28938, #26389 |
| UI | 导出持久化的完整对话，停止重复探测已禁用的工具，保留首次访问时的用户设置。 | #27432, #28646, #27365 |
| 选项 | 显式选择 FA 量化组合（已弃用的别名后来被移除），移除一个未使用的对端批次选项，并更正关于 load-mode 的建议。 | #28079, #28177, #28334 |

后续适配（2026-10-10）：#30112（2026-10-08 合并）的多 GPU 设计被应用到本分叉自己的专家缓存（`mgs-expert-cache.cpp`）上：每个 GPU 为分配给它的层持有一个池，请求的大小在各 GPU 之间分配，按 GPU 测量上传带宽，并且在流水线并行时不启用缓存。没有复制其代码，也没有集成上游的缓存本身（#29887）：本分叉保留自己的缓存，该缓存还会在 GPU 和 CPU 之间拆分每一步计算。

后续集成（2026-10-13，版本 0.1.8）：项目必须支持而本分叉尚不认识的架构，已从上游 master 移植过来；对于 DeepSeek-V4.1，则依据尚未合并的 pull request 和参考代码实现。

| 架构 | 已应用的修改 | 来源 |
| --- | --- | --- |
| Kimi-K3（`kimi-k3`） | KDA + MLA 混合架构、潜空间 MoE、残差混合、聊天解析器。 | #26185, #28121, #28068, #28466, #25444（部分） |
| GLM-5.3-Flash（`glm5-next`、`glm5v`） | 计算图、k-pool 索引器、MTP、视觉投影器、SWIGLU_CLAMP。 | #27773, #29745, #29824, #29958, #29994, #30042, #29928, #30017, #30097, #30257, #27930, #29805 |
| Qwen3.8-Flash-Next（`qwen4exp`） | 计算图、混合索引内存、超连接、逐层嵌入、QSA、MTP、行预取。 | #27742, #27880, #27941, #28023, #28123, #28569, #28751, #28896, #28901, #29599（部分）, #29622（部分）, #29751, #29761, #29819, #29825, #30160 |
| DeepSeek-V4.1-Flash（`deepseek41`） | 计算图、支持两种布局的加载器、缓存、Engram、DSML。 | 尚未合并的 #28696；以 ik_llama.cpp #2455、#2506、#2507 和 #2512 作为参考；antirez/ds4 的 `ds4_engram.c` 和 `gguf-tools/deepseek41_metadata.py`（MIT）；官方 DeepSeek-V4.1-Flash 推理代码（MIT） |

未移植：稀疏 flash-attention 内核（#27970、#28770），因此 QSA 和 DSA 层虽然结果正确，但在长上下文下不会提速；混合 token 与嵌入的批次（#29622 的其余部分）；张量延迟读取；DeepSeek-V4.1 的 DSpark、视觉以及 FP8/FP4 KV 模拟。

多 token CUDA 融合有意只覆盖本分叉中已有的操作；它没有引入 SWIGLU_CLAMP，后者是后来随 GLM-5.3-Flash（#27930）引入的。GLM 的自动 MTP 仍然要求完整的 GGUF 主干/头，并且恰好只有一个头。UI 初始化重构没有必要，因为本分叉以同步方式初始化持久化的设置；只移植了保留用户设置的那项修复。

推迟：CPU 分块 K/IQ 矩阵乘法（#27851）需要较新的工作区/量化块集成；加权专家归约和无条件 top-k 融合（#25952/#28432/#28422）需要分配器方面的依赖；Metal 相关工作（#28301/#28948）需要其重新组织后的内核/融合基础设施；OpenCL MTP 分派（#27637）需要较新的二进制内核。大范围的 UI 渲染重构（#28460）需要更大规模的 UI 迁移；行预取（#29599）和 Qwen4Exp MTP（#29761）后来随 Qwen3.8-Flash-Next 一起集成。JSON 清单列出了每个候选项的链接，并记录了其原因。转换工具、额外的二进制文件、CI/发布和测试不在这次仅限服务器的集成范围之内。

审查涵盖源码差异、声明/调用点、共享的 CUDA/HIP/MUSA 构建路径以及本地扩展的保留情况。没有进行构建、测试、GPU 基准测试或浏览器运行。上游的基准测试结论并不是对本分叉的测量；速度和运行时正确性仍有待在部署硬件上验证。

TLS：[OpenSSL 证书/密钥加载](https://docs.openssl.org/3.0/man3/SSL_CTX_use_certificate/) · [X.509 扩展](https://docs.openssl.org/3.0/man5/x509v3_config/)
