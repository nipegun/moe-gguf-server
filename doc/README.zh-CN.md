# moe-gguf-server

本 README 还提供以下语言版本：\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), **zh-CN**

## 这是什么

通过网页聊天和 HTTP 推理 API，在本地运行 GGUF 混合专家（MoE）模型。

与 llama.cpp 和 FreeToken 类似，但以 nipegun 式的手术刀般的精准度精心打造，力求实现尽可能快的推理速度。

## 截图

尚未添加截图。

## 架构与模型

`mgs` 可加载的 GGUF 架构及其各自的模型（详见[手册](MANUAL.zh-CN.md#supported-models)）：

- `qwen35moe`
  - Qwen3.6
- `qwen4exp`
  - Qwen3.8-Flash-Next
- `glm-dsa`
  - GLM-5.3
- `glm5-next`
  - GLM-5.3-Flash
- `kimi-k3`
  - Kimi-K3
- `deepseek41`
  - DeepSeek-V4.1-Flash
- `minimax-m3`
  - MiniMax-M3
- `nemotron_h_moe`
  - Nemotron-3
- `mimo2`
  - MiMo-V2.6
- `flint`
  - Flint

## 部署

请以 `root` 身份运行。

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## 快速构建

请选择与您的硬件相符的部分。以下命令适用于 Debian 13。

### Vulkan

适用于 AMD、Intel 和 NVIDIA GPU。以 `root` 身份运行：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

然后以普通用户身份，在任意文件夹中运行：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

适用于 NVIDIA GPU。以 `root` 身份运行：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

然后以普通用户身份，在任意文件夹中运行：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

适用于 ROCm 支持的 AMD GPU；较旧的 GPU（例如 Radeon RX 580）请使用 Vulkan。Debian 13 只提供 HIP 5.7，而构建需要 6.1 或更高版本，因此这些命令会从 AMD 软件源安装 ROCm 7.2.4。以 `root` 身份运行：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
mkdir -p -m 0755 /etc/apt/keyrings
curl -fsSL -o /etc/apt/keyrings/rocm.asc https://repo.radeon.com/rocm/rocm.gpg.key
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/7.2.4 noble main' > /etc/apt/sources.list.d/rocm.list
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/graphics/7.2.4/ubuntu noble main' >> /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update
apt-get install -y rocm-hip-runtime-dev hipblas-dev rocblas-dev hipblaslt-dev
```

将您的普通用户加入 `render` 和 `video` 组并重新登录，以便构建能够检测到 GPU。

然后以普通用户身份，在任意文件夹中运行：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### 仅 CPU

适用于没有 GPU 的计算机。以 `root` 身份运行：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

然后以普通用户身份，在任意文件夹中运行：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- 构建的所有选项和要求见[手册](MANUAL.zh-CN.md#local-build)。

## 阅读文档

浏览：

- [doc/MANUAL.zh-CN.md](MANUAL.zh-CN.md)：了解如何构建、安装和使用本项目，诊断构建失败，在 15 种界面语言中进行选择，并可选择让回复使用相同的地区语言变体。
- [doc/CODE.zh-CN.md](CODE.zh-CN.md)：了解 MoE 引擎、MTP 内存放置、源代码结构以及后端/前端构建约定（面向开发者和 AI 模型）。

## 赞助本项目

- **[雇用我](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**，为本项目添加新功能。
- **GitHub Sponsors。** 您也可以在 GitHub 上为本项目提供资金支持：[github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)。
