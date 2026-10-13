# moe-gguf-server

本 README 还提供以下语言版本：\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), **zh-CN**

## 这是什么

通过网页聊天和 HTTP 推理 API，在本地运行 GGUF 混合专家（MoE）模型。

与 llama.cpp 和 FreeToken 类似，但以 nipegun 式的手术刀般的精准度精心打造，力求实现尽可能快的推理速度。

## 截图

尚未添加截图。

## 部署

请以 `root` 身份运行。

**Debian**（需已安装 CUDA Toolkit；安装程序会调整 NCCL，使其与该 CUDA 版本相匹配）：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## 构建

首先安装所有依赖项：

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
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- 必须事先安装好[构建依赖项](MANUAL.zh-CN.md#local-build)。

## 阅读文档

浏览：

- [doc/MANUAL.zh-CN.md](MANUAL.zh-CN.md)：了解如何构建、安装和使用本项目，诊断构建失败，在 15 种界面语言中进行选择，并可选择让回复使用相同的地区语言变体。
- [doc/CODE.zh-CN.md](CODE.zh-CN.md)：了解 MoE 引擎、MTP 内存放置、源代码结构以及后端/前端构建约定（面向开发者和 AI 模型）。

## 赞助本项目

- **[雇用我](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**，为本项目添加新功能。
- **GitHub Sponsors。** 您也可以在 GitHub 上为本项目提供资金支持：[github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)。
