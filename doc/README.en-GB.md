# moe-gguf-server

This README is also available in:\
[de-DE](README.de-DE.md), **en-GB**, [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## What is it

Run GGUF Mixture of Experts (MoE) models locally through a web chat and an HTTP inference API.

Like llama.cpp and FreeToken but crafted with nipegunian scalpel precision to achieve the fastest inference speed possible.

## Screenshots

Screenshots have not been added yet.

## Deploy

Run it as `root`.

**Debian** (with an already installed CUDA Toolkit; the installer adjusts NCCL to match that CUDA version):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Build

Install all the dependencies first:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Then run as your regular user, from any folder:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- The [build dependencies](MANUAL.en-GB.md#local-build) must already be installed.

## Read the documentation

Explore:

- [doc/MANUAL.en-GB.md](MANUAL.en-GB.md) to build, install and use it, diagnose build failures, choose among 15 interface languages and optionally request replies in the same regional variety.
- [doc/CODE.en-GB.md](CODE.en-GB.md) to understand the MoE engine, MTP memory placement, source structure and backend/frontend build contracts (for developers and AI models).

## Sponsor this project

- **[Hire me](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** to add new features.
- **GitHub Sponsors.** You can also support the project financially on GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
