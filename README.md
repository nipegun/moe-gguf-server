# moe-gguf-server

This README is also available in:\
[de-DE](doc/README.de-DE.md), [en-GB](doc/README.en-GB.md), **en-US**, [es-AR](doc/README.es-AR.md), [es-ES](doc/README.es-ES.md), [fr-FR](doc/README.fr-FR.md), [he-IL](doc/README.he-IL.md), [hi-IN](doc/README.hi-IN.md), [it-IT](doc/README.it-IT.md), [ja-JP](doc/README.ja-JP.md), [ko-KR](doc/README.ko-KR.md), [pt-BR](doc/README.pt-BR.md), [pt-PT](doc/README.pt-PT.md), [ru-RU](doc/README.ru-RU.md), [zh-CN](doc/README.zh-CN.md)

## What is it

Run GGUF Mixture of Experts (MoE) models locally through a web chat and an HTTP inference API.

Like llama.cpp and FreeToken but crafted with nipegunian scalpel precision to achieve the fastest inference speed possible.

## Screenshots

Screenshots have not been added yet.

## Architectures and models

The GGUF architectures that `mgs` loads, each with its models (details in the [manual](doc/MANUAL.md#supported-models)):

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

## Deploy

Run it as `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Quick build

Choose the section that matches your hardware. The commands are for Debian 13.

### Vulkan

For AMD, Intel and NVIDIA GPUs. As `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Then, as your regular user, from any folder:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

For NVIDIA GPUs. As `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Then, as your regular user, from any folder:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

For AMD GPUs supported by ROCm; use Vulkan for older ones, such as the Radeon RX 580. Debian 13 only ships HIP 5.7 and the build needs 6.1 or later, so these commands install ROCm 7.2.4 from the AMD repository. As `root`:

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

Add your regular user to the `render` and `video` groups and log in again, so that the build can detect the GPU.

Then, as your regular user, from any folder:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### CPU only

For computers without a GPU. As `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Then, as your regular user, from any folder:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- All the options and requirements of the build are in the [manual](doc/MANUAL.md#local-build).

## Read the documentation

Explore:

- [doc/MANUAL.md](doc/MANUAL.md) to build, install and use it, diagnose build failures, choose among 15 interface languages and optionally request replies in the same regional variety.
- [doc/CODE.md](doc/CODE.md) to understand the MoE engine, MTP memory placement, source structure and backend/frontend build contracts (for developers and AI models).

## Sponsor this project

- **[Hire me](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** to add new features.
- **GitHub Sponsors.** You can also support the project financially on GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
