# moe-gguf-server

Este README também está disponível em:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), **pt-BR**, [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## O que é

Execute modelos GGUF Mixture of Experts (MoE) localmente por meio de um chat web e de uma API HTTP de inferência.

Como o llama.cpp e o FreeToken, mas elaborado com a precisão de bisturi nipeguniana para alcançar a maior velocidade de inferência possível.

## Capturas de tela

As capturas de tela ainda não foram adicionadas.

## Arquiteturas e modelos

As arquiteturas GGUF que o `mgs` carrega, cada uma com seus modelos (detalhes no [manual](MANUAL.pt-BR.md#supported-models)):

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

## Implantação

Execute-o como `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilação rápida

Escolha a seção que corresponde ao seu hardware. Os comandos são para o Debian 13.

### Vulkan

Para GPUs AMD, Intel e NVIDIA. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Depois, com seu usuário comum, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Para GPUs NVIDIA. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Depois, com seu usuário comum, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Para GPUs AMD compatíveis com o ROCm; para as mais antigas, como a Radeon RX 580, use Vulkan. O Debian 13 traz apenas o HIP 5.7 e a compilação precisa da 6.1 ou posterior, por isso estes comandos instalam o ROCm 7.2.4 a partir do repositório da AMD. Como `root`:

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

Adicione seu usuário comum aos grupos `render` e `video` e faça login novamente, para que a compilação consiga detectar a GPU.

Depois, com seu usuário comum, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### Somente CPU

Para computadores sem GPU. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Depois, com seu usuário comum, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Todas as opções e requisitos da compilação estão no [manual](MANUAL.pt-BR.md#local-build).

## Leia a documentação

Explore:

- [doc/MANUAL.pt-BR.md](MANUAL.pt-BR.md) para compilar, instalar e usar o projeto, diagnosticar falhas de compilação, escolher entre 15 idiomas da interface e, opcionalmente, pedir respostas na mesma variante regional.
- [doc/CODE.pt-BR.md](CODE.pt-BR.md) para entender o motor MoE, o posicionamento da memória do MTP, a estrutura do código-fonte e os contratos de compilação do backend e do frontend (para desenvolvedores e modelos de IA).

## Patrocine este projeto

- **[Contrate-me](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** para adicionar novos recursos.
- **GitHub Sponsors.** Você também pode apoiar o projeto financeiramente no GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
