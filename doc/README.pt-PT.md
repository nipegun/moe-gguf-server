# moe-gguf-server

Este README também está disponível em:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), **pt-PT**, [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## O que é

Execute modelos GGUF Mixture of Experts (MoE) localmente através de um chat web e de uma API HTTP de inferência.

Como o llama.cpp e o FreeToken, mas trabalhado com a precisão de bisturi nipeguniana para alcançar a maior velocidade de inferência possível.

## Capturas de ecrã

Ainda não foram adicionadas capturas de ecrã.

## Arquiteturas e modelos

As arquiteturas GGUF que o `mgs` carrega, cada uma com os seus modelos (detalhes no [manual](MANUAL.pt-PT.md#supported-models)):

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

## Implementação

Execute-o como `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilação rápida

Escolha a secção que corresponde ao seu hardware. Os comandos são para o Debian 13.

### Vulkan

Para GPU AMD, Intel e NVIDIA. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Depois, com o seu utilizador habitual, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Para GPU NVIDIA. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Depois, com o seu utilizador habitual, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Para GPU AMD suportadas pelo ROCm; para as mais antigas, como a Radeon RX 580, utilize Vulkan. O Debian 13 inclui apenas o HIP 5.7 e a compilação precisa da 6.1 ou posterior, pelo que estes comandos instalam o ROCm 7.2.4 a partir do repositório da AMD. Como `root`:

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

Adicione o seu utilizador habitual aos grupos `render` e `video` e volte a iniciar sessão, para que a compilação consiga detetar a GPU.

Depois, com o seu utilizador habitual, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### Apenas CPU

Para computadores sem GPU. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Depois, com o seu utilizador habitual, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Todas as opções e requisitos da compilação estão no [manual](MANUAL.pt-PT.md#local-build).

## Ler a documentação

Explore:

- [doc/MANUAL.pt-PT.md](MANUAL.pt-PT.md) para o compilar, instalar e utilizar, diagnosticar falhas de compilação, escolher entre os 15 idiomas da interface e, opcionalmente, pedir respostas na mesma variedade regional.
- [doc/CODE.pt-PT.md](CODE.pt-PT.md) para compreender o motor MoE, a colocação da memória MTP, a estrutura do código-fonte e os contratos de compilação do backend e do frontend (para programadores e modelos de IA).

## Patrocinar este projeto

- **[Contrate-me](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** para adicionar novas funcionalidades.
- **GitHub Sponsors.** Também pode apoiar financeiramente o projeto no GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
