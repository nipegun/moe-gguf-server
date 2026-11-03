# moe-gguf-server

Este README también está disponible en:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), **es-ES**, [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Qué es

Ejecuta modelos GGUF Mixture of Experts (MoE) localmente mediante un chat web y una API HTTP de inferencia.

Como llama.cpp y FreeToken, pero elaborado con precisión de bisturí nipeguniano para lograr la mayor velocidad de inferencia posible.

## Capturas de pantalla

Todavía no se han añadido capturas de pantalla.

## Arquitecturas y modelos

Las arquitecturas GGUF que carga `mgs`, cada una con sus modelos (detalles en el [manual](MANUAL.es-ES.md#supported-models)):

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

## Despliegue

Ejecútalo como `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilación rápida

Elige la sección que corresponda a tu hardware. Los comandos son para Debian 13.

### Vulkan

Para GPU de AMD, Intel y NVIDIA. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Después, con tu usuario habitual, desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Para GPU de NVIDIA. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Después, con tu usuario habitual, desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Para GPU de AMD compatibles con ROCm; para las más antiguas, como la Radeon RX 580, usa Vulkan. Debian 13 solo incluye HIP 5.7 y la compilación necesita la 6.1 o posterior, así que estos comandos instalan ROCm 7.2.4 desde el repositorio de AMD. Como `root`:

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

Añade tu usuario habitual a los grupos `render` y `video` y vuelve a iniciar sesión, para que la compilación pueda detectar la GPU.

Después, con tu usuario habitual, desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### Solo CPU

Para equipos sin GPU. Como `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Después, con tu usuario habitual, desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Todas las opciones y requisitos de la compilación están en el [manual](MANUAL.es-ES.md#local-build).

## Lee la documentación

Explora:

- [doc/MANUAL.es-ES.md](MANUAL.es-ES.md) para compilarlo, instalarlo, usarlo, diagnosticar fallos de compilación, elegir entre 15 idiomas de interfaz y pedir respuestas en la misma variante regional.
- [doc/CODE.es-ES.md](CODE.es-ES.md) para entender el motor MoE, la ubicación de MTP en memoria, la estructura del código y los contratos de compilación del backend y del frontend (para desarrolladores y modelos de IA).

## Patrocina este proyecto

- **[Contrátame](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** para añadir nuevas funcionalidades.
- **GitHub Sponsors.** También puedes apoyar el proyecto económicamente en GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
