# moe-gguf-server

Este README también está disponible en:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), **es-AR**, [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Qué es

Ejecutá modelos GGUF Mixture of Experts (MoE) localmente mediante un chat web y una API HTTP de inferencia.

Como llama.cpp y FreeToken, pero hecho con precisión de bisturí nipeguniano para lograr la mayor velocidad de inferencia posible.

## Capturas de pantalla

Todavía no se agregaron capturas de pantalla.

## Despliegue

Ejecutalo como `root`.

**Debian** (con un CUDA Toolkit ya instalado; el instalador ajusta NCCL a esa versión de CUDA):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilación

Instalá primero todas las dependencias:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Después, ejecutalo con tu usuario habitual, desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- Las [dependencias de compilación](MANUAL.es-AR.md#local-build) tienen que estar ya instaladas.

## Leé la documentación

Explorá:

- [doc/MANUAL.es-AR.md](MANUAL.es-AR.md) para compilarlo, instalarlo, usarlo, diagnosticar fallos de compilación, elegir entre 15 idiomas de interfaz y pedir respuestas en la misma variante regional.
- [doc/CODE.es-AR.md](CODE.es-AR.md) para entender el motor MoE, la ubicación de MTP en memoria, la estructura del código y los contratos de compilación del backend y del frontend (para desarrolladores y modelos de IA).

## Patrociná este proyecto

- **[Contratame](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** para agregar nuevas funcionalidades.
- **GitHub Sponsors.** También podés apoyar el proyecto económicamente en GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
