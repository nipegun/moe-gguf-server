# moe-gguf-server

Este README também está disponível em:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), **pt-BR**, [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## O que é

Execute modelos GGUF Mixture of Experts (MoE) localmente por meio de um chat web e de uma API HTTP de inferência.

Como o llama.cpp e o FreeToken, mas elaborado com a precisão de bisturi nipeguniana para alcançar a maior velocidade de inferência possível.

## Capturas de tela

As capturas de tela ainda não foram adicionadas.

## Implantação

Execute-o como `root`.

**Debian** (com um CUDA Toolkit já instalado; o instalador ajusta o NCCL para corresponder a essa versão do CUDA):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilação

Primeiro, instale todas as dependências:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Depois, execute como seu usuário comum, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- As [dependências de compilação](MANUAL.pt-BR.md#local-build) já devem estar instaladas.

## Leia a documentação

Explore:

- [doc/MANUAL.pt-BR.md](MANUAL.pt-BR.md) para compilar, instalar e usar o projeto, diagnosticar falhas de compilação, escolher entre 15 idiomas da interface e, opcionalmente, pedir respostas na mesma variante regional.
- [doc/CODE.pt-BR.md](CODE.pt-BR.md) para entender o motor MoE, o posicionamento da memória do MTP, a estrutura do código-fonte e os contratos de compilação do backend e do frontend (para desenvolvedores e modelos de IA).

## Patrocine este projeto

- **[Contrate-me](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** para adicionar novos recursos.
- **GitHub Sponsors.** Você também pode apoiar o projeto financeiramente no GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
