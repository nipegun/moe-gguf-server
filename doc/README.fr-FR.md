# moe-gguf-server

Ce README est également disponible en :\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), **fr-FR**, [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Qu'est-ce que c'est

Exécutez en local des modèles GGUF Mixture of Experts (MoE) au moyen d'un chat web et d'une API HTTP d'inférence.

Comme llama.cpp et FreeToken, mais façonné avec une précision de scalpel nipegunienne pour atteindre la vitesse d'inférence la plus élevée possible.

## Captures d'écran

Aucune capture d'écran n'a encore été ajoutée.

## Architectures et modèles

Les architectures GGUF que `mgs` charge, chacune avec ses modèles (détails dans le [manuel](MANUAL.fr-FR.md#supported-models)) :

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

## Déploiement

Exécutez-le en tant que `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilation rapide

Choisissez la section qui correspond à votre matériel. Les commandes sont prévues pour Debian 13.

### Vulkan

Pour les GPU AMD, Intel et NVIDIA. En tant que `root` :

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Ensuite, avec votre utilisateur habituel, depuis n'importe quel dossier :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Pour les GPU NVIDIA. En tant que `root` :

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Ensuite, avec votre utilisateur habituel, depuis n'importe quel dossier :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Pour les GPU AMD pris en charge par ROCm ; pour les plus anciens, comme la Radeon RX 580, utilisez Vulkan. Debian 13 ne fournit que HIP 5.7 et la compilation exige la 6.1 ou une version ultérieure ; ces commandes installent donc ROCm 7.2.4 depuis le dépôt d'AMD. En tant que `root` :

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

Ajoutez votre utilisateur habituel aux groupes `render` et `video` et reconnectez-vous, pour que la compilation puisse détecter le GPU.

Ensuite, avec votre utilisateur habituel, depuis n'importe quel dossier :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### CPU uniquement

Pour les ordinateurs sans GPU. En tant que `root` :

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Ensuite, avec votre utilisateur habituel, depuis n'importe quel dossier :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Toutes les options et tous les prérequis de la compilation se trouvent dans le [manuel](MANUAL.fr-FR.md#local-build).

## Lire la documentation

Explorez :

- [doc/MANUAL.fr-FR.md](MANUAL.fr-FR.md) pour le compiler, l'installer et l'utiliser, diagnostiquer les échecs de compilation, choisir parmi 15 langues d'interface et, si vous le souhaitez, demander des réponses dans la même variante régionale.
- [doc/CODE.fr-FR.md](CODE.fr-FR.md) pour comprendre le moteur MoE, le placement de MTP en mémoire, la structure des sources et les contrats de compilation du backend et du frontend (pour les développeurs et les modèles d'IA).

## Soutenir ce projet

- **[Engagez-moi](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** pour ajouter de nouvelles fonctionnalités.
- **GitHub Sponsors.** Vous pouvez aussi soutenir financièrement le projet sur GitHub : [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
