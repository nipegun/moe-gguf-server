# moe-gguf-server

Ce README est également disponible en :\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), **fr-FR**, [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Qu'est-ce que c'est

Exécutez en local des modèles GGUF Mixture of Experts (MoE) au moyen d'un chat web et d'une API HTTP d'inférence.

Comme llama.cpp et FreeToken, mais façonné avec une précision de scalpel nipegunienne pour atteindre la vitesse d'inférence la plus élevée possible.

## Captures d'écran

Aucune capture d'écran n'a encore été ajoutée.

## Déploiement

Exécutez-le en tant que `root`.

**Debian** (avec un CUDA Toolkit déjà installé ; l'installateur adapte NCCL à cette version de CUDA) :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilation

Installez d'abord toutes les dépendances :

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Ensuite, exécutez avec votre utilisateur habituel, depuis n'importe quel dossier :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- Les [dépendances de compilation](MANUAL.fr-FR.md#local-build) doivent déjà être installées.

## Lire la documentation

Explorez :

- [doc/MANUAL.fr-FR.md](MANUAL.fr-FR.md) pour le compiler, l'installer et l'utiliser, diagnostiquer les échecs de compilation, choisir parmi 15 langues d'interface et, si vous le souhaitez, demander des réponses dans la même variante régionale.
- [doc/CODE.fr-FR.md](CODE.fr-FR.md) pour comprendre le moteur MoE, le placement de MTP en mémoire, la structure des sources et les contrats de compilation du backend et du frontend (pour les développeurs et les modèles d'IA).

## Soutenir ce projet

- **[Engagez-moi](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** pour ajouter de nouvelles fonctionnalités.
- **GitHub Sponsors.** Vous pouvez aussi soutenir financièrement le projet sur GitHub : [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
