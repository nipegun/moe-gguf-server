# moe-gguf-server

Diese README ist auch verfügbar auf:\
**de-DE**, [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Was ist das

Führen Sie GGUF-Modelle mit Mixture of Experts (MoE) lokal über einen Web-Chat und eine HTTP-Inferenz-API aus.

Wie llama.cpp und FreeToken, aber mit nipegunscher Skalpellpräzision gefertigt, um die höchstmögliche Inferenzgeschwindigkeit zu erreichen.

## Screenshots

Es wurden noch keine Screenshots hinzugefügt.

## Bereitstellung

Führen Sie es als `root` aus.

**Debian** (mit einem bereits installierten CUDA Toolkit; das Installationsprogramm passt NCCL an diese CUDA-Version an):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Kompilierung

Installieren Sie zuerst alle Abhängigkeiten:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Führen Sie dann als Ihr normaler Benutzer in einem beliebigen Ordner Folgendes aus:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- Die [Build-Abhängigkeiten](MANUAL.de-DE.md#local-build) müssen bereits installiert sein.

## Dokumentation lesen

Entdecken Sie:

- [doc/MANUAL.de-DE.md](MANUAL.de-DE.md), um es zu kompilieren, zu installieren und zu verwenden, Build-Fehler zu diagnostizieren, eine der 15 Sprachen der Oberfläche zu wählen und optional Antworten in derselben regionalen Sprachvariante anzufordern.
- [doc/CODE.de-DE.md](CODE.de-DE.md), um die MoE-Engine, die Speicherplatzierung von MTP, die Struktur des Quellcodes und die Build-Verträge von Backend und Frontend zu verstehen (für Entwickler und KI-Modelle).

## Dieses Projekt unterstützen

- **[Beauftragen Sie mich](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**, um neue Funktionen hinzuzufügen.
- **GitHub Sponsors.** Sie können das Projekt auf GitHub auch finanziell unterstützen: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
