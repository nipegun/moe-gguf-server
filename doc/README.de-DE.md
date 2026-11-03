# moe-gguf-server

Diese README ist auch verfügbar auf:\
**de-DE**, [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Was ist das

Führen Sie GGUF-Modelle mit Mixture of Experts (MoE) lokal über einen Web-Chat und eine HTTP-Inferenz-API aus.

Wie llama.cpp und FreeToken, aber mit nipegunscher Skalpellpräzision gefertigt, um die höchstmögliche Inferenzgeschwindigkeit zu erreichen.

## Screenshots

Es wurden noch keine Screenshots hinzugefügt.

## Architekturen und Modelle

Die GGUF-Architekturen, die `mgs` lädt, jeweils mit ihren Modellen (Details im [Handbuch](MANUAL.de-DE.md#supported-models)):

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

## Bereitstellung

Führen Sie es als `root` aus.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Schnelle Kompilierung

Wählen Sie den Abschnitt, der zu Ihrer Hardware passt. Die Befehle gelten für Debian 13.

### Vulkan

Für GPUs von AMD, Intel und NVIDIA. Als `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Dann als Ihr normaler Benutzer, in einem beliebigen Ordner:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Für NVIDIA-GPUs. Als `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Dann als Ihr normaler Benutzer, in einem beliebigen Ordner:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Für AMD-GPUs, die ROCm unterstützt; für ältere wie die Radeon RX 580 verwenden Sie Vulkan. Debian 13 liefert nur HIP 5.7, der Build benötigt aber 6.1 oder neuer, deshalb installieren diese Befehle ROCm 7.2.4 aus dem AMD-Repository. Als `root`:

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

Fügen Sie Ihren normalen Benutzer den Gruppen `render` und `video` hinzu und melden Sie sich erneut an, damit der Build die GPU erkennen kann.

Dann als Ihr normaler Benutzer, in einem beliebigen Ordner:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### Nur CPU

Für Rechner ohne GPU. Als `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Dann als Ihr normaler Benutzer, in einem beliebigen Ordner:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Alle Optionen und Voraussetzungen des Builds stehen im [Handbuch](MANUAL.de-DE.md#local-build).

## Dokumentation lesen

Entdecken Sie:

- [doc/MANUAL.de-DE.md](MANUAL.de-DE.md), um es zu kompilieren, zu installieren und zu verwenden, Build-Fehler zu diagnostizieren, eine der 15 Sprachen der Oberfläche zu wählen und optional Antworten in derselben regionalen Sprachvariante anzufordern.
- [doc/CODE.de-DE.md](CODE.de-DE.md), um die MoE-Engine, die Speicherplatzierung von MTP, die Struktur des Quellcodes und die Build-Verträge von Backend und Frontend zu verstehen (für Entwickler und KI-Modelle).

## Dieses Projekt unterstützen

- **[Beauftragen Sie mich](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**, um neue Funktionen hinzuzufügen.
- **GitHub Sponsors.** Sie können das Projekt auf GitHub auch finanziell unterstützen: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
