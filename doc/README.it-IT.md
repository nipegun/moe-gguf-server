# moe-gguf-server

Questo README è disponibile anche in:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), **it-IT**, [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Che cos'è

Esegui in locale modelli GGUF Mixture of Experts (MoE) tramite una chat web e un'API HTTP di inferenza.

Come llama.cpp e FreeToken, ma realizzato con una precisione chirurgica nipeguniana per raggiungere la massima velocità di inferenza possibile.

## Screenshot

Non sono ancora stati aggiunti screenshot.

## Architetture e modelli

Le architetture GGUF che `mgs` carica, ciascuna con i suoi modelli (dettagli nel [manuale](MANUAL.it-IT.md#supported-models)):

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

## Distribuzione

Eseguilo come `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilazione rapida

Scegli la sezione adatta al tuo hardware. I comandi sono per Debian 13.

### Vulkan

Per GPU AMD, Intel e NVIDIA. Come `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Poi, con il tuo utente abituale, da qualsiasi cartella:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Per GPU NVIDIA. Come `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Poi, con il tuo utente abituale, da qualsiasi cartella:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Per GPU AMD supportate da ROCm; per quelle più vecchie, come la Radeon RX 580, usa Vulkan. Debian 13 include solo HIP 5.7 e la compilazione richiede la 6.1 o successiva, quindi questi comandi installano ROCm 7.2.4 dal repository di AMD. Come `root`:

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

Aggiungi il tuo utente abituale ai gruppi `render` e `video` ed effettua di nuovo l'accesso, così la compilazione può rilevare la GPU.

Poi, con il tuo utente abituale, da qualsiasi cartella:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### Solo CPU

Per computer senza GPU. Come `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Poi, con il tuo utente abituale, da qualsiasi cartella:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Tutte le opzioni e i requisiti della compilazione sono nel [manuale](MANUAL.it-IT.md#local-build).

## Leggi la documentazione

Esplora:

- [doc/MANUAL.it-IT.md](MANUAL.it-IT.md) per compilarlo, installarlo e usarlo, diagnosticare gli errori di compilazione, scegliere tra 15 lingue dell'interfaccia e, facoltativamente, chiedere risposte nella stessa variante regionale.
- [doc/CODE.it-IT.md](CODE.it-IT.md) per comprendere il motore MoE, il posizionamento in memoria di MTP, la struttura del codice sorgente e i contratti di compilazione del backend e del frontend (per sviluppatori e modelli di IA).

## Sponsorizza questo progetto

- **[Assumimi](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** per aggiungere nuove funzionalità.
- **GitHub Sponsors.** Puoi anche sostenere economicamente il progetto su GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
