# moe-gguf-server

Questo README è disponibile anche in:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), **it-IT**, [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Che cos'è

Esegui in locale modelli GGUF Mixture of Experts (MoE) tramite una chat web e un'API HTTP di inferenza.

Come llama.cpp e FreeToken, ma realizzato con una precisione chirurgica nipeguniana per raggiungere la massima velocità di inferenza possibile.

## Screenshot

Non sono ancora stati aggiunti screenshot.

## Distribuzione

Eseguilo come `root`.

**Debian** (con un CUDA Toolkit già installato; il programma di installazione adegua NCCL a quella versione di CUDA):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Compilazione

Installa prima tutte le dipendenze:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Poi eseguilo con il tuo utente abituale, da qualsiasi cartella:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- Le [dipendenze di compilazione](MANUAL.it-IT.md#local-build) devono essere già installate.

## Leggi la documentazione

Esplora:

- [doc/MANUAL.it-IT.md](MANUAL.it-IT.md) per compilarlo, installarlo e usarlo, diagnosticare gli errori di compilazione, scegliere tra 15 lingue dell'interfaccia e, facoltativamente, chiedere risposte nella stessa variante regionale.
- [doc/CODE.it-IT.md](CODE.it-IT.md) per comprendere il motore MoE, il posizionamento in memoria di MTP, la struttura del codice sorgente e i contratti di compilazione del backend e del frontend (per sviluppatori e modelli di IA).

## Sponsorizza questo progetto

- **[Assumimi](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** per aggiungere nuove funzionalità.
- **GitHub Sponsors.** Puoi anche sostenere economicamente il progetto su GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
