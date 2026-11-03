# moe-gguf-server

この README は次の言語でも読めます：\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), **ja-JP**, [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## 概要

GGUF 形式の Mixture of Experts（MoE）モデルを、Web チャットと HTTP 推論 API を通じてローカルで実行します。

llama.cpp や FreeToken と同様のものですが、可能な限り最速の推論速度を実現するため、nipegun 流のメスのような精度で作り込まれています。

## スクリーンショット

スクリーンショットはまだ追加されていません。

## アーキテクチャとモデル

`mgs` が読み込む GGUF アーキテクチャと、それぞれのモデル（詳細は[マニュアル](MANUAL.ja-JP.md#supported-models)を参照）：

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

## デプロイ

`root` として実行してください。

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## クイックビルド

お使いのハードウェアに合うセクションを選んでください。コマンドは Debian 13 用です。

### Vulkan

AMD、Intel、NVIDIA の GPU 向けです。`root` で実行します：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

次に、通常のユーザーで、任意のフォルダーから実行します：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

NVIDIA の GPU 向けです。`root` で実行します：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

次に、通常のユーザーで、任意のフォルダーから実行します：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

ROCm が対応している AMD の GPU 向けです。Radeon RX 580 などの古い GPU には Vulkan を使ってください。Debian 13 には HIP 5.7 しかなく、ビルドには 6.1 以降が必要なため、これらのコマンドは AMD のリポジトリから ROCm 7.2.4 をインストールします。`root` で実行します：

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

ビルドが GPU を検出できるように、通常のユーザーを `render` と `video` グループに追加し、ログインし直してください。

次に、通常のユーザーで、任意のフォルダーから実行します：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### CPU のみ

GPU のないコンピューター向けです。`root` で実行します：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

次に、通常のユーザーで、任意のフォルダーから実行します：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- ビルドのすべてのオプションと要件は[マニュアル](MANUAL.ja-JP.md#local-build)に記載されています。

## ドキュメントを読む

参照先：

- [doc/MANUAL.ja-JP.md](MANUAL.ja-JP.md)：ビルド、インストール、使い方、ビルド失敗の診断、15 種類のインターフェース言語からの選択、および必要に応じて同じ地域の言語バリエーションでの応答の要求について説明しています。
- [doc/CODE.ja-JP.md](CODE.ja-JP.md)：MoE エンジン、MTP のメモリ配置、ソースの構成、バックエンドとフロントエンドのビルド上の取り決めを理解するためのドキュメントです（開発者および AI モデル向け）。

## このプロジェクトを支援する

- 新機能の追加をご希望の場合は、**[私に依頼してください](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**。
- **GitHub Sponsors。** GitHub でプロジェクトを金銭的に支援することもできます：[github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)。
