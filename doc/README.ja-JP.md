# moe-gguf-server

この README は次の言語でも読めます：\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), **ja-JP**, [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## 概要

GGUF 形式の Mixture of Experts（MoE）モデルを、Web チャットと HTTP 推論 API を通じてローカルで実行します。

llama.cpp や FreeToken と同様のものですが、可能な限り最速の推論速度を実現するため、nipegun 流のメスのような精度で作り込まれています。

## スクリーンショット

スクリーンショットはまだ追加されていません。

## デプロイ

`root` として実行してください。

**Debian**（CUDA Toolkit がすでにインストールされていること。インストーラーは NCCL をその CUDA のバージョンに合わせて調整します）：

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## ビルド

まず、すべての依存関係をインストールします：

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
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- [ビルドの依存関係](MANUAL.ja-JP.md#local-build)がすでにインストールされている必要があります。

## ドキュメントを読む

参照先：

- [doc/MANUAL.ja-JP.md](MANUAL.ja-JP.md)：ビルド、インストール、使い方、ビルド失敗の診断、15 種類のインターフェース言語からの選択、および必要に応じて同じ地域の言語バリエーションでの応答の要求について説明しています。
- [doc/CODE.ja-JP.md](CODE.ja-JP.md)：MoE エンジン、MTP のメモリ配置、ソースの構成、バックエンドとフロントエンドのビルド上の取り決めを理解するためのドキュメントです（開発者および AI モデル向け）。

## このプロジェクトを支援する

- 新機能の追加をご希望の場合は、**[私に依頼してください](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**。
- **GitHub Sponsors。** GitHub でプロジェクトを金銭的に支援することもできます：[github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)。
