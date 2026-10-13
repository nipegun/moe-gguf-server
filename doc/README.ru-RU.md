# moe-gguf-server

Этот README также доступен на следующих языках:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), **ru-RU**, [zh-CN](README.zh-CN.md)

## Что это такое

Запускайте GGUF-модели Mixture of Experts (MoE) локально через веб-чат и HTTP API для инференса.

Как llama.cpp и FreeToken, но создан с нипегуновской точностью скальпеля, чтобы достичь максимально возможной скорости инференса.

## Скриншоты

Скриншоты пока не добавлены.

## Развёртывание

Выполняйте от имени `root`.

**Debian** (с уже установленным CUDA Toolkit; установщик подбирает NCCL под эту версию CUDA):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Сборка

Сначала установите все зависимости:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Затем выполните от имени обычного пользователя, из любой папки:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- [Зависимости для сборки](MANUAL.ru-RU.md#local-build) должны быть уже установлены.

## Читайте документацию

Изучите:

- [doc/MANUAL.ru-RU.md](MANUAL.ru-RU.md), чтобы собрать, установить и использовать проект, диагностировать ошибки сборки, выбрать один из 15 языков интерфейса и при желании запрашивать ответы на том же региональном варианте языка.
- [doc/CODE.ru-RU.md](CODE.ru-RU.md), чтобы разобраться в движке MoE, размещении MTP в памяти, структуре исходного кода и контрактах сборки backend и frontend (для разработчиков и моделей ИИ).

## Поддержать проект

- **[Наймите меня](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**, чтобы добавить новые функции.
- **GitHub Sponsors.** Вы также можете финансово поддержать проект на GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
