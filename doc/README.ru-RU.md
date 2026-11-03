# moe-gguf-server

Этот README также доступен на следующих языках:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), **ru-RU**, [zh-CN](README.zh-CN.md)

## Что это такое

Запускайте GGUF-модели Mixture of Experts (MoE) локально через веб-чат и HTTP API для инференса.

Как llama.cpp и FreeToken, но создан с нипегуновской точностью скальпеля, чтобы достичь максимально возможной скорости инференса.

## Скриншоты

Скриншоты пока не добавлены.

## Архитектуры и модели

Архитектуры GGUF, которые загружает `mgs`, и модели каждой из них (подробности в [руководстве](MANUAL.ru-RU.md#supported-models)):

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

## Развёртывание

Выполняйте от имени `root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## Быстрая сборка

Выберите раздел, который подходит к вашему оборудованию. Команды предназначены для Debian 13.

### Vulkan

Для GPU AMD, Intel и NVIDIA. От имени `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Затем от имени обычного пользователя, из любой папки:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

Для GPU NVIDIA. От имени `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

Затем от имени обычного пользователя, из любой папки:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

Для GPU AMD, которые поддерживает ROCm; для более старых, таких как Radeon RX 580, используйте Vulkan. В Debian 13 есть только HIP 5.7, а для сборки нужна версия 6.1 или новее, поэтому эти команды устанавливают ROCm 7.2.4 из репозитория AMD. От имени `root`:

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

Добавьте обычного пользователя в группы `render` и `video` и войдите в систему заново, чтобы сборка могла обнаружить GPU.

Затем от имени обычного пользователя, из любой папки:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### Только CPU

Для компьютеров без GPU. От имени `root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Затем от имени обычного пользователя, из любой папки:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- Все параметры и требования сборки описаны в [руководстве](MANUAL.ru-RU.md#local-build).

## Читайте документацию

Изучите:

- [doc/MANUAL.ru-RU.md](MANUAL.ru-RU.md), чтобы собрать, установить и использовать проект, диагностировать ошибки сборки, выбрать один из 15 языков интерфейса и при желании запрашивать ответы на том же региональном варианте языка.
- [doc/CODE.ru-RU.md](CODE.ru-RU.md), чтобы разобраться в движке MoE, размещении MTP в памяти, структуре исходного кода и контрактах сборки backend и frontend (для разработчиков и моделей ИИ).

## Поддержать проект

- **[Наймите меня](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**, чтобы добавить новые функции.
- **GitHub Sponsors.** Вы также можете финансово поддержать проект на GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
