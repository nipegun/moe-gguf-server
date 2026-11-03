# moe-gguf-server

קובץ README זה זמין גם בשפות:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), **he-IL**, [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## מה זה

הרצת מודלי GGUF מסוג Mixture of Experts (MoE) באופן מקומי, דרך צ'אט אינטרנטי ו-API הסקה מבוסס HTTP.

כמו llama.cpp ו-FreeToken, אך מלוטש בדיוק של אזמל ניפגוני כדי להשיג את מהירות ההסקה הגבוהה ביותר האפשרית.

## צילומי מסך

עדיין לא נוספו צילומי מסך.

## ארכיטקטורות ומודלים

ארכיטקטורות ה-GGUF ש-`mgs` טוען, כל אחת עם המודלים שלה (פרטים ב[מדריך](MANUAL.he-IL.md#supported-models)):

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

## פריסה

הרץ אותו כ-`root`.

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## בנייה מהירה

בחר את הסעיף שמתאים לחומרה שלך. הפקודות מיועדות ל-Debian 13.

### Vulkan

עבור GPU של AMD, Intel ו-NVIDIA. כ-`root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

לאחר מכן, כמשתמש הרגיל שלך, מכל תיקייה:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

עבור GPU של NVIDIA. כ-`root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

לאחר מכן, כמשתמש הרגיל שלך, מכל תיקייה:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

עבור GPU של AMD שנתמכים על ידי ROCm; לדגמים ישנים יותר, כמו Radeon RX 580, השתמש ב-Vulkan. Debian 13 כולל רק את HIP 5.7 והבנייה דורשת 6.1 ומעלה, ולכן הפקודות האלה מתקינות את ROCm 7.2.4 מהמאגר של AMD. כ-`root`:

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

הוסף את המשתמש הרגיל שלך לקבוצות `render` ו-`video` והתחבר מחדש, כדי שהבנייה תוכל לזהות את ה-GPU.

לאחר מכן, כמשתמש הרגיל שלך, מכל תיקייה:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### CPU בלבד

עבור מחשבים ללא GPU. כ-`root`:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

לאחר מכן, כמשתמש הרגיל שלך, מכל תיקייה:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- כל האפשרויות והדרישות של הבנייה נמצאות ב[מדריך](MANUAL.he-IL.md#local-build).

## קריאת התיעוד

עיין ב:

- [doc/MANUAL.he-IL.md](MANUAL.he-IL.md) כדי לבנות, להתקין ולהשתמש בו, לאבחן כשלי בנייה, לבחור מבין 15 שפות ממשק ולבקש, אם תרצה, תשובות באותה גרסה אזורית של השפה.
- [doc/CODE.he-IL.md](CODE.he-IL.md) כדי להבין את מנוע ה-MoE, את מיקום MTP בזיכרון, את מבנה קוד המקור ואת חוזי הבנייה של ה-backend וה-frontend (למפתחים ולמודלי בינה מלאכותית).

## חסות לפרויקט

- **[שכור אותי](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** כדי להוסיף תכונות חדשות.
- **GitHub Sponsors.** אפשר גם לתמוך בפרויקט כספית ב-GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
