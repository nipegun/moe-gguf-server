# moe-gguf-server

קובץ README זה זמין גם בשפות:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), **he-IL**, [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## מה זה

הרצת מודלי GGUF מסוג Mixture of Experts (MoE) באופן מקומי, דרך צ'אט אינטרנטי ו-API הסקה מבוסס HTTP.

כמו llama.cpp ו-FreeToken, אך מלוטש בדיוק של אזמל ניפגוני כדי להשיג את מהירות ההסקה הגבוהה ביותר האפשרית.

## צילומי מסך

עדיין לא נוספו צילומי מסך.

## פריסה

הרץ אותו כ-`root`.

**Debian** (עם CUDA Toolkit שכבר מותקן; תוכנית ההתקנה מתאימה את NCCL לגרסת CUDA זו):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## בנייה

התקן תחילה את כל התלויות:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

לאחר מכן הרץ כמשתמש הרגיל שלך, מכל תיקייה:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- [תלויות הבנייה](MANUAL.he-IL.md#local-build) חייבות להיות מותקנות מראש.

## קריאת התיעוד

עיין ב:

- [doc/MANUAL.he-IL.md](MANUAL.he-IL.md) כדי לבנות, להתקין ולהשתמש בו, לאבחן כשלי בנייה, לבחור מבין 15 שפות ממשק ולבקש, אם תרצה, תשובות באותה גרסה אזורית של השפה.
- [doc/CODE.he-IL.md](CODE.he-IL.md) כדי להבין את מנוע ה-MoE, את מיקום MTP בזיכרון, את מבנה קוד המקור ואת חוזי הבנייה של ה-backend וה-frontend (למפתחים ולמודלי בינה מלאכותית).

## חסות לפרויקט

- **[שכור אותי](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)** כדי להוסיף תכונות חדשות.
- **GitHub Sponsors.** אפשר גם לתמוך בפרויקט כספית ב-GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
