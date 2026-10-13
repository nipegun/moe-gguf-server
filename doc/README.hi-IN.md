# moe-gguf-server

यह README इन भाषाओं में भी उपलब्ध है:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), **hi-IN**, [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## यह क्या है

वेब चैट और HTTP इन्फ़रेंस API के ज़रिए GGUF Mixture of Experts (MoE) मॉडल अपने ही कंप्यूटर पर चलाएँ।

llama.cpp और FreeToken जैसा ही, लेकिन सबसे तेज़ संभव इन्फ़रेंस गति पाने के लिए nipegun की नश्तर जैसी सटीकता से गढ़ा गया।

## स्क्रीनशॉट

अभी तक कोई स्क्रीनशॉट नहीं जोड़ा गया है।

## परिनियोजन

इसे `root` के रूप में चलाएँ।

**Debian** (पहले से इंस्टॉल किए गए CUDA Toolkit के साथ; इंस्टॉलर NCCL को उसी CUDA संस्करण के अनुरूप समायोजित करता है):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## बिल्ड

पहले सभी निर्भरताएँ इंस्टॉल करें:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

फिर अपने सामान्य उपयोगकर्ता के रूप में, किसी भी फ़ोल्डर से चलाएँ:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- [बिल्ड निर्भरताएँ](MANUAL.hi-IN.md#local-build) पहले से इंस्टॉल होनी चाहिए।

## दस्तावेज़ पढ़ें

देखें:

- [doc/MANUAL.hi-IN.md](MANUAL.hi-IN.md): इसे बिल्ड, इंस्टॉल और उपयोग करने, बिल्ड की विफलताओं का निदान करने, इंटरफ़ेस की 15 भाषाओं में से चुनने और चाहें तो उत्तर उसी क्षेत्रीय भाषा-रूप में माँगने के लिए।
- [doc/CODE.hi-IN.md](CODE.hi-IN.md): MoE इंजन, MTP की मेमोरी में प्लेसमेंट, सोर्स कोड की संरचना और backend/frontend के बिल्ड अनुबंधों को समझने के लिए (डेवलपर्स और AI मॉडलों के लिए)।

## इस प्रोजेक्ट को प्रायोजित करें

- नई सुविधाएँ जुड़वाने के लिए **[मुझे काम पर रखें](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**।
- **GitHub Sponsors.** आप GitHub पर भी इस प्रोजेक्ट को आर्थिक सहायता दे सकते हैं: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)।
