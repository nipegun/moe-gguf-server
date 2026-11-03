# moe-gguf-server

यह README इन भाषाओं में भी उपलब्ध है:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), **hi-IN**, [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## यह क्या है

वेब चैट और HTTP इन्फ़रेंस API के ज़रिए GGUF Mixture of Experts (MoE) मॉडल अपने ही कंप्यूटर पर चलाएँ।

llama.cpp और FreeToken जैसा ही, लेकिन सबसे तेज़ संभव इन्फ़रेंस गति पाने के लिए nipegun की नश्तर जैसी सटीकता से गढ़ा गया।

## स्क्रीनशॉट

अभी तक कोई स्क्रीनशॉट नहीं जोड़ा गया है।

## आर्किटेक्चर और मॉडल

`mgs` जिन GGUF आर्किटेक्चर को लोड करता है, हर एक अपने मॉडलों के साथ (विवरण [मैनुअल](MANUAL.hi-IN.md#supported-models) में):

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

## परिनियोजन

इसे `root` के रूप में चलाएँ।

**Debian**

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## त्वरित बिल्ड

अपने हार्डवेयर के अनुसार अनुभाग चुनें। ये कमांड Debian 13 के लिए हैं।

### Vulkan

AMD, Intel और NVIDIA GPU के लिए। `root` के रूप में:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
apt-get install -y libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

फिर अपने सामान्य उपयोगकर्ता के रूप में, किसी भी फ़ोल्डर से:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

### CUDA

NVIDIA GPU के लिए। `root` के रूप में:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

फिर अपने सामान्य उपयोगकर्ता के रूप में, किसी भी फ़ोल्डर से:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cuda
```

### ROCm

ROCm द्वारा समर्थित AMD GPU के लिए; Radeon RX 580 जैसे पुराने GPU के लिए Vulkan का उपयोग करें। Debian 13 में केवल HIP 5.7 है और बिल्ड को 6.1 या उससे नया संस्करण चाहिए, इसलिए ये कमांड AMD के रिपॉज़िटरी से ROCm 7.2.4 इंस्टॉल करते हैं। `root` के रूप में:

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

अपने सामान्य उपयोगकर्ता को `render` और `video` समूहों में जोड़ें और फिर से लॉग इन करें, ताकि बिल्ड GPU का पता लगा सके।

फिर अपने सामान्य उपयोगकर्ता के रूप में, किसी भी फ़ोल्डर से:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

### केवल CPU

बिना GPU वाले कंप्यूटरों के लिए। `root` के रूप में:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

फिर अपने सामान्य उपयोगकर्ता के रूप में, किसी भी फ़ोल्डर से:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu
```

- बिल्ड के सभी विकल्प और आवश्यकताएँ [मैनुअल](MANUAL.hi-IN.md#local-build) में हैं।

## दस्तावेज़ पढ़ें

देखें:

- [doc/MANUAL.hi-IN.md](MANUAL.hi-IN.md): इसे बिल्ड, इंस्टॉल और उपयोग करने, बिल्ड की विफलताओं का निदान करने, इंटरफ़ेस की 15 भाषाओं में से चुनने और चाहें तो उत्तर उसी क्षेत्रीय भाषा-रूप में माँगने के लिए।
- [doc/CODE.hi-IN.md](CODE.hi-IN.md): MoE इंजन, MTP की मेमोरी में प्लेसमेंट, सोर्स कोड की संरचना और backend/frontend के बिल्ड अनुबंधों को समझने के लिए (डेवलपर्स और AI मॉडलों के लिए)।

## इस प्रोजेक्ट को प्रायोजित करें

- नई सुविधाएँ जुड़वाने के लिए **[मुझे काम पर रखें](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**।
- **GitHub Sponsors.** आप GitHub पर भी इस प्रोजेक्ट को आर्थिक सहायता दे सकते हैं: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)।
