# moe-gguf-server

이 README는 다음 언어로도 제공됩니다:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), **ko-KR**, [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## 소개

웹 채팅과 HTTP 추론 API를 통해 GGUF Mixture of Experts(MoE) 모델을 로컬에서 실행합니다.

llama.cpp나 FreeToken과 비슷하지만, 가능한 한 가장 빠른 추론 속도를 내기 위해 nipegun 특유의 메스 같은 정밀함으로 다듬었습니다.

## 스크린샷

아직 스크린샷이 추가되지 않았습니다.

## 배포

`root`로 실행하세요.

**Debian**(CUDA Toolkit이 이미 설치되어 있어야 하며, 설치 프로그램이 그 CUDA 버전에 맞게 NCCL을 조정합니다):

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

## 빌드

먼저 모든 의존성을 설치하세요:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit libnccl2 libnccl-dev
```

그런 다음 일반 사용자로, 아무 폴더에서나 실행하세요:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

- [빌드 의존성](MANUAL.ko-KR.md#local-build)이 미리 설치되어 있어야 합니다.

## 문서 읽기

살펴보기:

- [doc/MANUAL.ko-KR.md](MANUAL.ko-KR.md): 빌드, 설치, 사용 방법과 빌드 실패 진단, 15가지 인터페이스 언어 선택, 그리고 필요에 따라 같은 지역 변형으로 응답을 요청하는 방법을 다룹니다.
- [doc/CODE.ko-KR.md](CODE.ko-KR.md): MoE 엔진, MTP 메모리 배치, 소스 구조, 백엔드/프런트엔드 빌드 규약을 이해할 수 있습니다(개발자 및 AI 모델용).

## 이 프로젝트 후원하기

- 새로운 기능을 추가하려면 **[저를 고용하세요](mailto:nipegun@gmail.com?subject=About%20the%20moe-gguf-server%20repo)**.
- **GitHub Sponsors.** GitHub에서 프로젝트를 금전적으로 후원할 수도 있습니다: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
