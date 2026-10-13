# 사용자 매뉴얼

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

목차: [설치](#installation), [로컬 빌드](#local-build), [배포](#deployment),
[채팅](#chat), [모니터링](#monitor), [모델과 도구](#models), [지원 모델](#supported-models), [API](#api),
[요청 대기열](#request-queue), [보안 제한](#security-limits), [Vulkan](#vulkan), [성능](#performance), [제거된 호환성](#removed-compatibility),
[유지 관리](#maintenance), [라이선스](#license).

<a id="installation"></a>
## 설치 및 첫 실행

프로덕션 호스트에서 root로 설치하세요.
Debian/Ubuntu에서는 설치 프로그램을 표준 입력으로 바로 받아 실행할 수 있습니다:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

주변 파일을 미리 내려받을 필요는 없습니다.
진입 스크립트는 [이 저장소](https://github.com/nipegun/moe-gguf-server/)에서 `main` 소스 아카이브 전체를 내려받아 검사한 뒤, `/opt/moe-gguf-server-source`에 있는 관리용 사본을 동기화합니다.
다운로드와 빌드 캐시는 프로젝트의 `_/temp/` 아래에 보관됩니다.
진입 스크립트를 가져오려면 curl과 CA 인증서가 필요하며, 그 밖에 빠진 부트스트랩 의존성은 이후 자동으로 설치됩니다.

다른 소스 디렉터리를 선택하고 빌드 옵션을 전달하려면 다음과 같이 합니다:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

원격 모드에서 사용하는 디렉터리는 절대 경로여야 하며 로컬 디스크에 있어야 합니다.
설치 프로그램은 빈 디렉터리를 새로 사용하거나 소스 출처 표식이 있는 사본만 재사용하며, 관련 없는 디렉터리는 거부합니다.
잠금을 통해 그 사본에 동시에 설치하는 것을 막습니다.
이후의 `curl | bash` 실행은 `_/`와 빌드 캐시를 보존하면서, `main`에서 제거된 파일까지 포함해 소스 파일을 동기화합니다.
API 키, 모델, 배포된 서비스는 그 소스 트리 밖에 있으므로 변경되지 않습니다.
다운로드나 빌드가 실패하면 새 실행 파일은 설치되지 않으며, 다시 시도하려면 같은 명령을 다시 실행하면 됩니다.
Git 작업은 전혀 필요하지 않습니다.

따로 내려받은 단독 스크립트도 같은 원격 모드를 사용합니다.
완전한 로컬 사본에서 `bash deploy/install-update-reinstall-debian.sh`를 실행하면 소스를 내려받지 않고 그 사본을 빌드합니다.
`bash -s --` 뒤의 인수는 빌드 설치 프로그램으로 전달됩니다.
`--help`에는 root 권한도 소스 다운로드도 필요하지 않습니다.
`--no-system-deps`는 부트스트랩 패키지 설치도 비활성화하므로, curl, CA 인증서, tar, gzip, rsync, util-linux, coreutils가 이미 있어야 합니다.
설치 프로그램의 하위 프로세스는 파이프에서 스크립트를 읽어 갈 수 없으며, 모든 단계가 하나의 `/root/webapp-install.log` 출력 스트림을 공유합니다.

웹 구성은 여전히 별도의 단계이며, `/opt/moe-gguf-server-source/deploy/configure-web.sh` 또는 선택한 소스 디렉터리에 있는 같은 스크립트를 사용합니다.

Alpine은 여전히 완전한 로컬 사본과 `bash deploy/install-update-reinstall-alpine.sh`를 사용하며, 먼저 Bash를 설치해야 합니다.
Debian의 기본값은 CUDA/NCCL이고, Alpine의 기본값은 CPU이며 CUDA/NCCL 설치 경로를 지원하지 않습니다.
Debian에서 CPU 빌드를 하려면 `--cpu`를, Vulkan을 통해 AMD, Intel 또는 NVIDIA GPU를 사용하려면 `--vulkan`을 지정하세요([Vulkan](#vulkan)).
CUDA Toolkit과 NVIDIA 드라이버는 이미 설치되어 있어야 합니다.
`--no-nccl`은 NCCL을 명시적으로 비활성화하고, `--no-system-deps`는 패키지 관리를 관리자에게 맡깁니다.

로컬 UI에는 Node.js 20.x 계열이면 20.19 이상, 그렇지 않으면 22.12 이상이 필요합니다.
`--no-ui`를 지정하면 UI를 생략합니다.
그 밖의 옵션으로는 `--jobs N`, `--prefix PATH`, `--build-dir PATH`, `--cuda-architectures LIST`, `--no-openssl`, `--build-ui`가 있습니다.
`MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`, `MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND`, `CUDACXX`로 이에 대응하는 값을 환경 변수에서 재정의할 수 있습니다.
빌드 디렉터리와 스테이징 디렉터리는 서로 다르고 겹치지 않아야 하며, 둘 다 프로젝트 `_/temp/` 디렉터리의 하위 디렉터리여야 합니다.

설치 프로그램은 필요하면 일치하는 NCCL 패키지와 NVIDIA 저장소를 추가하지만, CUDA Toolkit이나 GPU 드라이버를 설치하거나 업그레이드하지는 않습니다.
내부 라이브러리는 정적으로 링크하며 `/usr/local/bin/mgs`를 설치합니다.
두 설치 프로그램 모두 sudo를 사용하지도, 설치하지도 않습니다.

Debian/Ubuntu에서는 감지된 CUDA 릴리스에 맞춰 선택한 정확히 같은 버전으로 `libnccl2`와 `libnccl-dev`를 함께 설치합니다.
이 APT 명령은 `--allow-downgrades`를 사용하므로, 이미 설치된 더 새로운 NCCL 패키지가 선택된 버전의 설치를 막지 않습니다.
이 플래그는 NCCL 설치 명령에만 전달됩니다.

배포판이 더 오래된 Node.js를 제공한다면 먼저 지원되는 버전을 설치하세요.
UI는 항상 로컬에서 컴파일됩니다. 빌드 과정에서 미리 빌드된 업스트림 자산을 내려받을 수는 없으며, 그런 자산에는 이 프로젝트의 번역과 API 경로가 없습니다.
UI 빌드가 실패하면 설치가 중단됩니다.

완전한 로컬 사본에서 CPU 빌드를 선택하거나 설치 옵션 목록을 보려면 다음을 실행합니다:

```bash
bash deploy/install-update-reinstall-debian.sh --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

Alpine에서는 설치 프로그램을 실행하기 전에 `apk add --no-cache bash`로 Bash를 설치하세요.

단일 모델을 실행하려면 다음과 같이 합니다:

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

다른 옵션은 필요하지 않습니다. `mgs`는 실행되는 머신에서 추론 속도가 가장 빠른 설정을 스스로 고릅니다.
자동 맞춤은 모델을 GPU에 배치하고, 시스템 메모리가 허락하면 MoE 전문가를 RAM에 두어 남는 VRAM이 [MoE 전문가 캐시](#expert-cache)가 되게 합니다. 또한 [마이크로 배치와 컨텍스트 크기](#context-size)도 정합니다.
CPU 스레드 수는 프로세서에 맞춰지며([CPU 스레드](#threads)), GGUF에서 찾은 MTP 레이어는 활성화되어 VRAM에 유지됩니다([MTP](#mtp-vram)).
명시적으로 지정한 옵션(`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`, `--override-tensor`…)은 자동 선택보다 우선합니다.
무엇이 선택되었는지는 시작 로그에 표시됩니다.

시작할 때 출력되는 HTTPS URL을 여세요. 보통 `https://127.0.0.1:11443/`입니다.
큰 모델은 로드하는 데 시간이 걸릴 수 있습니다.
모델 요청이 로드 중이라고 응답하는 동안에도 웹 인터페이스와 API 문서는 계속 사용할 수 있습니다.
`--help`는 상속된 엔진 옵션을 나열합니다.
GGUF 파일, GPU 메모리, 컨텍스트 길이, 모델별 프로젝터는 의도한 작업에 맞아야 합니다.

서버 코드는 모든 빌드 및 설치 방법에서 기본적으로 HTTPS를 활성화합니다.
11443부터 처음으로 사용 가능한 HTTPS 포트에 바인딩한 다음, 선택된 HTTPS 포트를 건너뛰며 11080부터 HTTP 리디렉션 포트를 바인딩합니다.
HTTP는 요청된 호스트, 경로, 쿼리를 유지한 채 실제 HTTPS 포트로 가는 영구 308 리디렉션으로 응답합니다.
로그에는 선택된 URL이 출력되며, 포트가 충돌하면 각 순서가 서로 독립적으로 다음 포트로 넘어갑니다.
`--port N`은 주 포트를 고정하며, 그 포트가 사용 중이면 실패합니다. `--port 0`을 지정하면 OS가 포트를 고릅니다.
`--http-port N`은 리디렉션 포트 검색의 시작점을 바꿉니다.
대응하는 환경 변수는 `MGS_ARG_PORT`와 `MGS_ARG_HTTP_PORT`입니다.

인증서 파일이 없으면 실행 파일은 처음 시작할 때 외부 `openssl` 명령 없이 자체 서명 인증서와 개인 키를 만들어 `~/.mgs/certificates/`(`fullchain.pem`과 `privkey.pem`, 키는 본인만 읽을 수 있음)에 저장합니다.
이후의 모든 실행과 모든 새 빌드가 이를 재사용하므로, 브라우저는 신뢰 예외를 유지하고 다시 불러올 필요가 없습니다. 새 쌍은 그 파일이 없거나, 서로 일치하지 않거나, 인증서가 만료된 경우(365일 후)에만 만들어집니다.
인증서는 `localhost`, 루프백 주소, `--host`로 지정한 호스트, 그리고 생성 당시의 머신 이름과 네트워크 주소를 포함합니다.
브라우저 경고를 완전히 없애려면 `~/.mgs/certificates/fullchain.pem`을 신뢰할 수 있는 인증서로 가져오고, 새 인증서를 만들려면 두 파일을 모두 삭제하세요.
그 폴더에 쓸 수 없으면 실행할 때마다 메모리 안에서 따로 인증서를 생성합니다.
클라이언트가 신뢰하는 인증서를 사용하려면 `--ssl-cert-file /path/fullchain.pem`과 `--ssl-key-file
/path/privkey.pem`을 함께 지정하세요. 둘 중 하나가 없거나 쌍이 유효하지 않으면 시작이 중단됩니다.
`--no-openssl`로 만든 빌드는 HTTPS를 제공할 수 없으므로 명시적인 `--http`(또는 `MGS_ARG_HTTPS=false`)가 필요합니다. 일반 HTTP는 11080부터 포트 검색을 시작합니다.
라우터의 자식 프로세스는 비공개 루프백 포트에서 명시적인 HTTP를 사용합니다.
Unix 소켓과 Vertex AI의 `AIP_MODE=PREDICTION`은 HTTP 전송을 그대로 유지합니다.

CMake 구성은 `build/CMakeLists.txt`에 있습니다.
설치 스크립트와 로컬 빌드 스크립트는 모두 소스 사본에 `build/`를 포함하고 이를 자동으로 선택합니다.
설치 프로그램은 기존 캐시가 다른 소스 디렉터리를 가리키면 생성된 CMake 구성을 초기화합니다.
로컬 빌드는 항상 `/tmp/` 아래의 새 디렉터리에서 시작하고 종료할 때 그 디렉터리를 제거합니다.
CMake를 직접 사용하려면 프로젝트 루트에서 `cmake -S build -B _/temp/cmake`를 실행하세요. 일반적인 설치 프로그램과 로컬 빌드 명령은 그대로입니다.
HTTPS는 기본적으로 시스템 OpenSSL을 사용합니다(`MGS_OPENSSL=ON`). CMake를 직접 사용하는 빌드에서 `-DMGS_BUILD_BORINGSSL=ON` 또는 `-DMGS_BUILD_LIBRESSL=ON`을 지정하면 대신 BoringSSL 또는 LibreSSL을 내려받아 빌드하고, 둘 다 없이 `-DMGS_OPENSSL=OFF`를 지정하면 HTTPS 없이 서버를 빌드합니다.

<a id="local-build"></a>
## 수동 실행을 위한 로컬 빌드

`build/build-local.sh`는 사용자의 컴퓨터에서 프로젝트를 컴파일하고, 직접 실행하는 독립형 사본을 해당 사용자용으로 설치합니다.
서비스를 설치하지 않고, 백그라운드에서 아무것도 시작하지 않으며, root나 sudo를 사용하지 않고, 패키지도 설치하지 않습니다.
일반 사용자로, 아무 폴더에서나 실행하세요:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

이 한 줄 명령은 Git 없이 `main` 소스 아카이브를 `/tmp` 아래의 비공개 빌드 폴더로 내려받습니다.
아카이브와 압축을 푼 소스는 종료할 때 다른 임시 빌드 파일과 함께 제거됩니다.
스크립트만 따로 복사한 경우에도 같은 방식으로 동작합니다.
완전한 로컬 프로젝트 사본에서 로컬 소스를 빌드하려면 다음을 실행합니다:

```bash
bash build/build-local.sh
```

이 스크립트를 절대 경로로 실행해도, 스크립트가 들어 있는 완전한 프로젝트를 사용합니다.
`Using local sources: ...` 메시지로 인터페이스 번역을 포함해 어느 폴더가 컴파일되는지 확인할 수 있습니다.
실행이 `Error: Compilation failed.`로 끝나면 새 실행 파일은 설치되지 않은 것이며, 기존 설치는 이전 인터페이스를 그대로 유지합니다.
`ninja: build stopped` 위에 있는 첫 번째 컴파일러 오류를 읽고, 그 원인을 바로잡은 뒤 같은 스크립트를 다시 실행하세요.
빌드와 설치가 성공하면 `mgs`를 다시 시작하고 웹 인터페이스를 새로 고치세요.

`std::vector`에 `mPushBack`이라는 멤버가 없다는 오류나 `subproc.cpp`의 `no declaration matches` 같은 오류는 소스의 이름이 서로 일치하지 않는다는 뜻입니다.
수정된 소스 파일을 사용해 빌드를 다시 실행하세요.
인터페이스 빌드에서도 구성 import가 소스 모듈이 내보내는 이름과 일치해야 하며, 누락된 export는 npm/Vite 단계에서 보고됩니다.

SvelteKit이 `Page options are ignored when router.type === 'hash'`를 보고하면서 `fLoad`를 지목한다면, 채팅 경로는 프레임워크가 요구하는 정확한 이름인 `load`를 export해야 합니다.
수정된 경로는 API 키 검증을 유지합니다.
빌드는 자산을 생성하고 인터페이스를 컴파일하기 전에 `svelte-kit sync`를 실행하며, 동기화가 실패하면 중단됩니다.
`Provisioning UI assets` 단계에서 실패하면 C++ 라이브러리가 이미 컴파일되었더라도 설치되지 않습니다.

한 줄 명령으로 옵션을 전달하려면 `bash -s --`를 사용합니다. 예를 들어 CPU 빌드는 다음과 같습니다:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help`는 소스 아카이브를 내려받거나 컴파일하지 않고 옵션을 보여 줍니다.

결과물은 `~/.local` 접두사 아래에 해당 사용자용으로 설치됩니다:

| 경로 | 내용 |
| --- | --- |
| `~/.local/bin/mgs` | 사용자가 실행하는 프로그램입니다. 자신의 위치를 기준으로 `../lib/mgs/`를 찾아, 번들된 로더를 통해 서버를 시작하는 작은 정적 런처입니다. |
| `~/.local/lib/mgs/` | 서버 바이너리(`mgs`, 웹 인터페이스와 API 문서가 내장됨)와 서버에 필요한 모든 공유 라이브러리입니다. glibc 동적 로더와 C 라이브러리, C++ 런타임, OpenMP, OpenSSL, glibc 이름 서비스 모듈, 그리고 CUDA 빌드에서는 CUDA 런타임, cuBLAS, NCCL이 들어 있습니다. |
| `~/.local/share/doc/mgs/licenses/` | 프로젝트의 PolyForm NonCommercial 1.0.0 라이선스(`LICENSE`)와 소스 트리의 `build/licenses/`에서 복사한 서드파티 라이선스 고지(llama.cpp, FreeToken, Swagger UI, cpp-httplib, nlohmann/json)입니다. |

이 설치는 스크립트를 실행한 사용자만 사용할 수 있습니다.
모든 것은 런처가 있는 위치를 기준으로 결정됩니다. 런처는 자신의 접두사에서 `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs`를 실행하고, 서버와 그 라이브러리는 `LD_LIBRARY_PATH`를 포함한 다른 어떤 경로보다 먼저 `lib/mgs/`를 검색합니다.
여전히 시스템에서 가져오는 유일한 라이브러리는 NVIDIA 드라이버(`libcuda.so.1`과 그것이 로드하는 `libnvidia-*` 라이브러리)입니다. 실행 중인 커널 모듈과 일치해야 하기 때문입니다. `--vulkan`을 사용할 때는 시스템의 Vulkan 드라이버도 마찬가지입니다([Vulkan](#vulkan)).
마지막으로 스크립트는 번들된 로더를 통해 서버가 로드할 라이브러리를 나열하고, 그중 하나라도 `lib/mgs/` 밖에서 오면 실패하며, 런처를 통해 `--version`을 실행합니다.

Debian은 `~/.local/bin` 폴더가 생기면 로그인할 때(`~/.profile`을 통해) 그 폴더를 `PATH`에 추가하므로, 다시 로그인한 뒤에는 이름만으로 `mgs`를 실행할 수 있습니다. 그 폴더가 `PATH`에 없는 동안에는 스크립트가 안내 메시지를 출력합니다.
원할 때 실행하고 Ctrl+C로 중지하세요:

```bash
mgs --model /path/model.gguf
```

전체 경로로도 실행할 수 있습니다: `~/.local/bin/mgs --model /path/model.gguf`.

채팅은 `https://127.0.0.1:11443/`, API 문서는 `https://127.0.0.1:11443/api/doc/`에서 여세요.
이 매뉴얼의 모든 옵션이 적용됩니다. 예를 들어 `--models-dir`은 다중 모델 라우터를 활성화합니다.

다음 요구 사항은 스크립트가 확인만 하므로 root로 직접 설치해야 합니다: `cmake`, C/C++ 컴파일러와 정적 C 라이브러리(`libc6-dev`를 함께 설치하는 `build-essential`), `patchelf`, `libssl-dev`(또는 `--no-openssl`), npm이 포함된 Node.js 20.19+/22.12+(또는 `--no-ui`), 이미 설치된 CUDA Toolkit(또는 `--cpu`), 개발 파일이 포함된 NCCL(`libnccl2`와 `libnccl-dev`, 또는 `--no-nccl`).
`nvcc`는 `CUDACXX`, `PATH` 또는 `/usr/local/cuda/bin/nvcc`에서 가져옵니다. Ninja가 설치되어 있으면 Ninja를 사용합니다.
기본 유틸리티로는 `tar`, `coreutils`, `util-linux`가 필요합니다.
원격 빌드에는 `curl`, `ca-certificates`, `gzip`도 필요하며, 어떤 의존성도 자동으로 설치되지 않습니다.

옵션: `--prefix PATH`(다른 설치 접두사), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`, `--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`, `--help`.

알아 두면 좋은 점:

- 빌드는 이 컴퓨터의 CPU와 컴파일할 때 장착되어 있던 GPU에 맞게 최적화됩니다. 설치본을 복사하지 말고 컴퓨터마다 다시 빌드하세요.
- 설치본은 접두사 단위로 옮길 수 있습니다. `bin/mgs`와 `lib/mgs/`를 `bin/`과 `lib/` 단계를 유지한 채 함께 옮기세요. 항상 `bin/mgs`로 시작하세요. `lib/mgs/mgs`는 내부용이므로 직접 실행하면 안 됩니다. 번들된 로더를 건너뛰게 되기 때문입니다.
- 스크립트를 다시 실행하면 설치가 업데이트됩니다. `bin/mgs`, `lib/mgs/`, `share/doc/mgs/licenses/`만 교체합니다. 이미 실행 중인 사본은 계속 동작하므로, 새 빌드를 사용하려면 다시 시작하세요.
- 스크립트는 `bin/mgs`, `lib/mgs/` 또는 그 라이선스 폴더가 이미 있는 접두사를, 그것들을 스크립트 자신이 만든 경우가 아니면 거부합니다. 스크립트는 `lib/mgs/`에 숨김 표식 `.moe-gguf-server-bundle`을 남깁니다.
- `--prefix PATH`는 사용자가 쓸 수 있는 다른 접두사 아래에 설치합니다. `/`나 프로젝트 안의 폴더는 지정할 수 없습니다.
- 스테이징된 소스, 빌드 파일, 캐시, 조립된 설치본, 로그는 비공개 `/tmp/moe-gguf-server-build.XXXXXXXX/` 폴더를 사용합니다. 결과를 접두사로 복사한 뒤 스크립트는 그 폴더를 제거합니다. 오류가 나거나 처리된 중단이 있을 때에도 정리합니다. 실행할 때마다 소스 트리나 `/opt/`에 쓰지 않고 처음부터 컴파일합니다. 로컬 빌드는 `--build-dir`을 받지 않습니다.
- 완성된 결과물을 복사하고 교체하는 동안 접두사가 잠기므로, 같은 설치를 동시에 업데이트할 수 없습니다.
- CUDA 빌드는 시작할 때 NVIDIA 드라이버가 필요합니다. 다른 GPU가 있는 컴퓨터에서는 `--vulkan`을, GPU가 없는 컴퓨터에서는 `--cpu`를 사용하세요.

<a id="vulkan"></a>
## Vulkan(AMD, Intel, NVIDIA GPU)

CUDA(기본값)와 CPU 외에도 `build/build-local.sh`와 Debian 설치 프로그램은 모두 `--vulkan`을 지정하면 AMD Radeon 카드처럼 CUDA가 없는 GPU를 위한 Vulkan 백엔드를 빌드합니다.
먼저 root로 빌드 의존성을 설치하세요:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

그런 다음 일반 사용자로 실행합니다:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

Debian 설치 프로그램도 같은 옵션을 받으며, 해당 Vulkan 패키지를 스스로 설치합니다:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices`는 GPU를 보여 줍니다. 예: `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- 로컬 설치는 Vulkan 로더(`libvulkan.so.1`)를 다른 라이브러리와 마찬가지로 번들에 포함하지만, Vulkan 드라이버는 시스템의 것을 사용합니다. 로더는 `/usr/share/vulkan/icd.d/`에 등록된 드라이버(AMD용 Mesa RADV, Intel용 ANV, NVIDIA 드라이버)를 읽고, 이 드라이버들은 커널 및 시스템과 일치해야 하므로 자신의 시스템 라이브러리를 로드합니다. 바이너리를 실행할 Debian 릴리스에서 빌드하세요.
- `mgs`를 실행하는 사용자에게는 GPU 렌더 노드(`/dev/dri/renderD*`)에 대한 접근 권한이 필요합니다. 데스크톱에 로그인한 사용자는 이미 권한이 있고, 다른 사용자는 `render` 그룹(및 `video`)에 속해야 합니다. `configure-web.sh`는 서비스 계정을 두 그룹 모두에 추가합니다.
- VRAM 내 MTP, MoE 전문가 캐시, 전문가 프리페치, `--fit`, 모니터링 페이지는 CUDA와 똑같이 동작합니다. AMD에서는 모니터링 페이지가 사용률, VRAM, 온도를 sysfs에서 읽습니다.
- `mgs`는 실제 VRAM(device-local 메모리 힙)만 GPU 메모리로 계산하며, GTT는 절대 포함하지 않습니다. GTT는 AMD나 Intel GPU가 매핑할 수 있는 시스템 RAM의 일부(보통 절반)로, 추가 메모리가 아니며 GPU는 이를 PCIe를 통해 VRAM보다 몇 배나 느리게 읽습니다. MTP 레이어, 밀집 가중치, KV 캐시, 전문가 캐시는 VRAM에 할당되고, 들어가지 않는 전문가는 일반 RAM에 남아 CPU가 계산합니다.
- `mgs`는 커널이 그 메모리를 실제로 어디에 두고 있는지(amdgpu, i915, xe가 제공하는 `/proc/self/fdinfo`) 확인합니다. 다른 프로그램이 VRAM을 채워 드라이버가 그 일부를 GTT로 옮기면 로그와 모니터링 페이지에 그 사실이 표시되며, 모니터링 페이지는 모든 AMD 또는 Intel GPU의 실제 위치(VRAM과 GTT)를 보여 줍니다. 전문가 캐시는 더 낮은 메모리 우선순위(`VK_EXT_memory_priority`)로 할당되므로, 드라이버는 MTP 레이어, 가중치, KV 캐시보다 전문가 캐시를 먼저 옮깁니다. ReBAR(resizable BAR)가 없으면 디바이스 버퍼는 CPU가 매핑할 수 있는 작은 VRAM 창(RX 580에서 256 MiB)도 사용하지 않습니다. 이 창은 데스크톱이 함께 쓰며, 드라이버가 가장 먼저 GTT로 비워 내는 영역입니다.
- RAM에 남는 모델 페이지를 고정(pin)할 수 있는 것은 CUDA뿐이므로, Vulkan에서는 전문가가 페이징 가능한 메모리를 거쳐 GPU에 도달합니다. `--load-mode none`을 지정하면 대신 전문가를 고정된 호스트 메모리로 복사하며, Vulkan을 사용하는 RTX 4060 Ti(Qwen3.6-35B-A3B Q4_K)에서 이 방식으로 전문가 업로드는 2.7 GB/s에서 13.1 GB/s로, 생성 속도는 27.5 tokens/s에서 36.6 tokens/s로 올라갔습니다. AMD와 Intel에서는 그 고정 메모리가 GTT에서 차감됩니다. 전문가가 GTT에 들어가지 않으면 로드가 `radv/amdgpu: Not enough memory for command submission`(`ErrorDeviceLost`)으로 실패하며, GTT가 16.8 GB인 RX 580에서 이 모델의 18.9 GB 전문가가 실제로 그렇게 실패했습니다. 전문가가 GTT와 남는 RAM에 모두 들어갈 때만 사용하세요.
- 모델 크기는 머신에 맞추세요. GGUF 전체가 남는 RAM과 VRAM을 합친 용량에서 KV 캐시와 계산 버퍼를 뺀 공간에 들어가야 합니다. VRAM 8 GB와 RAM 16 GB라면 가중치 약 18–20 GB가 한계이며, 그보다 큰 파일은 디스크에서 계속 다시 읽혀 생성이 매우 느려집니다. 위의 21.6 GB Q4_K는 RAM 15 GB인 RX 580에서 7.8 tokens/s, 같은 머신의 RAM을 31 GB로 늘린 뒤에는 15.9 tokens/s(`--spec-type none`에서는 10.0)에 도달했습니다.

<a id="deployment"></a>
## 프로덕션 배포

바이너리를 설치한 뒤, 소스 디렉터리(원격 설치에서는 `/opt/moe-gguf-server-source`)에서 다음을 실행하세요:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

도메인은 프로덕션 호스트로 해석되어야 합니다.
`moe-gguf-server-web` 서비스는 root 없이 `moe-gguf-server` 계정으로 전용 Apache 인스턴스를 실행합니다.
시작할 때마다 11080부터 처음으로 사용 가능한 HTTP 포트와 11443부터 처음으로 사용 가능한 HTTPS 포트를 선택합니다.
각 순서는 두 프로토콜에 같은 포트를 할당하지 않고 65535까지 서로 독립적으로 증가합니다.
IPv4/IPv6 TCP 수신 소켓을 검사하고, 선택 후에 포트가 사용 중이 된 것을 Apache가 감지하면 다시 시도합니다.
그 밖의 구성 실패는 포트 충돌로 오인되지 않고 시작을 중단시킵니다.

HTTP는 경로와 쿼리를 유지한 채 선택된 HTTPS 포트로 영구 리디렉션됩니다.
이 리디렉션은 캐시되지 않습니다.
Apache는 요청을 `127.0.0.1:8080`의 추론 서비스로 전달하며, 이 서비스는 `--http --port 8080`으로 명시적으로 시작되고 역시 `moe-gguf-server` 계정으로 실행됩니다.
설치에는 root가 필요하지만, 두 서비스 모두 root로 실행할 필요는 없습니다.
HAProxy는 구성되지 않으며, 공유 Apache 사이트를 가져오거나 수정하지도 않습니다.

실제 URL은 두 수신 소켓이 모두 열린 뒤에 공개됩니다:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

예를 들어 HTTP 11081과 HTTPS 11445는 `https://ai.example.com:11445/`로 리디렉션됩니다.
이 파일은 종료할 때 제거되고 시작할 때 다시 생성되며, 다시 시작하면 다른 포트가 선택될 수 있습니다.
클라이언트, 방화벽, 외부 프록시는 공개된 값을 사용해야 합니다.
포트는 [브라우저 출처(origin)](https://developer.mozilla.org/en-US/docs/Glossary/Origin)의 일부이므로, HTTPS 포트가 달라지면 로컬 기록과 설정도 따로 관리됩니다.
기존 대화는 원래 출처에 남아 있으며, 내보내기/가져오기로 옮길 수 있습니다.

시스템 계정에는 대화형 비밀번호가 없습니다.
모델은 `/opt/moe-gguf-server/models`에 저장되고, 프로그램 구성은 서비스 계정의 홈인 `/opt/moe-gguf-server/.mgs/config/`에 있습니다. API 키(`api.key`), 웹 도메인(`web-domain`), Apache 템플릿(`apache2.conf.in`), 인터페이스 설정(`ui-settings.json`)이 있으며, 모두 `moe-gguf-server` 소유이고 모드는 0600입니다.
자격 증명과 설치 출력은 `/root/webapp-credentials.txt`와 `/root/webapp-install.log`에 기록되며, 둘 다 모드 600입니다.
다시 구성해도 기존 키, 모델, 인증서는 보존됩니다.
TLS 개인 키는 root 소유이며 moe-gguf-server 그룹이 읽을 수 있으므로(모드 640), 권한이 없는 웹 프로세스도 이를 읽을 수 있습니다.

직접 준비한 TLS 인증서를 사용하려면 다음과 같이 합니다:

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

인증서는 `/opt/moe-gguf-server/certificates/`(`fullchain.pem`과 `privkey.pem`, `moe-gguf-server` 계정 소유이며 키는 이 계정만 읽을 수 있음)에 있습니다.
파일을 지정하지 않으면 스크립트는 이미 있는 인증서를 유지하거나, 처음 실행할 때 자체 서명 인증서를 만듭니다. 이후 다시 설치할 때마다 이 인증서를 재사용하므로 브라우저는 신뢰 예외를 유지합니다.
클라이언트에서 이 인증서를 신뢰하도록 설정하거나, 클라이언트가 이미 신뢰하는 인증서를 지정하세요.
지정한 파일은 그 폴더로 복사됩니다. 인증서를 갱신할 때는 그 사본을 새로 고치고 `moe-gguf-server-web`을 다시 시작해야 합니다. 스크립트는 Let's Encrypt 인증서를 발급받거나 갱신하지 않습니다.
원본 인증서를 갱신한 뒤에는 같은 명령을 다시 사용하세요.
스크립트의 `--binary` 옵션으로 기본값이 아닌 실행 파일 위치를 지정할 수 있습니다.

서비스는 라우터 모드를 사용합니다.
읽을 수 있는 GGUF 모델을 `/opt/moe-gguf-server/models`에 넣거나, 인터페이스의 모델 관리 기능을 사용하세요.
특수한 명령줄 옵션이 필요하면 Debian/Ubuntu에서는 systemd override를, Alpine에서는 OpenRC 서비스 구성을 사용하세요.
백엔드는 루프백에 바인딩된 상태로 두세요.
웹 구성 도구를 다시 실행하면 자신이 관리하는 서비스 파일을 다시 생성합니다.

포트를 다시 선택하려면 웹 서비스를 다시 시작하세요:

```bash
systemctl restart moe-gguf-server-web
```

Alpine에서는 `rc-service moe-gguf-server-web restart`를 사용합니다.
시작 로그는 Debian에서는 `journalctl -u moe-gguf-server-web`, Alpine에서는 `/var/log/moe-gguf-server-web.log`에 있습니다.
Apache 진단 정보는 `/opt/moe-gguf-server/_/temp/web/startup.log`와 `error.log`에 있고, 도메인 접근 로그는 계속 `/var/www/DOMAIN-logs/`에 남습니다.
독립형 템플릿은 `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`입니다.
`_/temp/web/`에 생성된 구성은 시작할 때마다 다시 만들어지므로 편집하지 마세요.

<a id="chat"></a>
## 인터페이스 사용

1. 서버 URL을 열고, 요청이 표시되면 API 키를 입력하세요. 설정 → 일반에서 입력할 수도 있습니다.
2. 모델을 선택하세요. 라우터 모드에서 로드된 모델이 없으면 사용 가능한 모델을 로드하세요.
3. 메시지를 입력하고 전송하세요. 기본적으로 Enter를 누르면 전송되고 Shift+Enter를 누르면 줄이 바뀝니다. 이 단축키는 설정에서 바꿀 수 있습니다.
4. 첨부 동작으로 파일을 추가하세요. 이미지, 오디오, 동영상을 사용하려면 모델이 해당 모달리티를 지원해야 합니다. PDF 파일은 텍스트로 추출하거나 이미지로 변환할 수 있으며, 비전 기능이 없는 모델은 텍스트 방식을 사용합니다.
5. 메시지 동작을 사용해 생성을 중지하거나, 지원되는 경우 추론을 건너뛰거나, 메시지를 편집하거나, 응답을 다시 생성하거나, 대화를 분기할 수 있습니다.

사이드바에서는 검색, 고정, 이름 바꾸기, 삭제, 일괄 선택을 할 수 있습니다.
대화는 IndexedDB를 사용해 브라우저에 저장되므로, 브라우저 저장소를 지우거나 브라우저 프로필을 바꾸기 전에 내보내세요.
설정은 서버가 `mgs`를 실행하는 사용자의 `~/.mgs/config/ui-settings.json`(배포된 서비스에서는 `/opt/moe-gguf-server/.mgs/config/ui-settings.json`)에 보관하므로, 인터페이스를 여는 모든 브라우저가 같은 설정을 공유합니다. 생성 매개변수, 표시 옵션, 테마, 언어, MCP 서버, 즐겨찾는 모델, 비활성화된 도구와 항상 허용된 도구, 기본 추론 노력이 여기에 포함됩니다.
API 키만은 각 브라우저에 남습니다. 나머지 설정을 불러오는 요청을 인증하는 데 쓰이기 때문이므로, 브라우저마다 한 번씩 입력하세요.
가져오기/내보내기는 설정에서 할 수 있습니다.
내보낸 파일에는 첨부 파일이 포함될 수 있으며, 명시적으로 선택한 경우 민감한 API 키나 MCP 인증 헤더도 포함될 수 있습니다.

언어는 설정 → 일반에서 선택합니다.
15가지 옵션은 `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN` 순서로 정렬되며, 기본값은 `en-US`입니다.
브라질 포르투갈어와 유럽 포르투갈어는 번역이 따로 있습니다.
히브리어는 오른쪽에서 왼쪽으로 쓰는 인터페이스를 사용합니다. 채팅 입력과 일반 사용자 메시지는 텍스트의 방향을 따르고, 코드와 터미널 출력은 왼쪽에서 오른쪽으로 표시됩니다.
언어를 바꾸면 선택한 언어가 서버 설정에 저장되고 인터페이스가 다시 로드됩니다. 서버가 이를 `/api/config.js`에 게시하므로 모든 브라우저가 그 언어로 시작합니다.
대기 중인 다른 설정은 먼저 저장하세요.
URL 쿼리 매개변수 `lang`은 해당 페이지에서 이 설정보다 우선하며, 선택을 저장할 수 없을 때에도 그 선택을 유지합니다.
예를 들어 인터페이스 URL 끝에 `?lang=ja-JP`를 붙이면 일본어가 선택됩니다.

설정 → 일반 → **항상 인터페이스 언어로 답변**은 기본적으로 꺼져 있습니다.
이 옵션을 켜고 설정을 저장하면, 기존 대화, 다시 생성한 답변, 이어 쓰기, 도구 호출 이후의 턴, 모델이 생성한 제목을 포함한 모든 새 답변에서 인터페이스의 언어와 지역 변형을 사용하도록 모델에 지시합니다.
`es-AR`의 경우 지시문은 아르헨티나식 voseo(“Seguí estos pasos”, “tenés”, “podés”)를 명시적으로 요청하고, `es-ES`는 스페인의 용법을 요청합니다.
영국 영어와 미국 영어, 브라질 포르투갈어와 유럽 포르투갈어도 각각 다른 지역별 지시문을 받습니다.

이 지시문은 모든 새 추론 요청과 KV 캐시 미리 채우기에 함께 전송되며, 저장된 시스템 프롬프트나 대화를 수정하지 않고 전송되는 시스템 메시지와 결합됩니다.
인터페이스 언어를 바꾸면 다시 로드한 뒤의 새 요청에 적용되며, 이미 생성 중인 응답은 처음 받은 지시문을 그대로 유지합니다.
설정을 끄면 이후 요청에는 지시문이 추가되지 않습니다.
기존 답변은 번역되지 않습니다.
코드, 명령, 이름, 인용문, 명시적으로 요청한 번역은 필요한 언어를 유지하고, 그 주변의 설명은 인터페이스 언어를 따릅니다.
일관성은 모델이 지시를 따르는 능력에 따라 달라집니다. 이 기능은 자동 번역이 아니며, 지시를 완벽하게 따른다는 보장도 아닙니다.
외부 API 클라이언트는 이 브라우저 설정의 영향을 받지 않습니다.

테마는 시스템, 밝게, 어둡게 중에서 선택합니다.
생성 설정에는 컨텍스트 사용량, 온도(temperature), 샘플러, 페널티, 에이전트 턴 수 제한이 포함됩니다.
일부 기능은 서버 플래그나 선택한 모델에 따라 달라집니다.

응답이 끝날 때마다 인터페이스는 서버가 유휴 상태일 때 대화를 다시 전송합니다. 이렇게 하면 사용자가 응답을 읽는 동안 대화가 이미 KV 캐시에 인코딩되어 있어 다음 턴이 더 빨리 시작됩니다.
기본적으로 켜져 있으며, 중지하려면 설정 → 개발 → **응답 후 KV 캐시 미리 채우기**를 끄세요.

서버가 다른 요청을 처리하는 동안 보낸 메시지는 [대기열](#request-queue)에서 기다리며, 그 응답에는 처리가 시작될 때까지 진행 텍스트 대신 안내가 표시됩니다.
다른 프로그램이 동시에 API를 사용하고 있으면 “대기열에서 대기 중: API로 직접 전송된 다른 요청을 먼저 처리하고 있습니다(앞선 요청: N).”라는 안내가 표시되고, 웹 인터페이스 자체의 요청(다른 탭이나 브라우저, 생성된 제목, KV 캐시 미리 채우기)이 먼저 처리되는 경우에도 같은 형식의 안내가 표시됩니다.
요청이 시작되면 안내는 바로 사라집니다.

<a id="monitor"></a>
## 모니터링 페이지와 상태 표시줄

**모니터링**은 사이드바의 **설정** 아래에 있습니다(주소 `#/monitor`).
열려 있는 동안 2초마다 새로 고쳐지며 다음을 보여 줍니다:

- **사용률**: 먼저 머신의 GPU마다 카드가 하나씩 표시됩니다(NVIDIA, AMD, Intel이며, 어떤 모델도 사용하지 않는 GPU도 포함). 카드 제목은 굵은 글씨의 레이블과 그 이름입니다(예: **gpu-1**: NVIDIA GeForce RTX 4060 Ti, GPU가 여러 개면 gpu-1, gpu-2 등). 카드에는 사용률, 메모리 컨트롤러(대역폭) 사용률, 사용 중인 VRAM, 온도, 한도 대비 소비 전력이 표시됩니다. 그다음 프로세서마다 카드가 하나씩 표시되며, 같은 방식으로 레이블과 모델명을 제목으로 합니다(예: **cpu-1**: AMD Ryzen 9 5950X 16-Core Processor, 프로세서가 여러 개면 cpu-1, cpu-2 등). 이 카드에는 사용률, 그 프로세서의 논리 CPU마다 하나씩 있는 작은 세로 막대, 시스템 평균 부하가 표시됩니다. 모든 사용률 막대는 녹색이며 사용률이 높아질수록 빨간색으로 채워집니다. 값을 알 수 없는 막대는 회색으로 남습니다. NVIDIA GPU는 드라이버의 NVML 라이브러리로, AMD GPU는 sysfs의 `gpu_busy_percent`로 읽습니다. 그 밖의 GPU(Intel, 또는 NVML이 없는 NVIDIA)는 사용률이 표시되지 않으며, 모델이 그 GPU를 사용하기 시작하면 추론 백엔드에서 이름과 메모리 정보를 가져옵니다.
- **메모리**: GPU마다 막대 하나(VRAM)와 RAM용 막대 하나가 있습니다. 아직 어떤 모델도 사용하지 않는 Intel GPU처럼 메모리 수치가 없는 GPU에는 막대가 없습니다. 각 막대는 용도별로 나뉘고 각 용도는 항상 같은 색으로 표시되며, 범례에는 모든 구간이 크기와 전체 대비 비율과 함께 한 줄에 하나씩, 막대와 같은 순서로 나열됩니다. 구간 위에 마우스를 올리면 크기를 볼 수 있습니다.

  | 구간 | 내용 |
  | --- | --- |
  | MTP / 초안 모델 가중치 | 모델의 MTP(nextn) 레이어, `--spec-draft-model`로 전달한 별도 MTP GGUF, 또는 초안 모델의 가중치입니다. |
  | 전문가 가중치 | 모델의 라우팅되는 전문가 텐서입니다. |
  | 밀집 가중치 | 그 밖의 모든 가중치입니다: 어텐션, 공유 전문가, 임베딩, 출력. |
  | KV 캐시 | 모델의 컨텍스트 메모리(KV 캐시 또는 순환 상태)입니다. |
  | MTP / 초안 모델 KV 캐시 | MTP 또는 초안 모델 컨텍스트의 컨텍스트 메모리입니다. |
  | 연산 버퍼 | 계산 그래프의 임시 버퍼입니다. |
  | 전문가 캐시 및 프리페치 | MoE 전문가 캐시(풀, 식별자, 스테이징)와 전문가 프리페치 슬롯입니다. |
  | 프롬프트 캐시 및 체크포인트 | `--cache-ram`으로 저장한 프롬프트와 슬롯의 컨텍스트 체크포인트(RAM)입니다. |
  | 기타 | 위 항목으로 설명되지 않는 사용 메모리입니다: 다른 프로세스, 데스크톱, 드라이버 컨텍스트, 멀티모달 프로젝터. |
  | 여유 | 디바이스 또는 RAM의 나머지 부분입니다. |

  모델 파일에서 매핑된 가중치는 커널이 그 페이지 캐시를 사용 가능하다고 보고하더라도 RAM으로 계산되므로, 구간의 합이 “사용 중”인 메모리보다 클 수 있습니다. 이 경우 막대에는 “기타”가 표시되지 않고 여유 메모리가 더 적게 표시됩니다. 전문가 캐시는 첫 요청 후 생성될 때 나타납니다. 라우터 모드에서는 로드된 모든 모델이 막대에 포함되며, 모니터링 페이지는 모델을 로드하거나 깨우지 않습니다.

모든 페이지 하단의 **상태 표시줄**은 먼저 모든 GPU의 사용률을, 그다음 모든 물리 프로세서의 사용률을 각각 별도의 막대로 보여 주며, 모니터링 페이지의 카드와 같은 레이블을 사용합니다: `gpu-1`, `gpu-2`... 그리고 `cpu-1`, `cpu-2`...
막대는 크기가 같고 2초마다 새로 고쳐지며, 모니터링 페이지의 막대처럼 녹색이고 사용률이 높아질수록 빨간색으로 채워집니다. 각 막대에 마우스를 올리면 디바이스 이름이 표시됩니다.
GPU가 없으면 빈 `GPU` 막대가 표시되고, 좁은 화면에서 막대가 다 들어가지 않으면 상태 표시줄이 가로로 스크롤됩니다.
클릭하면 모니터링 페이지가 열립니다.
모델을 로드하는 동안에도 계속 동작합니다.
기본적으로 표시되며, 숨기려면 설정 → 표시 → **상태 표시줄 표시**를 끄세요.

<a id="models"></a>
## 모델, 도구, MCP

`mgs`는 MoE 주 모델만 받습니다. GPT-2나 Phi-2 같은 밀집(dense) 모델은 가중치를 로드하기 전에 거부되며, 그 디코더 구현은 컴파일되지 않습니다. MoE 모델 안의 밀집 레이어와, 명시적으로 지정한 호환되는 MTP 전용 헤드는 유지됩니다.

<a id="supported-models"></a>
### 지원되는 모델 계열

| 계열 | GGUF 아키텍처 | 비고 |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | GGUF에 포함되어 있으면 MTP를 사용합니다. |
| Qwen3.8-Flash-Next | `qwen4exp` | GGUF에 포함되어 있으면 MTP를 사용합니다. 레이어별 임베딩 테이블(약 29 GB)은 필요할 때마다 모델 파일에서 읽으므로, 파일을 빠른 디스크에 두세요. 긴 컨텍스트는 희소 선택 아래에서 밀집 어텐션으로 계산됩니다. 결과는 정확하지만 희소 커널의 속도 향상은 얻지 못합니다. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP. 전용 `glm5v` mmproj로 이미지를 처리합니다. |
| Kimi-K3 | `kimi-k3` | 공개된 GGUF에는 MTP가 없습니다. Vulkan에서는 잔차 혼합이 CPU에서 실행됩니다. |
| DeepSeek-V4.1-Flash | `deepseek41` | 공개된 두 가지 레이아웃을 모두 로드합니다. antirez의 DwarfStar 파일(예: `antirez/deepseek-v4.1-flash-gguf`)과 llama.cpp 생태계의 파일입니다. Engram 테이블(파일 안에서 약 188 GiB)은 절대 로드되지 않습니다. 모든 토큰이 모델 파일에서 해당 행을 읽으므로 `mgs`가 실행되는 동안 파일에 계속 접근할 수 있어야 하며(가급적 SSD), Linux 또는 다른 POSIX 시스템이 필요합니다. 채팅 템플릿이 없는 파일은 내장 템플릿을 사용하며, `reasoning_effort`는 1부터 100까지의 숫자를 받습니다. MTP와 이미지는 지원하지 않습니다. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

Gemma 4와 V4.1보다 오래된 DeepSeek 모델은 더 이상 지원되지 않습니다([제거된 호환성](#removed-compatibility) 참조).

GPU를 여러 개 선택한 경우, MTP는 RAM으로 물러나기 전에 그 GPU를 모두 확인합니다. MTP의 메모리 추정치에는 출력 헤드가 포함됩니다. 모델에 통합된 MTP 버퍼는 다른 가중치보다 먼저 로드되며, `--spec-draft-model`로 전달한 외부 MTP GGUF는 `--fit off`에서도 주 모델보다 먼저 로드됩니다. 어떤 GPU를 사용할 수 있는지는 여전히 현재의 명시적 디바이스 선택이 결정합니다.

사고(thinking)를 제어하려면 `--reasoning on` 또는 `--reasoning off`를 사용하세요. `--chat-template-kwargs`를 통해 `enable_thinking`을 전달하면 거부됩니다. 문서에 나온 하이픈 형식의 옵션 이름을 사용하세요. 밑줄 표기는 거부됩니다. 목록 값은 옵션을 반복하지 말고 쉼표로 구분한 하나의 값으로 전달하세요. 추측(speculative) 모드를 조합할 수 있도록 `--spec-type`만 반복할 수 있습니다.

모델 선택기는 사용 가능한 모델, 로드된 모델, 즐겨찾는 모델을 보여 줍니다.
모델 세부 정보에는 컨텍스트 크기, 모달리티, 양자화, 채팅 템플릿이 포함됩니다.
라우터 모드는 로드된 모델마다 별도의 자식 프로세스를 관리하며, 요청은 `model` 필드나 쿼리 매개변수로 모델을 지정합니다.

MCP 서버는 URL과 선택적 인증 헤더로 구성합니다.
연결 로그, 사용 가능한 프롬프트, 리소스, 도구를 살펴볼 수 있습니다.
선택 사항인 서버 CORS 프록시는 엔진의 해당 플래그로 명시적으로 활성화해야 하며, 비활성 콘텐츠만 반환하고 페이지로 열리는 것을 거부합니다([보안 제한](#security-limits) 참조). HTTPS 페이지 안의 HTTP 리소스에 대한 브라우저 제한은 여전히 적용됩니다.
활성화된 MCP 서버 두 개가 같은 이름의 도구를 제공하면 어느 쪽도 모델에 제공되지 않고 인터페이스에 경고가 표시됩니다. 그 도구를 사용하려면 해당 서버 중 하나만 남기고 모두 비활성화하세요.
서버의 기본 제공 도구와 브라우저 JavaScript 샌드박스는 선택 사항입니다.
도구 승인과 에이전트 턴 수 제한은 인터페이스에서 계속 사용할 수 있습니다. 턴 수 제한(설정 → 에이전트 → **에이전트 턴 수**, 무한 루프를 피하기 위해 중지하기 전까지 도구를 실행하는 최대 주기 수)의 기본값은 50입니다.

스트림은 서버가 그 스트림을 보관하는 동안 저장된 대화 식별 정보와 커서를 사용해 다시 연결할 수 있습니다.
서버가 중지되었거나 스트림이 만료되었으면 재생을 보장할 수 없습니다.
인터페이스는 두 번째 생성 요청을 조용히 만드는 대신 재연결 실패를 보고합니다.

### 로컬 셸 MCP 서버

`--local-shell-mcp-server`를 지정해 `mgs`를 시작하면 웹 채팅의 모델이 `mgs`가 실행되는 머신의 터미널을 사용할 수 있습니다:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- 시작할 때 `mgs`는 어떤 셸이 설치되어 있는지 확인하고 셸마다 도구를 하나씩 제공합니다: `bash.run`(Bash), `zsh.run`(Zsh), `sh.run`(sh), `psh.run`(PowerShell, `pwsh`). 로그에는 찾은 셸이 나열되며, 설치되지 않은 셸에는 도구가 제공되지 않습니다.
- 인터페이스는 이를 감지해 MCP 서버에 **local shell mcp server**라는 이름으로 추가하고 활성화합니다. 그곳에서 비활성화하거나 특정 대화에서만 비활성화할 수 있으며, 그 선택은 유지됩니다. `mgs`를 이 옵션 없이 시작하면 항목이 사라집니다.
- 모든 도구는 `command`와, 선택적으로 `cwd`(작업 디렉터리, 기본값은 `mgs`를 실행하는 사용자의 홈 디렉터리)와 `timeout`(초 단위, 기본값 60, 최대 3600)을 받습니다. 응답은 `exit_code`, `stdout`, `stderr`와 함께 `signal`, `timed_out`, `cancelled`, `timeout`, `seconds`, `shell`, `cwd`를 담은 JSON 객체이며, 긴 출력은 앞부분과 끝부분만 남깁니다.
- 각 명령은 터미널 없이, 빈 입력으로 새 프로세스에서 실행됩니다. `cd`와 변수는 다음 호출로 이어지지 않으며, 대화형 프로그램(편집기, 질문)은 사용할 수 없습니다. 시간 제한이 만료되거나 요청이 취소되면 명령이 시작한 모든 프로세스가 종료됩니다. 백그라운드에 남는 프로세스는 출력을 리디렉션해야 합니다.
- 도구를 항상 허용하도록 선택하지 않는 한, 인터페이스는 호출할 때마다 권한을 묻습니다.
- `sudo`는 터미널 없이도 동작합니다(Linux). 비밀번호가 필요하면 사용자가 어느 페이지에 있든 페이지 중앙에 창이 열리고, sudo의 프롬프트(`[sudo] password for user:`), 비밀번호를 요청한 도구와 명령, 비밀번호 입력란, 그리고 그 아래에 **수락** 및 **취소** 버튼이 표시됩니다. 커서가 이미 비밀번호 입력란에 있으므로 바로 비밀번호를 입력할 수 있습니다. **Enter**를 누르면 수락하고 **Escape**를 누르면 취소하며, 창 바깥을 클릭해도 창은 닫히지 않습니다. 비밀번호는 `mgs`가 실행되는 머신의 sudo에만 전달됩니다. 대화에 저장되지 않고, 로그에 기록되지 않으며, 모델은 절대 볼 수 없습니다. 창이 기다리는 동안에는 명령의 시간 제한이 흐르지 않습니다. **취소**를 누르거나 5분 안에 아무도 응답하지 않으면 sudo는 실패합니다. 호출마다 새 프로세스이므로 sudo는 호출할 때마다 다시 묻고, 비밀번호가 틀리면 최대 세 번까지 묻습니다. 이때 창에는 이전 비밀번호가 틀렸을 수 있다는 경고가 표시됩니다. 명령은 `mgs`를 시작한 터미널과 분리된 자체 세션에서 실행되므로, sudo가 그 터미널에서 비밀번호를 묻는 일은 없습니다. 응답할 수 있는 것은 `mgs`의 웹 인터페이스뿐이며, 다른 MCP 클라이언트에서는 터미널이 없어 sudo가 이전처럼 실패합니다. 배포된 서비스는 보통 비밀번호도 sudo 권한도 없는 `moe-gguf-server` 계정으로 실행되므로, 이 기능은 사용자 본인의 계정으로 실행하는 `mgs`(예: `~/.local` 빌드)를 위한 것입니다. `mgs`에 `DISPLAY` 변수가 없으면 명령은 `DISPLAY=:mgs-askpass`를 받습니다. sudo는 `DISPLAY`가 있을 때만 터미널 없이 비밀번호 도우미를 사용하기 때문입니다. 창을 열려고 하는 프로그램은 즉시 실패합니다.
- 채팅에서 모든 호출은 `bash.run`처럼 해당 도구의 이름으로 표시됩니다. 펼치면 고정폭 글꼴의 회색 상자가 최대 세 개 표시됩니다. 명령만 담은 **stdin**, 출력을 담은 **stdout**, 오류 텍스트를 담은 **stderr**입니다. stdout은 명령이 무언가를 출력했을 때만, stderr는 오류가 있었을 때만 나타납니다. 종료 코드는 상자 제목 옆에 표시됩니다. 예: `stdout (코드 2):`(출력이 없으면 stdin 옆). stdin 상자는 흰색 테두리(밝은 테마에서는 진한 회색), stdout 상자는 녹색 테두리, stderr 상자는 빨간색 테두리이며, 모든 셸 도구(`bash.run`, `zsh.run`, `sh.run`, `psh.run`)에서 같습니다.
- 모든 호출을 자동으로 펼치려면 **설정 → 표시 → 로컬 셸 명령 자동 펼치기**를 켜세요. 각 상자는 내용이 확인되는 즉시 나타납니다. stdin은 모델이 명령을 작성하는 동안, stdout과 stderr는 명령이 끝날 때 나타납니다.
- 셸 도구를 호출할 때마다 `~/.mgs/local-shell-mcp-server/`(`mgs`를 실행하는 사용자의 홈, 배포된 서비스에서는 `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`)에 별도의 파일이 남습니다. 파일 이름은 호출한 로컬 날짜와 시각을 밀리초까지 나타내며(예: `y2026m10d12h23m59s59ms731.jsonl`), 같은 밀리초에 여러 호출이 시작되면 `-2`, `-3`…이 붙습니다. 한 줄에 JSON 객체 하나가 들어 있는 JSONL 파일입니다. 먼저 명령(`"event": "command"`, 도구, 셸, 작업 디렉터리, 시간 제한 포함)이 실행 전에 기록되고, 그다음 출력 전체가 잘리지 않고 도착하는 대로 조각마다 한 줄씩(`"event": "stdout"` 또는 `"stderr"`, 텍스트는 `data`에) 명령이 쓴 순서대로 기록되며, 마지막으로 결과(`"event": "result"`: 종료 코드, 시그널, 시간 초과 또는 취소 여부, 걸린 초, stdout과 stderr의 바이트 수) 또는 시작하지 못한 이유(`"event": "error"`)가 기록됩니다. 실행 전에 거부된 호출(명령 없음, 존재하지 않는 작업 디렉터리, 잘못된 시간 제한)도 요청된 그대로의 명령과 오류를 담은 파일을 남깁니다. 채팅에는 매우 긴 출력의 앞부분과 끝부분만 남지만 파일에는 모든 내용이 있으며, 명령이 실행되는 도중 `mgs`가 중지되더라도 그때까지의 출력은 이미 파일에 들어 있습니다. 호출의 stdout 전체를 읽으려면 `jq -j 'select(.event == "stdout") | .data' FILE`을 실행하세요. JSON에는 텍스트만 담을 수 있으므로 UTF-8이 아닌 바이트(바이너리 출력)는 `�`로 바뀌며, 기가바이트 단위로 출력하는 명령은 그만한 크기의 파일을 남깁니다. 명령의 출력에는 비밀 정보가 들어 있을 수 있으므로 폴더와 파일은 그 사용자만 읽을 수 있습니다. 이 파일은 아무것도 자동으로 삭제하지 않으므로, 더 이상 필요 없는 오래된 파일은 직접 지우세요.

배포된 서비스에서는 systemd override(`systemctl edit moe-gguf-server`)의 `[Service]` 섹션에 `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1`을 추가하고 `moe-gguf-server`를 다시 시작하세요. Alpine에서는 OpenRC 서비스 구성에서 같은 변수를 export하세요.

**보안.** 명령은 `mgs`를 실행하는 사용자의 권한(배포된 서비스에서는 `root`가 아닌 `moe-gguf-server` 계정)으로 실행됩니다. 인터페이스를 열거나 API를 호출할 수 있는 사람은 누구나 명령을 실행할 수 있으므로, 신뢰할 수 있는 환경에서만 이 옵션을 활성화하고 API 키로 서버를 보호하세요. 다른 사이트의 페이지에서 온 요청은 거부되고, 기본 CORS 출처는 localhost가 되며, 로그에는 모든 명령이 종료 코드와 함께 기록됩니다. sudo 비밀번호를 입력하기 전에 비밀번호 창에 표시된 명령을 꼭 읽어 보세요. 그 비밀번호는 바로 그 명령을 승인합니다. 서버는 부모 프로세스가 사용자를 위해 root로 실행 중인 sudo인 비밀번호 도우미에게만 비밀번호를 넘기므로, 명령이 스스로 도우미를 시작해 비밀번호를 읽어 낼 수는 없습니다.

<a id="api"></a>
## HTTP API

예제는 11443 포트를 사용합니다. 포트가 다르면 `ports.json`의 HTTPS 포트로 바꾸세요.

기본 API 루트는 `/api`입니다. UI는 그대로 `/`에 있습니다.
원래 서버의 루트 `/v1/...` 경로는 더 이상 등록되지 않습니다.
OpenAI 호환 클라이언트는 기본 URL로 `https://ai.example.com:11443/api/v1`을 사용해야 합니다.
요청/응답 필드 이름은 업스트림과의 호환성을 유지합니다. C/C++ 엔진 인터페이스는 현재 함수를 유지하지만, 지원 중단으로 표시되었던 함수는 유지하지 않습니다([제거된 호환성](#removed-compatibility) 참조).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

사용하는 셸에서 `MGS_API_KEY`를 설치된 키로 설정하세요.
네이티브 작업, 임베딩, 토큰화, 재순위화(reranking), 모델 관리, 재개 가능한 스트림은 `/api/doc/`의 번들된 Swagger UI에 문서화되어 있습니다. 그 OpenAPI JSON은 `/api/doc/openapi.json`에 있으며, 정적 JSON 파일 없이 서버가 요청마다 생성합니다.
이 목록에는 라우터 경로와 선택적 호환 경로를 포함해, 실행 중인 프로세스가 등록한 경로가 반영됩니다.
비활성화된 기능은 403을 반환할 수 있습니다.
키가 없거나 유효하지 않으면 401을, 모델을 로드하는 중이면 503을 반환합니다.
상태 확인(health) 경로와 모델 목록 경로는 `GET`과 `HEAD`에 대해 공개 접근을 유지합니다. 항상 키가 필요한 경로는 [보안 제한](#security-limits)을 참조하세요.

`--api-prefix /api/custom`을 지정하면 추론 경로가 `/api/custom` 아래로 옮겨집니다.
루트 UI는 그 접두사를 `/api/config.js`에서 받으며, 문서는 계속 `/api/doc/`에 있습니다.
문서 네임스페이스는 예약되어 있으므로 추론 접두사로 사용할 수 없습니다.
라우터의 자식 프로세스는 내부적으로 `/api`를 사용합니다.
서버는 접두사를 검증하며, `/api/` 밖의 값과 끝에 슬래시가 붙은 경로는 거부됩니다.

모니터링 페이지는 `GET /api/monitor`(전체 보고서), `GET /api/monitor/memory`(로드된 모델의 메모리 스냅샷), `GET /api/monitor/usage`(CPU 및 GPU 사용률만, 모델을 로드하는 동안에도 사용 가능)를 사용합니다.
이 경로들은 작업 큐가 아니라 추론 루프가 게시하는 카운터와 스냅샷을 읽으므로, 폴링해도 잠든 모델을 깨우거나 유휴 타이머를 초기화하지 않습니다.
각 필드는 Swagger에 설명되어 있습니다.

인터페이스는 `GET /api/ui/settings`(`{"settings": DOCUMENT}`, 처음 저장하기 전에는 `{"settings": null}`)로 설정을 읽고, `POST /api/ui/settings`(`{"settings": DOCUMENT}`, `~/.mgs/config/ui-settings.json` 전체를 모드 0600으로 교체)로 설정을 저장합니다.
두 경로 모두 API 키가 필요하며, 모델을 로드하는 동안에도 응답합니다.

<a id="request-queue"></a>
### 요청 대기열과 병렬 요청

기본적으로 `mgs`는 한 번에 요청 하나를 처리하며(`-np 1`) 그 요청에 컨텍스트 전체를 할당합니다. 따라서 웹 인터페이스에서 오든 API로 직접 오든, 모든 요청이 컨텍스트 토큰 전체를 사용할 수 있습니다.
그동안 도착한 요청은 하나의 대기열에서 기다리며 도착한 순서대로 처리됩니다. 인터페이스의 요청이 API 요청을 앞지르지 않으며, 그 반대도 마찬가지입니다.
API는 인터페이스 설정이 어떻든 똑같이 동작합니다.

여러 요청을 동시에 처리하려면 `-np N`(`--parallel N`, `MGS_ARG_N_PARALLEL`)으로 `mgs`를 시작하세요. 최대 N개의 요청이 병렬로 실행되고 컨텍스트가 그 요청들에 나뉘므로, 각 요청은 일부만 받습니다(로그의 `initializing` 줄에 있는 `n_ctx_slot`). `--kv-unified`를 지정하면 대신 컨텍스트 전체 크기의 KV 캐시 하나를 함께 사용합니다.
N개를 넘는 요청은 여전히 대기열에서 기다립니다.

웹 인터페이스는 요청에 `"mgs_client": "webui"`를 추가합니다.
스트리밍되는 인터페이스 요청 하나가 기다리는 동안, 그 스트림은 다음과 같은 이벤트로 시작합니다

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: 그 요청보다 앞서 실행 중이거나 대기 중인 요청 수, `ahead_api`와 `ahead_webui`: 그중 API로 직접 온 요청과 웹 인터페이스에서 온 요청의 수). 요청이 시작될 때는 모든 수가 0인 마지막 이벤트가 전송됩니다.
`mgs_client`를 보내지 않는 클라이언트는 이 이벤트를 절대 받지 않으므로, OpenAI 호환 클라이언트는 평소와 같은 스트림을 받습니다.
Swagger는 이 필드를 다른 completion 매개변수와 함께 설명합니다.

<a id="security-limits"></a>
### 보안 제한

이 제한은 적대적인 요청, 파일, 모델로부터 서버와 인터페이스를 보호합니다.
일반적인 사용에는 충분히 넉넉하며, 제한을 넘는 것은 서버를 멈추게 하는 대신 명확한 오류로 실패합니다.

- **API 키.** 상태 확인 경로와 모델 목록 경로는 `GET`과 `HEAD`에 대해서만 공개됩니다. 모델 목록 다시 불러오기(`GET /api/models?reload=1`)와 라우터 모드에서의 모델 추가 또는 삭제(`POST` 및 `DELETE /api/models`)에는 키가 구성되어 있으면 항상 키가 필요합니다.
- **다른 사이트의 페이지.** 서버 도구나 MCP 서버가 활성화되어 있고(`--tools`, `--agent`, `--mcp-servers-config`, `--mcp-servers-json` 또는 `--local-shell-mcp-server`) `--cors-origins`가 설정되지 않은 경우, 서버는 `Origin` 헤더가 다른 사이트를 가리키는 `GET`, `HEAD`, `OPTIONS` 이외의 모든 요청에 403으로 응답합니다. 인터페이스(Apache 뒤에 있는 경우 포함)와, curl이나 SDK처럼 `Origin`을 보내지 않는 클라이언트는 영향을 받지 않습니다. `POST /api/tools`는 `Content-Type: application/json`만 받습니다(그 외에는 415).
- **CORS 프록시.** `/api/cors-proxy`는 브라우저가 페이지, 프레임, 워커로 열 요청을 거부하며(403), 모든 응답을 비활성 콘텐츠로 반환합니다. 따라서 서버 주소에서 스크립트를 실행하거나 쿠키를 설정할 수 없습니다.
- **라우터 목록.** 라우터 모드에서 `/api/models`의 `status.args`와 `status.preset`은 `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`, `--mcp-servers-json`의 값 대신 `<redacted>`를 표시하며, 로그도 마찬가지입니다. 모델 프로세스는 여전히 실제 값을 받습니다.
- **느린 클라이언트.** 서버가 요청을 받는 동안 각 읽기는 최대 30초까지 기다리며(쓰기는 여전히 `--timeout`이 관리하므로 긴 응답이 잘리지 않음), 빈 스레드를 기다리는 수락된 연결은 최대 256개입니다.
- **요청 본문.** 배열이나 객체가 1024단계보다 깊게 중첩된 JSON 본문은 400으로 거부됩니다. JSON 스키마(`response_format`, `json_schema`)는 최대 256단계까지 중첩할 수 있으며, 빈 `pattern` 값과 반복할 대상이 없는 수량자는 거부됩니다.
- **원격 미디어.** `image_url`, `input_audio`, `input_video`의 URL은 공개 인터넷 주소만 가리킬 수 있습니다. 루프백, 사설 네트워크(10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10), 링크 로컬 주소(클라우드 메타데이터 주소 169.254.169.254 포함), IPv6 고유 로컬 주소, 그 밖의 비공개 주소는 리디렉션으로 그 주소에 도달하는 경우를 포함해 400으로 거부됩니다. 예를 들어 사용자 네트워크의 서버에서 이미지를 읽기 위해 이를 허용하려면 `--media-allow-private-urls`(`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`)로 `mgs`를 시작하세요. 모든 API 클라이언트를 신뢰할 수 있을 때만 그렇게 하세요. 각 다운로드에는 연결 10초, 전체 30초, 최대 5번의 리디렉션, 10 MiB의 제한이 있으며, 서버는 동시에 최대 8개의 다운로드만 실행합니다.
- **디코딩된 미디어.** 이미지는 한 변 최대 32768픽셀, 최대 100메가픽셀, 오디오는 최대 1시간, 동영상 프레임은 한 변 최대 8192픽셀이며 동영상 하나당 디코딩된 프레임은 최대 2 GiB, 요청 하나당 디코딩된 이미지와 오디오는 최대 2 GiB입니다.
- **모델의 채팅 템플릿.** `range()`는 최대 100만 개의 요소를 만들고, 만들어진 문자열은 256 MiB 미만, `indent()`와 `tojson()`의 들여쓰기는 256칸 미만, 중첩은 1024개 문장 미만으로 유지됩니다. 이를 넘는 템플릿은 템플릿 오류로 실패합니다.
- **RPC.** RPC 백엔드가 포함된 빌드에서 RPC 서버는 상대가 보낸 프레임 크기와 텐서 경계를 검사하고, 문제가 있는 연결만 닫습니다. 신뢰할 수 있는 머신에만 노출하세요. 여전히 한 번에 클라이언트 하나만 처리합니다.
- **인터페이스.** 메시지 안의 링크는 http, https, mailto, tel 스킴(또는 상대 주소)만 유지하며, 이미지 첨부 파일은 내장 이미지여야 합니다. 대화 가져오기는 최대 512 MB의 파일, 항목이 최대 10 000개인 ZIP 아카이브, JSONL 파일당 256 MB, 압축 해제 후 전체 1 GB까지 받습니다. 잘못된 형식의 레코드가 있거나 다른 대화에 속한 메시지가 있는 파일은 통째로 거부되며, 메시지가 이미 존재하는 대화는 건너뛰고 보고됩니다.

설치 명령은 여전히 HTTPS를 통해 현재 `main` 브랜치를 내려받으며, 서명이나 고정된 리비전은 확인하지 않습니다.

<a id="performance"></a>
## 메모리와 성능

호스트 메모리 등록과 전문가 프리페치는 기본적으로 활성화되어 있습니다.
프리페치는 특히 큰 프롬프트에서 전문가 전송을 계산과 겹쳐 수행합니다.
결과는 모델, 메모리 용량, RAM/PCIe 대역폭에 따라 달라집니다.

전문가를 CPU로 오프로드하는 모델에서는 `--cpu-moe` 또는 `--n-cpu-moe N`으로 VRAM 사용량을 줄일 수 있습니다.
호스트 등록은 매핑된 가중치 페이지를 고정(pin)하여 호스트에서 디바이스로의 복사를 빠르게 합니다.
고정된 메모리는 페이지 아웃될 수 없으므로 운영 체제가 쓸 RAM을 충분히 남겨 두세요.
이 등록 방식은 POSIX 전용이며, Windows는 이전의 로드 방식을 유지합니다.
`--load-mode none`은 호스트 등록이 전부 또는 일부 실패할 때만 권장되며, 일반적으로 필요한 옵션은 아닙니다.

전문가 프리페치는 기본적으로 VRAM 슬롯 세 개를 사용합니다.
각 슬롯에는 전송되는 가장 큰 전문가 텐서가 들어가야 합니다.
`GGML_SCHED_PREFETCH_EXPERTS`에 2–8 값을 지정하면 슬롯 수를 명시적으로 선택합니다.
메모리 할당이 실패하면 가능한 경우 할당된 슬롯을 최소 두 개 유지하고, 그렇지 않으면 일반 전송을 사용합니다.
프리페치는 배치의 전문가 선택 수가 전문가 수의 두 배 이상일 때만 시작됩니다.
프리페치 풀은 이를 처음 활성화한 GPU에 속하며, 다른 GPU는 선택적 복사를 유지합니다.
배치가 클수록 프리필에 도움이 될 수 있지만 메모리가 더 필요합니다.

진단을 위해 `GGML_CUDA_REGISTER_HOST=0` 또는 `GGML_SCHED_PREFETCH_EXPERTS=0`을 설정하면 해당 최적화를 비활성화할 수 있습니다.
의도적으로 끄려는 경우가 아니라면 일반 운영 중에는 이 변수들을 설정하지 마세요.

자동 MTP는 가중치를 할당하기 전에 모든 샤드를 포함해 GGUF 메타데이터와 텐서 이름을 검사합니다.
지원되는 아키텍처, 모델 본체, 그리고 통합된 모든 헤드가 필요합니다.
외부 헤드나 지원되지 않는 아키텍처로는 활성화되지 않습니다.
기본 초안 한도는 토큰 세 개입니다.
명시적인 `--spec-type` 선택이 우선하며, `--no-spec-mtp-auto`와 `--spec-type none`은 자동 선택을 비활성화합니다.
`--spec-draft-n-max`는 한도를 바꿉니다.
MTP는 생성에 영향을 주며, 프리필이나 첫 토큰까지의 시간에는 직접 영향을 주지 않습니다.
향상 정도는 수락률과 하드웨어에 따라 달라지며, 한도를 늘린다고 속도가 더 빨라진다는 보장은 없습니다.

<a id="mtp-vram"></a>
MTP 가중치는 GPU가 담을 수 있으면 항상 GPU 메모리에 머뭅니다. 모델에 이식된 MTP 레이어, `--spec-draft-model`로 전달한 별도의 MTP GGUF, 그리고 이들이 사용하는 출력 헤드가 여기에 해당합니다.
가중치를 시스템 메모리로 옮기는 옵션(`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, 이에 대응하는 `--spec-draft-*` 옵션, 자동 맞춤)은 이들에게 적용되지 않으며, `--gpu-layers` 값이 낮아도 GPU에 남습니다.
자동 맞춤은 모델의 나머지 부분을 그 주위에 배치하고, [MoE 전문가 캐시](#expert-cache)는 그 뒤에 남은 메모리를 사용합니다.
MTP 레이어가 512 MiB의 여유를 두고 남는 GPU 메모리에 들어가지 않으면, 로그에 경고가 표시되고 다른 레이어와 똑같이 배치됩니다.
별도의 MTP GGUF를 사용하면 주 모델에 이식된 MTP 레이어는 로드되지 않으므로 메모리를 두 번 차지하지 않습니다. `--spec-type none`을 지정하면 MTP 가중치는 하나도 로드되지 않습니다.
로드 로그는 `keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)` 같은 줄로 배치를 확인해 주며, 이 줄은 `-lv 4`에서 볼 수 있습니다.

<a id="expert-cache"></a>
### MoE 전문가 캐시

MoE 전문가가 시스템 메모리에 머물고 레이어의 나머지가 GPU에서 실행될 때, 서버는 모든 레이어에서 가장 최근에 사용한 전문가를 남는 VRAM을 차지하는 캐시에 보관합니다. GPU가 여러 개면 각 GPU가 자신에게 할당된 레이어를 자신의 VRAM에 캐시합니다.
생성되는 토큰마다(그리고 32토큰 미만의 모든 배치에서), 캐시에 있는 전문가는 GPU에서 실행되고, 캐시에 없는 전문가 중 일부는 캐시로 업로드되며, 나머지는 동시에 CPU에서 실행됩니다. 이 비율은 측정된 PCIe와 CPU 대역폭에 따라 정해집니다.
긴 프롬프트에서는 전문가 프리페치가 그대로 사용되며, 이제 프리페치는 캐시된 전문가를 GPU 안에서 복사하고 나머지만 업로드합니다.
출력은 반올림 차이를 제외하면 캐시 없이 실행한 결과와 같습니다. GPU와 CPU는 반올림 방식이 다르고 각각이 계산하는 전문가가 캐시 내용에 따라 달라지므로, 긴 탐욕적(greedy) 생성은 실행할 때마다 몇 토큰 뒤에 다르지만 똑같이 타당한 텍스트로 갈라질 수 있습니다. 공식 llama.cpp의 캐시도 같은 방식으로 동작합니다.

캐시는 기본적으로 활성화되어 있으며, 모델, MTP 컨텍스트, 멀티모달 프로젝터가 로드된 뒤 첫 요청에서 생성됩니다.
`--gpu-layers`, `--n-cpu-moe`, `--override-tensor`를 지정하지 않으면, 자동 맞춤은 여유(4 GiB 또는 RAM의 10%)를 두고 들어갈 때 모든 전문가를 시스템 메모리에 두고 남는 VRAM을 캐시에 맡깁니다(GPU가 여러 개면 남는 메모리에 따라 레이어를 분산하여 모든 GPU에 캐시가 생기게 합니다). 그렇지 않으면 이전처럼 전문가 레이어 전체를 GPU에 배치하고 캐시는 남은 공간을 사용합니다.
명시적인 배치 옵션은 존중됩니다. `--cpu-moe`는 캐시에 가장 많은 VRAM을 주고, `--n-cpu-moe N`은 일부 정적 레이어를 GPU에 유지합니다.
RAM에 남는 전문가는 고정되므로, RAM에는 그 전문가 전부와 운영 체제가 함께 들어가야 합니다.

<a id="context-size"></a>
모델이 VRAM에 들어가지 않고 `-c`를 지정하지 않으면, 자동 맞춤은 컨텍스트 크기도 정합니다. KV 캐시는 밀집 가중치를 뺀 나머지 VRAM의 최대 4분의 1을 모델의 학습 컨텍스트 범위 안에서 차지하고, 전문가(캐시 또는 전문가 레이어 전체)가 나머지를 사용합니다.
이전에는 이런 모델이 항상 모든 슬롯이 나눠 쓰는 최소 4096토큰만 받았기 때문에 긴 답변이 중간에 잘렸습니다(로그의 `truncated = 1`).
로그에는 `context size raised from 4096 to N tokens` 같은 줄과 `initializing` 줄의 `n_ctx_slot`으로 결과가 표시됩니다.
`-c N`은 컨텍스트를 명시적으로 설정합니다(컨텍스트가 클수록 캐시에 남는 VRAM이 줄어 생성 속도가 낮아집니다. 기준 머신에서 4096토큰일 때 35.8 tokens/s, 122624토큰일 때 33.9 tokens/s). `--fit-ctx N`은 최솟값을 올립니다.
MTP 컨텍스트는 학습 컨텍스트 전체만큼 VRAM을 예약하는 대신 모델 컨텍스트와 같은 크기를 사용합니다.

자동 맞춤은 컨텍스트보다 먼저 마이크로 배치(`-ub`), 즉 한 번의 처리에서 다루는 프롬프트 토큰 수를 정합니다.
전문가가 RAM에 있으면 처리할 때마다 캐시에 없는 전문가를 GPU로 업로드하므로, 더 크고 적은 횟수의 처리로 긴 프롬프트를 훨씬 빠르게 처리할 수 있으며 생성 속도는 변하지 않습니다.
`-b`나 `-ub`를 지정하지 않으면, 추가 버퍼가 밀집 가중치를 뺀 나머지 VRAM의 8분의 1에 들어가는 값 중 4096, 2048, 1024토큰 가운데 가장 큰 값을 선택하고, 배치(`-b`)도 함께 올립니다.
기준 머신에서 2514토큰 프롬프트는 4096을 사용했을 때 같은 생성 속도로 214 tokens/s에서 755 tokens/s가 되었습니다. 버퍼가 커지면 기본 컨텍스트가 짧아집니다(16 GB에서 MTP 없이 약 90000토큰, MTP 사용 시 61000토큰).
로그에는 `micro-batch size
raised from 512 to 4096 tokens (batch 4096)` 같은 줄로 선택 결과가 표시됩니다.

| 옵션 | 변수 | 효과 |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE`(`0`이면 비활성화) | 캐시를 활성화하거나 비활성화합니다(기본값은 활성화). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | 캐시 전체 크기를 고정하며, 각 GPU가 캐시하는 전문가에 따라 GPU 간에 나눕니다. `-1`(기본값)은 모든 GPU의 남는 VRAM을 사용합니다. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | 캐시와 프리페치 슬롯 외에 모든 GPU에서 비워 두는 VRAM입니다(기본값 1024). 다른 프로그램이 GPU를 사용하면 늘리세요. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | 진단용: 측정된 균형 대신, 캐시에 없는 전문가 중 업로드할 고정 비율(0–1)입니다. |
| — | `MGS_MOE_CACHE_STATS` | 진단용: N 디코딩 단계마다 캐시 카운터를 로그에 기록합니다(`-lv 4`에서 표시). |

`GET /api/expert-cache`는 크기, 캐시된 레이어와 슬롯, 적중과 실패, 업로드, CPU 전문가, 캐시에서 제공된 프리페치 바이트의 비율, 측정된 대역폭을 반환합니다(GPU가 여러 개면 크기와 카운터는 합산되고 업로드 대역폭은 평균값입니다).
`{"size_mib": N}`과 함께 `POST /api/expert-cache`를 보내면 재시작 없이 크기를 바꿉니다(`-1`은 자동, `0`은 캐시 해제). 다음 요청에서 적용되며 빈 캐시로 시작합니다:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

`--metrics`를 지정하면 `/api/metrics`에 `mgs:moe_cache_*` 카운터와 게이지(단계, 적중, 실패, 업로드, CPU 전문가, 프리페치 바이트, 크기, 슬롯, 대역폭)가 추가됩니다.

RTX 4060 Ti 16 GB에서 Qwen3.6-35B-A3B Q8_0(35 GiB, CPU 스레드 8개)으로 캐시 없는 `--n-cpu-moe 28`과 비교한 기준 측정 결과: 생성 26.9 → 35.7 tokens/s(+33%), 프롬프트 처리 767–778 → 718–730 tokens/s(−6%), MTP를 활성화하면 33.6 → 47.0 tokens/s(+40%).
같은 머신에서 자체 캐시(`-cmoe --moe-cache-mib 10240`)를 사용한 공식 llama.cpp는 26.7 tokens/s, MTP 사용 시 27.8 tokens/s에 도달했습니다.
향상 폭은 VRAM에 들어가는 전문가의 비율, 모델의 라우팅 지역성, PCIe와 RAM 대역폭에 따라 달라집니다.

<a id="threads"></a>
### CPU 스레드

CPU는 캐시에 없는 전문가를 계산합니다.
이 작업은 메모리 대역폭에 묶여 있어 적은 스레드로도 포화되며, 모든 논리 CPU를 사용하면 스레드들이 GPU를 구동하는 스레드와 경쟁하게 됩니다.
`-t`를 지정하지 않으면 `mgs`는 수학 연산에 유용한 코어를 사용하지만, 실행할 수 있는 논리 CPU의 절반을 넘지는 않습니다.
SMT(코어당 스레드 2개)가 있는 머신에서는 이전처럼 물리 코어 수가 되고, SMT를 드러내지 않는 가상 머신에서는 가상 CPU의 절반이 됩니다.
기준 머신(데스크톱 세션이 있는 가상 코어 16개, Qwen3.6-35B-A3B Q8_0, MTP 없음)에서 스레드 4~12개는 같은 속도(32.9–33.9 tokens/s)로 생성했고, 스레드 16개는 28.7 tokens/s였습니다.
시작 로그에는 `using 8 CPU threads for generation and 8 for batches` 같은 줄로 선택 결과가 표시되며, `-t`와 `-tb`로 명시적으로 설정할 수 있습니다.

<a id="checkpoints"></a>
### 컨텍스트 체크포인트

순환(recurrent) 레이어나 슬라이딩 윈도 레이어가 있는 모델(예: Qwen3.5/3.6/3.8 하이브리드, Nemotron-H, Granite 4)은 상태를 되돌릴 수 없으므로, 요청이 대화의 앞부분을 바꾸면 서버는 변경 지점 이전의 가장 최근 체크포인트에서 다시 시작합니다.
체크포인트는 사용자, 어시스턴트, 도구 메시지의 시작 지점(코딩 에이전트가 오래된 추론을 제거하거나 오래된 도구 결과를 잘라 내는 곳)과, 추가 프롬프트 처리 없이 최소 `--checkpoint-min-step`토큰마다(기본값 1024, 이전에는 8192) 배치됩니다.
`--ctx-checkpoints`(32)에 도달하면 이웃 체크포인트와 가장 가까운 체크포인트가 제거되므로, 나머지 체크포인트는 대화 전체에 고르게 분포한 상태를 유지합니다.
기준 에이전트 시나리오(14k 토큰, 도구 결과 8개)에서 뒤쪽의 도구 결과를 잘라 내는 작업은 19.3초에서 5.5초로, 앞쪽의 도구 결과는 18.0초에서 13.6초로 줄었습니다.
각 체크포인트는 순환 또는 슬라이딩 윈도 상태를 호스트 메모리에 보관합니다(Qwen3.6-35B-A3B의 경우 63 MiB).

<a id="upstream-options"></a>
### 업데이트된 옵션과 진단

호스트 등록이 성공하면 기본값인 `--load-mode mmap`을 유지하세요. `--load-mode none`은 등록이 실패할 때 대체 수단으로만 사용하세요. 이제 `--load-mode dio`는 최대 64 MiB의 추가 메모리를 거쳐 읽기를 스테이징합니다. 이는 모델 전체가 사용하는 RAM의 상한이 아닙니다.

`GGML_CUDA_PEER_MAX_BATCH_SIZE`는 런타임에 아무 효과가 없었으므로 제거되었습니다. 이전 로드 플래그(`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`)와 `GGML_CUDA_FA_ALL_QUANTS` 빌드 플래그는 더 이상 받지 않습니다. 대체 방법은 [제거된 호환성](#removed-compatibility)을 참조하세요. 수동 CMake 빌드에서 `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0`을 지정하면 해당 K/V 벡터 커널 조합이 추가되며, 쉼표로 구분한 목록이나 따옴표로 감싼 세미콜론 구분 목록도 받습니다. 기본 목록은 `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`이며, F16은 항상 포함됩니다. `all`은 컴파일 작업량을 늘립니다. 빠진 벡터 커널 조합은 F16 변환 대체 경로를 사용하므로 더 느릴 수 있습니다.

헤드가 하나인 완전한 GLM-4.5-Air(`glm4moe`) GGUF도 이제 자동 MTP를 활성화할 수 있습니다. 기존의 다른 적격성 검사와 명시적 옵션의 우선 규칙은 여전히 적용됩니다. MTP는 출력이 있는 그래프와 없는 그래프를 분리하고, 동시 시퀀스 전반에서 원래의 토큰 순서를 유지하며, 이미지 뒤에서는 실제 위치를 사용합니다.

자동 맞춤은 nextn(MTP) 레이어를 계산에 포함하므로, MTP가 비활성화되어 있을 때 첫 번째 레이어가 더 이상 CPU에 남지 않습니다. 이제 MTP 가중치는 항상 GPU 메모리에 머뭅니다. [MTP 배치](#mtp-vram)를 참조하세요. 기본 `--checkpoint-min-step`은 이제 8192 대신 1024입니다. [컨텍스트 체크포인트](#checkpoints)를 참조하세요. `--moe-cache`, `--no-moe-cache`, `--moe-cache-size`, `--moe-cache-reserve`는 [MoE 전문가 캐시](#expert-cache)를 제어합니다. 자동 맞춤은 VRAM보다 큰 모델에 더 이상 4096토큰 컨텍스트만 남기지 않으며, 이제 마이크로 배치도 선택합니다. [컨텍스트 크기](#context-size)를 참조하세요. 기본 CPU 스레드 수는 이제 최대 논리 CPU의 절반입니다. [CPU 스레드](#threads)를 참조하세요. 이제 `-np`의 기본값은 통합 KV 캐시를 공유하는 슬롯 네 개의 자동 모드가 아니라 컨텍스트 전체를 가진 슬롯 하나이며, `--kv-unified`는 기본적으로 꺼져 있습니다. [요청 대기열](#request-queue)을 참조하세요.

`--metrics`를 지정하면 인증된 `GET /api/metrics`에 `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total`, `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`이 포함됩니다. 위치 인덱스는 0부터 시작하며, 이 시리즈는 추측 요청이 완료된 뒤에 나타납니다. 수락률을 평가할 때는 수락 카운터와 초안 카운터의 증가분을 비교하고, 지연 시간은 따로 측정하세요. 수락률만으로는 속도 향상이 증명되지 않습니다. 라우터 모드에서는 `?model=MODEL_ID`로 모델을 선택하세요.

`--media-allow-private-urls`(`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, 기본값은 꺼짐)를 지정하면 요청의 미디어 URL이 루프백, 사설 주소, 그 밖의 비공개 주소에 접근할 수 있습니다. [보안 제한](#security-limits)을 참조하세요.

대화 내보내기는 저장된 전체 메시지 트리와 메타데이터를 읽습니다. 서버 도구가 403을 반환하면 새 메시지에서는 실패한 조회를 더 이상 반복하지 않으며, 도구 패널을 다시 열면 다시 시도합니다. 첫 방문 시의 서버 기본값은 API 키를 포함해 사용자가 이미 바꾼 설정을 보존합니다.

[구현 및 업스트림 출처](CODE.ko-KR.md#upstream-review).

<a id="removed-compatibility"></a>
## 제거된 호환성

이전 버전과의 호환을 위한 임시 처리 코드는 제거되었습니다.
이전 이름은 더 이상 변환되지 않으며 경고도 표시되지 않습니다. 서버는 제거된 명령줄 옵션을 잘못된 인수로 거부하고, 제거된 환경 변수는 무시하며, 제거된 경로에는 404로 응답합니다.
스크립트, 서비스 override, 프리셋, 클라이언트를 아래의 대체 항목으로 업데이트하세요.

| 제거됨 | 대신 사용할 항목 |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock`(둘을 함께 쓰려면 `mmap+mlock`) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | 없음. 아무 효과가 없었습니다. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft`, `--draft-p-min` 같은 이전 초안 모델 별칭 | 대응하는 `--spec-draft-*` 옵션. 예: `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf`, `--spec-draft-p-min`. `mgs --help`에 모두 나열됩니다. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` 또는 `--spec-ngram-mod-n-max` / `--spec-draft-n-min` 또는 `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | 선택한 n-gram 모드의 `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m`, `--spec-ngram-*-min-hits` 옵션 |
| 이전 업스트림 접두사를 사용하는 환경 변수, CMake 옵션, C API 심볼(이전 버전에서 이미 없앤 별칭 포함) | `MGS_` 접두사(C API에서는 `mgs_`)가 붙은 현재 이름. 예: `MGS_ARG_PORT`, `NO_` 변형 대신 `MGS_ARG_<NAME>=false`, 메모리 매핑·잠금·Direct I/O 변수 대신 `MGS_ARG_LOAD_MODE`, 초안 한도에는 `MGS_ARG_SPEC_DRAFT_N_MAX`와 `MGS_ARG_SPEC_DRAFT_N_MIN`, `MGS_CACHE` 또는 `MGS_BUILD_UI`. 백엔드 옵션에는 대응하는 `GGML_*` 옵션(예: `GGML_CUDA` 또는 `GGML_NATIVE`) |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` 또는 `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` 또는 `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| `reasoning_format` 값 `deepseek-legacy`(요청 필드 및 `--reasoning-format`) | `deepseek` 또는 `auto`. 추론은 `reasoning_content`로 반환됩니다. |
| `reasoning_budget_end_tag` 요청 필드 | `reasoning_budget_end_tags`(문자열 배열) |
| `generation_settings`의 `reasoning_in_content` | 없음. 이 필드는 제거되었습니다. |
| Hugging Face 버킷에서 미리 빌드된 업스트림 인터페이스 다운로드(`USE_PREBUILT_UI`와 `UI_HF_BUCKET` CMake 옵션, `HF_UI_VERSION` 환경 변수) | 없음. 인터페이스는 항상 `frontend/`에서 빌드되거나(`MGS_BUILD_UI=ON`) 제외됩니다. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| 빌드 번호와 커밋: CMake 변수 `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT`(`mgs-config.cmake`에도 있음)와 `build_info`, `system_fingerprint`, `User-Agent`의 `b0-unknown` 문자열 | `backend/cmake/build-info.cmake`에서 설정하는 `MGS_VERSION`. 이제 이 필드들에는 `0.1.0` 같은 버전이 들어갑니다. |
| Meta Llama 모델(`llama`, `llama4`, `llama-embed` GGUF 아키텍처와 Llama 4 비전 프로젝터) 및 내장 채팅 템플릿 `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3`, `llama4` | 없음. 지원되는 MoE 모델 중 하나를 사용하세요. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| 설치 프로그램 옵션 `--no-sudo` | 없음. 설치 프로그램은 항상 sudo 없이 root로 실행됩니다. |
| C API에서 지원 중단으로 표시되었던 함수와 필드: `mgs.h`의 `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, `mgs_token_*` 별칭(예: `mgs_token_bos`, `mgs_token_get_text`, `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, `mgs_context_params`의 `defrag_thold` 필드, `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; `ggml.h`의 `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace`, `ggml_upscale_ext`; `mtmd.h`의 `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny`, `image_marker` 필드 | 현재 함수: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*`(`mgs_vocab_cls` 대신 `mgs_vocab_bos`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos`, `media_marker`. 워밍업 실행은 호출하는 쪽에서 수행하며, `defrag_thold`는 아무 효과가 없었습니다. |
| Gemma 4: `gemma4`와 `gemma4-assistant` 아키텍처, `gemma4v`, `gemma4uv`, `gemma4a`, `gemma4ua` 프로젝터, `peg-gemma4` 채팅 형식, `gemma4` 토크나이저와 사전 토크나이저, `--vision-gemma-4b-default`와 `--vision-gemma-12b-default` 프리셋 | 없음. [지원되는 계열](#supported-models) 중 하나를 사용하세요. |
| V4.1보다 오래된 DeepSeek 모델: `deepseek`, `deepseek2`(이것으로 변환된 Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3, Youtu 파일 포함), `deepseek2-ocr`(`deepseekocr` 및 `deepseekocr2` 프로젝터 포함), `deepseek32`, `mistral4`, `deepseek4` 아키텍처, `deepseek-llm`과 `deepseek-coder` 사전 토크나이저, DeepSeek-V4 DSpark 초안 백본 | DeepSeek-V4.1-Flash(`deepseek41`). |
| `--jinja`, `--no-jinja`(`MGS_ARG_JINJA`)와 Jinja가 아닌 엔진의 내장 템플릿(`deepseek3`, `chatglm4`, `mistral-v7-tekken` 같은 이름을 지정한 `--chat-template`) | 없음. 채팅 템플릿은 항상 Jinja를 거칩니다. `--chat-template chatml`은 여전히 ChatML을 선택합니다. 다른 템플릿을 쓰려면 그 Jinja 소스를 `--chat-template` 또는 `--chat-template-file`로 전달하세요. `tools`가 포함된 요청에는 더 이상 어떤 플래그도 필요하지 않습니다. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| 한 옵션의 두 번째 긴 이름: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk`(라우터 프리셋 키도 같은 이름을 따릅니다) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | 옵션 이름을 따른 `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL` |
| `thinking_budget_tokens` 요청 필드 | `reasoning_budget_tokens`(Anthropic 클라이언트는 계속 `thinking.budget_tokens`를 보냅니다) |
| `POST /api/embeddings`와 `POST /api/v1/embeddings`의 `content` | `input` |
| `mgs.h`의 `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, `mgs_chat_message` 구조체, `MGS_FILE_MAGIC_GGLA`; `mtmd.h`의 `mtmd_image_tokens_get_id` | 없음. 채팅 서식은 `common/chat.h`의 Jinja 템플릿이 처리하며, 나머지는 쓰이는 곳이 없었습니다. |

CMake는 제거된 `-D` 변수를 적용하지 않고 사용되지 않은 변수로 보고하므로, 수동 빌드에서는 구성 출력을 확인하세요.

웹 인터페이스는 설정을 서버(`~/.mgs/config/ui-settings.json`)에 보관하며, 브라우저에는 API 키(`localStorage`의 `MgsUi.apiKey`)와 대화(`MgsUi` IndexedDB 데이터베이스)만 보관합니다.
이전 버전이 `localStorage`에 보관하던 설정(`MgsUi.config`, `MgsUi.userOverrides`, 즐겨찾는 모델, 도구 목록, 추론 노력, 닫은 MCP 추천, 언어)은 읽거나 마이그레이션하지 않으므로, 설정은 기본값에서 시작하고 API 키는 한 번 다시 입력해야 합니다.
이전 버전이 다른 이름으로 저장한 데이터, 독립된 `theme` 키, 오래된 메시지 안의 인라인 추론 및 도구 호출 표식, 이전 `context` 유형의 붙여 넣은 텍스트 첨부, 닫은 MCP 추천의 이전 목록 형식, 서버의 이전 `webui_settings` 필드도 읽거나 마이그레이션하지 않습니다.
이전 버전의 대화를 유지하려면 업그레이드하기 전에 이전 버전에서 내보낸 뒤, 업그레이드 후에 그 파일을 가져오세요.
대화 가져오기는 [보안 제한](#security-limits)에 나열된 한도 안에서 JSONL과 ZIP 내보내기만 받으며, 이전 JSON 내보내기 형식은 거부됩니다.
이전 설정은 삭제할 때까지 브라우저에 남아 있습니다. 이를 제거하려면 인터페이스 페이지에서 브라우저 콘솔(F12)을 열고, 인터페이스를 열었던 주소와 포트마다 한 번씩 `for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`를 실행하세요.

`-mu`/`--model-url`로 내려받은 파일은 `~/.cache/mgs/`에 캐시되므로, 이전 버전이 예전 캐시 폴더에 저장한 파일은 다시 내려받게 됩니다.
Hugging Face 다운로드는 계속 Hugging Face 캐시(`~/.cache/huggingface/hub`, 또는 `MGS_CACHE`, `HF_HUB_CACHE`, `HF_HOME`에 지정한 폴더)를 사용합니다.

### 오래된 모델 파일

다음과 같은 오래된 변환 파일을 위한 호환 코드도 제거되었습니다.
이런 파일을 사용하려면 원본 모델을 다시 변환하거나 현재 GGUF를 내려받으세요.

| 오래된 파일 | 현재 동작 |
| --- | --- |
| 2025년 8월 이전에 변환된 Grok-1(`grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` 또는 `grok.attn_logit_softcapping`이 없음) | 로드되지 않습니다: 키 누락. |
| 분할되지 않은 `blk.N.attn_kv_b` 텐서가 있는 Kimi-Linear | 로드되지 않습니다: `attn_k_b`/`attn_v_b` 누락. 이 경로는 이미 그래프를 구성하는 단계에서 실패했습니다. |
| 병합된 `ssm_in` 텐서만 있는 Qwen3-Next 파일 | 로드되지 않습니다: `blk.0.attn_qkv.weight` 텐서 누락. |
| `expert_gating_func`(`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned`(Granite 4.0-H) 또는 `full_attention_interval`(Qwen3-Next)이 없는 파일, 그리고 레이어별 배열 대신 단일 SWA 주기를 가진 Cohere2-MoE 파일 | 로드되지 않습니다: 키 누락. |
| `tokenizer.ggml.pre`가 없거나 사전 토크나이저 이름이 `llama3`, `llama-v3`, `jina-es`, `jina-de`, `llada-moe`인 BPE 어휘, 그리고 병합 규칙(merges)이 없는 Kimi-K2 토크나이저 | 로드되지 않습니다: 키 누락, 알 수 없는 사전 토크나이저 유형 또는 병합 규칙 누락. |
| GGUF 버전 1 또는 2 파일 | 로드되지 않습니다: GGUF 버전 3만 받습니다. |
| 예전 일회성 스크립트로 만든 mmproj 파일: 프로젝터 유형 `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler`(MiniCPM-V 2.x~4.5, MiniCPM-o 2.6), `adapter`(GLM-Edge-V), 그리고 프로젝터 유형이 없는 파일 | 프로젝터가 로드되지 않습니다: 알 수 없거나 누락된 프로젝터 유형. |
| `ffn_up`과 `ffn_down`이 뒤바뀐 예전 변환의 mmproj 파일 | 더 이상 보정하지 않으며, 프로젝터가 실패합니다. |
| `image_min_pixels`/`image_max_pixels`가 없는 Kimi-K2.5 mmproj와 `preproc_image_size`가 없는 Step3-VL | 프로젝터가 로드되지 않습니다: 키 또는 텐서 누락. |
| `tokenizer.ggml.fim_*_token_id` 키 대신 예전 `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id`, `tokenizer.ggml.middle_token_id` 키만 있는 파일 | 로드됩니다. 채우기(infill) 토큰은 텍스트(예: `<PRE>` 또는 `<\|fim_prefix\|>`)로 인식되며, 찾지 못하면 채우기가 동작하지 않습니다. |

현재 변환 파일이나 지원 계열에서 가장 많이 내려받은 공개 GGUF가 여전히 의존하는 대체 처리는 Hugging Face의 헤더를 확인한 뒤 유지했습니다.
`llama-bpe`, Tekken 및 유사한 사전 토크나이저의 기본 `add_bos`, `full_attention_interval`에서 도출하는 Qwen3.6의 순환 레이어 배치(가장 많이 내려받은 Qwen3.6 GGUF에는 `recurrent_layers`가 없음), 그리고 `attention.indexer.types`가 없는 GLM-5.2/5.3 파일을 위한 내장 GLM-5.2 인덱서 배치가 여기에 해당합니다.

<a id="maintenance"></a>
## 유지 관리와 문제 해결

- 설치된 버전은 `mgs --version`으로 확인하세요(예를 들어 `version: 0.1.0`을 출력합니다). 로컬 빌드는 마지막에 이 명령을 실행하며, CMake는 구성하는 동안 `mgs version: 0.1.0`을 출력합니다. 인터페이스는 설정 → 표시 → **빌드 버전 정보 표시**(오른쪽 아래 모서리)와 모델 정보 대화 상자의 **빌드 정보**에 같은 버전을 표시합니다. 프로젝트가 바뀔 때마다 마지막 숫자가 올라갑니다: 0.1.0, 0.1.1, 0.1.2 등.
- `main`에서 업데이트하려면 `curl | bash`를 다시 실행하고, 바이너리를 다시 빌드해 교체하려면 해당 로컬 설치 명령을 다시 실행하세요. 그런 다음 서비스를 다시 시작하세요. Debian에서는 `systemctl restart moe-gguf-server`, Alpine에서는 `rc-service moe-gguf-server restart`입니다.
- `/root/webapp-install.log`를 읽고, Debian에서는 `journalctl -u moe-gguf-server`, Alpine에서는 `/var/log/moe-gguf-server.log`를 확인하세요. Apache 로그는 `/var/www/DOMAIN-logs/` 아래에 있습니다.
- CUDA 컴파일을 사용할 수 없으면 Toolkit을 별도로 설치하거나 `--cpu`를 사용하세요. NCCL을 찾을 수 없으면 저장소 사용 가능 여부와 CUDA 버전을 확인하거나, 명시적으로 `--no-nccl`을 선택하세요.
- UI 컴파일이 실패하면 Node.js와 npm 버전, 의존성 다운로드, 로그를 확인하세요. 미리 빌드된 업스트림 인터페이스로 대체하지 마세요.
- 배포용 빌드에서는 소스 구조가 바뀐 뒤 `_/temp/` 안의 새 디렉터리를 사용하세요. SSHFS에 있는 프로젝트 사본에서는 실행 파일 실행이 금지될 수 있으므로, 프로덕션에서는 로컬 디스크를 사용하세요.
- 로컬 빌드의 경우 현재 `main`을 내려받으려면 위의 한 줄 명령을, 로컬 소스를 사용하려면 `bash build/build-local.sh`를 다시 실행하고, 직접 실행한 사본을 다시 시작하세요. 실패하거나 `lib/` 밖의 라이브러리가 보고되면 터미널 출력을 읽어 보세요. 임시 `build.log`는 빌드 디렉터리와 함께 삭제됩니다.
- 오래된 클라이언트에서 API 404 오류가 나면 보통 기본 URL을 `/api/v1`로 바꿔야 합니다. 설치된 API 키는 Authorization 헤더에 넣어야 합니다.

이 프로젝트의 규칙에 따라 코딩 어시스턴트는 테스트, 배포 실행, GPU 벤치마크, 모바일 브라우저 자동 검사를 수행하지 않습니다.
정적 구문 검사는 런타임 동작을 보장하지 않습니다.
구현 지도와 확장 지점은 [CODE.ko-KR.md](CODE.ko-KR.md)를 참조하세요.

<a id="license"></a>
## 라이선스

`mgs`는 [PolyForm NonCommercial License 1.0.0](../LICENSE)에 따라 배포됩니다.
개인 사용, 학습, 연구, 취미 프로젝트, 그리고 자선 단체, 교육 기관, 공공 연구 기관, 정부 기관의 사용 등 비상업적 목적이라면 어떤 용도로든 사용, 수정, 공유할 수 있습니다.
회사가 업무에 사용하는 것과 같은 상업적 사용에는 nipegun으로부터 별도의 라이선스를 받아야 합니다.

이 라이선스는 nipegun의 기여분에만 적용됩니다.
llama.cpp(MIT, The ggml authors), FreeToken(Apache 2.0), 번들된 라이브러리에서 온 부분은 각자의 라이선스를 유지합니다. 그 고지는 `build/licenses/`에 있으며, 로컬 빌드는 이를 `~/.local/share/doc/mgs/licenses/`에 설치합니다.
