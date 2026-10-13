# 기술 가이드

[Deutsch (Deutschland)](CODE.de-DE.md) · [English (United Kingdom)](CODE.en-GB.md) · [English (United States)](CODE.md) · [Español (Argentina)](CODE.es-AR.md) · [Español (España)](CODE.es-ES.md) · [Français (France)](CODE.fr-FR.md) · [עברית (ישראל)](CODE.he-IL.md) · [हिन्दी (भारत)](CODE.hi-IN.md) · [Italiano (Italia)](CODE.it-IT.md) · [日本語 (日本)](CODE.ja-JP.md) · [한국어(대한민국)](CODE.ko-KR.md) · [Português (Brasil)](CODE.pt-BR.md) · [Português (Portugal)](CODE.pt-PT.md) · [Русский (Россия)](CODE.ru-RU.md) · [中文（中国）](CODE.zh-CN.md)

[아키텍처 및 설계 결정](#architecture) · [모듈 맵](#modules) · [주요 심볼 색인](#symbols) · [주요 호출 경로](#flows) · [진입점 및 라우트](#routes) · [영향 분석](#impact) · [확장 지점](#extensions) · [업스트림 검토 — 2026-10-02](#upstream-review)

<a id="architecture"></a>
## 아키텍처 및 설계 결정

**MoE 모델 로딩과 MTP 우선순위.** 엔진은 `backend/src/CMakeLists.txt`에 명시적으로 나열된 소스 목록을 컴파일합니다. dense 전용 디코더 구현은 제거되었습니다. 공유 그래프는 이를 사용하는 MoE 구현(Phi-MoE, Granite, Nemotron, Ernie, Nomic 포함)에 속하며, MoE 모델 내부의 dense 블록은 여전히 필요합니다. `mgs_model_create`는 가중치를 로드하기 전에 전문가 수를 확인하고 dense 메인 모델을 거부합니다. MTP로 명시적으로 로드한 헤드 전용 GGUF는 예외로, dense 예측 헤드가 MoE 타깃 모델에 쓰일 수 있습니다. Qwen35에는 MTP 그래프만 남아 있습니다.

MTP 배치는 선택된 모든 GPU를 고려하고, 백엔드의 할당 크기와 정렬을 반영하며, 출력 헤드(또는 이와 묶인 토큰 임베딩)를 포함합니다. 먼저 모든 헤드를 담을 수 있는 GPU를 찾고, 그런 GPU가 없으면 가장 큰 그룹부터 여러 GPU에 나누어 배치합니다. 선택된 어떤 GPU에도 담을 수 없는 그룹만 RAM으로 밀려납니다. 별도의 `dMtpContexts`는 트렁크 컨텍스트보다 먼저 할당되고 업로드되며, 텐서 오버라이드로는 이 배치를 대체할 수 없습니다. 외부 MTP 가중치는 `--fit off`일 때를 포함해 항상 타깃 모델보다 먼저 미리 로드되고, 두 번째 사본을 로드하지 않은 채 추측 디코딩 초기화 단계로 넘겨집니다. fit 추정은 이미 할당된 가중치를 두 번 세지 않으면서 이들의 컨텍스트/연산 메모리를 예약합니다.

**소스 규칙과 CLI 검증.** 프로젝트 식별자는 프로젝트 규칙에 정해진 함수, 메서드, 매개변수, 값 접두사를 붙인 CamelCase를 사용합니다. 공개 C API, GGUF 이름, 프로토콜 필드, 서드파티 이름, 프레임워크가 요구하는 콜백/컴포넌트 이름은 기존 계약을 그대로 유지합니다. C++ 서식은 공백 두 칸 들여쓰기를 사용합니다. 명명 규칙은 어떤 Linux 빌드에서도 컴파일되지 않는 전처리기 분기(Windows, macOS, 디버그 매크로, `MGS_LLGUIDANCE`, `#if 0`)에도 적용됩니다. 이런 곳에서는 컴파일러가 옛 이름을 잡아낼 수 없으므로, 이름을 바꿀 때는 해당 분기를 직접 확인해야 합니다. `--chat-template-kwargs`는 `vEnableThinking`을 거부하므로 `--reasoning on` 또는 `--reasoning off`를 사용하세요. 밑줄이 들어간 옵션 표기는 변환되지 않으며, 명시적으로 반복할 수 있는 `--spec-type`을 제외하고 반복된 옵션은 거부됩니다. `Github-Repo-Push.sh`는 `set -euo pipefail`, `fCleanup`, 그 EXIT 트랩, 명시적으로 결과를 확인하는 `fMain`을 사용하며, 게시 단계가 하나라도 실패하면 스크립트가 중단됩니다.

`frontend/`에는 UI가, `backend/`에는 엔진, 서버, 의존성이,
`deploy/`에는 설치 프로그램, 서비스, 에셋 스크립트가, `build/`에는 로컬
자체 완결형 빌드 스크립트와 재배치 가능한 런처가, `doc/`에는 문서가, `_/`에는 로컬 작업 자료가 있습니다. 가져온 의존성은 그대로 보존됩니다.

Svelte 인터페이스는 `frontend/`에, C++ 엔진/서버는 `backend/`에, 설치 관련 파일은 `deploy/`에 있습니다. 엔진은 `src/`, `common/`, `include/`, `ggml/`, `vendor/`, `tools/` 사이의 상대적 관계를 그대로 유지합니다. `build/CMakeLists.txt`의 CMake 진입점은 백엔드에 처리를 위임하며, UI 라이브러리는 `frontend/`에서 빌드되어 실행 파일에 내장됩니다. CMake 타깃은 `mgs-engine`(엔진 라이브러리, `backend/src/`), `mgs-common-base`와 `mgs-common`(`backend/common/`), `mtmd`, `server-context`, `mgs-server-impl`(`backend/tools/`), `mgs-ui`(내장 인터페이스), 그리고 실행 파일 타깃 `mgs`이며, 빌드 산출물과 설치 모두 `mgs`라는 이름을 사용합니다.

**명명 규칙.** llama.cpp에서 물려받은 코드는 프로젝트 고유의 이름을 사용합니다. `backend/include/mgs.h`와 `mgs-cpp.h`의 공개 C API는 `mgs_` 및 `MGS_` 접두사를 쓰고, 엔진 소스는 `backend/src/mgs*.{h,cpp}`, 서버 진입 함수는 `mgs_server`, 환경 변수는 `MGS_*`(`MGS_ARG_*`, `MGS_API_KEY`, `MGS_CACHE`, `MGS_MOE_CACHE*`, `MGS_SERVER_*`), CMake 옵션은 `MGS_*`, Prometheus 메트릭은 `mgs:*`, CORS 프록시 헤더 접두사는 `x-mgs-proxy-header-`, 브라우저 저장소는 `MgsUi`, 다운로드 캐시는 `~/.cache/mgs/`, tinyBLAS CPU 커널은 `ggml-cpu/tinyblas/`의 `GGML_TINYBLAS`입니다. 남아 있는 `llama` 표기는 의도된 것으로, 다음이 이에 해당합니다. llama.cpp 라이선스 고지(`LICENSE`, `build/licenses/LICENSE-llama.cpp`)와 README 태그라인, 저장소 링크 없이 업스트림 프로젝트 자체를 언급하는 부분(출처, 비교), GGUF 파일 안에 저장된 값(SentencePiece 토크나이저 모델 `llama`와 `llama-bpe`, `llama4` 사전 토크나이저, 그리고 이를 나타내는 `cMgsVocabPreTypeLlama3`/`LLAMA4` 식별자와 `fUnicodeRegexSplitCustomLlama3`), 주석에 나오는 서드파티 모델과 코드의 이름(CodeLlama, Llama-3_1-Nemotron, LLaDA, Ultravox 모델, YaRN 참조 구현 `LlamaYaRNScaledRotaryEmbedding.py`), tinyBLAS 소스에 있는 참고 문헌 하나, virtgpu 백엔드가 사용하는 virglrenderer API 원격 처리(API remoting)의 `APIR_LLAMA_CPP_*` 변수, 그리고 ‘llamada’ 같은 스페인어 단어입니다. Meta의 Llama 아키텍처(`llama`, `llama4`, `llama-embed`)와 그 내장 채팅 템플릿, Llama 4 비전 프로젝터는 제거되었습니다.

HTTP 등록 단계에서는 공개되는 각 라우트에 API 접두사를 붙입니다. 핸들러는 접두사가 없는 정규 경로를 받으며, 라우터는 모델별 자식 프로세스로 요청을 전달할 때 `/api`를 덧붙입니다. 공개 인증 허용 목록은 완전한 외부 경로를 사용합니다. UI는 `/`에 그대로 있으며, `/api/config.js`가 애플리케이션 모듈이 로드되기 전에 접두사를 알려 줍니다. Swagger와 OpenAPI는 모델 로딩 중에도 `/api/doc/` 아래에서 제공됩니다. 라우트/메서드 레지스트리가 실시간 목록을 제공하고, completion 매개변수는 `server-schema`에서 가져옵니다.

JSON 카탈로그에는 15개 로캘의 인터페이스 번역이 들어 있습니다: `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`. `aLanguages`와 `dCatalogs`는 이들을 로캘 코드의 알파벳순으로 등록하며, `en-US`가 기본값이자 메시지 대체(fallback) 언어로 남습니다. 각 카탈로그에는 15개 언어 전체의 현지화된 이름이 포함됩니다. `fTranslate`는 대화 데이터를 수정하지 않고 이름 붙은 값을 보간합니다. 새 번역은 영어 복수형 접미사를 보간하는 대신 삭제/가져오기 메시지에 개수 레이블을 사용합니다. 언어를 바꾸면 그 값이 서버의 설정 문서에 저장되고 인터페이스가 다시 로드되므로, 모듈을 가져올 때 계산된 레이블도 함께 바뀝니다. `/api/config.js`는 모듈이 로드되기 전에 저장된 언어를 알려 주며, URL의 `?lang=`이 이를 재정의합니다. 루트 레이아웃은 HTML 언어와 방향을 설정하고(`rtl`은 히브리어에만 적용), 내비게이션은 논리적 위치 지정을 사용하며, 채팅 입력과 사용자의 일반 텍스트에는 `dir="auto"`를 쓰고, 코드/터미널 텍스트는 왼쪽에서 오른쪽 방향을 유지합니다. 프레임워크 계약, 프로토콜 필드, 가져온 심볼은 원래 이름을 유지하며, 프로젝트 고유의 함수는 필수 명명 접두사를 따릅니다.

**문서 언어.** README, MANUAL, CODE 문서는 인터페이스와 같은 15개 로캘로 제공됩니다. `README.md`(en-US)는 루트에 있고 그 번역본은 `doc/README.<locale>.md`이며, 매뉴얼과 이 가이드는 `doc/MANUAL.md`와 `doc/CODE.md`(en-US), 그리고 `doc/MANUAL.<locale>.md`와 `doc/CODE.<locale>.md`입니다. 모든 언어판은 나머지 모든 언어판을 로캘 코드의 알파벳순으로 연결하는 언어 링크 줄로 시작합니다. 제목에는 모든 언어에서 같은 `<a id>` 앵커가 붙어 있으므로, 문서 간 링크(예: `MANUAL.<locale>.md#local-build`)는 모든 언어판에서 작동합니다. 코드 블록, 명령, 경로, 옵션, 식별자는 모든 번역본에서 동일하며 설명 문장만 달라집니다. 히브리어판은 일반 Markdown입니다. GitHub와 Gitea가 각 단락의 방향을 감지하므로, 본문은 오른쪽에서 왼쪽으로 읽히고 코드 블록은 왼쪽에서 오른쪽 방향을 유지합니다.

설치에는 root 권한이 필요하며, 명령 실패를 명시적으로 처리하고 중간 파일은 `_/temp/` 아래에만 둡니다. 프로젝트의 변경 사항이 빠진 업스트림 인터페이스를 내려받는 대신 UI를 로컬에서 빌드합니다. 배포 과정에서는 권한이 없는 추론 서비스와 웹 서비스를 만듭니다. 전용 Apache 인스턴스는 시작할 때마다 HTTP 포트는 11080부터, HTTPS 포트는 11443부터 찾으며, 추론은 loopback:8080에 머뭅니다. HAProxy는 구성하지 않습니다. 프로그램 구성은 서비스 계정의 `~/.mgs/config/`인 `/opt/moe-gguf-server/.mgs/config/`에 있으며, 모두 이 계정이 소유하고 모드는 0600입니다. `configure-web.sh`가 쓰는 `api.key`, `web-domain`, `apache2.conf.in`(Apache `ServerRoot`)과 `mgs`가 쓰는 `ui-settings.json`이 여기에 해당하며, `/etc` 아래에는 시스템 통합(systemd 또는 OpenRC 유닛)만 남습니다. 대화는 브라우저에 저장되며, 백엔드 사용자 데이터베이스는 없습니다.

서버 자체는 기본적으로 HTTPS를 활성화합니다. `common_params`는 11443에서 시작하고 `server_http_context`는 처음 발견한 빈 포트에 바인딩하며, 이어서 11080부터 HTTP 포트를 찾습니다. 두 번째 리스너는 호스트, 경로, 쿼리를 유지한 채 `Cache-Control: no-store`와 함께 선택된 HTTPS 포트로 308을 반환합니다. `--port`는 메인 포트를 고정하며(0이면 OS에 맡김), `--http-port`는 HTTP 포트 탐색의 시작점을 바꿉니다. 주소가 이미 사용 중인 충돌일 때만 다음 포트로 넘어가고, 그 밖의 오류는 시작을 중단시킵니다. `server-tls.cpp`는 제공된 PEM 쌍을 로드하거나, 제공된 것이 없으면 `~/.mgs/certificates/`의 자체 서명 신원 정보(`fPersistentServerIdentity`)를 로드합니다. `fullchain.pem`과 `privkey.pem`이 서로 일치하고 인증서가 만료되지 않은 동안에는 이를 재사용하고, 그렇지 않으면 ECDSA P-256 인증서(유효 기간 365일, localhost, 루프백, 구성된 호스트, 머신 이름, 인터페이스 주소에 대한 SAN 포함)를 생성해 임시 파일과 이름 바꾸기를 거쳐 저장합니다(폴더 0700, 키 0600). 따라서 모든 실행과 모든 빌드가 같은 인증서를 제공하고, 브라우저는 신뢰 예외를 계속 유지합니다. 외부 명령은 호출하지 않으며, 폴더에 쓸 수 없으면 해당 실행은 메모리상의 신원 정보를 사용합니다. OpenSSL 지원이 없으면 HTTPS는 명시적으로 실패합니다. `--http`는 의도적인 평문 HTTP를 선택하며, 라우터 자식 프로세스와 8080 포트의 Apache 백엔드가 이를 사용합니다. Unix 소켓과 Vertex AI는 HTTP를 유지합니다. 소멸자는 두 리스너를 모두 중지하고 종료될 때까지 기다립니다(join). 이러한 결정은 설치 프로그램이나 런처와 무관하게 C++ 안에 구현되어 있습니다.

`build/build-local.sh`는 워크스테이션에서 수동으로 사용하기 위한 것으로, root가 필요 없고 패키지나 서비스를 설치하지 않으며 실행 중인 것을 아무것도 남기지 않습니다. 완전한 로컬 소스 트리를 복사하거나, `curl | bash`를 통해 또는 단독 스크립트로 실행된 경우에는 Git 없이 GitHub의 `main` 아카이브를 내려받습니다. 그런 다음 `/tmp/moe-gguf-server-build.XXXXXXXX/`에서 `mgs` 타깃을 빌드하고, 재배치 가능한 자체 완결형 사본을 접두사(기본값 `~/.local`, 현재 사용자 전용, `--prefix`로 변경 가능) 아래에 설치합니다: `bin/mgs`, `lib/mgs/`, `share/doc/mgs/licenses/`. `lib/mgs/`에는 서버 바이너리, `ldd`로 찾은 전이적 의존성 전체, glibc 동적 로더, 그리고 glibc가 `dlopen`으로 여는 NSS 모듈이 들어 있습니다. 커널은 절대 경로로 된 `PT_INTERP`만 인정하므로, `bin/mgs`는 `/proc/self/exe`를 읽어 `bin/`에서 접두사로 올라간 뒤 나머지 인수와 함께 `lib/mgs/<loader> lib/mgs/mgs`를 실행하는 정적 런처(`build/mgs-launcher.c`)입니다. 접두사 안의 어떤 것도 그 절대 위치에 의존하지 않습니다. 서버 바이너리와 번들된 라이브러리에는 `DT_RPATH` `$ORIGIN`이 설정되어 있으며, 이는 `LD_LIBRARY_PATH`보다 우선하기 때문에 의도적으로 선택한 것입니다. 이 경우 서버의 프로세스 이미지는 로더가 되므로, 런처가 `MGS_EXECUTABLE`을 내보내고 `fGetServerExecPath`가 그 값을 반환해 라우터 자식 프로세스가 런처를 통해 다시 진입하도록 합니다. NVIDIA 드라이버 라이브러리(`libcuda`, `libnvidia-*`)는 커널 모듈과 일치해야 하므로 절대 번들하지 않습니다. `lib/mgs/` 안의 표식 파일은 이 스크립트가 만들지 않은 설치를 덮어쓰지 못하도록 막습니다.

로컬 실행마다 `mktemp -d /tmp/moe-gguf-server-build.XXXXXXXX`로 만든 새 비공개 디렉터리를 사용합니다. 내려받은 아카이브, 그 멤버 목록, 복사하거나 압축을 푼 소스, CMake 출력, npm/Node/CUDA 캐시, 임시 파일, 로그, 조립된 `bundle/`은 모두 그 안에 머뭅니다. 컴파일러 자동 캐싱은 비활성화되어 있으므로(`GGML_CCACHE=OFF`, `CCACHE_DISABLE=1`) 빌드가 끝난 뒤에 살아남는 캐시 서버는 없습니다. 산출물이 준비되면 스크립트는 `flock`으로 접두사 디렉터리 자체를 잠그고, 완성된 파일을 최종 위치 옆의 임시 항목으로 복사한 다음, 같은 파일 시스템 안에서 이름을 바꿔 제자리에 놓습니다. `fCleanup`은 로그 기록 프로세스가 끝나기를 기다린 뒤, 성공했을 때와 실패했을 때, 그리고 처리된 INT/TERM/HUP 신호를 받았을 때 비공개 작업 공간을 제거합니다. 소스 트리나 `/opt/`에는 빌드 파일을 쓰지 않으며, 로컬 빌더에는 `--build-dir` 옵션이 없습니다.

진입 스크립트의 경로는 `fMain`을 호출하기 전에 `${BASH_SOURCE[0]:-}`로 캡처하므로, `set -u` 상태에서도 stdin으로 실행할 수 있습니다. 옵션은 소스를 찾기 전에 파싱되므로 `--help`에는 소스 트리도, 소스 다운로드도 필요하지 않습니다. `fHasProjectSources`는 빌드 진입점, 런처, 프런트엔드 매니페스트, 라이선스 디렉터리를 확인합니다. 완전한 로컬 사본은 기존 제외 규칙을 적용해 스테이징합니다. 그렇지 않으면 `fDownloadSources`가 이미 설치된 curl/gzip을 요구하고, 재시도를 포함해 HTTPS로 내려받은 뒤 아카이브 경로를 검증하고, 비공개 `source/` 디렉터리에 압축을 푼 다음 필수 파일을 확인합니다. `fMain`은 stdin을 `/dev/null`에서 리디렉션한 상태로 실행되므로, 빌드 도구가 파이프로 전달된 스크립트를 읽어 들일 수 없습니다. 원격 실행 시 옵션은 `bash -s --`로 전달하며, 패키지는 절대 설치하지 않습니다.


Debian 진입점은 독립적으로 동작합니다. 완전한 로컬 체크아웃을 감지하거나 GitHub에서 `main`을 내려받습니다. 원격 모드는 `/opt/moe-gguf-server-source`(`MGS_INSTALL_SOURCE_DIR`로 재정의 가능)를 사용하고, 소스 출처 표식을 확인하며, 동기화와 컴파일이 끝날 때까지 잠금을 유지합니다. `rsync --checksum --no-times --delete`는 더 이상 필요 없는 소스를 제거하고, `_/`는 보존하며, 변경된 파일에는 새 타임스탬프를 부여합니다. 헬퍼는 stdin이 닫힌 별도의 Bash에서 실행되며, 부모의 로그 스트림을 중복해서 쓰지 않습니다.

하위 호환성 계층은 유지하지 않습니다. 더 이상 사용되지 않는 CLI 별칭과 제거된 옵션의 잔재(tombstone), 중복된 긴 옵션 이름(각 옵션은 짧은 이름과 긴 이름을 하나씩만 가짐), `MGS_ARG_NO_*` 환경 변수 대체 처리와 이전 옵션 이름을 딴 환경 변수, 레거시 HTTP 라우트 별칭(`/completion`, `/embedding`, `/reranking`)과 중복된 요청 필드(`thinking_budget_tokens`, 임베딩 라우트의 `content`), `deepseek-legacy` 추론 형식, Jinja가 아닌 템플릿 엔진(`--jinja`/`--no-jinja`, `mgs-chat.cpp`의 내장 C++ 템플릿과 `mgs_chat_apply_template`: 모든 채팅 템플릿은 Jinja를 거치며, `chatml`은 내장 ChatML 소스를 가리킴), 옛 CMake 변수 이름, 브라우저 데이터 마이그레이션, 설치 프로그램의 호환용 심(shim)이 모두 제거되었으므로, 각 옵션, 라우트, 변수, 저장된 값에는 현재 이름 하나만 있습니다. 이 프로젝트는 한 번도 배포된 적이 없으므로 공개 mgs/GGML/mtmd C API에도 더 이상 사용되지 않는 심볼은 남아 있지 않으며, 현재 함수만 남아 있습니다(제거된 함수는 [MANUAL.ko-KR.md](MANUAL.ko-KR.md#removed-compatibility)에 나열되어 있음). 예외는 의도된 것으로, 다음과 같습니다. 지원 계열의 현재 변환본이나 가장 많이 다운로드된 공개 GGUF가 여전히 의존하는 대체 처리를 통한 GGUF 파일 로딩(Hugging Face에서 해당 파일의 헤더로 확인함: `full_attention_interval`에서 도출하는 Qwen3.6 순환(recurrent) 레이아웃과, `attention.indexer.types`가 없는 GLM-5.2/5.3 파일을 위한 내장 GLM-5.2 인덱서 레이아웃), 툴체인과 운영 체제의 버전 가드, 그리고 외부 프로토콜(OpenAI, Anthropic, TEI/Jina, GCP Vertex, MCP) 및 ggml-virtgpu의 virglrenderer 게스트 같은 외부 상대와의 상호 운용성입니다. 옛 파일을 위한 호환용 심도 제거되었습니다: Grok-1, MiniCPM, Kimi-Linear의 분할되지 않은 `attn_kv_b`(비 MLA) 경로, Qwen3-Next의 `ssm_in` 경로, 누락된 `expert_gating_func`, `rope.scaling.finetuned`, `full_attention_interval` 키, Cohere2-MoE의 스칼라 SWA 주기, 누락된 사전 토크나이저와 Kimi-K2 병합(merge) 정보, `llama3`, `llama-v3`, `jina-es`, `jina-de`, `llada-moe` 사전 토크나이저 이름, `rope.scale_linear`, 초기 Gemma 2, GGUF 버전 1과 2, 옛 `tokenizer.ggml.prefix/suffix/middle_token_id` 키, 그리고 `mtmd`에서는 옛 일회성 스크립트의 프로젝터(`mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler`, `adapter`, 프로젝터 유형이 없는 파일), 옛 수술식(surgery) 변환에서 뒤바뀐 FFN, 누락된 Kimi-K2.5 및 Step3-VL 키의 기본값입니다. 사용자의 결정에 따라 아키텍처 전체가 제거되기도 했습니다: Gemma 4와 V4.1보다 오래된 모든 DeepSeek 아키텍처입니다(아래의 지원 계열 참조).

수동으로 구성하려면 프로젝트 루트에서 `cmake -S build -B _/temp/cmake`를 실행합니다. 빌드 스크립트는 `build/`를 소스와 나란히 스테이징하고 `source/build`를 CMake 소스 디렉터리로 선택합니다. 배포 스크립트는 생성된 파일을 `_/temp/` 아래에 두며, 캐시된 구성이 다른 소스 디렉터리를 가리키면 `CMakeCache.txt`와 `CMakeFiles/`를 초기화합니다. 로컬 빌드는 항상 `/tmp/` 아래의 새로운 일회용 작업 공간을 사용합니다. cpp-httplib의 TLS 라이브러리는 `backend/CMakeLists.txt`에 선언된 옵션에 따라 `backend/vendor/cpp-httplib/CMakeLists.txt`에서 선택됩니다. `MGS_BUILD_BORINGSSL`과 `MGS_BUILD_LIBRESSL`(둘 다 기본값 off)은 해당 라이브러리를 내려받아 빌드하며, 이 순서대로 `MGS_OPENSSL`(기본값 on, 시스템 OpenSSL)보다 우선합니다. 셋 다 없으면 서버는 HTTPS를 지원하지 않습니다.

**버전.** 프로젝트 버전은 `backend/cmake/build-info.cmake`의 `MGS_VERSION` 한 곳에서만 정의되며(`0.1.0`부터 시작), 프로젝트에 변경이 있을 때마다 패치 번호가 올라갑니다(0.1.1, 0.1.2, ...). 로컬 빌드와 Debian 설치 프로그램은 같은 CMake 트리를 구성하므로 둘 다 같은 버전을 만들어 냅니다. CMake는 `mgs version: …`을 출력하고, `build-info.cpp`는 `mgs_version()`에서 이 버전을 반환하며, 이 값은 공유 라이브러리의 파일 버전(soname은 `.so.0` 유지), `mgs.pc`, `mgs-config.cmake`, 인터페이스의 `build.json`에도 설정됩니다. 이 프로젝트는 Git을 사용하지 않으므로 빌드 번호나 커밋이 없습니다. `--version`, `/api/props`의 `build_info` 속성, OpenAI `system_fingerprint`, 셸 MCP의 `serverInfo.version`, 다운로드 시의 `User-Agent`(`mgs/0.1.0`)는 모두 이 버전을 보고합니다. `deploy/scripts/ui-assets.cmake`는 `build.json`에 다른 버전이 들어 있으면 인터페이스를 다시 빌드하므로, 인터페이스 소스를 건드리지 않아도 재사용되는 `_/temp/` 빌드 트리에 버전 변경이 반영됩니다.

서식 구성은 폴더 단위로 적용됩니다. `.clang-format`은 `backend/`, `frontend/`, `build/`에 있고, `.editorconfig`는 이 폴더들과 `deploy/`, `tests/`에 있습니다. `build/licenses/`의 라이선스 고지는 `build/.editorconfig`를 상속합니다. 동등한 설정은 서로 일치하도록 유지하세요. 프로젝트 루트와 `doc/`의 파일은 UTF-8, LF, 공백 두 칸 들여쓰기로 편집기를 설정해야 합니다.

`doc/` 폴더에는 읽기용 문서가 들어 있습니다. Swagger 페이지는 `frontend/api.html`에 있으며 CMake가 이를 서버에 내장합니다. 백엔드는 요청이 들어올 때마다 등록된 라우트로부터 `/api/doc/openapi.json`을 생성하며, 정적인 `openapi.json` 소스 파일은 없습니다.

이 포크는 llama.cpp의 서버, 멀티모달 지원, GGML 백엔드를 유지하면서,
GPU 메모리와 시스템 RAM에 나뉘어 있는 가중치를 위한 MoE 최적화를 더했습니다. 호스트
등록과 전문가 프리페치는
thecodacus 포크의 `fable5/host-register` 및 `fable5/prefetch-experts` 브랜치에서
가져온 것입니다.
프리페치 슬롯은 gate/up/down 전문가 텐서를 담으며, 그 전송을
연산과 겹쳐서 수행합니다. 자동 MTP는 `draft-mtp`를 선택하기 전에 `nextn_predict_layers`, 모델 트렁크,
통합된 모든 헤드를 확인하며, 이 확인은 모든
GGUF 샤드를 대상으로 합니다. 튜닝과 제한 사항은 [매뉴얼](MANUAL.ko-KR.md#performance)을 참조하세요.

**MoE 전문가 캐시.** `backend/src/mgs-expert-cache.{h,cpp}`는 전문가가 시스템 메모리에 머무는 모든 레이어에 대해
가장 최근에 사용된 전문가를 GPU마다 하나씩 있는 디바이스 풀에 보관하며, 이 풀은 해당 GPU에 할당된 레이어들이 공유합니다(같은 GPU에서
전문가별 텐서 유형과 형태가 동일한 레이어 그룹마다 LRU 목록 하나). 슬롯 하나는 전문가 하나의 전문가별 텐서를 모두 담습니다.
가중치, 바이어스, 전문가별 스케일이 여기에 해당하며, gate/up 프로젝션은 분리되어 있을 수도 있고 병합되어 있을 수도 있습니다.
op 오프로드 한도(32 토큰, `GGML_OP_OFFLOAD_MIN_BATCH`)보다 작은 마이크로 배치에서는 `mBuildMoeFfn`이 전문가 체인을
두 번 구성합니다. 하나는 슬롯 식별자를 통해 풀을 읽는 디바이스 분기이고, 다른 하나는 호스트 가중치를 읽는 CPU 분기입니다.
먼저 CPU 분할 연산(`mPartition`)이 실행됩니다. 이 연산은 적중 수를 세고, 미스 가운데 대역폭 균형에 맞춘 몫
(약 m·B_upload/(B_upload+B_cpu), 최소 하나)을 업로드하며, 디바이스 분기보다 앞서 슬롯 식별자와 업로드 작업을
디바이스 스트림의 대기열에 넣습니다. 그동안 CPU 분기는 나머지 미스를 동시에 계산합니다.
다른 분기가 처리하는 전문가는 CPU 쪽에서 식별자 -1을 받고(CPU의 `mul_mat_id`/`add_id` 커널은 이를 건너뛰고 0을 씀),
디바이스 쪽에서는 빈 슬롯을 받습니다(위치 k는 빈 슬롯 k를 사용하므로, CUDA MMQ 헬퍼가 요구하는 대로 한 토큰 안에서
같은 슬롯이 반복되지 않음). 따라서 두 분기의 합은 반올림 오차 범위에서 일반 실행 결과와 같습니다.
스케줄러는 입력이 없는 분할(split)을 이전 분할과 동기화하되, 그 분할이 입력을 복사했거나 그 백엔드가 현재 버퍼 유형에
접근할 수 있을 때만 동기화합니다. 이렇게 하면 업스트림 #26040의 수정 사항을 유지하면서 두 분기가 겹쳐 실행될 수 있습니다.
업로드 대역폭은 캐시를 만들 때 실제 전문가로 측정하고, CPU 대역폭은 CPU 분기의 소요 시간으로부터 실행 중에 추정합니다.
큰 마이크로 배치에서는 전문가 프리페치를 그대로 사용합니다. `ggml_backend_sched_set_weight_upload`를 통해 캐시는 자신이 가진
전문가를 디바이스 간 복사로 프리페치 슬롯에 넣고, 나머지만 호스트 메모리에서 업로드합니다. 캐시는 컨텍스트를 할당하고 남은
디바이스 메모리에서 `--moe-cache-reserve`와 프리페치 슬롯을 뺀 만큼을 차지합니다. 캐시는 모든 컨텍스트와 프로젝터가 로드된 뒤
들어온 첫 요청 때 생성되며(공통 워밍업은 생성을 뒤로 미룸), 실행 중에 크기를 조정할 수 있고(`mgs_set_expert_cache_size`,
`POST /api/expert-cache`), `mgs_get_expert_cache_info`, `/api/expert-cache`, `/api/metrics`를 통해 카운터를 보고합니다.
설정은 `--moe-cache*` 옵션이 지정하는 `MGS_MOE_CACHE*` 환경 변수를 통해 전달되므로 공개 API는 바뀌지 않습니다.
캐시가 활성화되어 있고 RAM이 충분하면 `--fit`은 모든 전문가를 시스템 메모리에 두고 남는 디바이스 메모리를 캐시에 넘깁니다.
LRU 캐시가 정적인 전문가 레이어 전체보다 디코딩에 더 유리하기 때문입니다. GPU가 여러 개이면 dense 전용 레이어를
메모리 목표에 따라 여러 GPU에 분산하므로, 각 GPU는 자신이 맡은 레이어의 전문가를 캐시합니다. 업로드 대역폭은 GPU별로
측정하고, 요청된 크기는 각 GPU가 캐시하는 호스트 전문가의 양에 따라 GPU들에 나누며, 여러 마이크로 배치를 동시에 진행하는
파이프라인 병렬 처리에서는 캐시가 꺼진 상태로 유지됩니다. `--fit`은 로더가 오프로드하는 레이어로 번호를 매기는
nextn(MTP) 레이어도 함께 셉니다. 그렇지 않으면 MTP가 꺼져 있을 때 첫 번째 레이어가 CPU에 남았습니다.

**`--fit`에서의 컨텍스트 크기.** `fCommonParamsFitImpl`의 2단계는 사용자가 지정하지 않은 컨텍스트를 마치 모든 가중치가
디바이스 메모리에 들어가야 하는 것처럼 줄입니다. 그래서 예전에는 디바이스보다 큰 모델이, 전문가가 결국 시스템 메모리에
머무는데도 항상 `--fit-ctx`(4096 토큰)까지 떨어졌고 추론 응답이 잘렸습니다. dense 가중치가 들어가고 나면 `fRaiseContext`가
줄어든 컨텍스트와 학습 컨텍스트에서 각각 dense 전용 메모리 배치를 측정하고 선형 보간한 다음, KV 캐시와 그에 따라 커지는 버퍼가
dense 가중치를 올리고 남은 디바이스 메모리의 최대 4분의 1(`cContextShare`)을 차지할 때까지 컨텍스트를 늘립니다(256 단위로
내림하고 한 번 더 측정해 확인). 이 단계는 모든 전문가를 RAM에 두는 메모리 배치(전문가 캐시)가 선택된 뒤, 그리고 전문가 레이어
전체가 디바이스를 채우기 전에 실행되므로, 전문가는 항상 그 메모리의 4분의 3 이상을 확보합니다. 명시적으로 지정한 `--ctx-size`는
절대 바뀌지 않고, `--fit-ctx`는 최솟값으로 유지되며, KV 캐시가 시스템 메모리에 있을 때는 아무것도 바뀌지 않습니다.
초안 모델 컨텍스트는 타깃 모델의 컨텍스트 크기를 그대로 사용합니다(`mLoadModel`의 `vParamsDft.n_ctx = n_ctx`).
예전에는 컨텍스트 크기가 0이면 모델의 학습 컨텍스트에 맞춰 KV 캐시를 할당해, 전문가 캐시가 쓸 디바이스 메모리를 빼앗았습니다.

**자동 구성.** `mgs --model FILE`은 머신에 가장 빠른 설정을 스스로 골라야 하며, 명시적으로 지정한 옵션은 항상 우선합니다.
전문가가 시스템 메모리에 있으면 프롬프트의 모든 마이크로 배치가 디바이스에 없는 전문가를 PCIe로 스트리밍하므로
(RTX 4060 Ti에서 Qwen3.6-35B-A3B Q8_0의 경우 마이크로 배치당 약 2초), 프롬프트 처리 속도는 마이크로 배치 크기에 거의 비례해
빨라지는 반면 생성 속도는 이에 영향을 받지 않습니다. 2514 토큰짜리 프롬프트는 512에서 초당 215 토큰, 2048에서 408 토큰,
4096에서 737–755 토큰으로 빨라졌고, 생성 속도는 같았습니다. 컨텍스트를 늘리기 전에 `fRaiseBatch`는 선택된 메모리 배치에서
4096, 2048, 1024 토큰의 마이크로 배치를 시도하고, 추가 메모리가 dense 가중치를 올리고 남은 디바이스 메모리의 8분의 1(`cBatchShare`)
이내에 머무는 가장 큰 값을 유지합니다. 논리 배치도 함께 커집니다. 이 단계는 `-b`/`-ub`(또는 해당 변수)가 지정되지 않았을 때만
실행됩니다(`common_params::vBatchExplicit`, `pAutoBatch`로 전달). `common_init_result`는 결과로 나온 `n_batch`를 매개변수에 다시
복사합니다. 초안 모델과 MTP 컨텍스트가 이 매개변수를 복사하고, 타깃 모델의 모든 배치를 자신의 `n_batch` 크기 버퍼로 받기
때문입니다. 이들의 마이크로 배치는 기본값을 유지합니다. 기본 CPU 스레드 수(`fCommonDefaultThreads`, `-t`가 없을 때
`fPostprocessCpuParams`가 적용)는 수학 연산에 유용한 코어 수이며, 프로세스 선호도(affinity) 마스크에 있는 논리 CPU의 절반을
상한으로 합니다. MoE 추론에서 CPU가 맡는 부분은 메모리 대역폭에 묶여 있고, SMT가 보이지 않는 가상 머신에서 모든 논리 CPU를
사용하면 워커 스레드가 GPU를 구동하는 스레드와 경쟁했기 때문입니다(16 vCPU: 4~12 스레드는 서로 오차 범위 이내, 16 스레드는
13% 느림). SMT가 보이면 상한이 물리 코어 수와 같으므로, 베어 메탈 머신은 이전 기본값을 유지합니다. `common_init_result`는
스레드 수를 로그에 기록하고, `--fit`은 자신이 선택한 마이크로 배치와 컨텍스트를 로그에 기록합니다.

**MTP 배치.** MTP 가중치는 다른 모든 배치 규칙보다 우선해 디바이스 메모리에 놓입니다. 모델에 이식된 nextn 레이어와
`--spec-draft-model`로 전달한 MTP 전용 GGUF의 모든 레이어가 여기에 해당합니다.
`mgs_model_base::mLoadTensors`는 `n_gpu_layers`가 이들을 CPU에 남겨 두더라도 해당 레이어와 출력 레이어를 GPU에 할당하며,
`mgs_model_loader::mCreateTensor`는 이들을 호스트 메모리로 보내는 오버라이드(`--fit` 부분 레이어, `--cpu-moe`, `--n-cpu-moe`,
`--override-tensor`와 이에 대응하는 `--spec-draft-*` 옵션)를 무시합니다. 토큰 임베딩은 트렁크의 토큰 임베딩과 마찬가지로
CPU에 남습니다. 행 조회는 GPU에서 해도 얻는 것이 없기 때문입니다. MTP 레이어가 512 MiB의 여유를 두고 디바이스의 빈 메모리에
들어가지 않으면 일반 배치 규칙이 적용됩니다. `--fit`은 모델을 할당하지 않고 로드하는 방식으로 측정하므로, 디바이스에 놓인
MTP 가중치를 인식하고 나머지를 그 주위에 맞춰 배치합니다. 전문가 캐시는 남은 메모리를 차지합니다. 별도의 MTP GGUF를 사용하면
`fCommonModelParamsToMgs`는 사용되지 않는 사본을 유지하는 대신 타깃 모델에 이식된 MTP 레이어를
로드하지 않습니다.

**지원 계열.** 반드시 작동해야 하는 MoE 계열은 Qwen3.6(`qwen35moe`), Qwen3.8-Flash-Next(`qwen4exp`), GLM-5.3(`glm-dsa`), GLM-5.3-Flash(`glm5-next`), Kimi-K3(`kimi-k3`), DeepSeek-V4.1-Flash(`deepseek41`), MiniMax-M3(`minimax-m3`), Nemotron-3(`nemotron_h_moe`), MiMo-V2.6(`mimo2`)입니다. 사용자의 결정에 따라 Gemma 4(`gemma4`, `gemma4-assistant`, `gemma4v`, `gemma4uv`, `gemma4a`, `gemma4ua` 프로젝터, `peg-gemma4` 채팅 형식)와 V4.1보다 오래된 모든 DeepSeek 아키텍처가 제거되었습니다: `deepseek`, `deepseek2`(Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3, Youtu도 사용하던 아키텍처), `deepseekocr` 프로젝터를 포함한 `deepseek2-ocr`, `deepseek32`, `mistral4`, `deepseek4`. 이들의 GGUF는 알 수 없는 아키텍처로 간주되어 거부됩니다.

**Kimi-K3(`models/kimi-k3.cpp`).** KDA 선형 어텐션 레이어(유계 시그모이드 감쇠를 쓰는 풀랭크 게이트 `ssm_g`, 공유 함수 `fBuildGdnL2Norm`, `mBuildRecurrentAttn`의 채널별 게이트)가 시그모이드 출력 게이트를 갖춘 흡수형 `k_b`/`v_b` 형식의 MLA 레이어와 번갈아 배치됩니다. `ggml_dsv4_hc_pre`로 구성한 레이어 간 잔차 혼합은 `attn_res_score`, `ffn_res_score`, `output_res_score`를 읽습니다. MoE는 잠재(latent) 방식입니다. 라우터는 전체 폭에서, 896개의 전문가는 잠재 폭에서 동작하며(`ffn_routed_down` → 전문가 → `ffn_routed_norm` → `ffn_routed_up`), `situ` 활성화 함수(`cLlmFfnSitu`)는 `mBuildMoeFfn`에서 처리합니다. 이 모델에서는 `MGS_MAX_EXPERTS`가 1024입니다. 순환 상태는 롤백(`n_rs_seq`)을 지원합니다. 공개된 GGUF에는 MTP가 없습니다. 채팅 템플릿은 `<|open|>`/`<|sep|>`/`<|close|>`를 사용하며 `fCommonChatParamsInitKimiK3`가 파싱하고, 토크나이저는 `kimi-k2` 사전 토크나이저를 그대로 사용합니다.

**GLM-5.3-Flash(`models/glm5-next.cpp`)와 `glm5v`.** KDA 레이어, k-pool DSA 인덱서(`indexer_kpool_gate`, `indexer_kpool_ape`)를 갖춘 nope 전용 MLA 레이어, 융합된 hc 연산을 쓰는 하이퍼 커넥션 스트림, 그리고 `ggml_swiglu_clamp`를 쓰는 DeepSeek 방식의 MoE로 구성됩니다. 이 새 연산(CPU, CUDA, Vulkan)은 디코딩 중에도 CUDA mat-vec + GLU 융합을 유지합니다. NextN 블록은 DSA 레이어입니다. MTP는 `--model`만 지정해도 켜지며, 그 가중치는 VRAM 우선 배치를 따릅니다. 메모리는 `mgs_memory_hybrid_idx`입니다. `glm5v` 프로젝터는 glm4v 그래프를 재사용하며, 선택적인 post-conv 정규화, 클램프된 SwiGLU, 자체 전처리기, `<|begin_of_image|>`/`<|end_of_image|>` 마커를 갖습니다. `glm4`와 `glm5` 사전 토크나이저는 `ignore_merges`를 적용한 ChatGLM4를 사용합니다.

**Qwen3.8-Flash-Next(`models/qwen4exp.cpp`).** 저랭크 하이퍼 커넥션 믹서와 결합 단계(`ggml_dsv4_hc_pre_gated`, 항등 혼합을 쓰는 `ggml_dsv4_hc_post`), n-gram 해시 기반 레이어별 임베딩(그 테이블인 `per_layer_token_embd`는 약 29 GB이며, 매핑된 상태로 남아 필요할 때 행 프리페치와 함께 읽힘), 순환 캐시 안의 팽창(dilated) 합성곱 기록(`cache_ple_r_l`), 롤백을 지원하는 게이트 델타넷(gated delta-net) 레이어, QSA 블록 희소 어텐션 레이어(k-pool 인덱서 캐시, `ggml_lightning_indexer` 점수 계산, KQ 마스크 위에 얹는 top-k 선택 마스크로 이루어지며, 업스트림의 희소 커널이 이식되지 않았으므로 dense 플래시 어텐션으로 계산), 게이트가 있는 공유 전문가를 갖춘 MoE, IMROPE, 그리고 자체 하이퍼 커넥션 믹서와 초안 메모리를 갖춘 MTP 블록으로 구성됩니다. `mgs_memory_hybrid_idx`(`mgs-memory-hybrid-idx.{h,cpp}`)는 qwen4exp와 glm5-next를 위해 KV 캐시, 순환 캐시, k-pool 인덱서 캐시를 결합합니다. KV 셀은 n-gram 조회를 위해 토큰 ID(`vTok`)를 기록하며, `seq_pos`는 (위치, 셀) 쌍의 집합입니다.

**DeepSeek-V4.1-Flash(`models/deepseek41.cpp`, `mgs-kv-cache-dsv41.{h,cpp}`).** 이 아키텍처를 선언하는 공개 레이아웃은 두 가지입니다. 하나는 antirez의 DwarfStar 파일(`deepseek41.hidden_size`처럼 Hugging Face 구성에서 가져온 키, `deepseek41.config` JSON 문자열, 그리고 E4M3 값과 E8M0 스케일로 된 행당 264바이트의 I8 Engram 테이블)이고, 다른 하나는 llama.cpp 생태계의 파일(표준 키, ggml 유형으로 된 Engram 테이블)입니다. `fMgsDeepseek41AdaptMetadata`는 모델을 만들기 전에 첫 번째 레이아웃을 표준 키로 변환하며(점수 함수 → 게이팅 열거형, 스칼라 `swiglu_limit` → 레이어별 배열, 40개 또는 43개의 압축 비율, 소수의 누적 합으로 구한 Engram 오프셋), 로더는 Engram 프로젝션의 두 가지 이름을 모두 받아들입니다. 레이어 역할(KV 소스, 인덱스 키 소유자, top-k 소스)은 실제로 존재하는 텐서로부터 정해집니다. Engram 테이블(약 188 GiB)은 `cTensorExternal`로 생성됩니다. 크기는 집계되지만 절대 할당되지 않고, 프리페치는 그 앞에서 멈추며, 각 토큰의 행은 비공개 디스크립터에서 `pread`로 읽습니다(`mEngramReadRows`, E4M3+E8M0를 디코딩하거나 `to_float`로 역양자화). 캐시는 압축 비율마다 그룹 하나를 유지하고(비율 2는 게이트 풀링, 비율 1은 레이어 20부터 전체 길이 스트림), 인덱스 키 캐시는 키 소유자에만 두며, 리더(reader) 레이어는 소스 레이어의 저장소를 가리킵니다. Engram은 이전 토큰을 원시 캐시 셀에서 가져옵니다. 그래프는 지연된(lagged) 하이퍼 커넥션 혼합(원-핫 첫 복사, 마지막 FFN 혼합과 함께하는 최종 축약, HC 헤드 없음), q 헤드 정규화 없음, RoPE 이전의 잠재 표현에서 얻는 인덱스 키, top-k 재사용과 2단계 후보 블록 풀, 레이어 1과 14 앞의 Engram, 라우팅 전문가와 공유 전문가의 `ggml_swiglu_clamp`를 사용합니다. GGUF에 채팅 템플릿이 없으면 `DEEPSEEK41_TEMPLATE_SRC`(Jinja 소스)를 사용합니다. DSML 태그에는 앞에 공백이 붙으며, 서버는 `reasoning_effort`(1–100)를 템플릿에 전달합니다. MTP/DSpark와 비전은 없습니다. `dsv4_*` 및 `lightning_indexer` ggml 연산과 `vDsv4HcMult`/`aDsv4CompressRatios` 하이퍼파라미터는 Kimi-K3, GLM-5.3-Flash, Qwen3.8이 사용하므로 유지되며, Vulkan과 Metal에서는 이 연산들이 CPU에서 실행됩니다.

**컨텍스트 체크포인트.** 메모리를 롤백할 수 없는 모델(순환 레이어나 SWA 레이어)은
체크포인트에서 상태를 복원합니다. 체크포인트는 사용자, 어시스턴트 또는 도구 메시지를 시작하는 배치의 시작 지점(에이전트가
컨텍스트를 편집하는 경계), 프롬프트의 끝 근처, 그리고 이전 체크포인트보다 `--checkpoint-min-step` 토큰(기본값 1024) 이상
지난 모든 배치에서 만들어집니다. 메시지 경계에서 배치를 자르는 것은 그 배치의 마지막 마이크로 배치가 4분의 3 이상 찼을 때뿐입니다.
마이크로 배치가 하나 늘어날 때마다 오프로드된 전문가를 다시 스트리밍해야 하기 때문입니다.
`--ctx-checkpoints`에 도달하면 이웃과의 간격이 가장 좁은 체크포인트를 제거하므로, 남은 체크포인트는
컨텍스트 전체에 고르게 퍼져 있게 됩니다.

**모니터링과 상태 표시줄.** `GET /api/monitor`는 CPU 사용률(전체, 물리 프로세서별, 논리 CPU별 사용률을 `/proc/stat`의 변화량으로 계산합니다. 각 논리 CPU는 번호로 대응시켜 sysfs의 `physical_package_id`로 묶고, 각 프로세서의 모델명은 `/proc/cpuinfo`의 해당 블록에서 가져옵니다), RAM(`/proc/meminfo`), 모든 GPU(`dlopen("libnvidia-ml.so.1")`을 통한 NVML, 이어서 NVML이 나열하지 않은 GPU 드라이버의 모든 DRM 카드입니다. amdgpu는 sysfs에서 `gpu_busy_percent`와 VRAM을 읽고, i915, xe, nouveau, nvidia는 사용률 없이 표시되다가 모델이 사용하기 시작하면 해당 백엔드의 이름과 메모리로 보완됩니다. 디스플레이 전용 어댑터는 제외되며, 메모리 수치가 없는 GPU에는 UI가 VRAM 막대를 그리지 않습니다), 그리고 로드된 모델의 메모리를 용도별로 나누어 보고합니다. 메모리 조회는 절대 작업 대기열을 거치지 않습니다. 작업이 게시될 때마다 `vTimeLastTask`가 초기화되므로 2초마다 폴링하는 페이지 때문에 모델이 영원히 절전 상태에 들어가지 못하게 되고, `mCreateResponse`는 잠든 모델이 깨어날 때까지 기다리기 때문입니다. 대신 메인 루프가 `mLoadModel` 이후에, 슬롯이 작업하는 동안에는 최대 초당 한 번, 슬롯이 유휴 상태가 되면 항상 한 번(요청이 끝나면 그 프롬프트를 저장함) 스냅숏(`mRefreshMonitorSnapshot`)을 게시합니다. HTTP 스레드는 뮤텍스를 잡은 채 이를 복사하기만 하며, 절전 상태에 들어가면 `{"state": "sleeping"}`을 게시합니다. 스냅숏은 `mgs_get_memory_breakdown`(버퍼 유형별 모델, 컨텍스트, 연산 메모리), `mgs_model_get_weight_breakdown`(MTP 및 라우팅 전문가 가중치로, 모든 텐서를 훑기 때문에 로드 후 한 번만 계산), `mgs_get_expert_memory_breakdown`(전문가 캐시 버퍼와 스케줄러의 프리페치 슬롯으로, 일반 분석에서는 보이지 않음)을 합친 것입니다. 타깃 모델 위의 MTP 컨텍스트는 그 가중치를 공유하므로 KV 캐시와 연산 버퍼만 집계됩니다. 별도의 MTP GGUF는 MTP 가중치로, 그 밖의 초안 모델은 초안 가중치로 집계됩니다. 호스트 버퍼 유형(GPU의 고정(pinned) 호스트 메모리 포함)은 RAM 막대로, 디바이스 버퍼 유형은 해당 GPU로 들어가며, NVML과는 PCI 버스 ID(소문자, CUDA 가상 디바이스의 `-vN` 접미사 제외)로 대응시킵니다. `fMonitorReport`는 "other"(어떤 세그먼트로도 설명되지 않는 사용 메모리: 다른 프로세스, 드라이버 컨텍스트, 프로젝터)와 남은 여유 메모리를 추가합니다. 라우터 부모 프로세스는 자체 CUDA 컨텍스트를 만들게 되는 GPU 백엔드를 절대 초기화하지 않습니다. 대신 NVML로 GPU를 읽고 준비된 각 자식 프로세스에 `/api/monitor/memory`를 요청하며(제한 시간 1초), 로딩 중이거나 절전 중인 자식 프로세스는 자신의 상태만 보고합니다. `GET /api/monitor/usage`(사용률만 제공: 전체 CPU, 논리 CPU를 제외한 각 물리 프로세서, 각 GPU)는 모델이 로드되는 동안에도 허용됩니다. CPU와 GPU 측정값은 동시에 폴링하는 클라이언트들이 750 ms 동안 공유하며, 10초보다 오래된 CPU 기준값은 새로운 250 ms 구간으로 대체됩니다. UI에서 `cMonitorStore`는 구독자가 있을 때만 폴링하고 탭이 숨겨져 있는 동안에는 일시 중지하며, 상태 표시줄은 모니터링 페이지가 열려 있는 동안 그 보고서를 재사용합니다. 이 페이지는 사용률과 메모리만 그립니다. 보고서의 모델별 항목(상태, 상주 메모리, 전문가 캐시)은 다른 클라이언트를 위해 API에 남아 있습니다. 카드 제목은 굵게 표시한 디바이스 레이블(`gpu-N`과 `cpu-N`, 보고서 순서대로, GPU와 물리 프로세서마다 카드 하나)과 그 뒤의 디바이스 이름으로 구성됩니다. 모든 프로세서 카드에는 시스템 평균 부하가 표시되며, 모든 사용률 미터(`MonitorMeter`, 코어별 막대, 상태 표시줄)는 메모리 막대의 범주형 팔레트와 달리, 사용률만큼 빨간색(`--monitor-meter-used`)으로 채워지는 초록색 막대(`--monitor-meter-free`)입니다. 상태 표시줄은 화면 하단에 고정됩니다. `.has-status-bar`가 `--status-bar-height`를 설정하고 채팅 폼, 사이드바, 고정(sticky) 푸터, `body`의 안쪽 여백이 이 값을 더하므로, 상태 표시줄을 숨기면(설정 → 표시) 모든 오프셋이 예전과 똑같이 유지됩니다. 메모리 막대는 검증된 8가지 범주형 색상에 중립색 "other"를 더해 고정된 순서(`aMonitorSegmentGroups`)로 사용하므로, 서로 관련된 서버 세그먼트는 같은 색을 공유합니다(MTP와 초안 가중치, 전문가 캐시와 프리페치). 추론 메뉴는 항목의 크기를 내용에 맞춥니다(`w-max`, 이름에는 `whitespace-nowrap`). 전역 `min-width: 0`과 `overflow-wrap: anywhere` 때문에, 예전에는 힌트 옆에 있는 `flex-1` 이름이 한 글자씩 쪼개졌습니다.

**Vulkan 백엔드.** `build/build-local.sh --vulkan`과 Debian 설치 프로그램(`--vulkan`, 또는 `MGS_BUILD_BACKEND=vulkan`)은 CUDA를 끈 채 `-DGGML_VULKAN=ON`을 구성합니다. 셰이더는 빌드할 때 `glslc`로 컴파일되며, CMake에는 Vulkan과 SPIR-V 헤더(`libvulkan-dev`, `spirv-headers`)가 필요합니다. 설치 프로그램은 이 패키지들과 `mesa-vulkan-drivers`를 추가로 설치하며, 로컬 빌드는 `glslc`만 확인하고 구성이 실패하면 필요한 패키지 이름을 알려 주는 안내로 바꿔 보여 줍니다. 로컬 번들은 일반적인 `ldd` 의존성 수집을 통해 Vulkan 로더 `libvulkan.so.1`을 복사하지만, 드라이버(ICD)는 로더가 `/usr/share/vulkan/icd.d/`에서 열며 자체 시스템 라이브러리를 로드합니다(RADV는 LLVM, libdrm, X/Wayland 클라이언트 라이브러리를 끌어옴). 이미 `lib/mgs/`에 있는 soname은 공유되므로, 번들은 그것을 실행할 Debian 릴리스에서 빌드해야 합니다. Vulkan 디바이스는 `async`와 `events` 기능을 보고하므로 전문가 프리페치와 전문가 캐시가 그대로 동작하며, 그 `device_id`는 PCI 버스 ID이므로 모니터링은 이를 NVML 또는 amdgpu sysfs 카드와 대응시킵니다(이때 카드의 일반적인 이름은 백엔드 설명으로 바뀜). Vulkan은 `ggml_backend_register_host_buffer`를 내보내지 않으므로 매핑된 전문가 페이지는 페이징 가능한 상태로 남습니다. RTX 4060 Ti에서 Vulkan으로 측정한 결과(Qwen3.6-35B-A3B Q4_K), 전문가 업로드 속도는 mmap에서 2.7 GB/s, `--load-mode none`(고정 호스트 버퍼)에서 13.1 GB/s였고, 생성 속도는 각각 초당 27.5 토큰과 36.6 토큰이었습니다. 이제 로더는 CUDA 탓으로 돌리는 대신 매핑된 메모리를 고정할 수 있는 백엔드가 없다고 알립니다. 렌더 노드에는 `render` 그룹이 필요하며, `configure-web.sh`가 서비스 계정에 이 그룹을 부여합니다(`video` 그룹도 함께 부여하며, Alpine에서도 멱등적으로 동작).

**Vulkan에서의 실제 VRAM.** 독립형 GPU에서 `ggml-vulkan`은 device-local 메모리(VRAM)만 집계하고 할당합니다. AMD와 Intel GPU가 매핑할 수 있는 시스템 RAM의 일부(보통 절반)인 GTT는 GPU가 PCIe를 통해 읽는 메모리이므로 절대 GPU 메모리로 집계하지 않습니다. 세 가지 안전장치가 모델을 실제 VRAM에 머물게 합니다. resizable BAR가 없으면 CPU에서 보이는 device-local 힙은 데스크톱과 공유하는 작은 창(RX 580에서 256 MiB)인데, 예전에는 `ggml-vulkan`이 모든 디바이스 버퍼에 대해 이 힙을 먼저 시도했습니다. 그 결과 작은 버퍼들(약 99 MiB, MTP KV 캐시 포함)이 그곳에 놓였고, 커널이 그중 67 MiB를 GTT로 축출했습니다. 이제는 그 힙이 가장 큰 device-local 힙의 절반 미만이면 `disable_host_visible_vidmem`이 자동으로 설정되므로, 디바이스 버퍼는 큰 힙으로 갑니다(RX 580에서는 축출 0, 창에 남은 양 872 KiB). 전문가 캐시 풀은 `VK_EXT_memory_priority` 0.5를 쓰는 두 번째 Vulkan 버퍼 유형(`ggml_backend_dev_get_cache_buffer_type`으로 내보내고 `fPoolBufferType`이 선택)을 사용하므로, VRAM이 부족해지면 드라이버는 우선순위 1.0을 유지하는 MTP 레이어, 가중치, KV 캐시보다 캐시를 먼저 축출합니다. 비동기 복사 검사는 같은 GPU의 모든 디바이스 버퍼 유형을 허용합니다(`ggml_backend_vk_buffer_is_device_of`). 마지막으로 모든 추론 프로세스는 `/proc/self/fdinfo`(`fMonitorDrmResidency`: amdgpu, i915, xe가 게시하는 `drm-resident-*`/`drm-memory-*`와 `amd-evicted-vram`)를 읽어 모니터링 스냅숏에 첨부하고, GPU 할당 가운데 64 MiB(또는 2 %)를 넘는 양이 VRAM 밖에 있으면 로그에 한 번 경고하며(`mCheckVramResidency`), 모니터링 페이지는 실제 위치와 경고를 표시합니다. `--load-mode none`은 자동으로 선택되지 않습니다. AMD에서는 고정 호스트 메모리가 GTT 사용량으로 집계되며, 18.9 GB의 전문가와 16.8 GB의 GTT 조합에서 RADV가 "Not enough memory for command submission"(`ErrorDeviceLost`)으로 실패했기 때문입니다.

**로컬 셸 MCP 서버.** `--local-shell-mcp-server`는 `server_shell_mcp`(`backend/tools/server/server-shell-mcp.cpp`)를 켭니다. 이것은 `mgs` 안에 C++로 작성된 MCP Streamable HTTP 서버이므로, 자체 완결형 번들에는 Python이나 추가 패키지가 필요하지 않습니다. 별도의 리스너가 아니라 API와 같은 서버에 `<api prefix>/mcp/shell`로 등록되므로, 웹 UI는 어느 머신에서든 같은 출처, TLS 인증서, API 키, Apache 프록시를 통해 이 서버에 접근할 수 있습니다. 반면 `127.0.0.1` 포트였다면 서버 자신에서만 접근할 수 있었을 것입니다. 시작할 때 `mStart`는 절대 경로로 된 `PATH` 항목과 일반적인 디렉터리에서 `bash`, `zsh`, `sh`, `pwsh`/`powershell`을 찾고, 인터프리터가 존재하며 실행 가능한 도구만 노출합니다(`bash.run`, `zsh.run`, `sh.run`, `psh.run`). 서버는 상태를 유지하지 않습니다. POST 하나에 JSON-RPC 메시지 하나를 처리하고, JSON으로 응답하며, SSE 스트림(GET은 405 반환)과 세션은 없습니다. `fRunCommand`는 `posix_spawn`으로 새 세션에서 인터프리터를 시작합니다(자체 프로세스 그룹을 가지며 서버의 제어 터미널이 없으므로, sudo는 `mgs`를 시작한 터미널에 절대 프롬프트를 띄우지 않음). 입력은 `/dev/null`이고, 신호 처리는 기본값이며(서버는 SIGPIPE를 무시함), stdout과 stderr는 별도의 파이프로 받아 `poll`로 읽습니다. 제한 시간(기본 60초, 최대 3600초)이 지나거나 연결이 닫히면 그룹 전체를 종료하며, 셸이 끝난 뒤에는 남은 출력을 250 ms 동안 읽으므로 백그라운드 프로세스가 호출을 붙잡아 둘 수 없습니다. 각 스트림은 앞부분과 끝부분을 보존합니다(stdout은 8+8 KiB, stderr는 4+4 KiB). 셸 도구 호출은 인수가 거부된 경우에도 모두 자체 JSONL 파일(`server_shell_mcp_call_log`)을 남깁니다. 이 파일은 `mgs`를 실행하는 사용자의 `~/.mgs/local-shell-mcp-server/`에 생성되며, 호출 시점의 현지 시각을 밀리초까지 딴 이름을 갖습니다(`y2026m10d12h23m59s59ms731.jsonl`, `O_EXCL`로 생성하므로 같은 밀리초의 호출에는 `-2`, `-3`…이 붙음). 파일에는 실행 전에 쓰는 `command` 줄(도구, 셸, cwd, 제한 시간, 명령), 이어서 읽을 때마다 한 줄씩 도착 순서대로 전체 출력을 담는 `stdout` 또는 `stderr` 줄(`fRunCommand`는 모든 조각을 `pOutput`에 넘기고, `mOutput`은 두 번의 읽기에 걸쳐 나뉜 UTF-8 문자를 보류함), 마지막으로 `result` 줄(종료 코드, 신호, 시간 초과 여부, 취소 여부, 소요 초, 각 스트림의 바이트 수) 또는 `error` 줄이 기록됩니다(거부된 호출에는 요청 내용을 담은 `command` 줄과 `error` 줄이 남음). 출력은 절대 메모리에 보관하지 않으므로, 파일에는 도구 결과에서 잘린 부분까지 모두 들어 있고 실행 도중 `mgs`가 멈춰도 파일은 남습니다. 출력에 비밀 정보가 들어 있을 수 있으므로 폴더는 모드 0700, 파일은 0600으로 생성하며, 파일을 쓸 수 없으면 로그에 경고만 남깁니다. `mToolsCallResult`는 JSON 객체(`vExitCode`(종료된 셸을 회수하지 못하면 null), `stdout`, `stderr`, `signal`, `vTimedOut`, `cancelled`, `timeout`, `seconds`, `shell`, `cwd`)를 `structuredContent`와 텍스트 블록 양쪽으로 응답합니다. UI는 도구 결과의 텍스트만 보관하기 때문입니다. 모델은 같은 JSON을 읽고, 채팅은 한 스트림이 어디서 끝나는지 추측하지 않고 이를 상자로 나눕니다. `Origin` 헤더가 있는 요청은 이 서버의 페이지에서 와야 하고(`fOriginAllowed`, `Host`/`X-Forwarded-Host`와 비교하며 개발용 프록시를 위해 루프백 페이지는 허용), `Content-Type: application/json`을 사용해야 하므로 교차 사이트 요청이 차단됩니다. `Origin`이 없는 클라이언트는 API 키에 의존합니다. 다른 에이전트 기능과 마찬가지로, 이 옵션은 기본 CORS 출처를 localhost로 제한하고 시작 시 경고를 추가합니다. 라우터 자식 프로세스에는 이 옵션이 전달되지 않으며, 이 라우트는 모델이 로드되는 동안에도 응답합니다. `/api/props`는 `shell_mcp_server`를 게시합니다. `cMcpStore.mSyncShellServer`(루트 레이아웃에서 호출)는 `mgs-shell` 항목(서버의 `serverInfo.title`과 마찬가지로 이름은 `local shell mcp server`)을 처음에는 활성화된 상태로 추가하고, 그 URL은 현재 출처로, 이름은 그대로 유지하며, 서버가 더 이상 제공하지 않으면 항목을 제거합니다. `MCPService.mCreateTransport`는 같은 출처의 MCP 서버에 UI의 API 키를 보냅니다. 도구 호출은 여전히 UI의 권한 확인을 거칩니다. 채팅에서 `ChatMessageToolCallBlock`은 `bash.run`, `zsh.run`, `sh.run`, `psh.run` 호출(`fIsLocalShellTool`: `aShellMcpToolNames`에 있으면서 다른 MCP 서버가 소유하지 않은 이름)을 `ChatMessageToolCallBlockLocalShell`로 보냅니다. 이 컴포넌트는 도구 이름을 제목으로 유지하고, 다른 도구의 입력 상자와 같은 회색 고정폭 글꼴 상자를 보여 줍니다. stdin 상자에는 인수가 스트리밍되는 동안 부분 인수에서 파싱한 명령만 표시되고, stdout과 stderr 상자는 내용이 있을 때만 표시되며, 제목에 종료 코드가 붙습니다(`stdout (코드 2):`, 출력이 없으면 `stdin (코드 0):`). stdin 상자는 흰색 테두리(밝은 테마에서는 진한 회색), stdout 상자는 초록색 테두리, stderr 상자는 빨간색 테두리를 가지며, 네 도구 모두 같습니다(`SyntaxHighlightedCode`의 `borderColor`에 `--local-shell-stdin-border`, `--local-shell-stdout-border`, `--local-shell-stderr-border` 사용). 이 JSON 객체가 아닌 결과(인수 오류, 권한 거부, 요청 실패)는 stderr 상자에 표시됩니다. `expandLocalShellCommands` 설정(표시)은 `ChatMessageAgenticContent.fGetDefaultExpanded`가 이 호출들을 펼치도록 하므로, 각 상자는 내용이 알려지는 즉시 나타납니다.

**웹 UI 설정.** 인터페이스의 모든 설정은 서버가 `mgs`를 실행하는 사용자의 `~/.mgs/config/ui-settings.json`에 보관하는 JSON 문서 하나에 들어 있으므로(`backend/tools/server/server-ui-settings.cpp`: 폴더 0700, 파일 0600, 뮤텍스를 잡은 채 임시 파일에 쓴 뒤 기존 파일 위로 이름을 바꿔 덮어씀), 모든 브라우저가 이를 공유합니다. `GET /api/ui/settings`는 `{"settings": DOCUMENT}`를 반환하고(처음 저장하기 전에는 `null`), `POST /api/ui/settings`는 문서 전체를 교체합니다. 두 라우트 모두 API 키가 필요하고, 모든 모드에서 등록되며(라우터 부모 프로세스가 직접 처리), 모델이 로드되는 동안에도 응답합니다. `i18n.ts`가 모듈을 가져올 때 언어를 고정하므로, `/api/config.js`는 저장된 언어를 `globalThis.cUiLanguage`로 게시합니다(유효한 언어 태그일 때만). 인터페이스에서 `cUiSettingsStore`(`frontend/src/lib/stores/ui-settings.svelte.ts`)는 루트 레이아웃이 마운트될 때 문서를 로드하고, 각 섹션을 그 섹션을 등록한 스토어에 넘깁니다: `config`와 `userOverrides`(설정 스토어), `favoriteModels`(모델), `disabledToolKeys`(도구), `alwaysAllowedTools`(권한), `reasoningEffortDefault`(대화), `dismissedRecommendedMcpServers`(MCP 서버 추가 대화 상자). `language`는 `fSetLanguage`가 쓰며, 이 함수는 저장이 끝날 때까지 기다린 뒤 페이지를 다시 로드합니다. `mSet`은 마지막 변경 후 300 ms가 지나면 문서 전체를 저장하며, 아직 대기 중인 변경은 페이지가 언로드될 때 `keepalive`로 전송됩니다. 문서가 도착하기 전에는 아무것도 저장하지 않으므로, 일부만 반영된 상태가 파일을 덮어쓰는 일은 없습니다. 로드에 실패하면 5초마다 다시 시도하지만, 401/403이면 API 키가 입력될 때까지 기다립니다. 서버 기본값 동기화와 셸 MCP 항목은 `/api/props`와 문서가 모두 준비될 때까지 기다립니다. 문서가 없으면 첫 방문으로 간주하며, 이때 `--ui-config` 기본값이 한 번 적용됩니다. API 키만 브라우저에 남습니다(`MgsUi.apiKey`). 나머지를 로드하는 요청을 인증하는 것이 바로 API 키이기 때문입니다. 대화는 IndexedDB에, 스트림 재개 키와 빌드 버전 키는 `localStorage`에 남습니다.

**요청 대기열.** 서버는 기본적으로 슬롯 하나를 사용합니다(`-np 1`, `common_params::vNParallel`. 네 개의 슬롯이 통합 KV 캐시를 공유하던 업스트림의 자동 모드는 제거되었으므로 `--kv-unified`도 기본적으로 꺼져 있음). 각 요청은 웹 UI에서 오든 API로 직접 오든 전체 컨텍스트를 받으며, 나머지 요청은 `server_queue`의 지연(deferred) 대기열에서 기다렸다가 도착한 순서대로 실행됩니다(슬롯이 해제되면 첫 번째 지연 작업이 작업 대기열의 맨 앞으로 옮겨짐, `mPopDeferredTask`). `-np N`은 최대 N개의 요청을 병렬로 실행하며 컨텍스트를 그 요청들에 나눕니다. API는 인터페이스나 그 설정 문서에 의존하지 않습니다. 유일한 연결 고리는 웹 UI가 자신의 요청에 `"mgs_client": "webui"`(completion 스키마의 `field_str`로, `task_params::vFromWebui`를 설정함)를 표시한다는 점이며, 이를 통해 서버는 실행 중이거나 대기 중인 모든 요청의 출처를 압니다. 작업이 빈 슬롯을 찾지 못하면 `mDeferTask`는 사용 중인 슬롯(`mCountRunning`)과 그 앞에 있는 지연 작업(`server_queue::mDeferredTasks`)을 API와 웹 UI로 나누어 셉니다. 그리고 요청이 알림을 원하는 경우(`vQueueNotice`: 웹 UI에서 온 스트리밍 요청이면서, 이벤트 스트림의 문법이 고정된 Anthropic Messages와 OpenAI Responses를 제외한 형식)에는 `server_task_result_queue`를 보내며, 이것은 스트림에 `data: {"mgs_queue": {"ahead": N, "ahead_api": A, "ahead_webui": W}}` 형태로 도달합니다. 스트리밍 핸들러는 첫 결과를 받은 뒤에 헤더를 보내므로, 이 이벤트가 응답도 즉시 열어 줍니다. 대기 중이던 작업이 슬롯을 얻으면 모든 개수가 0인 마지막 알림을 받고, `mNotifyQueuePositions`가 아직 대기 중인 작업들의 정보를 갱신합니다. `mgs_client`를 보내지 않는 클라이언트는 이 이벤트를 절대 받지 않으므로, 그 OpenAI 호환 스트림은 바뀌지 않습니다. UI에서 `ChatService.mHandleStreamResponse`는 이 이벤트를 `onQueue`에 넘기고 첫 번째 일반 청크를 받으면 상태를 지웁니다. `cChatStore`는 이 상태를 대화별로 보관하며(`mGetQueueStatus`), `ChatMessageAssistantProcessingInfo`는 다른 어떤 진행 텍스트보다 먼저 `fQueueStatusText`를 표시합니다. 앞선 요청 중 하나라도 API로 직접 들어온 것이면 API용 문구를, 그렇지 않으면 웹 UI용 문구를 보여 주며, 두 문구 모두 앞선 요청의 수를 포함합니다. UI가 스스로 보내는 요청(응답 후 KV 캐시 미리 채우기, 생성된 제목)도 같은 표시를 달고 자기 차례를 기다립니다.

**sudo 비밀번호 프롬프트.** Linux에서 sudo가 있으면 `mStart`는 `server_shell_mcp_askpass`도 만듭니다. 이것은 전용 스레드가 처리하고 연결마다 스레드를 하나씩 쓰는 추상(abstract) Unix 소켓입니다(보호하거나 정리할 파일이 없음). `_meta`에 `mgs/callId`가 담긴 `tools/call`(UI가 `cMcpStore.mExecuteTool`에서 호출마다 하나씩 생성)은 임의의 토큰과 함께 등록되며, 그 명령에는 `SUDO_ASKPASS`(`mgs` 실행 파일 자체: 런처가 게시하는 `MGS_EXECUTABLE` 또는 `/proc/self/exe`), `MGS_LOCAL_SHELL_ASKPASS`(소켓 이름과 토큰), 그리고 서버에 DISPLAY가 없으면 `DISPLAY=:mgs-askpass`가 설정됩니다. sudo는 `DISPLAY`가 있을 때만 터미널 없이 askpass 헬퍼로 대체하기 때문입니다. sudo는 `mgs <prompt>`를 실행합니다. `mgs_server`가 가장 먼저 호출하는 `fServerShellMcpAskpassMain`은 이 경우(해당 변수, 단일 인수, sudo 부모 프로세스)를 인식하고 토큰과 프롬프트를 소켓으로 보냅니다. `mServe`는 `SO_PEERCRED`로 헬퍼가 같은 사용자에 속하는지, `fParentIsSudo`로 그 부모가 sudo인지 확인합니다(이름이 `sudo`이고, 실제 uid가 서버와 같으며, 유효 uid 또는 저장된 uid가 root인지 확인. 권한이 없는 프로세스는 setuid 프로세스의 실행 파일을 읽을 수 없으므로 이것이 가능한 가장 강력한 검사임). 따라서 명령이 비밀번호를 가로채려고 가짜 프롬프트로 헬퍼를 직접 실행할 수는 없습니다. 그다음 프롬프트는 응답을 기다리며(최대 5분), 그동안 `fRunCommand`는 대기한 시간만큼 마감 시각을 뒤로 미룹니다(`mPaused`). 루트 레이아웃에 한 번만 마운트되어 어느 페이지에서든 열리는 `DialogLocalShellAskpass`는 실행 중인 모든 호출(`mExecuteTool`이 도구와 명령으로 채우는 `cMcpStore.mActiveShellCalls`)에 대해 500 ms마다 `GET /api/mcp/shell/askpass?call=`을 폴링하고, 첫 번째 프롬프트를 화면 가운데의 대화 상자에 sudo 프롬프트, 도구, 명령과 함께 표시하며, 비밀번호를 입력한 뒤 sudo가 다시 묻는 경우에는 안내를 덧붙입니다(닫기 버튼이 없는 ShadCN `Dialog`입니다. 프롬프트가 나타나는 즉시 비밀번호 필드가 포커스를 받으며(`fFocusPassword`: `onOpenAutoFocus`는 필드의 ref가 바인딩되기 전에 실행되므로 필드의 attachment로 구현), 수락 또는 Enter(`fHandlePasswordKeydown`)로 응답하고, 취소 또는 Escape로 취소하며, 바깥을 클릭해도 아무 일도 일어나지 않습니다). `POST /api/mcp/shell/askpass`가 비밀번호나 취소를 전달하면, 헬퍼는 이를 sudo가 읽는 파이프에 쓰고 사본은 지워집니다. 비밀번호는 절대 도구 결과, 대화, 로그에 도달하지 않습니다. 오류(취소됨, 응답 없음, 명령 종료)는 `mgs: ...` 형태로 명령의 stderr에 전달됩니다. 다른 MCP 클라이언트는 호출 ID를 보내지 않으므로, 그 명령에는 askpass 변수가 설정되지 않고 sudo는 예전처럼 터미널이 없어 실패합니다.
**보안 강화.** 버전 0.1.9는 외부 정적 보안 검토에서 나온 지적 사항을 해결합니다(지적 사항 21건 중 20건 수정, 설치 프로그램에 관한 1건은 설계상 그대로 둠. 아래 참조).
- **HTTP(`server-http.cpp`):** 공개 경로는 GET/HEAD일 때만 API 키 검사를 건너뛰며, 키가 설정되어 있으면 `GET /models?reload=...`에는 키가 필요합니다. 도구와 MCP 서버가 켜는 `localhost` CORS 모드에서는, 메서드가 GET, HEAD, OPTIONS가 아닌 요청의 `Origin`이 루프백도 아니고 이 서버 자신(`Host` 또는 `X-Forwarded-Host`. Apache는 `ProxyPreserveHost On`으로 `Host`를 유지)도 아니면 라우팅 전에 403을 받습니다. `Origin`이 없는 요청(curl, SDK)은 통과합니다. `POST /tools`에는 `Content-Type: application/json`이 필요하므로(415), 교차 사이트 폼은 CORS가 거부하는 사전 요청(preflight) 없이는 이 라우트에 도달할 수 없습니다. 요청을 수신하는 동안 각 읽기는 최대 30초까지 기다립니다(`cHttpRequestReadTimeoutSec`). 쓰기에는 여전히 `--timeout`이 적용되므로 긴 생성이 잘리지 않습니다. 두 스레드 풀은 수락된 연결을 최대 256개까지 대기열에 넣으며(`cHttpMaxQueuedConnections`), `mPost`/`mDel`은 배열이나 객체가 1024단계보다 깊게 중첩된 본문을 어떤 핸들러가 파싱하기도 전에 거부합니다(`fJsonNestingExceeds`, `cHttpMaxRequestJsonDepth`). nlohmann이 문서를 재귀적으로 복사하고 덤프하기 때문입니다.
- **라우터(`server-models.cpp`):** `fRedactArgs`와 `fRedactPreset`은 `status.args`, `status.preset`, 자식 프로세스 생성 로그에서 비밀 옵션(`sSecretOptionEnvs`: HF 토큰, API 키와 키 파일, SSL 키, MCP 서버 구성)의 값을 `<redacted>`로 바꿉니다. 자식 프로세스는 여전히 실제 값을 받습니다. `mProxyRequest`는 자식 프로세스로 전달할 때 `Origin`을 제거합니다. 라우터가 이미 검사했기 때문입니다.
- **CORS 프록시(`server-cors-proxy.h`):** 탐색 및 임베딩 요청(`Sec-Fetch-Mode`가 navigate이거나 `Sec-Fetch-Dest`가 document, frame, object, worker인 요청)은 403을 받으며(`fProxyIsNavigation`), 프록시를 거친 모든 응답은 무력화됩니다(`fProxyMakeInert`). 샌드박스 CSP, `nosniff`, `Content-Disposition: attachment`, 동일 출처 CORP를 적용하고, 업스트림의 쿠키, CSP, CORS, HSTS, 리디렉션, 링크, 서비스 워커 헤더는 제거합니다. `Content-Type`, `mcp-session-id`, `WWW-Authenticate`는 UI의 MCP 클라이언트를 위해 유지됩니다.
- **원격 미디어(`download.{h,cpp}`, `server-common.cpp`):** `fCommonRemoteGetContent`는 연결 제한 시간과 전체 마감 시각(`set_max_timeout`과 본문 수신기 안의 검사)을 적용하고, 리디렉션을 직접 따라갑니다. `image_url`, `input_audio`, `input_video`(`vPublicHostsOnly`)의 경우 호스트 이름을 한 번만 확인(resolve)하며 모든 주소가 공개 주소여야 합니다. 루프백, RFC 1918, CGNAT, 링크 로컬(169.254.169.254 포함), 문서용, 멀티캐스트, 예약 범위는 허용하지 않고, IPv6에서는 2000::/3만 허용하며, 매핑된 주소, NAT64, 6to4 주소는 그 안의 IPv4 주소로 판단합니다. 연결은 검사한 주소에 고정되며(`set_hostname_addr_map`), 리디렉션마다 다시 검사합니다. `--media-allow-private-urls`는 주소 검사를 해제합니다. 미디어 다운로드에는 연결, 읽기, 쓰기마다 10초, 전체 30초, 리디렉션 5회, 10 MiB, 동시 8개(`media_download_slot`)의 제한이 있습니다. `ffprobe`와 `ffmpeg`는 입력이 요청 버퍼일 때 파이프만 읽습니다.
- **디코딩(`mtmd-helper.cpp`, `server-common.cpp`):** 이미지는 한 변당 최대 32768 px, 100메가픽셀까지이며 디코딩 전에 `stbi_info_from_memory`로 검사합니다. 오디오는 최대 1시간까지이며 샘플을 할당하기 전에 검사합니다. 비디오 프레임은 한 변당 최대 8192 px, 비디오당 RGB 프레임 2 GiB까지입니다. 요청당 디코딩된 이미지와 오디오는 최대 2 GiB입니다(`fProcessMtmdPrompt`).
- **파서:** GBNF 변환기(`json-schema-to-grammar.cpp`)는 빈 패턴과 반복할 대상이 없는 수량자를 거부합니다. 스키마 중첩, 중첩 방문, `$ref` 체인은 256으로(`cMaxSchemaDepth`, `fJsonExceedsDepth`가 반복 방식으로 측정), 패턴 그룹은 128로, 제외되는 속성 이름은 1024바이트로, `type` 목록은 항목 7개로 제한하며, 규칙 이름은 재귀적인 `std::regex` 대신 루프로 정리합니다. Jinja 런타임(`jinja/value.h`)은 `range()`를 원소 100만 개로, 만들어지는 문자열을 256 MiB 및 조각 1 Mi개로, `indent()`/`tojson()` 들여쓰기를 256으로, 문 실행 중첩을 1024로, 파싱 중첩을 256으로 제한합니다. `gguf.cpp`는 오버플로 검사에서 크기가 0인 차원으로 나누지 않고 건너뛰며, 빈 텐서는 여전히 유효합니다.
- **RPC(`ggml-rpc/`):** 가변 크기 `recv_msg`는 명령별 상한을 적용하고(SET_TENSOR: 헤더와 해당 연결에서 살아 있는 가장 큰 버퍼, GRAPH_COMPUTE: 클라이언트가 보낼 수 있는 최댓값인 `INT32_MAX`), `std::length_error`를 잡습니다. 상대방이 보낸 텐서 경계는 오버플로에 안전한 헬퍼(`rpc_checked_nbytes`, `rpc_tensor_region_ok`)로 검사하며, `GGML_ASSERT` 대신 오류로 응답합니다. 어떤 예외든 해당 연결만 닫으며, 버퍼를 해제하면 그 버퍼를 쓰는 저장된 그래프도 제거됩니다. RDMA 수신은 요청한 양보다 많이 복사하지 않고 나머지는 `rx_pending`에 보관하며, 큐 페어(queue pair)는 원격 접근 권한을 부여하지 않습니다.
- **웹 UI:** Markdown 렌더링에 실패하면 이스케이프된 텍스트를 표시합니다. 스킴이 http, https, mailto, tel이 아닌 링크(상대 링크는 제외)는 `href`를 잃고, 이미지 첨부 파일은 `data:image/...;base64`여야 하며, 깨진 이미지 안내는 DOM 노드로 만듭니다(`fCreateImageErrorFallback`). 대화 가져오기는 압축을 풀기 전에 파일(512 MB), ZIP 항목 수(10 000), 각 JSONL(256 MB), 압축 해제 후 총량(1 GB)을 검사하고, 모든 레코드를 검증하며, 각 메시지의 `convId`가 해당 대화와 일치하도록 요구하고, `bulkAdd`로 기록하되 메시지 ID가 이미 존재하는 대화는 건너뜁니다. 도구 이름마다 하나의 신원(`mResolveEnabledEntry`)이 모델, 승인, 실행에 쓰이며, 실행은 정확히 그 서버로 갑니다(`mExecuteTool(pToolCall, pServerId, pSignal)`). 활성화된 두 MCP 서버가 같은 이름을 제공하면 경고와 함께 그 도구를 제공하지 않으며, 서버의 모든 도구를 항상 허용하는 설정("다음 서버의 모든 도구 항상 허용")은 서버 ID 단위로 적용됩니다.
- **웹 UI 의존성(`frontend/package.json`, 0.1.10):** `npm audit`가 지적한 패키지 중 브라우저 번들에 포함되는 것은 수정된 릴리스로 옮겼습니다: DOMPurify 3.4.17, mermaid 11.17.2, `@modelcontextprotocol/sdk` 1.32.1. SvelteKit 2.70.3과 `@vite-pwa/assets-generator` 1.0.4(sharp 0.35.5, Node.js 20.9 이상)도 업데이트했으며, `npm audit fix`는 빌드 시점의 전이적 패키지를 허용 범위 안에서 갱신합니다. 직접 의존성의 버전은 정확히 고정된 상태로 유지됩니다. `npm audit`가 여전히 나열하는 항목은 빌드 도구(`@tailwindcss/typography`를 통한 `postcss-selector-parser`, Windows의 esbuild 개발 서버)나 KaTeX에만 해당합니다. KaTeX의 수정 버전(0.18.2)은 `rehype-katex`와 mermaid가 허용하는 `^0.16` 범위 밖에 있으며, 이 KaTeX 보안 권고는 이미 존재하는 프로토타입 오염이 있어야 악용할 수 있습니다.
- **해결하지 않은 항목:** `curl | bash` 설치 프로그램을 변경할 수 없는 리비전이나 서명에 고정하는 것(프로젝트 규칙상 `main`에서 설치해야 함), 헤더 블록 전체에 대한 절대 마감 시각(cpp-httplib는 읽기마다 제한 시간을 적용함), RPC 읽기 마감 시각과 동시 클라이언트, RPC 서버의 반복 방식 `create_node`, Jinja 템플릿에 대한 전역 CPU 및 메모리 예산, 요청의 `grammar`에 대한 `mgs-grammar.cpp`의 재귀 처리.

<a id="modules"></a>
## 모듈 맵

| 모듈 | 경로 | 역할 | 의존 대상 | 사용처 |
| --- | --- | --- | --- | --- |
| MoE 모델 구성 | `backend/src/CMakeLists.txt; backend/src/models/models.h; backend/src/mgs-model.cpp` | MoE 그래프와 그에 필요한 헤드/공유 블록을 컴파일하고, dense 메인 모델을 거부합니다. | GGUF 메타데이터, 모델 로더 | 모델 초기화 |
| 빌드 | `build/CMakeLists.txt; backend/CMakeLists.txt` | 엔진, 서버, 로컬 UI를 조합하고, 업스트림 타깃을 유지하면서 `mgs`를 출력합니다. | CMake, C/C++ 툴체인 | deploy/install-common.sh; build/build-local.sh |
| 버전 | `backend/cmake/build-info.cmake; backend/common/build-info.{h,cpp.in}; deploy/scripts/ui-assets.cmake; frontend/scripts/vite-plugin-build-info.ts` | `MGS_VERSION`을 한 곳에서 정의하고 바이너리(`mgs_version()`), 라이브러리 파일 버전, CMake 패키지, 인터페이스의 `build.json`에 전달합니다. | CMake | `--version`, `/api/props`, OpenAI 응답, 셸 MCP, 다운로드, UI |
| 네이티브 TLS | `backend/tools/server/server-tls.cpp; backend/tools/server/server-http.cpp` | 자동 인증서 또는 PEM 인증서, 기본 HTTPS, 빈 포트를 사용하는 HTTP 리디렉션을 제공합니다. | OpenSSL, cpp-httplib | 공개 실행 파일과 라우터 |
| HTTP | `backend/tools/server/server-http.cpp` | 접두사가 붙은 API를 등록하고, 요청을 인증하며(공개 경로는 GET/HEAD에만 허용), `localhost` CORS 모드에서 크로스 사이트 쓰기를 거부하고, 요청 읽기·연결 대기열·본문의 JSON 중첩 깊이를 제한하며, 루트 UI 에셋을 제공합니다. | cpp-httplib, mgs-ui | server.cpp |
| 원격 미디어 | `backend/common/download.{h,cpp}; backend/tools/server/server-common.cpp` | `image_url`, `input_audio`, `input_video` URL을 전체 마감 시간 안에 가져옵니다. 한 번 해석한 뒤 고정한 공개 주소에서만 가져오며(`--media-allow-private-urls`가 없을 때), 리디렉션은 하나씩 검사하고, 크기는 10 MiB, 동시 다운로드는 8개로 제한합니다. | cpp-httplib, getaddrinfo | 추론 라우트 |
| 미디어 디코딩 제한 | `backend/tools/mtmd/mtmd-helper.cpp; backend/tools/server/server-common.cpp` | 이미지는 디코딩하기 전에 크기를, 오디오는 메모리를 할당하기 전에 길이를 확인하고, 비디오 프레임 크기와 프레임 예산, 요청마다 디코딩된 총량을 확인합니다. | stb_image, miniaudio, ffmpeg | mtmd, 서버 |
| CORS 프록시 | `backend/tools/server/server-cors-proxy.h` | UI를 위해 MCP 엔드포인트를 가져오고, 탐색은 거부하며, 비활성(inert) 응답을 반환합니다. | http.h, server-models | UI MCP 클라이언트 |
| 추론 | `backend/tools/server/server-context.cpp` | 슬롯, 추측 디코딩 지표, 디코딩을 관리하고, 토큰 수를 세기 전에 절전 상태에서 깨어납니다. | mgs-common, mgs-engine, mtmd | HTTP, 라우터 |
| 요청 대기열 | `backend/tools/server/server-{context,queue,task,schema}.{cpp,h}; frontend/src/lib/services/chat.service.ts; frontend/src/lib/stores/chat.svelte.ts; frontend/src/lib/utils/progress.ts` | 기본적으로(`-np 1`) 요청을 도착 순서대로 한 번에 하나씩 실행합니다. 스트리밍되는 웹 UI 요청(`mgs_client`)에는 앞에서 실행 중이거나 대기 중인 API 요청과 웹 UI 요청의 수를 알려 주고, 채팅에 표시합니다. | server_queue, 슬롯, completion 스키마, 15개 카탈로그 | 채팅, API 클라이언트 |
| Kimi-K3 | `backend/src/models/kimi-k3.cpp` | KDA + MLA 하이브리드, 레이어 간 잔차 혼합, 잠재(latent) MoE를 구현합니다. | mgs-graph (KDA, MoE), ggml `dsv4_hc_pre` | 모델 그래프 |
| GLM-5.3-Flash 및 glm5v | `backend/src/models/glm5-next.cpp; backend/tools/mtmd/models/glm4v.cpp` | k-pool 인덱서와 MTP를 갖춘 KDA + MLA, 그리고 glm5v 비전 프로젝터를 구현합니다. | mgs-memory-hybrid-idx, ggml `swiglu_clamp` | 모델 그래프, mtmd |
| Qwen3.8-Flash-Next | `backend/src/models/qwen4exp.cpp` | 하이퍼 커넥션, n-gram 레이어별 임베딩, gated delta-net, QSA 어텐션, MTP를 구현합니다. | mgs-memory-hybrid-idx, ggml `dsv4_hc_*`, `lightning_indexer` | 모델 그래프 |
| 하이브리드 인덱스 메모리 | `backend/src/mgs-memory-hybrid-idx.{h,cpp}` | KV 캐시, 순환(recurrent) 캐시, k-pool 인덱서 캐시를 제공합니다. | mgs-kv-cache, mgs-memory-recurrent | qwen4exp, glm5-next |
| DeepSeek-V4.1 | `backend/src/models/deepseek41.cpp; backend/src/mgs-kv-cache-dsv41.{h,cpp}` | 두 가지 GGUF 레이아웃, 외부 Engram 테이블, 공유 압축 KV, 후보 풀을 다룹니다. | mgs-model-loader (`cTensorExternal`), ggml `dsv4_hc_*`, `lightning_indexer`, `swiglu_clamp` | 모델 그래프 |
| 라우터 | `backend/tools/server/server-models.cpp` | 모델 자식 프로세스를 관리하고, API 키 파일 인증을 부모 프로세스에 유지하며, 정규 API 요청을 `Origin` 없이 전달합니다. 모델 목록과 로그에서는 비밀 옵션을 가리고, 런처가 `MGS_EXECUTABLE`을 설정하면 그것을 다시 실행합니다. | 서브프로세스, HTTP, 스트림 | server.cpp |
| 서브프로세스 | `backend/common/subproc.{h,cpp}` | `mCreate`, `mAlive`, 스트림 접근자, `mCloseStdin`, `mTerminate`, `mJoin`으로 자식 핸들을 관리하며, 두 빌드 분기 모두 헤더와 일치합니다. 호출부와 비활성화된 빌드의 스텁에서는 벤더의 `subprocess_option_*` 이름을 그대로 유지합니다. | sheredom/subprocess.h, std::vector | 라우터, MCP stdio, 서버 도구 |
| 비디오 입력 | `backend/tools/mtmd/mtmd-helper.cpp` | ffprobe/ffmpeg 서브프로세스로 파일이나 버퍼를 디코딩합니다. 선언된 매개변수와 공개 필드 `ffmpeg_bin_dir`/`timestamp_interval_ms`를 전달하고, 명령 벡터는 `push_back`으로 구성합니다. | mtmd, 서브프로세스, ffmpeg/ffprobe | 멀티모달 입력 |
| 샘플링 버퍼 | `backend/common/sampling.cpp; backend/src/mgs-sampler.cpp` | 링 버퍼 고유의 `mPushBack` 메서드를 유지하며, `std::vector`로 변환할 때는 표준 `push_back` 메서드를 사용합니다. | std::vector | 공통 샘플러, 엔진 샘플러 |
| 프런트엔드 빌드 | `frontend/package.json; frontend/vite.config.ts; frontend/pwa-assets*.config.ts; frontend/scripts/vite-plugin-*.ts` | PWA 에셋을 생성하고 Vite로 컴파일하기 전에 `svelte-kit sync`를 실행하고, 동기화 오류를 전파합니다. 현재의 `dSveltekitPwaOptions`, `cBuildConfig`, PWA 상수, `cTab`/`cNewline`을 직접 가져옵니다. | frontend/src/lib/constants, SvelteKit, Vite, PWA 에셋 생성기 | npm run build, mgs-ui |
| 채팅 라우팅 | `frontend/src/routes/(chat)/+page.ts; frontend/src/routes/(chat)/chat/[id]/+page.ts` | 해시 라우터가 지원하는 프레임워크의 정확한 이름 `load`를 내보내고, 채팅을 렌더링하기 전에 `/api/props`로 API 키를 검증합니다. | SvelteKit PageLoad, api-key-validation | 새 대화와 기존 대화 |
| 비전 어텐션 | `backend/tools/mtmd/models/granite4-vision.cpp` | 정규화된 입력 텐서를 Q/K/V 프로젝션 변수와 분리해 둡니다. Granite의 셀프 어텐션은 세 프로젝션 모두와 그 잔차에 같은 정규화된 쿼리 입력을 사용합니다. | GGML, clip 그래프 | 비전 프로젝터 |
| 스트림 | `backend/tools/server/server-stream.cpp` | 생성 세션을 보존하고 커서 위치부터 재개합니다. | server-task, server-queue | 채팅, 라우터 |
| API 문서 | `backend/tools/server/server-api-doc.cpp; frontend/api.html` | 라우트 레지스트리와 completion 스키마로 OpenAPI를 구성하고, Swagger UI를 내장합니다. | server-schema, vendor/swagger-ui | /api/doc/ |
| 셸 MCP 서버 | `backend/tools/server/server-shell-mcp.{h,cpp}` | 설치된 셸을 감지하고 `/api/mcp/shell`에서 `bash.run`, `zsh.run`, `sh.run`, `psh.run`으로 MCP Streamable HTTP를 제공합니다. 명령은 자체 세션에서 시간 제한과 함께 실행하고, 호출마다 `~/.mgs/local-shell-mcp-server/` 아래의 개별 JSONL 파일에 기록합니다. | server-http, server-common, POSIX spawn | server.cpp; UI `cMcpStore` |
| sudo 비밀번호 요청 | `backend/tools/server/server-shell-mcp.{h,cpp}; backend/tools/server/server.cpp` | `mgs`를 SUDO_ASKPASS로 실행하고, 부모 프로세스가 sudo인 헬퍼만 받아들이며, UI가 응답할 때까지 각 요청을 붙잡아 두고 그동안 해당 명령의 시간 제한을 멈춥니다. | Linux 추상 Unix 소켓, /proc | 셸 MCP 서버, UI 비밀번호 대화 상자 |
| UI 셸 MCP 항목 | `frontend/src/lib/stores/mcp.svelte.ts; frontend/src/routes/+layout.svelte; frontend/src/lib/services/mcp.service.ts` | `/api/props`의 `shell_mcp_server`를 `local shell mcp server`라는 이름의 `mgs-shell` MCP 서버로 반영하고, 같은 출처(same-origin) MCP 요청을 인증합니다. | serverStore, settingsStore | MCP 설정과 채팅 |
| UI 로컬 셸 호출 | `frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockLocalShell.svelte; frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts; frontend/src/lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte` | 셸 MCP 도구 호출을 종료 코드와 함께 stdin, stdout, stderr 상자로 표시하고, `expandLocalShellCommands`가 켜져 있으면 자동으로 펼칩니다. | mcpStore, settingsStore, SyntaxHighlightedCode | 채팅 |
| UI 비밀번호 대화 상자 | `frontend/src/lib/services/local-shell-askpass.service.ts; frontend/src/lib/components/app/dialogs/DialogLocalShellAskpass.svelte; frontend/src/lib/stores/mcp.svelte.ts; frontend/src/routes/+layout.svelte` | 셸 호출마다 _meta에 호출 ID를 담아 보내고, 실행 중인 호출의 대기 중인 sudo 요청을 폴링해 가운데 정렬된 대화 상자에 표시하며, 그 대화 상자는 비밀번호나 취소를 전송합니다. | apiFetch, mcpStore, ShadCN Dialog | 루트 레이아웃 |
| 매개변수와 MTP | `backend/common/arg.cpp; backend/common/common.cpp` | `--load-mode`를 포함한 옵션을 파싱하고, 자동 MTP 전에 GGUF를 검사합니다(단일 헤드 glm4moe 포함). 별도의 MTP GGUF를 사용하면 타깃에 이식된 MTP 레이어는 로드하지 않습니다. | GGUF, mgs-engine | 서버, 모델 초기화 |
| 모델 로딩 | `backend/src/mgs-model-loader.cpp; backend/src/mgs-model.cpp; backend/src/mgs-mmap.cpp` | 가중치를 로드하고, 매핑을 보존/등록하며, Direct I/O 스테이징을 64 MiB로 제한합니다. MTP 가중치와 출력 레이어는 디바이스 메모리에 머물며 호스트 메모리로 보내는 오버라이드를 무시합니다. | GGML 백엔드 등록 | mgs-model |
| 전문가 프리페치 | `backend/ggml/src/ggml-backend.cpp` | 크기가 제한된 링 버퍼로 전문가 전송을 겹쳐 실행하고, 입력이 없는 분할을 동기화하며, 빈 ID는 건너뜁니다. 프리페치 업로드는 업로드 콜백(전문가 캐시)이 처리할 수 있으며, 입력이 없는 분할은 입력이 복사된 뒤이거나 공유 버퍼 유형일 때만 대기합니다. | GGML 디바이스와 버퍼 | 그래프 스케줄러 |
| 프런트엔드 | `frontend/src/routes; frontend/src/lib` | 채팅/설정을 렌더링하고 대화를 내보냅니다. 처리 중 표시기의 여백은 그 위치에서 산출하고, 현재 파일 업로드 오류는 getter로 노출합니다. ModelsStore는 인증 헤더를 직접 가져와 유틸리티 배럴의 청크 순환을 피합니다. 헤더 정제는 기본 가림 집합과 추가 가림 집합을 서로 구분해 둡니다. | Svelte, 브라우저 저장소, API | 브라우저 |
| 현지화 | `frontend/src/lib/i18n.ts; frontend/src/lib/locales; frontend/src/lib/components/app/settings/LanguageSelector.svelte` | 15개 로캘을 알파벳순으로 등록하고, JSON 메시지를 해석해 값을 보간하며, 선택한 언어를 서버의 설정 문서에 저장합니다. | 15개 JSON 카탈로그 | UI 컴포넌트, 구성, 루트 레이아웃 |
| 답변 언어 | `frontend/src/lib/utils/response-language.ts; frontend/src/lib/services/chat.service.ts; frontend/src/lib/constants/settings-{keys,registry}.ts; frontend/src/lib/locales` | 사용자가 켜야 하는 `respondInInterfaceLanguage` 설정(기본값 false)을 저장하고, 나가는 모든 completion과 KV 미리 채우기에 현재 로캘의 지역별 지시문을 추가합니다. | fGetLanguage, fTranslate, SettingsStore, 15개 카탈로그 | 일반 채팅, 재생성, 이어서 생성, 도구 턴, 생성된 제목 |
| 텍스트 방향 | `frontend/src/routes/+layout.svelte; frontend/src/app.css; frontend/src/lib/components/app/navigation/SidebarNavigation; frontend/src/lib/components/app/settings/SettingsChatDesktopSidebar.svelte; frontend/src/lib/components/app/chat/ChatForm/ChatFormTextarea.svelte; frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserBubble.svelte` | HTML 언어/방향을 설정하고, 내비게이션을 논리적 속성으로 배치하며, 채팅 텍스트의 방향을 따르고, 코드는 LTR로 격리합니다. | `fGetLanguage`, CSS 논리 속성, HTML `dir` | 모든 페이지, 채팅, 코드 미리보기 |
| 배포 | `deploy/install-common.sh; deploy/configure-web.sh` | 바이너리를 설치하고 Apache와 시스템 서비스를 구성합니다. | apt/apk, OpenSSL, systemd/OpenRC | 프로덕션 호스트 |
| Debian 부트스트랩 | `deploy/install-update-reinstall-debian.sh` | curl-to-Bash 방식을 위해 소스를 내려받거나 동기화하고, 잠금을 건 뒤 컴파일을 위임합니다. | curl, tar, rsync, flock, install-common.sh | 로컬/원격 Debian 설치 |
| 웹 시작 | `deploy/start-web.sh; deploy/apache2.conf.in` | HTTP/HTTPS 포트를 선택하고, Apache를 root 없이 시작하며, 바인드 충돌 시 다시 시도하고, 실제로 동작하는 URL을 게시합니다. | Bash, Apache, iproute2, flock | moe-gguf-server-web |
| 프로젝트 라이선스 | `LICENSE` | nipegun의 기여분에 적용되는 PolyForm NonCommercial License 1.0.0으로, `Required Notice:` 줄로 시작합니다. 전문(前文)은 llama.cpp의 원래 고지(The ggml authors)를 유지하고, 서드파티 부분은 각각 `build/licenses/`에 있는 해당 라이선스를 따르도록 안내합니다. 생성된 번들의 licenses/ 폴더에 `LICENSE`로 복사됩니다. | build/licenses/ | GitHub, build/build-local.sh |
| 라이선스 고지 | `build/licenses/` | 서드파티 라이선스 고지(ggml 저자들의 MIT 라이선스인 `LICENSE-llama.cpp`, `LICENSE-FreeToken`, `NOTICE-swagger-ui`가 딸린 `LICENSE-swagger-ui`, `LICENSE-cpp-httplib`, `LICENSE-jsonhpp`)를 보관하고, 생성된 번들의 licenses/ 폴더로 복사합니다. | 서드파티 고지 | LICENSE, build/build-local.sh |
| 로컬 빌드 | `build/build-local.sh; build/mgs-launcher.c` | 로컬 소스를 선택하거나 main을 HTTPS로 내려받고, 비공개 /tmp/ 작업 공간에서 컴파일한 뒤 재배치 가능한 설치본을 조립해 기본적으로 `~/.local` 아래(`bin/mgs`, `lib/mgs/`)에 설치하며, 종료할 때 정리합니다. 정적 런처는 NVIDIA 드라이버를 제외한 번들 라이브러리를 로드합니다. | CMake, ldd, patchelf, 정적 libc, CUDA Toolkit, Node.js; 다운로드용 curl/gzip | 워크스테이션에서의 수동 사용, 로컬 파일 또는 curl-to-Bash |
| Vulkan 백엔드 | `backend/ggml/src/ggml-vulkan; build/build-local.sh; deploy/install-common.sh; deploy/configure-web.sh` | `--vulkan`으로 Vulkan 백엔드를 빌드하고(로컬 빌드와 Debian 설치 프로그램), 빌드 패키지와 Mesa 드라이버를 설치하며, 로더는 번들에 포함하되 드라이버는 시스템의 것을 사용하고, 서비스 계정에 `render`와 `video` 그룹을 부여합니다. | glslc, libvulkan-dev, spirv-headers, Vulkan 드라이버 | CUDA를 사용하지 않는 AMD, Intel, NVIDIA GPU |
| MoE 커널 | `backend/ggml/src/ggml-{cuda,cpu,vulkan,sycl}` | 전문가를 라우팅하고, 타일/퓨전을 조정하며, 어텐션 꼬리 부분을 처리합니다. CPU의 `mul_mat_id`/`add_id`는 음수 전문가 식별자를 건너뜁니다. | GGML, 디바이스 컴파일러 | 추론 그래프 |
| MTP 실행 | `backend/src/mgs-{context,batch,model}.cpp; backend/src/models/glm4-moe.cpp; backend/common/{sampling,speculative,fit}.cpp` | 별도의 그래프 아레나를 캐시하고, 행 순서를 보존하며, 초안 헤드와 수락을 관리합니다. | mgs-engine, GGML | 서버 슬롯 |
| 초안 지표 | `backend/tools/server/server-{context,task}.{cpp,h}` | 초안/수락 카운터를 집계하고 Prometheus 지표를 게시합니다. | 슬롯, 작업 대기열 | /api/metrics |
| UI 영속화 | `frontend/src/lib/stores/{conversations,tools,permissions,models,settings}.svelte.ts` | 영속 저장된 대화 트리(IndexedDB)를 내보내고, 도구 사용 가능 여부, 권한, 즐겨찾기, 사용자 설정을 서버의 설정 문서에 보관합니다. | IndexedDB, UI 설정 | 채팅, 설정 |
| 대화 가져오기 | `frontend/src/lib/stores/conversations.svelte.ts; frontend/src/lib/services/database.service.ts; frontend/src/lib/constants/conversation-import.ts` | 압축을 풀기 전에 파일과 ZIP 항목의 크기를 제한하고, 레코드를 검증하며, 기존 메시지를 덮어쓰지 않고 새 대화를 추가합니다. | fflate, IndexedDB (Dexie) | 가져오기 대화 상자 |
| UI 링크와 이미지 안전성 | `frontend/src/lib/utils/safe-url.ts; frontend/src/lib/utils/image-error-fallback.ts; frontend/src/lib/components/app/content/MarkdownContent/plugins/rehype/{enhance-links,resolve-attachment-images}.ts; frontend/src/lib/components/app/content/MarkdownContent/MarkdownContent.svelte` | 안전한 링크 스킴과 `data:image` 첨부만 남기고, 깨진 이미지 안내를 DOM 노드로 구성하며, Markdown 처리에 실패하면 이스케이프된 텍스트를 표시합니다. | rehype, DOM | 메시지 렌더링 |
| UI 도구 식별 | `frontend/src/lib/stores/{tools,mcp,agentic}.svelte.ts` | 도구 이름마다 활성화된 항목을 하나로 결정해 모델, 승인, 실행이 모두 정확히 그 MCP 서버를 가리키게 하며, 두 서버가 함께 제공하는 이름은 보류합니다. | MCP 클라이언트, 권한 | 에이전트 루프 |
| UI 설정 | `backend/tools/server/server-ui-settings.{h,cpp}; frontend/src/lib/stores/ui-settings.svelte.ts; frontend/src/lib/services/ui-settings.service.ts` | 인터페이스의 설정 문서를 `~/.mgs/config/ui-settings.json`에 보관하고, `GET`/`POST /api/ui/settings`를 제공하며, 저장된 언어를 `/api/config.js`에 게시하고, 각 섹션을 그것을 소유한 스토어에 넘깁니다. | std::filesystem, POSIX 파일 API, API fetch | 설정, 모델, 도구, 권한, 대화, i18n, 루트 레이아웃 |
| MoE 전문가 캐시 | `backend/src/mgs-expert-cache.{h,cpp}` | 최근에 사용한 호스트 전문가를 GPU마다 LRU 풀에 보관하고, 작은 마이크로 배치를 디바이스와 CPU로 나누며, 프리페치 업로드를 풀에서 처리하고, 상태를 보고합니다. | GGML 백엔드와 스케줄러, mgs-model | mgs-context, mgs-graph, common, 서버 |
| 하이브리드 MoE 그래프 | `backend/src/mgs-graph.cpp` | 전문가 체인을 한 번 구성하거나, 캐시된 레이어에서는 CPU 분할 연산을 사이에 두고 두 번(디바이스 풀과 호스트 가중치) 구성한 뒤 두 분기를 더합니다. | mgs-expert-cache, GGML | 모델 그래프 |
| 자동 fit | `backend/common/fit.cpp` | 레이어를 디바이스 메모리에 맞추고, RAM이 허용하면 모든 전문가를 캐시용으로 시스템 메모리에 두며(GPU가 여러 개면 메모리에 따라 레이어를 GPU들에 분산), dense 가중치를 올리고 남은 디바이스 메모리 안에서 마이크로 배치(최대 4096, 8분의 1 이내)와 축소된 컨텍스트(최대 4분의 1)를 늘리고, nextn 레이어를 계산에 넣습니다. | mgs-ext, /proc/meminfo | 공통 초기화 |
| 컨텍스트 체크포인트 | `backend/tools/server/server-context.cpp; backend/common/chat.h` | 메시지 경계와 배치 시작 지점에서 체크포인트를 만들고, 이웃과 가장 가까운 체크포인트를 제거합니다. | 메시지 구간, mgs 상태 API | 서버 슬롯 |
| 모니터링(서버) | `backend/tools/server/server-monitor.{h,cpp}; backend/tools/server/server-context.cpp; backend/tools/server/server-models.cpp` | CPU, RAM, GPU 사용률을 읽고(procfs, NVML, sysfs), 모든 추론 루프의 메모리 스냅샷을 게시하며, 라우터 자식 프로세스의 정보를 집계하고, `/api/monitor*`를 제공합니다. | mgs-ext 메모리 API, libnvidia-ml (선택 사항, dlopen) | HTTP, UI 모니터링과 상태 표시줄 |
| 메모리 집계 | `backend/src/mgs-model.cpp; backend/src/mgs-context.cpp; backend/src/mgs-expert-cache.cpp; backend/ggml/src/ggml-backend.cpp` | 모델 가중치를 버퍼 유형별로 MTP와 라우팅 전문가로 나누고, 전문가 캐시 버퍼와 프리페치 슬롯을 보고합니다. | GGML 버퍼와 스케줄러 | 모니터링(서버) |
| 모니터링(UI) | `frontend/src/lib/components/app/monitor; frontend/src/lib/stores/monitor.svelte.ts; frontend/src/lib/services/monitor.service.ts; frontend/src/lib/constants/monitor.ts; frontend/src/routes/monitor` | 모니터링 페이지(사용률 카드와 범례가 있는 누적 메모리 막대)와 CPU·GPU 미터가 있는 고정 상태 표시줄을 제공합니다. 사용률 미터는 빨간색으로 채워지는 초록색 막대입니다. | `/api/monitor`, `/api/monitor/usage`, 설정 | 루트 레이아웃, 사이드바 |


<a id="symbols"></a>
## 주요 심볼 색인

| 심볼 | file:line | 용도 |
| --- | --- | --- |
| `mgs_model_create` | [backend/src/mgs-model.cpp:157](../backend/src/mgs-model.cpp#L157) | 모델을 구성하기 전에 전문가 수 또는 명시적인 MTP 전용 헤드인지를 검증합니다. |
| `main` | [backend/tools/server/main.cpp:3](../backend/tools/server/main.cpp#L3) | `mgs`의 CLI 진입점입니다. |
| `MGS_VERSION` | [backend/cmake/build-info.cmake:3](../backend/cmake/build-info.cmake#L3) | 프로젝트 버전입니다. 프로젝트를 변경할 때마다 패치 번호를 올립니다. |
| `mgs_version` | [backend/common/build-info.cpp.in:9](../backend/common/build-info.cpp.in#L9) | `--version`, `/api/props`, `system_fingerprint`, 셸 MCP, `User-Agent`에 쓰이는 프로젝트 버전을 반환합니다. |
| `mgs_server` | [backend/tools/server/server.cpp:89](../backend/tools/server/server.cpp#L89) | 프로세스를 초기화하고 라우트를 등록합니다. |
| `server_http_context::mInit` | [backend/tools/server/server-http.cpp:234](../backend/tools/server/server-http.cpp#L234) | HTTPS와 HTTP 리디렉션을 구성하고, API 접두사를 검증하며, 미들웨어를 설치합니다. |
| `server_http_context::mStart` | [backend/tools/server/server-http.cpp:654](../backend/tools/server/server-http.cpp#L654) | 서로 다른 포트에 바인드하고 두 리스너를 시작합니다. |
| `fBindServerPort` | [backend/tools/server/server-http.cpp:88](../backend/tools/server/server-http.cpp#L88) | 포트 충돌일 때만 다시 시도하며, 선택한 수신 소켓은 바인드된 상태로 유지합니다. |
| `fCreateHttpsServer` | [backend/tools/server/server-tls.cpp:302](../backend/tools/server/server-tls.cpp#L302) | PEM 인증서를 로드하거나, 처음 실행할 때 생성해 저장한 `~/.mgs/certificates/`의 자체 서명 아이덴티티를 로드합니다. |
| `fPersistentServerIdentity` | [backend/tools/server/server-tls.cpp:277](../backend/tools/server/server-tls.cpp#L277) | `~/.mgs/certificates/fullchain.pem`과 `privkey.pem`이 유효하면 재사용하고, 그렇지 않으면 자체 서명 아이덴티티를 생성해 저장합니다(0700/0600). |
| `server_http_context::mGet` | [backend/tools/server/server-http.cpp:856](../backend/tools/server/server-http.cpp#L856) | GET 라우트를 등록하고 그 문서화 메타데이터를 기록합니다. |
| `vMiddlewareValidateApiKey` | [backend/tools/server/server-http.cpp:402](../backend/tools/server/server-http.cpp#L402) | 공개 경로에 대한 GET/HEAD에서만 API 키 검사를 건너뜁니다. 비어 있지 않은 `reload`가 붙은 `GET /models`에는 키가 필요합니다. |
| `fOriginIsSameHost` | [backend/tools/server/server-http.cpp:187](../backend/tools/server/server-http.cpp#L187) | `Host` 또는 `X-Forwarded-Host`를 통해 이 서버를 가리키는 `Origin`을 받아들입니다(기본 포트는 무시). 라우팅 전 핸들러는 `localhost` CORS 모드에서 그 밖의 크로스 사이트 쓰기를 거부합니다. |
| `fJsonNestingExceeds` | [backend/tools/server/server-http.cpp:34](../backend/tools/server/server-http.cpp#L34) | 본문을 파싱하지 않고 JSON 텍스트로 훑어, 배열/객체 중첩이 한도보다 깊은지 알려 줍니다. |
| `fRejectDeepRequestBody` | [backend/tools/server/server-http.cpp:800](../backend/tools/server/server-http.cpp#L800) | `cHttpMaxRequestJsonDepth`(1024)보다 깊게 중첩된 POST 및 DELETE 본문에는 핸들러가 실행되기 전에 400으로 응답합니다. |
| `fRedactArgs` | [backend/tools/server/server-models.cpp:173](../backend/tools/server/server-models.cpp#L173) | 목록과 로그에서 비밀 옵션(`sSecretOptionEnvs`, 모든 표기, `--opt=value` 형식 포함)의 값을 `<redacted>`로 바꿉니다. |
| `fRedactPreset` | [backend/tools/server/server-models.cpp:161](../backend/tools/server/server-models.cpp#L161) | `status.preset`용으로 비밀 옵션을 `<redacted>`로 바꾼 프리셋 사본을 만듭니다. |
| `fProxyIsNavigation` | [backend/tools/server/server-cors-proxy.h:34](../backend/tools/server/server-cors-proxy.h#L34) | 브라우저가 문서, 프레임, 객체, 워커로 렌더링할 프록시 요청(`Sec-Fetch-Mode`, `Sec-Fetch-Dest`)을 거부합니다. |
| `fProxyMakeInert` | [backend/tools/server/server-cors-proxy.h:44](../backend/tools/server/server-cors-proxy.h#L44) | 프록시된 응답에서 업스트림의 능동적 헤더를 제거하고, 샌드박스 CSP, `nosniff`, `attachment`, same-origin CORP를 강제합니다. |
| `fCommonRemoteGetContent` | [backend/common/download.cpp:603](../backend/common/download.cpp#L603) | 연결 마감 시간과 전체 마감 시간, 수동 리디렉션을 적용해 내려받으며, `vPublicHostsOnly`가 설정되면 해석 뒤 고정한 공개 목적지에서만 내려받습니다. |
| `fRemoteResolveHost` | [backend/common/download.cpp:503](../backend/common/download.cpp#L503) | 주소 검사와 고정을 위해 남은 마감 시간 안에서 호스트를 한 번 해석합니다. |
| `fRemoteIpv4IsPublic` | [backend/common/download.cpp:438](../backend/common/download.cpp#L438) | 루프백, 사설, CGNAT, 링크 로컬, 문서용, 테스트용, 멀티캐스트, 예약, 브로드캐스트 IPv4 주소를 거부합니다. |
| `fRemoteIpv6IsPublic` | [backend/common/download.cpp:469](../backend/common/download.cpp#L469) | Teredo와 문서용 대역을 제외한 2000::/3만 허용합니다. 매핑 주소, 호환 주소, NAT64, 6to4 주소는 그 안의 IPv4로 판단합니다. |
| `fHandleMedia` | [backend/tools/server/server-common.cpp:872](../backend/tools/server/server-common.cpp#L872) | `media_download_slot` 아래에서 data URI, `--media-path` 파일 또는 원격 URL(`--media-allow-private-urls`가 없으면 공개 주소만)을 읽습니다. |
| `fJsonExceedsDepth` | [backend/common/json-schema-to-grammar.cpp:305](../backend/common/json-schema-to-grammar.cpp#L305) | 변환기가 재귀에 들어가기 전에 거부할 수 있도록 스키마 중첩 깊이를 반복 방식으로 측정합니다(`cMaxSchemaDepth`). |
| `recv_msg` | [backend/ggml/src/ggml-rpc/ggml-rpc.cpp:275](../backend/ggml/src/ggml-rpc/ggml-rpc.cpp#L275) | 선언된 크기가 해당 명령의 상한 이내일 때만 가변 길이 RPC 프레임을 수신합니다. |
| `rpc_server::deserialize_tensor` | [backend/ggml/src/ggml-rpc/ggml-rpc.cpp:1141](../backend/ggml/src/ggml-rpc/ggml-rpc.cpp#L1141) | 상대 측 텐서를 재구성하며, 데이터 범위가 오버플로하거나 버퍼를 벗어나면 `nullptr`을 반환합니다. |
| `socket_t::impl::rdma_recv` | [backend/ggml/src/ggml-rpc/transport.cpp:449](../backend/ggml/src/ggml-rpc/transport.cpp#L449) | 각 RDMA 완료에서 요청한 바이트까지만 복사하고, 나머지는 다음 읽기를 위해 `rx_pending`에 보관합니다. |
| `fRegisterApiDocumentation` | [backend/tools/server/server-api-doc.cpp:384](../backend/tools/server/server-api-doc.cpp#L384) | Swagger 뷰어, 번들 에셋, 실시간 OpenAPI를 제공합니다. |
| `server_shell_mcp::mStart` | [backend/tools/server/server-shell-mcp.cpp:1151](../backend/tools/server/server-shell-mcp.cpp#L1151) | bash, zsh, sh, pwsh/powershell을 감지하고, 지시문에 쓸 홈 디렉터리, 사용자, 호스트를 기록합니다. |
| `server_shell_mcp::mHandlePost` | [backend/tools/server/server-shell-mcp.cpp:1464](../backend/tools/server/server-shell-mcp.cpp#L1464) | Origin과 Content-Type을 확인하고, JSON-RPC 메시지 하나를 파싱해 initialize, ping, tools/list, tools/call에 응답합니다. 알림에는 202로 응답합니다. |
| `server_shell_mcp::mToolsCallResult` | [backend/tools/server/server-shell-mcp.cpp:1311](../backend/tools/server/server-shell-mcp.cpp#L1311) | command, cwd, timeout을 검증하고 명령을 실행한 뒤, exit_code, stdout, stderr를 담은 JSON 객체를 `structuredContent`와 텍스트로 응답합니다. |
| `fRunCommand` | [backend/tools/server/server-shell-mcp.cpp:474](../backend/tools/server/server-shell-mcp.cpp#L474) | 제어 터미널이 없는 새 세션에서 `posix_spawn`을 실행하고(`POSIX_SPAWN_SETSID`, 새 프로세스 그룹도 생성), 두 파이프를 `poll`로 감시하며, 읽은 내용은 모두 `pOutput`(호출 로그)에도 넘기고, 시간 초과나 연결 끊김 시 그룹을 종료합니다. |
| `server_shell_mcp_call_log` | [backend/tools/server/server-shell-mcp.cpp:340](../backend/tools/server/server-shell-mcp.cpp#L340) | `~/.mgs/local-shell-mcp-server/`에 호출마다 JSONL 파일 하나(`y2026m10d12h23m59s59ms731.jsonl`, 모드 0600)를 만듭니다. 실행 전에 `command` 줄을, 출력이 도착하는 대로 전체 출력을 `stdout`/`stderr` 줄로, 마지막에 `result` 또는 `error`를 기록합니다. |
| `fOriginAllowed` | [backend/tools/server/server-shell-mcp.cpp:140](../backend/tools/server/server-shell-mcp.cpp#L140) | Origin이 없는 요청, 이 서버의 출처에서 온 요청, 루프백에서 루프백으로 가는 요청을 허용합니다. |
| `cMcpStore.mSyncShellServer` | [frontend/src/lib/stores/mcp.svelte.ts:598](../frontend/src/lib/stores/mcp.svelte.ts#L598) | `/api/props`에 따라 `mgs-shell` 서버 항목을 추가하거나, 갱신하거나(URL과 이름), 제거합니다. |
| `fIsLocalShellTool` | [frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts:25](../frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts#L25) | `bash.run`, `zsh.run`, `sh.run`, `psh.run`이면 true를 반환합니다. 단, 다른 MCP 서버가 그 도구 이름을 소유하면 그렇지 않습니다. |
| `fParseLocalShellResult` | [frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts:44](../frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/local-shell-run.ts#L44) | 셸 호출의 JSON 객체(exit_code, stdout, stderr, timed_out, cancelled)를 읽으며, 그 밖의 결과에는 null을 반환합니다. |
| `ChatMessageToolCallBlockLocalShell` | [frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockLocalShell.svelte:1](../frontend/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockLocalShell.svelte#L1) | 도구 이름을 제목으로 하고, 종료 코드와 함께 stdin, stdout, stderr 상자를 표시합니다. |
| `server_shell_mcp_askpass::mServe` | [backend/tools/server/server-shell-mcp.cpp:1002](../backend/tools/server/server-shell-mcp.cpp#L1002) | `SO_PEERCRED`와 sudo 부모 프로세스를 확인하고, 요청을 게시한 뒤 UI를 기다리며(최대 5분), 비밀번호나 그 사유로 응답합니다. |
| `fParentIsSudo` | [backend/tools/server/server-shell-mcp.cpp:735](../backend/tools/server/server-shell-mcp.cpp#L735) | 부모 프로세스의 이름이 sudo이고, 그 부모의 실제 uid가 우리와 같으며 유효 uid 또는 저장된 uid가 root인 프로세스를 받아들입니다. |
| `fServerShellMcpAskpassMain` | [backend/tools/server/server-shell-mcp.cpp:1094](../backend/tools/server/server-shell-mcp.cpp#L1094) | sudo가 SUDO_ASKPASS로 실행할 때의 `mgs` 헬퍼 모드입니다. 요청을 보내고 비밀번호를 sudo의 파이프에 씁니다. |
| `server_shell_mcp::mHandleAskpassGet` | [backend/tools/server/server-shell-mcp.cpp:1525](../backend/tools/server/server-shell-mcp.cpp#L1525) | 호출에서 대기 중인 요청을 반환합니다(`?call=`). |
| `server_shell_mcp::mHandleAskpassPost` | [backend/tools/server/server-shell-mcp.cpp:1536](../backend/tools/server/server-shell-mcp.cpp#L1536) | Origin과 Content-Type을 확인하고 비밀번호(한 줄, 최대 4096바이트) 또는 취소를 전달합니다. |
| `fUiSettingsMakeGetHandler` | [backend/tools/server/server-ui-settings.cpp:167](../backend/tools/server/server-ui-settings.cpp#L167) | `GET /api/ui/settings`: 저장된 문서를 반환하며, 처음 저장하기 전에는 `{"settings": null}`을 반환합니다. |
| `fUiSettingsMakePostHandler` | [backend/tools/server/server-ui-settings.cpp:181](../backend/tools/server/server-ui-settings.cpp#L181) | `POST /api/ui/settings`: `{"settings": {...}}`를 검증하고 `~/.mgs/config/ui-settings.json`을 원자적으로 교체합니다(0600). |
| `fUiSettingsLanguage` | [backend/tools/server/server-ui-settings.cpp:142](../backend/tools/server/server-ui-settings.cpp#L142) | 저장된 인터페이스 언어로, 유효한 언어 태그일 때만 `/api/config.js`가 게시합니다. |
| `UiSettingsStore::mLoad` | [frontend/src/lib/stores/ui-settings.svelte.ts:82](../frontend/src/lib/stores/ui-settings.svelte.ts#L82) | 서버의 설정 문서를 한 번 로드하고 등록된 모든 섹션 로더를 실행합니다. API 키가 없는 경우가 아니면 다시 시도합니다. |
| `UiSettingsStore::mSet` | [frontend/src/lib/stores/ui-settings.svelte.ts:72](../frontend/src/lib/stores/ui-settings.svelte.ts#L72) | 섹션 하나를 변경하고 300 ms 뒤에 문서 전체를 저장합니다. 문서가 로드되기 전에는 무시됩니다. |
| `UiSettingsStore::mFlush` | [frontend/src/lib/stores/ui-settings.svelte.ts:96](../frontend/src/lib/stores/ui-settings.svelte.ts#L96) | 대기 중인 변경 사항을 즉시 저장하고 완료될 때까지 기다립니다(언어를 바꿔 다시 로드하기 전에 사용). |
| `cMcpStore.mActiveShellCalls` | [frontend/src/lib/stores/mcp.svelte.ts:1233](../frontend/src/lib/stores/mcp.svelte.ts#L1233) | 로컬 셸 MCP 서버에서 실행 중인 호출(`_meta`로 보낸 ID, 도구, 명령)로, `mExecuteTool`이 설정합니다. |
| `DialogLocalShellAskpass` | [frontend/src/lib/components/app/dialogs/DialogLocalShellAskpass.svelte:1](../frontend/src/lib/components/app/dialogs/DialogLocalShellAskpass.svelte#L1) | 루트 레이아웃의 가운데 정렬된 대화 상자입니다. 실행 중인 셸 호출의 sudo 요청을 폴링하고, 확인 또는 Enter(`fHandlePasswordKeydown`)나 취소 또는 Escape로 응답합니다. 요청이 표시되면 곧바로 비밀번호 필드가 포커스를 받습니다(`fFocusPassword`). |
| `LocalShellAskpassService` | [frontend/src/lib/services/local-shell-askpass.service.ts:16](../frontend/src/lib/services/local-shell-askpass.service.ts#L16) | `/api/mcp/shell/askpass`를 사용하는 `mPending`, `mAnswer`, `mCancel`을 제공합니다. |
| `MCPService.mCreateTransport` | [frontend/src/lib/services/mcp.service.ts:414](../frontend/src/lib/services/mcp.service.ts#L414) | MCP 전송 계층을 구성하고, 같은 출처의 서버와 CORS 프록시에 UI의 API 키를 추가합니다. |
| `fCompletionProperties` | [backend/tools/server/server-api-doc.cpp:32](../backend/tools/server/server-api-doc.cpp#L32) | server-schema의 completion 매개변수와 설명을 노출합니다. |
| `server_model_meta::mUpdateArgs` | [backend/tools/server/server-models.cpp:241](../backend/tools/server/server-models.cpp#L241) | 자식 프로세스에 명시적 HTTP, 비공개 포트, /api 접두사를 강제합니다. |
| `common_subproc` | [backend/common/subproc.h:21](../backend/common/subproc.h#L21) | `vProc`을 소유하고, `MGS_SUBPROCESS` 유무에 관계없이 같은 메서드를 노출합니다. 인수/환경 벡터는 null로 끝나며, stdin을 닫으면 join/소멸 전에 벤더의 `stdin_file`이 분리됩니다. |
| `fCommonParamsPrintInfo` | [backend/common/common.cpp:448](../backend/common/common.cpp#L448) | 빌드 및 시스템 정보를 로그로 남깁니다. 두 전처리기 분기 모두 `vBuildType`을 선언하며, `NDEBUG`(Release)일 때는 비어 있고 그렇지 않으면 ` (debug)`입니다. |
| `fCommonMaybeEnableEmbeddedMtp` | [backend/common/common.cpp:1539](../backend/common/common.cpp#L1539) | 명시적 옵션을 존중하면서 자동 추측 디코딩을 적용합니다. |
| `fDetectEmbeddedMtp` | [backend/common/common.cpp:1422](../backend/common/common.cpp#L1422) | 모든 GGUF 샤드를 검사해 지원되는 완전한 MTP 헤드를 찾습니다. |
| `fCommonModelParamsToMgs` | [backend/common/common.cpp:1933](../backend/common/common.cpp#L1933) | 공통 매개변수를 모델 매개변수로 변환합니다. 이식된 MTP 레이어는 이를 대체하는 별도의 MTP GGUF가 없을 때만 로드합니다. |
| `mgs_model_base::mLoadTensors` | [backend/src/mgs-model.cpp:1224](../backend/src/mgs-model.cpp#L1224) | 레이어를 디바이스에 할당하고, MTP 레이어와 출력 레이어는 들어갈 수 있으면 GPU에 둡니다. |
| `mgs_model_loader::mCreateTensor` | [backend/src/mgs-model-loader.cpp:1183](../backend/src/mgs-model-loader.cpp#L1183) | 각 가중치의 버퍼 유형을 고르며, MTP 가중치를 호스트 메모리로 옮기는 것을 제외한 오버라이드를 적용합니다. |
| `mgs_mmap::mRegisterHost` | [backend/src/mgs-mmap.cpp:639](../backend/src/mgs-mmap.cpp#L639) | 매핑된 페이지를 고정하고 그 등록 해제 콜백을 보관합니다. |
| `fInitializeExpertPrefetch` | [backend/ggml/src/ggml-backend.cpp:1650](../backend/ggml/src/ggml-backend.cpp#L1650) | 메모리가 부족할 때의 대체 경로를 갖추고 전문가 슬롯을 할당/재사용합니다. |
| `fApiUrl` | [frontend/src/lib/api-url.ts:8](../frontend/src/lib/api-url.ts#L8) | 서버가 게시한 API 접두사를 기준으로 UI 요청의 경로를 해석합니다. |
| `aLanguages` | [frontend/src/lib/i18n.ts:17](../frontend/src/lib/i18n.ts#L17) | 지원되는 로캘 코드를 알파벳순으로 담습니다. 여기서 `Language` 타입과 선택기 옵션이 파생됩니다. |
| `fApplyResponseLanguage` | [frontend/src/lib/utils/response-language.ts:6](../frontend/src/lib/utils/response-language.ts#L6) | 켜져 있으면 첫 번째 시스템 메시지(문자열 또는 콘텐츠 파트)의 사본에 현지화된 지시문을 덧붙이거나, 필요하면 시스템 메시지를 앞에 추가합니다. 저장된 기록과 호출자의 객체는 건드리지 않습니다. |
| `fGetLanguage` | [frontend/src/lib/i18n.ts:70](../frontend/src/lib/i18n.ts#L70) | URL/저장소로 선택된 로캘을 반환하며, 없으면 `en-US`를 반환합니다. 레이아웃은 이를 HTML 언어와 방향에 사용합니다. |
| `fTranslate` | [frontend/src/lib/i18n.ts:74](../frontend/src/lib/i18n.ts#L74) | 메시지를 해석하고 이름이 붙은 값을 보간합니다. |
| `fSetLanguage` | [frontend/src/lib/i18n.ts:81](../frontend/src/lib/i18n.ts#L81) | 언어를 저장소/URL에 저장하고 UI를 다시 로드합니다. |
| `fMain (installer)` | [deploy/install-common.sh:171](../deploy/install-common.sh#L171) | 옵션을 검증하고, 의존성을 설치하며, build/를 포함한 소스를 준비하고, 맞지 않는 CMake 메타데이터를 새로 고친 뒤 source/build에서 빌드합니다. |
| `fInstallNccl` | [deploy/install-common.sh:118](../deploy/install-common.sh#L118) | 감지된 CUDA 릴리스에 맞는 NCCL을 선택하고, 이 APT 명령에서 다운그레이드를 허용해 libnccl2/libnccl-dev를 정확히 같은 버전으로 설치합니다. |
| `fMain (Debian)` | [deploy/install-update-reinstall-debian.sh:168](../deploy/install-update-reinstall-debian.sh#L168) | 로컬/원격 모드를 선택하고 Debian 설치를 조율합니다. |
| `fDownloadSource (Debian)` | [deploy/install-update-reinstall-debian.sh:114](../deploy/install-update-reinstall-debian.sh#L114) | 캐시를 유지하면서 소스 아카이브를 검증하고 동기화합니다. |
| `fLockSourceDirectory (Debian)` | [deploy/install-update-reinstall-debian.sh:91](../deploy/install-update-reinstall-debian.sh#L91) | 같은 체크아웃을 사용하는 설치가 동시에 실행되지 않게 막습니다. |
| `fRunLocalInstaller (Debian)` | [deploy/install-update-reinstall-debian.sh:150](../deploy/install-update-reinstall-debian.sh#L150) | 함수/트랩을 격리하고 stdin을 닫은 별도의 Bash에서 헬퍼를 실행합니다. |
| `fMain (web deployment)` | [deploy/configure-web.sh:40](../deploy/configure-web.sh#L40) | 자격 증명, 인증서, 가상 호스트, 서비스 파일을 만듭니다. |
| `fMain (local build)` | [build/build-local.sh:261](../build/build-local.sh#L261) | 로컬 또는 원격 소스를 선택하고, 일회용 /tmp/ 작업 공간을 만들며, source/build에서 컴파일해 번들을 조립하고, 출력 디렉터리를 잠근 뒤 결과를 복사하고 검증합니다. |
| `fHasProjectSources` | [build/build-local.sh:96](../build/build-local.sh#L96) | 로컬 소스를 선택할 때와 내려받은 아카이브를 검증할 때, 필요한 소스 파일과 라이선스 디렉터리가 있는지 확인합니다. |
| `fDownloadSources` | [build/build-local.sh:107](../build/build-local.sh#L107) | main을 HTTPS로 내려받고, 아카이브 경로를 검증하며, 비공개 /tmp/ 작업 공간 안에 소스를 풀고, 필요한 프로젝트 파일을 확인합니다. |
| `fCleanup (local build)` | [build/build-local.sh:16](../build/build-local.sh#L16) | 로그 기록기를 닫고 끝날 때까지 기다리며, 대상의 불완전한 항목과 비공개 /tmp/ 작업 공간을 제거하고, 종료 상태를 보존합니다. |
| `fBundleDependencies` | [build/build-local.sh:167](../build/build-local.sh#L167) | NVIDIA 드라이버 라이브러리를 제외하고, 객체의 `ldd` 의존성 폐포를 `lib/`에 복사합니다. |
| `fBundleNameServiceModules` | [build/build-local.sh:184](../build/build-local.sh#L184) | 호스트 및 사용자 조회를 위해 glibc NSS 모듈과 그 의존성을 번들에 포함합니다. |
| `fPinLibrarySearchPaths` | [build/build-local.sh:196](../build/build-local.sh#L196) | 번들된 `RUNPATH` 항목을 `DT_RPATH` `$ORIGIN`으로 변환합니다. |
| `fReplaceDirectory` | [build/build-local.sh:211](../build/build-local.sh#L211) | 새로 조립한 폴더를 제자리에 바꿔 넣으며, 실패하면 이전 폴더를 복원합니다. |
| `fVerifyBundle` | [build/build-local.sh:228](../build/build-local.sh#L228) | 런처가 실행하는 방식 그대로 번들된 로더로 의존성을 나열하고, `lib/` 밖에서 해석된 라이브러리를 거부합니다. |
| `main (launcher)` | [build/mgs-launcher.c:27](../build/mgs-launcher.c#L27) | `bin/`을 기준으로 자신의 접두사를 해석하고, `MGS_EXECUTABLE`을 내보낸 뒤 `lib/mgs/<loader> lib/mgs/mgs`를 실행합니다. |
| `fGetServerExecPath` | [backend/tools/server/server-models.cpp:76](../backend/tools/server/server-models.cpp#L76) | 라우터 자식 프로세스용 실행 파일을 고르며, `MGS_EXECUTABLE`을 우선합니다. |
| `fReadOccupiedPorts` | [deploy/start-web.sh:40](../deploy/start-web.sh#L40) | 다른 서비스를 멈추지 않고 IPv4/IPv6 리스너를 읽습니다. |
| `fFindFreePort` | [deploy/start-web.sh:52](../deploy/start-web.sh#L52) | 예약된 HTTPS 포트와 바인드가 거부된 포트를 제외하고 다음 빈 포트를 찾습니다. |
| `fOwnsListener` | [deploy/start-web.sh:65](../deploy/start-web.sh#L65) | 리스너가 Apache 자식 프로세스의 것인지 확인합니다. |
| `fPublishPorts` | [deploy/start-web.sh:72](../deploy/start-web.sh#L72) | 바인드한 뒤 포트, URL, PID를 원자적으로 게시합니다. |
| `fMain` | [deploy/start-web.sh:82](../deploy/start-web.sh#L82) | Apache를 감독하고 종료 신호를 전달합니다. |
| `ggml_backend_sched_compute_splits` | [backend/ggml/src/ggml-backend.cpp:1714](../backend/ggml/src/ggml-backend.cpp#L1714) | 분할을 동기화하고, 선택적/프리페치 전문가 복사를 디스패치합니다. |
| `ggml_cuda_mul_mat_id_needs_sync` | [backend/ggml/src/ggml-cuda/ggml-cuda.cu:1870](../backend/ggml/src/ggml-cuda/ggml-cuda.cu#L1870) | 비동기 전문가 커널에서 그래프 캡처를 유지합니다. |
| `ggml_cuda_fattn_vec_instances` | [backend/ggml/cmake/common.cmake:53](../backend/ggml/cmake/common.cmake#L53) | CUDA, HIP, MUSA용으로 컴파일할 K/V 양자화 쌍을 선택합니다. |
| `mgs_context::mGetGfResPrev` | [backend/src/mgs-context.cpp:2365](../backend/src/mgs-context.cpp#L2365) | 출력 유무에 따라 아레나를 선택하고, 필요할 때 지연 생성합니다. |
| `mgs_context::mOutputReorder` | [backend/src/mgs-context.cpp:2257](../backend/src/mgs-context.cpp#L2257) | 마스킹되지 않은 NextN 행과 레이어 입력 행의 원래 토큰 순서를 복원합니다. |
| `mgs_model_glm4_moe::graph_mtp` | [backend/src/models/glm4-moe.cpp:140](../backend/src/models/glm4-moe.cpp#L140) | 단일 헤드 GLM MTP 디코더를 구성합니다. |
| `fCommonSamplerSampleAndAcceptN` | [backend/common/sampling.cpp:655](../backend/common/sampling.cpp#L655) | 끝부분이 아닌 첫 번째 EOG에서 추측 디코딩의 수락을 끝냅니다. |
| `server_routes::mHandleCountTokens` | [backend/tools/server/server-context.cpp:5730](../backend/tools/server/server-context.cpp#L5730) | 응답이 서버를 깨운 뒤에만 모델 포인터를 사용합니다. |
| `ConversationsStore::mGetConversationsForExport` | [frontend/src/lib/stores/conversations.svelte.ts:542](../frontend/src/lib/stores/conversations.svelte.ts#L542) | 영속 저장된 메타데이터와 전체 메시지 트리를 선택한 순서대로 읽습니다. |
| `ConversationsStore::mParseImportFile` | [frontend/src/lib/stores/conversations.svelte.ts:1061](../frontend/src/lib/stores/conversations.svelte.ts#L1061) | 파일 크기를 확인한 뒤, 압축을 풀기 전에 `unzipSync`로 ZIP 항목을 거릅니다(`.jsonl`만, 항목 수와 선언된 크기). |
| `ConversationsStore::mParseSessionsJsonl` | [frontend/src/lib/stores/conversations.svelte.ts:986](../frontend/src/lib/stores/conversations.svelte.ts#L986) | 가져온 모든 대화와 메시지를 검증하고, 각 `convId`가 자신의 대화와 일치하도록 요구합니다. |
| `DatabaseService.mImportConversations` | [frontend/src/lib/services/database.service.ts:574](../frontend/src/lib/services/database.service.ts#L574) | 각 대화와 그 메시지를 한 트랜잭션 안에서 `bulkAdd`로 추가하며, 메시지 ID가 이미 있으면 그 대화 전체를 건너뜁니다. |
| `fCreateImageErrorFallback` | [frontend/src/lib/utils/image-error-fallback.ts:10](../frontend/src/lib/utils/image-error-fallback.ts#L10) | DOM 노드와 `textContent`로 깨진 이미지 안내를 구성하며, http(s) URL만 링크로 만듭니다. |
| `fIsSafeLinkUrl` | [frontend/src/lib/utils/safe-url.ts:50](../frontend/src/lib/utils/safe-url.ts#L50) | 상대 링크와 http, https, mailto, tel 스킴을 허용합니다. `fIsDataImageUrl`은 `data:image/...;base64`만 받아들입니다. |
| `ToolsStore::mResolveEnabledEntry` | [frontend/src/lib/stores/tools.svelte.ts:320](../frontend/src/lib/stores/tools.svelte.ts#L320) | 노출, 승인, 실행을 위해 도구 이름의 활성화된 항목 하나를 결정합니다. 두 서버가 함께 제공하는 MCP 이름은 어떤 항목으로도 결정되지 않습니다(`mConflictingMcpToolNames`). |
| `MCPStore::mExecuteTool` | [frontend/src/lib/stores/mcp.svelte.ts:1175](../frontend/src/lib/stores/mcp.svelte.ts#L1175) | 승인된 바로 그 서버 ID에서, 그 서버가 해당 도구를 광고하는지 확인한 뒤 도구를 실행합니다. |
| `mgs_expert_cache::mFCreate` | [backend/src/mgs-expert-cache.cpp:181](../backend/src/mgs-expert-cache.cpp#L181) | 전문가가 호스트에 있는 GPU 레이어를 선택하고, GPU마다 풀 하나의 크기를 정해 할당하며, 모든 GPU의 업로드 대역폭을 측정합니다. |
| `mgs_expert_cache::mPartition` | [backend/src/mgs-expert-cache.cpp:838](../backend/src/mgs-expert-cache.cpp#L838) | 마이크로 배치를 적중, 업로드, CPU 전문가로 나누고, 식별자와 업로드를 대기열에 넣습니다. |
| `mgs_expert_cache::mUploadWeights` | [backend/src/mgs-expert-cache.cpp:701](../backend/src/mgs-expert-cache.cpp#L701) | 프리페치 슬롯을 채웁니다. 캐시된 전문가는 디바이스 간 복사로, 나머지는 호스트 메모리에서 가져옵니다. |
| `mgs_expert_cache::mFillInfo` | [backend/src/mgs-expert-cache.cpp:972](../backend/src/mgs-expert-cache.cpp#L972) | 크기, 내용, 카운터를 보고합니다. |
| `llm_graph_context::mBuildMoeFfn` | [backend/src/mgs-graph.cpp:1500](../backend/src/mgs-graph.cpp#L1500) | 일반 전문가 체인 또는 하이브리드 디바이스 분기와 CPU 분기를 구성합니다. |
| `ggml_backend_sched_set_weight_upload` | [backend/ggml/src/ggml-backend.cpp:2197](../backend/ggml/src/ggml-backend.cpp#L2197) | 전문가 프리페치가 사용하는 업로드 콜백을 설치합니다. |
| `mgs_set_expert_cache_size` | [backend/src/mgs-context.cpp:4065](../backend/src/mgs-context.cpp#L4065) | 새 캐시 크기를 요청하며, 다음 디코딩에서 적용됩니다. |
| `mgs_get_expert_cache_info` | [backend/src/mgs-context.cpp:4063](../backend/src/mgs-context.cpp#L4063) | 모니터링용으로 캐시 상태를 반환합니다. |
| `fCommonMoeCacheSize` | [backend/common/common.cpp:1067](../backend/common/common.cpp#L1067) | 옵션과 환경 변수로부터 구성된 캐시 크기를 결정합니다. |
| `fHostAvailableMemory` | [backend/common/fit.cpp:185](../backend/common/fit.cpp#L185) | RAM에 두는 전문가에 쓸 수 있는 시스템 메모리를 읽습니다. |
| `fRaiseContext` (`fCommonParamsFitImpl` 안의 람다) | [backend/common/fit.cpp:619](../backend/common/fit.cpp#L619) | `--fit`으로 줄어든 컨텍스트를, KV 캐시가 dense 가중치를 올리고 남은 디바이스 메모리의 최대 4분의 1을 차지할 때까지 늘립니다. |
| `fRaiseBatch` (`fCommonParamsFitImpl` 안의 람다) | [backend/common/fit.cpp:678](../backend/common/fit.cpp#L678) | 전문가가 시스템 메모리에 머물고 `-b`/`-ub`가 지정되지 않았을 때 마이크로 배치와 논리 배치를 늘립니다(4096, 2048 또는 1024). |
| `fCommonDefaultThreads` | [backend/common/common.cpp:252](../backend/common/common.cpp#L252) | 기본 CPU 스레드 수로, 연산에 유용한 코어 수이되 친화성 마스크에 포함된 논리 CPU의 절반을 넘지 않습니다. |
| `server_routes::vPostExpertCache` | [backend/tools/server/server-context.cpp:5464](../backend/tools/server/server-context.cpp#L5464) | `pSizeMib`를 검증하고 캐시 크기 조정을 대기열에 넣습니다. |
| `fExpertCacheJson` | [backend/tools/server/server-context.cpp:872](../backend/tools/server/server-context.cpp#L872) | `/expert-cache`와 `/metrics`용으로 캐시 상태를 변환합니다. |
| `mCreateCheckpoint` | [backend/tools/server/server-context.cpp:2575](../backend/tools/server/server-context.cpp#L2575) | 체크포인트를 저장하며, 이웃과 가장 가까운 체크포인트를 제거합니다. |
| `common_chat_msg_spans::mIsTurnStart` | [backend/common/chat.h:178](../backend/common/chat.h#L178) | 사용자, 어시스턴트, 도구 메시지의 시작을 감지합니다. |
| `mgs_model_get_weight_breakdown` | [backend/src/mgs-model.cpp:2755](../backend/src/mgs-model.cpp#L2755) | 모델에 로드된 가중치를 버퍼 유형별로 MTP(nextn) 레이어와 라우팅 전문가(`*exps.*`)로 나눕니다. |
| `mgs_get_expert_memory_breakdown` | [backend/src/mgs-context.cpp:4086](../backend/src/mgs-context.cpp#L4086) | 전문가 캐시 버퍼와 프리페치 슬롯의 메모리를 버퍼 유형별로 반환합니다. |
| `ggml_backend_sched_get_prefetch_size` | [backend/ggml/src/ggml-backend.cpp:2204](../backend/ggml/src/ggml-backend.cpp#L2204) | 스케줄러의 전문가 프리페치 슬롯이 차지하는 바이트 수와 버퍼 유형을 반환합니다. |
| `mgs_expert_cache::mAddMemory` | [backend/src/mgs-expert-cache.cpp:995](../backend/src/mgs-expert-cache.cpp#L995) | 캐시의 풀, 슬롯 식별자, 스테이징 버퍼를 버퍼 유형별로 합산합니다. |
| `server_context_impl::mGetMonitorSnapshot` | [backend/tools/server/server-context.cpp:920](../backend/tools/server/server-context.cpp#L920) | 컨텍스트나 작업 대기열을 건드리지 않고, 게시된 메모리 스냅샷을 HTTP 스레드용으로 복사합니다. |
| `server_context_impl::mBuildMonitorSnapshot` | [backend/tools/server/server-context.cpp:1002](../backend/tools/server/server-context.cpp#L1002) | 타깃 컨텍스트와 초안 컨텍스트의 메모리 세그먼트를 GPU별로, 그리고 RAM에 대해 구성하고, 프롬프트 캐시와 체크포인트도 더합니다. |
| `server_context_impl::mRefreshMonitorSnapshot` | [backend/tools/server/server-context.cpp:1132](../backend/tools/server/server-context.cpp#L1132) | 로드한 뒤 스냅샷을 게시하며, 슬롯이 작업하는 동안에는 초당 최대 한 번, 유휴 상태일 때는 항상 게시합니다. |
| `fMonitorReport` | [backend/tools/server/server-monitor.cpp:698](../backend/tools/server/server-monitor.cpp#L698) | CPU, RAM, GPU 측정값을 스냅샷과 결합합니다. GPU별(PCI ID 기준) 및 RAM의 세그먼트, "other", 여유 메모리로 구성됩니다. |
| `fMonitorUsageJson` | [backend/tools/server/server-monitor.cpp:641](../backend/tools/server/server-monitor.cpp#L641) | 상태 표시줄용 CPU 사용률(전체 및 물리 프로세서별)과 GPU 사용률을 반환합니다. |
| `fCpuJson` | [backend/tools/server/server-monitor.cpp:183](../backend/tools/server/server-monitor.cpp#L183) | `/proc/stat`의 변화량으로 계산한 CPU 사용률로, 전체와 물리 프로세서별(`processors`: 모델, 사용률, 번호가 붙은 논리 CPU) 값을 750 ms 동안 공유합니다. 10초보다 오래된 기준값은 250 ms 간격의 측정으로 대체됩니다. |
| `fNvml` | [backend/tools/server/server-monitor.cpp:444](../backend/tools/server/server-monitor.cpp#L444) | `dlopen`으로 `libnvidia-ml.so.1`을 한 번 열고, 모니터가 사용하는 NVML 호출을 해석합니다. |
| `fCollectMonitorSnapshots` | [backend/tools/server/server-models.cpp:1593](../backend/tools/server/server-models.cpp#L1593) | 준비된 모든 라우터 자식 프로세스에 `/api/monitor/memory`를 요청합니다. 로딩 중이거나 절전 중인 자식 프로세스는 상태만 보고합니다. |
| `MonitorStore::mSubscribeReport` | [frontend/src/lib/stores/monitor.svelte.ts:39](../frontend/src/lib/stores/monitor.svelte.ts#L39) | 참조 횟수로 관리되는 `/monitor` 폴링을 2초마다 수행하며, 탭이 숨겨져 있는 동안에는 멈춥니다. 상태 표시줄에도 데이터를 공급합니다. |
| `MonitorStore::mSubscribeUsage` | [frontend/src/lib/stores/monitor.svelte.ts:24](../frontend/src/lib/stores/monitor.svelte.ts#L24) | 보고서를 폴링하지 않는 동안, 참조 횟수로 관리되는 `/monitor/usage` 폴링을 2초마다 수행합니다. |
| `aMonitorSegmentGroups` | [frontend/src/lib/constants/monitor.ts:23](../frontend/src/lib/constants/monitor.ts#L23) | 메모리 막대 세그먼트의 고정된 순서, 레이블, 색상 변수를 정의합니다. |
| `MonitorScreen` | [frontend/src/lib/components/app/monitor/MonitorScreen.svelte:1](../frontend/src/lib/components/app/monitor/MonitorScreen.svelte#L1) | 모니터링 페이지입니다. 사용률 카드(GPU를 먼저 `gpu-N`으로, 이어서 물리 프로세서마다 카드 하나를 `cpu-N`으로 표시하며, 각 레이블은 굵은 글씨로 디바이스 이름 앞에 둠)와, 막대 순서대로 한 줄에 세그먼트 하나씩 범례를 보여 주는 메모리 막대(`MonitorMemoryBar`)로 구성됩니다. |
| `StatusBar` | [frontend/src/lib/components/app/monitor/StatusBar.svelte:1](../frontend/src/lib/components/app/monitor/StatusBar.svelte#L1) | 크기가 같은 미터를 GPU마다 하나씩(`gpu-1`, `gpu-2`...), 이어서 물리 프로세서마다 하나씩(`cpu-1`, `cpu-2`...) 표시하는 하단 고정 막대로, 레이블은 모니터링 카드와 같습니다. 다 들어가지 않으면 옆으로 스크롤되며, 누르면 모니터링 페이지가 열립니다. |
| `fMonitorDrmResidency` | [backend/tools/server/server-monitor.cpp:347](../backend/tools/server/server-monitor.cpp#L347) | 커널 DRM 드라이버가 프로세스의 GPU 메모리를 어디에 두고 있는지(VRAM, GTT, 축출된 VRAM)를 `/proc/self/fdinfo`에서 PCI ID별로 읽습니다. |
| `server_context_impl::mCheckVramResidency` | [backend/tools/server/server-context.cpp:1155](../backend/tools/server/server-context.cpp#L1155) | 모델이 GPU에 할당한 것의 일부가 VRAM 밖에 있을 때와 다시 돌아왔을 때 한 번씩 경고합니다. |
| `fPoolBufferType` | [backend/src/mgs-expert-cache.cpp:135](../backend/src/mgs-expert-cache.cpp#L135) | 백엔드가 축출 우선순위가 더 낮은 버퍼 유형을 내보내면, 전문가 캐시 풀에 그 유형을 선택합니다. |
| `ggml_backend_vk_device_get_cache_buffer_type` | [backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp:20805](../backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L20805) | `VK_EXT_memory_priority` 0.5가 적용된 Vulkan 디바이스 메모리로, `ggml_backend_dev_get_cache_buffer_type`으로 내보내집니다. |
| `ggml_backend_vk_buffer_is_device_of` | [backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp:18281](../backend/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L18281) | 비동기 복사에서 백엔드 GPU의 모든 디바이스 버퍼 유형(기본 유형과 캐시 유형)을 받아들입니다. |
| `server_context_impl::mDeferTask` | [backend/tools/server/server-context.cpp:2685](../backend/tools/server/server-context.cpp#L2685) | 빈 슬롯을 찾지 못한 작업을 대기열에 넣고, 첫 번째 대기열 알림(앞에 있는 사용 중인 슬롯과 대기 중인 작업을 출처별로)을 보냅니다. |
| `server_context_impl::mNotifyQueuePositions` | [backend/tools/server/server-context.cpp:2672](../backend/tools/server/server-context.cpp#L2672) | 작업이 시작되거나 다시 대기열에 들어간 뒤, 알림을 요청한 모든 대기 작업에 현재 위치를 보냅니다. |
| `server_queue::mDeferredTasks` | [backend/tools/server/server-queue.h:93](../backend/tools/server/server-queue.h#L93) | 지연된 작업의 ID, 인덱스, 출처, 알림 플래그를 도착 순서대로 담습니다. |
| `server_task_result_queue` | [backend/tools/server/server-task.h:476](../backend/tools/server/server-task.h#L476) | `{"mgs_queue": {"ahead", "ahead_api", "ahead_webui"}}`로 직렬화되는 대기열 알림으로, 스트림을 끝내지 않습니다. |
| `ChatService.mHandleStreamResponse` | [frontend/src/lib/services/chat.service.ts:754](../frontend/src/lib/services/chat.service.ts#L754) | completion의 SSE 스트림을 파싱하고, `mgs_queue` 이벤트는 `onQueue`에 넘기며, 첫 번째 일반 청크가 오면 이를 지웁니다. |
| `ChatStore::mGetQueueStatus` | [frontend/src/lib/stores/chat.svelte.ts:585](../frontend/src/lib/stores/chat.svelte.ts#L585) | 대화의 요청이 슬롯을 기다리는 동안의 대기열 상태이며, 그렇지 않으면 null입니다. |
| `fQueueStatusText` | [frontend/src/lib/utils/progress.ts:15](../frontend/src/lib/utils/progress.ts#L15) | 채팅의 대기열 알림입니다. 앞에 있는 요청 수와 함께 API 변형 또는 웹 UI 변형을 반환하며, 요청이 기다리지 않으면 null을 반환합니다. |
| `mgs_model_deepseek41` | [backend/src/models/models.h:258](../backend/src/models/models.h#L258) | DeepSeek-V4.1-Flash 모델로, 두 GGUF 레이아웃, 외부 Engram 테이블, 공유 압축 KV, 후보 풀을 다룹니다. |
| `fMgsDeepseek41AdaptMetadata` | [backend/src/models/deepseek41.cpp:140](../backend/src/models/deepseek41.cpp#L140) | 모델을 생성하기 전에 `deepseek41` GGUF의 DwarfStar 키를 표준 키로 변환합니다. |
| `mgs_model_deepseek41::mEngramReadRows` | [backend/src/models/deepseek41.cpp:380](../backend/src/models/deepseek41.cpp#L380) | `pread`로 Engram 행을 읽고 E4M3+E8M0을 디코딩하거나 F32로 역양자화합니다. |
| `mgs_kv_cache_dsv41` | [backend/src/mgs-kv-cache-dsv41.h:87](../backend/src/mgs-kv-cache-dsv41.h#L87) | DeepSeek-V4.1 캐시로, 원시 SWA 캐시에 더해 비율마다 압축 그룹 하나를 두며 읽기 레이어들이 이를 공유합니다. |
| `mgs_model_loader::cTensorExternal` | [backend/src/mgs-model-loader.h:73](../backend/src/mgs-model-loader.h#L73) | 텐서 플래그로, 집계에는 포함되지만 할당되지는 않습니다. 프리페치는 이 플래그가 붙은 첫 번째 텐서 앞에서 멈춥니다. |
| `mgs_model_glm5_next` | [backend/src/models/models.h:344](../backend/src/models/models.h#L344) | GLM-5.3-Flash 모델로, k-pool DSA 인덱서를 갖춘 KDA + nope 전용 MLA, 하이퍼 커넥션, MTP를 구현합니다. |
| `mgs_model_qwen4exp` | [backend/src/models/models.h:749](../backend/src/models/models.h#L749) | Qwen3.8-Flash-Next 모델로, 저랭크 하이퍼 커넥션, n-gram 레이어별 임베딩, gated delta-net, QSA, MTP를 구현합니다. |
| `mgs_memory_hybrid_idx` | [backend/src/mgs-memory-hybrid-idx.h:20](../backend/src/mgs-memory-hybrid-idx.h#L20) | qwen4exp와 glm5-next의 KV 캐시, 순환 캐시, k-pool 인덱서 캐시입니다. |
| `mgs_model_kimi_k3` | [backend/src/models/models.h:855](../backend/src/models/models.h#L855) | Kimi-K3 모델로, KDA + MLA 하이브리드, 레이어 간 잔차 혼합, 896개 전문가를 가진 잠재 MoE를 구현합니다. |
| `fCommonChatParamsInitKimiK3` | [backend/common/chat.cpp:1838](../backend/common/chat.cpp#L1838) | Kimi-K3 채팅 형식(`<|open|>`/`<|sep|>`/`<|close|>`)으로, 생각, 응답, 도구 섹션을 다룹니다. |
| `DEEPSEEK41_TEMPLATE_SRC` | [backend/common/chat.cpp:642](../backend/common/chat.cpp#L642) | `deepseek41` GGUF에 템플릿이 없을 때 사용하는 내장 Jinja 템플릿입니다. |
| `ggml_swiglu_clamp` | [backend/ggml/include/ggml.h:1346](../backend/ggml/include/ggml.h#L1346) | 입력을 클램프하는 SwiGLU입니다(CPU, 융합된 mat-vec 경로를 쓰는 CUDA, Vulkan). |
| `ggml_dsv4_hc_pre_gated` | [backend/ggml/include/ggml.h:2572](../backend/ggml/include/ggml.h#L2572) | 원소별 시그모이드 게이트가 붙은 하이퍼 커넥션 사전 혼합입니다(qwen4exp, CPU와 CUDA). |


<a id="flows"></a>
## 주요 호출 경로

1. **설치:** Debian 파일/stdin 진입 → 검증 → 로그와 잠금 → 로컬 체크아웃 또는 GitHub `main` 아카이브 → 소스 검증/동기화 → stdin을 닫은 자식 Bash → `install-common.sh:fMain` → 의존성 → `build/`를 포함한 안정적인 `_/temp/source` 스테이징 → `-S _/temp/source/build`를 지정한 CMake → `bin/mgs` 설치. Alpine은 로컬 체크아웃에서 헬퍼를 직접 호출합니다.
2. **배포:** `configure-web.sh` → 서비스 계정 → API 키/인증서 → 독립형 템플릿과 서비스 → `start-web.sh` → 점유된 리스너 → 11080+/11443+ 선택 → Apache 바인드 → 두 리스너의 소유권 확인 → `ports.json`. 실제 바인드 충돌이면 선택 단계로 돌아가고, 그 밖의 실패는 서비스를 중지합니다.
3. **채팅:** 컴포넌트 → `ChatService.mSendMessage` → 정규화된 메시지와 사용자 지정 JSON → 현재의 `respondInInterfaceLanguage` 플래그와 UI 로캘을 사용하는 `fApplyResponseLanguage` → `fApiUrl` → HTTP 등록 → 인증/준비 상태 확인 → `server_routes` → 작업 대기열/슬롯 → mgs/GGML 평가 → JSON 또는 SSE → 브라우저 저장소와 렌더링. `mGenerateTitle`과 모든 에이전트 턴도 같은 경로를 사용하며, `mPreEncode`는 KV 캐시를 준비하기 전에 같은 시스템 지시문을 적용합니다.
4. **라우터:** 외부 핸들러 → 정규 내부 경로 → 모델 선택 → 자식 프로세스의 `/api` 요청 → 응답 또는 보존된 스트림. 조회, 재개, 취소도 같은 계약을 따릅니다.
5. **MTP:** 공통 초기화 → `fCommonMaybeEnableEmbeddedMtp` → `fDetectEmbeddedMtp` → 모든 GGUF 샤드의 헤더/텐서 → 아키텍처/트렁크/헤드 검사 → 명시적으로 재정의하지 않았다면 `draft-mtp` 선택 → 모델 로드: `mgs_model_base::mLoadTensors`가 MTP 레이어와 출력 레이어를 GPU에 배치 → `mgs_model_loader::mCreateTensor`가 이들에 대해서는 호스트 메모리로 보내는 오버라이드를 건너뜀.
6. **MoE:** 매핑된 가중치 → 호스트 등록 → 그래프 분할 → `fInitializeExpertPrefetch` → 별도 백엔드 전송 → 준비/해제 이벤트 → 연산. 메모리가 부족하면 슬롯 수를 줄이거나 일반 복사로 되돌아갑니다.
7. **언어:** URL `lang`(우선) / 저장소 → 15개 카탈로그 중 하나(기본값 `en-US`) → `fTranslate` → Svelte가 이스케이프한 텍스트. 루트 레이아웃 → HTML `lang`/`dir` → 논리적 내비게이션과 양방향 채팅 텍스트. 선택기 → 저장 → 다시 로드.
8. **갱신된 추론:** 마이크로 배치의 원래 인덱스 → 출력 유무별로 독립된 아레나 → CPU/CUDA/HIP/Vulkan/SYCL 커널 → NextN 행 순서 복원 → EOG까지 수락 → 완료된 슬롯의 추측 디코딩 지표.
9. **내보내기/도구:** 선택한 ID → 영속 저장된 대화/메시지 읽기 → JSONL/ZIP. 도구 HTTP 403 → 비활성 상태 유지 → 패널에서 명시적으로 재시도.
10. **로컬 빌드:** 파일 또는 stdin으로 실행한 `build/build-local.sh` → 옵션 → 로컬/원격 선택 → `/tmp/moe-gguf-server-build.XXXXXXXX/`의 비공개 작업 공간과 로그 → 툴체인, CUDA, Node.js 확인 → 로컬 복사 또는 HTTPS 아카이브 다운로드/검증/`build/`를 포함해 `source/`에 압축 해제 → `cmake/`에서 `-S source/build`를 지정한 CMake → `mgs` → 정적 런처 → 로더, `ldd` 의존성 폐포, NSS 모듈을 `bundle/lib/mgs/`에 배치 → 서버 바이너리와 `DT_RPATH` `$ORIGIN` → `bundle/lib/mgs/`에 표식 → `build/licenses/`와 루트의 `LICENSE`를 `bundle/licenses/`에 배치 → 접두사 잠금 → 완성된 산출물을 최종 위치 옆에 복사 → `lib/mgs/`, `bin/mgs`, `share/doc/mgs/licenses/`를 원자적으로 교체 → `<loader> --list` 검증 → 런처를 통한 `--version` → `fCleanup`이 비공개 작업 공간을 제거. 실행 시: 런처 → `/proc/self/exe` → `MGS_EXECUTABLE` → `bin/`을 기준으로 한 접두사 → `execv(lib/mgs/<loader>, lib/mgs/mgs, arguments)` → 라우터 자식 프로세스도 런처를 거쳐 다시 진입.


11. **네이티브 웹 시작:** `common_params` / CLI → `server_http_context::mInit` → `fCreateHttpsServer`(PEM, 또는 `fPersistentServerIdentity`: `~/.mgs/certificates/`를 로드하거나 생성) → `start` → HTTPS용 `fBindServerPort` → 별도의 HTTP 포트 → 리스너 스레드 두 개 → 로그에 실제 URL 기록 → 중지 시 두 스레드를 join.
12. **MoE 전문가 캐시:** 로드한 뒤의 첫 요청 → `mgs_context::mDecode` → `mgs_expert_cache::mFCreate` → 대상이 되는 레이어와 그룹 → 모든 GPU의 예산 → 모든 GPU의 풀과 식별자 버퍼 → GPU별 대역폭 측정 → 하이브리드 그래프와 업로드 콜백을 사용하는 `mSchedReserve`. 32 토큰 미만의 마이크로 배치: 디바이스에서 라우터 실행 → CPU에서 `mPartition`(선택 결과 복사, LRU 적중, 균형 잡힌 업로드, 식별자) → 풀을 사용하는 디바이스 분기 → 동시에 호스트 가중치를 사용하는 CPU 분기 → `mMeasureCpu` → 디바이스에서 병합 → 라우터 가중치 적용. 큰 마이크로 배치: 프리페치 → `mUploadWeights`(캐시된 전문가는 디바이스 간 복사, 나머지는 호스트에서 업로드) → 연산. `POST /api/expert-cache` → 작업 대기열 → `mgs_set_expert_cache_size` → 다음 디코딩에서 캐시와 그래프를 다시 구성.
13. **컨텍스트 체크포인트:** 프롬프트 배치 채우기 → 4분의 3 이상 찬 마이크로 배치 안의 메시지 시작 지점, 또는 마지막 사용자 메시지 → 배치 분할 → 배치 시작 지점에 체크포인트(메시지 시작, 프롬프트 끝, 또는 직전 체크포인트 이후 `--checkpoint-min-step`) → `mCreateCheckpoint`(이웃과 가장 가까운 것을 제거) → 이후 위치 p에서 갈라지는 요청은 p 이전의 가장 최근 체크포인트를 복원 → 그 뒷부분만 처리.
14. **자동 구성(fit, 배치, 컨텍스트, 스레드):** 인수 파싱 → `fPostprocessCpuParams` → `-t`가 없으면 `fCommonDefaultThreads` → `mLoadModel`이 측정한 초안/MTP 컨텍스트를 디바이스 여유분에 더함 → `common_init_result` → `fCommonFitParams`(`pAutoBatch` = `-b`/`-ub` 없음) → `fCommonParamsFitImpl`: 초기 매개변수로 측정 → 모델이 들어가지 않으면 2단계에서 지정되지 않은 컨텍스트를 `--fit-ctx`로 축소 → dense 전용 레이어를 뒤에서부터 앞으로 → 모든 전문가를 RAM에(전문가 캐시) 두거나 전문가 레이어 전체를 이동 → `fRaiseBatch` → `fRaiseContext`(둘 다 전문가 레이어 전체를 옮기기 전에 실행) → `n_batch`를 매개변수에 다시 복사 → 스레드 로그 → 모델과 타깃 컨텍스트 로드 → 타깃의 `n_ctx`와 `n_batch`로 초안 컨텍스트 생성 → 첫 요청에서 전문가 캐시가 남은 디바이스 메모리를 차지.
15. **모니터링:** `mLoadModel` → `mgs_model_get_weight_breakdown`(한 번) → `mRefreshMonitorSnapshot`(로드한 뒤, 슬롯이 작업하는 동안 최대 초당 1회, 유휴 상태에서는 항상) → `mBuildMonitorSnapshot`(타깃 컨텍스트와 초안 컨텍스트의 `mgs_get_memory_breakdown`과 `mgs_get_expert_memory_breakdown`, 프롬프트 캐시와 체크포인트, GPU별 `ggml_backend_dev_get_props`) → 뮤텍스. `GET /api/monitor` → `vGetMonitor`(라우터: `vGetRouterMonitor` → `fCollectMonitorSnapshots` → 자식 프로세스의 `/api/monitor/memory`) → `fMonitorReport`(`fCpuJson`, `fRamJson`, `fGpusJson` → NVML/sysfs, GPU별(PCI ID 기준) 및 RAM의 세그먼트, "other", 여유 메모리) → `cMonitorStore` → `MonitorScreen`. 상태 표시줄: `StatusBar` → `cMonitorStore.mSubscribeUsage` → `GET /api/monitor/usage` → 2초마다 `fMonitorUsageJson`.
16. **Vulkan 빌드:** `--vulkan` → `vBuildBackend=vulkan` → (설치 프로그램: `libvulkan-dev spirv-headers glslc mesa-vulkan-drivers`) → `glslc` 확인 → `-DGGML_VULKAN=ON -DGGML_CUDA=OFF`를 지정한 CMake(실패하면 필요한 패키지 이름을 알려 줌) → 셰이더와 `mgs` → 로컬 빌드: `lib/mgs/`에 `libvulkan.so.1` → 실행 시: 로더 → `/usr/share/vulkan/icd.d/`의 ICD → `Vulkan0` 디바이스 → CUDA와 마찬가지로 `--fit`, 전문가 캐시, 프리페치.
17. **로컬 셸 MCP 서버:** `--local-shell-mcp-server` → `vShellMcpServer`(기본값은 localhost CORS, 라우터 자식 프로세스에서는 제거) → `mgs_server` → `server_shell_mcp::mStart`(셸 감지) → `POST/GET/DELETE /mcp/shell` 등록과 경고 → `/api/props`의 `shell_mcp_server` → 루트 레이아웃 → `cMcpStore.mSyncShellServer` → `mgs-shell` 항목 → 상태 확인과 `MCPService.connect`(StreamableHTTP, 같은 출처의 API 키) → `initialize`/`tools/list` → 모델의 도구 호출 → 권한 요청 → `tools/call` → `mHandlePost` → `fOriginAllowed` → `mToolsCallResult` → `server_shell_mcp_call_log::mOpen`과 `command` 줄 → `fRunCommand`(`posix_spawn`, `poll`, 그룹 종료. 읽을 때마다 → `pOutput` → `mOutput` → `stdout`/`stderr` 줄) → `mFlushOutput` → `result` 또는 `error` 줄 → 텍스트와 `structuredContent`로 된 JSON 객체(`vExitCode`, `stdout`, `stderr`...) → 도구 메시지 → `ChatMessageToolCallBlock` → `fIsLocalShellTool` → `ChatMessageToolCallBlockLocalShell`(`fParseLocalShellCommand`, `fParseLocalShellResult`) → stdin, stdout, stderr 상자. `expandLocalShellCommands`가 켜져 있으면 기본으로 펼쳐짐.
18. **sudo 비밀번호 요청:** `cMcpStore.mExecuteTool`(`_meta`의 `mgs/callId`, `dShellCalls`) → `mToolsCallResult` → `server_shell_mcp_askpass::mRegister`와 `mEnvironment`(`SUDO_ASKPASS`, `MGS_LOCAL_SHELL_ASKPASS`, `DISPLAY`) → `fRunCommand` → 터미널 없는 sudo → `mgs <prompt>` → `mgs_server` → `fServerShellMcpAskpassMain` → 추상 소켓 → `mServe`(`SO_PEERCRED`, `fParentIsSudo`, 토큰) → 대기 중인 요청과 일시 정지된 마감 시간(`mPaused`) → `DialogLocalShellAskpass`(`mActiveShellCalls`, 500 ms마다 `LocalShellAskpassService.mPending`) → 가운데 정렬된 비밀번호 대화 상자 → `mAnswer`/`mCancel` → `mHandleAskpassPost` → `server_shell_mcp_askpass::mAnswer` → 헬퍼에 응답 → sudo가 읽는 파이프에 비밀번호 기록(또는 stderr에 `mgs: ...`) → 명령이 끝나면 `mUnregister`.
19. **요청 대기열:** `ChatService.mSendMessage`(`mgs_client: "webui"`, `stream`) 또는 API 클라이언트 → `server_routes` → `fEvalMgsCmplSchema`(`vFromWebui`) → `vQueueNotice` → 작업 대기열 → `mProcessSingleTask` → `mGetAvailableSlot` → 빈 슬롯 없음 → `mDeferTask`(`mCountRunning`, `server_queue::mDeferredTasks`) → `server_queue::mDefer` → `server_task_result_queue` → 첫 번째 스트림 결과 → `data: {"mgs_queue": ...}` → `mHandleStreamResponse` → `onQueue` → `ChatStore::mSetQueueStatus` → `fQueueStatusText` → `ChatMessageAssistantProcessingInfo`. 슬롯 해제 → `vCallbackOnRelease` → `mPopDeferredTask`(첫 번째 지연 작업을 맨 앞으로) → `mProcessSingleTask` → `mLaunchSlotWithTask` → 0이 담긴 알림 → `mNotifyQueuePositions` → 첫 토큰 → `onQueue(null)`.
20. **DeepSeek-V4.1 로딩과 Engram:** `mgs_model_loader` → `fMgsDeepseek41AdaptMetadata`(DwarfStar 키 → 표준 키) → `mgs_model_deepseek41` 하이퍼파라미터와 텐서(Engram 테이블에는 `cTensorExternal`, 프리페치는 그 앞에서 멈춤) → 레이어 역할 → `mgs_kv_cache_dsv41` → ubatch마다 `llm_graph_input_dsv41_engram::mSetInput`(토큰 기록에 대한 n-gram 해시 → `pread`를 통한 `mEngramReadRows` → F32 입력) → 레이어 1과 14 앞에서 `mBuildEngram` → 인덱스 키와 후보 풀을 사용해 공유 압축 KV에 대한 어텐션 → `ggml_swiglu_clamp`를 사용하는 MoE.
21. **원격 미디어와 디코딩:** 요청 파트(`image_url`, `input_audio`, `input_video`) → `fHandleMedia` → data URI, `--media-path` 파일 또는 `media_download_slot`(동시 8개) → `fCommonRemoteGetContent`(`--media-allow-private-urls`가 없으면 `vPublicHostsOnly`) → `fRemoteResolveHost` → 모든 주소에 대해 `fRemoteIpv4IsPublic`/`fRemoteIpv6IsPublic` → `set_hostname_addr_map` → 30초, 10 MiB 이내의 GET → 리디렉션마다 다시 검사(최대 5회) → `fProcessMtmdPrompt`(요청당 디코딩 2 GiB) → 디코딩 전 `stbi_info_from_memory`, 할당 전 오디오 길이, 비디오 프레임 크기와 예산 → mtmd 청크. 거부된 주소는 해당 옵션 이름을 알려 주는 400 응답이 됩니다.

<a id="routes"></a>
## 진입점 및 라우트

| 라우트/URL/명령 | 핸들러 | 파일 |
| --- | --- | --- |
| `mgs` | `main → mgs_server` | `backend/tools/server/main.cpp` |
| `bash build/build-local.sh` | `fMain (local build)` | `build/build-local.sh` |
| `curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh` &#124; `bash -s -- [options]` | `fMain (local build) → fDownloadSources` | `build/build-local.sh` |
| `<prefix>/bin/mgs` | `main (launcher) → main → mgs_server` | `build/mgs-launcher.c` |
| `/` | 내장 UI 에셋 | `backend/tools/server/server-http.cpp` |
| `/api/config.js` | 런타임 API 접두사와 저장된 인터페이스 언어 | `backend/tools/server/server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp` |
| `POST /api/models` | `vModelsRoutes->vPostRouterModels` | `backend/tools/server/server.cpp` |
| `POST /api/models/load` | `vModelsRoutes->vPostRouterModelsLoad` | `backend/tools/server/server.cpp` |
| `POST /api/models/unload` | `vModelsRoutes->vPostRouterModelsUnload` | `backend/tools/server/server.cpp` |
| `GET /api/models/sse` | `vModelsRoutes->vGetRouterModelsSse` | `backend/tools/server/server.cpp` |
| `DELETE /api/models` | `vModelsRoutes->vDelRouterModels` | `backend/tools/server/server.cpp` |
| `GET /api/health` | `routes.vGetHealth` | `backend/tools/server/server.cpp` |
| `GET /api/v1/health` | `routes.vGetHealth` | `backend/tools/server/server.cpp` |
| `GET /api/metrics` | `routes.vGetMetrics` | `backend/tools/server/server.cpp` |
| `GET /api/props` | `routes.vGetProps` | `backend/tools/server/server.cpp` |
| `POST /api/props` | `routes.vPostProps` | `backend/tools/server/server.cpp` |
| `GET /api/models` | `routes.vGetModels`(공개, `reload`에는 API 키 필요) | `backend/tools/server/server.cpp` |
| `GET /api/v1/models` | `routes.vGetModels`(공개, `reload`에는 API 키 필요) | `backend/tools/server/server.cpp` |
| `POST /api/completions` | `routes.vPostCompletions` | `backend/tools/server/server.cpp` |
| `POST /api/v1/completions` | `routes.vPostCompletionsOai` | `backend/tools/server/server.cpp` |
| `POST /api/chat/completions` | `routes.vPostChatCompletions` | `backend/tools/server/server.cpp` |
| `POST /api/v1/chat/completions` | `routes.vPostChatCompletions` | `backend/tools/server/server.cpp` |
| `POST /api/v1/chat/completions/control` | `routes.vPostControl` | `backend/tools/server/server.cpp` |
| `POST /api/v1/responses` | `routes.vPostResponsesOai` | `backend/tools/server/server.cpp` |
| `POST /api/responses` | `routes.vPostResponsesOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/audio/transcriptions` | `routes.vPostTranscriptionsOai` | `backend/tools/server/server.cpp` |
| `POST /api/audio/transcriptions` | `routes.vPostTranscriptionsOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/messages` | `routes.vPostAnthropicMessages` | `backend/tools/server/server.cpp` |
| `POST /api/infill` | `routes.vPostInfill` | `backend/tools/server/server.cpp` |
| `POST /api/embeddings` | `routes.vPostEmbeddings` | `backend/tools/server/server.cpp` |
| `POST /api/v1/embeddings` | `routes.vPostEmbeddingsOai` | `backend/tools/server/server.cpp` |
| `POST /api/rerank` | `routes.vPostRerank` | `backend/tools/server/server.cpp` |
| `POST /api/v1/rerank` | `routes.vPostRerank` | `backend/tools/server/server.cpp` |
| `POST /api/tokenize` | `routes.vPostTokenize` | `backend/tools/server/server.cpp` |
| `POST /api/detokenize` | `routes.vPostDetokenize` | `backend/tools/server/server.cpp` |
| `POST /api/apply-template` | `routes.vPostApplyTemplate` | `backend/tools/server/server.cpp` |
| `POST /api/chat/completions/input_tokens` | `routes.vPostChatCompletionsTok` | `backend/tools/server/server.cpp` |
| `POST /api/v1/chat/completions/input_tokens` | `routes.vPostChatCompletionsTok` | `backend/tools/server/server.cpp` |
| `POST /api/responses/input_tokens` | `routes.vPostResponsesTokOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/responses/input_tokens` | `routes.vPostResponsesTokOai` | `backend/tools/server/server.cpp` |
| `POST /api/v1/messages/count_tokens` | `routes.vPostAnthropicCountTokens` | `backend/tools/server/server.cpp` |
| `GET /api/lora-adapters` | `routes.vGetLoraAdapters` | `backend/tools/server/server.cpp` |
| `POST /api/lora-adapters` | `routes.vPostLoraAdapters` | `backend/tools/server/server.cpp` |
| `GET /api/expert-cache` | `routes.vGetExpertCache` | `backend/tools/server/server.cpp` |
| `POST /api/expert-cache` | `routes.vPostExpertCache` | `backend/tools/server/server.cpp` |
| `GET /api/monitor` | `routes.vGetMonitor`(라우터: `vModelsRoutes->vGetRouterMonitor`) | `backend/tools/server/server.cpp` |
| `GET /api/monitor/memory` | `routes.vGetMonitorMemory`(라우터: `vModelsRoutes->vGetRouterMonitorMemory`) | `backend/tools/server/server.cpp` |
| `GET /api/monitor/usage` | `routes.vGetMonitorUsage`(로딩 중에도 동작) | `backend/tools/server/server.cpp` |
| `GET /api/ui/settings` | `fUiSettingsMakeGetHandler`(로딩 중에도 동작) | `backend/tools/server/server.cpp` |
| `POST /api/ui/settings` | `fUiSettingsMakePostHandler`(로딩 중에도 동작) | `backend/tools/server/server.cpp` |
| `#/monitor` (UI) | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |
| `GET /api/slots` | `routes.vGetSlots` | `backend/tools/server/server.cpp` |
| `POST /api/slots/:id_slot` | `routes.vPostSlots` | `backend/tools/server/server.cpp` |
| `GET /api/v1/stream` | `vStreamGetH` | `backend/tools/server/server.cpp` |
| `POST /api/v1/streams/lookup` | `vStreamsLookupH` | `backend/tools/server/server.cpp` |
| `DELETE /api/v1/stream` | `vStreamDeleteH` | `backend/tools/server/server.cpp` |
| `GET /api/cors-proxy` | `vProxyHandlerGet`(탐색 불가, 비활성 응답) | `backend/tools/server/server.cpp` |
| `POST /api/cors-proxy` | `vProxyHandlerPost`(탐색 불가, 비활성 응답) | `backend/tools/server/server.cpp` |
| `GET /api/tools` | `tools.vHandleGet` | `backend/tools/server/server.cpp` |
| `POST /api/tools` | `tools.vHandlePost`(`application/json`만 허용) | `backend/tools/server/server.cpp` |
| `mgs --local-shell-mcp-server` | `vShellMcpServer → server_shell_mcp::mStart` | `backend/common/arg.cpp; backend/tools/server/server.cpp` |
| `POST /api/mcp/shell` | `vShellMcp.mHandlePost`(로딩 중에도 동작) | `backend/tools/server/server.cpp` |
| `GET /api/mcp/shell`, `DELETE /api/mcp/shell` | `vShellMcp.mHandleUnsupported` (405) | `backend/tools/server/server.cpp` |
| `GET /api/mcp/shell/askpass?call=<id>` | `vShellMcp.mHandleAskpassGet`(로딩 중에도 동작) | `backend/tools/server/server.cpp` |
| `POST /api/mcp/shell/askpass` | `vShellMcp.mHandleAskpassPost`(로딩 중에도 동작) | `backend/tools/server/server.cpp` |
| sudo가 `SUDO_ASKPASS`로 실행하는 `mgs <prompt>` | `fServerShellMcpAskpassMain` | `backend/tools/server/server.cpp; backend/tools/server/server-shell-mcp.cpp` |
| `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` | `mRegisterGcpCompat` | `backend/tools/server/server-http.cpp` |
| `HTTP 11080+ (executable)` | 선택된 HTTPS로 308 리디렉션 | `backend/tools/server/server-http.cpp` |
| `HTTPS 11443+ (executable)` | TLS를 통한 네이티브 UI와 API | `backend/tools/server/server-http.cpp; server-tls.cpp` |
| `HTTP 11080+ (Apache)` | 선택된 HTTPS로 영구 리디렉션 | `deploy/apache2.conf.in` |
| `HTTPS 11443+ (Apache)` | ProxyPass → 127.0.0.1:8080 | `deploy/apache2.conf.in` |

<a id="impact"></a>
## 영향 분석

- **답변 언어:** 지시문은 사용자 지정 JSON 다음에 적용해야 `messages`를 바꿔도 켜 둔 설정을 우회할 수 없습니다. 로캘 지시문을 대화에 저장하지 말고, 새 요청마다 현재 플래그를 읽으세요. KV 접두사를 보존하려면 completion과 미리 채우기가 모두 같은 헬퍼를 사용해야 합니다. 스트림 재연결은 이미 실행 중인 생성을 재사용하므로 그 언어 지시문을 바꿀 수 없습니다. 15개 카탈로그의 `responseLanguage.*` 항목을 모두 맞춰 두세요. 아르헨티나/스페인 스페인어, 영국/미국 영어, 브라질/유럽 포르투갈어에는 지역별 지침이 명시되어 있습니다. 지시를 따르는지는 여전히 모델에 달려 있으며, 외부 API 호출자에게는 영향이 없습니다.

- **원격 설치:** URL, GitHub 아카이브 구조, 진입 스크립트/헬퍼 프로토콜을 바꾸면 `curl | bash`에 영향을 줍니다. 도움말 옵션을 서로 맞추고 저장소 출처는 고정해 두세요. 표식과 잠금이 관리 대상 트리를 보호하므로, 동기화 후에도 남아야 하는 로컬 수정은 그곳에 두지 마세요.
- **네이티브 TLS:** 직접 접속하는 클라이언트는 HTTPS를 사용하고 인증서를 신뢰하거나 알려진 PEM 쌍을 제공해야 합니다. Apache 백엔드와 라우터 자식 프로세스에는 `--http`를 명시적으로 지정하며, 자식 프로세스는 공개 TLS 변수를 상속하지 않습니다. 명시적으로 지정한 메인 포트는 절대 몰래 바뀌지 않습니다. OpenSSL 없이 빌드하면 HTTP를 명시적으로 지정해야 합니다.
- **웹 포트:** 리디렉션, 공개 주소, 방화벽, 브라우저 저장소의 출처에 영향을 줍니다. `ports.json`은 두 리스너가 모두 실행된 Apache 프로세스의 것이 된 뒤에만 게시됩니다. 배타적 잠금이 런처의 동시 실행을 막습니다. HTTP를 찾기 전에 HTTPS를 먼저 예약하므로, 범위가 겹치더라도 두 서비스가 같은 포트를 쓰지 않습니다. 포트가 바닥나면 명시적 오류가 발생합니다.
- **접두사와 미들웨어:** 모든 HTTP 클라이언트, 인증, 라우터, 스트림, 프런트엔드, Swagger에 영향을 줍니다. `server-http`, `server-models`, `api-url`과 그 문서를 함께 변경하세요.
- **요청 강화:** 크로스 사이트 검사는 `Host` 또는 `X-Forwarded-Host`가 공개 주소를 가리킨다는 데 기대고 있습니다. `X-Forwarded-Host`를 보내지 않고 `Host`를 다시 쓰는 리버스 프록시를 쓰면, 도구나 MCP 서버가 켜져 있을 때 UI의 쓰기가 403으로 실패합니다(`deploy/apache2.conf.in`은 `ProxyPreserveHost On`을 유지합니다). `cHttpRequestReadTimeoutSec`을 늘리거나 대기열 한도를 없애면 느린 클라이언트로 인한 자원 고갈이 다시 생기고, `cHttpMaxRequestJsonDepth`를 `cMaxSchemaDepth`의 약 두 배보다 낮추면 변환기가 받아들이는 JSON 스키마가 거부됩니다.
- **비밀 옵션:** 자격 증명을 담는 옵션은 `sSecretOptionEnvs`에 등록해야 합니다. 그렇지 않으면 그 값이 공개 라우터 목록과 프로세스 생성 로그에 나타납니다.
- **원격 미디어 정책:** `common_remote_params`의 기본값은 Hugging Face와 Docker 매니페스트에 대해 기존 동작을 유지하며, `vPublicHostsOnly`를 설정하는 것은 미디어 경로뿐입니다. 리디렉션을 자동으로 따라가거나 검사 뒤에 이름으로 다시 연결하면 DNS 리바인딩과 리디렉션 기반 SSRF가 다시 가능해집니다. 디코딩 제한은 `mtmd-helper.cpp`와 `server-common.cpp`에 있는 범용 상수이며, 이를 낮추면 정상적인 대용량 미디어가 거부됩니다.
- **UI 도구 식별:** 승인, 권한, 실행은 `mResolveEnabledEntry`가 반환한 항목과 그 서버 ID를 사용해야 합니다. 다시 이름만으로 해석하면(UI에서는 첫 번째 일치, `MCPStore`에서는 마지막에 기록한 것) 한 서버에 대한 승인으로 다른 서버의 도구, 로컬 셸까지 실행될 수 있습니다.
- **completion 스키마:** 추론 검증, 기본값, OpenAPI 문서에 영향을 줍니다. 확장할 때는 프로토콜 필드 이름을 유지해야 합니다.
- **카탈로그/선택기:** 모든 화면과 모듈을 가져올 때 초기화되는 레이블에 영향을 줍니다. 15개 카탈로그 전체에서 키, 언어 이름, 데이터 매개변수를 맞춰 두세요. 번역에서 개수 레이블을 쓰는 경우에는 영어의 복수 접미사 매개변수를 생략할 수 있습니다. HTML 태그와 `{{USER}}`, `{{ASSISTANT}}` 같은 제목 템플릿 토큰은 유지하세요. 새 로캘은 `aLanguages`와 `dCatalogs` 양쪽에 등록하세요. 히브리어는 내비게이션 방향에도 영향을 주며, 코드는 LTR로 유지해야 합니다. 프로토콜 식별자나 사용자 데이터는 번역하지 마세요.
- **등록/프리페치:** RAM/VRAM 사용량, GPU 이벤트, 매핑/버퍼 수명에 영향을 줍니다. 등록된 페이지나 슬롯은 사용 중인 동안 절대 해제하지 마세요.
- **MTP:** 가중치를 할당하기 전의 디코더 선택에 영향을 줍니다. 명시적 옵션의 우선순위를 유지하고 모든 GGUF 샤드를 검사하세요. MTP 가중치는 호스트 메모리로 보내는 오버라이드를 무시하고, `--fit`이 트렁크를 배치하기 전이자 전문가 캐시의 크기가 정해지기 전에 디바이스 메모리를 차지합니다. 따라서 `mLoadTensors`의 MTP 레이어 할당을 바꾸면 MTP가 있는 모든 모델의 VRAM 사용량이 달라집니다.
- **CMake/설치 프로그램:** 모든 빌드 플랫폼에 영향을 줍니다. 이전 구조의 캐시를 재사용하거나 로컬 UI 빌드 실패를 숨기지 마세요.
- **버전:** `MGS_VERSION`은 공유 라이브러리 파일의 이름이 되고(`libmgs-engine.so.0.1.0`, soname은 `.so.0` 그대로), `/api/props`, `system_fingerprint`, 셸 MCP의 `serverInfo.version`, 다운로드의 `User-Agent`, UI의 `build.json`에 전달됩니다. `MAJOR.MINOR.PATCH` 형식을 유지하세요. UI 빌드는 `[A-Za-z0-9._-]` 밖의 문자를 거부합니다.
- **Vulkan:** 번들은 자신이 로드하는 시스템 드라이버와 C/C++ 런타임을 공유하므로, 더 오래되었거나 더 새로운 Debian 릴리스에서 빌드한 번들은 ICD를 로드하지 못할 수 있습니다. 고정(pinned) 메모리를 이용한 전문가 업로드는 CUDA에만 있으므로, `--load-mode` 기본값이나 호스트 버퍼 유형을 바꾸면 Vulkan의 전송 속도가 달라집니다. `render` 그룹이 없는 서비스 계정은 GPU를 열 수 없습니다. 디바이스 버퍼는 device-local 메모리에 머물러야 하므로, MoE 서비스에서는 `GGML_VK_ALLOW_SYSMEM_FALLBACK`이나 `GGML_VK_PREFER_HOST_MEMORY`를 절대 켜지 마세요. 캐시와 비슷한 새 디바이스 할당에는 캐시 버퍼 유형을 써야 하며, `ggml-vulkan`의 새 비동기 경로는 GPU의 모든 디바이스 버퍼 유형을 받아들여야 합니다.
- **로컬 번들:** 새로 링크된 라이브러리는 `ldd`를 통해 자동으로 포함됩니다. `OUTPUT_NAME`, 라우터의 재실행 경로(`fGetServerExecPath`, `MGS_EXECUTABLE`), 로더의 명시적 실행 모드, 라이브러리 검색 순서를 바꾸면 `build/build-local.sh`와 런처에 영향을 줍니다. `lib/mgs/mgs`를 직접 실행하면 번들된 로더를 건너뛰게 되며, 이는 지원되지 않습니다. 드라이버 라이브러리는 계속 제외해야 합니다.
- **MoE 전문가 캐시:** 디바이스 메모리(풀은 컨텍스트를 할당하고 남은 메모리를 차지하며, `--moe-cache-reserve`는 다른 프로세스와 지연 할당되는 버퍼를 위한 몫), 스케줄러의 분할 동기화, CPU의 `mul_mat_id`/`add_id` 커널(음수 식별자), 모든 MoE 아키텍처의 그래프(`mBuildMoeFfn`)에 영향을 줍니다. 전문가 곱셈용 디바이스 커널은 한 토큰 안에서 서로 다른 슬롯 식별자를 받아들여야 합니다. `--fit` 배치를 바꾸면 필요한 RAM이 달라집니다. 모든 전문가를 RAM에 두는 방식은 `MemAvailable`이 부족하면 전문가 레이어 전체를 옮기는 방식으로 되돌아갑니다. GPU가 여러 개면 각 GPU가 자신의 레이어를 위한 풀을 따로 가지며, 레이어 분할(`--fit`, `--tensor-split`)에 따라 각 GPU가 어떤 전문가를 캐시할지 정해집니다.
- **배치 크기와 스레드:** `fRaiseBatch`의 `cBatchShare`는 디바이스 메모리를 내주고 프롬프트 속도를 얻는 균형점입니다. 마이크로 배치는 컨텍스트보다 먼저 정해지므로, 마이크로 배치가 커지면 기본 컨텍스트도 짧아집니다(16 GB에서 Qwen3.6-35B-A3B Q8_0: 512일 때 122880 토큰, 4096일 때 89856 토큰). 서버는 타깃의 `mgs_n_batch`/`mgs_n_ubatch`에 따라 프롬프트를 자르고 컨텍스트 체크포인트를 배치하므로, 그 위치는 선택된 크기를 따릅니다. 초안 컨텍스트나 MTP 컨텍스트가 자신의 `n_batch`보다 큰 배치를 받게 만드는 변경은 그 배치 버퍼를 넘칩니다. `fCommonDefaultThreads`는 전문가가 RAM에 있는 모든 MoE 레이어의 CPU 몫을 바꿉니다. 상한을 바꾸기 전에 가상 머신과 베어 메탈 양쪽에서 `--spec-type none`으로 생성 속도를 측정하세요.
- **`--fit`의 컨텍스트 크기:** `fRaiseContext`의 `cContextShare`는 컨텍스트와 전문가 사이의 균형점입니다. 여기서 허용하는 KV 캐시의 모든 바이트는 전문가 캐시 풀이나 디바이스에 배치된 전문가 레이어 전체에서 빠져나가므로, 몫을 키우면 생성 속도가 떨어지고 줄이면 기본 컨텍스트가 짧아집니다. 초안 컨텍스트는 타깃 컨텍스트를 따르므로, `mLoadModel`이 그 크기를 정하는 방식을 바꾸면 MTP나 초안 모델이 있는 모든 모델에서 전문가 캐시에 남는 디바이스 메모리가 달라집니다.
- **컨텍스트 체크포인트:** 간격과 배치는 호스트 메모리(슬롯당 최대 `--ctx-checkpoints`개의 상태)와 미리 채우기의 배치 처리에 영향을 줍니다. 분할이 하나 늘 때마다 오프로드된 전문가를 한 번 더 거쳐야 합니다.
- **제거된 호환성:** 클라이언트, 라우터 프리셋, 서비스 오버라이드, 빌드 스크립트는 현재 이름을 사용해야 합니다. 제거된 별칭은 이제 옵션으로 쓰면 알 수 없는 옵션으로 실패하고, 환경 변수로 쓰면 무시되며, 라우트로 쓰면 404를 반환합니다. 이름을 다시 도입하려면 `common/arg.cpp`, `server.cpp` 또는 CMake에 추가하고 CODE와 MANUAL의 각 언어판에 문서화해야 합니다. README는 개요나 설치 명령이 바뀔 때만 갱신하세요.
- **모니터링:** 스냅샷은 컨텍스트를 건드릴 수 있는 유일한 스레드인 메인 루프에서 구성됩니다. `mgs_get_memory_breakdown` 밖에서 할당된 버퍼(새 캐시나 풀)는 내역 호출이 이를 보고할 때까지 "other"로 표시됩니다. 모니터링 요청을 절대 작업 대기열로 보내지 마세요. 유휴 타이머가 초기화되고 절전 중인 모델이 깨어납니다. 라우터 집계는 자식 프로세스의 계약(`/api` 접두사, API 키 없음)에 기대고 있습니다. 새로 추가하는 고정 UI 요소나 하단에 고정되는 UI 요소에는 `--status-bar-height`를 더해야 합니다.
- **셸 MCP 서버:** API를 사용할 수 있는(또는 UI를 열 수 있는) 사람은 누구나 서비스 사용자로 명령을 실행할 수 있으므로, 신뢰할 수 있는 환경에서만 사용하고 API 키와 함께 써야 합니다. 라우트, `shell_mcp_server` 속성, `mgs-shell` 식별자를 바꾸면 UI 동기화와 저장된 설정에 영향을 줍니다. Origin이나 Content-Type 검사를 바꾸면 크로스 사이트 명령 실행이 다시 가능해집니다. `fRunCommand`를 바꾸면 프로세스 그룹 정리, 시간 제한, 출력 한도에 영향을 줍니다. `mToolsCallResult`의 JSON 필드를 바꾸면 모델이 읽는 내용, 결과 전체를 stderr 텍스트로 표시하게 되는 `fParseLocalShellResult`, 호출 로그의 `result` 줄에 영향을 줍니다. `server_shell_mcp_call_log`의 `cCallLogFolder`, 파일 이름, 0700/0600 모드를 바꾸면 호출 로그의 위치가 바뀌거나 명령 출력이 다른 사용자에게 노출됩니다.
- **웹 UI 설정:** 문서는 모든 브라우저가 공유하며 마지막에 저장한 것이 이깁니다. 저장할 때는 항상 모든 섹션을 보내므로, `cUiSettingsStore` 밖에 설정을 보관하는 스토어는 다른 브라우저에서 그 설정을 잃게 됩니다. 문서가 로드되기 전에 저장하면 파일이 불완전한 상태로 덮어써지며, `mSet`이 이를 막습니다. 섹션이나 구성 키의 이름을 바꾸면 이전 값은 읽히지 않습니다(마이그레이션 없음). 파일에는 MCP 헤더가 들어갈 수 있어 모드가 0600이며, API 키를 가진 사람은 누구나 API를 통해 읽을 수 있습니다.
- **sudo 비밀번호 요청:** 비밀번호는 브라우저, API(HTTPS와 API 키), 추상 소켓을 거칩니다. `fParentIsSudo`, `SO_PEERCRED` 검사, 토큰 조회를 바꾸면 명령이 가짜 요청으로 비밀번호를 읽을 수 있게 됩니다. `fServerShellMcpAskpassMain`이나 `mgs_server` 시작 부분에서의 그 위치를 바꾸면 헬퍼가 망가지거나, 명령 안에서 일반적으로 실행한 `mgs`가 헬퍼처럼 동작하게 됩니다. `cAskpassDisplay`나 `DISPLAY`를 설정하는 조건을 바꾸면 sudo가 헬퍼를 쓰지 않게 됩니다. 명령을 자체 세션 없이(`POSIX_SPAWN_SETSID`) 시작하면 sudo가 대신 `mgs`를 시작한 터미널에서 비밀번호를 묻습니다. 루트 레이아웃에서 `DialogLocalShellAskpass`를 마운트 해제하면 요청이 만료될 때까지 응답을 받지 못합니다. `mgs/callId`를 바꾸면 `cMcpStore.mExecuteTool`과 UI의 폴링에 영향을 줍니다. `/mcp/shell/askpass`의 요청 본문을 로그에 남기면 비밀번호가 로그에 기록됩니다.
- **요청 대기열:** 기본값인 슬롯 하나는 모든 요청에 컨텍스트 전체를 줍니다. `-np`의 기본값을 올리면(또는 자동 모드를 되살리면) 컨텍스트가 슬롯들 사이에 나뉘어 긴 답변이 짧아집니다. `mgs_queue` 이벤트는 `mgs_client: "webui"`로 표시된 요청에만 전달됩니다. 다른 클라이언트에 보내거나 Anthropic Messages 또는 OpenAI Responses 스트림에 넣으면 그 파서가 망가집니다. 모든 스트리밍 핸들러는 `server_task_result_queue`를 첫 번째 결과나 그 뒤의 결과로 받아들여야 하며, 새 지연 경로는 모두 `mDeferTask`를 거쳐야 대기 중인 요청이 자기 위치를 유지합니다. 다시 대기열에 들어간 작업은 업스트림과 마찬가지로 지연 대기열의 맨 뒤로 갑니다.
- **제거된 아키텍처:** Gemma 4와 V4.1보다 오래된 모든 DeepSeek 아키텍처는 제거되었습니다. 하나를 다시 도입하려면 그 아키텍처 항목, 테이블, 그래프, 어휘, 채팅 관련 부분을 복원하고 `RulesForThisProjectOnly.md`에 다시 추가해야 합니다. `deepseek2`로 변환된 다른 모델(Kimi-K2.x, GLM-4.7-Flash 등)의 GGUF는 더 이상 로드되지 않습니다.
- **공유 DSV4 구성 요소:** `ggml_dsv4_hc_pre`, `ggml_dsv4_hc_pre_gated`, `ggml_dsv4_hc_comb`, `ggml_dsv4_hc_post`, `ggml_lightning_indexer`, `vDsv4HcMult`, `aDsv4CompressRatios`, 융합된 hc cparams는 Kimi-K3, GLM-5.3-Flash, Qwen3.8, DeepSeek-V4.1이 함께 사용하므로, 여기를 바꾸면 네 모델 모두에 영향을 줍니다. Vulkan이나 Metal 커널이 없으므로 그 백엔드에서는 CPU에서 실행됩니다.
- **하이브리드 인덱스 메모리:** qwen4exp와 glm5-next가 공유합니다. KV 셀은 토큰 ID를 저장하고 `seq_pos`는 (위치, 셀) 쌍을 담으며, n-gram 임베딩과 상태 복원이 여기에 기대고 있습니다.
- **외부 텐서:** `cTensorExternal` 텐서는 절대 할당되지 않으며, 프리페치는 그 첫 번째 텐서에서 멈춥니다. 텐서 데이터를 순회하는 코드(양자화, 저장, 메모리 보고)는 이를 건너뛰어야 합니다. Engram 행을 읽으려면 실행하는 동안 내내 모델 파일에 POSIX `pread`를 할 수 있어야 합니다.

프로젝트 고유 규칙에 따라 테스트를 작성하거나 실행하는 것은 금지되어 있습니다. 검토는 정적으로만 이루어지며, 런타임, GPU, 원격 설치, 모바일 화면 넘침의 정확성을 보증하지 않습니다. `tests/`는 실행 가능한 테스트 없이 예약된 상태로 남아 있습니다.


<a id="extensions"></a>
## 확장 지점

- **엔드포인트:** 핸들러를 구현하고 `server.cpp`에서 `get`, `post`, `del`로 등록합니다. 등록하면 접두사가 붙고 목록에 포함됩니다. `fRouteDescription`과 `fOperation`에 매개변수, 응답, 예시를 추가하세요. 라우터 작업이라면 인증과 전달 방식을 검토하세요.
- **UI 텍스트:** 15개 JSON 카탈로그 모두에 안정적인 키를 추가하고, 값에는 `{p0}`, `{p1}` 등을 써서 `fTranslate`로 해석합니다. 도움말 HTML은 신뢰할 수 있는 내용이어야 하며, 사용자 콘텐츠를 정제하지 않은 HTML로 삽입하지 마세요. 로캘을 추가하려면 `i18n.ts`에서 그 전체 JSON을 가져와 등록하고, `aLanguages`를 알파벳순으로 유지하며, 모든 카탈로그에 `language.<locale>` 레이블을 추가하고, `README`, `MANUAL`, `CODE` 번역을 추가한 뒤 모든 언어판의 언어 막대에 그 링크를 넣으세요.
- **CLI 옵션:** `common/arg.cpp`와 그 매개변수 구조체를 확장하고, 공개 계약을 유지하며, 동작을 문서화하고, 영향을 받는 심볼/모듈/흐름 행을 갱신합니다. 자격 증명을 담는 옵션은 `sSecretOptionEnvs`(`server-models.cpp`)에도 추가합니다.
- **설치:** `fMain`과 `fShowHelp`를 확장하고, 시스템을 변경하기 전에 검증하며, 모든 실패를 전파하고, CODE와 MANUAL의 각 언어판을 갱신합니다. 설치 명령이 바뀌면 README의 각 언어판도 갱신하세요.
- **로컬 번들:** 다른 시스템 라이브러리를 `lib/mgs/`에서 제외하려면 `fVerifyBundle`도 따르는 `fIsDriverLibrary`를 확장합니다. 새 보조 파일은 임시 `bundle/` 디렉터리 아래에서 조립하고, 완성된 산출물을 접두사 안의 최종 위치 옆에 복사한 다음 `fReplaceDirectory`로 디렉터리를 교체하세요. 새로 설치되는 항목은 `fMain`의 소유권 검사에 추가하세요.
- **전문가별 텐서:** `mgs_expert_role`에 역할을 추가하고, `fLayerExpertTensors`와 `mBuildMoeFfn`의 `vTensors` 묶음에 매핑한 뒤, 풀 텐서를 디바이스 분기에 넘깁니다. 그러면 풀, 업로드, 프리페치 재사용이 그 텐서도 처리합니다.
- **빌드 백엔드:** 두 스크립트(`build-local.sh`와 `install-common.sh`, 그리고 Debian 진입 스크립트의 전달 목록)에 옵션을 추가하고, 그 패키지를 `fInstallSystemDependencies`에, `-DGGML_*` 플래그를 구성 인수에, 메시지를 요약에 추가합니다. 바이너리가 직접 링크하는 드라이버 라이브러리는 `fIsDriverLibrary`에 추가합니다.
- **새 버전:** 프로젝트를 변경할 때마다 `backend/cmake/build-info.cmake`에 있는 `MGS_VERSION`의 패치 번호를 올립니다(0.1.1, 0.1.2, ...). 바이너리, 라이브러리, 인터페이스가 모두 여기서 버전을 가져오므로 다른 곳은 수정할 필요가 없습니다.
- **메모리 세그먼트:** `mBuildMonitorSnapshot`(`fAdd`), `aMonitorSegments`, `MonitorSegmentKey`에 키를 추가한 다음, `aMonitorSegmentGroups`의 그룹 하나에 배치하고(범주형 색상 여덟 개가 모두 사용 중이므로 아홉 번째 색상을 추가하지 말고 기존 그룹에 합치세요), MANUAL의 세그먼트 표와 `fRouteDescription`에 문서화합니다.
- **셸 도구:** `server_shell_mcp::mStart`의 `lCandidates`에 후보(도구 이름, 제목, 실행 파일 이름, 명령 앞에 놓이는 인수)를 추가합니다. 감지, `tools/list`, `tools/call`, 로그는 별도의 변경 없이 이를 반영합니다. 채팅에서 stdin, stdout, stderr 상자로 표시되도록 `aShellMcpToolNames`에 도구 이름을 추가하고, `fRouteDescription`의 라우트 설명, `--local-shell-mcp-server` 도움말, `expandLocalShellCommands`의 도움말 텍스트, MANUAL을 갱신하세요.
- **인터페이스가 보관하는 설정:** 단순한 설정은 `settings-registry.ts`에 넣으며 `config` 섹션과 함께 저장됩니다. 새 섹션에는 `UiSettingsDocument`의 필드, 그 섹션을 소유한 스토어에서 `cUiSettingsStore.mRegister`로 등록한 로더, 변경할 때마다 호출하는 `cUiSettingsStore.mSet`이 필요합니다. `server-api-doc.cpp`의 `/ui/settings` 설명에 문서화하세요. 설정을 `localStorage`에 보관하지 마세요.
- **다른 클라이언트를 위한 대기열 알림:** `mgs_queue` 이벤트를 받으려는 클라이언트는 `"stream": true`와 함께 `"mgs_client": "webui"`를 보냅니다. 다른 출처 값을 쓰려면 `task_params`의 필드, `fMakeMgsCmplSchema`의 `mgs_client` 필드에 있는 그 처리기, `mCountRunning`, `mDeferTask`, `mNotifyQueuePositions`, `server_task_result_queue`의 전용 카운터, 그리고 `fQueueStatusText`와 15개 카탈로그의 번역된 텍스트가 필요합니다.

구현 참고 자료: [Svelte 컴파일러](https://svelte.dev/docs/svelte/svelte-compiler), [OpenAPI 서버](https://swagger.io/docs/specification/v3_0/api-host-and-base-path/), [Apache mod_proxy](https://httpd.apache.org/docs/2.4/mod/mod_proxy.html). Swagger UI 5.30.2는 라이선스 고지와 함께 `backend/vendor/swagger-ui/`에 벤더링되어 있습니다.



[Apache Listen과 재시작 동작](https://httpd.apache.org/docs/2.4/bind.html)

[GitHub 소스 아카이브](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives)

<a id="upstream-review"></a>
## 업스트림 검토 — 2026-10-02

8월 4일분을 포함해 업스트림 기록 항목 1,103개를 선별 검토했습니다. 그중 1,081개는 8월 5일부터 10월 2일까지의 항목이며, 마지막은 **a8c9a4e7ccba**(08:18:51 UTC)입니다. Git 없이 소스 아카이브/API를 사용했습니다. 비교 기준 스냅샷 **5788b510a1e3**은 8월 4일 09:12 UTC의 것으로, 참고용일 뿐 로컬 포크의 SHA로 입증된 것은 아닙니다. 이 작업은 커밋 38개 또는 그중 적용 가능한 부분을 선택적으로 통합한 것이며, 업스트림 헤드로의 전체 업그레이드가 아닙니다. 기존의 공개 mgs/GGML 헤더, 로컬 호스트 등록, 전문가 프리페치, API 접두사, 번역, 배포 구조는 그대로 유지됩니다.

로컬 커밋/파일 목록: `_/upstream/upstream-review-2026-10-02.json`(게시되는 소스에서는 제외).

| 영역 | 적용된 변경 | 업스트림 커밋 |
| --- | --- | --- |
| 스케줄러 | 입력이 없는 분할의 동기화, 인위적인 분할 감소, 빈 전문가 ID 처리. | #26040, #28387, #28739 |
| CUDA 그래프 | 비동기 MoE 경로에서 그래프 유지, 출력이 있는 경우와 없는 경우에 별도의 MTP 아레나 사용. | #26802, #28549 |
| CUDA 커널 | 선택된 전문가 10개에 대한 빠른 라우팅, 작은 배치의 GLU/top-k 융합, warp 경쟁 상태 수정. | #27978, #27621, #28475 |
| CUDA 튜닝 | 양자화 형식에 따른 MMVQ/MMQ 임곗값, Pascal/Volta 튜닝, 분기 없는 Q4_K/Q5_K, Spark L2 프리페치. | #26079, #26264, #28912, #29753, #26705 |
| HIP / Vulkan / SYCL | RDNA/Vulkan에서 전문가별 타일 크기 조정, Vulkan의 비활성 작업 감소, SYCL의 IQ MoE 디스패치. | #28552, #28935, #29182, #25483, #28476 |
| MTP 메모리 | 로드된 헤드만 fit에 반영, 융합된 텐서를 올바르게 건너뛰기, KV 레이어 필터링, 초안 임베딩 설정 격리. | #26605, #29014, #28630, #26352 |
| MTP 정확성 | 배치 행 순서 보존, EOG에서 중지, 비활성 초안 건너뛰기, 멀티모달 위치 사용. | #29019, #29638, #27404, #28715 |
| GLM-4.5-Air | 완전한 단일 헤드 glm4moe GGUF를 위한 실행 가능한 MTP 그래프와 자동 감지. | #26534 |
| CPU 어텐션 | 벡터화된 F16 변환, 패딩된 softcap 마스킹을 올바르게 처리하는 타일 방식의 어텐션 꼬리 처리. | #26947, #29423 |
| Direct I/O | 텐서 전체를 복제하는 대신 추가 스테이징 할당을 64 MiB로 제한. | #29749 |
| 서버 | 토큰 수를 세기 전에 안전하게 깨어나기, 라우터의 API 키 파일 인증을 부모 프로세스에 유지, 초안 지표 노출. | #29309, #28938, #26389 |
| UI | 영속 저장된 전체 대화 내보내기, 비활성 도구 확인 요청의 반복 중지, 첫 방문 시 사용자 설정 보존. | #27432, #28646, #27365 |
| 옵션 | FA 양자화 쌍을 명시적으로 선택(지원 중단된 별칭은 이후 제거됨), 사용하지 않는 피어 배치 옵션 제거, load-mode 안내 수정. | #28079, #28177, #28334 |

이후의 적용(2026-10-10): #30112(2026-10-08 병합)의 다중 GPU 설계를 이 포크 고유의 전문가 캐시(`mgs-expert-cache.cpp`)에 적용했습니다. GPU마다 할당된 레이어를 위한 풀 하나, 요청된 크기의 GPU 간 분배, GPU별 업로드 대역폭, 파이프라인 병렬 처리 시 캐시 미사용이 그 내용입니다. 그 코드는 복사하지 않았으며, 업스트림 캐시 자체(#29887)도 통합하지 않았습니다. 이 포크는 모든 단계를 GPU와 CPU로 나누기도 하는 자체 캐시를 유지합니다.

이후의 통합(2026-10-13, 버전 0.1.8): 프로젝트가 지원해야 하지만 이 포크에 없던 아키텍처를 업스트림 master에서 이식했으며, DeepSeek-V4.1은 열려 있는 풀 리퀘스트와 참조 코드를 바탕으로 구현했습니다.

| 아키텍처 | 적용된 변경 | 출처 |
| --- | --- | --- |
| Kimi-K3 (`kimi-k3`) | KDA + MLA 하이브리드, 잠재 MoE, 잔차 혼합, 채팅 파서. | #26185, #28121, #28068, #28466, #25444 (일부) |
| GLM-5.3-Flash (`glm5-next`, `glm5v`) | 그래프, k-pool 인덱서, MTP, 비전 프로젝터, SWIGLU_CLAMP. | #27773, #29745, #29824, #29958, #29994, #30042, #29928, #30017, #30097, #30257, #27930, #29805 |
| Qwen3.8-Flash-Next (`qwen4exp`) | 그래프, 하이브리드 인덱스 메모리, 하이퍼 커넥션, 레이어별 임베딩, QSA, MTP, 행 프리페치. | #27742, #27880, #27941, #28023, #28123, #28569, #28751, #28896, #28901, #29599 (일부), #29622 (일부), #29751, #29761, #29819, #29825, #30160 |
| DeepSeek-V4.1-Flash (`deepseek41`) | 그래프, 두 레이아웃용 로더, 캐시, Engram, DSML. | 열려 있는 #28696, 참고용 ik_llama.cpp #2455, #2506, #2507, #2512, antirez/ds4의 `ds4_engram.c`와 `gguf-tools/deepseek41_metadata.py`(MIT), 공식 DeepSeek-V4.1-Flash 추론 코드(MIT) |

이식하지 않은 것: 희소 flash-attention 커널(#27970, #28770). 따라서 QSA와 DSA 레이어는 올바르게 동작하지만 긴 컨텍스트에서 속도가 빨라지지는 않습니다. 토큰과 임베딩이 섞인 배치(#29622의 나머지 부분), 지연 텐서 읽기, DeepSeek-V4.1의 DSpark, 비전, FP8/FP4 KV 시뮬레이션도 이식하지 않았습니다.

다중 토큰 CUDA 융합은 의도적으로 이 포크에 이미 있는 연산만 다룹니다. 이 작업에서는 SWIGLU_CLAMP를 도입하지 않았으며, 이는 나중에 GLM-5.3-Flash와 함께 들어왔습니다(#27930). GLM 자동 MTP는 여전히 완전한 GGUF 트렁크/헤드와 정확히 하나의 헤드를 요구합니다. 이 포크는 영속 저장된 설정을 동기적으로 초기화하므로 UI 초기화 리팩터링은 필요하지 않았고, 사용자 설정 보존 수정만 이식했습니다.

보류한 것: CPU 타일 방식 K/IQ 행렬 곱(#27851)은 더 새로운 작업 공간/양자화 블록 통합이 필요합니다. 가중 전문가 축소와 무조건적 top-k 융합(#25952/#28432/#28422)은 할당자 의존성이 필요합니다. Metal 작업(#28301/#28948)은 재구성된 커널/융합 인프라가 필요합니다. OpenCL MTP 디스패치(#27637)는 더 새로운 바이너리 커널이 필요합니다. 광범위한 UI 렌더링 리팩터링(#28460)은 더 큰 규모의 UI 마이그레이션이 필요합니다. 행 프리페치(#29599)와 Qwen4Exp MTP(#29761)는 나중에 Qwen3.8-Flash-Next와 함께 통합했습니다. JSON 목록은 각 후보를 연결하고 그 사유를 기록합니다. 변환 도구, 추가 바이너리, CI/릴리스, 테스트는 이 서버 전용 통합의 범위 밖에 있습니다.

검토 범위는 소스 diff, 선언/호출 지점, 공유되는 CUDA/HIP/MUSA 빌드 경로, 로컬 확장의 보존입니다. 빌드, 테스트, GPU 벤치마크, 브라우저 실행은 수행하지 않았습니다. 업스트림이 주장하는 벤치마크 수치는 이 포크를 측정한 것이 아니며, 속도와 런타임 정확성은 배포 하드웨어에서 검증해야 합니다.

TLS: [OpenSSL 인증서/키 로딩](https://docs.openssl.org/3.0/man3/SSL_CTX_use_certificate/) · [X.509 확장](https://docs.openssl.org/3.0/man5/x509v3_config/).
