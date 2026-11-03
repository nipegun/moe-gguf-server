#!/bin/bash

set -euo pipefail

vTemporaryDirectory=''
vLogEnabled=false
vLogProcess=''

fCleanup() {
  local vExitStatus=$?
  if [[ -n "$vTemporaryDirectory" && -d "$vTemporaryDirectory" ]]; then
    rm -rf -- "$vTemporaryDirectory" || true
  fi
  if [[ "$vLogEnabled" == true ]]; then
    printf 'Installer finished with status %s.\n' "$vExitStatus"
    exec 1>&3 2>&4
    wait "$vLogProcess" || true
    exec 3>&- 4>&-
  fi
}

trap fCleanup EXIT

fFail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

fRequireCommand() {
  local pCommand="$1"
  command -v -- "$pCommand" >/dev/null 2>&1 || fFail "Required command not found: $pCommand"
}

fShowHelp() {
  printf '%s\n' \
    'Build, install and deploy moe-gguf-server in one step. Run the installer as root.' \
    'It compiles mgs, installs it in PREFIX/bin/mgs and configures the inference and web services.' \
    '' \
    'Build options:' \
    '  --prefix PATH              Installation prefix (default: /usr/local).' \
    '  --build-dir PATH           Build directory inside PROJECT/_/temp/.' \
    '  --jobs N                   Number of parallel build jobs.' \
    '  --cuda                     Use an existing CUDA Toolkit (Debian default).' \
    '  --rocm                     Use an existing ROCm installation (HIP) for AMD GPUs (Debian).' \
    '  --vulkan                   Build the Vulkan backend for AMD, Intel or NVIDIA GPUs (Debian).' \
    '  --cpu                      Build without GPU support (Alpine default).' \
    '  --cuda-architectures LIST  CMAKE_CUDA_ARCHITECTURES value.' \
    '  --rocm-targets LIST        AMD GPU targets, comma separated (default: GPUs present, such as gfx1100).' \
    '  --no-ui                    Build without the web interface.' \
    '  --no-openssl               Build without native HTTPS support.' \
    '  --no-nccl                  Disable NCCL explicitly.' \
    '  --no-system-deps           Do not install system dependencies.' \
    '' \
    'Web options (passed to deploy/configure-web.sh):' \
    '  --domain NAME              Web domain (default: hostname -f, or hostname if that fails).' \
    '  --certificate PATH         Certificate chain (PEM); requires --private-key.' \
    '  --private-key PATH         Private key (PEM); requires --certificate.' \
    '  --proxy-ip IP              Reverse proxy in front of Apache: PROXY protocol on HTTPS and' \
    '                             X-Forwarded-For on HTTP are accepted only from this address.' \
    '  -h, --help                 Show this help.' \
    '' \
    'Environment: CUDACXX selects the CUDA compiler and HIPCXX the ROCm clang compiler.' \
    'Build and staging directories must remain inside PROJECT/_/temp/.' \
    'The Debian entry point downloads sources to /opt/moe-gguf-server-source when run through curl | bash.' \
    'Local checkouts must be stored on a local production filesystem.' \
    'CUDA Toolkit, ROCm and the GPU drivers are never installed or upgraded by this script.' \
    'With --vulkan it installs the Vulkan build packages and the Mesa Vulkan drivers.' \
    'Reconfigure the web services later without rebuilding: deploy/configure-web.sh --help.' \
    'Records: /root/webapp-install.log and /root/webapp-credentials.txt (mode 600).'
}

fDetectJobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc 2>/dev/null || printf '1\n'
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n'
  else
    printf '1\n'
  fi
}

fResolveExecutable() {
  local pCandidate="$1"
  if [[ "$pCandidate" == */* ]]; then
    if [[ -x "$pCandidate" ]]; then
      printf '%s\n' "$pCandidate"
    fi
  else
    command -v -- "$pCandidate" 2>/dev/null || true
  fi
  return 0
}

fDetectRocmTargets() {
  local pHipClangDirectory="$1"
  local pRocmPath="$2"
  local vCandidate vTool vListing vTarget vTargets=''
  # amdgpu-arch prints one target per GPU; rocminfo lists them as "Name: gfx1100" agents.
  for vCandidate in "${pHipClangDirectory}/amdgpu-arch" "${pRocmPath}/llvm/bin/amdgpu-arch" amdgpu-arch \
    "${pRocmPath}/bin/rocminfo" rocminfo; do
    vTool="$(fResolveExecutable "$vCandidate")"
    [[ -n "$vTool" ]] || continue
    vListing="$("$vTool" 2>/dev/null)" || vListing=''
    while IFS= read -r vTarget; do
      [[ -n "$vTarget" ]] || continue
      [[ ",${vTargets}," == *",${vTarget},"* ]] || vTargets+="${vTargets:+,}${vTarget}"
    done < <(printf '%s\n' "$vListing" | sed -nE 's/^[[:space:]]*(Name:[[:space:]]+)?(gfx[0-9a-z]+)[[:space:]]*$/\2/p')
    [[ -z "$vTargets" ]] || break
  done
  printf '%s\n' "$vTargets"
}

fWarnUnsupportedRocmTargets() {
  local pTargets="$1"
  local vTarget
  local -a aTargets=()
  IFS=, read -r -a aTargets < <(printf '%s\n' "$pTargets")
  for vTarget in "${aTargets[@]}"; do
    if [[ "$vTarget" =~ ^gfx[678][0-9][0-9]([^0-9a-z]|$) ]]; then
      printf 'Warning: current ROCm releases do not support %s (GCN up to Polaris, such as the Radeon RX 580, gfx803); use --vulkan for that GPU.\n' "$vTarget"
    fi
  done
}

fRunAptGet() {
  env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 "$@" || fFail 'apt-get failed.'
}

fFindNcclVersion() {
  local pCudaRelease="$1"
  local vPackage vVersion vRepository
  local vVersions
  vVersions="$(apt-cache madison libnccl2)" || fFail 'Cannot read the NCCL package index.'
  while IFS='|' read -r vPackage vVersion vRepository; do
    vVersion="${vVersion//[[:space:]]/}"
    if [[ "$vVersion" == *"+cuda${pCudaRelease}" ]]; then
      printf '%s\n' "$vVersion"
    fi
  done < <(printf '%s\n' "$vVersions")
}

fAddNvidiaRepository() {
  local pRepositoryOs="$1"
  local pRepositoryArchitecture="$2"
  local cKeyringVersion=1.1-1
  local vKeyringPackage vKeyringUrl
  vTemporaryDirectory="$(mktemp -d "${vTemporaryRoot}/nvidia-keyring.XXXXXXXX")" || fFail 'Cannot create keyring directory.'
  vKeyringPackage="${vTemporaryDirectory}/cuda-keyring.deb"
  vKeyringUrl="https://developer.download.nvidia.com/compute/cuda/repos/${pRepositoryOs}/${pRepositoryArchitecture}/cuda-keyring_${cKeyringVersion}_all.deb"
  printf '%s\n' 'Adding the NVIDIA repository for NCCL only (no Toolkit or driver installation).'
  curl --fail --location --retry 3 --output "$vKeyringPackage" "$vKeyringUrl" || fFail 'Cannot download the NVIDIA keyring.'
  dpkg -i "$vKeyringPackage" || fFail 'Cannot install the NVIDIA keyring.'
  rm -rf -- "$vTemporaryDirectory" || fFail 'Cannot remove the temporary NVIDIA keyring.'
  vTemporaryDirectory=''
}

fInstallNccl() {
  local pRepositoryOs="$1"
  local pRepositoryArchitecture="$2"
  local vNcclVersion vAvailableVersions
  vNcclVersion="$(fFindNcclVersion "$vCudaVersion" | sort -V | tail -n 1)" || fFail 'Cannot select NCCL.'
  if [[ -z "$vNcclVersion" ]]; then
    fAddNvidiaRepository "$pRepositoryOs" "$pRepositoryArchitecture"
    fRunAptGet update
    vNcclVersion="$(fFindNcclVersion "$vCudaVersion" | sort -V | tail -n 1)" || fFail 'Cannot select NCCL.'
  fi
  [[ -n "$vNcclVersion" ]] || fFail "NCCL is unavailable for CUDA ${vCudaVersion}."
  vAvailableVersions="$(apt-cache madison libnccl-dev | cut -d '|' -f 2 | tr -d ' ')" || fFail 'Cannot read libnccl-dev versions.'
  printf '%s\n' "$vAvailableVersions" | grep -Fxq -- "$vNcclVersion" || fFail "libnccl-dev ${vNcclVersion} is unavailable."
  printf 'Installing NCCL %s for CUDA %s.\n' "$vNcclVersion" "$vCudaVersion"
  # The matching CUDA build may have a lower package version than the installed NCCL.
  fRunAptGet install -y --allow-downgrades --no-install-recommends "libnccl2=${vNcclVersion}" "libnccl-dev=${vNcclVersion}"
}

fInstallSystemDependencies() {
  local vRepositoryOs vRepositoryArchitecture vDpkgArchitecture
  local -a aPackages
  if [[ "$vDistribution" == alpine ]]; then
    fRequireCommand apk
    aPackages=(build-base ca-certificates ccache cmake coreutils curl gzip libgomp ninja pkgconf tar)
    [[ "$vBuildOpenSsl" != ON ]] || aPackages+=(openssl-dev)
    [[ "$vBuildUi" != ON ]] || aPackages+=(nodejs npm)
    apk add --no-cache "${aPackages[@]}" || fFail 'Cannot install Alpine dependencies.'
    return 0
  fi
  fRequireCommand apt-get
  fRequireCommand dpkg
  aPackages=(build-essential ca-certificates ccache cmake curl gzip libgomp1 ninja-build pkg-config tar)
  [[ "$vBuildOpenSsl" != ON ]] || aPackages+=(libssl-dev)
  [[ "$vBuildUi" != ON ]] || aPackages+=(nodejs npm)
  # Vulkan headers, SPIR-V headers and shader compiler for the build; Mesa provides the AMD and Intel drivers.
  [[ "$vBuildBackend" != vulkan ]] || aPackages+=(libvulkan-dev spirv-headers glslc mesa-vulkan-drivers)
  fRunAptGet update
  fRunAptGet install -y --no-install-recommends "${aPackages[@]}"
  if [[ "$vBuildBackend" == cuda && "$vUseNccl" == ON ]]; then
    case "$vDistribution" in
      debian) vRepositoryOs="debian${VERSION_ID%%.*}" ;;
      ubuntu) vRepositoryOs="ubuntu${VERSION_ID//./}" ;;
    esac
    vDpkgArchitecture="$(dpkg --print-architecture)" || fFail 'Cannot detect the package architecture.'
    case "$vDpkgArchitecture" in
      amd64) vRepositoryArchitecture=x86_64 ;;
      arm64) vRepositoryArchitecture=sbsa ;;
      *) fFail "NCCL installation is not supported on ${vDpkgArchitecture}." ;;
    esac
    fInstallNccl "$vRepositoryOs" "$vRepositoryArchitecture"
  fi
}

fMain() {
  local vScriptDirectory vProjectDirectory vTemporaryRoot
  local vBuildDirectory vStagedSourceDirectory vInstallPrefix=/usr/local vBuildJobs='' vCachedSourceDirectory
  local vBuildBackend="$cDefaultBackend" vBuildUi=ON vBuildOpenSsl=ON vInstallDependencies=ON vUseNccl=ON
  local vCudaArchitectures='' vCudaEnabled=OFF vNcclEnabled=OFF vCudaCompiler='' vCudaVersion=''
  local vVulkanEnabled=OFF
  local vRocmTargets='' vHipEnabled=OFF vHipConfig='' vHipCompiler='' vHipClangDirectory='' vRocmPath='' vRocmVersion=''
  local vDistribution vNodeMajor vNcclIncludeDirectory vNcclLibrary vServerBinary='' vCandidate vLibraryListing
  local -a aGeneratorArguments=() aConfigureArguments=() aWebArguments=()
  vScriptDirectory="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || fFail 'Cannot locate the installer.'
  vProjectDirectory="$(CDPATH= cd -- "${vScriptDirectory}/.." && pwd -P)" || fFail 'Cannot locate the project.'
  vTemporaryRoot="${vProjectDirectory}/_/temp"
  vBuildDirectory="${vTemporaryRoot}/build"
  vStagedSourceDirectory="${vTemporaryRoot}/source"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --prefix|--build-dir|--jobs|--cuda-architectures|--rocm-targets|--domain|--certificate|--private-key|--proxy-ip)
        [[ $# -ge 2 ]] || fFail "Missing value after $1."
        case "$1" in
          --prefix) vInstallPrefix="$2" ;;
          --build-dir) vBuildDirectory="$2" ;;
          --jobs) vBuildJobs="$2" ;;
          --cuda-architectures) vCudaArchitectures="$2" ;;
          --rocm-targets) vRocmTargets="$2" ;;
          --domain|--certificate|--private-key|--proxy-ip) aWebArguments+=("$1" "$2") ;;
        esac
        shift 2
        ;;
      --cuda) vBuildBackend=cuda; shift ;;
      --rocm) vBuildBackend=rocm; shift ;;
      --vulkan) vBuildBackend=vulkan; shift ;;
      --cpu) vBuildBackend=cpu; shift ;;
      --no-ui) vBuildUi=OFF; shift ;;
      --no-openssl) vBuildOpenSsl=OFF; shift ;;
      --no-nccl) vUseNccl=OFF; shift ;;
      --no-system-deps) vInstallDependencies=OFF; shift ;;
      -h|--help) fShowHelp; return 0 ;;
      *) fFail "Unknown option: $1" ;;
    esac
  done

  [[ $EUID -eq 0 ]] || fFail 'Run this installer as root; sudo is not used.'
  [[ -f "${vProjectDirectory}/build/CMakeLists.txt" ]] || fFail 'Project root not found.'
  [[ -r /etc/os-release ]] || fFail 'Cannot identify the distribution.'
  . /etc/os-release
  vDistribution="${ID:-}"
  case "${cInstallerDistribution}:${vDistribution}" in
    debian:debian|debian:ubuntu|alpine:alpine) ;;
    *) fFail "This installer does not support ${vDistribution}." ;;
  esac

  umask 077
  touch /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot create installation records.'
  chmod 600 /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot protect installation records.'
  fRequireCommand tee
  if [[ "${MGS_INSTALLER_PARENT_LOG:-false}" != true ]]; then
    exec 3>&1 4>&2
    exec > >(tee -a /root/webapp-install.log) 2>&1
    vLogProcess=$!
    vLogEnabled=true
  fi
  printf '\nmoe-gguf-server installation: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  [[ -f "${vScriptDirectory}/configure-web.sh" ]] || fFail 'The web configuration script is missing.'
  # Validate the web options before the build, so a typo does not surface after a long compilation.
  MGS_INSTALLER_PARENT_LOG=true /bin/bash "${vScriptDirectory}/configure-web.sh" --check "${aWebArguments[@]}" < /dev/null ||
    fFail 'Invalid web options.'

  if [[ "$vDistribution" == alpine && "$vInstallDependencies" == ON ]]; then
    apk add --no-cache coreutils || fFail 'Cannot install the path canonicalization tools.'
  fi
  fRequireCommand realpath
  mkdir -p "$vTemporaryRoot" || fFail 'Cannot create the project temporary directory.'
  vTemporaryRoot="$(realpath -m -- "$vTemporaryRoot")" || fFail 'Cannot resolve the temporary directory.'
  vBuildDirectory="$(realpath -m -- "$vBuildDirectory")" || fFail 'Cannot resolve the build directory.'
  vStagedSourceDirectory="$(realpath -m -- "$vStagedSourceDirectory")" || fFail 'Cannot resolve the staging directory.'
  [[ "$vBuildDirectory" == "$vTemporaryRoot/"* ]] || fFail 'The build directory must be inside PROJECT/_/temp/.'
  [[ "$vStagedSourceDirectory" == "$vTemporaryRoot/"* ]] || fFail 'The source staging directory must be inside PROJECT/_/temp/.'
  [[ "$vStagedSourceDirectory" != "$vBuildDirectory" && "$vStagedSourceDirectory" != "$vBuildDirectory/"* && "$vBuildDirectory" != "$vStagedSourceDirectory/"* ]] || fFail 'Build and staging directories must be separate and cannot contain one another.'
  [[ "$vInstallPrefix" == /* ]] || fFail '--prefix must be an absolute path.'
  [[ -n "$vBuildJobs" ]] || vBuildJobs="$(fDetectJobs)"
  [[ "$vBuildJobs" =~ ^[1-9][0-9]*$ ]] || fFail '--jobs must be a positive integer.'
  if [[ "$vBuildBackend" == cuda ]]; then
    [[ "$vDistribution" != alpine ]] || fFail 'This Alpine installer supports CPU builds; use Debian for CUDA/NCCL.'
    vCudaEnabled=ON
    [[ "$vUseNccl" != ON ]] || vNcclEnabled=ON
    if [[ -n "${CUDACXX:-}" ]]; then
      vCudaCompiler="$(fResolveExecutable "$CUDACXX")"
    elif command -v nvcc >/dev/null 2>&1; then
      vCudaCompiler="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
      vCudaCompiler=/usr/local/cuda/bin/nvcc
    fi
    [[ -n "$vCudaCompiler" ]] || fFail 'Install CUDA Toolkit first, set CUDACXX, or use --rocm, --vulkan or --cpu. This installer never installs the Toolkit.'
    vCudaVersion="$("$vCudaCompiler" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')" || fFail 'Cannot read the CUDA version.'
    vCudaVersion="${vCudaVersion%%$'\n'*}"
    [[ -n "$vCudaVersion" ]] || fFail 'Cannot identify the CUDA Toolkit version.'
    printf 'Using existing CUDA Toolkit %s: %s\n' "$vCudaVersion" "$vCudaCompiler"
  fi
  if [[ "$vBuildBackend" == rocm ]]; then
    [[ "$vDistribution" != alpine ]] || fFail 'This Alpine installer supports CPU builds; use Debian for ROCm.'
    vHipEnabled=ON
    vHipConfig="$(fResolveExecutable hipconfig)"
    if [[ -z "$vHipConfig" && -x /opt/rocm/bin/hipconfig ]]; then
      vHipConfig=/opt/rocm/bin/hipconfig
    fi
    [[ -n "$vHipConfig" ]] || fFail 'Install ROCm first (HIP, hipBLAS and rocBLAS with their development files), or use --cuda, --vulkan or --cpu. This installer never installs ROCm.'
    vRocmPath="$("$vHipConfig" -R)" || fFail 'Cannot read the ROCm path from hipconfig.'
    vRocmPath="${vRocmPath%%$'\n'*}"
    vHipClangDirectory="$("$vHipConfig" -l)" || fFail 'Cannot read the ROCm clang path from hipconfig.'
    vHipClangDirectory="${vHipClangDirectory%%$'\n'*}"
    if [[ -n "${HIPCXX:-}" ]]; then
      vHipCompiler="$(fResolveExecutable "$HIPCXX")"
    else
      vHipCompiler="$(fResolveExecutable "${vHipClangDirectory}/clang")"
    fi
    [[ -n "$vHipCompiler" && -n "$vRocmPath" ]] || fFail 'Cannot locate the ROCm clang compiler; set HIPCXX.'
    vRocmVersion="$("$vHipConfig" -v)" || fFail 'Cannot read the HIP version.'
    vRocmVersion="${vRocmVersion%%$'\n'*}"
    if [[ -z "$vRocmTargets" ]]; then
      vRocmTargets="$(fDetectRocmTargets "$vHipClangDirectory" "$vRocmPath")"
      [[ -n "$vRocmTargets" ]] || fFail 'No AMD GPU was detected; pass --rocm-targets with the GPU targets (for example gfx1100).'
    fi
    [[ "$vRocmTargets" =~ ^gfx[0-9A-Za-z:+-]+(,gfx[0-9A-Za-z:+-]+)*$ ]] || fFail '--rocm-targets must be a comma separated list such as gfx1100,gfx1030.'
    fWarnUnsupportedRocmTargets "$vRocmTargets"
    # The HIP compiler and ggml locate the ROCm runtime and device libraries through these variables.
    export HIP_PATH="$vRocmPath"
    export ROCM_PATH="$vRocmPath"
    printf 'Using existing ROCm (HIP %s): %s, targets %s\n' "$vRocmVersion" "$vHipCompiler" "$vRocmTargets"
  fi
  if [[ "$vBuildBackend" == vulkan ]]; then
    [[ "$vDistribution" != alpine ]] || fFail 'This Alpine installer supports CPU builds; use Debian for Vulkan.'
    vVulkanEnabled=ON
  fi

  if [[ "$vInstallDependencies" == ON ]]; then
    fInstallSystemDependencies
  fi
  if [[ "$vVulkanEnabled" == ON ]]; then
    fRequireCommand glslc
  fi
  for vCandidate in cmake install rm tar; do
    fRequireCommand "$vCandidate"
  done
  if [[ "$vBuildUi" == ON ]]; then
    fRequireCommand node
    fRequireCommand npm
    vNodeMajor="$(node -p 'process.versions.node.split(".").map(Number).slice(0, 2).join(" ")')" || fFail 'Cannot identify Node.js.'
    local vNodeMinor="${vNodeMajor#* }"
    vNodeMajor="${vNodeMajor%% *}"
    [[ "$vNodeMajor" -eq 20 && "$vNodeMinor" -ge 19 || "$vNodeMajor" -eq 22 && "$vNodeMinor" -ge 12 || "$vNodeMajor" -gt 22 ]] || fFail 'The local interface requires Node.js 20.19+ or 22.12+. Install a supported version or use --no-ui.'
    export npm_config_cache="${vTemporaryRoot}/npm-cache"
    export TMPDIR="$vTemporaryRoot"
  fi

  if [[ -f "${vBuildDirectory}/CMakeCache.txt" ]]; then
    vCachedSourceDirectory="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${vBuildDirectory}/CMakeCache.txt")" || fFail 'Cannot read the CMake source directory.'
    if [[ "$vCachedSourceDirectory" != "${vStagedSourceDirectory}/build" ]]; then
      printf '%s\n' 'Resetting CMake configuration for the current source directory.'
      rm -f -- "${vBuildDirectory}/CMakeCache.txt" || fFail 'Cannot remove the stale CMake cache.'
      rm -rf -- "${vBuildDirectory}/CMakeFiles" || fFail 'Cannot remove stale CMake metadata.'
    fi
  fi
  if [[ ! -f "${vBuildDirectory}/CMakeCache.txt" ]] && command -v ninja >/dev/null 2>&1; then
    aGeneratorArguments=(-G Ninja)
  fi
  aConfigureArguments=(
    -S "${vStagedSourceDirectory}/build"
    -B "$vBuildDirectory"
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${vInstallPrefix}"
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_NATIVE=ON
    "-DGGML_CUDA=${vCudaEnabled}"
    "-DGGML_CUDA_NCCL=${vNcclEnabled}"
    "-DGGML_VULKAN=${vVulkanEnabled}"
    "-DGGML_HIP=${vHipEnabled}"
    "-DMGS_BUILD_UI=${vBuildUi}"
    "-DMGS_OPENSSL=${vBuildOpenSsl}"
  )
  [[ -z "$vCudaCompiler" ]] || aConfigureArguments+=("-DCMAKE_CUDA_COMPILER=${vCudaCompiler}")
  [[ -z "$vCudaArchitectures" ]] || aConfigureArguments+=("-DCMAKE_CUDA_ARCHITECTURES=${vCudaArchitectures}")
  [[ -z "$vHipCompiler" ]] || aConfigureArguments+=("-DCMAKE_HIP_COMPILER=${vHipCompiler}" "-DGPU_TARGETS=${vRocmTargets//,/;}")

  printf 'Staging sources in %s.\n' "$vStagedSourceDirectory"
  rm -rf -- "$vStagedSourceDirectory" || fFail 'Cannot remove the previous staged source.'
  install -d -m 0755 "$vStagedSourceDirectory" || fFail 'Cannot create the staged source directory.'
  tar -C "$vProjectDirectory" \
    --exclude='./.git' --exclude='./.agents' --exclude='./.claude' --exclude='./.codex' --exclude='./.aws' \
    --exclude='./_' --exclude='./build-*' --exclude='./cmake-build-*' \
    --exclude='*/node_modules' --exclude='*/.svelte-kit' --exclude='*/dist' \
    -cf - . | tar -C "$vStagedSourceDirectory" -xf - || fFail 'Cannot stage the project.'
  printf 'Configuring moe-gguf-server (%s) in %s.\n' "$vBuildBackend" "$vBuildDirectory"
  if ! cmake "${aGeneratorArguments[@]}" "${aConfigureArguments[@]}"; then
    [[ "$vVulkanEnabled" != ON ]] || fFail 'CMake configuration failed. The Vulkan backend needs libvulkan-dev, spirv-headers and glslc.'
    [[ "$vHipEnabled" != ON ]] || fFail 'CMake configuration failed. The ROCm backend needs HIP 6.1 or later with the hipBLAS and rocBLAS development files.'
    fFail 'CMake configuration failed.'
  fi
  if [[ "$vNcclEnabled" == ON ]]; then
    vNcclIncludeDirectory="$(sed -n 's/^NCCL_INCLUDE_DIR:PATH=//p' "${vBuildDirectory}/CMakeCache.txt")" || fFail 'Cannot read NCCL include configuration.'
    vNcclLibrary="$(sed -n 's/^NCCL_LIBRARY:FILEPATH=//p' "${vBuildDirectory}/CMakeCache.txt")" || fFail 'Cannot read NCCL library configuration.'
    vNcclIncludeDirectory="${vNcclIncludeDirectory%%$'\n'*}"
    vNcclLibrary="${vNcclLibrary%%$'\n'*}"
    [[ -n "$vNcclIncludeDirectory" && "$vNcclIncludeDirectory" != *-NOTFOUND && -n "$vNcclLibrary" && "$vNcclLibrary" != *-NOTFOUND ]] || fFail 'NCCL was not detected; install the matching NCCL packages or explicitly use --no-nccl.'
    printf 'NCCL enabled: %s (headers: %s).\n' "$vNcclLibrary" "$vNcclIncludeDirectory"
  fi
  cmake --build "$vBuildDirectory" --config Release --target mgs --parallel "$vBuildJobs" || fFail 'Compilation failed.'
  vServerBinary="${vBuildDirectory}/bin/mgs"
  [[ -x "$vServerBinary" ]] || fFail 'Compilation did not produce mgs.'
  if [[ "$vHipEnabled" == ON ]]; then
    # Like the CUDA libraries, the ROCm libraries are found through the run path that CMake records in the binary.
    vLibraryListing="$(ldd "$vServerBinary")" || fFail 'Cannot list the shared libraries of mgs.'
    if [[ "$vLibraryListing" == *'=> not found'* ]]; then
      printf '%s\n' "$vLibraryListing" >&2
      fFail "Some ROCm libraries cannot be found at run time; add ${vRocmPath}/lib to /etc/ld.so.conf.d/ and run ldconfig."
    fi
  fi
  vInstallPrefix="${vInstallPrefix%/}"
  install -d -m 0755 "${vInstallPrefix}/bin" || fFail 'Cannot create the installation directory.'
  install -C -m 0755 "$vServerBinary" "${vInstallPrefix}/bin/mgs" || fFail 'Cannot install mgs.'
  printf '\nInstalled: %s/bin/mgs\n' "$vInstallPrefix"
  if [[ "$vBuildBackend" == cuda ]]; then
    printf '%s\n' 'CUDA MoE optimizations are enabled by default.'
  elif [[ "$vBuildBackend" == rocm ]]; then
    printf '%s\n' 'ROCm backend enabled. Users that run mgs need access to /dev/kfd and /dev/dri (groups render and video);' \
      'configure-web.sh adds the service account to them.'
  elif [[ "$vBuildBackend" == vulkan ]]; then
    printf '%s\n' 'Vulkan backend enabled. Users that run mgs need access to /dev/dri (groups render and video);' \
      'configure-web.sh adds the service account to them.'
  fi
  printf '%s\n' 'MTP activates automatically for GGUF models with supported integrated MTP heads.'

  # The web services are part of the same deployment; configure-web.sh can rerun alone to reconfigure them.
  if [[ "$vInstallDependencies" == OFF ]]; then
    aWebArguments+=(--no-system-deps)
  fi
  printf '\nConfiguring the inference and web services.\n'
  MGS_INSTALLER_PARENT_LOG=true /bin/bash "${vScriptDirectory}/configure-web.sh" \
    --binary "${vInstallPrefix}/bin/mgs" "${aWebArguments[@]}" < /dev/null || fFail 'The web configuration failed.'
}
