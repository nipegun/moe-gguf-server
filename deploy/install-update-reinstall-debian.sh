#!/bin/bash

set -euo pipefail

readonly cRepositoryUrl=https://github.com/nipegun/moe-gguf-server
readonly cArchiveUrl=https://github.com/nipegun/moe-gguf-server/archive/refs/heads/main.tar.gz
readonly cEntryScript="${BASH_SOURCE[0]:-}"
vDownloadDirectory=''
vLogEnabled=false
vLogProcess=''

fCleanup() {
  local vExitStatus=$?
  if [[ -n "$vDownloadDirectory" && -d "$vDownloadDirectory" ]]; then
    rm -rf -- "$vDownloadDirectory" || true
  fi
  if [[ "$vLogEnabled" == true ]]; then
    printf 'Debian installer finished with status %s.\n' "$vExitStatus"
    exec 1>&3 2>&4
    wait "$vLogProcess" || true
    exec 3>&- 4>&-
  fi
}

trap fCleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fFail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

fShowHelp() {
  printf '%s\n' \
    'Install, update or reinstall moe-gguf-server on Debian/Ubuntu as root.' \
    'Run from a complete local checkout, a standalone download, or curl | bash.' \
    '' \
    'Options forwarded to the build installer:' \
    '  --prefix PATH              Installation prefix (default: /usr/local).' \
    '  --build-dir PATH           Build directory inside PROJECT/_/temp/.' \
    '  --jobs N                   Number of parallel build jobs.' \
    '  --cuda                     Use an existing CUDA Toolkit (default).' \
    '  --vulkan                   Build the Vulkan backend (AMD, Intel or NVIDIA GPUs).' \
    '  --cpu                      Build without GPU support.' \
    '  --cuda-architectures LIST  CMAKE_CUDA_ARCHITECTURES value.' \
    '  --build-ui                 Build the local web interface (default).' \
    '  --no-ui                    Build without the web interface.' \
    '  --no-openssl               Build without native HTTPS support.' \
    '  --no-nccl                  Disable NCCL explicitly.' \
    '  --no-system-deps           Require dependencies to be installed already.' \
    '  -h, --help                 Show this help without downloading sources.' \
    '' \
    'Remote source directory: MGS_INSTALL_SOURCE_DIR (default: /opt/moe-gguf-server-source).' \
    'Build environment: MGS_BUILD_DIR, MGS_SOURCE_STAGE_DIR, MGS_INSTALL_PREFIX,' \
    'MGS_BUILD_JOBS, MGS_BUILD_BACKEND and CUDACXX.' \
    'Remote mode downloads main from https://github.com/nipegun/moe-gguf-server without Git.' \
    'Pass pipeline options with: curl -fsSL INSTALLER_URL | bash -s -- --vulkan --jobs 4'
}

fStartLog() {
  umask 077
  touch /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot create installation records.'
  chmod 600 /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot protect installation records.'
  command -v tee >/dev/null 2>&1 || fFail 'The installer requires tee from coreutils.'
  exec 3>&1 4>&2
  exec > >(tee -a /root/webapp-install.log) 2>&1
  vLogProcess=$!
  vLogEnabled=true
  printf '\nDebian installer entry: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

fEnsureDownloadDependencies() {
  local pInstallDependencies="$1"
  local vCommand vMissing=false
  for vCommand in curl tar gzip rsync flock realpath; do
    if ! command -v "$vCommand" >/dev/null 2>&1; then
      printf 'Missing source download dependency: %s\n' "$vCommand"
      vMissing=true
    fi
  done
  [[ -s /etc/ssl/certs/ca-certificates.crt ]] || vMissing=true
  if [[ "$vMissing" == true ]]; then
    [[ "$pInstallDependencies" == true ]] || fFail 'Install curl, ca-certificates, tar, gzip, rsync, util-linux and coreutils before using --no-system-deps.'
    env DEBIAN_FRONTEND=noninteractive apt-get update || fFail 'Cannot update bootstrap package lists.'
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl tar gzip rsync util-linux coreutils || fFail 'Cannot install source download dependencies.'
  fi
}

fLockSourceDirectory() {
  local pSourceDirectory="$1"
  command -v flock >/dev/null 2>&1 || fFail 'Install util-linux to provide the installer lock.'
  mkdir -p "${pSourceDirectory}/_/temp" || fFail 'Cannot create the project temporary directory.'
  exec 9>"${pSourceDirectory}/_/temp/installer.lock" || fFail 'Cannot open the installer lock.'
  flock -n 9 || fFail 'Another installer is already using this source directory.'
}

fValidateManagedSource() {
  local pSourceDirectory="$1"
  local vEntry vOrigin
  if [[ -f "${pSourceDirectory}/_/source-origin" ]]; then
    IFS= read -r vOrigin < "${pSourceDirectory}/_/source-origin" || fFail 'Cannot read the source origin.'
    [[ "$vOrigin" == "$cRepositoryUrl" ]] || fFail 'The source directory belongs to another repository.'
  else
    # Only an empty directory (apart from our workspace) can become a managed copy.
    for vEntry in "${pSourceDirectory}"/* "${pSourceDirectory}"/.[!.]* "${pSourceDirectory}"/..?*; do
      [[ -e "$vEntry" || -L "$vEntry" ]] || continue
      [[ "${vEntry##*/}" == _ ]] || fFail 'Refusing to overwrite an unmanaged source directory; choose another MGS_INSTALL_SOURCE_DIR or run its local installer.'
    done
  fi
}

fDownloadSource() {
  local pSourceDirectory="$1"
  local vEntry vMember vArchive vSource
  [[ "$pSourceDirectory" == /* && "$pSourceDirectory" != / ]] || fFail 'MGS_INSTALL_SOURCE_DIR must be an absolute project directory.'
  [[ ! -L "$pSourceDirectory" ]] || fFail 'The managed source directory cannot be a symbolic link.'
  fValidateManagedSource "$pSourceDirectory"
  mkdir -p "$pSourceDirectory" || fFail 'Cannot create the managed source directory.'
  [[ "$(stat -c '%u' "$pSourceDirectory")" == 0 ]] || fFail 'The managed source directory must be owned by root.'
  fLockSourceDirectory "$pSourceDirectory"
  fValidateManagedSource "$pSourceDirectory"
  chmod 700 "$pSourceDirectory" || fFail 'Cannot protect the managed source directory.'
  vDownloadDirectory="$(mktemp -d "${pSourceDirectory}/_/temp/download.XXXXXXXX")" || fFail 'Cannot create the download directory.'
  vArchive="${vDownloadDirectory}/source.tar.gz"
  vSource="${vDownloadDirectory}/source"
  mkdir -p "$vSource" || fFail 'Cannot create the extraction directory.'
  printf 'Downloading source snapshot: %s\n' "$cArchiveUrl"
  curl --fail --location --show-error --silent --proto '=https' --proto-redir '=https' \
    --connect-timeout 30 --retry 3 --output "$vArchive" "$cArchiveUrl" || fFail 'Cannot download the main source archive.'
  tar -tzf "$vArchive" > "${vDownloadDirectory}/members.txt" || fFail 'The downloaded source archive is invalid or incomplete.'
  while IFS= read -r vMember; do
    [[ "$vMember" == moe-gguf-server-main/* && "/${vMember}/" != */../* && "/${vMember}/" != */./* ]] || fFail 'The source archive contains an unexpected path.'
  done < "${vDownloadDirectory}/members.txt"
  tar -xzf "$vArchive" --strip-components=1 --no-same-owner --no-same-permissions \
    -C "$vSource" || fFail 'Cannot extract the source archive.'
  for vEntry in build/CMakeLists.txt backend/CMakeLists.txt deploy/install-common.sh deploy/configure-web.sh frontend/package.json; do
    [[ -f "${vSource}/${vEntry}" && ! -L "${vSource}/${vEntry}" ]] || fFail "The source archive is missing ${vEntry}."
  done
  printf '%s\n' "$cRepositoryUrl" > "${pSourceDirectory}/_/source-origin" || fFail 'Cannot record the managed source origin.'
  # Delete obsolete source files while preserving local caches and administrative data.
  # Changed content gets a fresh timestamp, including when main moves to an older snapshot.
  rsync -a --checksum --no-times --delete \
    --exclude='/_/' --exclude='/.git/' --exclude='/.agents/' --exclude='/.codex/' --exclude='/.aws/' \
    "${vSource}/" "${pSourceDirectory}/" || fFail 'Cannot update the managed source copy.'
  printf 'Sources ready in %s.\n' "$pSourceDirectory"
}

fRunLocalInstaller() {
  local pSourceDirectory="$1"
  shift
  # A separate shell keeps the downloaded helper's functions and EXIT trap isolated.
  MGS_INSTALLER_PARENT_LOG=true /bin/bash -c '
set -euo pipefail
readonly cInstallerDistribution=debian
readonly cDefaultBackend=cuda
vSourceDirectory="$1"
shift
. "${vSourceDirectory}/deploy/install-common.sh"
if ! fMain "$@"; then
  printf "%s\n" "Debian build installation failed." >&2
  exit 1
fi
' moe-gguf-server-installer "$pSourceDirectory" "$@" < /dev/null || fFail 'The Debian build installer failed.'
}

fMain() {
  local vSourceDirectory='' vScriptDirectory vArgument
  local vInstallDependencies=true vShowHelp=false
  local -a aInstallerArguments=("$@")
  while [[ $# -gt 0 ]]; do
    vArgument="$1"
    case "$vArgument" in
      --prefix|--build-dir|--jobs|--cuda-architectures)
        [[ $# -ge 2 ]] || fFail "Missing value after ${vArgument}."
        shift 2
        ;;
      --no-system-deps) vInstallDependencies=false; shift ;;
      -h|--help) vShowHelp=true; shift ;;
      --cuda|--vulkan|--cpu|--build-ui|--no-ui|--no-openssl|--no-nccl) shift ;;
      *) fFail "Unknown option: ${vArgument}" ;;
    esac
  done
  if [[ "$vShowHelp" == true ]]; then
    fShowHelp
    return 0
  fi
  [[ $EUID -eq 0 ]] || fFail 'Run the installer as root; sudo is never used.'
  [[ -r /etc/os-release ]] || fFail 'Cannot identify the distribution.'
  . /etc/os-release
  case "${ID:-}" in
    debian|ubuntu) ;;
    *) fFail 'This installer supports Debian and Ubuntu only.' ;;
  esac
  fStartLog
  if [[ -n "$cEntryScript" && -f "$cEntryScript" ]]; then
    vScriptDirectory="$(CDPATH= cd -- "$(dirname -- "$cEntryScript")" && pwd -P)" || fFail 'Cannot locate the local installer.'
    if [[ -f "${vScriptDirectory}/install-common.sh" && -f "${vScriptDirectory}/../build/CMakeLists.txt" && -d "${vScriptDirectory}/../backend" ]]; then
      vSourceDirectory="$(CDPATH= cd -- "${vScriptDirectory}/.." && pwd -P)" || fFail 'Cannot locate the local source tree.'
    fi
  fi
  if [[ -n "$vSourceDirectory" ]]; then
    fLockSourceDirectory "$vSourceDirectory"
    printf 'Installing the local source tree: %s\n' "$vSourceDirectory"
  else
    fEnsureDownloadDependencies "$vInstallDependencies"
    [[ "${MGS_INSTALL_SOURCE_DIR:-/opt/moe-gguf-server-source}" == /* ]] || fFail 'MGS_INSTALL_SOURCE_DIR must be absolute.'
    vSourceDirectory="$(realpath -m -- "${MGS_INSTALL_SOURCE_DIR:-/opt/moe-gguf-server-source}")" || fFail 'Cannot resolve the source directory.'
    fDownloadSource "$vSourceDirectory"
  fi
  fRunLocalInstaller "$vSourceDirectory" "${aInstallerArguments[@]}"
  printf 'Web deployment command: bash %q --domain YOUR_DOMAIN\n' "${vSourceDirectory}/deploy/configure-web.sh"
}

if ! fMain "$@" < /dev/null; then
  printf '%s\n' 'Debian installation failed.' >&2
  exit 1
fi
