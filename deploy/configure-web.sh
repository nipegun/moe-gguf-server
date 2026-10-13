#!/bin/bash

set -euo pipefail

# Configuration of the program lives with the service account, in ~/.mgs/config/ of /opt/moe-gguf-server
readonly cConfigurationDirectory=/opt/moe-gguf-server/.mgs/config

vLogEnabled=false
vLogProcess=''

fCleanup() {
  local vExitStatus=$?
  if [[ "$vLogEnabled" == true ]]; then
    printf 'Web configuration finished with status %s.\n' "$vExitStatus"
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

fWaitForWebPorts() {
  local vTick
  for ((vTick=0; vTick<600; vTick++)); do
    if [[ -s /opt/moe-gguf-server/_/temp/web/ports.json ]]; then
      cat /opt/moe-gguf-server/_/temp/web/ports.json || fFail 'Cannot read the selected web ports.'
      return 0
    fi
    sleep 0.1 || true
  done
  fFail 'The web service did not publish its ports; inspect the moe-gguf-server-web service log.'
}

fMain() {
  local vDomain='' vCertificate='' vPrivateKey='' vBinary=/usr/local/bin/mgs
  local vScriptDirectory vApiKey vModuleDirectory vSubjectAltName vLogFile vGroup
  vScriptDirectory="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || fFail 'Cannot locate deployment templates.'
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain|--certificate|--private-key|--binary)
        [[ $# -ge 2 ]] || fFail "Missing value after $1."
        case "$1" in
          --domain) vDomain="$2" ;;
          --certificate) vCertificate="$2" ;;
          --private-key) vPrivateKey="$2" ;;
          --binary) vBinary="$2" ;;
        esac
        shift 2
        ;;
      -h|--help)
        printf '%s\n' 'Configure non-root inference and web services.' \
          'Every web startup selects HTTP from 11080 and HTTPS from 11443, incrementing occupied ports.' \
          'Run as root: deploy/configure-web.sh --domain ai.example.com' \
          'Optional: --certificate /path/fullchain.pem --private-key /path/privkey.pem' \
          'Optional: --binary /usr/local/bin/mgs' \
          'The certificate lives in /opt/moe-gguf-server/certificates/: supplied files are copied there; without them, the' \
          'certificate already there is kept, or a self-signed certificate is created once and reused by every reinstall.'
        return 0
        ;;
      *) fFail "Unknown option: $1" ;;
    esac
  done
  [[ $EUID -eq 0 ]] || fFail 'Run this script as root; sudo is not used.'
  [[ "$vDomain" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ && "$vDomain" != *..* ]] || fFail '--domain must be a DNS hostname or IPv4 address.'
  [[ "$vBinary" =~ ^/[a-zA-Z0-9_./+-]+$ && -x "$vBinary" ]] || fFail '--binary must identify an executable using a plain absolute path.'
  [[ -r /etc/os-release ]] || fFail 'Cannot identify the distribution.'
  . /etc/os-release
  case "${ID:-}" in
    debian|ubuntu|alpine) ;;
    *) fFail 'Only Debian, Ubuntu and Alpine are supported.' ;;
  esac
  [[ -z "$vCertificate" && -z "$vPrivateKey" || -r "$vCertificate" && -r "$vPrivateKey" ]] || fFail 'Supply both a readable certificate and a readable private key.'
  umask 077
  touch /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot create installation records.'
  chmod 600 /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot protect installation records.'
  exec 3>&1 4>&2
  exec > >(tee -a /root/webapp-install.log) 2>&1
  vLogProcess=$!
  vLogEnabled=true
  printf '\nConfiguring automatic HTTP/HTTPS ports for %s at %s.\n' "$vDomain" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  case "$ID" in
    debian|ubuntu)
      env DEBIAN_FRONTEND=noninteractive apt-get update || fFail 'Cannot update package lists.'
      env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apache2-bin openssl iproute2 util-linux || fFail 'Cannot install the web service dependencies.'
      if ! id moe-gguf-server >/dev/null 2>&1; then
        useradd --system --user-group --create-home --home-dir /opt/moe-gguf-server --shell /usr/sbin/nologin moe-gguf-server || fFail 'Cannot create the service account.'
      fi
      # The Vulkan backend opens the GPU render nodes in /dev/dri; CUDA devices need no group.
      for vGroup in render video; do
        if getent group "$vGroup" >/dev/null 2>&1; then
          usermod -a -G "$vGroup" moe-gguf-server || fFail "Cannot add the service account to the ${vGroup} group."
        fi
      done
      vModuleDirectory=/usr/lib/apache2/modules
      ;;
    alpine)
      apk add --no-cache apache2 apache2-ssl apache2-proxy openssl iproute2 flock || fFail 'Cannot install the web service dependencies.'
      if ! id moe-gguf-server >/dev/null 2>&1; then
        addgroup -S moe-gguf-server || fFail 'Cannot create the service group.'
        adduser -S -D -h /opt/moe-gguf-server -s /sbin/nologin -G moe-gguf-server moe-gguf-server || fFail 'Cannot create the service account.'
      fi
      for vGroup in render video; do
        # BusyBox addgroup fails when the account is already a member, so check first (idempotent reruns).
        if getent group "$vGroup" >/dev/null 2>&1 && ! id -nG moe-gguf-server | tr ' ' '\n' | grep -Fxq "$vGroup"; then
          addgroup moe-gguf-server "$vGroup" || fFail "Cannot add the service account to the ${vGroup} group."
        fi
      done
      vModuleDirectory=/usr/lib/apache2
      ;;
  esac
  install -d -o moe-gguf-server -g moe-gguf-server -m 0750 /opt/moe-gguf-server /opt/moe-gguf-server/models || fFail 'Cannot create model storage.'
  install -d -m 0755 "/var/www/${vDomain}" || fFail 'Cannot create the Apache document directory.'
  install -d -o moe-gguf-server -g moe-gguf-server -m 0750 "/var/www/${vDomain}-logs" \
    /opt/moe-gguf-server/_ /opt/moe-gguf-server/_/temp /opt/moe-gguf-server/_/temp/web || fFail 'Cannot create web runtime directories.'
  for vLogFile in "/var/www/${vDomain}-logs/error.log" "/var/www/${vDomain}-logs/access.log"; do
    touch "$vLogFile" || fFail 'Cannot create the Apache log file.'
    chown moe-gguf-server:moe-gguf-server "$vLogFile" || fFail 'Cannot set web log ownership.'
    chmod 640 "$vLogFile" || fFail 'Cannot protect the web log.'
  done
  install -d -o moe-gguf-server -g moe-gguf-server -m 0700 /opt/moe-gguf-server/.mgs "$cConfigurationDirectory" ||
    fFail 'Cannot create configuration directory.'
  if [[ ! -s "${cConfigurationDirectory}/api.key" ]]; then
    openssl rand -hex 32 > "${cConfigurationDirectory}/api.key" || fFail 'Cannot generate the API key.'
  fi
  chown moe-gguf-server:moe-gguf-server "${cConfigurationDirectory}/api.key" || fFail 'Cannot set API key ownership.'
  chmod 600 "${cConfigurationDirectory}/api.key" || fFail 'Cannot protect the API key.'
  vApiKey="$(cat "${cConfigurationDirectory}/api.key")" || fFail 'Cannot read the API key.'
  sed -i '/^moe-gguf-server API key:/d' /root/webapp-credentials.txt || fFail 'Cannot update credential records.'
  printf 'moe-gguf-server API key: %s\n' "$vApiKey" >> /root/webapp-credentials.txt || fFail 'Cannot record the API key.'

  # The certificate lives with the service account: a self-signed one is created once and every reinstall reuses
  # it, so browsers keep their trust exception.
  install -d -o moe-gguf-server -g moe-gguf-server -m 0750 /opt/moe-gguf-server/certificates || fFail 'Cannot create certificate directory.'
  if [[ -n "$vCertificate" ]]; then
    if [[ "$(realpath "$vCertificate")" != /opt/moe-gguf-server/certificates/fullchain.pem ]]; then
      install -m 0644 "$vCertificate" /opt/moe-gguf-server/certificates/fullchain.pem || fFail 'Cannot install the certificate.'
    fi
    if [[ "$(realpath "$vPrivateKey")" != /opt/moe-gguf-server/certificates/privkey.pem ]]; then
      install -m 0600 "$vPrivateKey" /opt/moe-gguf-server/certificates/privkey.pem || fFail 'Cannot install the private key.'
    fi
  elif [[ ! -s /opt/moe-gguf-server/certificates/fullchain.pem || ! -s /opt/moe-gguf-server/certificates/privkey.pem ]]; then
    vSubjectAltName="DNS:${vDomain}"
    [[ ! "$vDomain" =~ ^[0-9.]+$ ]] || vSubjectAltName="IP:${vDomain}"
    openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365 \
      -subj "/CN=${vDomain}" -addext "subjectAltName=${vSubjectAltName}" \
      -keyout /opt/moe-gguf-server/certificates/privkey.pem \
      -out /opt/moe-gguf-server/certificates/fullchain.pem || fFail 'Cannot create the certificate.'
  fi
  chown moe-gguf-server:moe-gguf-server /opt/moe-gguf-server/certificates/privkey.pem \
    /opt/moe-gguf-server/certificates/fullchain.pem || fFail 'Cannot set certificate ownership.'
  chmod 600 /opt/moe-gguf-server/certificates/privkey.pem || fFail 'Cannot protect the private key.'
  chmod 644 /opt/moe-gguf-server/certificates/fullchain.pem || fFail 'Cannot set certificate permissions.'
  sed -e "s/@DOMAIN@/${vDomain}/g" -e "s|@MODULE_DIRECTORY@|${vModuleDirectory}|g" \
    "${vScriptDirectory}/apache2.conf.in" > "${cConfigurationDirectory}/apache2.conf.in" || fFail 'Cannot write the isolated Apache template.'
  printf '%s\n' "$vDomain" > "${cConfigurationDirectory}/web-domain" || fFail 'Cannot record the web domain.'
  chown moe-gguf-server:moe-gguf-server "${cConfigurationDirectory}/apache2.conf.in" "${cConfigurationDirectory}/web-domain" ||
    fFail 'Cannot set web configuration ownership.'
  chmod 600 "${cConfigurationDirectory}/apache2.conf.in" "${cConfigurationDirectory}/web-domain" ||
    fFail 'Cannot set web configuration permissions.'
  install -d -m 0755 /usr/local/libexec/moe-gguf-server || fFail 'Cannot create the service launcher directory.'
  install -m 0755 "${vScriptDirectory}/start-web.sh" /usr/local/libexec/moe-gguf-server/start-web.sh || fFail 'Cannot install the web launcher.'

  if [[ "$ID" == alpine ]]; then
    sed "s|@BINARY@|${vBinary}|g" "${vScriptDirectory}/moe-gguf-server.openrc.in" > /etc/init.d/moe-gguf-server || fFail 'Cannot write the OpenRC service.'
    chmod 755 /etc/init.d/moe-gguf-server || fFail 'Cannot enable the OpenRC script.'
    install -m 0755 "${vScriptDirectory}/moe-gguf-server-web.openrc.in" /etc/init.d/moe-gguf-server-web || fFail 'Cannot install the web OpenRC service.'
    rc-update add moe-gguf-server default || fFail 'Cannot enable the inference service.'
    rc-update add moe-gguf-server-web default || fFail 'Cannot enable the web service.'
    rc-service moe-gguf-server restart || fFail 'Cannot restart the inference service.'
    rc-service moe-gguf-server-web restart || fFail 'Cannot restart the web service.'
  else
    sed "s|@BINARY@|${vBinary}|g" "${vScriptDirectory}/moe-gguf-server.service.in" > /etc/systemd/system/moe-gguf-server.service || fFail 'Cannot write the systemd service.'
    chmod 644 /etc/systemd/system/moe-gguf-server.service || fFail 'Cannot set service permissions.'
    install -m 0644 "${vScriptDirectory}/moe-gguf-server-web.service.in" /etc/systemd/system/moe-gguf-server-web.service || fFail 'Cannot install the web systemd service.'
    systemctl daemon-reload || fFail 'Cannot reload systemd.'
    systemctl enable moe-gguf-server moe-gguf-server-web || fFail 'Cannot enable services.'
    systemctl restart moe-gguf-server || fFail 'Cannot restart the inference service.'
    systemctl restart moe-gguf-server-web || fFail 'Cannot restart the web service.'
  fi
  fWaitForWebPorts
  printf '%s\n' 'Live web addresses: /opt/moe-gguf-server/_/temp/web/ports.json'
  printf '%s\n' 'Credentials: /root/webapp-credentials.txt (root only).'
}

if ! fMain "$@"; then
  printf '%s\n' 'Web configuration failed.' >&2
  exit 1
fi
