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

fShowHelp() {
  printf '%s\n' \
    'Configure the non-root inference and web services of moe-gguf-server.' \
    'The installers run this script after installing mgs; run it alone to reconfigure without rebuilding.' \
    'Every web startup selects HTTP from 11080 and HTTPS from 11443, incrementing occupied ports.' \
    'Apache reaches the inference service through the Unix socket /opt/moe-gguf-server/_/temp/web/mgs.sock.' \
    '' \
    'Run as root: deploy/configure-web.sh [options]' \
    '  --domain NAME              Web domain (default: hostname -f, or hostname if that fails).' \
    '  --certificate PATH         Certificate chain (PEM); requires --private-key.' \
    '  --private-key PATH         Private key (PEM); requires --certificate.' \
    '  --proxy-ip IP              Reverse proxy in front of Apache (IPv4 or IPv6). HTTPS requires the PROXY' \
    '                             protocol only from this address and HTTP trusts its X-Forwarded-For header;' \
    '                             every other address keeps connecting directly. Without it, no proxy is set.' \
    '  --binary PATH              mgs executable (default: /usr/local/bin/mgs).' \
    '  --no-system-deps           Do not install Apache, OpenSSL, iproute2 or flock.' \
    '  --check                    Validate the options and exit without changing anything.' \
    '  -h, --help                 Show this help.' \
    '' \
    'The certificate lives in /opt/moe-gguf-server/certificates/: supplied files are copied there; without them, the' \
    'certificate already there is kept, or a self-signed certificate is created once and reused by every reinstall.' \
    'The settings are kept in /opt/moe-gguf-server/.mgs/config/ (api.key, web-domain, web-proxy-ip, apache2.conf.in).' \
    'Records: /root/webapp-install.log and /root/webapp-credentials.txt (mode 600).'
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

fIsValidDomain() {
  local pDomain="$1"
  [[ "$pDomain" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ && "$pDomain" != *..* ]]
}

fDetectDomain() {
  local vDomain
  vDomain="$(hostname -f 2>/dev/null)" || vDomain=''
  if ! fIsValidDomain "$vDomain"; then
    # uname -n prints the same host name when the hostname command is not installed.
    vDomain="$(hostname 2>/dev/null || uname -n)" || vDomain=''
  fi
  fIsValidDomain "$vDomain" || return 1
  printf '%s\n' "$vDomain"
}

fNormalizeIpv4() {
  local pAddress="$1"
  local vOctet vNormalized=''
  local -a aOctets=()
  [[ "$pAddress" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || return 1
  IFS=. read -r -a aOctets < <(printf '%s\n' "$pAddress")
  for vOctet in "${aOctets[@]}"; do
    [[ $((10#$vOctet)) -le 255 ]] || return 1
    vNormalized+="${vNormalized:+.}$((10#$vOctet))"
  done
  printf '%s\n' "$vNormalized"
}

fExpandIpv6() {
  # Prints the 32 hexadecimal digits of an IPv6 address, or fails if the text is not one.
  local pAddress="$1"
  local vIpv4 vHead vTail vGroup vPart vIndex vExpanded=''
  local -a aOctets=() aHead=() aTail=() aGroups=()
  [[ "$pAddress" =~ ^[0-9A-Fa-f:.]+$ && "$pAddress" == *:* && "$pAddress" != *:::* ]] || return 1
  # An embedded IPv4 address (::ffff:192.0.2.1) becomes the last two groups.
  if [[ "$pAddress" == *.* ]]; then
    vIpv4="$(fNormalizeIpv4 "${pAddress##*:}")" || return 1
    IFS=. read -r -a aOctets < <(printf '%s\n' "$vIpv4")
    printf -v vPart '%x:%x' "$((aOctets[0] * 256 + aOctets[1]))" "$((aOctets[2] * 256 + aOctets[3]))"
    pAddress="${pAddress%:*}:${vPart}"
  fi
  if [[ "$pAddress" == *::* ]]; then
    vHead="${pAddress%%::*}"
    vTail="${pAddress#*::}"
    [[ "$vTail" != *::* && "$vHead" != :* && "$vTail" != *: ]] || return 1
    IFS=: read -r -a aHead < <(printf '%s\n' "$vHead")
    IFS=: read -r -a aTail < <(printf '%s\n' "$vTail")
    [[ $((${#aHead[@]} + ${#aTail[@]})) -le 7 ]] || return 1
    aGroups=("${aHead[@]}")
    for ((vIndex = ${#aHead[@]} + ${#aTail[@]}; vIndex < 8; vIndex++)); do
      aGroups+=(0)
    done
    aGroups+=("${aTail[@]}")
  else
    [[ "$pAddress" != :* && "$pAddress" != *: ]] || return 1
    IFS=: read -r -a aGroups < <(printf '%s\n' "$pAddress")
    [[ ${#aGroups[@]} -eq 8 ]] || return 1
  fi
  for vGroup in "${aGroups[@]}"; do
    [[ "$vGroup" =~ ^[0-9A-Fa-f]{1,4}$ ]] || return 1
    printf -v vPart '%04x' "$((16#$vGroup))"
    vExpanded+="$vPart"
  done
  printf '%s\n' "$vExpanded"
}

fCompressIpv6() {
  # Writes 32 hexadecimal digits in the RFC 5952 text form (the longest run of zero groups becomes "::").
  local pHex="$1"
  local vIndex vGroup vBestStart=-1 vBestLength=0 vRunStart=-1 vRunLength=0 vText=''
  local -a aGroups=()
  for ((vIndex = 0; vIndex < 8; vIndex++)); do
    printf -v vGroup '%x' "$((16#${pHex:vIndex*4:4}))"
    aGroups+=("$vGroup")
    if [[ "$vGroup" == 0 ]]; then
      if [[ "$vRunStart" -lt 0 ]]; then
        vRunStart="$vIndex"
        vRunLength=0
      fi
      vRunLength=$((vRunLength + 1))
      if [[ "$vRunLength" -gt "$vBestLength" ]]; then
        vBestStart="$vRunStart"
        vBestLength="$vRunLength"
      fi
    else
      vRunStart=-1
    fi
  done
  if [[ "$vBestLength" -lt 2 ]]; then
    vBestStart=-1
  fi
  for ((vIndex = 0; vIndex < 8; vIndex++)); do
    if [[ "$vIndex" -eq "$vBestStart" ]]; then
      vText+='::'
      vIndex=$((vIndex + vBestLength - 1))
      continue
    fi
    if [[ -n "$vText" && "$vText" != *:: ]]; then
      vText+=':'
    fi
    vText+="${aGroups[vIndex]}"
  done
  printf '%s\n' "$vText"
}

fHexToBits() {
  local pHex="$1"
  local vIndex vNibble vBits=''
  for ((vIndex = 0; vIndex < ${#pHex}; vIndex++)); do
    vNibble=$((16#${pHex:vIndex:1}))
    vBits+="$(((vNibble >> 3) & 1))$(((vNibble >> 2) & 1))$(((vNibble >> 1) & 1))$((vNibble & 1))"
  done
  printf '%s\n' "$vBits"
}

fBitsToHex() {
  local pBits="$1"
  local vIndex vNibble vHex=''
  for ((vIndex = 0; vIndex < ${#pBits}; vIndex += 4)); do
    printf -v vNibble '%x' "$((2#${pBits:vIndex:4}))"
    vHex+="$vNibble"
  done
  printf '%s\n' "$vHex"
}

fComplementNetworks() {
  # Every address of the family except one: for each bit, the network that shares the earlier bits and differs in it.
  local pBits="$1"
  local pFamily="$2"
  local vLength="${#pBits}" vIndex vOctet vFlipped vZeros vNetwork vHex vAddress vNetworks=''
  for ((vIndex = 0; vIndex < vLength; vIndex++)); do
    if [[ "${pBits:vIndex:1}" == 1 ]]; then
      vFlipped=0
    else
      vFlipped=1
    fi
    printf -v vZeros '%*s' "$((vLength - vIndex - 1))" ''
    vNetwork="${pBits:0:vIndex}${vFlipped}${vZeros// /0}"
    if [[ "$pFamily" == 4 ]]; then
      vAddress=''
      for ((vOctet = 0; vOctet < 32; vOctet += 8)); do
        vAddress+="${vAddress:+.}$((2#${vNetwork:vOctet:8}))"
      done
    else
      vHex="$(fBitsToHex "$vNetwork")"
      # Apache rejects IPv4-mapped networks (::ffff:0:0/96); IPv4 clients are matched by the IPv4 networks.
      [[ "$vHex" != 00000000000000000000ffff* ]] || continue
      vAddress="$(fCompressIpv6 "$vHex")"
    fi
    vNetworks+="${vNetworks:+ }${vAddress}/$((vIndex + 1))"
  done
  printf '%s\n' "$vNetworks"
}

fProxyExceptions() {
  # Networks exempt from the PROXY protocol: every IPv4 and IPv6 address except the proxy.
  local pProxyIp="$1"
  local pFamily="$2"
  local vHex vBits vNetworks
  local -a aOctets=()
  if [[ "$pFamily" == 4 ]]; then
    IFS=. read -r -a aOctets < <(printf '%s\n' "$pProxyIp")
    printf -v vHex '%02x%02x%02x%02x' "${aOctets[0]}" "${aOctets[1]}" "${aOctets[2]}" "${aOctets[3]}"
    vBits="$(fHexToBits "$vHex")"
    vNetworks="$(fComplementNetworks "$vBits" 4)"
    # Apache accepts no /0 network, so all of IPv6 is written as its two halves.
    printf '%s ::/1 8000::/1\n' "$vNetworks"
  else
    vHex="$(fExpandIpv6 "$pProxyIp")"
    vBits="$(fHexToBits "$vHex")"
    vNetworks="$(fComplementNetworks "$vBits" 6)"
    printf '%s 0.0.0.0/1 128.0.0.0/1\n' "$vNetworks"
  fi
}

fMain() {
  local vDomain='' vCertificate='' vPrivateKey='' vBinary=/usr/local/bin/mgs vProxyIp='' vProxyFamily=''
  local vInstallDependencies=true vCheckOnly=false
  local vScriptDirectory vApiKey vModuleDirectory vSubjectAltName vLogFile vGroup vCandidate vProxyHex vProxyExceptions
  local -a aTemplateArguments=() aWebSettings=()
  vScriptDirectory="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || fFail 'Cannot locate deployment templates.'
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain|--certificate|--private-key|--binary|--proxy-ip)
        [[ $# -ge 2 ]] || fFail "Missing value after $1."
        case "$1" in
          --domain) vDomain="$2" ;;
          --certificate) vCertificate="$2" ;;
          --private-key) vPrivateKey="$2" ;;
          --binary) vBinary="$2" ;;
          --proxy-ip) vProxyIp="$2" ;;
        esac
        shift 2
        ;;
      --no-system-deps) vInstallDependencies=false; shift ;;
      --check) vCheckOnly=true; shift ;;
      -h|--help)
        fShowHelp
        return 0
        ;;
      *) fFail "Unknown option: $1" ;;
    esac
  done
  [[ $EUID -eq 0 ]] || fFail 'Run this script as root; sudo is not used.'
  if [[ -z "$vDomain" ]]; then
    vDomain="$(fDetectDomain)" || fFail 'Cannot derive the web domain from hostname -f or hostname; pass --domain.'
  fi
  fIsValidDomain "$vDomain" || fFail '--domain must be a DNS hostname or IPv4 address.'
  [[ -r /etc/os-release ]] || fFail 'Cannot identify the distribution.'
  . /etc/os-release
  case "${ID:-}" in
    debian|ubuntu|alpine) ;;
    *) fFail 'Only Debian, Ubuntu and Alpine are supported.' ;;
  esac
  [[ -z "$vCertificate" && -z "$vPrivateKey" || -r "$vCertificate" && -r "$vPrivateKey" ]] || fFail 'Supply both a readable certificate and a readable private key.'
  if [[ -n "$vProxyIp" ]]; then
    if vCandidate="$(fNormalizeIpv4 "$vProxyIp")"; then
      vProxyIp="$vCandidate"
      vProxyFamily=4
    elif vProxyHex="$(fExpandIpv6 "$vProxyIp")"; then
      if [[ "$vProxyHex" == 00000000000000000000ffff* ]]; then
        # An IPv4-mapped address is configured as the IPv4 address it carries.
        vProxyIp="$((16#${vProxyHex:24:2})).$((16#${vProxyHex:26:2})).$((16#${vProxyHex:28:2})).$((16#${vProxyHex:30:2}))"
        vProxyFamily=4
      else
        vProxyIp="$(fCompressIpv6 "$vProxyHex")"
        vProxyFamily=6
      fi
    else
      fFail '--proxy-ip must be an IPv4 or IPv6 address.'
    fi
  fi
  if [[ "$vCheckOnly" == true ]]; then
    if [[ -n "$vProxyIp" ]]; then
      printf 'Web options are valid: domain %s, trusted proxy %s.\n' "$vDomain" "$vProxyIp"
    else
      printf 'Web options are valid: domain %s, no proxy.\n' "$vDomain"
    fi
    return 0
  fi
  [[ "$vBinary" =~ ^/[a-zA-Z0-9_./+-]+$ && -x "$vBinary" ]] || fFail '--binary must identify an executable using a plain absolute path.'
  umask 077
  touch /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot create installation records.'
  chmod 600 /root/webapp-install.log /root/webapp-credentials.txt || fFail 'Cannot protect installation records.'
  # The installers already copy everything they print, this script included, into the installation log.
  if [[ "${MGS_INSTALLER_PARENT_LOG:-false}" != true ]]; then
    exec 3>&1 4>&2
    exec > >(tee -a /root/webapp-install.log) 2>&1
    vLogProcess=$!
    vLogEnabled=true
  fi
  printf '\nConfiguring automatic HTTP/HTTPS ports for %s at %s.\n' "$vDomain" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  case "$ID" in
    debian|ubuntu)
      if [[ "$vInstallDependencies" == true ]]; then
        env DEBIAN_FRONTEND=noninteractive apt-get update || fFail 'Cannot update package lists.'
        env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apache2-bin openssl iproute2 util-linux || fFail 'Cannot install the web service dependencies.'
      fi
      if ! id moe-gguf-server >/dev/null 2>&1; then
        useradd --system --user-group --create-home --home-dir /opt/moe-gguf-server --shell /usr/sbin/nologin moe-gguf-server || fFail 'Cannot create the service account.'
      fi
      # The ROCm and Vulkan backends open /dev/kfd and the GPU render nodes in /dev/dri; CUDA devices need no group.
      for vGroup in render video; do
        if getent group "$vGroup" >/dev/null 2>&1; then
          usermod -a -G "$vGroup" moe-gguf-server || fFail "Cannot add the service account to the ${vGroup} group."
        fi
      done
      vModuleDirectory=/usr/lib/apache2/modules
      ;;
    alpine)
      if [[ "$vInstallDependencies" == true ]]; then
        apk add --no-cache apache2 apache2-ssl apache2-proxy openssl iproute2 flock || fFail 'Cannot install the web service dependencies.'
      fi
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
  for vCandidate in openssl ss flock; do
    command -v "$vCandidate" >/dev/null 2>&1 || fFail "Required command not found: ${vCandidate}."
  done
  [[ -x /usr/sbin/apache2 || -x /usr/sbin/httpd ]] || fFail 'Apache is not installed.'
  for vCandidate in mod_ssl.so mod_proxy.so mod_proxy_http.so; do
    [[ -f "${vModuleDirectory}/${vCandidate}" ]] || fFail "The Apache module ${vModuleDirectory}/${vCandidate} is missing."
  done
  if [[ -n "$vProxyIp" ]]; then
    [[ -f "${vModuleDirectory}/mod_remoteip.so" ]] || fFail "--proxy-ip needs the Apache module ${vModuleDirectory}/mod_remoteip.so."
  fi
  install -d -o moe-gguf-server -g moe-gguf-server -m 0750 /opt/moe-gguf-server /opt/moe-gguf-server/models || fFail 'Cannot create model storage.'
  install -d -m 0755 "/var/www/${vDomain}" || fFail 'Cannot create the Apache document directory.'
  # The web runtime directory also holds the Unix socket where the inference service listens for Apache.
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
  # Every credential created by the deployment, rewritten in place on each run.
  sed -i -e '/^moe-gguf-server API key:/d' -e '/^moe-gguf-server system account:/d' /root/webapp-credentials.txt ||
    fFail 'Cannot update credential records.'
  printf '%s\n' 'moe-gguf-server system account: moe-gguf-server (no password; login disabled)' \
    "moe-gguf-server API key: ${vApiKey}" >> /root/webapp-credentials.txt || fFail 'Cannot record the API key.'

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

  # Template lines that start with @PROXY@ are kept only with --proxy-ip; the result persists the choice, and
  # start-web.sh renders it again with the selected ports on every start.
  aTemplateArguments=(-e "s/@DOMAIN@/${vDomain}/g" -e "s|@MODULE_DIRECTORY@|${vModuleDirectory}|g")
  aWebSettings=("${cConfigurationDirectory}/apache2.conf.in" "${cConfigurationDirectory}/web-domain")
  if [[ -n "$vProxyIp" ]]; then
    vProxyExceptions="$(fProxyExceptions "$vProxyIp" "$vProxyFamily")" || fFail 'Cannot compute the addresses exempt from the PROXY protocol.'
    aTemplateArguments+=(-e 's/^@PROXY@//' -e "s|@PROXY_IP@|${vProxyIp}|g" -e "s|@PROXY_EXCEPTIONS@|${vProxyExceptions}|g")
    printf '%s\n' "$vProxyIp" > "${cConfigurationDirectory}/web-proxy-ip" || fFail 'Cannot record the proxy address.'
    aWebSettings+=("${cConfigurationDirectory}/web-proxy-ip")
    printf 'Trusted proxy: %s (PROXY protocol on HTTPS, X-Forwarded-For on HTTP).\n' "$vProxyIp"
  else
    aTemplateArguments+=(-e '/^@PROXY@/d')
    rm -f -- "${cConfigurationDirectory}/web-proxy-ip" || fFail 'Cannot remove the previous proxy address.'
  fi
  sed "${aTemplateArguments[@]}" "${vScriptDirectory}/apache2.conf.in" > "${cConfigurationDirectory}/apache2.conf.in" ||
    fFail 'Cannot write the isolated Apache template.'
  printf '%s\n' "$vDomain" > "${cConfigurationDirectory}/web-domain" || fFail 'Cannot record the web domain.'
  chown moe-gguf-server:moe-gguf-server "${aWebSettings[@]}" || fFail 'Cannot set web configuration ownership.'
  chmod 600 "${aWebSettings[@]}" || fFail 'Cannot set web configuration permissions.'
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
