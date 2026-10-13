#!/bin/bash

set -euo pipefail

readonly cConfigurationDirectory=/opt/moe-gguf-server/.mgs/config
readonly cRuntimeDirectory=/opt/moe-gguf-server/_/temp/web
readonly cHttpStart=11080
readonly cHttpsStart=11443
readonly cLastPort=65535
vApachePid=''
vStopping=false
vOwnsLock=false

fCleanup() {
  if [[ -n "$vApachePid" ]]; then
    kill -TERM "$vApachePid" 2>/dev/null || true
    wait "$vApachePid" 2>/dev/null || true
  fi
  if [[ "$vOwnsLock" == true ]]; then
    rm -f -- "${cRuntimeDirectory}/ports.json" "${cRuntimeDirectory}/ports.json.next" || true
  fi
}

trap fCleanup EXIT

fStop() {
  vStopping=true
  if [[ -n "$vApachePid" ]]; then
    kill -TERM "$vApachePid" 2>/dev/null || true
  fi
}

trap fStop TERM INT

fFail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

fReadOccupiedPorts() {
  local vSockets vState vReceiveQueue vSendQueue vAddress vPeer vRest vPort
  vSockets="$(ss -H -ltn)" || fFail 'Cannot inspect IPv4/IPv6 TCP listeners.'
  dOccupiedPorts=()
  while read -r vState vReceiveQueue vSendQueue vAddress vPeer vRest; do
    vPort="${vAddress##*:}"
    if [[ "$vPort" =~ ^[0-9]+$ ]]; then
      dOccupiedPorts["$vPort"]=1
    fi
  done < <(printf '%s\n' "$vSockets")
}

fFindFreePort() {
  local pFirstPort="$1" pReservedPort="$2"
  local vCandidate
  for ((vCandidate=pFirstPort; vCandidate<=cLastPort; vCandidate++)); do
    if [[ "$vCandidate" != "$pReservedPort" && -z "${dOccupiedPorts[$vCandidate]+present}" && -z "${dRejectedPorts[$vCandidate]+present}" ]]; then
      printf '%s\n' "$vCandidate"
      return 0
    fi
  done
  printf 'No free TCP port from %s through %s.\n' "$pFirstPort" "$cLastPort" >&2
  return 1
}

fOwnsListener() {
  local pPort="$1"
  local vSockets
  vSockets="$(ss -H -ltnp "sport = :${pPort}")" || fFail 'Cannot inspect the web listener owner.'
  [[ "$vSockets" == *"pid=${vApachePid},"* ]]
}

fPublishPorts() {
  printf '{"http_port":%s,"https_port":%s,"http_url":"http://%s:%s/","https_url":"https://%s:%s/","pid":%s}\n' \
    "$vHttpPort" "$vHttpsPort" "$vDomain" "$vHttpPort" "$vDomain" "$vHttpsPort" "$vApachePid" \
    > "${cRuntimeDirectory}/ports.json.next" || fFail 'Cannot record the selected ports.'
  chmod 640 "${cRuntimeDirectory}/ports.json.next" || fFail 'Cannot protect the port record.'
  mv -f -- "${cRuntimeDirectory}/ports.json.next" "${cRuntimeDirectory}/ports.json" || fFail 'Cannot publish the selected ports.'
  printf 'HTTP: http://%s:%s/\nHTTPS: https://%s:%s/\nAPI documentation: https://%s:%s/api/doc/\n' \
    "$vDomain" "$vHttpPort" "$vDomain" "$vHttpsPort" "$vDomain" "$vHttpsPort"
}

fMain() {
  local vApacheBinary vDomain vHttpPort vHttpsPort vTick vReady vStatus vRejected
  local -A dOccupiedPorts=() dRejectedPorts=()
  [[ $EUID -ne 0 ]] || fFail 'Run the web service as moe-gguf-server, not root.'
  [[ "$(id -un)" == moe-gguf-server ]] || fFail 'The web service requires the moe-gguf-server account.'
  command -v ss >/dev/null 2>&1 || fFail 'Install iproute2 before starting the web service.'
  command -v flock >/dev/null 2>&1 || fFail 'Install flock before starting the web service.'
  if [[ -x /usr/sbin/apache2 ]]; then
    vApacheBinary=/usr/sbin/apache2
  elif [[ -x /usr/sbin/httpd ]]; then
    vApacheBinary=/usr/sbin/httpd
  else
    fFail 'Apache is not installed.'
  fi
  IFS= read -r vDomain < "${cConfigurationDirectory}/web-domain" || fFail 'Cannot read the web domain.'
  [[ "$vDomain" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ && "$vDomain" != *..* ]] || fFail 'Invalid web domain.'
  umask 077
  mkdir -p "$cRuntimeDirectory" || fFail 'Cannot create the web runtime directory.'
  exec 9>"${cRuntimeDirectory}/launcher.lock" || fFail 'Cannot open the web service lock.'
  flock -n 9 || fFail 'The application web service is already running.'
  vOwnsLock=true
  rm -f -- "${cRuntimeDirectory}/ports.json" || fFail 'Cannot clear the previous port record.'

  while [[ "$vStopping" == false ]]; do
    fReadOccupiedPorts
    vHttpsPort="$(fFindFreePort "$cHttpsStart" 0)" || fFail 'No HTTPS port is available.'
    vHttpPort="$(fFindFreePort "$cHttpStart" "$vHttpsPort")" || fFail 'No HTTP port is available.'
    sed -e "s/@HTTP_PORT@/${vHttpPort}/g" -e "s/@HTTPS_PORT@/${vHttpsPort}/g" \
      "${cConfigurationDirectory}/apache2.conf.in" > "${cRuntimeDirectory}/apache2.conf" || fFail 'Cannot render Apache configuration.'
    : > "${cRuntimeDirectory}/startup.log" || fFail 'Cannot initialize the startup log.'
    : > "${cRuntimeDirectory}/error.log" || fFail 'Cannot initialize the Apache diagnostic log.'
    printf 'Starting web service: HTTP %s, HTTPS %s.\n' "$vHttpPort" "$vHttpsPort"
    [[ "$vStopping" == false ]] || return 0
    LC_ALL=C "$vApacheBinary" -f "${cRuntimeDirectory}/apache2.conf" -DFOREGROUND \
      -E "${cRuntimeDirectory}/startup.log" 2>>"${cRuntimeDirectory}/startup.log" &
    vApachePid=$!
    vReady=false
    for ((vTick=0; vTick<300; vTick++)); do
      [[ "$vStopping" == false ]] || break
      kill -0 "$vApachePid" 2>/dev/null || break
      if fOwnsListener "$vHttpPort" && fOwnsListener "$vHttpsPort"; then
        vReady=true
        fPublishPorts
        break
      fi
      sleep 0.1 || true
    done
    if [[ "$vStopping" == false && "$vReady" == false ]] && kill -0 "$vApachePid" 2>/dev/null; then
      fFail 'Apache did not open both selected ports within 30 seconds.'
    fi
    vStatus=0
    wait "$vApachePid" || vStatus=$?
    # A signal can interrupt wait before the child has terminated.
    if [[ "$vStopping" == true ]]; then
      wait "$vApachePid" 2>/dev/null || true
      vApachePid=''
      return 0
    fi
    vApachePid=''
    rm -f -- "${cRuntimeDirectory}/ports.json" || fFail 'Cannot clear the port record.'
    if [[ "$vReady" == true ]]; then
      fFail "Apache stopped unexpectedly (status ${vStatus})."
    fi
    # Apache may switch from its startup log to ErrorLog before binding listeners.
    cat "${cRuntimeDirectory}/error.log" >> "${cRuntimeDirectory}/startup.log" || fFail 'Cannot collect Apache startup diagnostics.'
    if ! grep -Eq 'Address already in use|\(98\).*AH00072' "${cRuntimeDirectory}/startup.log"; then
      cat "${cRuntimeDirectory}/startup.log" >&2 || true
      fFail "Apache startup failed (status ${vStatus}); see the startup log."
    fi
    # The socket snapshot is advisory: retry only ports rejected by the actual bind.
    vRejected=false
    if grep -Eq "AH00072:.*address.*:${vHttpPort}([^0-9]|$)" "${cRuntimeDirectory}/startup.log"; then
      dRejectedPorts["$vHttpPort"]=1
      vRejected=true
    fi
    if grep -Eq "AH00072:.*address.*:${vHttpsPort}([^0-9]|$)" "${cRuntimeDirectory}/startup.log"; then
      dRejectedPorts["$vHttpsPort"]=1
      vRejected=true
    fi
    [[ "$vRejected" == true ]] || fFail 'Cannot identify the conflicting Apache listener.'
    printf '%s\n' 'A selected port became busy during startup; selecting the next free port.'
  done
}

if ! fMain "$@"; then
  printf '%s\n' 'Web service startup failed.' >&2
  exit 1
fi
