# Benutzerhandbuch

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Inhalt: [Installation](#installation), [lokaler Build](#local-build), [Bereitstellung](#deployment),
[Chat](#chat), [Monitor](#monitor), [Modelle und Werkzeuge](#models), [unterstützte Modelle](#supported-models), [API](#api),
[Anfragewarteschlange](#request-queue), [Sicherheitsgrenzen](#security-limits), [Vulkan](#vulkan), [Leistung](#performance), [entfernte Kompatibilität](#removed-compatibility),
[Wartung](#maintenance), [Lizenz](#license).

<a id="installation"></a>
## Installation und erster Start

Installieren Sie als root auf dem Produktionsrechner. Unter Debian/Ubuntu kann das
Installationsprogramm direkt über die Standardeingabe übergeben werden:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

Es müssen vorher keine weiteren Dateien heruntergeladen werden. Das Einstiegsskript lädt das
vollständige Quellarchiv von `main` aus [diesem Repository](https://github.com/nipegun/moe-gguf-server/) herunter, prüft
es und synchronisiert eine verwaltete Kopie in `/opt/moe-gguf-server-source`. Downloads
und Build-Caches bleiben unter `_/temp/` des Projekts. Curl und CA-Zertifikate
werden benötigt, um das Einstiegsskript abzurufen; fehlende Bootstrap-Abhängigkeiten werden anschließend
automatisch installiert.

Ein anderes Quellverzeichnis wählen und Build-Optionen übergeben Sie so:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

Das entfernte Verzeichnis muss absolut angegeben sein und auf einem lokalen Datenträger liegen. Das Installationsprogramm
übernimmt nur ein leeres Verzeichnis oder verwendet eine Kopie wieder, die seine Herkunftsmarkierung trägt;
fremde Verzeichnisse werden abgelehnt. Eine Sperre verhindert gleichzeitige Installationen in
diese Kopie. Spätere `curl | bash`-Läufe synchronisieren die Quelldateien, einschließlich der Dateien,
die aus `main` entfernt wurden, und erhalten dabei `_/` und die Build-Caches. API-Schlüssel, Modelle und
bereitgestellte Dienste liegen außerhalb dieses Quellbaums und werden nicht verändert. Ein fehlgeschlagener
Download oder Build installiert keine neue ausführbare Datei; führen Sie denselben Befehl erneut aus, um es
noch einmal zu versuchen. Es sind keine Git-Operationen erforderlich.

Ein eigenständig heruntergeladenes Skript verwendet denselben entfernten Modus. Wird
`bash deploy/install-update-reinstall-debian.sh` aus einer vollständigen lokalen Arbeitskopie ausgeführt,
kompiliert es diese Arbeitskopie, ohne Quellen herunterzuladen. Argumente nach `bash -s --`
werden an das Build-Installationsprogramm weitergereicht. `--help` benötigt weder root noch einen Download
der Quellen. `--no-system-deps` deaktiviert auch die Installation der Bootstrap-Pakete: curl,
CA-Zertifikate, tar, gzip, rsync, util-linux und coreutils müssen bereits vorhanden sein.
Unterprozesse des Installationsprogramms können das Skript nicht aus der Pipe lesen; alle Phasen teilen sich
einen gemeinsamen Ausgabestrom in `/root/webapp-install.log`.

Die Webkonfiguration bleibt ein eigener Schritt mit
`/opt/moe-gguf-server-source/deploy/configure-web.sh` oder dem entsprechenden Skript im
gewählten Quellverzeichnis.

Alpine verwendet weiterhin eine vollständige lokale Arbeitskopie und
`bash deploy/install-update-reinstall-alpine.sh`, wobei Bash zuvor installiert werden muss.
Debian verwendet standardmäßig CUDA/NCCL; Alpine verwendet standardmäßig die CPU und unterstützt den
Installationsweg mit CUDA/NCCL nicht. Verwenden Sie `--cpu` für einen CPU-Build unter Debian oder `--vulkan` für GPUs von AMD, Intel oder NVIDIA
über Vulkan ([Vulkan](#vulkan)). CUDA Toolkit und NVIDIA-Treiber müssen bereits installiert sein. `--no-nccl` deaktiviert NCCL ausdrücklich;
`--no-system-deps` überlässt die Paketverwaltung dem Administrator.

Die lokale Oberfläche benötigt Node.js 20.19+ in der Reihe 20.x oder 22.12+. `--no-ui` lässt sie weg. Weitere
Optionen sind `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST`, `--no-openssl` und `--build-ui`.
`MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`,
`MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND` und `CUDACXX` bieten entsprechende
Überschreibungen über Umgebungsvariablen. Build- und Staging-Verzeichnisse müssen verschiedene,
sich nicht überschneidende Unterverzeichnisse des Verzeichnisses `_/temp/` des Projekts sein.

Das Installationsprogramm fügt bei Bedarf passende NCCL-Pakete und das Repository von NVIDIA hinzu,
installiert oder aktualisiert aber niemals das CUDA Toolkit oder GPU-Treiber. Es bindet die internen
Bibliotheken statisch ein und installiert `/usr/local/bin/mgs`. Keines der beiden
Installationsprogramme verwendet oder installiert sudo.

Unter Debian/Ubuntu werden `libnccl2` und `libnccl-dev` gemeinsam in genau derselben
Version installiert, die für die erkannte CUDA-Version ausgewählt wurde. Dieser APT-Befehl verwendet
`--allow-downgrades`, damit ein neueres bereits installiertes NCCL-Paket die Installation
der ausgewählten Version nicht blockiert. Das Flag wird nur an den Installationsbefehl für NCCL übergeben.

Liefert die Distribution eine ältere Node.js-Version aus, installieren Sie zuerst eine unterstützte Version.
Die Oberfläche wird immer lokal kompiliert; der Build kann keine vorkompilierten Upstream-Assets
herunterladen, denen die Übersetzungen und API-Routen dieses Projekts fehlen. Ein fehlgeschlagener Build der Oberfläche
bricht die Installation ab.

Aus einer vollständigen lokalen Arbeitskopie wählen Sie einen CPU-Build oder lassen sich die Installationsoptionen auflisten:

```bash
bash deploy/install-update-reinstall-debian.sh --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

Installieren Sie unter Alpine Bash mit `apk add --no-cache bash`, bevor Sie das dortige Installationsprogramm ausführen.

Ein einzelnes Modell starten Sie mit:

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

Keine weitere Option ist nötig: `mgs` wählt die Einstellungen, die auf dem Rechner, auf dem es läuft, die höchste
Inferenzgeschwindigkeit liefern. Die automatische Anpassung legt das Modell auf die GPU und behält, wenn der Arbeitsspeicher
es zulässt, die MoE-Experten im RAM, sodass der freie VRAM zum [MoE-Expertencache](#expert-cache) wird; außerdem wählt sie
[den Micro-Batch und die Kontextgröße](#context-size). Die Anzahl der CPU-Threads richtet sich nach dem Prozessor
([CPU-Threads](#threads)), und die im GGUF gefundenen MTP-Schichten werden aktiviert und im VRAM gehalten
([MTP](#mtp-vram)). Jede ausdrücklich angegebene Option (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`,
`--override-tensor`…) hat Vorrang vor der automatischen Auswahl. Das Startprotokoll zeigt, was gewählt wurde.

Öffnen Sie die beim Start ausgegebene HTTPS-URL, normalerweise `https://127.0.0.1:11443/`.
Das Laden eines großen Modells kann eine Weile dauern. Die Weboberfläche
und die API-Dokumentation bleiben verfügbar, während Modellanfragen melden, dass
noch geladen wird. `--help` listet die geerbten Engine-Optionen auf. GGUF-Dateien, GPU-Speicher,
Kontextlänge und modellspezifische Projektoren müssen zur vorgesehenen Arbeitslast passen.

Der Servercode aktiviert HTTPS standardmäßig für jede Build- und Installationsmethode.
Er bindet den ersten freien HTTPS-Port ab 11443 und anschließend einen HTTP-Weiterleitungsport
ab 11080, wobei der gewählte HTTPS-Port übersprungen wird. HTTP antwortet mit einer permanenten
308-Weiterleitung auf den tatsächlichen HTTPS-Port und behält dabei den angeforderten Host, Pfad und die Abfrage bei.
Das Protokoll gibt die gewählten URLs aus; Portkonflikte lassen jede Folge unabhängig voneinander weiterlaufen.
`--port N` legt den Hauptport fest und schlägt fehl, wenn er belegt ist; mit `--port 0` wählt ihn das
Betriebssystem. `--http-port N` ändert den Startwert der Suche nach dem Weiterleitungsport.
Die entsprechenden Umgebungsvariablen sind `MGS_ARG_PORT` und `MGS_ARG_HTTP_PORT`.

Ohne Zertifikatsdateien erzeugt die ausführbare Datei beim ersten Start ein selbstsigniertes Zertifikat und
seinen privaten Schlüssel, ohne einen externen `openssl`-Befehl zu benötigen,
und speichert beide in `~/.mgs/certificates/` (`fullchain.pem` und
`privkey.pem`, wobei nur Sie den Schlüssel lesen können). Jeder spätere Lauf und jeder neue Build
verwendet sie wieder, sodass der Browser seine Vertrauensausnahme behält und nicht neu geladen werden muss; ein neues
Paar wird nur erzeugt, wenn diese Dateien fehlen, nicht zusammenpassen oder das Zertifikat
abgelaufen ist (nach 365 Tagen). Das Zertifikat deckt `localhost`, die Loopback-Adressen,
den mit `--host` angegebenen Host sowie den Rechnernamen und die Netzwerkadressen ab,
die bei seiner Erzeugung bestanden; um die Browserwarnung ganz zu vermeiden, importieren Sie
`~/.mgs/certificates/fullchain.pem` als vertrauenswürdig, und löschen Sie beide Dateien, um ein
neues zu erzeugen. Kann in diesen Ordner nicht geschrieben werden, erzeugt jeder Lauf sein eigenes Zertifikat im
Arbeitsspeicher. Geben Sie sowohl `--ssl-cert-file /path/fullchain.pem` als auch `--ssl-key-file
/path/privkey.pem` an, um ein Zertifikat zu verwenden, dem Ihre Clients vertrauen. Ein fehlendes oder ungültiges Paar bricht
den Start ab. Mit `--no-openssl` erstellte Builds können kein HTTPS bereitstellen und benötigen ausdrücklich
`--http` (oder `MGS_ARG_HTTPS=false`); reines HTTP beginnt seine Portsuche bei 11080.
Router-Kindprozesse verwenden ausdrücklich HTTP auf privaten Loopback-Ports. Unix-Sockets und
`AIP_MODE=PREDICTION` von Vertex AI behalten ihren HTTP-Transport.

Die CMake-Konfiguration befindet sich in `build/CMakeLists.txt`. Sowohl die Installations- als auch die lokalen
Build-Skripte nehmen `build/` in ihre Quellkopie auf und wählen es automatisch aus.
Die Installationsprogramme setzen die erzeugte CMake-Konfiguration zurück, wenn ein vorhandener Cache auf
ein anderes Quellverzeichnis verweist. Lokale Builds beginnen immer in einem frischen Verzeichnis unter
`/tmp/` und entfernen es beim Beenden. Für die direkte Verwendung von CMake führen Sie
`cmake -S build -B _/temp/cmake` im Stammverzeichnis des Projekts aus; die üblichen Befehle für Installation und
lokalen Build bleiben gleich. HTTPS verwendet standardmäßig das OpenSSL des Systems
(`MGS_OPENSSL=ON`); bei einem direkten CMake-Build lädt `-DMGS_BUILD_BORINGSSL=ON` oder
`-DMGS_BUILD_LIBRESSL=ON` stattdessen BoringSSL bzw. LibreSSL herunter und kompiliert es, und
`-DMGS_OPENSSL=OFF` ohne eine dieser Optionen erstellt den Server ohne HTTPS.

<a id="local-build"></a>
## Lokaler Build für die manuelle Verwendung

`build/build-local.sh` kompiliert das Projekt auf Ihrem eigenen Computer und installiert eine
eigenständige Kopie für Ihren Benutzer, die Sie von Hand starten. Es installiert keine Dienste, startet
nichts im Hintergrund, verwendet niemals root oder sudo und installiert niemals Pakete.
Führen Sie es als Ihr normaler Benutzer in einem beliebigen Ordner aus:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

Der Einzeiler lädt das Quellarchiv von `main` ohne Git in den privaten
Build-Ordner unter `/tmp` herunter. Das Archiv und die entpackten Quellen werden beim Beenden zusammen mit
den übrigen temporären Build-Dateien entfernt. Eine eigenständige Kopie des Skripts funktioniert
genauso. Aus einer vollständigen lokalen Kopie des Projekts kompilieren Sie Ihre lokalen Quellen mit:

```bash
bash build/build-local.sh
```

Wird dieses Skript über seinen absoluten Pfad ausgeführt, verwendet es ebenfalls das vollständige Projekt, in dem
es liegt. Die Meldung `Using local sources: ...` bestätigt, welcher Ordner
kompiliert wird, einschließlich der Übersetzungen seiner Oberfläche. Endet der Lauf mit
`Error: Compilation failed.`, wurde keine neue ausführbare Datei installiert; eine vorhandene
Installation behält ihre bisherige Oberfläche. Lesen Sie den ersten Compilerfehler oberhalb von
`ninja: build stopped`, beheben Sie seine Ursache und führen Sie dasselbe Skript erneut aus. Sobald
Build und Installation erfolgreich waren, starten Sie `mgs` neu und laden die Weboberfläche neu.

Fehler wie ein `std::vector` ohne Mitglied namens `mPushBack` oder
`no declaration matches` in `subproc.cpp` weisen auf inkonsistente Namen in den Quellen hin.
Verwenden Sie die korrigierten Quelldateien und führen Sie den Build erneut aus. Der Build der Oberfläche
setzt außerdem voraus, dass ihre Konfigurationsimporte zu den Namen passen, die die Quellmodule
exportieren; ein fehlender Export wird vom npm/Vite-Schritt gemeldet.

Meldet SvelteKit `Page options are ignored when router.type === 'hash'`
und nennt dabei `fLoad`, müssen die Chat-Routen genau den Namen `load` des Frameworks
exportieren. Die korrigierten Routen behalten die Prüfung des API-Schlüssels bei. Der Build führt
`svelte-kit sync` aus, bevor er Assets erzeugt und die Oberfläche kompiliert, und
bricht ab, wenn die Synchronisierung fehlschlägt. Ein Fehler bei `Provisioning UI assets` verhindert
die Installation auch dann, wenn die C++-Bibliotheken bereits kompiliert wurden.

Um Optionen über den Einzeiler zu übergeben, verwenden Sie `bash -s --`; zum Beispiel für einen CPU-Build:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` zeigt die Optionen an, ohne das Quellarchiv herunterzuladen oder zu kompilieren.

Das Ergebnis wird für Ihren Benutzer unter dem Präfix `~/.local` installiert:

| Pfad | Inhalt |
| --- | --- |
| `~/.local/bin/mgs` | Das Programm, das Sie ausführen: ein kleiner statischer Starter, der `../lib/mgs/` relativ zu sich selbst findet und den Server über den mitgelieferten Loader startet. |
| `~/.local/lib/mgs/` | Die Server-Binärdatei (`mgs`, mit eingebetteter Weboberfläche und API-Dokumentation) und jede gemeinsam genutzte Bibliothek, die sie benötigt: der dynamische Loader und die C-Bibliothek von glibc, die C++-Laufzeitbibliothek, OpenMP, OpenSSL, die Namensdienstmodule von glibc und in CUDA-Builds die CUDA-Laufzeit, cuBLAS und NCCL. |
| `~/.local/share/doc/mgs/licenses/` | Die Lizenz PolyForm NonCommercial 1.0.0 des Projekts (`LICENSE`) und die Lizenzhinweise Dritter, kopiert aus `build/licenses/` des Quellbaums (llama.cpp, FreeToken, Swagger UI, cpp-httplib und nlohmann/json). |

Nur der Benutzer, der das Skript ausführt, kann diese Installation verwenden. Alles wird relativ zum Ort des Starters
aufgelöst: Er führt `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` aus seinem Präfix aus, und der Server und seine Bibliotheken durchsuchen
`lib/mgs/` vor allem anderen, einschließlich `LD_LIBRARY_PATH`. Die einzige Bibliothek, die weiterhin vom System genommen wird, ist der
NVIDIA-Treiber (`libcuda.so.1` und die von ihm geladenen `libnvidia-*`-Bibliotheken), weil er zum laufenden Kernelmodul
passen muss; mit `--vulkan` außerdem der Vulkan-Treiber des Systems ([Vulkan](#vulkan)). Am Ende listet das Skript über den mitgelieferten Loader die Bibliotheken auf, die der Server laden würde, schlägt fehl, wenn eine davon
von außerhalb von `lib/mgs/` stammt, und führt `--version` über den Starter aus.

Debian fügt `~/.local/bin` bei der Anmeldung zum `PATH` hinzu, sobald dieser Ordner existiert (über `~/.profile`), sodass Sie nach einer erneuten
Anmeldung `mgs` über seinen Namen ausführen können; das Skript gibt einen Hinweis aus, solange es nicht im `PATH` liegt. Starten Sie es, wann immer Sie
möchten, und beenden Sie es mit Strg+C:

```bash
mgs --model /path/model.gguf
```

Der vollständige Pfad funktioniert ebenfalls: `~/.local/bin/mgs --model /path/model.gguf`.

Öffnen Sie `https://127.0.0.1:11443/` für den Chat und `https://127.0.0.1:11443/api/doc/` für
die API-Dokumentation. Jede Option dieses Handbuchs gilt; zum Beispiel
aktiviert `--models-dir` den Router für mehrere Modelle.

Voraussetzungen, die Sie selbst als root installieren, weil das Skript sie nur
prüft: `cmake`, ein C/C++-Compiler und die statische C-Bibliothek (`build-essential`, das
`libc6-dev` mitbringt), `patchelf`, `libssl-dev` (oder
`--no-openssl`), Node.js 20.19+/22.12+ mit npm (oder `--no-ui`), ein vorhandenes CUDA
Toolkit (oder `--cpu`) und NCCL mit seinen Entwicklungsdateien (`libnccl2` und
`libnccl-dev`, oder `--no-nccl`). `nvcc` wird aus `CUDACXX`, dem `PATH` oder
`/usr/local/cuda/bin/nvcc` genommen. Ninja wird verwendet, wenn es installiert ist.
Zu den grundlegenden Werkzeugen gehören `tar`, `coreutils` und `util-linux`. Entfernte Builds benötigen außerdem
`curl`, `ca-certificates` und `gzip`; es werden keine Abhängigkeiten automatisch installiert.

Optionen: `--prefix PATH` (ein anderes Installationspräfix), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`
und `--help`.

Gut zu wissen:

- Der Build ist für die CPU dieses Computers und für die beim Kompilieren vorhandenen GPUs
  optimiert. Kompilieren Sie auf jedem Computer neu, statt die Installation zu kopieren.
- Die Installation lässt sich als Präfix verschieben: Verschieben Sie `bin/mgs` und `lib/mgs/` gemeinsam und behalten Sie dabei ihre Ebenen `bin/` und
  `lib/` bei. Starten Sie immer `bin/mgs`; `lib/mgs/mgs` ist intern und darf nicht direkt ausgeführt werden, weil damit
  der mitgelieferte Loader umgangen würde.
- Ein erneuter Lauf des Skripts aktualisiert die Installation. Er ersetzt nur `bin/mgs`, `lib/mgs/` und
  `share/doc/mgs/licenses/`. Eine bereits laufende Kopie funktioniert weiter; starten Sie sie neu, um den neuen Build zu verwenden.
- Das Skript lehnt ein Präfix ab, in dem `bin/mgs`, `lib/mgs/` oder dieser Lizenzordner bereits existieren, sofern es sie nicht
  selbst angelegt hat; es hinterlässt eine versteckte Markierung `.moe-gguf-server-bundle` in `lib/mgs/`.
- `--prefix PATH` installiert unter einem anderen Präfix, in das Ihr Benutzer schreiben kann; es darf weder `/` noch ein Ordner innerhalb des
  Projekts sein.
- Bereitgestellte Quellen, Build-Dateien, Caches, die zusammengestellte Installation und das Protokoll verwenden einen
  privaten Ordner `/tmp/moe-gguf-server-build.XXXXXXXX/`. Nachdem das Ergebnis
  in das Präfix kopiert wurde, entfernt das Skript diesen Ordner. Es räumt auch bei Fehlern
  und abgefangenen Unterbrechungen auf. Jeder Lauf kompiliert von Grund auf, ohne in
  den Quellbaum oder nach `/opt/` zu schreiben. Lokale Builds akzeptieren `--build-dir` nicht.
- Das Präfix wird beim Kopieren und Ersetzen fertiger Artefakte gesperrt,
  was gleichzeitige Aktualisierungen derselben Installation verhindert.
- Ein CUDA-Build benötigt zum Starten den NVIDIA-Treiber; verwenden Sie `--vulkan` auf Computern mit einer anderen GPU und `--cpu` auf
  Computern ohne GPU.

<a id="vulkan"></a>
## Vulkan (GPUs von AMD, Intel und NVIDIA)

Neben CUDA (Standard) und CPU kompilieren sowohl `build/build-local.sh` als auch das Debian-Installationsprogramm mit
`--vulkan` das Vulkan-Backend, für GPUs ohne CUDA wie AMD-Radeon-Karten. Installieren Sie zuerst als root die Build-Abhängigkeiten:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Dann als Ihr normaler Benutzer:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

Das Debian-Installationsprogramm akzeptiert dieselbe Option und installiert diese Vulkan-Pakete selbst:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` zeigt die GPU an, zum Beispiel `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- Die lokale Installation bündelt den Vulkan-Loader (`libvulkan.so.1`) wie jede andere Bibliothek, aber der Vulkan-Treiber
  stammt vom System: Der Loader liest die in `/usr/share/vulkan/icd.d/` registrierten Treiber (Mesa RADV für AMD,
  ANV für Intel, den NVIDIA-Treiber), und diese laden ihre eigenen Systembibliotheken, weil sie zum Kernel
  und zum System passen müssen. Kompilieren Sie auf der Debian-Version, auf der die Binärdatei laufen wird.
- Der Benutzer, der `mgs` ausführt, benötigt den Render-Knoten der GPU (`/dev/dri/renderD*`). Ein am Desktop angemeldeter Benutzer
  hat ihn bereits; andere Benutzer benötigen die Gruppe `render` (und `video`). `configure-web.sh` fügt das Dienstkonto beiden hinzu.
- MTP im VRAM, der MoE-Expertencache, das Vorabladen der Experten, `--fit` und der Monitor funktionieren wie mit CUDA. Bei AMD liest der
  Monitor Auslastung, VRAM und Temperatur aus sysfs.
- `mgs` zählt nur den echten VRAM (die gerätelokalen Speicher-Heaps) als GPU-Speicher, niemals GTT. GTT ist der Teil des
  System-RAM (normalerweise die Hälfte davon), den eine GPU von AMD oder Intel einblenden kann: Es ist kein zusätzlicher Speicher, und die GPU liest ihn
  über PCIe, um ein Vielfaches langsamer als ihren VRAM. Die MTP-Schichten, die dichten Gewichte, der KV-Cache und der Expertencache werden
  im VRAM reserviert; die Experten, die nicht hineinpassen, bleiben im normalen RAM, wo die CPU sie berechnet.
- `mgs` prüft, wo der Kernel diesen Speicher tatsächlich hält (`/proc/self/fdinfo`, veröffentlicht von amdgpu, i915 und xe). Wenn
  ein anderes Programm den VRAM füllt und der Treiber einen Teil davon nach GTT verschiebt, melden das Protokoll und der Monitor dies, und der
  Monitor zeigt den tatsächlichen Ort (VRAM und GTT) jeder GPU von AMD oder Intel an. Der Expertencache wird mit einer
  niedrigeren Speicherpriorität (`VK_EXT_memory_priority`) reserviert, sodass der Treiber ihn vor den MTP-Schichten, den Gewichten oder dem
  KV-Cache verschiebt. Ohne Resizable BAR meiden Gerätepuffer außerdem das kleine VRAM-Fenster, das die CPU einblenden kann (256 MiB bei einer
  RX 580), das sich der Desktop teilt und das der Treiber als Erstes nach GTT leert.
- Nur CUDA kann die Modellseiten fixieren, die im RAM bleiben, daher erreichen die Experten die GPU mit Vulkan über auslagerbaren
  Speicher. `--load-mode none` kopiert sie stattdessen in fixierten Host-Speicher, was auf einer RTX 4060 Ti über Vulkan
  (Qwen3.6-35B-A3B Q4_K) das Hochladen der Experten von 2,7 auf 13,1 GB/s und die Generierung von 27,5 auf 36,6 Token/s steigerte. Bei
  AMD und Intel zählt dieser fixierte Speicher gegen GTT: Wenn die Experten nicht hineinpassen, schlägt das Laden mit
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`) fehl, wie es bei den 18,9 GB an Experten dieses
  Modells mit 16,8 GB GTT auf einer RX 580 geschah. Verwenden Sie es nur, wenn die Experten in GTT und in den freien RAM passen.
- Wählen Sie die Modellgröße passend zum Rechner: Das gesamte GGUF sollte in den freien RAM plus VRAM passen, abzüglich des KV-Caches und der
  Rechenpuffer. Mit 8 GB VRAM und 16 GB RAM bedeutet das etwa 18–20 GB an Gewichten; größere Dateien werden
  immer wieder von der Festplatte gelesen, und die Generierung wird sehr langsam. Das oben genannte Q4_K mit 21,6 GB erreichte 7,8 Token/s auf einer
  RX 580 mit 15 GB RAM und 15,9 Token/s, sobald der Rechner 31 GB hatte (10,0 mit `--spec-type none`).

<a id="deployment"></a>
## Bereitstellung in der Produktion

Führen Sie im Quellverzeichnis (`/opt/moe-gguf-server-source` bei entfernten Installationen)
nach der Installation der Binärdatei Folgendes aus:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

Die Domain muss auf den Produktionsrechner auflösen. Der Dienst `moe-gguf-server-web`
betreibt eine eigene Apache-Instanz als `moe-gguf-server`, ohne root.
Jeder Start wählt den ersten freien HTTP-Port ab 11080 und den ersten freien HTTPS-Port
ab 11443. Jede Folge läuft unabhängig bis 65535 weiter, ohne
beiden Protokollen denselben Port zuzuweisen. Er untersucht die IPv4/IPv6-TCP-Listener
und versucht es erneut, wenn Apache feststellt, dass ein Port nach der Auswahl belegt wurde. Andere
Konfigurationsfehler brechen den Start ab, statt mit Portkonflikten verwechselt zu werden.

HTTP leitet dauerhaft auf den gewählten HTTPS-Port weiter und behält dabei Pfad und Abfrage bei.
Die Weiterleitung wird nicht zwischengespeichert. Apache leitet Anfragen an die Inferenz auf
`127.0.0.1:8080` weiter, die ausdrücklich mit `--http --port 8080` gestartet wird und ebenfalls als
`moe-gguf-server` läuft. Die Installation erfordert root;
keiner der beiden Dienste muss als root laufen. Es wird kein HAProxy konfiguriert, und keine gemeinsam genutzten
Apache-Sites werden importiert oder verändert.

Die tatsächlichen URLs werden veröffentlicht, nachdem beide Listener geöffnet wurden:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

Beispielsweise leiten HTTP 11081 und HTTPS 11445 auf
`https://ai.example.com:11445/` weiter. Die Datei wird beim Herunterfahren entfernt und beim Start
neu erzeugt; ein Neustart kann andere Ports wählen. Clients, Firewalls und jeder
externe Proxy müssen die veröffentlichten Werte verwenden. Der Port ist Teil des
[Browser-Ursprungs](https://developer.mozilla.org/en-US/docs/Glossary/Origin), daher hat ein
anderer HTTPS-Port einen eigenen lokalen Verlauf und eigene Einstellungen. Bestehende Konversationen
bleiben in ihrem ursprünglichen Ursprung; Export und Import können sie übertragen.

Das Systemkonto hat kein interaktives Passwort. Modelle werden in
`/opt/moe-gguf-server/models` gespeichert, und die Konfiguration des Programms liegt im Home-Verzeichnis des
Dienstkontos, `/opt/moe-gguf-server/.mgs/config/`: der API-Schlüssel (`api.key`), die Web-Domain
(`web-domain`), die Apache-Vorlage (`apache2.conf.in`) und die Einstellungen der
Oberfläche (`ui-settings.json`), alle im Besitz von `moe-gguf-server` mit dem Modus 0600.
Zugangsdaten und die Ausgabe der Installation werden in `/root/webapp-credentials.txt`
und `/root/webapp-install.log` festgehalten, beide mit dem Modus 600. Eine Neukonfiguration erhält vorhandene
Schlüssel, Modelle und Zertifikate. Der private TLS-Schlüssel gehört root und ist für die Gruppe
moe-gguf-server lesbar (Modus 640), sodass der unprivilegierte Webprozess ihn lesen kann.

Ihr eigenes TLS-Zertifikat verwenden Sie mit:

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

Das Zertifikat liegt in `/opt/moe-gguf-server/certificates/` (`fullchain.pem` und
`privkey.pem`, im Besitz des Kontos `moe-gguf-server`, wobei nur dieses den Schlüssel lesen kann).
Werden keine Dateien angegeben, behält das Skript das bereits vorhandene Zertifikat oder erzeugt beim ersten
Mal ein selbstsigniertes, das jede spätere Neuinstallation wiederverwendet, sodass Browser
ihre Vertrauensausnahme behalten. Vertrauen Sie ihm auf Ihrem Client oder geben Sie ein Zertifikat an, dem Ihre
Clients bereits vertrauen. Angegebene Dateien werden in diesen Ordner kopiert; bei der Erneuerung des Zertifikats
müssen diese Kopien aktualisiert und `moe-gguf-server-web` neu gestartet werden. Das Skript bezieht oder erneuert keine Let's-Encrypt-Zertifikate.
Verwenden Sie denselben Befehl nach der Erneuerung der Quellzertifikate. Die Option
`--binary` des Skripts unterstützt einen vom Standard abweichenden Speicherort der ausführbaren Datei.

Der Dienst verwendet den Router-Modus. Legen Sie lesbare GGUF-Modelle in
`/opt/moe-gguf-server/models` ab, oder verwenden Sie die Modellverwaltung in der Oberfläche. Für
spezielle Befehlszeilenoptionen verwenden Sie unter Debian/Ubuntu ein systemd-Override oder unter Alpine die
OpenRC-Dienstkonfiguration. Lassen Sie das Backend an Loopback gebunden.
Der Web-Konfigurator erzeugt seine verwaltete Dienstdatei bei einem erneuten Aufruf neu.

Starten Sie den Webdienst neu, um die Ports erneut auszuwählen:

```bash
systemctl restart moe-gguf-server-web
```

Unter Alpine verwenden Sie `rc-service moe-gguf-server-web restart`. Die Startprotokolle stehen unter Debian in
`journalctl -u moe-gguf-server-web` und unter Alpine in `/var/log/moe-gguf-server-web.log`.
Die Diagnosemeldungen von Apache stehen in `/opt/moe-gguf-server/_/temp/web/startup.log`
und `error.log`; die Zugriffsprotokolle der Domain bleiben in `/var/www/DOMAIN-logs/`. Die eigenständige
Vorlage ist `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. Bearbeiten Sie nicht die erzeugte
Konfiguration in `_/temp/web/`; sie wird bei jedem Start neu erstellt.

<a id="chat"></a>
## Verwendung der Oberfläche

1. Öffnen Sie die URL des Servers und geben Sie seinen API-Schlüssel ein, wenn Sie dazu aufgefordert werden, oder unter Einstellungen → Allgemein.
2. Wählen Sie ein Modell aus. Laden Sie im Router-Modus ein verfügbares Modell, wenn keines geladen ist.
3. Geben Sie eine Nachricht ein und senden Sie sie. Die Eingabetaste sendet standardmäßig; Umschalt+Eingabe fügt einen
   Zeilenumbruch ein. Diese Tastenkürzel lassen sich in den Einstellungen ändern.
4. Fügen Sie Dateien über die Anhangsaktion hinzu. Bilder, Audio und Video erfordern passende
   Modalitäten des Modells. PDF-Dateien können als Text extrahiert oder in Bilder umgewandelt werden;
   Modelle ohne Bildverarbeitung verwenden den Textweg.
5. Beenden Sie die Generierung, überspringen Sie den Denkprozess, sofern unterstützt, bearbeiten Sie eine Nachricht, generieren Sie eine
   Antwort erneut oder zweigen Sie eine Konversation über die Nachrichtenaktionen ab.

Die Seitenleiste unterstützt Suchen, Anheften, Umbenennen, Löschen und Mehrfachauswahl.
Konversationen werden im Browser mit IndexedDB gespeichert; exportieren Sie sie, bevor Sie den
Browserspeicher leeren oder das Browserprofil wechseln. Die Einstellungen werden vom Server in
`~/.mgs/config/ui-settings.json` des Benutzers gespeichert, der `mgs` ausführt (im bereitgestellten Dienst
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), sodass jeder Browser, der die
Oberfläche öffnet, sie teilt: Generierungsparameter, Anzeigeoptionen, Farbschema, Sprache, MCP-Server,
Lieblingsmodelle, deaktivierte und immer erlaubte Werkzeuge sowie der standardmäßige
Denkaufwand. Nur der API-Schlüssel bleibt in jedem Browser, weil er die Anfrage authentifiziert,
die den Rest lädt; geben Sie ihn daher in jedem Browser einmal ein. Import und Export sind in den
Einstellungen verfügbar. Exporte können Anhänge und, wenn ausdrücklich ausgewählt, vertrauliche
API-Schlüssel oder MCP-Autorisierungs-Header enthalten.

Wählen Sie die Sprache unter Einstellungen → Allgemein. Die 15 Optionen sind in der Reihenfolge `de-DE`,
`en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`,
`ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN` angeordnet; Standard ist `en-US`.
Brasilianisches und europäisches Portugiesisch haben getrennte Übersetzungen. Hebräisch verwendet eine
Oberfläche von rechts nach links; die Chateingabe und einfache Benutzernachrichten folgen der Richtung ihres
Textes, während Code und Terminalausgaben von links nach rechts angezeigt werden.
Ein Sprachwechsel speichert die Auswahl in den Einstellungen des Servers und
lädt die Oberfläche neu; der Server veröffentlicht sie in `/api/config.js`, sodass jeder Browser
in dieser Sprache startet. Speichern Sie vorher andere ausstehende Einstellungen. Der URL-Abfrageparameter `lang`
überschreibt sie für diese Seite und behält die Auswahl bei, wenn sie nicht gespeichert werden kann.
Hängen Sie zum Beispiel `?lang=ja-JP` an die URL der Oberfläche an, um Japanisch auszuwählen.

Einstellungen → Allgemein → **Immer in der Sprache der Benutzeroberfläche antworten** ist standardmäßig
deaktiviert. Aktivieren Sie die Option und speichern Sie die Einstellungen, um das Modell anzuweisen, für jede neue
Antwort die Sprache und die regionale Variante der Oberfläche zu verwenden, auch in
bestehenden Konversationen, bei erneut generierten Antworten, Fortsetzungen, Runden nach Werkzeugaufrufen
und vom Modell generierten Titeln. Für `es-AR` verlangt die Anweisung ausdrücklich
den argentinischen Voseo: „Seguí estos pasos“, „tenés“ und „podés“; `es-ES` verlangt den
spanischen Sprachgebrauch Spaniens. Britisches/amerikanisches Englisch und brasilianisches/europäisches
Portugiesisch erhalten ebenfalls eigene regionale Anweisungen.

Die Anweisung begleitet jede neue Inferenzanfrage und jede Vorbereitung des KV-Caches;
sie wird mit der ausgehenden Systemnachricht kombiniert, ohne Ihren gespeicherten
System-Prompt oder die Konversation zu bearbeiten. Ein Wechsel der Oberflächensprache wirkt sich auf neue
Anfragen nach dem Neuladen aus; eine bereits in Generierung befindliche Antwort behält die
Anweisung, die sie erhalten hat. Das Deaktivieren der Einstellung beendet ihr Hinzufügen bei nachfolgenden
Anfragen. Bestehende Antworten werden nicht übersetzt. Code, Befehle, Namen, Zitate
und ausdrücklich angeforderte Übersetzungen behalten die erforderliche Sprache; die
umgebenden Erklärungen folgen der Sprache der Oberfläche. Wie konsequent die Anweisung befolgt wird, hängt von der Fähigkeit
des Modells ab; dies ist weder eine automatische Übersetzung noch eine Garantie für vollständige Einhaltung.
Externe API-Clients sind von dieser Browsereinstellung nicht betroffen.

Wählen Sie für das Farbschema System, Hell oder Dunkel. Zu den Generierungseinstellungen gehören die
Kontextverwendung, Temperatur, Sampler, Strafen und die Grenzen für Agentenrunden. Manche Funktionen
hängen von Server-Flags oder vom ausgewählten Modell ab.

Nach jeder Antwort sendet die Oberfläche die Konversation erneut, sobald der Server
untätig ist, damit sie bereits im KV-Cache kodiert ist, während Sie lesen, und die nächste Runde
schneller beginnt. Dies ist standardmäßig aktiviert; deaktivieren Sie Einstellungen → Entwicklung → **KV-Cache nach jeder
Antwort vorbereiten**, um es zu unterbinden.

Während der Server eine andere Anfrage bearbeitet, wartet eine von Ihnen gesendete Nachricht in der
[Warteschlange](#request-queue), und ihre Antwort zeigt bis zu ihrem Beginn einen Hinweis statt des Fortschrittstexts an:
„In der Warteschlange: Zuerst wird eine andere Anfrage bearbeitet, die direkt an die API gesendet wurde
(Anfragen davor: N).“, wenn ein Programm gleichzeitig die API verwendet, oder den entsprechenden Hinweis
für eine Anfrage der Weboberfläche selbst (ein anderer Tab oder Browser, ein generierter Titel oder die Vorbereitung des
KV-Caches). Der Hinweis verschwindet, sobald Ihre Anfrage beginnt.

<a id="monitor"></a>
## Monitor und Statusleiste

**Monitor** befindet sich in der Seitenleiste unter **Einstellungen** (Adresse `#/monitor`). Solange er geöffnet ist, aktualisiert er sich alle
zwei Sekunden und zeigt:

- **Auslastung**: zuerst eine Karte pro GPU des Rechners (NVIDIA, AMD oder Intel, auch wenn kein Modell sie verwendet), betitelt mit ihrer
  fett gedruckten Kennung und ihrem Namen (zum Beispiel **gpu-1**: NVIDIA GeForce RTX 4060 Ti; bei mehreren GPUs gpu-1, gpu-2
  und so weiter), mit ihrer Auslastung, der Auslastung des Speichercontrollers (Bandbreite), dem belegten VRAM, der Temperatur und der Leistungsaufnahme im Verhältnis zu ihrem
  Grenzwert; dann eine Karte pro Prozessor, auf die gleiche Weise mit seinem Modell betitelt (zum Beispiel **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; bei mehreren Prozessoren cpu-1, cpu-2 und so weiter), mit seiner Auslastung, einem kleinen senkrechten Balken pro
  logischer CPU dieses Prozessors und der durchschnittlichen Systemlast. Jeder Auslastungsbalken ist grün und füllt sich mit zunehmender Auslastung
  rot; ein Balken, dessen Wert unbekannt ist, bleibt grau. GPUs von NVIDIA werden über die NVML-Bibliothek des Treibers gelesen und GPUs von AMD
  über `gpu_busy_percent` in sysfs; jede andere GPU (Intel oder NVIDIA ohne NVML) zeigt keine Auslastung an, und ihr
  Name und ihr Speicher stammen vom Inferenz-Backend, sobald ein Modell sie verwendet.
- **Speicher**: ein Balken pro GPU (VRAM) und einer für den RAM; eine GPU ohne Speicherangaben, etwa eine GPU von Intel, die noch kein
  Modell verwendet, hat keinen Balken. Jeder Balken ist nach Verwendungszweck unterteilt, jeder Zweck immer in derselben Farbe, und seine
  Legende listet jedes Segment mit seiner Größe und seinem Anteil am Gesamtwert auf, eines pro Zeile und in derselben Reihenfolge wie im
  Balken. Fahren Sie mit dem Mauszeiger über ein Segment, um seine Größe zu sehen.

  | Segment | Inhalt |
  | --- | --- |
  | MTP- / Entwurfsmodellgewichte | Gewichte der MTP-Schichten (nextn) des Modells, eines separaten MTP-GGUF, das mit `--spec-draft-model` übergeben wurde, oder eines Entwurfsmodells. |
  | Expertengewichte | Geroutete Expertentensoren des Modells. |
  | Dichte Gewichte | Alle übrigen Gewichte: Attention, gemeinsame Experten, Embeddings und Ausgabe. |
  | KV-Cache | Kontextspeicher des Modells (KV-Cache oder rekurrenter Zustand). |
  | KV-Cache für MTP / Entwurfsmodell | Kontextspeicher des MTP- oder Entwurfskontexts. |
  | Berechnungspuffer | Temporäre Puffer der Rechengraphen. |
  | Expertencache und Vorabladen | MoE-Expertencache (Pool, Kennungen und Staging) und die Slots für das Vorabladen der Experten. |
  | Prompt-Cache und Prüfpunkte | Mit `--cache-ram` gespeicherte Prompts und die Kontext-Prüfpunkte der Slots (RAM). |
  | Sonstiges | Belegter Speicher, den nichts von oben Genanntem erklärt: andere Prozesse, der Desktop, Treiberkontexte, der multimodale Projektor. |
  | Frei | Der Rest des Geräts oder des RAM. |

  Aus der Modelldatei eingeblendete Gewichte zählen als RAM, auch wenn der Kernel diesen Seitencache als verfügbar meldet, sodass die
  Segmente zusammen mehr ergeben können als der „belegte“ Speicher; der Balken zeigt dann kein „Sonstiges“ und weniger freien Speicher. Der
  Expertencache erscheint nach der ersten Anfrage, wenn er erstellt wird. Im Router-Modus umfassen die Balken jedes geladene
  Modell, und der Monitor lädt oder weckt niemals ein Modell.

Die **Statusleiste** am unteren Rand jeder Seite zeigt zuerst die Auslastung jeder GPU und dann die jedes physischen
Prozessors, jeweils in einem eigenen Balken, mit denselben Kennungen wie die Karten des Monitors: `gpu-1`, `gpu-2`... und `cpu-1`,
`cpu-2`... Die Balken haben dieselbe Größe, aktualisieren sich alle zwei Sekunden und sind wie die des Monitors grün und füllen sich
mit zunehmender Auslastung rot; der Hover-Text jedes Balkens nennt den Namen des Geräts. Ohne GPU zeigt sie einen leeren
`GPU`-Balken, und wenn die Balken auf einem schmalen Bildschirm nicht hineinpassen, scrollt die Statusleiste seitwärts. Klicken Sie darauf, um den
Monitor zu öffnen. Sie funktioniert weiter, während das Modell lädt. Sie wird standardmäßig angezeigt; deaktivieren Sie Einstellungen → Anzeige → **Statusleiste
anzeigen**, um sie auszublenden.

<a id="models"></a>
## Modelle, Werkzeuge und MCP

`mgs` akzeptiert ausschließlich MoE-Hauptmodelle. Dense-Modelle wie GPT-2 oder Phi-2 werden abgelehnt, bevor ihre Gewichte geladen werden; ihre Decoder-Implementierungen werden nicht kompiliert. Dichte Schichten innerhalb eines MoE-Modells und ein ausdrücklich angegebener, kompatibler reiner MTP-Kopf bleiben erhalten.

<a id="supported-models"></a>
### Unterstützte Modellfamilien

| Familie | GGUF-Architektur | Hinweise |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP, wenn das GGUF es enthält. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP, wenn das GGUF es enthält. Seine schichtweise Embedding-Tabelle (etwa 29 GB) wird bei Bedarf aus der Modelldatei gelesen, halten Sie die Datei daher auf einem schnellen Datenträger. Lange Kontexte werden unter der dünnbesetzten Auswahl mit dichter Attention berechnet: korrekt, aber ohne die Beschleunigung dünnbesetzter Kernel. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; Bilder mit seinem mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Die veröffentlichten GGUFs enthalten kein MTP. Mit Vulkan läuft seine Residualmischung auf der CPU. |
| DeepSeek-V4.1-Flash | `deepseek41` | Lädt beide veröffentlichten Formate: die DwarfStar-Dateien von antirez (etwa `antirez/deepseek-v4.1-flash-gguf`) und die Dateien des llama.cpp-Ökosystems. Seine Engram-Tabelle (etwa 188 GiB innerhalb der Datei) wird nie geladen: Jedes Token liest seine Zeilen aus der Modelldatei, daher muss die Datei verfügbar bleiben, solange `mgs` läuft, vorzugsweise auf einer SSD, und Linux oder ein anderes POSIX-System ist erforderlich. Dateien ohne Chat-Vorlage verwenden eine integrierte, und `reasoning_effort` akzeptiert eine Zahl von 1 bis 100. Kein MTP und keine Bilder. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

Gemma 4 und die DeepSeek-Modelle vor V4.1 werden nicht mehr unterstützt (siehe [entfernte Kompatibilität](#removed-compatibility)).

Bei mehreren ausgewählten GPUs prüft MTP alle, bevor es auf den RAM ausweicht. Seine Speicherschätzung schließt den Ausgabekopf ein. Integrierte MTP-Puffer werden vor den übrigen Gewichten geladen; ein externes MTP-GGUF, das mit `--spec-draft-model` übergeben wird, wird vor dem Hauptmodell geladen, auch mit `--fit off`. Die aktuelle ausdrückliche Geräteauswahl bestimmt weiterhin, welche GPUs verfügbar sind.

Verwenden Sie `--reasoning on` oder `--reasoning off`, um das Denken zu steuern. Die Übergabe von `enable_thinking` über `--chat-template-kwargs` wird abgelehnt. Verwenden Sie die dokumentierten Optionsnamen mit Bindestrich; Schreibweisen mit Unterstrich werden abgelehnt. Übergeben Sie Listenwerte als kommagetrennten Wert, statt eine Option zu wiederholen. Nur `--spec-type` bleibt wiederholbar, um spekulative Modi zu kombinieren.

Die Modellauswahl zeigt verfügbare, geladene und Lieblingsmodelle. Die Modelldetails
umfassen Kontextgröße, Modalitäten, Quantisierung und die Chat-Vorlage. Der Router-Modus
verwaltet separate Kindprozesse für geladene Modelle; Anfragen geben das Modell
über das Feld oder den Abfrageparameter `model` an.

MCP-Server werden per URL mit optionalen Autorisierungs-Headern konfiguriert. Sie können
Verbindungsprotokolle, verfügbare Prompts, Ressourcen und Werkzeuge einsehen. Der optionale
CORS-Proxy des Servers muss ausdrücklich mit dem entsprechenden Flag der Engine aktiviert
werden; er gibt nur inerte Inhalte zurück und weigert sich, als Seite geöffnet zu werden (siehe
[Sicherheitsgrenzen](#security-limits)). Browserbeschränkungen für HTTP-Ressourcen innerhalb von HTTPS-Seiten gelten weiterhin.
Wenn zwei aktivierte MCP-Server ein Werkzeug mit demselben Namen anbieten, wird keine der beiden Kopien dem Modell angeboten, und die
Oberfläche zeigt eine Warnung an: Deaktivieren Sie alle bis auf einen dieser Server, um das Werkzeug zu verwenden.
Die integrierten Werkzeuge des Servers und die JavaScript-Sandbox des Browsers sind optional.
Werkzeugfreigaben und die Grenze für Agentenrunden bleiben in der Oberfläche verfügbar. Die Rundengrenze
(Einstellungen → Agenten → **Agentenrunden**, die maximale Anzahl von Werkzeugausführungszyklen
vor dem Anhalten, um Endlosschleifen zu vermeiden) beträgt standardmäßig 50.

Streams können sich anhand der gespeicherten Konversationsidentität und des Cursors erneut verbinden, solange der
Server den Stream vorhält. Ein angehaltener Server oder ein abgelaufener Stream kann keine
garantierte Wiedergabe liefern. Die Oberfläche meldet eine fehlgeschlagene Wiederverbindung, statt
stillschweigend eine zweite Generierungsanfrage zu erstellen.

### Lokaler Shell-MCP-Server

Starten Sie `mgs` mit `--local-shell-mcp-server`, damit die Modelle des Web-Chats das Terminal des Rechners verwenden können, auf dem
`mgs` läuft:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- Beim Start prüft `mgs`, welche Shells installiert sind, und bietet für jede von ihnen ein Werkzeug an: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) und `psh.run` (PowerShell, `pwsh`). Das Protokoll listet die gefundenen Shells auf; eine Shell,
  die nicht installiert ist, erhält kein Werkzeug.
- Die Oberfläche erkennt ihn, fügt ihn den MCP-Servern als **local shell mcp server** hinzu und aktiviert ihn. Sie können ihn dort
  oder für eine einzelne Konversation deaktivieren, und Ihre Wahl bleibt erhalten. Wenn `mgs` ohne die Option startet, verschwindet
  der Eintrag.
- Jedes Werkzeug nimmt `command` und optional `cwd` (Arbeitsverzeichnis; standardmäßig das Home-Verzeichnis des Benutzers,
  der `mgs` ausführt) und `timeout` (Sekunden; standardmäßig 60, höchstens 3600) entgegen. Die Antwort ist ein JSON-Objekt mit
  `exit_code`, `stdout` und `stderr` sowie `signal`, `timed_out`, `cancelled`, `timeout`, `seconds`, `shell` und
  `cwd`; bei langen Ausgaben bleiben nur Anfang und Ende erhalten.
- Jeder Befehl läuft in einem neuen Prozess ohne Terminal und mit leerer Eingabe: `cd` und Variablen werden nicht in
  den nächsten Aufruf übernommen, und interaktive Programme (Editoren, Rückfragen) können nicht verwendet werden. Wenn das Zeitlimit abläuft oder die
  Anfrage abgebrochen wird, wird jeder Prozess beendet, den der Befehl gestartet hat. Im Hintergrund zurückgelassene Prozesse müssen
  ihre Ausgabe umleiten.
- Die Oberfläche bittet vor jedem Aufruf um Erlaubnis, es sei denn, Sie entscheiden sich, das Werkzeug immer zu erlauben.
- `sudo` funktioniert ohne Terminal (Linux): Wenn es ein Passwort benötigt, öffnet sich in der Mitte der Seite ein Fenster,
  auf welcher Seite Sie sich auch befinden, mit der Eingabeaufforderung von sudo (`[sudo] password for user:`), dem Werkzeug und dem Befehl, der
  danach fragt, einem Passwortfeld und darunter den Schaltflächen **Akzeptieren** und **Abbrechen**. Der Cursor steht bereits im
  Passwortfeld, sodass Sie das Passwort sofort eingeben können; die **Eingabetaste** bestätigt und **Escape** bricht ab; ein Klick
  außerhalb des Fensters schließt es nicht. Das Passwort geht nur an sudo auf dem Rechner, auf dem `mgs`
  läuft: Es wird nicht in der Konversation gespeichert, nicht in die Protokolle geschrieben, und das Modell sieht es nie. Während das
  Fenster wartet, läuft das Zeitlimit des Befehls nicht weiter; wenn Sie **Abbrechen** drücken oder niemand innerhalb von 5
  Minuten antwortet, schlägt sudo fehl. sudo fragt bei jedem Aufruf erneut, weil jeder ein neuer Prozess ist, und bis zu dreimal, wenn
  das Passwort falsch ist; das Fenster warnt dann, dass das vorherige Passwort möglicherweise falsch ist. Befehle laufen in einer eigenen
  Sitzung, ohne das Terminal, in dem `mgs` gestartet wurde, sodass sudo in diesem Terminal nie nach dem Passwort fragt.
  Nur die Weboberfläche von `mgs` kann antworten: Mit anderen MCP-Clients schlägt sudo wie bisher mangels Terminal fehl.
  Der bereitgestellte Dienst läuft unter dem Konto `moe-gguf-server`, das normalerweise weder ein Passwort noch eine sudo-Berechtigung
  hat; dies ist daher für ein mit Ihrem eigenen Benutzer ausgeführtes `mgs` gedacht (zum Beispiel den Build in `~/.local`). Wenn `mgs`
  keine Variable `DISPLAY` hat, erhalten die Befehle `DISPLAY=:mgs-askpass`, weil sudo seinen Passwort-Helfer
  ohne Terminal nur verwendet, wenn `DISPLAY` existiert; Programme, die versuchen, ein Fenster zu öffnen, schlagen sofort fehl.
- Im Chat wird jeder Aufruf mit dem Namen seines Werkzeugs angezeigt, etwa `bash.run`. Aufgeklappt zeigt er bis zu drei
  graue Kästen in Festbreitenschrift: **stdin** mit dem reinen Befehl, **stdout** mit der Ausgabe und **stderr** mit
  dem Fehlertext. stdout erscheint nur, wenn der Befehl etwas ausgegeben hat, und stderr nur, wenn es einen Fehler gab;
  der Exit-Code steht neben ihrem Titel, zum Beispiel `stdout (Code 2):` (neben stdin, wenn es keine Ausgabe gibt).
  Der stdin-Kasten hat einen weißen Rahmen (dunkelgrau im hellen Farbschema), der stdout-Kasten einen grünen und der stderr-Kasten
  einen roten, bei jedem Shell-Werkzeug (`bash.run`, `zsh.run`, `sh.run` und `psh.run`).
- Aktivieren Sie **Einstellungen → Anzeige → Lokale Shell-Befehle automatisch aufklappen**, damit sich jeder Aufruf von selbst öffnet: Jeder
  Kasten erscheint, sobald sein Inhalt bekannt ist, stdin, während das Modell den Befehl schreibt, und stdout und stderr, wenn
  der Befehl endet.
- Jeder Aufruf eines Shell-Werkzeugs hinterlässt eine eigene Datei in `~/.mgs/local-shell-mcp-server/` (im Home-Verzeichnis des Benutzers, der
  `mgs` ausführt; im bereitgestellten Dienst `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), benannt nach dem lokalen
  Datum und der Uhrzeit des Aufrufs bis auf die Millisekunde, zum Beispiel `y2026m10d12h23m59s59ms731.jsonl`; `-2`, `-3`… werden
  angehängt, wenn mehrere Aufrufe in derselben Millisekunde beginnen. Es ist eine JSONL-Datei, ein JSON-Objekt pro Zeile: zuerst der Befehl (`"event": "command"`,
  mit dem Werkzeug, der Shell, dem Arbeitsverzeichnis und dem Zeitlimit), geschrieben, bevor er läuft; dann die gesamte Ausgabe,
  ohne Kürzungen, so wie sie eintrifft, eine Zeile pro Stück (`"event": "stdout"` oder `"stderr"`, mit dem Text in `data`) in
  der Reihenfolge, in der der Befehl sie geschrieben hat; und zuletzt das Ergebnis (`"event": "result"`: Exit-Code, Signal, ob das Zeitlimit überschritten
  oder der Befehl abgebrochen wurde, die benötigten Sekunden und die Bytes von stdout und stderr) oder der Grund, warum er nicht starten konnte
  (`"event": "error"`). Auch ein vor der Ausführung abgelehnter Aufruf (kein Befehl, ein nicht existierendes Arbeitsverzeichnis, ein
  ungültiges Zeitlimit) hinterlässt seine Datei, mit dem Befehl, wie er angefordert wurde, und dem Fehler. Der Chat behält von sehr langen Ausgaben nur
  Anfang und Ende, aber die Datei enthält alles, und sie enthält die Ausgabe bereits, wenn `mgs`
  anhält, während der Befehl läuft. Um die gesamte stdout-Ausgabe eines Aufrufs zu lesen, führen Sie `jq -j 'select(.event == "stdout") | .data'
  FILE` aus. JSON enthält nur Text, daher werden Bytes, die kein UTF-8 sind (binäre Ausgabe), durch `�` ersetzt; ein Befehl, der
  Gigabytes ausgibt, hinterlässt eine Datei dieser Größe. Nur dieser Benutzer kann den Ordner und die Dateien lesen, weil die Ausgabe
  eines Befehls Geheimnisse enthalten kann; nichts löscht sie, entfernen Sie daher die alten, wenn Sie sie nicht mehr benötigen.

Fügen Sie im bereitgestellten Dienst `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` zum Abschnitt `[Service]` eines systemd-Overrides
(`systemctl edit moe-gguf-server`) hinzu und starten Sie `moe-gguf-server` neu; unter Alpine exportieren Sie dieselbe Variable in
der OpenRC-Dienstkonfiguration.

**Sicherheit.** Befehle laufen mit den Berechtigungen des Benutzers, der `mgs` ausführt (im bereitgestellten Dienst das
Konto `moe-gguf-server`, niemals `root`). Jeder, der die Oberfläche öffnen oder die API aufrufen kann, kann sie ausführen, aktivieren Sie
die Option daher nur in vertrauenswürdigen Umgebungen und schützen Sie den Server mit einem API-Schlüssel. Anfragen von Seiten anderer Websites
werden abgelehnt, der standardmäßige CORS-Ursprung wird localhost, und das Protokoll verzeichnet jeden Befehl mit seinem Exit-Code. Lesen Sie
den im Passwortfenster angezeigten Befehl, bevor Sie ein sudo-Passwort eingeben: Das Passwort autorisiert diesen Befehl. Der
Server übergibt es nur an einen Passwort-Helfer, dessen Elternprozess sudo ist, das als root für Ihren Benutzer läuft, sodass ein Befehl
den Helfer nicht selbst starten kann, um das Passwort auszulesen.

<a id="api"></a>
## HTTP-API

Die Beispiele verwenden 11443; ersetzen Sie ihn durch den HTTPS-Port aus `ports.json`, falls er abweicht.

Die Standardwurzel der API ist `/api`. Die Oberfläche bleibt unter `/`. Die Wurzelrouten `/v1/...`
des ursprünglichen Servers werden nicht mehr registriert. OpenAI-kompatible Clients müssen
`https://ai.example.com:11443/api/v1` als Basis-URL verwenden. Die Feldnamen von Anfragen und Antworten
bleiben mit Upstream kompatibel; die C/C++-Schnittstellen der Engine behalten ihre aktuellen Funktionen,
aber nicht diejenigen, die als veraltet markiert waren (siehe [entfernte Kompatibilität](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Setzen Sie `MGS_API_KEY` in Ihrer eigenen Shell auf den installierten Schlüssel. Native Operationen,
Embeddings, Tokenisierung, Reranking, Modellverwaltung und fortsetzbare Streams sind
in der mitgelieferten Swagger UI unter `/api/doc/` dokumentiert; ihr OpenAPI-JSON liegt unter
`/api/doc/openapi.json` und wird vom Server bei jeder Anfrage ohne statische JSON-Datei erzeugt. Die Übersicht spiegelt die Routen wider, die der
laufende Prozess registriert hat, einschließlich der Router- und der optionalen Kompatibilitätsrouten. Deaktivierte
Funktionen können 403 zurückgeben. Ein fehlender oder ungültiger Schlüssel ergibt 401, und das Laden eines Modells
ergibt 503. Die Routen für den Gesundheitsstatus und die Modellliste bleiben für `GET` und `HEAD` öffentlich zugänglich; siehe
[Sicherheitsgrenzen](#security-limits) für die Routen, die immer den Schlüssel benötigen.

`--api-prefix /api/custom` verlegt die Inferenzrouten unter `/api/custom`.
Die Oberfläche an der Wurzel erhält dieses Präfix von `/api/config.js`; die Dokumentation bleibt unter `/api/doc/`.
Der Namensraum der Dokumentation ist reserviert und kann nicht als Inferenzpräfix
verwendet werden. Router-Kindprozesse verwenden intern `/api`. Der Server prüft die Präfixe;
Werte außerhalb von `/api/` und Pfade mit abschließenden Schrägstrichen werden abgelehnt.

Der Monitor verwendet `GET /api/monitor` (vollständiger Bericht), `GET /api/monitor/memory` (Speicher-Momentaufnahmen der geladenen Modelle)
und `GET /api/monitor/usage` (nur CPU- und GPU-Auslastung, auch während das Modell lädt verfügbar). Sie lesen Zähler und
Momentaufnahmen, die von der Inferenzschleife veröffentlicht werden, niemals die Aufgabenwarteschlange, sodass ihre Abfrage weder ein schlafendes Modell weckt noch
dessen Leerlauf-Timer zurücksetzt. Ihre Felder sind in Swagger beschrieben.

Die Oberfläche liest ihre Einstellungen mit `GET /api/ui/settings` (`{"settings": DOCUMENT}` oder
`{"settings": null}` vor dem ersten Speichern) und speichert sie mit `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, das `~/.mgs/config/ui-settings.json` vollständig ersetzt, mit dem Modus 0600). Beide erfordern
den API-Schlüssel und antworten auch, während das Modell lädt.

<a id="request-queue"></a>
### Anfragewarteschlange und parallele Anfragen

Standardmäßig bearbeitet `mgs` eine Anfrage nach der anderen (`-np 1`) und gibt ihr den gesamten Kontext, sodass jede Anfrage
alle Kontext-Token nutzen kann, ob sie aus der Weboberfläche oder direkt von der API kommt. Die Anfragen, die
in der Zwischenzeit eintreffen, warten in einer einzigen Warteschlange und werden in der Reihenfolge ihres Eintreffens bearbeitet: Anfragen der Oberfläche
überholen keine API-Anfragen und umgekehrt. Die API verhält sich gleich, unabhängig von den Einstellungen
der Oberfläche.

Um mehrere Anfragen gleichzeitig zu bearbeiten, starten Sie `mgs` mit `-np N` (`--parallel N`, `MGS_ARG_N_PARALLEL`):
Bis zu N Anfragen laufen parallel, und der Kontext wird unter ihnen aufgeteilt, sodass jede einen Anteil erhält (`n_ctx_slot` in
der Zeile `initializing` des Protokolls); mit `--kv-unified` teilen sie sich stattdessen einen einzigen KV-Cache für den gesamten
Kontext. Anfragen über N hinaus warten weiterhin in der Warteschlange.

Die Weboberfläche fügt ihren Anfragen `"mgs_client": "webui"` hinzu. Während eine ihrer gestreamten Anfragen wartet,
beginnt der Stream mit Ereignissen wie

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: Anfragen, die vor ihr laufen oder warten; `ahead_api` und `ahead_webui`: wie viele davon direkt
an die API bzw. aus der Weboberfläche kamen) und einem letzten Ereignis mit allen Zählern auf 0, wenn sie beginnt. Clients, die
`mgs_client` nicht senden, erhalten diese Ereignisse nie, sodass OpenAI-kompatible Clients den üblichen Stream bekommen. Swagger
beschreibt das Feld zusammen mit den übrigen Completion-Parametern.

<a id="security-limits"></a>
### Sicherheitsgrenzen

Diese Grenzen schützen den Server und die Oberfläche vor feindseligen Anfragen, Dateien und Modellen. Sie sind großzügig
genug für die normale Verwendung: Was sie überschreitet, schlägt mit einer klaren Fehlermeldung fehl, statt den Server anzuhalten.

- **API-Schlüssel.** Die Routen für den Gesundheitsstatus und die Modellliste sind nur für `GET` und `HEAD` öffentlich. Das Neuladen der Modellliste
  (`GET /api/models?reload=1`) und das Hinzufügen oder Löschen von Modellen im Router-Modus (`POST` und `DELETE /api/models`)
  benötigen den Schlüssel, sobald einer konfiguriert ist.
- **Seiten anderer Websites.** Wenn Serverwerkzeuge oder MCP-Server aktiviert sind (`--tools`, `--agent`,
  `--mcp-servers-config`, `--mcp-servers-json` oder `--local-shell-mcp-server`) und `--cors-origins` nicht gesetzt ist,
  antwortet der Server mit 403 auf jede Anfrage außer `GET`, `HEAD` oder `OPTIONS`, deren `Origin`-Header
  eine andere Website nennt. Die Oberfläche (auch hinter Apache) und Clients, die kein `Origin` senden, etwa curl oder die SDKs,
  sind nicht betroffen. `POST /api/tools` akzeptiert nur `Content-Type: application/json` (sonst 415).
- **CORS-Proxy.** `/api/cors-proxy` verweigert Anfragen, die ein Browser als Seite, Frame oder Worker öffnen würde
  (403), und gibt jede Antwort als inerten Inhalt zurück: Er kann keine Skripte ausführen und keine Cookies auf der Adresse des Servers setzen.
- **Router-Auflistungen.** Im Router-Modus zeigen `status.args` und `status.preset` von `/api/models` `<redacted>`
  statt der Werte von `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  und `--mcp-servers-json`; das Protokoll verfährt ebenso. Die Modellprozesse erhalten weiterhin die echten Werte.
- **Langsame Clients.** Während der Server eine Anfrage empfängt, wartet jeder Lesevorgang höchstens 30 s (`--timeout` regelt weiterhin
  das Schreiben, sodass lange Antworten nicht abgeschnitten werden), und höchstens 256 angenommene Verbindungen warten auf einen freien Thread.
- **Anfragekörper.** JSON-Körper, die tiefer als 1024 Arrays oder Objekte verschachtelt sind, werden mit 400 abgelehnt. JSON-Schemas
  (`response_format`, `json_schema`) dürfen bis zu 256 Ebenen tief verschachtelt sein; leere `pattern`-Werte und Quantoren ohne
  etwas zu Wiederholendes werden abgelehnt.
- **Entfernte Medien.** URLs in `image_url`, `input_audio` und `input_video` dürfen nur auf öffentliche Internetadressen
  verweisen. Loopback, private Netze (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10), Link-Local-Adressen
  (einschließlich der Cloud-Metadatenadresse 169.254.169.254), eindeutige lokale IPv6-Adressen und andere
  nicht öffentliche Adressen werden mit 400 abgelehnt, auch wenn eine Weiterleitung zu ihnen führt. Um sie zuzulassen, etwa um
  Bilder von einem Server in Ihrem Netz zu lesen, starten Sie `mgs` mit `--media-allow-private-urls`
  (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); tun Sie dies nur, wenn jeder API-Client vertrauenswürdig ist. Jeder Download hat 10 s für den
  Verbindungsaufbau, insgesamt 30 s, bis zu 5 Weiterleitungen und 10 MiB, und der Server führt höchstens 8 Downloads gleichzeitig aus.
- **Dekodierte Medien.** Bilder bis zu 32768 Pixel pro Seite und 100 Megapixel, Audio bis zu einer Stunde, Videoframes bis
  zu 8192 Pixel pro Seite und 2 GiB dekodierter Frames pro Video sowie höchstens 2 GiB dekodierter Bilder und Audiodaten pro
  Anfrage.
- **Chat-Vorlagen des Modells.** `range()` erzeugt höchstens eine Million Elemente, erzeugte Zeichenketten bleiben unter
  256 MiB, die Einrückung von `indent()` und `tojson()` unter 256 Leerzeichen und die Verschachtelung unter 1024 Anweisungen; eine Vorlage,
  die sie überschreitet, schlägt mit einem Vorlagenfehler fehl.
- **RPC.** In einem Build mit RPC-Backend prüft der RPC-Server die Framegrößen und Tensorgrenzen, die seine
  Gegenstellen senden, und schließt nur die betroffene Verbindung. Machen Sie ihn nur vertrauenswürdigen Rechnern zugänglich: Er bedient weiterhin jeweils nur einen Client
  zur selben Zeit.
- **Oberfläche.** Links in Nachrichten behalten nur die Schemata http, https, mailto und tel (oder relative Adressen), und
  Bildanhänge müssen eingebettete Bilder sein. Der Import von Konversationen akzeptiert Dateien bis zu 512 MB, ZIP-Archive mit bis
  zu 10 000 Einträgen, 256 MB pro JSONL-Datei und insgesamt 1 GB entpackt. Eine Datei mit fehlerhaften Datensätzen oder mit einer
  Nachricht, die zu einer anderen Konversation gehört, wird vollständig abgelehnt; eine Konversation, deren Nachrichten bereits existieren,
  wird übersprungen und gemeldet.

Die Installationsbefehle laden weiterhin den aktuellen Branch `main` über HTTPS herunter; sie prüfen weder eine Signatur noch
eine feste Revision.

<a id="performance"></a>
## Speicher und Leistung

Die Registrierung von Host-Speicher und das Vorabladen der Experten sind standardmäßig aktiviert. Das Vorabladen
überlappt Expertenübertragungen mit der Berechnung, insbesondere bei großen Prompts.
Die Ergebnisse hängen vom Modell, von der Speicherkapazität und von der RAM-/PCIe-Bandbreite ab.

Bei Modellen mit auf die CPU ausgelagerten Experten können `--cpu-moe` oder `--n-cpu-moe N` den
VRAM-Verbrauch senken. Die Host-Registrierung fixiert eingeblendete Gewichtsseiten für schnellere Kopien vom Host zum Gerät.
Fixierter Speicher kann nicht ausgelagert werden, lassen Sie daher genug RAM für das Betriebssystem
frei. Dieser Registrierungsweg ist POSIX-spezifisch; Windows behält sein bisheriges
Ladeverhalten. `--load-mode none` wird nur empfohlen, wenn die Host-Registrierung ganz
oder teilweise fehlschlägt, nicht als normale Voraussetzung.

Das Vorabladen der Experten verwendet standardmäßig drei VRAM-Slots. Jeder Slot muss den größten
übertragenen Expertentensor aufnehmen können. Die Werte 2–8 in `GGML_SCHED_PREFETCH_EXPERTS` wählen
ausdrücklich eine Slotanzahl. Schlägt die Speicherreservierung fehl, werden nach Möglichkeit mindestens zwei reservierte Slots
behalten; andernfalls werden normale Übertragungen verwendet. Das Vorabladen beginnt erst,
wenn der Batch mindestens doppelt so viele Expertenauswahlen wie Experten enthält. Sein Pool
gehört der ersten GPU, die es aktiviert; andere GPUs behalten selektive Kopien.
Größere Batches können beim Prefill helfen, benötigen aber mehr Speicher.

Zur Diagnose setzen Sie `GGML_CUDA_REGISTER_HOST=0` oder
`GGML_SCHED_PREFETCH_EXPERTS=0`, um die entsprechende Optimierung zu deaktivieren.
Setzen Sie diese Variablen im normalen Betrieb nicht, es sei denn, Sie verzichten absichtlich darauf.

Das automatische MTP untersucht die GGUF-Metadaten und Tensornamen vor der Reservierung der Gewichte,
einschließlich jedes Shards. Es erfordert eine unterstützte Architektur, einen Modellstamm und
alle integrierten Köpfe. Ein externer Kopf oder eine nicht unterstützte Architektur
aktiviert es nicht. Die Standardgrenze für Entwürfe liegt bei drei Token. Eine ausdrückliche Auswahl mit `--spec-type`
hat Vorrang; `--no-spec-mtp-auto` und `--spec-type none` deaktivieren
die automatische Auswahl. `--spec-draft-n-max` ändert die Grenze. MTP wirkt sich auf die Generierung aus,
nicht direkt auf den Prefill oder die Zeit bis zum ersten Token. Die Verbesserung hängt von der Akzeptanzrate
und der Hardware ab; eine höhere Grenze garantiert keine höhere Geschwindigkeit.

<a id="mtp-vram"></a>
MTP-Gewichte bleiben immer im GPU-Speicher, wenn die GPU sie aufnehmen kann: die auf das Modell aufgepfropften MTP-Schichten, ein
separates MTP-GGUF, das mit `--spec-draft-model` übergeben wird, und der Ausgabekopf, den sie verwenden. Die Optionen, die Gewichte in den
Arbeitsspeicher verschieben (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, ihre `--spec-draft-*`-Entsprechungen und die automatische
Anpassung), gelten für sie nicht, und sie bleiben auch bei einem niedrigen `--gpu-layers` auf der GPU. Die automatische Anpassung platziert den
Rest des Modells um sie herum, und der [MoE-Expertencache](#expert-cache) verwendet den danach verbleibenden Speicher. Passen die MTP-Schichten
mit einer Reserve von 512 MiB nicht in den freien GPU-Speicher, zeigt das Protokoll eine Warnung an, und sie werden wie jede
andere Schicht platziert. Mit einem separaten MTP-GGUF werden die auf das Hauptmodell aufgepfropften MTP-Schichten nicht geladen, sodass sie nicht
doppelt Speicher belegen; mit `--spec-type none` wird kein MTP-Gewicht geladen. Das Ladeprotokoll bestätigt die Platzierung mit einer Zeile wie
`keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)`, sichtbar mit `-lv 4`.

<a id="expert-cache"></a>
### MoE-Expertencache

Wenn MoE-Experten im Arbeitsspeicher bleiben und der Rest der Schicht auf einer GPU läuft, hält der Server die zuletzt
verwendeten Experten aller Schichten in einem Cache, der den freien VRAM belegt;
bei mehreren GPUs speichert jede GPU die ihr zugewiesenen Schichten in ihrem eigenen VRAM. Für jedes generierte Token (und jeden
Batch unter 32 Token) laufen die im Cache gefundenen Experten auf der GPU, ein Teil der fehlenden wird
in den Cache hochgeladen, und der Rest läuft gleichzeitig auf der CPU; der Anteil richtet sich nach den gemessenen PCIe- und CPU-Bandbreiten.
Lange Prompts behalten das Vorabladen der Experten, das die zwischengespeicherten Experten nun innerhalb der GPU kopiert und nur
die übrigen hochlädt. Die Ausgabe stimmt bis auf Rundung mit dem Lauf ohne Cache überein. Die GPU und die CPU runden unterschiedlich, und welche Experten
jede von ihnen berechnet, hängt vom Inhalt des Caches ab, sodass eine lange Greedy-Generierung von einem Lauf zum nächsten nach einigen Token
zu einem anderen, aber ebenso gültigen Text abweichen kann; der Cache des offiziellen llama.cpp verhält sich genauso.

Der Cache ist standardmäßig aktiviert und wird von der ersten Anfrage erstellt, sobald das Modell, der MTP-Kontext und der
multimodale Projektor geladen sind. Ohne `--gpu-layers`, `--n-cpu-moe` oder `--override-tensor` hält die automatische
Anpassung jeden Experten im Arbeitsspeicher, wenn er mit einer Reserve (4 GiB oder 10 % des RAM) hineinpasst, und überlässt den freien VRAM dem Cache (bei mehreren GPUs verteilt sie die Schichten
nach ihrem freien Speicher auf sie, sodass jede GPU einen Cache hat); andernfalls platziert sie wie bisher ganze Expertenschichten auf der GPU, und der Cache verwendet, was
übrig bleibt. Ausdrückliche Platzierungsoptionen werden respektiert: `--cpu-moe` gibt dem Cache den meisten VRAM, und
`--n-cpu-moe N` behält einige statische Schichten auf der GPU. Die im RAM gehaltenen Experten sind fixiert, daher muss der RAM
sie alle plus das Betriebssystem aufnehmen.

<a id="context-size"></a>
Wenn das Modell nicht in den VRAM passt und `-c` nicht angegeben ist, wählt die automatische Anpassung auch die Kontextgröße: Der
KV-Cache belegt höchstens ein Viertel des nach den dichten Gewichten verbleibenden VRAM, bis zum Trainingskontext des
Modells, und die Experten (der Cache oder ganze Expertenschichten) erhalten den Rest. Früher erhielten solche Modelle immer das
Minimum von 4096 Token, das sich alle Slots teilten, wodurch lange Antworten abgeschnitten wurden (`truncated = 1` im Protokoll). Das
Protokoll zeigt das Ergebnis in einer Zeile wie `context size raised from 4096 to N tokens` und in `n_ctx_slot` der
Zeile `initializing`. `-c N` legt den Kontext ausdrücklich fest (mehr Kontext lässt dem Cache weniger VRAM und senkt die
Generierungsgeschwindigkeit: 35,8 Token/s mit 4096 Token und 33,9 mit 122624 auf dem Referenzrechner), und
`--fit-ctx N` erhöht das Minimum. Der MTP-Kontext erhält dieselbe Größe wie der Modellkontext, statt
VRAM für den gesamten Trainingskontext zu reservieren.

Vor dem Kontext wählt die automatische Anpassung den Micro-Batch (`-ub`), die Anzahl der Prompt-Token, die in
einem Durchgang verarbeitet werden. Mit den Experten im RAM lädt jeder Durchgang die Experten, die nicht im Cache sind, auf die GPU hoch, sodass weniger,
größere Durchgänge lange Prompts viel schneller verarbeiten, während sich die Generierung nicht ändert. Sofern weder `-b` noch `-ub` angegeben ist,
nimmt sie den größten Wert von 4096, 2048 und 1024 Token, dessen zusätzliche Puffer in ein Achtel des nach den
dichten Gewichten verbleibenden VRAM passen, und erhöht den Batch (`-b`) entsprechend. Auf dem Referenzrechner stieg ein Prompt mit 2514 Token mit 4096 von 214 auf
755 Token/s, bei gleicher Generierungsgeschwindigkeit; die größeren Puffer verkürzen den Standardkontext (etwa 90000
Token ohne MTP und 61000 mit MTP bei 16 GB). Das Protokoll zeigt die Wahl in einer Zeile wie `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

| Option | Variable | Wirkung |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE` (`0` deaktiviert) | Aktiviert oder deaktiviert den Cache (standardmäßig aktiviert). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | Feste Gesamtgröße des Caches, aufgeteilt auf die GPUs nach den Experten, die jede zwischenspeichert; `-1` (Standard) verwendet den freien VRAM jeder GPU. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | Auf jeder GPU neben dem Cache und den Vorablade-Slots frei gelassener VRAM (Standard 1024). Erhöhen Sie ihn, wenn andere Programme die GPU verwenden. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnose: fester Anteil (0–1) fehlender Experten, der statt des gemessenen Gleichgewichts hochgeladen wird. |
| — | `MGS_MOE_CACHE_STATS` | Diagnose: protokolliert die Cache-Zähler alle N Dekodierschritte (sichtbar mit `-lv 4`). |

`GET /api/expert-cache` gibt die Größe, die zwischengespeicherten Schichten und Slots, Treffer und Fehltreffer, Uploads, CPU-Experten, den
Anteil der aus dem Cache bedienten Vorablade-Bytes und die gemessenen Bandbreiten zurück (bei mehreren GPUs
werden Größen und Zähler addiert, und die Upload-Bandbreite ist ihr Mittelwert). `POST /api/expert-cache` mit
`{"size_mib": N}` ändert die Größe ohne Neustart (`-1` automatisch, `0` gibt den Cache frei); die nächste Anfrage
wendet sie an und beginnt mit einem leeren Cache:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Mit `--metrics` fügt `/api/metrics` Zähler und Messwerte `mgs:moe_cache_*` hinzu (Schritte, Treffer, Fehltreffer, Uploads,
CPU-Experten, Vorablade-Bytes, Größe, Slots und Bandbreiten).

Referenzmessung auf einer RTX 4060 Ti 16 GB mit Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 CPU-Threads), verglichen mit
`--n-cpu-moe 28` ohne Cache: Generierung 26,9 → 35,7 Token/s (+33 %), Prompt-Verarbeitung 767–778 →
718–730 Token/s (−6 %) und mit aktiviertem MTP 33,6 → 47,0 Token/s (+40 %). Auf demselben Rechner erreichte das offizielle
llama.cpp mit seinem eigenen Cache (`-cmoe --moe-cache-mib 10240`) 26,7 Token/s und 27,8 mit MTP. Die Gewinne hängen vom Anteil der Experten ab,
der in den VRAM passt, von der Routing-Lokalität des Modells und von den PCIe- und RAM-Bandbreiten.

<a id="threads"></a>
### CPU-Threads

Die CPU berechnet die Experten, die nicht im Cache sind. Diese Arbeit ist durch die Speicherbandbreite begrenzt und
ist schon mit wenigen Threads gesättigt, während die Verwendung jeder logischen CPU die Threads mit demjenigen konkurrieren lässt, der
die GPU ansteuert. Ohne `-t` verwendet `mgs` die für Berechnungen nützlichen Kerne, aber nie mehr als die Hälfte der logischen CPUs, auf denen es
laufen darf. Auf einem Rechner mit SMT (zwei Threads pro Kern) ist das wie bisher die Anzahl der physischen Kerne; in einer
virtuellen Maschine, die kein SMT anzeigt, ist es die Hälfte der virtuellen CPUs. Auf dem Referenzrechner (16 virtuelle
Kerne mit einer Desktopsitzung, Qwen3.6-35B-A3B Q8_0, ohne MTP) generierten 4 bis 12 Threads mit derselben Geschwindigkeit
(32,9–33,9 Token/s) und 16 Threads mit 28,7 Token/s. Das Startprotokoll zeigt die Wahl in einer Zeile wie
`using 8 CPU threads for generation and 8 for batches`; `-t` und `-tb` legen sie ausdrücklich fest.

<a id="checkpoints"></a>
### Kontext-Prüfpunkte

Modelle mit rekurrenten Schichten oder Schichten mit gleitendem Fenster (zum Beispiel die Hybride Qwen3.5/3.6/3.8, Nemotron-H oder Granite 4)
können ihren Zustand nicht zurücksetzen; wenn eine Anfrage einen früheren Teil der Konversation ändert, setzt der
Server daher beim neuesten Prüfpunkt vor der Änderung fort. Prüfpunkte werden am Anfang von Benutzer-,
Assistenten- und Werkzeugnachrichten gesetzt (wo Coding-Agenten alten Denkprozess entfernen oder alte Werkzeugergebnisse kürzen) und mindestens
alle `--checkpoint-min-step` Token (Standard 1024, früher 8192), ohne zusätzliche Prompt-Durchgänge. Wenn
`--ctx-checkpoints` (32) erreicht ist, wird der Prüfpunkt entfernt, der seinen Nachbarn am nächsten liegt, sodass die übrigen
über die Konversation verteilt bleiben. Im Referenzszenario mit einem Agenten (14k Token, acht Werkzeugergebnisse) sank das Kürzen eines späten
Werkzeugergebnisses von 19,3 s auf 5,5 s und das eines frühen von 18,0 s auf 13,6 s. Jeder Prüfpunkt hält den
rekurrenten Zustand oder den Zustand des gleitenden Fensters im Host-Speicher (63 MiB für Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Geänderte Optionen und Diagnose

Behalten Sie den Standard `--load-mode mmap` bei, wenn die Host-Registrierung gelingt. Verwenden Sie `--load-mode none` nur als Ausweichlösung, wenn die Registrierung fehlschlägt. `--load-mode dio` puffert Lesevorgänge nun über höchstens 64 MiB zusätzlichen Speicher; das ist keine Obergrenze für den gesamten RAM des Modells.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` wurde entfernt, weil es zur Laufzeit keine Wirkung hatte. Die alten Lade-Flags (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) und das Build-Flag `GGML_CUDA_FA_ALL_QUANTS` werden nicht mehr akzeptiert; ihre Ersetzungen finden Sie unter [entfernte Kompatibilität](#removed-compatibility). Bei einem manuellen CMake-Build fügt `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` diese K/V-Kombination der Vektorkernel hinzu; kommagetrennte oder in Anführungszeichen gesetzte, durch Semikolon getrennte Listen werden akzeptiert. Die Standardliste ist `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 ist immer enthalten. `all` erhöht den Kompilierungsaufwand. Fehlende Kombinationen der Vektorkernel verwenden eine Ausweichlösung mit F16-Konvertierung und können langsamer sein.

Vollständige GGUFs von GLM-4.5-Air (`glm4moe`) mit einem einzigen Kopf können nun das automatische MTP aktivieren. Die übrigen bestehenden Eignungsprüfungen und der Vorrang ausdrücklicher Optionen gelten weiterhin. MTP trennt Graphen mit und ohne Ausgaben, erhält die ursprüngliche Tokenreihenfolge über gleichzeitige Sequenzen hinweg und verwendet nach Bildern die tatsächlichen Positionen.

Die automatische Anpassung zählt die nextn-Schichten (MTP) mit, sodass die erste Schicht nicht mehr auf der CPU bleibt, wenn MTP deaktiviert ist. MTP-Gewichte bleiben nun immer im GPU-Speicher; siehe [MTP-Platzierung](#mtp-vram). Der Standardwert von `--checkpoint-min-step` ist nun 1024 statt 8192; siehe [Kontext-Prüfpunkte](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` und `--moe-cache-reserve` steuern den [MoE-Expertencache](#expert-cache). Die automatische Anpassung lässt Modelle, die größer als der VRAM sind, nicht mehr mit einem Kontext von 4096 Token zurück und wählt nun auch den Micro-Batch; siehe [Kontextgröße](#context-size). Die Standardanzahl der CPU-Threads beträgt nun höchstens die Hälfte der logischen CPUs; siehe [CPU-Threads](#threads). `-np` verwendet nun standardmäßig einen Slot mit dem gesamten Kontext statt des automatischen Modus mit vier Slots, die sich einen gemeinsamen KV-Cache teilen, und `--kv-unified` ist standardmäßig deaktiviert; siehe [Anfragewarteschlange](#request-queue).

Mit `--metrics` enthält ein authentifiziertes `GET /api/metrics` die Werte `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` und `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Die Positionsindizes beginnen bei null; diese Reihe erscheint, nachdem eine spekulative Anfrage abgeschlossen ist. Vergleichen Sie bei der Bewertung der Akzeptanz die Differenzen der Zähler für akzeptierte und entworfene Token, und messen Sie die Latenz separat; die Akzeptanz allein beweist keine Beschleunigung. Wählen Sie im Router-Modus das Modell mit `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, standardmäßig deaktiviert) erlaubt den Medien-URLs von Anfragen, Loopback-, private und andere nicht öffentliche Adressen zu erreichen; siehe [Sicherheitsgrenzen](#security-limits).

Konversationsexporte lesen den gespeicherten vollständigen Nachrichtenbaum und die Metadaten. Wenn Serverwerkzeuge 403 zurückgeben, wiederholen neue Nachrichten diese fehlgeschlagene Abfrage nicht mehr; das erneute Öffnen des Werkzeugbereichs versucht es erneut. Die Serverstandards beim ersten Besuch erhalten Einstellungen, die der Benutzer bereits geändert hat, einschließlich des API-Schlüssels.

[Implementierung und Upstream-Quellen](CODE.de-DE.md#upstream-review).

<a id="removed-compatibility"></a>
## Entfernte Kompatibilität

Die Kompatibilitätsschichten für ältere Versionen wurden entfernt. Alte Namen werden weder
übersetzt noch mit einer Warnung bedacht: Der Server lehnt entfernte Befehlszeilenoptionen als
ungültige Argumente ab, ignoriert entfernte Umgebungsvariablen und antwortet bei
entfernten Routen mit 404. Aktualisieren Sie Skripte, Dienst-Overrides, Voreinstellungen und Clients auf die
folgenden Ersetzungen.

| Entfernt | Stattdessen verwenden |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (oder `mmap+mlock`, um beides zu kombinieren) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Nichts; es hatte keine Wirkung. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Alte Aliase für Entwurfsmodelle wie `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` oder `--draft-p-min` | Die passende Option `--spec-draft-*`, zum Beispiel `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` oder `--spec-draft-p-min`. `mgs --help` listet sie alle auf. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` oder `--spec-ngram-mod-n-max` / `--spec-draft-n-min` oder `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | Die Option `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` bzw. `--spec-ngram-*-min-hits` des gewählten N-Gramm-Modus |
| Umgebungsvariablen, CMake-Optionen und Symbole der C-API mit dem früheren Upstream-Präfix, einschließlich der Aliase, die frühere Versionen bereits aufgegeben hatten | Der aktuelle Name mit dem Präfix `MGS_` (`mgs_` in der C-API), zum Beispiel `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` statt einer `NO_`-Variante, `MGS_ARG_LOAD_MODE` statt der Variablen für Memory-Mapping, Sperren und Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` und `MGS_ARG_SPEC_DRAFT_N_MIN` für die Entwurfsgrenzen, `MGS_CACHE` oder `MGS_BUILD_UI`; für die Backend-Optionen die passende Option `GGML_*`, zum Beispiel `GGML_CUDA` oder `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` oder `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` oder `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Wert `deepseek-legacy` von `reasoning_format` (Anfragefeld und `--reasoning-format`) | `deepseek` oder `auto`; der Denkprozess wird in `reasoning_content` zurückgegeben |
| Anfragefeld `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (Array von Zeichenketten) |
| `reasoning_in_content` in `generation_settings` | Nichts; das Feld wurde entfernt. |
| Download einer vorkompilierten Upstream-Oberfläche aus einem Hugging-Face-Bucket (die CMake-Optionen `USE_PREBUILT_UI` und `UI_HF_BUCKET` und die Umgebungsvariable `HF_UI_VERSION`) | Nichts; die Oberfläche wird immer aus `frontend/` kompiliert (`MGS_BUILD_UI=ON`) oder weggelassen. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Build-Nummer und Commit: die CMake-Variablen `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (auch in `mgs-config.cmake`) und die Zeichenkette `b0-unknown` von `build_info`, `system_fingerprint` und dem `User-Agent` | `MGS_VERSION`, festgelegt in `backend/cmake/build-info.cmake`; diese Felder enthalten nun die Version, etwa `0.1.0` |
| Meta-Llama-Modelle (die GGUF-Architekturen `llama`, `llama4` und `llama-embed` und der Bildprojektor von Llama 4) sowie die integrierten Chat-Vorlagen `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` und `llama4` | Nichts; verwenden Sie eines der unterstützten MoE-Modelle. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Installationsoption `--no-sudo` | Nichts; die Installationsprogramme laufen immer als root ohne sudo. |
| Funktionen und Felder, die die C-APIs als veraltet markiert hatten: in `mgs.h` `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, die Aliase `mgs_token_*` (zum Beispiel `mgs_token_bos`, `mgs_token_get_text` oder `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, das Feld `defrag_thold` von `mgs_context_params` und `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; in `ggml.h` `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` und `ggml_upscale_ext`; in `mtmd.h` `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` und das Feld `image_marker` | Die aktuelle Funktion: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` statt `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` und `media_marker`. Aufwärmläufe erledigt der Aufrufer, und `defrag_thold` hatte keine Wirkung. |
| Gemma 4: die Architekturen `gemma4` und `gemma4-assistant`, die Projektoren `gemma4v`, `gemma4uv`, `gemma4a` und `gemma4ua`, das Chat-Format `peg-gemma4`, der Tokenizer und Pre-Tokenizer `gemma4` sowie die Voreinstellungen `--vision-gemma-4b-default` und `--vision-gemma-12b-default` | Nichts; verwenden Sie eine der [unterstützten Familien](#supported-models). |
| DeepSeek-Modelle vor V4.1: die Architekturen `deepseek`, `deepseek2` (auch die damit konvertierten Dateien von Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 und Youtu), `deepseek2-ocr` (mit den Projektoren `deepseekocr` und `deepseekocr2`), `deepseek32`, `mistral4` und `deepseek4`, die Pre-Tokenizer `deepseek-llm` und `deepseek-coder` sowie das DSpark-Entwurfs-Backbone von DeepSeek-V4 | DeepSeek-V4.1-Flash (`deepseek41`). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) und die integrierten Vorlagen der Nicht-Jinja-Engine (`--chat-template` mit einem Namen wie `deepseek3`, `chatglm4` oder `mistral-v7-tekken`) | Nichts; Chat-Vorlagen laufen immer über Jinja. `--chat-template chatml` wählt weiterhin ChatML; für eine andere Vorlage übergeben Sie ihren Jinja-Quelltext mit `--chat-template` oder `--chat-template-file`. Anfragen mit `tools` benötigen kein Flag mehr. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| Zweite lange Namen einer Option: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk` (die Schlüssel der Router-Voreinstellungen folgen denselben Namen) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, benannt nach ihren Optionen |
| Anfragefeld `thinking_budget_tokens` | `reasoning_budget_tokens` (Anthropic-Clients senden weiterhin `thinking.budget_tokens`) |
| `content` in `POST /api/embeddings` und `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, die Struktur `mgs_chat_message` und `MGS_FILE_MAGIC_GGLA` in `mgs.h`; `mtmd_image_tokens_get_id` in `mtmd.h` | Nichts; die Chat-Formatierung übernehmen die Jinja-Vorlagen von `common/chat.h`, und die übrigen hatten keine Verwendung. |

CMake meldet entfernte `-D`-Variablen als unbenutzt, statt sie anzuwenden; prüfen Sie daher
die Konfigurationsausgabe manueller Builds.

Die Weboberfläche speichert ihre Einstellungen auf dem Server (`~/.mgs/config/ui-settings.json`)
und im Browser nur ihren API-Schlüssel (`MgsUi.apiKey` in `localStorage`) und ihre
Konversationen (die IndexedDB-Datenbank `MgsUi`). Sie liest oder migriert die
Einstellungen nicht, die ältere Versionen in `localStorage` gespeichert haben (`MgsUi.config`,
`MgsUi.userOverrides`, die Lieblingsmodelle, die Werkzeuglisten, den Denkaufwand,
die verworfenen MCP-Empfehlungen und die Sprache), sodass die Einstellungen mit
ihren Standardwerten beginnen und der API-Schlüssel einmal erneut eingegeben werden muss. Ebenso wenig liest oder
migriert sie Daten, die ältere Versionen unter anderen Namen gespeichert haben, den eigenständigen Schlüssel `theme`,
eingebettete Markierungen für Denkprozess und Werkzeugaufrufe in alten Nachrichten, Anhänge mit eingefügtem Text
vom alten Typ `context`, das alte Listenformat der verworfenen MCP-Empfehlungen
oder das frühere Feld `webui_settings` des Servers. Um die
Konversationen einer älteren Version zu behalten, exportieren Sie sie vor dem Upgrade aus dieser Version und importieren
den Export anschließend. Der Import von Konversationen akzeptiert nur die JSONL- und ZIP-Exporte, innerhalb der unter
[Sicherheitsgrenzen](#security-limits) aufgeführten Grenzen; das alte JSON-Exportformat wird abgelehnt. Die alten Einstellungen bleiben im Browser, bis sie gelöscht werden: Um
sie zu entfernen, öffnen Sie auf der Seite der Oberfläche die Browserkonsole (F12) und führen
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
einmal für jede Adresse und jeden Port aus, mit denen die Oberfläche geöffnet wurde.

Mit `-mu`/`--model-url` heruntergeladene Dateien werden in `~/.cache/mgs/` zwischengespeichert, sodass die
Dateien, die ältere Versionen in ihrem früheren Cache-Ordner gespeichert haben, erneut heruntergeladen werden.
Downloads von Hugging Face verwenden weiterhin den Hugging-Face-Cache (`~/.cache/huggingface/hub`
oder den Ordner in `MGS_CACHE`, `HF_HUB_CACHE` oder `HF_HOME`).

### Alte Modelldateien

Auch der Kompatibilitätscode für die folgenden alten Konvertierungen wurde entfernt.
Konvertieren Sie das ursprüngliche Modell erneut oder laden Sie ein aktuelles GGUF herunter, um sie zu verwenden.

| Alte Datei | Jetziges Verhalten |
| --- | --- |
| Grok-1, konvertiert vor August 2025 (ohne `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` oder `grok.attn_logit_softcapping`) | Lädt nicht: fehlender Schlüssel. |
| Kimi-Linear mit dem ungeteilten Tensor `blk.N.attn_kv_b` | Lädt nicht: `attn_k_b`/`attn_v_b` fehlen. Dieser Weg schlug bereits beim Aufbau des Graphen fehl. |
| Qwen3-Next-Dateien nur mit dem zusammengeführten Tensor `ssm_in` | Lädt nicht: fehlender Tensor `blk.0.attn_qkv.weight`. |
| Dateien ohne `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) oder `full_attention_interval` (Qwen3-Next) sowie Cohere2-MoE-Dateien mit einer einzigen SWA-Periode statt des schichtweisen Arrays | Lädt nicht: fehlender Schlüssel. |
| BPE-Vokabulare ohne `tokenizer.ggml.pre` oder mit den Pre-Tokenizer-Namen `llama3`, `llama-v3`, `jina-es`, `jina-de` oder `llada-moe` sowie Kimi-K2-Tokenizer ohne Merges | Lädt nicht: fehlender Schlüssel, unbekannter Pre-Tokenizer-Typ oder fehlende Merges. |
| Dateien der GGUF-Version 1 oder 2 | Lädt nicht: Nur die GGUF-Version 3 wird akzeptiert. |
| mmproj-Dateien der alten Einzelskripte: Projektortypen `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x bis 4.5, MiniCPM-o 2.6) und `adapter` (GLM-Edge-V) sowie Dateien ohne Projektortyp | Der Projektor lädt nicht: unbekannter oder fehlender Projektortyp. |
| mmproj-Dateien alter Konvertierungen mit vertauschtem `ffn_up` und `ffn_down` | Wird nicht mehr korrigiert; der Projektor schlägt fehl. |
| Kimi-K2.5-mmproj ohne `image_min_pixels`/`image_max_pixels` und Step3-VL ohne `preproc_image_size` | Der Projektor lädt nicht: fehlender Schlüssel oder Tensor. |
| Dateien, die nur die alten Schlüssel `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` und `tokenizer.ggml.middle_token_id` statt der Schlüssel `tokenizer.ggml.fim_*_token_id` enthalten | Lädt; die Infill-Token werden an ihrem Text erkannt (zum Beispiel `<PRE>` oder `<\|fim_prefix\|>`), und Infill funktioniert nicht, wenn sie nicht gefunden werden. |

Beibehalten wurden die Ausweichlösungen, von denen aktuelle Konvertierungen oder die meistheruntergeladenen veröffentlichten GGUFs der
unterstützten Familien noch abhängen, geprüft anhand ihrer Header auf
Hugging Face: das standardmäßige `add_bos` der Pre-Tokenizer `llama-bpe`, Tekken und ähnlicher;
die aus `full_attention_interval` abgeleitete Anordnung der rekurrenten Schichten von Qwen3.6 (die meistheruntergeladenen GGUFs von Qwen3.6 haben kein
`recurrent_layers`); und die integrierte Indexer-Anordnung von GLM-5.2 für Dateien von GLM-5.2/5.3
ohne `attention.indexer.types`.

<a id="maintenance"></a>
## Wartung und Fehlerbehebung

- Prüfen Sie mit `mgs --version`, welche Version installiert ist (es gibt zum Beispiel `version: 0.1.0` aus); der lokale Build
  führt den Befehl am Ende aus, und CMake gibt beim Konfigurieren `mgs version: 0.1.0` aus. Die Oberfläche zeigt dieselbe Version
  unter Einstellungen → Anzeige → **Informationen zur Build-Version anzeigen** (untere rechte Ecke) und unter **Build-Informationen** im
  Dialog mit den Modellinformationen. Jede Änderung am Projekt erhöht die letzte Zahl: 0.1.0, 0.1.1, 0.1.2 usw.
- Wiederholen Sie `curl | bash`, um von `main` zu aktualisieren, oder den passenden lokalen Installationsbefehl, um die Binärdatei neu zu kompilieren und zu ersetzen.
  Starten Sie den Dienst anschließend neu: `systemctl restart moe-gguf-server` unter Debian
  oder `rc-service moe-gguf-server restart` unter Alpine.
- Lesen Sie `/root/webapp-install.log`, `journalctl -u moe-gguf-server` unter Debian
  oder `/var/log/moe-gguf-server.log` unter Alpine. Die Apache-Protokolle liegen unter
  `/var/www/DOMAIN-logs/`.
- Wenn die CUDA-Kompilierung nicht verfügbar ist, installieren Sie das Toolkit separat oder verwenden Sie
  `--cpu`. Wenn NCCL nicht gefunden wird, prüfen Sie die Verfügbarkeit des Repositorys und die CUDA-Version,
  oder wählen Sie ausdrücklich `--no-nccl`.
- Wenn die Kompilierung der Oberfläche fehlschlägt, prüfen Sie die Versionen von Node.js und npm, die Downloads der Abhängigkeiten
  und das Protokoll. Ersetzen Sie sie nicht durch eine vorkompilierte Upstream-Oberfläche.
- Verwenden Sie für Bereitstellungs-Builds nach Änderungen an der Struktur der Quellen ein frisches Verzeichnis innerhalb von `_/temp/`.
  Eine Projektkopie auf SSHFS kann ausführbare Dateien unterbinden; verwenden Sie in der Produktion einen lokalen Datenträger.
- Führen Sie für den lokalen Build den obigen Einzeiler erneut aus, um das aktuelle `main` herunterzuladen, oder
  `bash build/build-local.sh`, um Ihre lokalen Quellen zu verwenden, und starten Sie die von Hand gestartete Kopie
  neu. Lesen Sie die Terminalausgabe, wenn der Lauf fehlschlägt oder Bibliotheken
  außerhalb von `lib/` meldet. Die temporäre Datei `build.log` wird mit dem Build-Verzeichnis gelöscht.
- API-Fehler 404 in älteren Clients erfordern meist, die Basis-URL auf
  `/api/v1` zu ändern. Der installierte API-Schlüssel gehört in den Authorization-Header.

Nach den Regeln dieses Projekts führt der Programmierassistent keine Tests, Bereitstellungsläufe, GPU-Benchmarks oder automatisierten
Prüfungen in mobilen Browsern durch. Statische Syntaxprüfungen
sind keine Garantie für das Laufzeitverhalten. Siehe [CODE.de-DE.md](CODE.de-DE.md) für die
Implementierungsübersicht und die Erweiterungspunkte.

<a id="license"></a>
## Lizenz

`mgs` wird unter der [PolyForm NonCommercial License 1.0.0](../LICENSE) vertrieben. Sie dürfen es für jeden
nicht kommerziellen Zweck verwenden, verändern und weitergeben: persönliche Nutzung, Studium, Forschung, Hobbyprojekte
sowie die Nutzung durch gemeinnützige Organisationen, Bildungseinrichtungen, öffentliche Forschungseinrichtungen und
staatliche Einrichtungen. Eine kommerzielle Nutzung, etwa durch ein Unternehmen für seine Geschäftstätigkeit, erfordert eine
separate Lizenz von nipegun.

Die Lizenz deckt nur die Beiträge von nipegun ab. Die Teile, die von llama.cpp (MIT, The
ggml authors), FreeToken (Apache 2.0) und den mitgelieferten Bibliotheken stammen, behalten ihre eigenen Lizenzen; ihre
Hinweise befinden sich in `build/licenses/`, und der lokale Build installiert sie in
`~/.local/share/doc/mgs/licenses/`.
