# Benutzerhandbuch

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Inhalt: [Installation](#installation), [lokaler Build](#local-build), [Bereitstellung](#deployment),
[Chat](#chat), [Monitor](#monitor), [Modelle und Werkzeuge](#models), [Modelle auswählen](#settings-models), [Modellseite](#model-page), [unterstützte Modelle](#supported-models), [API](#api),
[Anfragewarteschlange](#request-queue), [Sicherheitsgrenzen](#security-limits), [Vulkan](#vulkan), [ROCm](#rocm), [Leistung](#performance), [entfernte Kompatibilität](#removed-compatibility),
[Wartung](#maintenance), [Lizenz](#license).

<a id="installation"></a>
## Installation und erster Start

Installieren Sie als root auf dem Produktionsrechner. Unter Debian/Ubuntu kann das
Installationsprogramm direkt über die Standardeingabe übergeben werden:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

Das Installationsprogramm stellt in einem einzigen Schritt bereit: Es kompiliert `mgs`, installiert es in
`/usr/local/bin/mgs` und konfiguriert die Inferenz- und Webdienste
([Bereitstellung](#deployment)). Es müssen vorher keine weiteren Dateien heruntergeladen werden. Das Einstiegsskript lädt das
vollständige Quellarchiv von `main` aus [diesem Repository](https://github.com/nipegun/moe-gguf-server/) herunter, prüft
es und synchronisiert eine verwaltete Kopie in `/opt/moe-gguf-server-source`. Downloads
und Build-Caches bleiben unter `_/temp/` des Projekts. Curl und CA-Zertifikate
werden benötigt, um das Einstiegsskript abzurufen; fehlende Bootstrap-Abhängigkeiten werden anschließend
automatisch installiert.

Build- und Weboptionen übergeben Sie so:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com --cpu --jobs 4
```

Die verwaltete Kopie liegt immer in `/opt/moe-gguf-server-source`, gehört root und hat den
Modus 700. Das Installationsprogramm übernimmt sie nur, solange sie leer ist, oder verwendet sie wieder, wenn sie seine
Herkunftsmarkierung trägt; fremde Inhalte werden abgelehnt. Eine Sperre verhindert gleichzeitige Installationen in
diese Kopie. Spätere `curl | bash`-Läufe synchronisieren die Quelldateien, einschließlich der Dateien,
die aus `main` entfernt wurden, und erhalten dabei `_/` und die Build-Caches. API-Schlüssel, Modelle und
Zertifikate liegen außerhalb dieses Quellbaums und bleiben erhalten. Ein fehlgeschlagener
Download, Build oder eine fehlgeschlagene Optionsprüfung installiert keine neue ausführbare Datei; führen Sie denselben Befehl erneut aus, um es
noch einmal zu versuchen. Es sind keine Git-Operationen erforderlich.

Ein eigenständig heruntergeladenes Skript verwendet denselben entfernten Modus. Wird
`bash deploy/install-update-reinstall-debian.sh` aus einer vollständigen lokalen Arbeitskopie ausgeführt,
kompiliert es diese Arbeitskopie, ohne Quellen herunterzuladen. Argumente nach `bash -s --`
werden an das Build-Installationsprogramm weitergereicht. `--help` benötigt weder root noch einen Download
der Quellen. `--no-system-deps` deaktiviert auch die Installation der Bootstrap-Pakete: curl,
CA-Zertifikate, tar, gzip, rsync, util-linux und coreutils müssen bereits vorhanden sein.
Unterprozesse des Installationsprogramms können das Skript nicht aus der Pipe lesen; alle Phasen teilen sich
einen gemeinsamen Ausgabestrom in `/root/webapp-install.log`.

Die Weboptionen werden vor dem Build geprüft, sodass ein Fehler nicht erst nach einer langen
Kompilierung auffällt: `--domain NAME` (standardmäßig die Ausgabe von `hostname -f` oder, wenn diese kein gültiger
Name ist, von `hostname` oder `uname -n`), `--certificate PATH` und `--private-key PATH` (immer zusammen) und
`--proxy-ip IP` (siehe [Bereitstellung](#deployment)). `/opt/moe-gguf-server-source/deploy/configure-web.sh`
bleibt verfügbar, um die Webkonfiguration später ohne Neukompilierung zu ändern.

Alpine verwendet weiterhin eine vollständige lokale Arbeitskopie und
`bash deploy/install-update-reinstall-alpine.sh`, wobei Bash zuvor installiert werden muss; es
baut nur für die CPU. Unter Debian wählen Sie das Backend:

- CUDA (`--cuda`, der Standard) für NVIDIA-GPUs. Das CUDA Toolkit und der NVIDIA-Treiber
  müssen bereits installiert sein; das Installationsprogramm fügt die NCCL-Pakete hinzu, die zu dieser
  CUDA-Version passen (siehe unten). `--no-nccl` deaktiviert NCCL ausdrücklich.
- ROCm (`--rocm`) für von ROCm unterstützte AMD-GPUs; `--rocm-targets LIST` wählt die
  GPU-Targets. ROCm muss bereits installiert sein ([ROCm](#rocm)).
- Vulkan (`--vulkan`) für GPUs von AMD, Intel oder NVIDIA ([Vulkan](#vulkan)).
- Nur CPU (`--cpu`) für Rechner ohne GPU.

`--no-system-deps` überlässt die Paketverwaltung dem Administrator.

Die lokale Oberfläche benötigt Node.js 20.19+ in der Reihe 20.x oder 22.12+. `--no-ui` lässt sie weg. Weitere
Optionen sind `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST` und `--no-openssl`. `CUDACXX` wählt den CUDA-Compiler
und `HIPCXX` den clang-Compiler von ROCm. Build- und Staging-Verzeichnisse müssen verschiedene,
sich nicht überschneidende Unterverzeichnisse des Verzeichnisses `_/temp/` des Projekts sein.

Das Installationsprogramm fügt bei Bedarf passende NCCL-Pakete und das Repository von NVIDIA hinzu,
installiert oder aktualisiert aber niemals das CUDA Toolkit, ROCm oder GPU-Treiber. Es bindet die internen
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
bash deploy/install-update-reinstall-debian.sh --domain ai.example.com --cpu
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

Führen Sie `mgs` mit Ihrem eigenen Benutzer aus: Es braucht nie root. Als root gestartet, protokolliert es die
Warnung `mgs is running as root: run it with an unprivileged user instead` (der
lokale Shell-MCP-Server würde dem Modell eine root-Shell geben) und startet weiter.

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

Bis Version 0.1.34 brach die Suche beim zweiten Port ab: Ein `mgs`, das gestartet wurde, während 11443 belegt war (zum Beispiel durch ein anderes `mgs`), beendete sich mit `couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 11444`, statt den nächsten freien Port zu nehmen.

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
| `~/.local/lib/mgs/` | Die Server-Binärdatei (`mgs`, mit eingebetteter Weboberfläche und API-Dokumentation) und jede gemeinsam genutzte Bibliothek, die sie benötigt: der dynamische Loader und die C-Bibliothek von glibc, die C++-Laufzeitbibliothek, OpenMP, OpenSSL, die Namensdienstmodule von glibc und in CUDA-Builds die CUDA-Laufzeit, cuBLAS und NCCL; in ROCm-Builds die ROCm-Bibliotheken (HIP, hipBLAS, rocBLAS, hipBLASLt…) und die GPU-Kernelordner von rocBLAS und hipBLASLt (`rocblas/library`, `hipblaslt/library`). |
| `~/.local/share/doc/mgs/licenses/` | Die Lizenz PolyForm NonCommercial 1.0.0 des Projekts (`LICENSE`) und die Lizenzhinweise Dritter, kopiert aus `build/licenses/` des Quellbaums (llama.cpp, FreeToken, Swagger UI, cpp-httplib und nlohmann/json). |

Nur der Benutzer, der das Skript ausführt, kann diese Installation verwenden. Alles wird relativ zum Ort des Starters
aufgelöst: Er führt `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` aus seinem Präfix aus, und der Server und seine Bibliotheken durchsuchen
`lib/mgs/` vor allem anderen, einschließlich `LD_LIBRARY_PATH`. Die einzige Bibliothek, die weiterhin vom System genommen wird, ist der
NVIDIA-Treiber (`libcuda.so.1` und die von ihm geladenen `libnvidia-*`-Bibliotheken), weil er zum laufenden Kernelmodul
passen muss; mit `--vulkan` außerdem der Vulkan-Treiber des Systems ([Vulkan](#vulkan)); mit `--rocm` wird vom System nur der
amdgpu-Kerneltreiber (`/dev/kfd`, `/dev/dri`) verwendet, und der Starter verweist rocBLAS und hipBLASLt auf die in
`lib/mgs/` mitgelieferten Kernelordner ([ROCm](#rocm)). Am Ende listet das Skript über den mitgelieferten Loader die Bibliotheken auf, die der Server laden würde, schlägt fehl, wenn eine davon
von außerhalb von `lib/mgs/` stammt, und führt `--version` über den Starter aus.

Debian fügt `~/.local/bin` bei der Anmeldung zum `PATH` hinzu, sobald dieser Ordner existiert (über `~/.profile`), sodass Sie nach einer erneuten
Anmeldung `mgs` über seinen Namen ausführen können; das Skript gibt einen Hinweis aus, solange es nicht im `PATH` liegt. Starten Sie es, wann immer Sie
möchten, und beenden Sie es mit Strg+C:

```bash
mgs --model /path/model.gguf
```

Der vollständige Pfad funktioniert ebenfalls: `~/.local/bin/mgs --model /path/model.gguf`.

Öffnen Sie `https://127.0.0.1:11443/` für den Chat und `https://127.0.0.1:11443/api/doc/` für
die API-Dokumentation. Jede Option dieses Handbuchs gilt. Führen Sie `mgs` ohne `--model` aus, um den
Router für mehrere Modelle zu starten, und wählen Sie den Modellordner unter Einstellungen → Modelle
([Modelle auswählen und laden](#settings-models)).

Voraussetzungen, die Sie selbst als root installieren, weil das Skript sie nur
prüft: `cmake`, ein C/C++-Compiler und die statische C-Bibliothek (`build-essential`, das
`libc6-dev` mitbringt), `patchelf`, `libssl-dev` (oder
`--no-openssl`), Node.js 20.19+/22.12+ mit npm (oder `--no-ui`) und das Toolkit des
gewählten Backends: für CUDA (der Standard) ein vorhandenes CUDA Toolkit und NCCL mit seinen
Entwicklungsdateien (`libnccl2` und `libnccl-dev`, oder `--no-nccl`); für `--rocm` ein
vorhandenes ROCm ([ROCm](#rocm)); für `--vulkan` die Vulkan-Pakete ([Vulkan](#vulkan));
`--cpu` benötigt keines. `nvcc` wird aus `CUDACXX`, dem `PATH` oder
`/usr/local/cuda/bin/nvcc` genommen. Ninja wird verwendet, wenn es installiert ist.
Zu den grundlegenden Werkzeugen gehören `tar`, `coreutils` und `util-linux`. Entfernte Builds benötigen außerdem
`curl`, `ca-certificates` und `gzip`; es werden keine Abhängigkeiten automatisch installiert.

Optionen: `--prefix PATH` (ein anderes Installationspräfix), `--jobs N`, `--cuda`, `--rocm`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--rocm-targets LIST`, `--no-ui`, `--no-openssl`, `--no-nccl`
und `--help`. `CUDACXX` wählt den CUDA-Compiler und `HIPCXX` den clang-Compiler von ROCm.

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
- Ein CUDA-Build benötigt zum Starten den NVIDIA-Treiber; verwenden Sie `--rocm` oder `--vulkan` auf Computern mit einer anderen GPU und
  `--cpu` auf Computern ohne GPU.

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
- Eine integrierte GPU (die Grafik eines Prozessors von AMD oder Intel) hat keinen eigenen VRAM: Jeder Speicher-Heap, den sie meldet,
  ist System-RAM. `mgs` zählt diesen Speicher und den RAM als ein gemeinsames Budget und erstellt darauf keinen Expertencache; siehe
  [integrierte GPUs](#integrated-gpu).
- `mgs` prüft, wo der Kernel diesen Speicher tatsächlich hält (`/proc/self/fdinfo`, veröffentlicht von amdgpu, i915 und xe). Wenn
  ein anderes Programm den VRAM füllt und der Treiber einen Teil davon nach GTT verschiebt, melden das Protokoll und der Monitor dies, und der
  Monitor zeigt den tatsächlichen Ort (VRAM und GTT) jeder diskreten GPU von AMD oder Intel an (eine integrierte GPU hat keinen eigenen
  VRAM und wird daher nicht geprüft). Der Expertencache wird mit einer
  niedrigeren Speicherpriorität (`VK_EXT_memory_priority`, immer verwendet, wenn die GPU es anbietet, ohne eine Variable zum Einschalten)
  reserviert, sodass der Treiber ihn vor den MTP-Schichten, den Gewichten oder dem KV-Cache verschiebt. Ohne Resizable BAR meiden
  Gerätepuffer außerdem das kleine VRAM-Fenster, das die CPU einblenden kann (256 MiB bei einer RX 580), das sich der Desktop teilt und
  das der Treiber als Erstes nach GTT leert, und dieses Fenster zählt für sie nicht als freier Speicher.
- Nur CUDA kann die Modellseiten fixieren, die im RAM bleiben, daher erreichen die Experten die GPU mit Vulkan über auslagerbaren
  Speicher. `mgs` kopiert sie über einen kleinen Staging-Ring und stellt die Kopie auf die GPU in die Warteschlange, ohne darauf zu
  warten, sodass GPU und CPU weiterarbeiten, während die Experten hochgeladen werden: Auf einer RTX 4060 Ti mit Qwen3.6-35B-A3B Q8_0
  steigerte das die Prompt-Verarbeitung von 455 auf 515 Token/s und die Generierung von 20,6 auf 23,2 Token/s. `--load-mode none`
  kopiert die Experten stattdessen in fixierten Host-Speicher; vor dem Staging-Ring steigerte es auf derselben Karte mit dem
  Q4_K-Modell das Hochladen der Experten von 2,7 auf 13,1 GB/s und die Generierung von 27,5 auf 36,6 Token/s. Bei
  AMD und Intel zählt dieser fixierte Speicher gegen GTT: Wenn die Experten nicht hineinpassen, schlägt das Laden mit
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`) fehl, wie es bei den 18,9 GB an Experten dieses
  Modells mit 16,8 GB GTT auf einer RX 580 geschah. Verwenden Sie es nur, wenn die Experten in GTT und in den freien RAM passen.
- Mit einer NVIDIA-GPU über Vulkan und mit BF16-Experten auf jeder GPU wandeln die Expertenprodukte ihre Eingabe zuerst in einem
  Arbeitspuffer um. Dieser Puffer belegte früher 512 KiB pro Prompt-Token, daher schlugen Prompts mit einigen tausend Tokens mit
  `ErrorOutOfDeviceMemory` fehl, sobald der Expertencache den VRAM gefüllt hatte, es sei denn, `--moe-cache-reserve` wurde erhöht.
  Jetzt belegt er wenige KiB pro Token (32 MiB für einen Micro-Batch mit 4096 Tokens von Qwen3.6-35B-A3B), deutlich innerhalb der
  Standardreserve. Auf einer RTX 4060 Ti mit Qwen3.6-35B-A3B Q8_0 wurde dadurch auch die Prompt-Verarbeitung schneller, von 515 auf
  549 Token/s.
- Wählen Sie die Modellgröße passend zum Rechner: Das gesamte GGUF sollte in den freien RAM plus VRAM passen, abzüglich des KV-Caches und der
  Rechenpuffer. Mit 8 GB VRAM und 16 GB RAM bedeutet das etwa 18–20 GB an Gewichten; größere Dateien werden
  immer wieder von der Festplatte gelesen, und die Generierung wird sehr langsam. Das oben genannte Q4_K mit 21,6 GB erreichte 7,8 Token/s auf einer
  RX 580 mit 15 GB RAM und 15,9 Token/s, sobald der Rechner 31 GB hatte (10,0 mit `--spec-type none`).

<a id="rocm"></a>
## ROCm (AMD-GPUs)

Sowohl `build/build-local.sh` als auch das Debian-Installationsprogramm kompilieren mit `--rocm` das ROCm-Backend (HIP), für
AMD-GPUs, die die installierte ROCm-Version unterstützt. Keines der beiden Skripte installiert ROCm: Installieren Sie selbst als root
ROCm 6.1 oder neuer, mit HIP, hipBLAS und rocBLAS samt ihren Entwicklungsdateien. Installieren Sie für den lokalen Build außerdem
als root die allgemeinen Build-Abhängigkeiten (das Debian-Installationsprogramm installiert sie selbst):

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Dann als Ihr normaler Benutzer:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

Das Debian-Installationsprogramm akzeptiert dieselbe Option:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --rocm
```

- Die Skripte finden ROCm über `hipconfig` (im `PATH` oder `/opt/rocm/bin/hipconfig`) und verwenden den clang-Compiler,
  den es nennt; `HIPCXX` wählt einen anderen. Eine fehlgeschlagene Konfiguration nennt, was fehlt: HIP 6.1 oder neuer mit den
  Entwicklungsdateien von hipBLAS und rocBLAS.
- Die GPU-Targets sind die der beim Kompilieren vorhandenen GPUs, gelesen mit `amdgpu-arch` oder `rocminfo`. Ohne eine
  erkannte GPU oder um für andere zu bauen, übergeben Sie `--rocm-targets` mit einer kommagetrennten Liste, zum Beispiel
  `--rocm-targets gfx1100,gfx1030`.
- Aktuelle ROCm-Versionen unterstützen keine GCN-GPUs bis Polaris (`gfx6xx` bis `gfx8xx`, wie die Radeon RX 580,
  `gfx803`): Die Skripte warnen vor diesen Targets; verwenden Sie für sie `--vulkan`.
- Die lokale Installation kopiert jede ROCm-Bibliothek (HIP, hipBLAS, rocBLAS, hipBLASLt…) nach `lib/mgs/`, zusammen mit
  den GPU-Kernelordnern, die rocBLAS und hipBLASLt zur Laufzeit laden (`rocblas/library`, `hipblaslt/library`); der
  Starter verweist beide Bibliotheken auf diese Kopien (`ROCBLAS_TENSILE_LIBPATH`, `HIPBLASLT_TENSILE_LIBPATH`), sodass sie
  nie die Kernel einer anderen ROCm-Version lesen. Nur der amdgpu-Kerneltreiber stammt vom System. Die Kernelordner
  belegen Gigabytes.
- Das `/usr/local/bin/mgs` des Debian-Installationsprogramms lädt die ROCm-Bibliotheken des Systems über den beim
  Kompilieren hinterlegten Run-Path; wenn einige nicht gefunden werden, bricht das Installationsprogramm ab und fordert auf, den
  `lib`-Ordner von ROCm zu `/etc/ld.so.conf.d/` hinzuzufügen und `ldconfig` auszuführen.
- Der Benutzer, der `mgs` ausführt, benötigt `/dev/kfd` und `/dev/dri`, also die Gruppen `render` und `video`; das
  Installationsprogramm fügt das Dienstkonto beiden hinzu.

<a id="deployment"></a>
## Bereitstellung in der Produktion

Das Installationsprogramm ([Installation](#installation)) baut, installiert und konfiguriert die
Webdienste in einem einzigen Lauf:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com
```

Ohne `--domain` ist die Domain die Ausgabe von `hostname -f` (oder von `hostname` oder
`uname -n`, wenn diese kein gültiger Name ist). Die Domain muss auf den Produktionsrechner
auflösen. Um die Webkonfiguration später ohne Neukompilierung zu ändern, führen Sie als root im
Quellverzeichnis (`/opt/moe-gguf-server-source` bei entfernten Installationen) Folgendes aus:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

`configure-web.sh` akzeptiert dieselben Weboptionen wie das Installationsprogramm (`--domain`,
`--certificate`, `--private-key`, `--proxy-ip`), dazu `--binary PATH` für einen abweichenden Ort der
ausführbaren Datei, `--no-system-deps` und `--check`, das nur die Optionen validiert. Jeder Lauf schreibt die
gesamte Konfiguration aus den übergebenen Optionen neu: Ein erneuter Lauf ohne `--proxy-ip` entfernt den Proxy,
und ohne `--domain` kehrt die Domain zu der von `hostname -f` zurück.

Der Dienst `moe-gguf-server-web`
betreibt eine eigene Apache-Instanz als `moe-gguf-server`, ohne root.
Jeder Start wählt den ersten freien HTTP-Port ab 11080 und den ersten freien HTTPS-Port
ab 11443. Jede Folge läuft unabhängig bis 65535 weiter, ohne
beiden Protokollen denselben Port zuzuweisen. Er untersucht die IPv4/IPv6-TCP-Listener
und versucht es erneut, wenn Apache feststellt, dass ein Port nach der Auswahl belegt wurde. Andere
Konfigurationsfehler brechen den Start ab, statt mit Portkonflikten verwechselt zu werden.

HTTP leitet dauerhaft auf den gewählten HTTPS-Port weiter und behält dabei Pfad und Abfrage bei.
Die Weiterleitung wird nicht zwischengespeichert. Der Inferenzdienst (`moe-gguf-server`, der ebenfalls als
`moe-gguf-server` läuft und mit `--http` gestartet wird) lauscht auf keinem TCP-Port: Er lauscht auf dem
Unix-Socket `/opt/moe-gguf-server/_/temp/web/mgs.sock`, den nur dieses Konto öffnen kann, und Apache leitet die
Anfragen dorthin weiter. Vor dem Lauschen löscht `mgs` eine Socket-Datei, die ein nicht mehr laufender Server
hinterlassen hat, aber nie den Socket eines Servers, der noch antwortet. Die Installation erfordert root;
keiner der beiden Dienste muss als root laufen. Es wird kein HAProxy konfiguriert, und keine gemeinsam genutzten
Apache-Sites werden importiert oder verändert.

Das Anhalten des Inferenzdienstes (`systemctl stop moe-gguf-server` oder `systemctl restart moe-gguf-server`) beendet ihn wie Strg+C oder ein `SIGTERM`, wenn Sie `mgs` selbst starten, geordnet innerhalb von ein, zwei Sekunden, auch mitten in einem langen Prompt: Laufende Anfragen erhalten einen Fehler 503 (`The server is shutting down.`), bei Streaming endet ihr Stream. Ein zweites Strg+C oder `SIGTERM` beendet den Prozess sofort, und während das Modell geladen wird, tut das bereits das erste.

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
(`web-domain`), der vertrauenswürdige Proxy (`web-proxy-ip`, nur mit `--proxy-ip`), die
Apache-Vorlage (`apache2.conf.in`) und die Einstellungen der
Oberfläche (`ui-settings.json`), alle im Besitz von `moe-gguf-server` mit dem Modus 0600.
Die Ausgabe der Installation wird in `/root/webapp-install.log` festgehalten, und
`/root/webapp-credentials.txt` enthält alle Zugangsdaten, die die Bereitstellung erstellt: das
Dienstkonto (`moe-gguf-server`, ohne Passwort und mit deaktivierter Anmeldung) und
den API-Schlüssel. Beide Dateien haben den Modus 600. Eine Neukonfiguration erhält vorhandene
Schlüssel, Modelle und Zertifikate.

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
Verwenden Sie denselben Befehl nach der Erneuerung der Quellzertifikate.

Wenn vor Apache ein Reverse-Proxy (HAProxy, nginx…) steht, geben Sie seine Adresse
mit `--proxy-ip` (IPv4 oder IPv6) an, dem Installationsprogramm oder `configure-web.sh`:

```bash
bash deploy/configure-web.sh --domain ai.example.com --proxy-ip 10.7.10.1
```

Apache lädt dann `mod_remoteip`. Bei HTTPS verlangt es den PROXY-Protokoll-Header
nur von dieser Adresse (`RemoteIPProxyProtocol On`, wobei jede andere Adresse in
`RemoteIPProxyProtocolExceptions` aufgeführt ist), sodass die übrigen Rechner weiterhin direkt verbinden; bei
HTTP vertraut es dem `X-Forwarded-For`-Header dieser Adresse (`RemoteIPHeader`,
`RemoteIPTrustedProxy`). Die Zugriffsprotokolle verzeichnen die echte Adresse des Clients. Ohne
`--proxy-ip` wird kein Proxy konfiguriert.

Der Dienst verwendet den Router-Modus. Legen Sie lesbare GGUF-Modelle in
`/opt/moe-gguf-server/models` ab, oder verwenden Sie die Modellverwaltung in der Oberfläche. Für
spezielle Befehlszeilenoptionen verwenden Sie unter Debian/Ubuntu ein systemd-Override oder unter Alpine die
OpenRC-Dienstkonfiguration. Lassen Sie den Inferenzdienst auf seinem Unix-Socket.
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
2. Wählen Sie ein Modell aus. Laden Sie im Router-Modus ein verfügbares Modell, wenn keines geladen ist; solange es keines gibt,
   verweist die Modellauswahl auf Einstellungen → Modelle, wo Sie den Ordner Ihrer Modelle wählen
   ([Modelle auswählen und laden](#settings-models)).
3. Geben Sie eine Nachricht ein und senden Sie sie. Die Eingabetaste sendet standardmäßig; Umschalt+Eingabe fügt einen
   Zeilenumbruch ein. Diese Tastenkürzel lassen sich in den Einstellungen ändern.
4. Fügen Sie Dateien über die Anhangsaktion hinzu. Bilder, Audio und Video erfordern passende
   Modalitäten des Modells. PDF-Dateien können als Text extrahiert oder in Bilder umgewandelt werden;
   Modelle ohne Bildverarbeitung verwenden den Textweg.
5. Beenden Sie die Generierung, überspringen Sie den Denkprozess, sofern unterstützt, bearbeiten Sie eine Nachricht, generieren Sie eine
   Antwort erneut oder zweigen Sie eine Konversation über die Nachrichtenaktionen ab.

Die drei Striche oben in der Seitenleiste klappen sie aus und ein. Sie zeigt **Neuer Chat**, **Monitor**, **Modell**
und **Einstellungen**, darunter die Konversationen: **Angeheftet**, **Letzte Chats** (die fünf, mit denen Sie zuletzt
gearbeitet haben; ein Chat, an den Sie erneut eine Nachricht senden, rückt dorthin auf) und **Chats** (die übrigen, die
neuesten zuerst). Jeder Chat hat eine Sprechblase mit einer Nummer von 01 bis 99, die er bei seiner Erstellung erhält
und behält, solange es ihn gibt (eine durch das Löschen eines Chats frei gewordene Nummer geht an den nächsten neuen),
sodass er sich auch bei eingeklappter Seitenleiste erkennen lässt, wenn nur die Symbole des Menüs und die Sprechblasen
zu sehen sind. Chats lassen sich anheften, umbenennen, löschen und mehrfach auswählen. Das Suchfeld neben den drei
Strichen (Strg+K, auf einem Mac ⌘K, klappt die Seitenleiste aus und setzt den Fokus darauf) findet ohne Rücksicht auf
Groß-/Kleinschreibung oder Akzente die Chats nach ihrem Titel
und nach dem Inhalt ihrer Nachrichten, die Seiten und ihre Teile, jede Einstellung, die Modelle, die Werkzeuge und die
MCP-Server. Klicken Sie auf ein Ergebnis oder drücken Sie für das erste die Eingabetaste, um seinen Chat oder seine
Seite zu öffnen, wo das Gefundene kurz hervorgehoben wird.
Konversationen werden vom Server in `~/.mgs/chats/` des Benutzers gespeichert, der `mgs` ausführt (im bereitgestellten Dienst
`/opt/moe-gguf-server/.mgs/chats/`), eine Datei pro Chat, benannt nach dem Datum seiner Erstellung in UTC
(`2026-11-03@14-59-59.723.jsonl`), sodass jeder Browser, der die Oberfläche öffnet, dieselben Chats sieht, und eine geöffnete Seite
zeigt innerhalb weniger Sekunden die Chats an, die ein anderer Browser erstellt, ändert oder löscht. Die Einstellungen werden vom Server in
`~/.mgs/config/ui-settings.json` des Benutzers gespeichert, der `mgs` ausführt (im bereitgestellten Dienst
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), sodass jeder Browser, der die
Oberfläche öffnet, sie teilt: Generierungsparameter, Anzeigeoptionen, Farbschema, Sprache, MCP-Server,
Lieblingsmodelle, deaktivierte und immer erlaubte Werkzeuge, der standardmäßige
Denkaufwand sowie der Modellordner unter Einstellungen → Modelle. Nur der API-Schlüssel bleibt in jedem Browser, weil er die
Anfrage authentifiziert, die den Rest lädt; geben Sie ihn daher in jedem Browser einmal ein. Import und Export sind in den
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

Wählen Sie für das Farbschema System, Hell oder Dunkel. Wie die Sprache veröffentlicht der Server das
gespeicherte Farbschema in `/api/config.js`, sodass jede Seite schon darin geöffnet wird, bevor sie gezeichnet wird,
ohne etwas im Browser zu speichern; mit System folgt die Oberfläche sofort dem hellen oder dunklen Modus des
Betriebssystems, auch während sie geöffnet ist. Zu den Generierungseinstellungen gehören die
Kontextverwendung, Temperatur, Sampler, Strafen und die Grenzen für Agentenrunden. Manche Funktionen
hängen von Server-Flags oder vom ausgewählten Modell ab.

Nach jeder Antwort sendet die Oberfläche die Konversation erneut, sobald der Server
untätig ist, damit sie bereits im KV-Cache kodiert ist, während Sie lesen, und die nächste Runde
schneller beginnt. Dies ist standardmäßig aktiviert; deaktivieren Sie Einstellungen → Entwicklung → **KV-Cache nach jeder
Antwort vorbereiten**, um es zu unterbinden.

Während der Server eine andere Anfrage bearbeitet, wartet eine von Ihnen gesendete Nachricht in der
[Warteschlange](#request-queue), und ihre Antwort zeigt bis zu ihrem Beginn einen Hinweis statt des Fortschrittstexts an:
„In der Warteschlange: Zuerst wird eine andere Anfrage bearbeitet, die direkt an die API gesendet wurde
(Anfragen davor: N).“, während eine Anfrage läuft, die ein Programm direkt an die API gesendet hat, oder andernfalls
„In der Warteschlange: Zuerst wird eine andere Anfrage aus der Weboberfläche bearbeitet (Anfragen davor: N).“
(ein anderer Tab oder Browser, ein generierter Titel oder die Vorbereitung des KV-Caches). N zählt die Anfragen,
die vor Ihrer laufen oder warten. Der Hinweis verschwindet, sobald Ihre Anfrage beginnt.

<a id="monitor"></a>
## Monitor und Statusleiste

**Monitor** befindet sich in der Seitenleiste unter **Neuer Chat** (Adresse `#/monitor`). Solange er geöffnet ist, aktualisiert er sich alle
zwei Sekunden und zeigt:

- **Auslastung**: zuerst eine Karte pro GPU des Rechners (NVIDIA, AMD oder Intel, auch wenn kein Modell sie verwendet), betitelt mit ihrer
  fett gedruckten Kennung und ihrem Namen (zum Beispiel **gpu-1**: NVIDIA GeForce RTX 4060 Ti; bei mehreren GPUs gpu-1, gpu-2
  und so weiter), mit ihrer Auslastung, der Auslastung des Speichercontrollers (Bandbreite), dem belegten VRAM, der Temperatur und der Leistungsaufnahme im Verhältnis zu ihrem
  Grenzwert; dann eine Karte pro Prozessor, auf die gleiche Weise mit seinem Modell betitelt (zum Beispiel **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; bei mehreren Prozessoren cpu-1, cpu-2 und so weiter), mit seiner Auslastung, einem kleinen senkrechten Balken pro
  logischer CPU dieses Prozessors und der durchschnittlichen Systemlast. Jeder Auslastungsbalken ist grün und füllt sich mit zunehmender Auslastung
  rot; ein Balken, dessen Wert unbekannt ist, bleibt grau. GPUs von NVIDIA werden über die NVML-Bibliothek des Treibers gelesen und GPUs von AMD
  über `gpu_busy_percent` in sysfs; jede andere GPU (Intel oder NVIDIA ohne NVML) zeigt keine Auslastung an, und ihr
  Name und ihr Speicher stammen vom Inferenz-Backend, sobald ein Modell sie verwendet. Auf einem breiten Bildschirm stehen
  die Karten zu zweit in einer Reihe, jede wie die Speicherbalken über die halbe Breite; bei einer ungeraden Anzahl von
  Karten enthält die letzte Reihe drei.
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

`mgs` lädt ausschließlich MoE-Modelle. Beim Start liest es den Header des mit `--model` übergebenen GGUF (und des mit `--spec-draft-model` übergebenen); handelt es sich um ein Dense-Modell wie GPT-2 oder Phi-2, protokolliert `mgs` einen Fehler wie diesen und beendet sich mit dem Exit-Code 1, bevor es irgendetwas lädt:

```text
srv    mgs_server: the model '/path/to/model.gguf' is not a Mixture of Experts (MoE) model
srv    mgs_server: mgs only loads MoE models; dense models are not supported
srv    mgs_server: exiting due to a dense model
```

Im Router-Modus beendet sich nur der Prozess dieses Modells: `/api/v1/models` markiert es als fehlgeschlagen, und die übrigen Modelle arbeiten weiter. Decoder-Implementierungen für Dense-Modelle werden nicht kompiliert, aber die dichten Schichten innerhalb eines MoE-Modells und ein ausdrücklich angegebener, kompatibler reiner MTP-Kopf (`--spec-draft-model`) bleiben erhalten.

<a id="settings-models"></a>
### Modelle unter Einstellungen → Modelle auswählen und laden

Ohne `--model` gestartet, läuft `mgs` im Router-Modus: Es lädt von sich aus kein Modell und listet die Modelle auf, die es findet,
sodass Sie sie über die Oberfläche laden können. Einstellungen → Modelle hat zwei Teile:

- **Modellordner.** Klicken Sie auf **Ordner wählen**: Ein Ordnerbrowser für den Computer, auf dem `mgs` läuft, öffnet
  sich im aktuellen Modellordner (oder in Ihrem Home-Verzeichnis). Klicken Sie auf einen Ordner, um ihn zu öffnen, gehen
  Sie mit dem Pfeil eine Ebene nach oben und mit dem Haus zu Ihrem Home-Verzeichnis, oder geben Sie einen Pfad in sein
  Feld ein und drücken Sie die Eingabetaste (ein führendes `~` steht für Ihr Home-Verzeichnis); **Versteckte Ordner
  anzeigen** fügt die Ordner hinzu, deren Name mit einem Punkt beginnt, und der Ordnerbrowser zählt die `.gguf`-Dateien
  des angezeigten Ordners. Klicken Sie auf **Wählen**: Der absolute Pfad dieses Ordners wird in das Feld eingetragen und
  sofort verwendet. Der Ordner kann eine `.gguf`-Datei pro Modell enthalten oder einen Unterordner pro Modell mit den
  Teilen eines aufgeteilten Modells (`NAME-00001-of-0000N.gguf`) und seinem multimodalen Projektor (`mmproj*.gguf`).
  `mgs` prüft, ob es den Ordner lesen kann, speichert ihn zusammen mit den übrigen Einstellungen und durchsucht ihn
  sofort; jeder spätere Start durchsucht denselben Ordner. Um ihn erneut einzulesen, nachdem Sie Dateien hinzugefügt
  oder entfernt haben, wählen Sie denselben Ordner noch einmal. Dense-GGUF-Dateien, darunter separate MTP-Köpfe,
  erscheinen nicht in der Liste, weil `mgs` nur MoE-Modelle lädt. Wenn Sie einen anderen Ordner wählen, werden die
  laufenden Modelle des vorherigen entladen. Die nach `~/.cache/mgs/` heruntergeladenen Modelle werden ebenfalls
  aufgeführt. Wurde `mgs` mit `--models-dir` gestartet, hat dieser Ordner Vorrang: **Ordner wählen** ist deaktiviert,
  und die Seite weist darauf hin.
- **Verfügbare Modelle.** Jedes Modell mit seinem Status (nicht geladen, wird geladen mit seinem Prozentsatz, geladen, ruhend oder
  Laden fehlgeschlagen), seiner Datei und seiner Größe auf dem Datenträger sowie einer Schaltfläche zum Laden oder Entladen. Ein hier
  geladenes Modell wird auch im Chat ausgewählt.

Welche Modelle geladen sind und wie sie geladen wurden, steht auf der [Modellseite](#model-page), auf die das Ende von
Einstellungen → Modelle verweist. Die Modellauswahl des Chats listet dieselben Modelle auf und lädt das gewählte;
solange die Liste leer ist, verweist sie auf Einstellungen → Modelle. Wurde `mgs` mit `--model` gestartet, weist
Einstellungen → Modelle nur darauf hin, dass sich das Modell nicht ändern lässt, ohne `mgs` neu zu starten.

Es ist immer nur ein Modell geladen: Wenn Sie ein anderes laden, hier, über die Modellauswahl des Chats oder über die
API, wird das bisher geladene entladen (`--models-max N` erlaubt N gleichzeitig). Ein neuer Chat verwendet das Modell, das in
diesem Moment geladen ist. Ein Chat behält sein eigenes Modell, das seiner letzten Antwort: Wenn Sie ihm eine Nachricht
senden, während ein anderes Modell geladen ist, wird dieses entladen und zuerst das Modell des Chats geladen. Liegt das
Modell des Chats nicht mehr dort, wo es war (es ist aus der Liste verschwunden oder ist jetzt eine andere Datei), weist
ein Dialog darauf hin und lässt Sie die Nachricht abbrechen, die im Eingabefeld bleibt, oder sie mit dem geladenen
Modell senden.

Der Ordner wird in `~/.mgs/config/ui-settings.json` zusammen mit den übrigen Einstellungen der Oberfläche als Abschnitt
`modelsFolder` gespeichert. `mgs` schreibt ihn, wenn Sie den Ordner wählen, und liest ihn bei jedem Start:

```json
{
  "modelsFolder": "/home/user/models"
}
```

Der bereitgestellte Dienst (`deploy/`) startet `mgs` mit `--models-dir /opt/moe-gguf-server/models`, daher ist das der Ordner,
den er auflistet.

<a id="model-page"></a>
### Modellseite

**Modell**, in der Seitenleiste nach **Monitor** (Adresse `#/model`), zeigt das geladene Modell in zwei Kästen:
**Modelldatei**, mit dem, was seine GGUF-Datei enthält (Beschreibung, Architektur, Datei, Größe auf dem Datenträger,
Parameter, Schichten, MTP-Schichten, Trainingskontext und multimodaler Projektor), und **Ladeparameter**, mit der Art,
wie es geladen wurde: die Parameter, die `mgs` selbst gewählt oder die Sie übergeben haben (Kontextgröße,
parallele Slots und Kontext pro Slot, Batch und Micro-Batch, Threads für Generierung und Batches, Flash Attention,
KV-Cache-Typen, `--fit`, MTP, spekulatives Dekodieren mit seiner maximalen Anzahl von Entwurfs-Tokens, MoE-Expertencache
und Geräte) und im Router-Modus die Befehlszeile seines Prozesses. Jeder eingeschaltete Zustand (Flash Attention,
`--fit`, MTP, spekulatives Dekodieren und der MoE-Expertencache) wird grün angezeigt. Bei Flash Attention steht, ob
`auto` sie eingeschaltet hat; MTP zeigt, wie viele Entwurfs-Tokens es in jedem Schritt anfordert, oder dass es sie noch
misst (0 bis 3), oder dass sich der Entwurf auf diesem Rechner nicht lohnt; und der MoE-Expertencache gibt an, ob er
erstellt wurde, mit seiner automatischen oder angeforderten Größe: Darüber entscheidet die erste Anfrage nach dem Laden
des Modells, bis dahin wird das angezeigt, und auf einer integrierten GPU oder wenn bereits alle Experten im VRAM
liegen, bleibt er ausgeschaltet. Diese aktuellen Zustände stammen aus dem [Monitor](#monitor)-Bericht, der auch den
Speicher enthält, den jedes Gerät nach Zweck belegt. Solange kein Modell geladen ist, verweist die Seite auf
Einstellungen → Modelle.

<a id="supported-models"></a>
### Unterstützte Modellfamilien

| Familie | GGUF-Architektur | Hinweise |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP, wenn das GGUF es enthält. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP, wenn das GGUF es enthält. Seine schichtweise Embedding-Tabelle (etwa 29 GB) wird bei Bedarf aus der Modelldatei gelesen, halten Sie die Datei daher auf einem schnellen Datenträger. Lange Kontexte werden unter der dünnbesetzten Auswahl mit dichter Attention berechnet: korrekt, aber ohne die Beschleunigung dünnbesetzter Kernel. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; Bilder mit seinem mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Die veröffentlichten GGUFs enthalten kein MTP. |
| DeepSeek-V4.1-Flash | `deepseek41` | Lädt beide veröffentlichten Formate: die DwarfStar-Dateien von antirez (etwa `antirez/deepseek-v4.1-flash-gguf`) und die Dateien des llama.cpp-Ökosystems. Seine Engram-Tabelle (etwa 188 GiB innerhalb der Datei) wird nie geladen: Jedes Token liest seine Zeilen aus der Modelldatei, daher muss die Datei verfügbar bleiben, solange `mgs` läuft, vorzugsweise auf einer SSD, und Linux oder ein anderes POSIX-System ist erforderlich. Dateien ohne Chat-Vorlage verwenden eine integrierte, und `reasoning_effort` akzeptiert eine Zahl von 1 bis 100. Kein MTP und keine Bilder. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |
| Flint | `flint` | Die veröffentlichten GGUFs enthalten kein MTP. |

Gemma 4 und die DeepSeek-Modelle vor V4.1 werden nicht mehr unterstützt (siehe [entfernte Kompatibilität](#removed-compatibility)). Ebenso wenig die übrigen von llama.cpp geerbten MoE-Architekturen wie gpt-oss, Qwen3-MoE, Qwen3-Next, GLM-4.x, Granite 4, MiniMax-M2 oder Kimi-Linear.

Bei mehreren ausgewählten GPUs prüft MTP alle, bevor es auf den RAM ausweicht. Seine Speicherschätzung schließt den Ausgabekopf ein. Integrierte MTP-Puffer werden vor den übrigen Gewichten geladen; ein externes MTP-GGUF, das mit `--spec-draft-model` übergeben wird, wird vor dem Hauptmodell geladen, auch mit `--fit off`, und schaltet MTP von selbst ein (siehe [automatisches MTP](#performance)); ein MTP, das sich nur in einem anderen GGUF befindet, wird nie verwendet, sofern es nicht auf diese Weise übergeben wird. Die aktuelle ausdrückliche Geräteauswahl bestimmt weiterhin, welche GPUs verfügbar sind.

Verwenden Sie `--reasoning on` oder `--reasoning off`, um das Denken zu steuern. Ist das Denken ausgeschaltet (mit dieser Option, mit dem Denk-Schalter des Chats oder in einer Anfrage mit `enable_thinking: false` in `chat_template_kwargs` bzw. `reasoning_effort: "none"`), schließt `mgs` außerdem jeden Denkblock, sobald das Modell ihn öffnet; so antworten auch Modelle, deren Chat-Vorlage diese Einstellung ignoriert, etwa Flint, ohne nachzudenken. Die Übergabe von `enable_thinking` über `--chat-template-kwargs` wird abgelehnt. Verwenden Sie die dokumentierten Optionsnamen mit Bindestrich; Schreibweisen mit Unterstrich werden abgelehnt. Übergeben Sie Listenwerte als kommagetrennten Wert, statt eine Option zu wiederholen. Nur `--spec-type` bleibt wiederholbar, um spekulative Modi zu kombinieren.

Die Modellauswahl zeigt verfügbare, geladene und Lieblingsmodelle. Die Modelldetails
umfassen Kontextgröße, Modalitäten, Quantisierung und die Chat-Vorlage. Der Router-Modus
verwaltet separate Kindprozesse für geladene Modelle; Anfragen geben das Modell
über das Feld oder den Abfrageparameter `model` an.

MCP-Server werden unter Einstellungen → MCP-Server per URL mit optionalen Autorisierungs-Headern konfiguriert. Sie können
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

Schalten Sie **Lokaler Shell-MCP-Server** unter Einstellungen → MCP-Server ein oder starten Sie `mgs` mit `--local-shell-mcp-server`,
damit die Modelle des Web-Chats das Terminal des Rechners verwenden können, auf dem `mgs` läuft:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- Der Schalter wirkt sofort, ohne `mgs` neu zu starten, und wird im Abschnitt `localShellMcpServer` von
  `~/.mgs/config/ui-settings.json` gespeichert, sodass spätere Starts ihn wieder einschalten (`POST /api/mcp/shell/enabled` mit
  `{"enabled": true}` bewirkt dasselbe). Mit `--local-shell-mcp-server` bleibt er eingeschaltet, und der Schalter ist deaktiviert.
  Ohne API-Schlüssel kann nur ein Browser auf dem Rechner, auf dem `mgs` läuft, den Schalter verwenden, sodass niemand sonst im
  Netzwerk dort eine Shell einschalten kann; mit einem API-Schlüssel kann es jeder, der den Schlüssel hat.

- Beim Start prüft `mgs`, welche Shells installiert sind, und bietet für jede von ihnen ein Werkzeug an: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) und `psh.run` (PowerShell, `pwsh`). Das Protokoll listet die gefundenen Shells auf; eine Shell,
  die nicht installiert ist, erhält kein Werkzeug.
- Die Oberfläche erkennt ihn, fügt ihn den MCP-Servern als **local shell mcp server** hinzu und aktiviert ihn. Sie können ihn dort
  oder für eine einzelne Konversation deaktivieren, und Ihre Wahl bleibt erhalten. Wenn der Server ausgeschaltet wird,
  verschwindet der Eintrag.
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
  (`"event": "error"`). Auch ein vor der Ausführung abgelehnter Aufruf (ein unbekanntes Werkzeug, kein Befehl, ein nicht existierendes
  Arbeitsverzeichnis, ein ungültiges Zeitlimit) hinterlässt seine Datei, mit dem Befehl, wie er angefordert wurde, und dem Fehler. Der Chat behält von sehr langen Ausgaben nur
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
werden abgelehnt (der standardmäßige CORS-Ursprung ist localhost), und das Protokoll verzeichnet jeden Befehl mit seinem Exit-Code. Lesen Sie
den im Passwortfenster angezeigten Befehl, bevor Sie ein sudo-Passwort eingeben: Das Passwort autorisiert diesen Befehl. Der
Server übergibt es nur an einen Passwort-Helfer, dessen Elternprozess sudo ist, das als root für Ihren Benutzer läuft, sodass ein Befehl
den Helfer nicht selbst starten kann, um das Passwort auszulesen.

<a id="api"></a>
## HTTP-API

Die Beispiele verwenden 11443; ersetzen Sie ihn durch den HTTPS-Port aus `ports.json`, falls er abweicht.

Die Standardwurzel der API ist `/api`. Die Oberfläche bleibt unter `/`. Die Wurzelrouten `/v1/...`
des ursprünglichen Servers werden nicht mehr registriert. OpenAI-kompatible Clients müssen
`https://ai.example.com:11443/api/v1` als Basis-URL verwenden: Die OpenAI-Routen (`/api/v1/models`,
`/api/v1/chat/completions`, `/api/v1/responses`, `/api/v1/audio/transcriptions` und ihre
`input_tokens`-Varianten) existieren nur unter `/api/v1` und der Gesundheitsstatus nur unter `/api/health`.
`/api/rerank` und `/api/v1/rerank` bleiben beide für TEI/Jina-Clients erhalten, und die nativen Routen
(`/api/completions`, `/api/embeddings`…) sind andere Handler als ihre gleichnamigen Routen unter `/api/v1`.
Die Feldnamen von Anfragen und Antworten
bleiben mit Upstream kompatibel; die C/C++-Schnittstellen der Engine behalten ihre aktuellen Funktionen,
aber nicht diejenigen, die als veraltet markiert waren oder keine Verwendung hatten (siehe [entfernte Kompatibilität](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Setzen Sie `MGS_API_KEY` in Ihrer eigenen Shell auf den installierten Schlüssel (nur `curl` verwendet ihn; `mgs` liest seinen Schlüssel aus
`--api-key`, `MGS_ARG_API_KEY` oder `--api-key-file`). Native Operationen,
Embeddings, Tokenisierung, Reranking, Modellverwaltung und fortsetzbare Streams sind
in der mitgelieferten Swagger UI unter `/api/doc/` dokumentiert; ihr OpenAPI-JSON liegt unter
`/api/doc/openapi.json` und wird vom Server bei jeder Anfrage ohne statische JSON-Datei erzeugt. Die Übersicht spiegelt die Routen wider, die der
laufende Prozess registriert hat, einschließlich der Router- und der optionalen Kompatibilitätsrouten. Deaktivierte
Funktionen können 403 zurückgeben. Ein fehlender oder ungültiger Schlüssel ergibt 401, und das Laden eines Modells
ergibt 503. `/api/health` und `/api/v1/models` bleiben für `GET` und `HEAD` öffentlich zugänglich; siehe
[Sicherheitsgrenzen](#security-limits) für die Routen, die immer den Schlüssel benötigen.

`--api-prefix /api/custom` verlegt die Inferenzrouten unter `/api/custom`.
Die Oberfläche an der Wurzel erhält dieses Präfix von `/api/config.js`; die Dokumentation bleibt unter `/api/doc/`.
Der Namensraum der Dokumentation ist reserviert und kann nicht als Inferenzpräfix
verwendet werden. Router-Kindprozesse verwenden intern `/api`. Der Server prüft die Präfixe;
Werte außerhalb von `/api/` und Pfade mit abschließenden Schrägstrichen werden abgelehnt.

Der Monitor verwendet `GET /api/monitor` (vollständiger Bericht), `GET /api/monitor/memory` (Speicher-Momentaufnahmen der geladenen Modelle)
und `GET /api/monitor/usage` (nur CPU- und GPU-Auslastung, auch während das Modell lädt verfügbar). Sie lesen Zähler und
Momentaufnahmen, die von der Inferenzschleife veröffentlicht werden, niemals die Aufgabenwarteschlange, sodass ihre Abfrage weder ein schlafendes Modell weckt noch
dessen Leerlauf-Timer zurücksetzt. Jedes Modell des Berichts trägt `expert_cache` (mit `active` und `pending`) und, mit MTP,
`mtp`: `draft_n` (die in jedem Schritt angeforderten Entwurfs-Tokens, 0, wenn sich der Entwurf nicht gelohnt hat, -1, während sie
gemessen werden), `draft_n_max` und `measuring`. Ihre Felder sind in Swagger beschrieben.

Die Oberfläche liest ihre Einstellungen mit `GET /api/ui/settings` (`{"settings": DOCUMENT}` oder
`{"settings": null}` vor dem ersten Speichern) und speichert sie mit `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, das `~/.mgs/config/ui-settings.json` vollständig ersetzt, mit dem Modus 0600). Beide erfordern
den API-Schlüssel und antworten auch, während das Modell lädt.

Die Chats verwenden `GET /api/chats` (die Konversationen und ihre `version`), `GET /api/chats/version`,
`GET /api/chats/search?q=WORDS&limit=N` (die neueste Nachricht des Benutzers oder des Assistenten jedes Chats, der jedes
Wort enthält, ohne Rücksicht auf Groß-/Kleinschreibung oder Akzente) sowie `GET`, `POST` und `DELETE /api/chats/{id_chat}`. Jede Datei beginnt mit
der Konversation und enthält danach eine Nachricht pro Zeile; die Dateien speichern die Datumsangaben im nativen Format
`2026-11-03@14-59-59.723` (UTC) und die API in Millisekunden seit der Unix-Epoche. Ein Chat, der ohne gültige Nummer gespeichert wird, erhält
die niedrigste freie Nummer. Sie erfordern den API-Schlüssel und antworten auch, während das Modell lädt.

Im Router-Modus liefert `GET /api/models/folder` den Modellordner: `folder`, `source` (`cli` für `--models-dir`,
`settings` für den Abschnitt `modelsFolder` von `~/.mgs/config/ui-settings.json` oder `none`), `exists`, `model_count`,
`settings_file` und `cache_dir`. `POST /api/models/folder` mit `{"folder": "/absolute/path"}` (ein führendes `~` steht für das
Home-Verzeichnis; `""` entfernt ihn) prüft, ob der Ordner gelesen werden kann, speichert ihn in diesem Abschnitt, durchsucht die Modelle erneut und liefert dasselbe Objekt; es antwortet mit 400,
wenn der Ordner mit `--models-dir` festgelegt wurde oder nicht gelesen werden kann.
`GET /api/models/folder/browse?path=/absolute/path` versorgt den Ordnerbrowser: Es liefert `path`, `parent` (leer bei
`/`), `home`, `folders` (die nach Namen sortierten Unterordner, jeweils mit `name`, `path` und `hidden`) und
`gguf_files` (wie viele `.gguf`-Dateien direkt in dem Ordner liegen); ohne `path` listet es den Modellordner auf, sonst
das Home-Verzeichnis, und es antwortet mit 400, wenn der Pfad kein Ordner ist, der gelesen werden kann. Alle drei
erfordern den API-Schlüssel. Im Router-Modus trägt jeder Eintrag von `GET /api/v1/models` `path`, `file_size` (Bytes,
alle Teile eines aufgeteilten Modells) und `mmproj`, und ein laufendes Modell ergänzt `meta` und `load`, die
Konfiguration, mit der es geladen wurde (`architecture`, `description`, `n_layer`, `n_layer_mtp`, `n_ctx`, `n_ctx_slot`,
`n_parallel`, `n_batch`, `n_ubatch`, `n_threads`, `n_threads_batch`, `flash_attn`, `flash_attn_enabled` (ob der Kontext
sie verwendet, auch mit `auto`), `cache_type_k`, `cache_type_v`, `fit`, `mtp`, `speculative`, `draft_n_max`,
`moe_cache_mib`, `devices` und `mmproj`). Mit `--model` trägt auch der
einzige Eintrag `load`.

<a id="request-queue"></a>
### Anfragewarteschlange und parallele Anfragen

Standardmäßig bearbeitet `mgs` eine Anfrage nach der anderen (`-np 1`) und gibt ihr den gesamten Kontext, sodass jede Anfrage
alle Kontext-Token nutzen kann, ob sie aus der Weboberfläche oder direkt von der API kommt. Die Anfragen, die
in der Zwischenzeit eintreffen, warten in einer einzigen Warteschlange und werden in der Reihenfolge ihres Eintreffens bearbeitet: Anfragen der Oberfläche
überholen keine API-Anfragen und umgekehrt, und eine Anfrage, die erneut warten muss, behält ihren Platz.
Die API verhält sich gleich, unabhängig von den Einstellungen der Oberfläche.

Um mehrere Anfragen gleichzeitig zu bearbeiten, starten Sie `mgs` mit `-np N` (`--parallel N`, `MGS_ARG_PARALLEL`):
Bis zu N Anfragen laufen parallel, und der Kontext wird unter ihnen aufgeteilt, sodass jede einen Anteil erhält (`n_ctx_slot` in
der Zeile `initializing` des Protokolls); mit `--kv-unified` teilen sie sich stattdessen einen einzigen KV-Cache für den gesamten
Kontext. Anfragen über N hinaus warten weiterhin in der Warteschlange.

Die Weboberfläche fügt ihren Anfragen `"mgs_client": "webui"` hinzu. Während eine ihrer gestreamten Anfragen wartet,
beginnt der Stream mit Ereignissen wie

```text
data: {"mgs_queue":{"ahead":2,"running_api":1,"running_webui":0,"waiting_api":0,"waiting_webui":1}}
```

(`ahead`: Anfragen, die vor ihr laufen oder warten; `running_api` und `running_webui`: wie viele davon gerade laufen
und direkt an die API bzw. aus der Weboberfläche kamen; `waiting_api` und `waiting_webui`: dasselbe für die
Anfragen, die vor ihr in der Warteschlange warten) und einem letzten Ereignis mit allen Zählern auf 0, wenn sie beginnt. Clients, die
`mgs_client` nicht senden, erhalten diese Ereignisse nie, sodass OpenAI-kompatible Clients den üblichen Stream bekommen. Swagger
beschreibt das Feld zusammen mit den übrigen Completion-Parametern.

<a id="security-limits"></a>
### Sicherheitsgrenzen

Diese Grenzen schützen den Server und die Oberfläche vor feindseligen Anfragen, Dateien und Modellen. Sie sind großzügig
genug für die normale Verwendung: Was sie überschreitet, schlägt mit einer klaren Fehlermeldung fehl, statt den Server anzuhalten.

- **API-Schlüssel.** Die Routen für den Gesundheitsstatus und die Modellliste (`/api/health` und `/api/v1/models`) sind nur für `GET`
  und `HEAD` öffentlich. Das Neuladen der Modellliste (`GET /api/v1/models?reload=1`) und das Hinzufügen oder Löschen von Modellen im Router-Modus (`POST` und `DELETE /api/models`)
  benötigen den Schlüssel, sobald einer konfiguriert ist.
- **Seiten anderer Websites.** Standardmäßig (`--cors-origins` nicht gesetzt, was `localhost` bedeutet) dürfen nur die Seiten von
  `localhost` und dieses Servers selbst die API aufrufen: Der Server antwortet mit 403 auf jede Anfrage außer `GET`,
  `HEAD` oder `OPTIONS`, deren `Origin`-Header eine andere Website nennt, und der Browser lässt andere Websites seine
  Antworten nicht lesen. Die Oberfläche (auch hinter Apache) und Clients, die kein `Origin` senden, etwa curl oder die SDKs,
  sind nicht betroffen. Damit die Seiten anderer Websites die API aufrufen können, führen Sie deren Ursprünge in `--cors-origins` auf (`*`
  lässt jede Website zu: Setzen Sie dann einen API-Schlüssel). `POST /api/tools` akzeptiert nur `Content-Type: application/json` (sonst
  415).
- **DNS-Rebinding.** Mit einfachem HTTP (`--http`) auf einem TCP-Port und ohne API-Schlüssel antwortet der Server mit 403
  (`Host not allowed`) auf jede Anfrage, deren `Host`-Header weder eine IP-Adresse noch `localhost` (oder ein auf `.localhost`
  endender Name) noch der Name des Rechners (auch mit `.local`) noch der mit `--host` angegebene Name ist, sodass eine Seite
  einer anderen Website, die ihren eigenen Namen auf diesen Rechner verweisen lässt, den Server nicht nutzen kann. HTTPS, ein
  API-Schlüssel, der Unix-Socket der Bereitstellung und Vertex AI (`AIP_MODE=PREDICTION`) sind nicht betroffen. Um einen Server
  mit einfachem HTTP über einen anderen Namen zu erreichen, setzen Sie einen API-Schlüssel.
- **CORS-Proxy.** `/api/cors-proxy` verweigert Anfragen, die ein Browser als Seite, Frame oder Worker öffnen würde
  (403), und gibt jede Antwort als inerten Inhalt zurück: Er kann keine Skripte ausführen und keine Cookies auf der Adresse des Servers setzen.
- **Router-Auflistungen.** Im Router-Modus zeigen `status.args` und `status.preset` von `/api/v1/models` `<redacted>`
  statt der Werte von `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  und `--mcp-servers-json`; das Protokoll verfährt ebenso. Die Modellprozesse erhalten weiterhin die echten Werte.
- **Vertex AI.** Mit `AIP_MODE=PREDICTION` erreicht das `@requestFormat` jeder Instanz nur die Inferenzrouten
  (Completions, Chat-Completions, Responses, Messages, Infill, Embeddings, Rerank, Tokenize, Detokenize,
  Apply-Template und Token-Zählung); jede andere Route, einschließlich der Vorhersageroute selbst, antwortet für diese
  Instanz mit einem Fehler.
- **Modell-Downloads.** Das Hugging-Face-Token (`--hf-token`, `HF_TOKEN`) geht nur an den Hugging-Face-Endpunkt
  (`MGS_MODEL_ENDPOINT`, standardmäßig `https://huggingface.co/`); ein Modell, Projektor oder Entwurfsmodell, das von
  einer anderen URL heruntergeladen wird, wird ohne das Token angefordert.
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
  Alle Medien einer Anfrage zusammen dürfen höchstens 256 Elemente und 512 MiB umfassen, und ihre entfernten Downloads haben
  insgesamt 120 s; mit `--media-path` gelesene Dateien zählen zu denselben Grenzen und werden vor dem Lesen geprüft.
- **Dekodierte Medien.** Bilder bis zu 32768 Pixel pro Seite und 100 Megapixel, Audio bis zu einer Stunde, Videoframes bis
  zu 8192 Pixel pro Seite und 2 GiB dekodierter Frames pro Video sowie höchstens 2 GiB dekodierter Bilder, Audiodaten und
  Videoframes pro Anfrage (einschließlich der Frames all ihrer Videos).
- **Projektoren.** Ein multimodaler Projektor, dessen Metadaten nicht zu seinem Graphen passen (zum Beispiel ein
  Patch-Merge-Faktor unter 2 in einem Projektor, der Patches zusammenführt), lädt nicht und meldet einen Fehler, statt den
  Server anzuhalten.
- **Serverwerkzeuge.** `read_file` mit einem Zeilenbereich liest die Datei blockweise und behält nur diese Zeilen, wobei
  höchstens 256 MiB der Datei durchsucht werden; ohne `end_line` werden Dateien über 16 KB abgelehnt.
- **Chat-Vorlagen des Modells.** `range()` erzeugt höchstens eine Million Elemente, erzeugte Zeichenketten bleiben unter
  256 MiB, die Einrückung von `indent()` und `tojson()` unter 256 Leerzeichen und die Verschachtelung unter 1024 Anweisungen; eine Vorlage,
  die sie überschreitet, schlägt mit einem Vorlagenfehler fehl.
- **Oberfläche.** Links in Nachrichten behalten nur die Schemata http, https, mailto und tel (oder relative Adressen), und
  Bildanhänge müssen eingebettete Bilder sein. Der Import von Konversationen akzeptiert Dateien bis zu 512 MB, ZIP-Archive mit bis
  zu 10 000 Einträgen, 256 MB pro JSONL-Datei und insgesamt 1 GB entpackt. Eine Datei mit fehlerhaften Datensätzen oder mit einer
  Nachricht, die zu einer anderen Konversation gehört, wird vollständig abgelehnt; eine Konversation, deren Nachrichten bereits existieren
  oder deren Nachrichten in einer Schleife aufeinander verweisen, wird übersprungen und gemeldet. Vom Modell geschriebene SVG-Bilder
  bleiben in ihrem eigenen Feld und können den Rest der Seite nicht verdecken. Das Auflisten der Ressourcen oder Ressourcenvorlagen
  eines MCP-Servers endet nach 100 Seiten, 10 000 Elementen, 60 s oder einem wiederholten Cursor. Eine MCP-Ressource wird immer von
  dem Server gelesen, zwischengespeichert und angehängt, von dem sie ausgewählt wurde, auch wenn ein anderer Server dieselbe URI
  anbietet. Eine **Immer erlauben**-Freigabe eines MCP-Werkzeugs gilt, solange sein Server dieselbe Adresse, denselben Transport,
  dieselbe Proxy-Nutzung und dieselben Header behält: Ändert sich eines davon, fragt die Oberfläche erneut.

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

`mgs` berücksichtigt neben dem freien RAM auch die Speichergrenze seiner cgroup (ein
systemd-Dienst mit `MemoryMax`, ein Container). Die automatische Anpassung behält die
MoE-Experten nur dann im RAM, wenn sie in diese Grenze passen, und legt sonst ganze
Expertenschichten auf die GPU. Bleiben die Experten im RAM, belegt der Prompt-Cache
(`--cache-ram`, standardmäßig 8 GiB) höchstens den Speicher, der nach ihnen und der Marge
bleibt; das Protokoll meldet das als `prompt cache limited to N MiB`, und `--cache-ram` legt
ihn explizit fest. Nur die Modellseiten, die im RAM bleiben, werden fixiert, und nur so viele, dass vom noch verfügbaren Speicher eine Reserve (4 GiB oder 10 % des RAM) bleibt; darüber hinaus zeigt das Protokoll `pinning X of Y GiB of mapped weights`, und der Rest nutzt auslagerbaren Speicher. In einer cgroup mit 30 GB und dem 37,6 GB
großen Qwen3.6-35B-A3B Q8_0 hing Version 0.1.17 beim Laden, bis das System sie beendete; jetzt
startet sie in wenigen Minuten und generiert. Bis Version 0.1.22 fixierte `mgs` mit ganzen Expertenschichten auf der GPU auch die bereits auf die GPU kopierten Experten, sodass auf einem Desktop mit wenig freiem RAM und ohne Swap das ganze System beim Laden des Modells nicht mehr reagieren konnte.

Sind die Modelldateien größer als der noch verfügbare Speicher (oder als das Limit der cgroup, in der es läuft), liest `mgs` sie beim Laden nicht vollständig, was nur ihre eigenen ersten Seiten und den Seitencache aller anderen Programme verdrängen würde: Es liest, was auf die GPU kommt und was es fixiert, und den Rest bei der ersten Verwendung. Mit Qwen3.8-Flash-Next (113 GB) in 40 GiB Speicher und einer SATA-Platte sank das Laden von über 5 Minuten auf etwa anderthalb Minuten.

Das Vorabladen der Experten verwendet standardmäßig drei VRAM-Slots. Jeder Slot muss den größten
übertragenen Expertentensor aufnehmen können. Die Werte 2–8 in `GGML_SCHED_PREFETCH_EXPERTS` wählen
ausdrücklich eine Slotanzahl. Schlägt die Speicherreservierung fehl, werden nach Möglichkeit mindestens zwei reservierte Slots
behalten; andernfalls werden normale Übertragungen verwendet. Das Vorabladen beginnt erst,
wenn der Batch mindestens doppelt so viele Expertenauswahlen wie Experten enthält. Sein Pool
gehört der ersten GPU, die es aktiviert; andere GPUs behalten selektive Kopien.
Größere Batches können beim Prefill helfen, benötigen aber mehr Speicher. Auf einer [integrierten GPU](#integrated-gpu) gibt es kein Vorabladen: Einen Slot zu füllen wäre eine Kopie von RAM zu RAM durch die CPU, und die Slots würden dem Page Cache der Experten RAM wegnehmen, deshalb werden die benutzten Experten bei Bedarf kopiert (mit Qwen3.8-Flash-Next auf einer Radeon 860M stieg der Prompt von 15 auf 33 Tokens/s und die Generierung von 4,3 auf 5,0 Tokens/s).

Jeder Slot hat die Größe, die die GPU für den größten Expertentensor anlegt, einschließlich der Auffüllung, die CUDA an Zeilen anhängt, deren Länge kein Vielfaches von 512 Elementen ist, und lässt `--moe-cache-reserve` (standardmäßig 1024 MiB) frei für den Speicher, den die Berechnung während der Ausführung belegt; passt ein Slot nicht, arbeitet das Vorabladen mit den Slots weiter, die passen, sofern es mindestens zwei sind, oder kopiert stattdessen die verwendeten Experten. Bis Version 0.1.31 ließen die Slots diese Auffüllung weg, sodass Qwen3.8-Flash-Next, dessen Experten Zeilen mit 640 Elementen haben, beim ersten langen Prompt mit `CUDA error: an illegal memory access was encountered` (oder `an internal operation failed`) anhalten konnte, sobald die GPU Platz für zwei Slots hatte, zum Beispiel mit MTP auf einer RTX 4060 Ti. Mit Qwen3.8-Flash-Next auf der RTX 4060 Ti verarbeitet MTP mit 2 Tokens den Prompt jetzt mit 12 Token/s und generiert mit 1,31 Token/s.

Zur Diagnose setzen Sie `GGML_CUDA_REGISTER_HOST=0`, `GGML_SCHED_PREFETCH_EXPERTS=0` oder
`GGML_VK_STAGING_RING=0` (Hochladen mit Vulkan über den Staging-Ring), um die entsprechende Optimierung zu deaktivieren.
Setzen Sie diese Variablen im normalen Betrieb nicht, es sei denn, Sie verzichten absichtlich darauf.

Das automatische MTP untersucht die GGUF-Metadaten und Tensornamen vor der Reservierung der Gewichte,
einschließlich jedes Shards. Für das in `--model` integrierte MTP erfordert es eine unterstützte
Architektur, einen Modellstamm und alle integrierten Köpfe. Ein separates MTP-GGUF, das mit
`--spec-draft-model` übergeben wird, aktiviert es ebenfalls, wenn es vollständige MTP-Köpfe deklariert, ohne
Modellstamm; ein Entwurfsmodell ohne MTP-Köpfe behält den spekulativen Typ, den Sie wählen. Ein MTP, das sich
nur in einem anderen GGUF befindet, wird nie gesucht, und eine nicht unterstützte Architektur aktiviert es
nicht. Die Standardgrenze für Entwürfe liegt bei zwei Token: Mit MTP generierten zwei Token auf einer RTX
4060 Ti und auf einer integrierten Radeon 860M genauso schnell wie drei, und ein verworfener Entwurf verschwendet
weniger Prüfaufwand. Jede ausdrückliche Auswahl mit `--spec-type`
hat Vorrang und schaltet die automatische Auswahl aus; `--spec-type none`
deaktiviert MTP. `--spec-draft-n-max` ändert die Grenze. MTP wirkt sich auf die Generierung aus,
nicht direkt auf den Prefill oder die Zeit bis zum ersten Token. Die Verbesserung hängt von der Akzeptanzrate
und der Hardware ab; eine höhere Grenze garantiert keine höhere Geschwindigkeit.

Ob MTP die Generierung beschleunigt, hängt vom Rechner ab: Jeder Schritt prüft die Draft-Tokens mit dem ganzen Modell, was wenig kostet, wenn die Experten in der GPU liegen, und viel, wenn die meisten aus dem RAM oder von der Festplatte kommen. Wenn MTP automatisch aktiviert wird, misst `mgs` es daher während der ersten Antworten: Nach kurzem Aufwärmen wechselt es Schritt für Schritt zwischen 0 (kein Draft), 1, 2 und 3 Draft-Tokens, bis jede Zahl 64 Tokens generiert hat, insgesamt etwa 500 Tokens, behält die schnellste Zahl, wenn der Draft mehr als 5 % schneller ist als kein Draft, und meldet sie im Protokoll mit einer Zeile wie `MTP drafts 2 tokens from now on, the fastest on this machine`. Das Ergebnis wird pro Modell, Rechner und Version von `mgs` in `~/.mgs/tuning/mtp.json` gespeichert, und spätere Starts verwenden es direkt; lohnte sich der Draft nicht, meldet das Protokoll `MTP drafting is not worth it on this machine`, und ab dem nächsten Start wird MTP gar nicht geladen, wodurch sein VRAM auch den Experten, dem Kontext und dem Mikro-Batch bleibt. Das Löschen der Datei (oder ihres Eintrags) führt zu einer neuen Messung. `--spec-draft-n-max N` legt die Zahl der Draft-Tokens fest und schaltet die Messung ab, `--spec-type draft-mtp` lässt MTP eingeschaltet, und ein separates, mit `--spec-draft-model` übergebenes MTP-GGUF wird immer verwendet. Auf der Radeon 860M generierte Qwen3.8-Flash-Next mit 0 / 1 / 2 / 3 Draft-Tokens 5,12 / 7,09 / 7,34 / 6,97 Token/s und behält 2; auf der RTX 4060 Ti generierte es 1,81 / 1,79 / 1,30 / 1,59, läuft daher ab dem nächsten Start ohne MTP, mit 23-24 Token/s beim Prompt und 1,69-1,70 bei der Generierung statt 12 und 1,4, während Qwen3.6-35B-A3B dort 26,38 / 33,92 / 36,21 / 35,41 generierte und 2 behält.

<a id="mtp-vram"></a>
MTP-Gewichte bleiben immer im GPU-Speicher, wenn die GPU sie aufnehmen kann: die auf das Modell aufgepfropften MTP-Schichten, ein
separates MTP-GGUF, das mit `--spec-draft-model` übergeben wird, und der Ausgabekopf, den sie verwenden. Die Optionen, die Gewichte in den
Arbeitsspeicher verschieben (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, ihre `--spec-draft-*`-Entsprechungen und die automatische
Anpassung), gelten für sie nicht, und sie bleiben auch bei einem niedrigen `--gpu-layers` auf der GPU. Die automatische Anpassung platziert den
Rest des Modells um sie herum, und der [MoE-Expertencache](#expert-cache) verwendet den danach verbleibenden Speicher. Wenn eine
MTP-Schicht in den freien Speicher keiner ausgewählten GPU passt, zeigt das Protokoll eine Warnung an (`no selected GPU has enough free
memory for MTP layer 0; using system memory`), und nur diese Schicht bleibt im Arbeitsspeicher; ein Tensor, dessen Typ die GPU
nicht aufnehmen kann, bleibt ebenfalls dort. Auf einer [integrierten GPU](#integrated-gpu) ist der für MTP gezählte freie Speicher höchstens das,
was der System-RAM noch hergeben kann. Mit einem separaten MTP-GGUF werden die auf das Hauptmodell aufgepfropften MTP-Schichten nicht geladen, sodass sie nicht
doppelt Speicher belegen; mit `--spec-type none` wird kein MTP-Gewicht geladen. Das Ladeprotokoll bestätigt die Platzierung mit Zeilen wie
`MTP layer 0 assigned to CUDA0 (856.36 MiB)`, sichtbar mit `-lv 4`.

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
`--n-cpu-moe N` behält einige statische Schichten auf der GPU. Die im RAM gehaltenen Experten werden fixiert, solange der noch verfügbare Speicher diese Reserve behält; der Rest bleibt auslagerbar, sodass das System auch bei knappem RAM weiter reagiert.

Jede Schicht behält ihren eigenen, gleich großen Anteil am Cache, sodass eine Häufung von Fehlzugriffen in einigen Schichten
den Cache der anderen nie leert, und innerhalb jedes Anteils sind Experten, die erneut verwendet werden, vor denen geschützt,
die nur einmal verwendet werden: Ein in den Cache hochgeladener Experte kommt zuerst in einen Probebereich und wechselt in
einen geschützten Bereich (höchstens die Hälfte des Anteils), wenn er erneut verwendet wird, und neue Experten ersetzen zuerst
den Probebereich. Wenn nur ein Teil der fehlenden Experten hochgeladen werden kann, kommen die zuletzt und am häufigsten
verwendeten zuerst. Mit Qwen3.6-35B-A3B Q8_0 auf einer RTX 4060 Ti ging die Generierung von 37,4 auf 38,1 Token/s (mit MTP von
45,2 auf 47,1) und die Cache-Trefferquote von 72,3 auf 74,1 %. Die Anteile halten den Cache auch dann nützlich, wenn ein
Schritt mehr Experten bringt, als der ganze Cache fassen kann, wie mit MTP auf GPUs mit weniger VRAM, wo eine einzige, von
allen Schichten geteilte Liste fast nichts behielt.

<a id="integrated-gpu"></a>
Auf einer integrierten GPU (der Grafik eines Prozessors von AMD oder Intel über Vulkan oder NVIDIA Jetson und DGX Spark
über CUDA) ist der Speicher, den die GPU meldet, ein Teil des System-RAM. Die automatische Anpassung zählt beides als ein einziges
Budget: Die GPU darf höchstens das nutzen, was der RAM nach den Puffern, die auf der CPU bleiben, noch hergeben kann. Wenn das ganze
Modell dort hineinpasst (falls nötig mit reduziertem Kontext, wie bei einer diskreten GPU), läuft alles auf der GPU; andernfalls
kommen ganze Expertenschichten bis zu dieser Grenze auf die GPU, und die CPU berechnet den Rest. Auf einer integrierten GPU gibt es
keinen Expertencache, auch nicht mit `--cpu-moe` oder `--n-cpu-moe`, weil er Experten aus dem RAM in denselben RAM kopieren und sie
doppelt vorhalten würde. Das Protokoll zeigt das Budget in einer Zeile wie
`Vulkan0 is an integrated GPU that shares the system memory: N MiB usable` und die Entscheidung in `no MoE expert cache on an integrated GPU`. Ist das Modell gemappt (`--load-mode mmap`, die Voreinstellung), zählen die Gewichte, die bei allen Schichten auf der GPU auf der CPU bleiben (die zeilenweise gelesenen Token- und schichtweisen Embedding-Tabellen), nicht gegen dieses Budget, weil nur die gelesenen Zeilen im Speicher bleiben: Qwen3.8-Flash-Next, dessen schichtweise Embeddings 28 GB belegen, erhält den gesamten Speicher, den die GPU meldet, statt dessen, was danach übrig bleibt. Auf einer solchen GPU startet sich `mgs` beim Start außerdem einmal mit `OMP_WAIT_POLICY=passive` neu, sofern die Variable nicht schon gesetzt ist, damit die CPU-Threads schlafen statt aktiv zu warten, während die GPU arbeitet: Sie teilen sich mit ihr die Leistung des Prozessors, und mit aktiv wartenden Threads lief die GPU einer Radeon 860M mit etwa dem halben Takt. Das Protokoll zeigt es in der Zeile `the OpenMP threads of the CPU wait passively`; `OMP_WAIT_POLICY=active` behält die aktiv wartenden Threads bei.

<a id="context-size"></a>
Wenn das Modell nicht in den VRAM passt und `-c` nicht angegeben ist, wählt die automatische Anpassung auch die Kontextgröße: Der
KV-Cache belegt höchstens ein Viertel des nach den dichten Gewichten verbleibenden VRAM, bis zum Trainingskontext des
Modells, und die Experten (der Cache oder ganze Expertenschichten) erhalten den Rest. Früher erhielten solche Modelle immer das
Minimum von 4096 Token, das sich alle Slots teilten, wodurch lange Antworten abgeschnitten wurden (`truncated = 1` im Protokoll). Das
Protokoll zeigt das Ergebnis in einer Zeile wie `context size raised from 4096 to N tokens` und in `n_ctx_slot` der
Zeile `initializing`. `-c N` legt den Kontext ausdrücklich fest (mehr Kontext lässt dem Cache weniger VRAM und senkt die
Generierungsgeschwindigkeit: 35,8 Token/s mit 4096 Token und 33,9 mit 122624 auf dem Referenzrechner), und
`--fit-ctx N` erhöht das Minimum. Der MTP-Kontext erhält dieselbe Größe wie der Modellkontext, statt
VRAM für den gesamten Trainingskontext zu reservieren. Füllen ganze Expertenschichten die GPU, weil die Experten nicht für den Cache im RAM bleiben (auf einer [integrierten GPU](#integrated-gpu) immer), belegt der KV-Cache stattdessen höchstens ein Sechzehntel, weil jede Expertenschicht außerhalb der GPU bei jedem Token aus dem RAM gelesen wird, oder von der Platte, wenn das Modell größer als der RAM ist: Qwen3.8-Flash-Next erhält auf einer Radeon 860M mit 46 GB RAM 37376 Tokens und generiert 5,3 Tokens/s, wo es mit einem Viertel 137216 Tokens und 4,9 Tokens/s erhielt.

Die automatische Anpassung zählt den Speicher des MTP-Kontexts für die Kontextgröße, die sie wählt, nicht für den Trainingskontext des Modells: Bis Version 0.1.32 zählte sie dafür mit Qwen3.8-Flash-Next auf einer RTX 4060 Ti 1300 MiB VRAM, wo er etwa 160 MiB belegt, und wählte einen kürzeren Kontext für das Modell. Mit Qwen3.8-Flash-Next und MTP mit 2 Tokens auf der RTX 4060 Ti zählt der Fit für den MTP-Kontext jetzt 130 MiB plus 4,57 KiB pro Kontext-Token und gibt dem Modell einen Kontext von 7936 statt 6912 Tokens.

Vor dem Kontext wählt die automatische Anpassung den Micro-Batch (`-ub`), die Anzahl der Prompt-Token, die in
einem Durchgang verarbeitet werden. Mit den Experten im RAM lädt jeder Durchgang die Experten, die nicht im Cache sind, auf die GPU hoch, sodass weniger,
größere Durchgänge lange Prompts viel schneller verarbeiten, während sich die Generierung nicht ändert. Sofern weder `-b` noch `-ub` angegeben ist,
nimmt sie den größten Wert von 4096, 2048 und 1024 Token, dessen zusätzliche Puffer in ein Achtel des nach den
dichten Gewichten verbleibenden VRAM passen, und erhöht den Batch (`-b`) entsprechend. Auf dem Referenzrechner stieg ein Prompt mit 2514 Token mit 4096 von 214 auf
755 Token/s, bei gleicher Generierungsgeschwindigkeit; die größeren Puffer verkürzen den Standardkontext (etwa 90000
Token ohne MTP und 61000 mit MTP bei 16 GB). Das Protokoll zeigt die Wahl in einer Zeile wie `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

Wenn ganze Expertenschichten die GPU füllen, lassen sie unter der Marge weniger Speicher übrig, als der nächste Expertentensor braucht, und nichts anderes nutzt ihn; die automatische Anpassung vergrößert dann den Mikro-Batch mit diesem Speicher, sodass kein Experte die GPU verlässt. Das Protokoll zeigt das in einer Zeile wie `micro-batch size raised from 2048 to 4096 tokens (batch 4096) with the device memory that the whole expert layers leave unused`. Ob er reicht, hängt vom freien VRAM ab: Mit Qwen3.8-Flash-Next und MTP auf einer RTX 4060 Ti reichte er mit der Standardmarge nicht, mit `--fit-target 800` schon. In diesem Lauf wurde der Prompt mit 21-22 statt 12 Token/s verarbeitet, bei einer Generierung von 1,41-1,46 Token/s (vorher 1,39).

Bei einer ausdrücklichen Platzierung (`--gpu-layers`, `--tensor-split`, `--override-tensor`, `--cpu-moe` oder `--n-cpu-moe`)
behält die automatische Anpassung sie unverändert bei und passt nur den Kontext und den Micro-Batch, die Sie nicht festgelegt haben, um sie herum an,
statt abzubrechen; das Protokoll meldet `the device placement was set explicitly`.

| Option | Variable | Wirkung |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_ARG_MOE_CACHE` (`0` oder `false` deaktiviert) | Aktiviert oder deaktiviert den Cache (standardmäßig aktiviert). |
| `--moe-cache-size MiB` | `MGS_ARG_MOE_CACHE_SIZE` | Feste Gesamtgröße des Caches, aufgeteilt auf die GPUs nach den Experten, die jede zwischenspeichert; `-1` (Standard) verwendet den freien VRAM jeder GPU. |
| `--moe-cache-reserve MiB` | `MGS_ARG_MOE_CACHE_RESERVE` | Auf jeder GPU neben dem Cache und den Vorablade-Slots frei gelassener VRAM (Standard 1024). Erhöhen Sie ihn, wenn andere Programme die GPU verwenden. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnose: fester Anteil (0–1) fehlender Experten, der statt des gemessenen Gleichgewichts hochgeladen wird. |
| — | `MGS_MOE_CACHE_STATS` | Diagnose: protokolliert die Cache-Zähler alle N Dekodierschritte (sichtbar mit `-lv 4`). |
| — | `MGS_MOE_CACHE_PROTECTED` | Diagnose: Anteil (0–1) am Cache-Anteil jeder Schicht für erneut verwendete Experten (Standard 0,5); `0` ergibt eine einfache LRU-Reihenfolge pro Schicht. |

`GET /api/expert-cache` gibt den Zustand des Caches – ob er existiert (`active`) oder die nächste Anfrage ihn noch erstellen
oder eine neue Größe anwenden muss (`pending`) –, die Größe, die zwischengespeicherten Schichten und Slots, Treffer und Fehltreffer, Uploads, CPU-Experten, den
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
laufen darf, außer auf einer CPU mit bis zu 8 logischen CPUs ohne SMT, auf der es alle ihre Kerne verwendet: Bei so wenigen
Kernen ist die CPU der begrenzende Faktor (auf einem i7-7700K mit 4 Kernen und einer RX 580, bei dem die CPU 40 der
41 Expertenschichten berechnet, generierten 4 Threads 17 % schneller als 2). Auf einem Rechner mit SMT (zwei Threads pro
Kern) ist die Obergrenze wie bisher die Anzahl der physischen Kerne; in einer größeren virtuellen Maschine, die kein SMT
anzeigt, ist es die Hälfte der virtuellen CPUs. Auf dem Referenzrechner (16 virtuelle
Kerne mit einer Desktopsitzung, Qwen3.6-35B-A3B Q8_0, ohne MTP) generierten 4 bis 12 Threads mit derselben Geschwindigkeit
(32,9–33,9 Token/s) und 16 Threads mit 28,7 Token/s. Das Startprotokoll zeigt die Wahl in einer Zeile wie
`using 8 CPU threads for generation and 8 for batches`; `-t` und `-tb` legen sie ausdrücklich fest.

<a id="kv-cache-type"></a>
### Typ des KV-Caches

Ohne `-ctk` (`--cache-type-k`) oder `-ctv` (`--cache-type-v`) gibt `mgs` dem KV-Cache die Genauigkeit der Modellgewichte, gemessen am Typ, der die meisten Expertengewichte enthält: Nicht quantisierte Modelle (F16, BF16, F32) behalten einen `f16`-Cache, Modelle mit mehr als 5 Bit pro Gewicht (Q5_K_M, Q6_K, Q8_0) erhalten `q8_0` und die übrigen (Q4_K_M, IQ4_XS, MXFP4, Q3_K, Q2_K…) `q4_0` – die beiden quantisierten Typen, die Flash Attention in allen Backends liest. Die Regel ist in allen Backends dieselbe (CPU, CUDA, ROCm und Vulkan). Ein quantisierter Cache braucht Flash Attention, daher bleibt der Cache bei `f16`, wenn Flash Attention deaktiviert ist (`-fa off`) oder mit diesem Typ auf der GPU nicht laufen kann. Das Startprotokoll zeigt die Wahl in einer Zeile wie `KV cache type chosen from the weights of the model: q8_0`, und die Seite **Modell** zeigt die verwendeten Typen.

Ein `q8_0`-Cache belegt etwa halb so viel Speicher wie ein `f16`-Cache und ein `q4_0`-Cache etwa 28 %, sodass die automatische Anpassung im selben Speicher viel mehr Kontext vergibt oder diesen Speicher dem Expertencache überlässt, auf Kosten einer etwas ungenaueren Attention und einiger Prozent Geschwindigkeit. Das Lesen eines quantisierten Caches kostet in allen Backends etwas zusätzliche Arbeit, und CUDA (NVIDIA-GPUs) und ROCm (AMD-GPUs mit einem `--rocm`-Build), die dieselben Flash-Attention-Kernel nutzen, lesen ihn nur direkt, wenn sehr wenige Token auf einmal verarbeitet werden (die normale Generierung, nicht der Prompt und nicht die MTP-Prüfung mehrerer Token) und die Attention-Köpfe 64, 128 oder 256 Werte haben; sonst wandeln sie ihn bei jedem Schritt in `f16` um. Auf einer Radeon 860M (Vulkan) mit Qwen3.6-35B-A3B Q6_K und ohne MTP generierte `q8_0` nach einem kurzen Prompt 3,8 % weniger Token pro Sekunde als `f16` und nach einem Prompt von 8060 Token 2,7 % weniger und verarbeitete diesen Prompt um 1,3 % langsamer. Wird nur eine der beiden Optionen angegeben, bleibt die andere bei `f16`, und `-ctk f16 -ctv f16` stellt den `f16`-Cache wieder her, wenn der Speicher reicht und diese wenigen Prozent wichtiger sind als der Kontext. Der MTP-Kontext und das Entwurfsmodell von `--spec-draft-model` folgen derselben Regel; `--spec-draft-type-k` und `--spec-draft-type-v` legen ihre Typen fest.

<a id="checkpoints"></a>
### Kontext-Prüfpunkte

Modelle mit rekurrenten Schichten oder Schichten mit gleitendem Fenster (zum Beispiel die Hybride Qwen3.5/3.6/3.8 oder Nemotron-H)
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

MTP trennt Graphen mit und ohne Ausgaben, erhält die ursprüngliche Tokenreihenfolge über gleichzeitige Sequenzen hinweg und verwendet nach Bildern die tatsächlichen Positionen.

Die automatische Anpassung zählt die nextn-Schichten (MTP) mit, sodass die erste Schicht nicht mehr auf der CPU bleibt, wenn MTP deaktiviert ist. MTP-Gewichte bleiben nun immer im GPU-Speicher; siehe [MTP-Platzierung](#mtp-vram). Der Standardwert von `--checkpoint-min-step` ist nun 1024 statt 8192; siehe [Kontext-Prüfpunkte](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` und `--moe-cache-reserve` steuern den [MoE-Expertencache](#expert-cache). Die automatische Anpassung lässt Modelle, die größer als der VRAM sind, nicht mehr mit einem Kontext von 4096 Token zurück und wählt nun auch den Micro-Batch; siehe [Kontextgröße](#context-size). Die Standardanzahl der CPU-Threads beträgt nun höchstens die Hälfte der logischen CPUs; siehe [CPU-Threads](#threads). `-np` verwendet nun standardmäßig einen Slot mit dem gesamten Kontext statt des automatischen Modus mit vier Slots, die sich einen gemeinsamen KV-Cache teilen, und `--kv-unified` ist standardmäßig deaktiviert; siehe [Anfragewarteschlange](#request-queue). Bei einer integrierten GPU zählt die automatische Anpassung den GPU-Speicher und den RAM nun als ein gemeinsames Budget und erstellt keinen MoE-Expertencache mehr; siehe [integrierte GPUs](#integrated-gpu). Seit 0.1.19 bricht die automatische Anpassung nicht mehr ab, wenn die Platzierung ausdrücklich angegeben ist: Sie behält diese Platzierung bei und passt den Kontext und den Micro-Batch um sie herum an; siehe [Kontextgröße](#context-size).

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
| Umgebungsvariablen, CMake-Optionen und Symbole der C-API mit dem früheren Upstream-Präfix, einschließlich der Aliase, die frühere Versionen bereits aufgegeben hatten | Der aktuelle Name mit dem Präfix `MGS_` (`mgs_` in der C-API), zum Beispiel `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` statt einer `NO_`-Variante, `MGS_ARG_LOAD_MODE` statt der Variablen für Memory-Mapping, Sperren und Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` und `MGS_ARG_SPEC_DRAFT_N_MIN` für die Entwurfsgrenzen oder `MGS_BUILD_UI`; für die Backend-Optionen die passende Option `GGML_*`, zum Beispiel `GGML_CUDA` oder `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `HF_HOME` und `MGS_CACHE` | Nichts; jeder Download, auch von Hugging Face, geht nach `~/.cache/mgs/` (unter `XDG_CACHE_HOME`, wenn gesetzt). |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` oder `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Wert `deepseek-legacy` von `reasoning_format` (Anfragefeld und `--reasoning-format`) | `deepseek` oder `auto`; der Denkprozess wird in `reasoning_content` zurückgegeben |
| Anfragefeld `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (Array von Zeichenketten) |
| `reasoning_in_content` in `generation_settings` | Nichts; das Feld wurde entfernt. |
| Download einer vorkompilierten Upstream-Oberfläche aus einem Hugging-Face-Bucket (die CMake-Optionen `USE_PREBUILT_UI` und `UI_HF_BUCKET` und die Umgebungsvariable `HF_UI_VERSION`) | Nichts; die Oberfläche wird immer aus `frontend/` kompiliert (`MGS_BUILD_UI=ON`) oder weggelassen. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Build-Nummer und Commit: die CMake-Variablen `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (auch in `mgs-config.cmake`) und die Zeichenkette `b0-unknown` von `build_info`, `system_fingerprint` und dem `User-Agent` | `MGS_VERSION`, festgelegt in `backend/cmake/build-info.cmake`; diese Felder enthalten nun die Version, etwa `0.1.19` |
| Meta-Llama-Modelle (die GGUF-Architekturen `llama`, `llama4` und `llama-embed` und der Bildprojektor von Llama 4) sowie die integrierten Chat-Vorlagen `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` und `llama4` | Nichts; verwenden Sie eines der unterstützten MoE-Modelle. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Installationsoption `--no-sudo` | Nichts; die Installationsprogramme laufen immer als root ohne sudo. |
| Funktionen und Felder, die die C-APIs als veraltet markiert hatten: in `mgs.h` `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, die Aliase `mgs_token_*` (zum Beispiel `mgs_token_bos`, `mgs_token_get_text` oder `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, das Feld `defrag_thold` von `mgs_context_params` und `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; in `ggml.h` `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` und `ggml_upscale_ext`; in `mtmd.h` `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` und das Feld `image_marker` | Die aktuelle Funktion: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` statt `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` und `media_marker`. Aufwärmläufe erledigt der Aufrufer, und `defrag_thold` hatte keine Wirkung. |
| Gemma 4: die Architekturen `gemma4` und `gemma4-assistant`, die Projektoren `gemma4v`, `gemma4uv`, `gemma4a` und `gemma4ua`, das Chat-Format `peg-gemma4`, der Tokenizer und Pre-Tokenizer `gemma4` sowie die Voreinstellungen `--vision-gemma-4b-default` und `--vision-gemma-12b-default` | Nichts; verwenden Sie eine der [unterstützten Familien](#supported-models). |
| DeepSeek-Modelle vor V4.1: die Architekturen `deepseek`, `deepseek2` (auch die damit konvertierten Dateien von Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 und Youtu), `deepseek2-ocr` (mit den Projektoren `deepseekocr` und `deepseekocr2`), `deepseek32`, `mistral4` und `deepseek4`, die Pre-Tokenizer `deepseek-llm` und `deepseek-coder`, das DSpark-Entwurfs-Backbone von DeepSeek-V4 und die Erkennung der Chat-Vorlage von DeepSeek-V3.2 (DSML mit `function_calls`) | DeepSeek-V4.1-Flash (`deepseek41`). |
| Die übrigen von llama.cpp geerbten MoE-Architekturen: `grok`, `nomic-bert-moe`, `qwen2moe`, `qwen3moe`, `qwen3next`, `qwen3vlmoe`, `phimoe`, `minicpm`, `jamba`, `cohere2moe`, `dbrx`, `olmoe`, `arctic`, `glm4moe`, `exaone-moe`, `granitemoe`, `granitehybrid`, `bailingmoe`, `bailingmoe2`, `dots1`, `afmoe`, `laguna`, `ernie4_5-moe`, `hunyuan-moe`, `hy_v3`, `gpt-oss`, `lfm2moe`, `smallthinker`, `llada-moe`, `grovemoe`, `minimax-m2`, `rnd1`, `mistral3`, `step35`, `kimi-linear` und `mellum`, sowie die Chat-Formate von Ministral 3, gpt-oss, Functionary v3.2, Kimi-K2, Cohere2-MoE, LFM2, GigaChat3 und MiniCPM5 | Nichts; verwenden Sie eine der [unterstützten Familien](#supported-models). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) und die integrierten Vorlagen der Nicht-Jinja-Engine (`--chat-template` mit einem Namen wie `deepseek3`, `chatglm4` oder `mistral-v7-tekken`) | Nichts; Chat-Vorlagen laufen immer über Jinja. `--chat-template chatml` wählt weiterhin ChatML; für eine andere Vorlage übergeben Sie ihren Jinja-Quelltext mit `--chat-template` oder `--chat-template-file`. Anfragen mit `tools` benötigen kein Flag mehr. |
| `-hfr` | `-hf` |
| Zweite lange Namen einer Option: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe` (die Schlüssel der Router-Voreinstellungen folgen denselben Namen) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, benannt nach ihren Optionen |
| Anfragefeld `thinking_budget_tokens` | `reasoning_budget_tokens` (Anthropic-Clients senden weiterhin `thinking.budget_tokens`) |
| `content` in `POST /api/embeddings` und `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, die Struktur `mgs_chat_message` und `MGS_FILE_MAGIC_GGLA` in `mgs.h`; `mtmd_image_tokens_get_id` in `mtmd.h` | Nichts; die Chat-Formatierung übernehmen die Jinja-Vorlagen von `common/chat.h`, und die übrigen hatten keine Verwendung. |
| `MGS_API_KEY`, `MGS_ARG_BATCH`, `MGS_ARG_UBATCH`, `MGS_ARG_N_PREDICT`, `MGS_ARG_N_PARALLEL`, `MGS_ARG_N_GPU_LAYERS`, `MGS_ARG_EMBEDDINGS`, `MGS_ARG_RERANKING`, `MGS_ARG_LOG_VERBOSITY`, `MGS_ARG_ROPE_SCALING_TYPE`, `MGS_ARG_SPEC_DRAFT_HF_REPO`, `MGS_ARG_STATIC_PATH`, `MGS_ARG_ENDPOINT_METRICS`, `MGS_ARG_ENDPOINT_PROPS`, `MGS_ARG_ENDPOINT_SLOTS` | `MGS_ARG_API_KEY`, `MGS_ARG_BATCH_SIZE`, `MGS_ARG_UBATCH_SIZE`, `MGS_ARG_PREDICT`, `MGS_ARG_PARALLEL`, `MGS_ARG_GPU_LAYERS`, `MGS_ARG_EMBEDDING`, `MGS_ARG_RERANK`, `MGS_ARG_VERBOSITY`, `MGS_ARG_ROPE_SCALING`, `MGS_ARG_SPEC_DRAFT_HF`, `MGS_ARG_PATH`, `MGS_ARG_METRICS`, `MGS_ARG_PROPS`, `MGS_ARG_SLOTS`: Jede Variable ist `MGS_ARG_`, gefolgt vom langen Namen ihrer Option in Großbuchstaben, mit `_` statt `-` |
| `MGS_MOE_CACHE`, `MGS_MOE_CACHE_MIB`, `MGS_MOE_CACHE_RESERVE_MIB` | `--moe-cache`/`--no-moe-cache`, `--moe-cache-size` und `--moe-cache-reserve` oder `MGS_ARG_MOE_CACHE`, `MGS_ARG_MOE_CACHE_SIZE` und `MGS_ARG_MOE_CACHE_RESERVE` |
| `MTMD_BACKEND_DEVICE`, `MTMD_DEBUG_EMBEDDINGS` und die CMake-Option `MTMD_VIDEO` | `MGS_MTMD_BACKEND_DEVICE`, `MGS_MTMD_DEBUG_EMBEDDINGS` und `MGS_MTMD_VIDEO` |
| `--spec-mtp-auto`, `--no-spec-mtp-auto` (`MGS_ARG_SPEC_MTP_AUTO`) | Nichts: MTP schaltet sich von selbst ein; `--spec-type none` deaktiviert es, und jeder andere `--spec-type` ersetzt es. |
| `--spec-default` | `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, die Werte, die es setzte |
| Die Modellvoreinstellungen `--tts-oute-default`, `--embd-gemma-default`, `--fim-qwen-1.5b-default`, `--fim-qwen-3b-default`, `--fim-qwen-7b-default`, `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec`, `--fim-qwen-30b-default`, `--gpt-oss-20b-default` und `--gpt-oss-120b-default` | Nichts; übergeben Sie das Modell mit `--model` oder `-hf`. |
| Die Optionen der Befehlszeilenwerkzeuge, die `mgs` nicht enthält, und ihre Variablen, etwa `-p`/`--prompt`, `-f`/`--file`, `-sys`/`--system-prompt`, `-i`/`--interactive`, `-cnv`/`--conversation`, `--hellaswag`, `--winogrande`, `--kl-divergence`, `--save-all-logits`, `--ppl-stride`, `-o`/`--output`, `--chunk`, `--image`, `--audio`, `--video`, `-pps`, `-tgs`, `-npp`, `--diffusion-steps`, `-lr`/`--learning-rate`, `-mv`/`--model-vocoder`, `-hfv`/`--hf-repo-v`, `--tts-speaker-file`, `-fitp`/`--fit-print`, `--show-timings`, `--prompt-cache`, `--tensor-filter`, `-gan`/`--grp-attn-n` sowie die Download-Optionen `--mtp`, `--dflash` und `--eagle3` | Nichts; der Server hat sie nie verwendet. |
| Die Werte `draft-eagle3`, `draft-dflash` und `draft-dspark` von `--spec-type`, mit den Entwurfsmodellen Eagle3, DFlash und DSpark | `draft-mtp`, `draft-simple` oder einer der N-Gramm-Typen |
| Die ggml-Backends außer CPU, CUDA, Vulkan und ROCm (HIP): BLAS, CANN, ET, Hexagon, Metal, MUSA, OpenCL, OpenVINO, RPC, SYCL, virtgpu, WebGPU, zDNN und ZenDNN, mit ihren CMake-Optionen (etwa `GGML_BLAS`, `GGML_METAL`, `GGML_RPC` oder `GGML_SYCL`) und `--rpc` | CPU, CUDA (`--cuda`), ROCm (`--rocm`) oder Vulkan (`--vulkan`) |
| In `mgs.h` `mgs_get_logits`, `mgs_get_embeddings`, `mgs_encode`, `mgs_set_causal_attn`, `mgs_model_has_encoder`, `mgs_model_has_decoder`, `mgs_model_decoder_start_token`, `mgs_model_is_recurrent`, `mgs_get_ctx_other` und das Feld `ctx_other` von `mgs_context_params`; in `mgs-ext.h` `mgs_set_embeddings_layer_inp`, `mgs_get_embeddings_layer_inp`, `mgs_model_target_layer_ids` und `mgs_model_target_layer_ids_n`; in `mtmd.h` `mtmd_decode_use_non_causal`, `mtmd_image_tokens_get_n_tokens` und `mtmd_image_tokens_get_n_pos` | `mgs_get_logits_ith`, `mgs_get_embeddings_ith` oder `mgs_get_embeddings_seq`, `mgs_decode` (das auch Modelle ohne KV-Cache ausführt), `mtmd_input_chunk_get_n_tokens` und `mtmd_input_chunk_get_n_pos`; die übrigen dienten Encoder-Decoder-, reinen Embedding- oder dichten Modellen oder dem zwischen Kontexten geteilten KV-Cache und haben keinen Ersatz. |
| Die multimodalen Projektoren von Gemma 3 (`gemma3`, `gemma3nv`, `gemma3na` und MobileNetV5) | Nichts; verwenden Sie den Projektor einer der [unterstützten Familien](#supported-models). |
| Die OpenAI-Routen ohne `/v1` (`GET /api/models`, `POST /api/chat/completions`, `POST /api/responses`, `POST /api/audio/transcriptions`, `POST /api/chat/completions/input_tokens`, `POST /api/responses/input_tokens`) und `GET /api/v1/health` | Dieselbe Route unter `/api/v1` (`GET /api/v1/models`, `POST /api/v1/chat/completions`…) und `GET /api/health`. `POST /api/models` und `DELETE /api/models` des Router-Modus bleiben. |
| Die Felder `ahead_api` und `ahead_webui` der Ereignisse `mgs_queue` | `running_api`, `running_webui`, `waiting_api` und `waiting_webui` ([Anfragewarteschlange](#request-queue)) |
| Installationsvariablen `MGS_INSTALL_SOURCE_DIR`, `MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`, `MGS_BUILD_JOBS` und `MGS_BUILD_BACKEND` sowie die Option `--build-ui` der Installationsprogramme und von `build/build-local.sh` | `--build-dir`, `--prefix`, `--jobs` und `--cuda`, `--rocm`, `--vulkan` oder `--cpu`; die Quellen kommen immer nach `/opt/moe-gguf-server-source`, und die Oberfläche wird immer gebaut, sofern nicht `--no-ui` angegeben ist. |
| Die Konfiguration der Webdienste als eigener Schritt nach der Installation | Das Installationsprogramm konfiguriert sie im selben Lauf (`--domain`, `--certificate`, `--private-key`, `--proxy-ip`); `configure-web.sh` bleibt für eine Neukonfiguration ohne Neukompilierung. |
| Der Inferenzdienst auf `127.0.0.1:8080` (`--port 8080`) hinter Apache | Der Unix-Socket `/opt/moe-gguf-server/_/temp/web/mgs.sock` ([Bereitstellung](#deployment)) |

CMake meldet entfernte `-D`-Variablen als unbenutzt, statt sie anzuwenden; prüfen Sie daher
die Konfigurationsausgabe manueller Builds.

Die Weboberfläche speichert ihre Einstellungen (`~/.mgs/config/ui-settings.json`) und ihre Chats (`~/.mgs/chats/`) auf dem Server
und im Browser nur ihren API-Schlüssel (`MgsUi.apiKey` in `localStorage`). Sie liest oder migriert die
Einstellungen nicht, die ältere Versionen in `localStorage` gespeichert haben (`MgsUi.config`,
`MgsUi.userOverrides`, die Lieblingsmodelle, die Werkzeuglisten, den Denkaufwand,
die verworfenen MCP-Empfehlungen, die Sprache und den Farbmodus der früheren Theme-Bibliothek), sodass die Einstellungen mit
ihren Standardwerten beginnen und der API-Schlüssel einmal erneut eingegeben werden muss. Ebenso wenig liest oder
migriert sie Daten, die ältere Versionen unter anderen Namen gespeichert haben, den eigenständigen Schlüssel `theme`,
eingebettete Markierungen für Denkprozess und Werkzeugaufrufe in alten Nachrichten, Anhänge mit eingefügtem Text
vom alten Typ `context`, das alte Listenformat der verworfenen MCP-Empfehlungen
oder das frühere Feld `webui_settings` des Servers. Ebenso wenig liest sie die Konversationen, die ältere Versionen im Browser gespeichert haben (die IndexedDB-Datenbank `MgsUi`):
Um sie zu behalten, exportieren Sie sie vor dem Upgrade aus der älteren Version und importieren den Export anschließend, wodurch
sie auf dem Server gespeichert werden. Der Import von Konversationen akzeptiert nur die JSONL- und ZIP-Exporte, innerhalb der unter
[Sicherheitsgrenzen](#security-limits) aufgeführten Grenzen; das alte JSON-Exportformat wird abgelehnt. Die alten Einstellungen bleiben im Browser, bis sie gelöscht werden: Um
sie zu entfernen, öffnen Sie auf der Seite der Oberfläche die Browserkonsole (F12) und führen
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
einmal für jede Adresse und jeden Port aus, mit denen die Oberfläche geöffnet wurde.

Jeder Download wird in `~/.cache/mgs/` zwischengespeichert (unter `XDG_CACHE_HOME`, wenn gesetzt):
mit `-mu`/`--model-url` heruntergeladene Dateien und, mit ihrer eigenen Struktur
(`~/.cache/mgs/hub/models--<organization>--<repo>/`), Downloads von Hugging Face. Keine
Variable ändert diesen Ordner. Die Dateien, die ältere Versionen in einem anderen Cache-Ordner gespeichert haben,
einschließlich des Hugging-Face-Caches (`~/.cache/huggingface/hub`), werden erneut heruntergeladen.

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

- Prüfen Sie mit `mgs --version`, welche Version installiert ist (es gibt zum Beispiel `version: 0.1.57` aus); der lokale Build
  führt den Befehl am Ende aus, und CMake gibt beim Konfigurieren `mgs version: 0.1.57` aus. Die Oberfläche zeigt dieselbe Version
  unter Einstellungen → Anzeige → **Informationen zur Build-Version anzeigen** (untere rechte Ecke) und unter **Build-Informationen** im
  Dialog mit den Modellinformationen. Jede Änderung am Projekt erhöht die letzte Zahl: 0.1.0, 0.1.1, 0.1.2 usw.
- Wiederholen Sie `curl | bash`, um von `main` zu aktualisieren, oder den passenden lokalen Installationsbefehl, um die Binärdatei neu zu kompilieren und zu ersetzen.
  Übergeben Sie dieselben Weboptionen erneut (`--domain`, `--proxy-ip`…), weil jeder Lauf die Webkonfiguration neu schreibt. Das
  Installationsprogramm startet am Ende beide Dienste neu; um den Inferenzdienst von Hand neu zu starten, führen Sie
  `systemctl restart moe-gguf-server` unter Debian oder `rc-service moe-gguf-server restart` unter Alpine aus.
- Lesen Sie `/root/webapp-install.log`, `journalctl -u moe-gguf-server` unter Debian
  oder `/var/log/moe-gguf-server.log` unter Alpine. Die Apache-Protokolle liegen unter
  `/var/www/DOMAIN-logs/`.
- Wenn die CUDA-Kompilierung nicht verfügbar ist, installieren Sie das Toolkit separat oder verwenden Sie
  `--rocm`, `--vulkan` oder `--cpu`. Wenn NCCL nicht gefunden wird, prüfen Sie die Verfügbarkeit des Repositorys und die CUDA-Version,
  oder wählen Sie ausdrücklich `--no-nccl`. Wenn ROCm nicht gefunden wird, installieren Sie es mit seinen Entwicklungsdateien oder setzen Sie
  `HIPCXX` ([ROCm](#rocm)).
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

Siehe [CODE.de-DE.md](CODE.de-DE.md) für die Implementierungsübersicht und die Erweiterungspunkte.

<a id="license"></a>
## Lizenz

`mgs` wird unter der [PolyForm NonCommercial License 1.0.0](../LICENSE) vertrieben. Sie dürfen es für jeden
nicht kommerziellen Zweck (NonCommercial) verwenden, verändern und weitergeben: persönliche Nutzung, Studium, Forschung, Hobbyprojekte
sowie die Nutzung durch gemeinnützige Organisationen, Bildungseinrichtungen, öffentliche Forschungseinrichtungen und
staatliche Einrichtungen. Eine kommerzielle Nutzung, etwa durch ein Unternehmen für seine Geschäftstätigkeit, erfordert eine
separate Lizenz von nipegun.

Die Teile, die von llama.cpp (MIT, The ggml authors), FreeToken (Apache 2.0) und den
mitgelieferten Bibliotheken stammen, behalten ihre eigenen Lizenzen; ihre Hinweise befinden sich in `build/licenses/`, und der lokale
Build installiert sie zusammen mit `LICENSE` in `~/.local/share/doc/mgs/licenses/`.
