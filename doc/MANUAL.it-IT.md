# Manuale utente

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Indice: [installazione](#installation), [compilazione locale](#local-build), [distribuzione](#deployment),
[chat](#chat), [monitoraggio](#monitor), [modelli e strumenti](#models), [modelli supportati](#supported-models), [API](#api),
[coda delle richieste](#request-queue), [limiti di sicurezza](#security-limits), [Vulkan](#vulkan), [prestazioni](#performance), [compatibilità rimossa](#removed-compatibility),
[manutenzione](#maintenance), [licenza](#license).

<a id="installation"></a>
## Installazione e primo avvio

Installa come root sull'host di produzione. Debian/Ubuntu accetta il programma di
installazione direttamente dallo standard input:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

Non occorre scaricare prima nessun file accessorio. Lo script di ingresso scarica
l'archivio completo dei sorgenti di `main` da [questo repository](https://github.com/nipegun/moe-gguf-server/), lo
verifica e sincronizza una copia gestita in `/opt/moe-gguf-server-source`. I download
e le cache di compilazione restano sotto la cartella `_/temp/` del progetto. Curl e i
certificati CA servono per scaricare lo script di ingresso; le dipendenze di avvio
mancanti vengono poi installate automaticamente.

Per scegliere un'altra cartella dei sorgenti e passare opzioni di compilazione, usa:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

La cartella remota deve essere un percorso assoluto e trovarsi su un disco locale. Il
programma di installazione adotta solo una cartella vuota o riutilizza una copia che
porta il suo marcatore di origine dei sorgenti; le cartelle estranee vengono rifiutate.
Un lock impedisce installazioni simultanee in quella copia. Le esecuzioni successive di
`curl | bash` sincronizzano i file sorgente, compresi quelli rimossi da `main`, preservando
`_/` e le cache di compilazione. Le chiavi API, i modelli e i servizi distribuiti si
trovano fuori da quell'albero dei sorgenti e non vengono modificati. Un download o una
compilazione non riusciti non installano un nuovo eseguibile; riesegui lo stesso comando
per riprovare. Non serve alcuna operazione Git.

Uno script scaricato da solo usa la stessa modalità remota. Eseguire
`bash deploy/install-update-reinstall-debian.sh` da una copia locale completa compila
quella copia senza scaricare i sorgenti. Gli argomenti dopo `bash -s --` vengono
inoltrati al programma di installazione della compilazione. `--help` non richiede né root
né il download dei sorgenti. `--no-system-deps` disattiva anche l'installazione dei
pacchetti di avvio: curl, i certificati CA, tar, gzip, rsync, util-linux e coreutils
devono essere già presenti. I sottoprocessi del programma di installazione non possono
consumare lo script dalla pipe; tutte le fasi condividono un unico flusso di output
`/root/webapp-install.log`.

La configurazione web resta un passaggio separato che usa
`/opt/moe-gguf-server-source/deploy/configure-web.sh` o il suo equivalente nella
cartella dei sorgenti scelta.

Alpine usa ancora una copia locale completa e
`bash deploy/install-update-reinstall-alpine.sh`, dopo aver installato Bash.
Debian usa per impostazione predefinita CUDA/NCCL; Alpine usa per impostazione predefinita la CPU e non supporta
il percorso di installazione CUDA/NCCL. Usa `--cpu` per una compilazione Debian per CPU o `--vulkan` per GPU AMD, Intel o NVIDIA
tramite Vulkan ([Vulkan](#vulkan)). Il CUDA Toolkit e i driver NVIDIA devono essere già installati. `--no-nccl` disattiva esplicitamente NCCL;
`--no-system-deps` lascia la gestione dei pacchetti all'amministratore.

L'interfaccia locale richiede Node.js 20.19+ nella serie 20.x oppure 22.12+. `--no-ui` la omette. Altre
opzioni sono `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST`, `--no-openssl` e `--build-ui`.
`MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`,
`MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND` e `CUDACXX` forniscono le corrispondenti
sostituzioni tramite variabili d'ambiente. Le cartelle di compilazione e di staging devono essere
sottocartelle distinte e non sovrapposte della cartella `_/temp/` del progetto.

Il programma di installazione aggiunge i pacchetti NCCL corrispondenti e il repository di NVIDIA quando serve,
ma non installa né aggiorna mai il CUDA Toolkit o i driver della GPU. Collega staticamente le
librerie interne e installa `/usr/local/bin/mgs`. Nessuno dei due
programmi di installazione usa o installa sudo.

Su Debian/Ubuntu, `libnccl2` e `libnccl-dev` vengono installati insieme alla stessa
versione esatta selezionata per la release CUDA rilevata. Questo comando APT usa
`--allow-downgrades`, così un pacchetto NCCL installato più recente non blocca l'installazione
della versione selezionata. Il flag viene passato solo al comando di installazione di NCCL.

Se la distribuzione include un Node.js più vecchio, installa prima una versione supportata.
L'interfaccia viene sempre compilata in locale; la compilazione non può scaricare risorse
precompilate di upstream, che non hanno le traduzioni né le route API di questo progetto. Una
compilazione dell'interfaccia non riuscita interrompe l'installazione.

Da una copia locale completa, scegli una compilazione per CPU o elenca le opzioni di installazione:

```bash
bash deploy/install-update-reinstall-debian.sh --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

Su Alpine, installa Bash con `apk add --no-cache bash` prima di eseguire il suo programma di installazione.

Esegui un singolo modello con:

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

Non serve nessun'altra opzione: `mgs` sceglie le impostazioni che offrono la massima velocità di inferenza sulla
macchina su cui gira. L'adattamento automatico colloca il modello sulla GPU e, quando la memoria di sistema lo consente,
mantiene gli esperti MoE nella RAM, così che la VRAM libera diventi la [cache degli esperti MoE](#expert-cache); sceglie
inoltre il [micro-batch e la dimensione del contesto](#context-size). Il numero di thread della CPU segue il processore
([Thread della CPU](#threads)) e i layer MTP presenti nel GGUF vengono attivati e mantenuti nella VRAM
([MTP](#mtp-vram)). Qualsiasi opzione indicata esplicitamente (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`,
`--override-tensor`…) prevale sulla scelta automatica. Il log di avvio mostra che cosa è stato scelto.

Apri l'URL HTTPS stampato all'avvio, normalmente `https://127.0.0.1:11443/`.
Il caricamento di un modello di grandi dimensioni può richiedere tempo. L'interfaccia
web e la documentazione dell'API restano disponibili mentre le richieste al modello
segnalano il caricamento. `--help` elenca le opzioni ereditate del motore. I file GGUF, la
memoria della GPU, la lunghezza del contesto e i proiettori specifici del modello devono
essere adeguati al carico di lavoro previsto.

Il codice del server attiva HTTPS per impostazione predefinita con ogni metodo di compilazione e di installazione.
Si mette in ascolto sulla prima porta HTTPS disponibile a partire da 11443, poi su una porta di reindirizzamento HTTP
a partire da 11080, saltando la porta HTTPS scelta. HTTP risponde con un reindirizzamento permanente 308
verso la porta HTTPS effettiva, preservando host, percorso e query richiesti.
Il log stampa gli URL selezionati; i conflitti di porta fanno avanzare ciascuna sequenza in modo indipendente.
`--port N` fissa la porta principale e fallisce se è occupata; `--port 0` lascia che sia il sistema operativo
a sceglierla. `--http-port N` cambia l'inizio della ricerca della porta di reindirizzamento.
Le variabili d'ambiente corrispondenti sono `MGS_ARG_PORT` e `MGS_ARG_HTTP_PORT`.

Senza file di certificato, l'eseguibile crea un certificato autofirmato e la sua
chiave privata al primo avvio, senza richiedere un comando `openssl` esterno,
e li salva in `~/.mgs/certificates/` (`fullchain.pem` e
`privkey.pem`, con la chiave leggibile solo da te). Ogni esecuzione successiva, e ogni nuova compilazione,
li riutilizza, così il browser mantiene la sua eccezione di attendibilità e non serve ricaricare; una nuova
coppia viene creata solo quando quei file mancano, non corrispondono o il certificato
è scaduto (dopo 365 giorni). Il certificato copre `localhost`, gli indirizzi di
loopback, l'host indicato con `--host`, e il nome della macchina e gli indirizzi di rete
che aveva quando è stato creato; per eliminare del tutto l'avviso del browser, importa
`~/.mgs/certificates/fullchain.pem` come attendibile, ed elimina entrambi i file per crearne
uno nuovo. Se quella cartella non è scrivibile, ogni esecuzione genera il proprio certificato in
memoria. Fornisci sia `--ssl-cert-file /path/fullchain.pem` sia `--ssl-key-file
/path/privkey.pem` per usare un certificato considerato attendibile dai tuoi client. Una coppia mancante o non valida interrompe
l'avvio. Le compilazioni fatte con `--no-openssl` non possono servire HTTPS e richiedono un
`--http` esplicito (o `MGS_ARG_HTTPS=false`); l'HTTP semplice inizia la ricerca della porta da 11080.
I processi figli del router usano HTTP esplicito su porte di loopback private. I socket Unix e
`AIP_MODE=PREDICTION` di Vertex AI mantengono il loro trasporto HTTP.

La configurazione di CMake si trova in `build/CMakeLists.txt`. Sia gli script di installazione sia quelli
di compilazione locale includono `build/` nella loro copia dei sorgenti e lo selezionano automaticamente.
I programmi di installazione reimpostano la configurazione CMake generata se una cache esistente punta
a un'altra cartella dei sorgenti. Le compilazioni locali partono sempre da una cartella nuova sotto
`/tmp/` e la rimuovono all'uscita. Per usare CMake direttamente, esegui
`cmake -S build -B _/temp/cmake` dalla radice del progetto; i comandi abituali di installazione e di
compilazione locale restano gli stessi. HTTPS usa per impostazione predefinita l'OpenSSL di sistema
(`MGS_OPENSSL=ON`); in una compilazione CMake diretta, `-DMGS_BUILD_BORINGSSL=ON` o
`-DMGS_BUILD_LIBRESSL=ON` scarica e compila invece BoringSSL o LibreSSL, e
`-DMGS_OPENSSL=OFF` senza nessuno dei due compila il server senza HTTPS.

<a id="local-build"></a>
## Compilazione locale per l'uso manuale

`build/build-local.sh` compila il progetto sul tuo computer e installa una
copia autonoma per il tuo utente, che avvii a mano. Non installa servizi, non avvia
nulla in background, non usa mai root né sudo e non installa mai pacchetti.
Eseguilo con il tuo utente abituale, da qualsiasi cartella:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

Il comando su una riga scarica l'archivio dei sorgenti di `main` senza Git nella cartella
di compilazione privata sotto `/tmp`. L'archivio e i sorgenti estratti vengono rimossi
insieme agli altri file temporanei di compilazione all'uscita. Una copia autonoma dello
script funziona allo stesso modo. Da una copia locale completa del progetto, compila i
tuoi sorgenti locali con:

```bash
bash build/build-local.sh
```

Anche eseguire questo script tramite il suo percorso assoluto usa il progetto completo
che lo contiene. Il messaggio `Using local sources: ...` conferma quale cartella viene
compilata, comprese le traduzioni della sua interfaccia. Se l'esecuzione termina con
`Error: Compilation failed.`, non è stato installato nessun nuovo eseguibile; un'installazione
esistente mantiene la sua interfaccia precedente. Leggi il primo errore del compilatore sopra
`ninja: build stopped`, correggine la causa e riesegui lo stesso script. Quando la
compilazione e l'installazione riescono, riavvia `mgs` e ricarica l'interfaccia web.

Errori come `std::vector` senza un membro chiamato `mPushBack` o
`no declaration matches` in `subproc.cpp` indicano nomi incoerenti nei sorgenti.
Usa i file sorgente corretti e riesegui la compilazione. La compilazione dell'interfaccia
richiede inoltre che gli import della sua configurazione corrispondano ai nomi esportati
dai moduli sorgente; un export mancante viene segnalato dal passaggio npm/Vite.

Se SvelteKit segnala `Page options are ignored when router.type === 'hash'`
e nomina `fLoad`, le route della chat devono esportare esattamente il nome `load`
del framework. Le route corrette mantengono la convalida della chiave API. La compilazione
esegue `svelte-kit sync` prima di generare le risorse e compilare l'interfaccia, e
si interrompe se la sincronizzazione fallisce. Un errore in `Provisioning UI assets`
impedisce comunque l'installazione anche quando le librerie C++ sono già state compilate.

Per passare opzioni tramite il comando su una riga, usa `bash -s --`; per esempio, una compilazione per CPU:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` mostra le opzioni senza scaricare l'archivio dei sorgenti né compilare.

Il risultato viene installato per il tuo utente sotto il prefisso `~/.local`:

| percorso | contenuto |
| --- | --- |
| `~/.local/bin/mgs` | Il programma che esegui: un piccolo launcher statico che trova `../lib/mgs/` rispetto a sé stesso e avvia il server tramite il loader incluso. |
| `~/.local/lib/mgs/` | Il binario del server (`mgs`, con l'interfaccia web e la documentazione dell'API incorporate) e tutte le librerie condivise di cui ha bisogno: il loader dinamico e la libreria C di glibc, il runtime C++, OpenMP, OpenSSL, i moduli name-service di glibc e, nelle compilazioni CUDA, il runtime CUDA, cuBLAS e NCCL. |
| `~/.local/share/doc/mgs/licenses/` | La licenza PolyForm NonCommercial 1.0.0 del progetto (`LICENSE`) e gli avvisi di licenza di terze parti copiati da `build/licenses/` dell'albero dei sorgenti (llama.cpp, FreeToken, Swagger UI, cpp-httplib e nlohmann/json). |

Solo l'utente che esegue lo script può usare questa installazione. Tutto viene risolto rispetto alla posizione del
launcher: esegue `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` dal suo prefisso, e il server e le sue librerie cercano
in `lib/mgs/` prima che altrove, compreso `LD_LIBRARY_PATH`. L'unica libreria ancora presa dal sistema è il
driver NVIDIA (`libcuda.so.1` e le librerie `libnvidia-*` che carica), perché deve corrispondere al modulo del kernel
in esecuzione; con `--vulkan`, anche il driver Vulkan del sistema ([Vulkan](#vulkan)). Alla fine lo script elenca, tramite il loader incluso, le librerie che il server caricherebbe, fallisce se qualcuna
proviene da fuori `lib/mgs/` ed esegue `--version` tramite il launcher.

Debian aggiunge `~/.local/bin` al `PATH` all'accesso, una volta che quella cartella esiste (tramite `~/.profile`), quindi dopo aver
effettuato di nuovo l'accesso puoi eseguire `mgs` per nome; lo script stampa un promemoria finché non è nel `PATH`. Avvialo quando
vuoi e fermalo con Ctrl+C:

```bash
mgs --model /path/model.gguf
```

Funziona anche il percorso completo: `~/.local/bin/mgs --model /path/model.gguf`.

Apri `https://127.0.0.1:11443/` per la chat e `https://127.0.0.1:11443/api/doc/` per
la documentazione dell'API. Si applicano tutte le opzioni di questo manuale; per esempio,
`--models-dir` attiva il router multimodello.

Requisiti, che installi tu stesso come root perché lo script si limita a verificarli:
`cmake`, un compilatore C/C++ e la libreria C statica (`build-essential`, che
porta con sé `libc6-dev`), `patchelf`, `libssl-dev` (oppure
`--no-openssl`), Node.js 20.19+/22.12+ con npm (oppure `--no-ui`), un CUDA
Toolkit già installato (oppure `--cpu`) e NCCL con i suoi file di sviluppo (`libnccl2` e
`libnccl-dev`, oppure `--no-nccl`). `nvcc` viene preso da `CUDACXX`, dal `PATH` o da
`/usr/local/cuda/bin/nvcc`. Ninja viene usato se installato.
Le utilità di base comprendono `tar`, `coreutils` e `util-linux`. Le compilazioni remote richiedono
anche `curl`, `ca-certificates` e `gzip`; nessuna dipendenza viene installata automaticamente.

Opzioni: `--prefix PATH` (un altro prefisso di installazione), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`
e `--help`.

Buono a sapersi:

- La compilazione è ottimizzata per la CPU di questo computer e per le GPU presenti durante la
  compilazione. Ricompila su ogni computer invece di copiare l'installazione.
- L'installazione è rilocabile come prefisso: sposta `bin/mgs` e `lib/mgs/` insieme, mantenendo i loro livelli `bin/` e
  `lib/`. Avvia sempre `bin/mgs`; `lib/mgs/mgs` è interno e non va eseguito direttamente, perché
  scavalcherebbe il loader incluso.
- Rieseguire lo script aggiorna l'installazione. Sostituisce solo `bin/mgs`, `lib/mgs/` e
  `share/doc/mgs/licenses/`. Una copia già in esecuzione continua a funzionare; riavviala per usare la nuova compilazione.
- Lo script rifiuta un prefisso in cui `bin/mgs`, `lib/mgs/` o quella cartella delle licenze esistono già, a meno che non sia stato lui
  a crearli; lascia un marcatore nascosto `.moe-gguf-server-bundle` in `lib/mgs/`.
- `--prefix PATH` installa sotto un altro prefisso in cui il tuo utente può scrivere; non può essere `/` né una cartella all'interno
  del progetto.
- I sorgenti preparati, i file di compilazione, le cache, l'installazione assemblata e il log usano una
  cartella privata `/tmp/moe-gguf-server-build.XXXXXXXX/`. Dopo aver copiato il risultato
  nel prefisso, lo script rimuove quella cartella. Fa pulizia anche in caso di errori
  e di interruzioni gestite. Ogni esecuzione compila da zero, senza scrivere nell'albero
  dei sorgenti né in `/opt/`. Le compilazioni locali non accettano `--build-dir`.
- Il prefisso viene bloccato durante la copia e la sostituzione degli artefatti completati,
  impedendo aggiornamenti simultanei della stessa installazione.
- Una compilazione CUDA ha bisogno del driver NVIDIA per avviarsi; usa `--vulkan` sui computer con un'altra GPU e `--cpu` sui
  computer senza GPU.

<a id="vulkan"></a>
## Vulkan (GPU AMD, Intel e NVIDIA)

Oltre a CUDA (predefinito) e alla CPU, sia `build/build-local.sh` sia il programma di installazione Debian compilano il backend Vulkan
con `--vulkan`, per le GPU senza CUDA come le schede AMD Radeon. Installa prima come root le dipendenze di compilazione:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Poi, con il tuo utente abituale:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

Il programma di installazione Debian accetta la stessa opzione e installa da sé quei pacchetti Vulkan:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` mostra la GPU, per esempio `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- L'installazione locale include il loader Vulkan (`libvulkan.so.1`) come qualsiasi altra libreria, ma il driver Vulkan
  proviene dal sistema: il loader legge i driver registrati in `/usr/share/vulkan/icd.d/` (Mesa RADV per AMD,
  ANV per Intel, il driver NVIDIA) e questi caricano le proprie librerie di sistema, perché devono corrispondere al kernel
  e al sistema. Compila sulla stessa release di Debian che eseguirà il binario.
- L'utente che esegue `mgs` ha bisogno del nodo di rendering della GPU (`/dev/dri/renderD*`). Un utente che ha effettuato l'accesso al desktop
  ce l'ha già; gli altri utenti hanno bisogno del gruppo `render` (e di `video`). `configure-web.sh` aggiunge l'account del servizio a entrambi.
- MTP nella VRAM, la cache degli esperti MoE, il prefetch degli esperti, `--fit` e la pagina Monitoraggio funzionano come con CUDA. Su AMD la
  pagina Monitoraggio legge utilizzo, VRAM e temperatura da sysfs.
- `mgs` conta come memoria della GPU solo la VRAM reale (gli heap di memoria device-local), mai la GTT. La GTT è la parte della
  RAM di sistema (normalmente la metà) che una GPU AMD o Intel può mappare: non è memoria aggiuntiva, e la GPU la legge
  tramite PCIe, molte volte più lentamente della sua VRAM. I layer MTP, i pesi densi, la cache KV e la cache degli esperti vengono
  allocati nella VRAM; gli esperti che non ci stanno restano nella RAM normale, dove li calcola la CPU.
- `mgs` verifica dove il kernel tiene davvero quella memoria (`/proc/self/fdinfo`, pubblicato da amdgpu, i915 e xe). Se
  un altro programma riempie la VRAM e il driver ne sposta una parte nella GTT, il log e la pagina Monitoraggio lo segnalano, e la
  pagina Monitoraggio mostra la posizione reale (VRAM e GTT) di ogni GPU AMD o Intel. La cache degli esperti viene allocata con una
  priorità di memoria più bassa (`VK_EXT_memory_priority`), così il driver la sposta prima dei layer MTP, dei pesi o della
  cache KV. Senza resizable BAR, i buffer del dispositivo evitano anche la piccola finestra di VRAM che la CPU può mappare (256 MiB su una
  RX 580), che il desktop condivide e che il driver svuota per prima nella GTT.
- Solo CUDA può bloccare in memoria (pinning) le pagine del modello che restano nella RAM, quindi con Vulkan gli esperti raggiungono la GPU tramite
  memoria paginabile. `--load-mode none` li copia invece nella memoria host bloccata, cosa che su una RTX 4060 Ti tramite Vulkan
  (Qwen3.6-35B-A3B Q4_K) ha portato i caricamenti degli esperti da 2,7 a 13,1 GB/s e la generazione da 27,5 a 36,6 token/s. Su
  AMD e Intel quella memoria bloccata conta nella GTT: quando gli esperti non ci stanno, il caricamento fallisce con
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), come è successo con i 18,9 GB di esperti di quel
  modello con 16,8 GB di GTT su una RX 580. Usala solo quando gli esperti stanno nella GTT e nella RAM libera.
- Dimensiona il modello in base alla macchina: l'intero GGUF dovrebbe stare nella RAM libera più la VRAM, meno la cache KV e i
  buffer di calcolo. Con 8 GB di VRAM e 16 GB di RAM ciò significa circa 18–20 GB di pesi; i file più grandi vengono riletti
  dal disco continuamente e la generazione diventa molto lenta. Il Q4_K da 21,6 GB citato sopra ha raggiunto 7,8 token/s su una
  RX 580 con 15 GB di RAM e 15,9 token/s quando la macchina ha avuto 31 GB (10,0 con `--spec-type none`).

<a id="deployment"></a>
## Distribuzione in produzione

Dalla cartella dei sorgenti (`/opt/moe-gguf-server-source` per le installazioni remote),
dopo aver installato il binario, esegui:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

Il dominio deve risolversi nell'host di produzione. Il servizio `moe-gguf-server-web`
esegue un'istanza dedicata di Apache come `moe-gguf-server`, senza root.
Ogni avvio seleziona la prima porta HTTP disponibile a partire da 11080 e la prima porta HTTPS
disponibile a partire da 11443. Ciascuna sequenza avanza in modo indipendente fino a 65535, senza
assegnare la stessa porta a entrambi i protocolli. Ispeziona i listener TCP IPv4/IPv6
e riprova se Apache rileva che una porta è stata occupata dopo la selezione. Gli altri
errori di configurazione interrompono l'avvio invece di essere scambiati per conflitti di porta.

HTTP reindirizza in modo permanente alla porta HTTPS selezionata, preservando percorso e query.
Il reindirizzamento non viene memorizzato nella cache. Apache inoltra le richieste di inferenza a
`127.0.0.1:8080`, avviato esplicitamente con `--http --port 8080` e anch'esso eseguito come
`moe-gguf-server`. L'installazione richiede root;
nessuno dei due servizi deve essere eseguito come root. Non viene configurato nessun HAProxy e non vengono
importati né modificati siti Apache condivisi.

Gli URL effettivi vengono pubblicati dopo che entrambi i listener si sono aperti:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

Per esempio, HTTP 11081 e HTTPS 11445 reindirizzano a
`https://ai.example.com:11445/`. Il file viene rimosso allo spegnimento e rigenerato
all'avvio; un riavvio può scegliere porte diverse. I client, i firewall e qualsiasi
proxy esterno devono usare i valori pubblicati. La porta fa parte
dell'[origine del browser](https://developer.mozilla.org/en-US/docs/Glossary/Origin), quindi una
porta HTTPS diversa ha cronologia e impostazioni locali separate. Le conversazioni esistenti
restano nella loro origine iniziale; l'esportazione e l'importazione possono trasferirle.

L'account di sistema non ha una password interattiva. I modelli vengono archiviati in
`/opt/moe-gguf-server/models` e la configurazione del programma si trova nella home
dell'account del servizio, `/opt/moe-gguf-server/.mgs/config/`: la chiave API (`api.key`), il dominio
web (`web-domain`), il modello di configurazione di Apache (`apache2.conf.in`) e le impostazioni
dell'interfaccia (`ui-settings.json`), tutti di proprietà di `moe-gguf-server` con modo 0600.
Le credenziali e l'output dell'installazione vengono registrati in `/root/webapp-credentials.txt`
e `/root/webapp-install.log`, entrambi con modo 600. La riconfigurazione preserva le chiavi,
i modelli e i certificati esistenti. La chiave privata TLS è di proprietà di root e leggibile dal gruppo
moe-gguf-server (modo 640), il che consente al processo web non privilegiato di leggerla.

Usa il tuo certificato TLS con:

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

Il certificato si trova in `/opt/moe-gguf-server/certificates/` (`fullchain.pem` e
`privkey.pem`, di proprietà dell'account `moe-gguf-server`, con la chiave leggibile solo da esso).
Senza file forniti, lo script mantiene il certificato già presente oppure, la prima
volta, ne crea uno autofirmato, che ogni reinstallazione successiva riutilizza, così i browser
mantengono la loro eccezione di attendibilità. Consideralo attendibile sul tuo client oppure fornisci un certificato
che i tuoi client considerano già attendibile. I file forniti vengono copiati in quella cartella; il rinnovo del certificato
deve aggiornare quelle copie e riavviare `moe-gguf-server-web`. Lo script non ottiene né rinnova certificati Let's Encrypt.
Usa lo stesso comando dopo aver rinnovato i certificati di origine. L'opzione
`--binary` dello script consente di indicare una posizione non predefinita dell'eseguibile.

Il servizio usa la modalità router. Metti modelli GGUF leggibili in
`/opt/moe-gguf-server/models`, oppure usa la gestione dei modelli nell'interfaccia. Per
opzioni specializzate della riga di comando, usa un override di systemd su Debian/Ubuntu o la
configurazione del servizio OpenRC su Alpine. Mantieni il backend associato al loopback.
Il configuratore web rigenera il suo file di servizio gestito quando viene rieseguito.

Riavvia il servizio web per selezionare di nuovo le porte:

```bash
systemctl restart moe-gguf-server-web
```

Su Alpine usa `rc-service moe-gguf-server-web restart`. I log di avvio si trovano in
`journalctl -u moe-gguf-server-web` su Debian o in `/var/log/moe-gguf-server-web.log`
su Alpine. La diagnostica di Apache si trova in `/opt/moe-gguf-server/_/temp/web/startup.log`
e `error.log`; i log di accesso del dominio restano in `/var/www/DOMAIN-logs/`. Il modello
autonomo è `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. Non modificare la configurazione
generata in `_/temp/web/`; viene ricostruita a ogni avvio.

<a id="chat"></a>
## Uso dell'interfaccia

1. Apri l'URL del server e inserisci la sua chiave API quando richiesto, oppure in Impostazioni → Generali.
2. Seleziona un modello. In modalità router, carica uno dei modelli disponibili se non ne è caricato nessuno.
3. Scrivi un messaggio e invialo. Per impostazione predefinita Invio invia; Maiusc+Invio inserisce un
   a capo. Queste scorciatoie si possono cambiare nelle impostazioni.
4. Aggiungi file con l'azione di allegato. Immagini, audio e video richiedono modalità
   compatibili del modello. I file PDF possono essere estratti come testo o convertiti in immagini;
   i modelli senza visione usano il percorso del testo.
5. Ferma la generazione, salta il ragionamento quando è supportato, modifica un messaggio, rigenera una
   risposta o crea una diramazione di una conversazione con le azioni dei messaggi.

La barra laterale consente di cercare, fissare, rinominare, eliminare e selezionare più elementi insieme.
Le conversazioni vengono archiviate nel browser tramite IndexedDB; esportale prima di cancellare
i dati del browser o di cambiare profilo del browser. Le impostazioni vengono conservate dal server in
`~/.mgs/config/ui-settings.json` dell'utente che esegue `mgs` (nel servizio distribuito,
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), quindi tutti i browser che aprono
l'interfaccia le condividono: parametri di generazione, opzioni di visualizzazione, tema, lingua, server
MCP, modelli preferiti, strumenti disattivati e sempre consentiti e lo sforzo di ragionamento
predefinito. Solo la chiave API resta in ciascun browser, perché autentica la richiesta
che carica il resto, quindi inseriscila una volta in ogni browser. L'importazione e l'esportazione sono disponibili nelle
impostazioni. Le esportazioni possono contenere allegati e, se selezionate esplicitamente, chiavi API
sensibili o intestazioni di autorizzazione MCP.

Scegli la lingua in Impostazioni → Generali. Le 15 opzioni sono ordinate `de-DE`,
`en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`,
`ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; quella predefinita è `en-US`.
Il portoghese brasiliano e quello europeo hanno traduzioni separate. L'ebraico usa un'interfaccia
da destra a sinistra; l'input della chat e i messaggi semplici dell'utente seguono la direzione del
loro testo, mentre il codice e l'output del terminale vengono mostrati da sinistra a destra.
Cambiare lingua salva la scelta nelle impostazioni del server e
ricarica l'interfaccia; il server la pubblica in `/api/config.js`, quindi ogni browser
si avvia in quella lingua. Salva prima le altre impostazioni in sospeso. Il parametro di query `lang`
dell'URL la sostituisce per quella pagina e mantiene la scelta quando non è possibile salvarla.
Per esempio, aggiungi `?lang=ja-JP` all'URL dell'interfaccia per selezionare il giapponese.

Impostazioni → Generali → **Rispondi sempre nella lingua dell’interfaccia** è disattivata per
impostazione predefinita. Attivala e salva le impostazioni per indicare al modello di usare la
lingua e la variante regionale dell'interfaccia per ogni nuova risposta, anche nelle
conversazioni esistenti, nelle risposte rigenerate, nelle continuazioni, nei turni dopo le chiamate
agli strumenti e nei titoli generati dal modello. Per `es-AR`, l'istruzione chiede esplicitamente
il voseo argentino: «Seguí estos pasos», «tenés» e «podés»; `es-ES` chiede
l'uso dello spagnolo di Spagna. L'inglese britannico/americano e il portoghese brasiliano/europeo
ricevono anch'essi istruzioni regionali distinte.

L'istruzione accompagna ogni nuova richiesta di inferenza e ogni precaricamento della cache KV;
viene combinata con il messaggio di sistema in uscita senza modificare il prompt di sistema
salvato né la conversazione. Cambiare la lingua dell'interfaccia influisce sulle nuove
richieste dopo il ricaricamento; una risposta già in fase di generazione mantiene
l'istruzione che ha ricevuto. Disattivare l'impostazione smette di aggiungerla alle richieste
successive. Le risposte esistenti non vengono tradotte. Il codice, i comandi, i nomi, le citazioni
e le traduzioni richieste esplicitamente mantengono la lingua necessaria; le spiegazioni
che li circondano seguono la lingua dell'interfaccia. La capacità del modello
di seguire l'istruzione determina la coerenza; non si tratta di una traduzione automatica
né di una garanzia di rispetto perfetto. I client esterni dell'API non sono
influenzati da questa impostazione del browser.

Scegli Sistema, Chiaro o Scuro per il tema. Le impostazioni di generazione comprendono l'utilizzo
del contesto, la temperatura, i campionatori, le penalità e i limiti dei turni dell'agente. Alcune funzionalità
dipendono dai flag del server o dal modello selezionato.

Dopo ogni risposta l'interfaccia invia di nuovo la conversazione, quando il server è
inattivo, così che sia già codificata nella cache KV mentre leggi e il turno successivo
inizi prima. È attivo per impostazione predefinita; disattiva Impostazioni → Sviluppo → **Precarica cache KV
dopo ogni risposta** per fermarlo.

Mentre il server sta elaborando un'altra richiesta, un messaggio che invii attende nella
[coda](#request-queue), e la sua risposta mostra un avviso al posto del testo di avanzamento finché
non inizia: «In coda: prima viene elaborata un'altra richiesta inviata direttamente all'API
(richieste prima della tua: N).» quando un programma usa l'API nello stesso momento, oppure lo stesso avviso
per una richiesta della stessa interfaccia web (un'altra scheda o un altro browser, un titolo generato o il
precaricamento della cache KV). L'avviso scompare non appena la tua richiesta inizia.

<a id="monitor"></a>
## Monitoraggio e barra di stato

**Monitoraggio** si trova sotto **Impostazioni** nella barra laterale (indirizzo `#/monitor`). Mentre è aperto si aggiorna ogni
due secondi e mostra:

- **Utilizzo**: per prima una scheda per ogni GPU della macchina (NVIDIA, AMD o Intel, anche se nessun modello la usa), intitolata con
  la sua etichetta in grassetto e il suo nome (per esempio **gpu-1**: NVIDIA GeForce RTX 4060 Ti; con più GPU, gpu-1, gpu-2 e
  così via), con il suo utilizzo, l'utilizzo del controller di memoria (larghezza di banda), la VRAM in uso, la temperatura e il consumo
  rispetto al suo limite; poi una scheda per ogni processore, intitolata allo stesso modo con il suo modello (per esempio **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; con più processori, cpu-1, cpu-2 e così via), con il suo utilizzo, una piccola barra verticale per ogni
  CPU logica di quel processore e il carico medio del sistema. Ogni barra di utilizzo è verde e si riempie di rosso man mano che
  l'utilizzo cresce; una barra il cui valore è sconosciuto resta grigia. Le GPU NVIDIA vengono lette tramite la libreria NVML del driver e le GPU AMD
  tramite `gpu_busy_percent` in sysfs; qualsiasi altra GPU (Intel, o NVIDIA senza NVML) non mostra l'utilizzo, e il suo
  nome e la sua memoria provengono dal backend di inferenza quando un modello la usa.
- **Memoria**: una barra per ogni GPU (VRAM) e una per la RAM; una GPU senza dati sulla memoria, come una GPU Intel che nessun
  modello usa ancora, non ha una barra. Ogni barra è suddivisa per uso, ciascun uso sempre con lo stesso colore, e la sua
  legenda elenca ogni segmento con la sua dimensione e la sua quota sul totale, uno per riga e nello stesso ordine della
  barra. Passa il puntatore su un segmento per vederne la dimensione.

  | Segmento | Contenuto |
  | --- | --- |
  | Pesi MTP / del modello bozza | Pesi dei layer MTP (nextn) del modello, di un GGUF MTP separato passato con `--spec-draft-model` o di un modello bozza. |
  | Pesi degli esperti | Tensori degli esperti instradati del modello. |
  | Pesi densi | Tutti gli altri pesi: attenzione, esperti condivisi, embedding e output. |
  | Cache KV | Memoria del contesto del modello (cache KV o stato ricorrente). |
  | Cache KV MTP / del modello bozza | Memoria del contesto MTP o del contesto del modello bozza. |
  | Buffer di calcolo | Buffer temporanei dei grafi di calcolo. |
  | Cache degli esperti e precaricamento | Cache degli esperti MoE (pool, identificatori e staging) e gli slot di precaricamento degli esperti. |
  | Cache dei prompt e checkpoint | Prompt salvati con `--cache-ram` e checkpoint del contesto degli slot (RAM). |
  | Altro | Memoria usata che nessuna delle voci precedenti spiega: altri processi, il desktop, i contesti del driver, il proiettore multimodale. |
  | Libera | Il resto del dispositivo o della RAM. |

  I pesi mappati dal file del modello contano come RAM anche quando il kernel segnala quella page cache come disponibile, quindi i
  segmenti possono sommare più della memoria «in uso»; la barra allora non mostra «Altro» e mostra meno memoria libera. La
  cache degli esperti compare dopo la prima richiesta, quando viene creata. In modalità router le barre includono ogni modello
  caricato, e la pagina Monitoraggio non carica né risveglia mai un modello.

La **barra di stato** in fondo a ogni pagina mostra prima l'utilizzo di ogni GPU e poi quello di ogni processore
fisico, ciascuno nella propria barra, con le stesse etichette delle schede di Monitoraggio: `gpu-1`, `gpu-2`... e `cpu-1`,
`cpu-2`... Le barre hanno la stessa dimensione, si aggiornano ogni due secondi e, come quelle di Monitoraggio, sono verdi e si riempiono
di rosso man mano che l'utilizzo cresce; il testo al passaggio del puntatore di ciascuna indica il nome del dispositivo. Senza GPU mostra una barra
`GPU` vuota, e quando le barre non stanno in uno schermo stretto la barra di stato scorre lateralmente. Fai clic su di essa per aprire
Monitoraggio. Continua a funzionare mentre il modello si carica. È mostrata per impostazione predefinita; disattiva Impostazioni → Visualizzazione → **Mostra barra
di stato** per nasconderla.

<a id="models"></a>
## Modelli, strumenti e MCP

`mgs` accetta solo modelli principali MoE. I modelli densi come GPT-2 o Phi-2 vengono rifiutati prima che i loro pesi siano caricati; le loro implementazioni del decoder non vengono compilate. I layer densi all'interno di un modello MoE e una testa solo MTP compatibile fornita esplicitamente vengono mantenuti.

<a id="supported-models"></a>
### Famiglie di modelli supportate

| Famiglia | Architettura GGUF | Note |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP quando il GGUF lo include. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP quando il GGUF lo include. La sua tabella di embedding per layer (circa 29 GB) viene letta dal file del modello su richiesta, quindi tieni il file su un disco veloce. I contesti lunghi vengono calcolati con attenzione densa sotto la selezione sparsa: il risultato è corretto, ma senza l'accelerazione dei kernel sparsi. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; immagini con il suo mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | I GGUF pubblicati non includono MTP. Con Vulkan il suo mixing residuo viene eseguito sulla CPU. |
| DeepSeek-V4.1-Flash | `deepseek41` | Carica entrambi i layout pubblicati: i file DwarfStar di antirez (come `antirez/deepseek-v4.1-flash-gguf`) e i file dell'ecosistema llama.cpp. La sua tabella Engram (circa 188 GiB all'interno del file) non viene mai caricata: ogni token legge le sue righe dal file del modello, quindi il file deve restare disponibile mentre `mgs` è in esecuzione, preferibilmente su un SSD, ed è necessario Linux o un altro sistema POSIX. I file senza un modello di chat usano uno integrato, e `reasoning_effort` accetta un numero da 1 a 100. Niente MTP e niente immagini. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

Gemma 4 e i modelli DeepSeek precedenti a V4.1 non sono più supportati (vedi [compatibilità rimossa](#removed-compatibility)).

Con più GPU selezionate, MTP le controlla tutte prima di ripiegare sulla RAM. La sua stima della memoria include la testa di output. I buffer MTP integrati vengono caricati prima degli altri pesi; un GGUF MTP esterno passato con `--spec-draft-model` viene caricato prima del modello principale, anche con `--fit off`. La selezione esplicita attuale dei dispositivi determina comunque quali GPU sono disponibili.

Usa `--reasoning on` o `--reasoning off` per controllare il ragionamento. Passare `enable_thinking` tramite `--chat-template-kwargs` viene rifiutato. Usa i nomi delle opzioni documentati con il trattino; le grafie con il trattino basso vengono rifiutate. Passa i valori di tipo lista come un unico valore separato da virgole invece di ripetere un'opzione. Solo `--spec-type` resta ripetibile per comporre modalità speculative.

Il selettore dei modelli mostra i modelli disponibili, caricati e preferiti. I dettagli
del modello includono la dimensione del contesto, le modalità, la quantizzazione e il modello di chat. La modalità
router gestisce processi figli separati per i modelli caricati; le richieste identificano il modello
tramite il campo `model` o il parametro di query.

I server MCP si configurano tramite URL con intestazioni di autorizzazione facoltative. Puoi
esaminare i log di connessione, i prompt, le risorse e gli strumenti disponibili. Il proxy CORS
facoltativo del server deve essere attivato esplicitamente con il flag corrispondente
del motore; restituisce solo contenuto inerte e rifiuta di essere aperto come pagina (vedi
[limiti di sicurezza](#security-limits)). Le restrizioni del browser sulle risorse HTTP all'interno delle pagine HTTPS restano valide.
Quando due server MCP attivati offrono uno strumento con lo stesso nome, nessuna delle due copie viene offerta al modello e
l'interfaccia mostra un avviso: disattiva tutti quei server tranne uno per usare lo strumento.
Gli strumenti integrati del server e la sandbox JavaScript del browser sono facoltativi.
Le approvazioni degli strumenti e il limite dei turni dell'agente restano disponibili nell'interfaccia. Il limite dei turni
(Impostazioni → Agente → **Turni dell'agente**, il numero massimo di cicli di esecuzione degli strumenti
prima di fermarsi, per evitare cicli infiniti) è 50 per impostazione predefinita.

Gli stream possono riconnettersi usando l'identità e il cursore della conversazione salvati finché il
server conserva lo stream. Un server fermato o uno stream scaduto non può garantire una
riproduzione. L'interfaccia segnala una riconnessione non riuscita invece di
creare silenziosamente una seconda richiesta di generazione.

### Server MCP della shell locale

Avvia `mgs` con `--local-shell-mcp-server` per consentire ai modelli della chat web di usare il terminale della macchina su cui
`mgs` è in esecuzione:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- All'avvio `mgs` controlla quali shell sono installate e offre uno strumento per ciascuna di esse: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) e `psh.run` (PowerShell, `pwsh`). Il log elenca le shell trovate; una shell
  non installata non riceve nessuno strumento.
- L'interfaccia lo rileva, lo aggiunge ai server MCP come **local shell mcp server** e lo attiva. Puoi disattivarlo lì,
  o per una singola conversazione, e la tua scelta viene mantenuta. Quando `mgs` si avvia senza l'opzione, la voce
  scompare.
- Ogni strumento accetta `command` e, facoltativamente, `cwd` (cartella di lavoro; per impostazione predefinita la cartella home dell'utente
  che esegue `mgs`) e `timeout` (secondi; 60 per impostazione predefinita, 3600 al massimo). La risposta è un oggetto JSON con
  `exit_code`, `stdout` e `stderr`, più `signal`, `timed_out`, `cancelled`, `timeout`, `seconds`, `shell` e
  `cwd`; gli output lunghi conservano solo l'inizio e la fine.
- Ogni comando viene eseguito in un nuovo processo senza terminale e con input vuoto: `cd` e le variabili non si conservano
  nella chiamata successiva, e i programmi interattivi (editor, domande) non si possono usare. Quando scade il timeout o la
  richiesta viene annullata, ogni processo avviato dal comando viene terminato. I processi lasciati in background devono
  reindirizzare il loro output.
- L'interfaccia chiede il permesso prima di ogni chiamata, a meno che tu non scelga di consentire sempre lo strumento.
- `sudo` funziona senza terminale (Linux): quando ha bisogno di una password, si apre una finestra al centro della pagina,
  in qualunque pagina tu ti trovi, con il messaggio di sudo (`[sudo] password for user:`), lo strumento e il comando che
  la chiede, un campo per la password e, sotto di esso, i pulsanti **Accetta** e **Annulla**. Il cursore è già nel
  campo della password, quindi puoi digitarla subito; **Invio** accetta ed **Esc** annulla; un clic
  fuori dalla finestra non la chiude. La password arriva solo a sudo sulla macchina su cui `mgs`
  è in esecuzione: non viene salvata nella conversazione, non viene scritta nei log e il modello non la vede mai. Mentre la
  finestra è in attesa, il timeout del comando smette di scorrere; se premi **Annulla**, o nessuno risponde entro 5
  minuti, sudo fallisce. sudo chiede di nuovo a ogni chiamata, perché ognuna è un nuovo processo, e fino a tre volte quando
  la password è sbagliata; la finestra avvisa allora che la password precedente potrebbe essere sbagliata. I comandi vengono eseguiti in una sessione
  propria, senza il terminale da cui è stato avviato `mgs`, quindi sudo non chiede mai la password in quel terminale.
  Solo l'interfaccia web di `mgs` può rispondere: con altri client MCP, sudo fallisce come prima per mancanza di un terminale.
  Il servizio distribuito viene eseguito come account `moe-gguf-server`, che normalmente non ha né una password né i permessi
  sudo, quindi questa funzione è pensata per `mgs` eseguito con il tuo utente (per esempio, la compilazione in `~/.local`). Quando `mgs`
  non ha la variabile `DISPLAY`, i comandi ricevono `DISPLAY=:mgs-askpass`, perché sudo usa il suo programma di aiuto per la password
  senza terminale solo quando `DISPLAY` esiste; i programmi che tentano di aprire una finestra falliscono subito.
- Nella chat, ogni chiamata viene mostrata con il nome del suo strumento, come `bash.run`. Espansa, mostra fino a tre
  caselle grigie con un carattere a spaziatura fissa: **stdin** con il solo comando, **stdout** con l'output e **stderr** con
  il testo dell'errore. stdout compare solo quando il comando ha stampato qualcosa e stderr solo quando c'è stato un errore;
  il codice di uscita va accanto al loro titolo, per esempio `stdout (codice 2):` (accanto a stdin quando non c'è output).
  La casella stdin ha un bordo bianco (grigio scuro con il tema chiaro), la casella stdout uno verde e la casella stderr
  uno rosso, con ogni strumento della shell (`bash.run`, `zsh.run`, `sh.run` e `psh.run`).
- Attiva **Impostazioni → Visualizzazione → Espandi automaticamente i comandi della shell locale** per aprire ogni chiamata da sola: ogni
  casella compare non appena il suo contenuto è noto, stdin mentre il modello scrive il comando e stdout e stderr quando
  il comando termina.
- Ogni chiamata a uno strumento della shell lascia un proprio file in `~/.mgs/local-shell-mcp-server/` (nella home dell'utente che
  esegue `mgs`; nel servizio distribuito, `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), con il nome formato dalla data
  e dall'ora locali della chiamata fino al millisecondo, per esempio `y2026m10d12h23m59s59ms731.jsonl`; `-2`, `-3`… vengono
  aggiunti quando più chiamate iniziano nello stesso millisecondo. È un file JSONL, un oggetto JSON per riga: per primo il comando (`"event": "command"`,
  con lo strumento, la shell, la cartella di lavoro e il timeout), scritto prima che venga eseguito; poi l'intero output,
  senza tagli, man mano che arriva, una riga per ogni frammento (`"event": "stdout"` o `"stderr"`, con il testo in `data`)
  nell'ordine in cui il comando lo ha scritto; e infine il risultato (`"event": "result"`: codice di uscita, segnale, se è scaduto il timeout
  o è stato annullato, i secondi impiegati e i byte di stdout e stderr) oppure il motivo per cui non è potuto partire
  (`"event": "error"`). Anche una chiamata rifiutata prima dell'esecuzione (nessun comando, una cartella di lavoro che non esiste, un
  timeout non valido) lascia il suo file, con il comando così come è stato richiesto e l'errore. La chat conserva solo
  l'inizio e la fine degli output molto lunghi, ma il file contiene tutto, e contiene già l'output se `mgs`
  si ferma mentre il comando è in esecuzione. Per leggere l'intero stdout di una chiamata, esegui `jq -j 'select(.event == "stdout") | .data'
  FILE`. JSON contiene solo testo, quindi i byte che non sono UTF-8 (output binario) vengono sostituiti con `�`; un comando che
  stampa gigabyte lascia un file di quelle dimensioni. Solo quell'utente può leggere la cartella e i file, perché l'output
  di un comando può contenere segreti; niente li elimina, quindi rimuovi quelli vecchi quando non ti servono più.

Nel servizio distribuito, aggiungi `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` alla sezione `[Service]` di un override
di systemd (`systemctl edit moe-gguf-server`) e riavvia `moe-gguf-server`; su Alpine, esporta la stessa variabile nella
configurazione del servizio OpenRC.

**Sicurezza.** I comandi vengono eseguiti con i permessi dell'utente che esegue `mgs` (nel servizio distribuito, l'account
`moe-gguf-server`, mai `root`). Chiunque possa aprire l'interfaccia o chiamare l'API può eseguirli, quindi attiva
l'opzione solo in ambienti fidati e proteggi il server con una chiave API. Le richieste provenienti da pagine di altri siti
vengono rifiutate, l'origine CORS predefinita diventa localhost e il log registra ogni comando con il suo codice di uscita. Leggi
il comando mostrato nella finestra della password prima di digitare una password di sudo: la password autorizza quel comando. Il
server la consegna solo a un programma di aiuto per la password il cui processo padre è sudo eseguito come root per il tuo utente, quindi un comando
non può avviare da solo il programma di aiuto per leggere la password.

<a id="api"></a>
## API HTTP

Gli esempi usano 11443; sostituiscila con la porta HTTPS di `ports.json` se è diversa.

La radice predefinita dell'API è `/api`. L'interfaccia resta in `/`. Le route radice `/v1/...`
del server originale non vengono più registrate. I client compatibili con OpenAI devono
usare `https://ai.example.com:11443/api/v1` come URL di base. I nomi dei campi di richiesta e di risposta
mantengono la compatibilità con upstream; le interfacce C/C++ del motore conservano le loro funzioni attuali,
ma non quelle che erano contrassegnate come deprecate (vedi [compatibilità rimossa](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Imposta `MGS_API_KEY` sulla chiave installata nella tua shell. Le operazioni native,
gli embedding, la tokenizzazione, il reranking, la gestione dei modelli e gli stream riprendibili sono
documentati nella Swagger UI inclusa in `/api/doc/`; il suo JSON OpenAPI si trova in
`/api/doc/openapi.json`, generato dal server a ogni richiesta senza un file JSON statico. L'inventario riflette le route registrate dal
processo in esecuzione, comprese le route del router e quelle facoltative di compatibilità. Le funzionalità
disattivate possono restituire 403. Una chiave mancante o non valida restituisce 401 e il caricamento del modello
restituisce 503. Le route di stato di salute e dell'elenco dei modelli mantengono l'accesso pubblico per `GET` e `HEAD`; vedi
[limiti di sicurezza](#security-limits) per le route che richiedono sempre la chiave.

`--api-prefix /api/custom` sposta le route di inferenza sotto `/api/custom`.
L'interfaccia radice riceve quel prefisso da `/api/config.js`; la documentazione resta in `/api/doc/`.
Lo spazio dei nomi della documentazione è riservato e non può essere usato come prefisso
di inferenza. I processi figli del router usano internamente `/api`. Il server convalida i prefissi;
i valori fuori da `/api/` e i percorsi con barre finali vengono rifiutati.

La pagina Monitoraggio usa `GET /api/monitor` (rapporto completo), `GET /api/monitor/memory` (istantanee della memoria dei modelli caricati)
e `GET /api/monitor/usage` (solo utilizzo di CPU e GPU, disponibile anche mentre il modello si carica). Leggono contatori e
istantanee pubblicati dal ciclo di inferenza, mai la coda dei task, quindi interrogarli periodicamente non risveglia un modello inattivo né
azzera il suo timer di inattività. I loro campi sono descritti in Swagger.

L'interfaccia legge le sue impostazioni con `GET /api/ui/settings` (`{"settings": DOCUMENT}`, oppure
`{"settings": null}` prima del primo salvataggio) e le salva con `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, che sostituisce per intero `~/.mgs/config/ui-settings.json`, con modo 0600). Entrambe richiedono
la chiave API e rispondono anche mentre il modello si carica.

<a id="request-queue"></a>
### Coda delle richieste e richieste parallele

Per impostazione predefinita `mgs` elabora una richiesta alla volta (`-np 1`) e le assegna l'intero contesto, così ogni richiesta può
usare tutti i token del contesto, sia che provenga dall'interfaccia web sia che arrivi direttamente dall'API. Le richieste che
arrivano nel frattempo attendono in un'unica coda e vengono elaborate nell'ordine di arrivo: le richieste dell'interfaccia non
scavalcano quelle dell'API, né viceversa. L'API si comporta allo stesso modo qualunque siano le impostazioni
dell'interfaccia.

Per elaborare più richieste contemporaneamente, avvia `mgs` con `-np N` (`--parallel N`, `MGS_ARG_N_PARALLEL`):
fino a N richieste vengono eseguite in parallelo e il contesto viene suddiviso tra di esse, quindi ciascuna ne riceve una parte (`n_ctx_slot` nella
riga `initializing` del log); con `--kv-unified` condividono invece un'unica cache KV dell'intero
contesto. Le richieste oltre N continuano ad attendere nella coda.

L'interfaccia web aggiunge `"mgs_client": "webui"` alle sue richieste. Mentre una delle sue richieste in streaming attende, lo
stream inizia con eventi come

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: richieste in esecuzione o in attesa prima di essa; `ahead_api` e `ahead_webui`: quante di esse sono arrivate direttamente
all'API o dall'interfaccia web), e un ultimo evento con tutti i conteggi a 0 quando inizia. I client che non
inviano `mgs_client` non ricevono mai questi eventi, quindi i client compatibili con OpenAI ricevono lo stream consueto. Swagger
descrive il campo insieme agli altri parametri di completamento.

<a id="security-limits"></a>
### Limiti di sicurezza

Questi limiti proteggono il server e l'interfaccia da richieste, file e modelli ostili. Sono abbastanza generosi
per un uso normale: ciò che li supera fallisce con un errore chiaro invece di fermare il server.

- **Chiave API.** Le route di stato di salute e dell'elenco dei modelli sono pubbliche solo per `GET` e `HEAD`. Ricaricare l'elenco dei modelli
  (`GET /api/models?reload=1`) e aggiungere o eliminare modelli in modalità router (`POST` e `DELETE /api/models`)
  richiede la chiave ogni volta che ne è configurata una.
- **Pagine di altri siti.** Quando gli strumenti del server o i server MCP sono attivati (`--tools`, `--agent`,
  `--mcp-servers-config`, `--mcp-servers-json` o `--local-shell-mcp-server`) e `--cors-origins` non è impostato,
  il server risponde 403 a qualsiasi richiesta diversa da `GET`, `HEAD` o `OPTIONS` la cui intestazione `Origin` indichi
  un altro sito. L'interfaccia (anche dietro Apache) e i client che non inviano `Origin`, come curl o gli SDK,
  non sono interessati. `POST /api/tools` accetta solo `Content-Type: application/json` (altrimenti 415).
- **Proxy CORS.** `/api/cors-proxy` rifiuta le richieste che un browser aprirebbe come pagina, frame o worker
  (403) e restituisce ogni risposta come contenuto inerte: non può eseguire script né impostare cookie sull'indirizzo del server.
- **Elenchi del router.** In modalità router, `status.args` e `status.preset` di `/api/models` mostrano `<redacted>`
  al posto dei valori di `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  e `--mcp-servers-json`; il log fa lo stesso. I processi dei modelli ricevono comunque i valori reali.
- **Client lenti.** Mentre il server riceve una richiesta, ogni lettura attende al massimo 30 s (`--timeout` continua a regolare
  le scritture, quindi le risposte lunghe non vengono troncate), e al massimo 256 connessioni accettate attendono un thread libero.
- **Corpi delle richieste.** I corpi JSON annidati oltre 1024 livelli di array o oggetti vengono rifiutati con 400. Gli schemi JSON
  (`response_format`, `json_schema`) possono annidarsi fino a 256 livelli; i valori `pattern` vuoti e i quantificatori senza
  nulla da ripetere vengono rifiutati.
- **Contenuti multimediali remoti.** Gli URL in `image_url`, `input_audio` e `input_video` possono puntare solo a indirizzi
  pubblici di Internet. Il loopback, le reti private (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10), gli indirizzi
  link-local (compreso l'indirizzo dei metadati cloud 169.254.169.254), gli indirizzi IPv6 unique local e gli altri
  indirizzi non pubblici vengono rifiutati con 400, anche quando vi conduce un reindirizzamento. Per consentirli, per esempio per
  leggere immagini da un server della tua rete, avvia `mgs` con `--media-allow-private-urls`
  (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); fallo solo quando tutti i client dell'API sono fidati. Ogni download ha 10 s per
  connettersi, 30 s in totale, fino a 5 reindirizzamenti e 10 MiB, e il server esegue al massimo 8 download alla volta.
- **Contenuti multimediali decodificati.** Immagini fino a 32768 pixel per lato e 100 megapixel, audio fino a un'ora, fotogrammi video fino
  a 8192 pixel per lato e 2 GiB di fotogrammi decodificati per video, e al massimo 2 GiB di immagini e audio decodificati per
  richiesta.
- **Modelli di chat del modello.** `range()` produce al massimo un milione di elementi, le stringhe costruite restano sotto
  i 256 MiB, l'indentazione di `indent()` e `tojson()` sotto i 256 spazi e l'annidamento sotto le 1024 istruzioni; un modello
  che li supera fallisce con un errore di modello.
- **RPC.** In una compilazione con il backend RPC, il server RPC verifica le dimensioni dei frame e i limiti dei tensori inviati dai suoi
  peer e chiude solo la connessione responsabile. Esponilo solo a macchine fidate: continua a servire un client
  alla volta.
- **Interfaccia.** I link nei messaggi conservano solo gli schemi http, https, mailto e tel (o indirizzi relativi), e
  gli allegati di immagine devono essere immagini incorporate. L'importazione delle conversazioni accetta file fino a 512 MB, archivi ZIP con
  fino a 10.000 voci, 256 MB per file JSONL e 1 GB estratto in totale. Un file con record malformati, o con un
  messaggio che appartiene a un'altra conversazione, viene rifiutato per intero; una conversazione i cui messaggi esistono già
  viene saltata e segnalata.

I comandi di installazione scaricano comunque il ramo `main` attuale tramite HTTPS; non verificano una firma né
una revisione fissa.

<a id="performance"></a>
## Memoria e prestazioni

La registrazione della memoria host e il prefetch degli esperti sono attivati per impostazione predefinita. Il prefetch
sovrappone i trasferimenti degli esperti al calcolo, in particolare con prompt di grandi dimensioni.
I risultati dipendono dal modello, dalla capacità di memoria e dalla larghezza di banda della RAM e di PCIe.

Per i modelli con esperti scaricati sulla CPU, `--cpu-moe` o `--n-cpu-moe N` possono ridurre
l'uso della VRAM. La registrazione host blocca in memoria le pagine dei pesi mappati per copie
più rapide dall'host al dispositivo. La memoria bloccata non può essere spostata nello swap, quindi lascia abbastanza RAM per il sistema
operativo. Questo percorso di registrazione è specifico di POSIX; Windows mantiene il suo
comportamento di caricamento precedente. `--load-mode none` è consigliato solo quando la registrazione host fallisce
del tutto o in parte, non come requisito normale.

Il prefetch degli esperti usa per impostazione predefinita tre slot di VRAM. Ogni slot deve poter contenere il più grande
tensore di esperti trasferito. I valori 2–8 in `GGML_SCHED_PREFETCH_EXPERTS` scelgono
esplicitamente il numero di slot. Se l'allocazione della memoria fallisce, vengono mantenuti almeno due slot allocati
quando possibile; altrimenti si usano i trasferimenti normali. Il prefetch inizia solo
quando il batch ha almeno il doppio di selezioni di esperti rispetto al numero di esperti. Il suo pool
appartiene alla prima GPU che lo attiva; le altre GPU mantengono copie selettive.
Batch più grandi possono aiutare il prefill ma richiedono più memoria.

Per la diagnostica, imposta `GGML_CUDA_REGISTER_HOST=0` o
`GGML_SCHED_PREFETCH_EXPERTS=0` per disattivare l'ottimizzazione corrispondente.
Non impostare quelle variabili durante il funzionamento normale, a meno che tu non voglia rinunciare deliberatamente all'ottimizzazione.

L'MTP automatico esamina i metadati GGUF e i nomi dei tensori prima dell'allocazione dei pesi,
compresi tutti gli shard. Richiede un'architettura supportata, un tronco del modello e
tutte le teste integrate. Una testa esterna o un'architettura non supportata non
lo attivano. Il limite predefinito di bozza è di tre token. La selezione esplicita di `--spec-type`
ha la precedenza; `--no-spec-mtp-auto` e `--spec-type none` disattivano
la selezione automatica. `--spec-draft-n-max` cambia il limite. MTP influisce sulla generazione,
non direttamente sul prefill né sul tempo al primo token. Il miglioramento dipende dal tasso di
accettazione e dall'hardware; aumentare il limite non garantisce più velocità.

<a id="mtp-vram"></a>
I pesi MTP restano sempre nella memoria della GPU quando la GPU può contenerli: i layer MTP innestati nel modello, un
GGUF MTP separato passato con `--spec-draft-model` e la testa di output che usano. Le opzioni che spostano i pesi nella
memoria di sistema (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, i loro equivalenti `--spec-draft-*` e l'adattamento
automatico) non si applicano a essi, e restano sulla GPU anche con un `--gpu-layers` basso. L'adattamento automatico colloca il
resto del modello attorno a essi, e la [cache degli esperti MoE](#expert-cache) usa la memoria che rimane dopo. Se i layer
MTP non stanno nella memoria libera della GPU con un margine di 512 MiB, il log mostra un avviso e vengono collocati come qualsiasi
altro layer. Con un GGUF MTP separato, i layer MTP innestati nel modello principale non vengono caricati, così non occupano
memoria due volte; con `--spec-type none`, non viene caricato nessun peso MTP. Il log di caricamento conferma la collocazione con una riga
come `keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)`, visibile con `-lv 4`.

<a id="expert-cache"></a>
### Cache degli esperti MoE

Quando gli esperti MoE restano nella memoria di sistema e il resto del layer viene eseguito su una GPU, il server mantiene gli esperti
usati più di recente di tutti i layer in una cache che occupa la VRAM libera;
con più GPU, ogni GPU mette in cache i layer che le sono assegnati nella propria VRAM. Per ogni token generato (e per qualsiasi
batch inferiore a 32 token), gli esperti trovati nella cache vengono eseguiti sulla GPU, una parte di quelli mancanti viene caricata
nella cache e il resto viene eseguito contemporaneamente sulla CPU; la parte segue le larghezze di banda misurate di PCIe e
della CPU. I prompt lunghi mantengono il prefetch degli esperti, che ora copia gli esperti in cache all'interno della GPU e carica solo
gli altri. L'output corrisponde a quello dell'esecuzione senza cache a meno degli arrotondamenti. La GPU e la CPU arrotondano in modo diverso e gli esperti
che ciascuna calcola dipendono dal contenuto della cache, quindi una lunga generazione greedy può divergere verso un testo diverso ma ugualmente
valido dopo alcuni token, da un'esecuzione all'altra; la cache del llama.cpp ufficiale si comporta allo stesso modo.

La cache è attivata per impostazione predefinita e viene creata dalla prima richiesta, una volta caricati il modello, il contesto MTP e il
proiettore multimodale. Senza `--gpu-layers`, `--n-cpu-moe` o `--override-tensor`, l'adattamento
automatico mantiene tutti gli esperti nella memoria di sistema quando ci stanno con un margine (4 GiB o il 10% della RAM) e lascia la VRAM libera alla cache (con più GPU, distribuisce i layer
tra di esse in base alla loro memoria libera, così ogni GPU ha una cache); altrimenti colloca interi layer di esperti sulla GPU come prima e la cache usa ciò che
rimane. Le opzioni esplicite di collocazione vengono rispettate: `--cpu-moe` dà alla cache la massima VRAM e
`--n-cpu-moe N` mantiene alcuni layer statici sulla GPU. Gli esperti mantenuti nella RAM sono bloccati in memoria, quindi la RAM deve contenerli
tutti più il sistema operativo.

<a id="context-size"></a>
Quando il modello non sta nella VRAM e `-c` non è indicato, l'adattamento automatico sceglie anche la dimensione del contesto: la
cache KV occupa al massimo un quarto della VRAM rimasta dopo i pesi densi, fino al contesto di addestramento del
modello, e gli esperti (la cache o interi layer di esperti) si tengono il resto. In precedenza questi modelli ricevevano sempre il
minimo di 4096 token condiviso da tutti gli slot, il che troncava le risposte lunghe (`truncated = 1` nel log). Il
log mostra il risultato in una riga come `context size raised from 4096 to N tokens` e in `n_ctx_slot` della
riga `initializing`. `-c N` imposta il contesto esplicitamente (più contesto lascia meno VRAM alla cache e riduce la
velocità di generazione: 35,8 token/s con 4096 token e 33,9 con 122624 sulla macchina di riferimento) e
`--fit-ctx N` alza il minimo. Il contesto MTP assume la stessa dimensione del contesto del modello invece di riservare
VRAM per l'intero contesto di addestramento.

Prima del contesto, l'adattamento automatico sceglie il micro-batch (`-ub`), il numero di token del prompt elaborati in
un singolo passaggio. Con gli esperti nella RAM, ogni passaggio carica sulla GPU gli esperti che non sono nella cache, quindi passaggi
meno numerosi e più grandi elaborano i prompt lunghi molto più velocemente, mentre la generazione non cambia. A meno che non sia indicato `-b` o `-ub`,
sceglie il più grande tra 4096, 2048 e 1024 token i cui buffer aggiuntivi stanno in un ottavo della VRAM rimasta dopo i
pesi densi, e aumenta di conseguenza il batch (`-b`). Sulla macchina di riferimento un prompt di 2514 token è passato da 214 a
755 token/s con 4096, alla stessa velocità di generazione; i buffer più grandi accorciano il contesto predefinito (circa 90.000
token senza MTP e 61.000 con MTP su 16 GB). Il log mostra la scelta in una riga come `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

| Opzione | Variabile | Effetto |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE` (`0` disattiva) | Attiva o disattiva la cache (attivata per impostazione predefinita). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | Dimensione totale fissa della cache, suddivisa tra le GPU in base agli esperti che ciascuna mette in cache; `-1` (predefinito) usa la VRAM libera di ogni GPU. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | VRAM lasciata libera su ogni GPU oltre alla cache e agli slot di prefetch (predefinito 1024). Aumentala quando altri programmi usano la GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnostica: quota fissa (0–1) di esperti mancanti caricati al posto del bilanciamento misurato. |
| — | `MGS_MOE_CACHE_STATS` | Diagnostica: registra nel log i contatori della cache ogni N passi di decodifica (visibile con `-lv 4`). |

`GET /api/expert-cache` restituisce la dimensione, i layer e gli slot in cache, gli hit e i miss, i caricamenti, gli esperti sulla CPU, la
quota di byte di prefetch serviti dalla cache e le larghezze di banda misurate (con più GPU,
dimensioni e contatori si sommano e la larghezza di banda di caricamento è la loro media). `POST /api/expert-cache` con
`{"size_mib": N}` cambia la dimensione senza riavviare (`-1` automatica, `0` libera la cache); la richiesta successiva
la applica e parte con una cache vuota:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Con `--metrics`, `/api/metrics` aggiunge contatori e indicatori `mgs:moe_cache_*` (passi, hit, miss, caricamenti,
esperti sulla CPU, byte di prefetch, dimensione, slot e larghezze di banda).

Misurazione di riferimento su una RTX 4060 Ti da 16 GB con Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 thread della CPU), confrontata con
`--n-cpu-moe 28` senza cache: generazione 26,9 → 35,7 token/s (+33%), elaborazione del prompt 767–778 →
718–730 token/s (−6%), e con MTP attivato 33,6 → 47,0 token/s (+40%). Sulla stessa macchina, il llama.cpp
ufficiale con la propria cache (`-cmoe --moe-cache-mib 10240`) ha raggiunto 26,7 token/s, e 27,8 con MTP. I guadagni dipendono dalla quota di esperti
che sta nella VRAM, dalla località dell'instradamento del modello e dalle larghezze di banda di PCIe e della RAM.

<a id="threads"></a>
### Thread della CPU

La CPU calcola gli esperti che non sono nella cache. Quel lavoro è limitato dalla larghezza di banda della memoria e
si satura con pochi thread, mentre usare tutte le CPU logiche fa competere i thread con quello che pilota
la GPU. Senza `-t`, `mgs` usa i core utili per il calcolo, ma mai più della metà delle CPU logiche su cui può
essere eseguito. Su una macchina con SMT (due thread per core) è il numero di core fisici, come prima; in una
macchina virtuale che non mostra SMT è la metà delle CPU virtuali. Sulla macchina di riferimento (16 core
virtuali con una sessione desktop, Qwen3.6-35B-A3B Q8_0, senza MTP), da 4 a 12 thread hanno generato alla stessa velocità
(32,9–33,9 token/s) e 16 thread a 28,7 token/s. Il log di avvio mostra la scelta in una riga come
`using 8 CPU threads for generation and 8 for batches`; `-t` e `-tb` li impostano esplicitamente.

<a id="checkpoints"></a>
### Checkpoint del contesto

I modelli con layer ricorrenti o a finestra scorrevole (per esempio gli ibridi Qwen3.5/3.6/3.8, Nemotron-H o Granite 4)
non possono riportare indietro il loro stato, quindi quando una richiesta modifica una parte precedente della conversazione il
server riprende dal checkpoint più recente anteriore alla modifica. I checkpoint vengono collocati all'inizio dei messaggi dell'utente,
dell'assistente e degli strumenti (dove gli agenti di programmazione rimuovono il ragionamento vecchio o accorciano i risultati vecchi degli strumenti) e almeno
ogni `--checkpoint-min-step` token (predefinito 1024, in precedenza 8192) senza passaggi aggiuntivi sul prompt. Quando
si raggiunge `--ctx-checkpoints` (32), viene rimosso il checkpoint più vicino ai suoi vicini, così gli altri restano
distribuiti lungo la conversazione. Nello scenario di agente di riferimento (14k token, otto risultati di strumenti), accorciare un risultato
di strumento recente è passato da 19,3 s a 5,5 s e uno iniziale da 18,0 s a 13,6 s. Ogni checkpoint conserva lo
stato ricorrente o della finestra scorrevole nella memoria host (63 MiB per Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Opzioni aggiornate e diagnostica

Mantieni il valore predefinito `--load-mode mmap` quando la registrazione host riesce. Usa `--load-mode none` solo come ripiego quando la registrazione fallisce. `--load-mode dio` ora prepara le letture usando al massimo 64 MiB di memoria aggiuntiva; non è un limite alla RAM totale del modello.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` è stata rimossa perché non aveva alcun effetto in esecuzione. I vecchi flag di caricamento (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) e il flag di compilazione `GGML_CUDA_FA_ALL_QUANTS` non vengono più accettati; vedi [compatibilità rimossa](#removed-compatibility) per i loro sostituti. Per una compilazione CMake manuale, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` aggiunge quella combinazione K/V di kernel vettoriali; sono accettati elenchi separati da virgole o elenchi tra virgolette separati da punto e virgola. L'elenco predefinito è `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 è sempre incluso. `all` aumenta il lavoro di compilazione. Le combinazioni di kernel vettoriali mancanti usano un ripiego di conversione F16 e possono essere più lente.

I GGUF completi a testa singola di GLM-4.5-Air (`glm4moe`) ora possono attivare l'MTP automatico. Gli altri controlli di idoneità esistenti e la precedenza delle opzioni esplicite continuano ad applicarsi. MTP separa i grafi con e senza output, preserva l'ordine originale dei token tra sequenze concorrenti e usa le posizioni reali dopo le immagini.

L'adattamento automatico conta i layer nextn (MTP), quindi il primo layer non resta più sulla CPU quando MTP è disattivato. I pesi MTP ora restano sempre nella memoria della GPU; vedi [collocazione di MTP](#mtp-vram). Il valore predefinito di `--checkpoint-min-step` ora è 1024 invece di 8192; vedi [checkpoint del contesto](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` e `--moe-cache-reserve` controllano la [cache degli esperti MoE](#expert-cache). L'adattamento automatico non lascia più i modelli più grandi della VRAM con un contesto di 4096 token e ora sceglie anche il micro-batch; vedi [dimensione del contesto](#context-size). Il numero predefinito di thread della CPU ora è al massimo la metà delle CPU logiche; vedi [Thread della CPU](#threads). `-np` ora usa per impostazione predefinita un solo slot con l'intero contesto, invece della modalità automatica con quattro slot che condividono una cache KV unificata, e `--kv-unified` è disattivato per impostazione predefinita; vedi [coda delle richieste](#request-queue).

Con `--metrics`, `GET /api/metrics` autenticato include `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` e `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Gli indici di posizione partono da zero; quella serie compare dopo il completamento di una richiesta speculativa. Confronta le differenze dei contatori dei token accettati e proposti quando valuti l'accettazione, e misura la latenza separatamente; l'accettazione da sola non dimostra un'accelerazione. In modalità router, seleziona il modello con `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, disattivato per impostazione predefinita) consente agli URL dei contenuti multimediali delle richieste di raggiungere il loopback, gli indirizzi privati e gli altri indirizzi non pubblici; vedi [limiti di sicurezza](#security-limits).

Le esportazioni delle conversazioni leggono l'albero completo dei messaggi e i metadati salvati. Se gli strumenti del server restituiscono 403, i nuovi messaggi smettono di ripetere quella ricerca fallita; riaprire il pannello degli strumenti la riprova. I valori predefiniti del server alla prima visita preservano le impostazioni già modificate dall'utente, compresa la chiave API.

[Implementazione e sorgenti upstream](CODE.it-IT.md#upstream-review).

<a id="removed-compatibility"></a>
## Compatibilità rimossa

Gli adattatori di retrocompatibilità sono stati rimossi. I vecchi nomi non vengono più
tradotti né segnalati con un avviso: il server rifiuta le opzioni della riga di comando rimosse come
argomenti non validi, ignora le variabili d'ambiente rimosse e risponde 404 per le
route rimosse. Aggiorna script, override dei servizi, preset e client ai
sostituti indicati di seguito.

| Rimosso | Usa invece |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (o `mmap+mlock` per combinarli entrambi) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Niente; non aveva alcun effetto. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Vecchi alias del modello bozza come `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` o `--draft-p-min` | L'opzione `--spec-draft-*` corrispondente, per esempio `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` o `--spec-draft-p-min`. `mgs --help` le elenca tutte. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` o `--spec-ngram-mod-n-max` / `--spec-draft-n-min` o `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | L'opzione `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` e `--spec-ngram-*-min-hits` della modalità n-gram scelta |
| Variabili d'ambiente, opzioni CMake e simboli dell'API C con il precedente prefisso di upstream, compresi gli alias che le versioni precedenti avevano già eliminato | Il nome attuale con il prefisso `MGS_` (`mgs_` nell'API C), per esempio `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` invece di una variante `NO_`, `MGS_ARG_LOAD_MODE` invece delle variabili di mappatura in memoria, di blocco e di Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` e `MGS_ARG_SPEC_DRAFT_N_MIN` per i limiti di bozza, `MGS_CACHE` o `MGS_BUILD_UI`; per le opzioni del backend, l'opzione `GGML_*` corrispondente, per esempio `GGML_CUDA` o `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` o `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` o `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valore `deepseek-legacy` di `reasoning_format` (campo della richiesta e `--reasoning-format`) | `deepseek` o `auto`; il ragionamento viene restituito in `reasoning_content` |
| Campo della richiesta `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (array di stringhe) |
| `reasoning_in_content` in `generation_settings` | Niente; il campo è stato rimosso. |
| Download di un'interfaccia precompilata di upstream da un bucket di Hugging Face (le opzioni CMake `USE_PREBUILT_UI` e `UI_HF_BUCKET` e la variabile d'ambiente `HF_UI_VERSION`) | Niente; l'interfaccia viene sempre compilata da `frontend/` (`MGS_BUILD_UI=ON`) oppure omessa. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Numero di compilazione e commit: le variabili CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (anche in `mgs-config.cmake`) e la stringa `b0-unknown` di `build_info`, `system_fingerprint` e dello `User-Agent` | `MGS_VERSION`, impostata in `backend/cmake/build-info.cmake`; quei campi ora riportano la versione, come `0.1.0` |
| Modelli Meta Llama (le architetture GGUF `llama`, `llama4` e `llama-embed` e il proiettore di visione di Llama 4) e i modelli di chat integrati `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` e `llama4` | Niente; usa uno dei modelli MoE supportati. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Opzione del programma di installazione `--no-sudo` | Niente; i programmi di installazione vengono sempre eseguiti come root senza sudo. |
| Funzioni e campi che le API C avevano contrassegnato come deprecati: in `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, gli alias `mgs_token_*` (per esempio `mgs_token_bos`, `mgs_token_get_text` o `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, il campo `defrag_thold` di `mgs_context_params` e `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; in `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` e `ggml_upscale_ext`; in `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` e il campo `image_marker` | La funzione attuale: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` invece di `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` e `media_marker`. Le esecuzioni di riscaldamento sono a carico del chiamante, e `defrag_thold` non aveva alcun effetto. |
| Gemma 4: le architetture `gemma4` e `gemma4-assistant`, i proiettori `gemma4v`, `gemma4uv`, `gemma4a` e `gemma4ua`, il formato di chat `peg-gemma4`, il tokenizzatore e il pre-tokenizzatore `gemma4`, e i preset `--vision-gemma-4b-default` e `--vision-gemma-12b-default` | Niente; usa una delle [famiglie supportate](#supported-models). |
| Modelli DeepSeek precedenti a V4.1: le architetture `deepseek`, `deepseek2` (anche i file Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 e Youtu convertiti con essa), `deepseek2-ocr` (con i proiettori `deepseekocr` e `deepseekocr2`), `deepseek32`, `mistral4` e `deepseek4`, i pre-tokenizzatori `deepseek-llm` e `deepseek-coder` e la backbone di bozza DSpark di DeepSeek-V4 | DeepSeek-V4.1-Flash (`deepseek41`). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) e i modelli integrati del motore non Jinja (`--chat-template` con un nome come `deepseek3`, `chatglm4` o `mistral-v7-tekken`) | Niente; i modelli di chat passano sempre per Jinja. `--chat-template chatml` seleziona ancora ChatML; per un altro modello passa il suo sorgente Jinja con `--chat-template` o `--chat-template-file`. Le richieste con `tools` non hanno più bisogno di alcun flag. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| Secondi nomi lunghi di un'opzione: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk` (le chiavi dei preset del router seguono gli stessi nomi) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, con il nome delle rispettive opzioni |
| Campo della richiesta `thinking_budget_tokens` | `reasoning_budget_tokens` (i client Anthropic continuano a inviare `thinking.budget_tokens`) |
| `content` in `POST /api/embeddings` e `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, la struct `mgs_chat_message` e `MGS_FILE_MAGIC_GGLA` in `mgs.h`; `mtmd_image_tokens_get_id` in `mtmd.h` | Niente; la formattazione della chat è affidata ai modelli Jinja di `common/chat.h`, e gli altri non avevano alcun uso. |

CMake segnala le variabili `-D` rimosse come inutilizzate invece di applicarle, quindi controlla
l'output della configurazione delle compilazioni manuali.

L'interfaccia web conserva le sue impostazioni sul server (`~/.mgs/config/ui-settings.json`),
e nel browser solo la sua chiave API (`MgsUi.apiKey` in `localStorage`) e le sue
conversazioni (il database IndexedDB `MgsUi`). Non legge né migra le
impostazioni che le versioni precedenti conservavano in `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, i modelli preferiti, gli elenchi degli strumenti, lo sforzo di ragionamento,
i suggerimenti MCP ignorati e la lingua), quindi le impostazioni ripartono dai
loro valori predefiniti e la chiave API va inserita di nuovo una volta. Non legge né
migra nemmeno i dati salvati dalle versioni precedenti con altri nomi, la chiave autonoma `theme`,
i marcatori di ragionamento e di chiamata agli strumenti incorporati nei vecchi messaggi, gli
allegati di testo incollato del vecchio tipo `context`, il vecchio formato a elenco dei suggerimenti
MCP ignorati né il precedente campo `webui_settings` del server. Per conservare le
conversazioni di una versione precedente, esportale da essa prima dell'aggiornamento e importa
l'esportazione dopo. L'importazione delle conversazioni accetta solo le esportazioni JSONL e ZIP, entro i limiti elencati
in [limiti di sicurezza](#security-limits); il vecchio formato di esportazione JSON viene rifiutato. Le vecchie impostazioni restano nel browser finché non vengono eliminate: per
rimuoverle, apri la console del browser (F12) sulla pagina dell'interfaccia ed esegui
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
una volta per ogni indirizzo e porta con cui è stata aperta l'interfaccia.

I file scaricati con `-mu`/`--model-url` vengono messi in cache in `~/.cache/mgs/`, quindi
quelli che le versioni precedenti salvavano nella loro vecchia cartella di cache vengono scaricati di nuovo.
I download da Hugging Face continuano a usare la cache di Hugging Face (`~/.cache/huggingface/hub`,
o la cartella indicata in `MGS_CACHE`, `HF_HUB_CACHE` o `HF_HOME`).

### Vecchi file di modello

È stato rimosso anche il codice di compatibilità per le seguenti vecchie conversioni.
Riconverti il modello originale, oppure scarica un GGUF attuale, per usarli.

| Vecchio file | Comportamento attuale |
| --- | --- |
| Grok-1 convertito prima di agosto 2025 (senza `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` o `grok.attn_logit_softcapping`) | Non si carica: chiave mancante. |
| Kimi-Linear con il tensore `blk.N.attn_kv_b` non suddiviso | Non si carica: mancano `attn_k_b`/`attn_v_b`. Quel percorso falliva già durante la costruzione del grafo. |
| File Qwen3-Next con solo il tensore unito `ssm_in` | Non si carica: manca il tensore `blk.0.attn_qkv.weight`. |
| File senza `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) o `full_attention_interval` (Qwen3-Next), e file Cohere2-MoE con un unico periodo SWA invece dell'array per layer | Non si carica: chiave mancante. |
| Vocabolari BPE senza `tokenizer.ggml.pre` o con i nomi di pre-tokenizzatore `llama3`, `llama-v3`, `jina-es`, `jina-de` o `llada-moe`, e tokenizzatori Kimi-K2 senza merge | Non si carica: chiave mancante, tipo di pre-tokenizzatore sconosciuto o merge mancanti. |
| File GGUF di versione 1 o 2 | Non si carica: è accettata solo la versione 3 di GGUF. |
| File mmproj dei vecchi script occasionali: tipi di proiettore `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (da MiniCPM-V 2.x a 4.5, MiniCPM-o 2.6) e `adapter` (GLM-Edge-V), e file senza tipo di proiettore | Il proiettore non si carica: tipo di proiettore sconosciuto o mancante. |
| File mmproj di vecchie conversioni con `ffn_up` e `ffn_down` scambiati | Non viene più corretto; il proiettore fallisce. |
| mmproj di Kimi-K2.5 senza `image_min_pixels`/`image_max_pixels` e Step3-VL senza `preproc_image_size` | Il proiettore non si carica: chiave o tensore mancante. |
| File che riportano solo le vecchie chiavi `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` e `tokenizer.ggml.middle_token_id` invece di quelle `tokenizer.ggml.fim_*_token_id` | Si carica; i token di infill vengono riconosciuti dal loro testo (per esempio `<PRE>` o `<\|fim_prefix\|>`), e l'infill non funziona se non vengono trovati. |

Sono stati mantenuti i ripieghi da cui dipendono ancora le conversioni attuali o i GGUF pubblicati più scaricati delle
famiglie supportate, verificati sulle loro intestazioni su
Hugging Face: il valore predefinito di `add_bos` dei pre-tokenizzatori `llama-bpe`, Tekken e
simili; la disposizione dei layer ricorrenti di Qwen3.6 derivata da
`full_attention_interval` (i GGUF di Qwen3.6 più scaricati non hanno
`recurrent_layers`); e la disposizione integrata dell'indicizzatore di GLM-5.2 per i file GLM-5.2/5.3
senza `attention.indexer.types`.

<a id="maintenance"></a>
## Manutenzione e risoluzione dei problemi

- Controlla quale versione è installata con `mgs --version` (stampa, per esempio, `version: 0.1.0`); la compilazione locale
  lo esegue alla fine e CMake stampa `mgs version: 0.1.0` durante la configurazione. L'interfaccia mostra la stessa versione
  in Impostazioni → Visualizzazione → **Mostra informazioni sulla versione della build** (angolo in basso a destra) e sotto **Informazioni sulla compilazione** nella
  finestra di dialogo delle informazioni sul modello. Ogni modifica al progetto aumenta l'ultimo numero: 0.1.0, 0.1.1, 0.1.2, ecc.
- Ripeti `curl | bash` per aggiornare da `main`, oppure il comando di installazione locale appropriato per ricompilare e sostituire il binario.
  Riavvia poi il servizio: `systemctl restart moe-gguf-server` su Debian,
  oppure `rc-service moe-gguf-server restart` su Alpine.
- Leggi `/root/webapp-install.log`, `journalctl -u moe-gguf-server` su Debian,
  oppure `/var/log/moe-gguf-server.log` su Alpine. I log di Apache si trovano sotto
  `/var/www/DOMAIN-logs/`.
- Se la compilazione CUDA non è disponibile, installa il Toolkit separatamente oppure usa
  `--cpu`. Se NCCL non viene trovato, verifica la disponibilità del repository e la versione di CUDA,
  oppure scegli esplicitamente `--no-nccl`.
- Se la compilazione dell'interfaccia fallisce, controlla le versioni di Node.js e npm, i download delle dipendenze
  e il log. Non sostituirla con un'interfaccia precompilata di upstream.
- Per le compilazioni di distribuzione, usa una cartella nuova all'interno di `_/temp/` dopo modifiche alla struttura dei sorgenti.
  Una copia del progetto su SSHFS può vietare gli eseguibili; in produzione usa un disco locale.
- Per la compilazione locale, riesegui il comando su una riga indicato sopra per scaricare il `main` attuale, oppure
  `bash build/build-local.sh` per usare i tuoi sorgenti locali, e riavvia la copia che
  hai avviato a mano. Leggi l'output del terminale se fallisce o segnala librerie
  fuori da `lib/`. Il file temporaneo `build.log` viene eliminato insieme alla cartella di compilazione.
- Gli errori API 404 nei client più vecchi di solito richiedono di cambiare l'URL di base in
  `/api/v1`. La chiave API installata va nell'intestazione Authorization.

In base alle regole di questo progetto, l'assistente di programmazione non esegue test, esecuzioni di distribuzione, benchmark della GPU né
verifiche automatiche su browser mobili. I controlli statici della sintassi
non sono una garanzia del comportamento in esecuzione. Vedi [CODE.md](CODE.it-IT.md) per la
mappa dell'implementazione e i punti di estensione.

<a id="license"></a>
## Licenza

`mgs` è distribuito con la [PolyForm NonCommercial License 1.0.0](../LICENSE). Puoi usarlo,
modificarlo e condividerlo per qualsiasi scopo non commerciale: uso personale, studio, ricerca, progetti amatoriali
e uso da parte di organizzazioni di beneficenza, istituti di istruzione, enti pubblici di ricerca e
istituzioni governative. L'uso commerciale, come quello di un'azienda che lo usa per la propria attività, richiede una
licenza separata da parte di nipegun.

La licenza copre solo i contributi di nipegun. Le parti che provengono da llama.cpp (MIT, The
ggml authors), FreeToken (Apache 2.0) e dalle librerie incluse mantengono le proprie licenze; i loro
avvisi si trovano in `build/licenses/` e la compilazione locale li installa in
`~/.local/share/doc/mgs/licenses/`.
