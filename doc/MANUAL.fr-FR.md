# Manuel d'utilisation

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Sommaire : [installation](#installation), [compilation locale](#local-build), [déploiement](#deployment),
[chat](#chat), [supervision](#monitor), [modèles et outils](#models), [choix des modèles](#settings-models), [page Modèle](#model-page), [modèles pris en charge](#supported-models), [API](#api),
[file d'attente des requêtes](#request-queue), [limites de sécurité](#security-limits), [Vulkan](#vulkan), [ROCm](#rocm), [performances](#performance), [compatibilité supprimée](#removed-compatibility),
[maintenance](#maintenance), [licence](#license).

<a id="installation"></a>
## Installation et premier lancement

Installez-le en tant que root sur l'hôte de production. Sous Debian/Ubuntu, l'installateur peut être
passé directement sur l'entrée standard :

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

L'installateur déploie en une seule étape : il compile `mgs`, l'installe dans
`/usr/local/bin/mgs` et configure les services d'inférence et web
([déploiement](#deployment)). Aucun fichier voisin ne doit être téléchargé au préalable. Le script d'entrée télécharge l'archive
complète des sources de `main` depuis [ce dépôt](https://github.com/nipegun/moe-gguf-server/), la
vérifie et synchronise une copie gérée dans `/opt/moe-gguf-server-source`. Les téléchargements
et les caches de compilation restent sous le `_/temp/` du projet. Curl et les certificats d'autorité (CA)
sont nécessaires pour récupérer le script d'entrée ; les dépendances d'amorçage manquantes sont ensuite
installées automatiquement.

Passez des options de compilation et web avec :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com --cpu --jobs 4
```

La copie gérée se trouve toujours dans `/opt/moe-gguf-server-source`, propriété de root avec le mode
700. L'installateur ne l'adopte que lorsqu'elle est vide ou ne la réutilise que lorsqu'elle porte son
marqueur d'origine des sources ; un contenu sans rapport est refusé. Un verrou empêche les installations simultanées dans cette copie. Les exécutions
suivantes de `curl | bash` synchronisent les fichiers sources, y compris les fichiers supprimés de `main`,
tout en préservant `_/` et les caches de compilation. Les clés API, les modèles et les certificats
se trouvent hors de cette arborescence des sources et sont conservés. Un téléchargement, une
compilation ou une vérification des options qui échoue n'installe pas de nouvel exécutable ; relancez la même commande pour réessayer.
Aucune opération Git n'est nécessaire.

Un script téléchargé isolément utilise le même mode distant. Exécuter
`bash deploy/install-update-reinstall-debian.sh` depuis une copie locale complète compile cette copie
sans télécharger les sources. Les arguments placés après `bash -s --` sont transmis à l'installateur de
compilation. `--help` ne requiert ni root ni téléchargement des sources. `--no-system-deps` désactive
aussi l'installation des paquets d'amorçage : curl, les certificats CA, tar, gzip, rsync, util-linux et
coreutils doivent déjà être présents. Les sous-processus de l'installateur ne peuvent pas consommer le
script depuis le tube ; toutes les phases partagent un même flux de sortie `/root/webapp-install.log`.

Les options web sont vérifiées avant la compilation, de sorte qu'une erreur n'apparaît pas après une
longue compilation : `--domain NAME` (par défaut la sortie de `hostname -f`, ou celle de `hostname` ou de
`uname -n` lorsque ce n'est pas un nom valide), `--certificate PATH` et `--private-key PATH` (toujours
ensemble) et `--proxy-ip IP` (voir [déploiement](#deployment)). `/opt/moe-gguf-server-source/deploy/configure-web.sh`
reste disponible pour modifier plus tard la configuration web sans recompiler.

Alpine utilise toujours une copie locale complète et
`bash deploy/install-update-reinstall-alpine.sh`, avec Bash installé au préalable ; il ne compile que
pour le CPU. Sous Debian, choisissez le backend :

- CUDA (`--cuda`, par défaut) pour les GPU NVIDIA. Le CUDA Toolkit et le pilote NVIDIA doivent déjà
  être installés ; l'installateur ajoute les paquets NCCL correspondant à cette version de CUDA (voir plus
  bas). `--no-nccl` désactive explicitement NCCL.
- ROCm (`--rocm`) pour les GPU AMD pris en charge par ROCm ; `--rocm-targets LIST` choisit les cibles
  GPU. ROCm doit déjà être installé ([ROCm](#rocm)).
- Vulkan (`--vulkan`) pour les GPU AMD, Intel ou NVIDIA ([Vulkan](#vulkan)).
- CPU uniquement (`--cpu`), pour les machines sans GPU.

`--no-system-deps` laisse la gestion des paquets à l'administrateur.

L'interface locale nécessite Node.js 20.19+ dans la branche 20.x, ou 22.12+. `--no-ui` l'omet. Les autres
options sont `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST` et `--no-openssl`. `CUDACXX` choisit le compilateur CUDA et `HIPCXX` le
compilateur clang de ROCm. Les répertoires de compilation et de préparation doivent être des descendants
distincts, sans chevauchement, du répertoire `_/temp/` du projet.

L'installateur ajoute les paquets NCCL correspondants et le dépôt de NVIDIA si nécessaire, mais n'installe
ni ne met jamais à niveau le CUDA Toolkit, ROCm ou les pilotes GPU. Il lie les bibliothèques internes de manière
statique et installe `/usr/local/bin/mgs`. Aucun des deux installateurs n'utilise ni n'installe sudo.

Sous Debian/Ubuntu, `libnccl2` et `libnccl-dev` sont installés ensemble dans la même version exacte,
choisie pour la version de CUDA détectée. Cette commande APT utilise
`--allow-downgrades` afin qu'un paquet NCCL installé plus récent ne bloque pas l'installation
de la version choisie. Cette option n'est passée qu'à la commande d'installation de NCCL.

Si la distribution fournit une version plus ancienne de Node.js, installez d'abord une version prise en
charge. L'interface est toujours compilée localement ; la compilation ne peut pas télécharger les
ressources précompilées d'amont, auxquelles manquent les traductions et les routes d'API de ce projet.
Un échec de la compilation de l'interface arrête l'installation.

Depuis une copie locale complète, choisissez une compilation CPU ou listez les options d'installation :

```bash
bash deploy/install-update-reinstall-debian.sh --domain ai.example.com --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

Sous Alpine, installez Bash avec `apk add --no-cache bash` avant d'exécuter son installateur.

Exécutez un seul modèle avec :

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

Aucune autre option n'est nécessaire : `mgs` choisit la configuration qui donne la plus grande vitesse
d'inférence sur la machine où il s'exécute. L'ajustement automatique place le modèle sur le GPU et, lorsque
la mémoire système le permet, garde les experts MoE en RAM afin que la VRAM libre devienne le
[cache d'experts MoE](#expert-cache) ; il choisit aussi le
[micro-lot et la taille du contexte](#context-size). Le nombre de threads CPU suit le processeur
([threads CPU](#threads)), et les couches MTP trouvées dans le GGUF sont activées et conservées en VRAM
([MTP](#mtp-vram)). Toute option donnée explicitement (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`,
`--override-tensor`…) prévaut sur le choix automatique. Le journal de démarrage indique ce qui a été choisi.

Exécutez `mgs` avec votre propre utilisateur : il n'a jamais besoin de root. Lancé en tant que root, il
journalise l'avertissement `mgs is running as root: run it with an unprivileged user instead` (le serveur
MCP de shell local donnerait au modèle un shell root) et poursuit le démarrage.

Ouvrez l'URL HTTPS affichée au démarrage, normalement `https://127.0.0.1:11443/`.
Le chargement d'un grand modèle peut prendre du temps. L'interface web et la documentation de l'API
restent disponibles pendant que les requêtes adressées au modèle signalent le chargement. `--help` liste
les options héritées du moteur. Les fichiers GGUF, la mémoire GPU, la longueur du contexte et les
projecteurs propres au modèle doivent correspondre à la charge de travail prévue.

Le code du serveur active HTTPS par défaut pour toutes les méthodes de compilation et d'installation.
Il se lie au premier port HTTPS disponible à partir de 11443, puis à un port de redirection HTTP à partir
de 11080, en sautant le port HTTPS choisi. HTTP répond par une redirection permanente 308 vers le port
HTTPS effectif, en conservant l'hôte, le chemin et la chaîne de requête demandés. Le journal affiche les
URL sélectionnées ; en cas de conflit de ports, chaque séquence avance indépendamment.
`--port N` fixe le port principal et échoue s'il est occupé ; `--port 0` laisse le système d'exploitation
le choisir. `--http-port N` change le début de la recherche du port de redirection.
Les variables d'environnement correspondantes sont `MGS_ARG_PORT` et `MGS_ARG_HTTP_PORT`.

Jusqu'à la version 0.1.34, la recherche s'arrêtait au deuxième port : un `mgs` lancé alors que le 11443 était occupé (par exemple par un autre `mgs`) se terminait avec `couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 11444` au lieu de prendre le port libre suivant.

Sans fichiers de certificat, l'exécutable crée un certificat autosigné et sa clé privée lors de son premier
démarrage, sans nécessiter de commande `openssl` externe, et les enregistre dans `~/.mgs/certificates/`
(`fullchain.pem` et `privkey.pem`, la clé n'étant lisible que par vous). Chaque exécution ultérieure, et
chaque nouvelle compilation, les réutilise, de sorte que le navigateur conserve son exception de confiance
et n'a pas besoin de recharger la page ; une nouvelle paire n'est créée que lorsque ces fichiers manquent, ne
correspondent pas ou que le certificat a expiré (au bout de 365 jours). Le certificat couvre `localhost`, les
adresses de bouclage, l'hôte indiqué avec `--host`, ainsi que le nom de la machine et les adresses réseau
qu'elle avait au moment de sa création ; pour faire disparaître complètement l'avertissement du navigateur,
importez `~/.mgs/certificates/fullchain.pem` comme certificat de confiance, et supprimez les deux fichiers
pour en créer un nouveau. Si ce dossier n'est pas accessible en écriture, chaque exécution génère son propre
certificat en mémoire. Fournissez à la fois `--ssl-cert-file /path/fullchain.pem` et `--ssl-key-file
/path/privkey.pem` pour utiliser un certificat reconnu par vos clients. Une paire manquante ou non valide
arrête le démarrage. Les compilations faites avec `--no-openssl` ne peuvent pas servir HTTPS et exigent un
`--http` explicite (ou `MGS_ARG_HTTPS=false`) ; le HTTP simple commence sa recherche de port à 11080.
Les processus enfants du routeur utilisent HTTP explicitement sur des ports de bouclage privés. Les sockets
Unix et le `AIP_MODE=PREDICTION` de Vertex AI conservent leur transport HTTP.

La configuration CMake se trouve dans `build/CMakeLists.txt`. Les scripts d'installation et de
compilation locale incluent tous deux `build/` dans leur copie des sources et le sélectionnent
automatiquement. Les installateurs réinitialisent la configuration CMake générée si un cache existant
pointe vers un autre répertoire des sources. Les compilations locales démarrent toujours dans un
répertoire neuf sous `/tmp/` et le suppriment à la sortie. Pour utiliser CMake directement, exécutez
`cmake -S build -B _/temp/cmake` depuis la racine du projet ; les commandes habituelles de l'installateur
et de la compilation locale restent les mêmes. HTTPS utilise par défaut l'OpenSSL du système
(`MGS_OPENSSL=ON`) ; dans une compilation CMake directe, `-DMGS_BUILD_BORINGSSL=ON` ou
`-DMGS_BUILD_LIBRESSL=ON` télécharge et compile BoringSSL ou LibreSSL à la place, et
`-DMGS_OPENSSL=OFF` sans l'un ni l'autre compile le serveur sans HTTPS.

<a id="local-build"></a>
## Compilation locale pour une utilisation manuelle

`build/build-local.sh` compile le projet sur votre propre ordinateur et installe une copie autonome
pour votre utilisateur, que vous lancez à la main. Il n'installe aucun service, ne démarre rien en
arrière-plan, n'utilise jamais root ni sudo et n'installe jamais de paquets.
Exécutez-le avec votre utilisateur habituel, depuis n'importe quel dossier :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

La commande d'une ligne télécharge l'archive des sources de `main`, sans Git, dans le dossier de
compilation privé sous `/tmp`. L'archive et les sources extraites sont supprimées avec les autres
fichiers temporaires de compilation à la sortie. Une copie isolée du script fonctionne de la même
manière. Depuis une copie locale complète du projet, compilez vos sources locales avec :

```bash
bash build/build-local.sh
```

Exécuter ce script par son chemin absolu utilise également le projet complet qui le contient. Le
message `Using local sources: ...` confirme quel dossier est compilé, y compris ses traductions de
l'interface. Si l'exécution se termine par `Error: Compilation failed.`, aucun nouvel exécutable n'a
été installé ; une installation existante conserve son interface précédente. Lisez la première erreur
du compilateur au-dessus de `ninja: build stopped`, corrigez-en la cause et relancez le même script.
Une fois la compilation et l'installation réussies, redémarrez `mgs` et rechargez l'interface web.

Des erreurs comme `std::vector` n'ayant aucun membre nommé `mPushBack` ou
`no declaration matches` dans `subproc.cpp` indiquent des noms incohérents dans les sources.
Utilisez les fichiers sources corrigés et relancez la compilation. La compilation de l'interface exige
aussi que ses imports de configuration correspondent aux noms exportés par les modules sources ; un
export manquant est signalé par l'étape npm/Vite.

Si SvelteKit signale `Page options are ignored when router.type === 'hash'`
et nomme `fLoad`, les routes du chat doivent exporter le nom exact `load` du
framework. Les routes corrigées conservent la validation de la clé API. La compilation exécute
`svelte-kit sync` avant de générer les ressources et de compiler l'interface, et
s'arrête si la synchronisation échoue. Un échec à l'étape `Provisioning UI assets` empêche toujours
l'installation, même lorsque les bibliothèques C++ ont déjà été compilées.

Pour passer des options par la commande d'une ligne, utilisez `bash -s --` ; par exemple, pour une
compilation CPU :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` affiche les options sans télécharger l'archive des sources ni compiler.

Le résultat est installé pour votre utilisateur sous le préfixe `~/.local` :

| chemin | contenu |
| --- | --- |
| `~/.local/bin/mgs` | Le programme que vous exécutez : un petit lanceur statique qui trouve `../lib/mgs/` par rapport à son propre emplacement et démarre le serveur au moyen du chargeur fourni. |
| `~/.local/lib/mgs/` | Le binaire du serveur (`mgs`, avec l'interface web et la documentation de l'API intégrées) et toutes les bibliothèques partagées dont il a besoin : le chargeur dynamique et la bibliothèque C de glibc, l'environnement d'exécution C++, OpenMP, OpenSSL, les modules de service de noms de glibc et, dans les compilations CUDA, l'environnement d'exécution CUDA, cuBLAS et NCCL ; dans les compilations ROCm, les bibliothèques ROCm (HIP, hipBLAS, rocBLAS, hipBLASLt…) et les dossiers de noyaux GPU de rocBLAS et hipBLASLt (`rocblas/library`, `hipblaslt/library`). |
| `~/.local/share/doc/mgs/licenses/` | La licence PolyForm NonCommercial 1.0.0 du projet (`LICENSE`) et les avis de licence de tiers copiés depuis le dossier `build/licenses/` de l'arborescence des sources (llama.cpp, FreeToken, Swagger UI, cpp-httplib et nlohmann/json). |

Seul l'utilisateur qui exécute le script peut utiliser cette installation. Tout est résolu par rapport
à l'emplacement du lanceur : il exécute `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` depuis son préfixe,
et le serveur et ses bibliothèques cherchent dans `lib/mgs/` avant tout le reste, y compris
`LD_LIBRARY_PATH`. La seule bibliothèque encore prise dans le système est le pilote NVIDIA
(`libcuda.so.1` et les bibliothèques `libnvidia-*` qu'il charge), car il doit correspondre au module du
noyau en cours d'exécution ; avec `--vulkan`, s'y ajoute le pilote Vulkan du système ([Vulkan](#vulkan)) ;
avec `--rocm`, seul le pilote noyau amdgpu (`/dev/kfd`, `/dev/dri`) est utilisé depuis le système, et le
lanceur oriente rocBLAS et hipBLASLt vers les dossiers de noyaux fournis dans `lib/mgs/` ([ROCm](#rocm)).
À la fin, le script liste, au moyen du chargeur fourni, les bibliothèques que le serveur chargerait,
échoue si l'une d'elles provient de l'extérieur de `lib/mgs/`, et exécute `--version` via le lanceur.

Debian ajoute `~/.local/bin` au `PATH` à l'ouverture de session dès que ce dossier existe (via
`~/.profile`) : après vous être reconnecté, vous pouvez donc exécuter `mgs` par son nom ; le script
affiche un rappel tant que ce dossier n'est pas dans le `PATH`. Lancez-le quand vous le souhaitez et
arrêtez-le avec Ctrl+C :

```bash
mgs --model /path/model.gguf
```

Le chemin complet fonctionne aussi : `~/.local/bin/mgs --model /path/model.gguf`.

Ouvrez `https://127.0.0.1:11443/` pour le chat et `https://127.0.0.1:11443/api/doc/` pour
la documentation de l'API. Toutes les options de ce manuel s'appliquent. Lancez `mgs` sans `--model` pour
démarrer le routeur multimodèle et choisir le dossier des modèles dans Paramètres → Modèles
([choisir et charger des modèles](#settings-models)).

Prérequis, que vous installez vous-même en tant que root, car le script se contente de les vérifier :
`cmake`, un compilateur C/C++ et la bibliothèque C statique (`build-essential`, qui
entraîne `libc6-dev`), `patchelf`, `libssl-dev` (ou
`--no-openssl`), Node.js 20.19+/22.12+ avec npm (ou `--no-ui`) et la chaîne d'outils du backend
choisi : pour CUDA (par défaut), un CUDA Toolkit déjà installé et NCCL avec ses fichiers de
développement (`libnccl2` et `libnccl-dev`, ou `--no-nccl`) ; pour `--rocm`, un ROCm déjà installé
([ROCm](#rocm)) ; pour `--vulkan`, les paquets Vulkan ([Vulkan](#vulkan)) ; `--cpu` n'en nécessite aucune. `nvcc` est pris dans `CUDACXX`, le `PATH` ou
`/usr/local/cuda/bin/nvcc`. Ninja est utilisé s'il est installé.
Les utilitaires de base comprennent `tar`, `coreutils` et `util-linux`. Les compilations distantes
nécessitent aussi `curl`, `ca-certificates` et `gzip` ; aucune dépendance n'est installée automatiquement.

Options : `--prefix PATH` (un autre préfixe d'installation), `--jobs N`, `--cuda`, `--rocm`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--rocm-targets LIST`, `--no-ui`, `--no-openssl`, `--no-nccl`
et `--help`. `CUDACXX` choisit le compilateur CUDA et `HIPCXX` le compilateur clang de ROCm.

Bon à savoir :

- La compilation est optimisée pour le CPU de cet ordinateur et pour les GPU présents au moment de
  la compilation. Recompilez sur chaque ordinateur au lieu de copier l'installation.
- L'installation est déplaçable en tant que préfixe : déplacez ensemble `bin/mgs` et `lib/mgs/`, en
  conservant leurs niveaux `bin/` et `lib/`. Démarrez toujours `bin/mgs` ; `lib/mgs/mgs` est interne et
  ne doit pas être exécuté directement, car il contournerait le chargeur fourni.
- Relancer le script met à jour l'installation. Il ne remplace que `bin/mgs`, `lib/mgs/` et
  `share/doc/mgs/licenses/`. Une copie déjà en cours d'exécution continue de fonctionner ; redémarrez-la
  pour utiliser la nouvelle compilation.
- Le script refuse un préfixe où `bin/mgs`, `lib/mgs/` ou ce dossier de licences existent déjà, sauf
  s'il les a créés lui-même ; il laisse un marqueur caché `.moe-gguf-server-bundle` dans `lib/mgs/`.
- `--prefix PATH` installe sous un autre préfixe dans lequel votre utilisateur peut écrire ; ce ne peut
  être ni `/` ni un dossier situé à l'intérieur du projet.
- Les sources préparées, les fichiers de compilation, les caches, l'installation assemblée et le journal
  utilisent un dossier privé `/tmp/moe-gguf-server-build.XXXXXXXX/`. Après avoir copié le résultat
  dans le préfixe, le script supprime ce dossier. Il fait aussi le ménage en cas d'erreur
  et d'interruption gérée. Chaque exécution compile à partir de zéro, sans écrire dans
  l'arborescence des sources ni dans `/opt/`. Les compilations locales n'acceptent pas `--build-dir`.
- Le préfixe est verrouillé pendant la copie et le remplacement des artefacts terminés,
  ce qui empêche les mises à jour simultanées de la même installation.
- Une compilation CUDA a besoin du pilote NVIDIA pour démarrer ; utilisez `--rocm` ou `--vulkan` sur les
  ordinateurs dotés d'un autre GPU et `--cpu` sur ceux qui n'ont pas de GPU.

<a id="vulkan"></a>
## Vulkan (GPU AMD, Intel et NVIDIA)

Outre CUDA (par défaut) et le CPU, `build/build-local.sh` comme l'installateur Debian compilent le
backend Vulkan avec `--vulkan`, pour les GPU sans CUDA, comme les cartes AMD Radeon. Installez d'abord
les dépendances de compilation en tant que root :

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Ensuite, avec votre utilisateur habituel :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

L'installateur Debian accepte la même option et installe lui-même ces paquets Vulkan :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` affiche le GPU, par exemple `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- L'installation locale inclut le chargeur Vulkan (`libvulkan.so.1`) comme n'importe quelle autre
  bibliothèque, mais le pilote Vulkan provient du système : le chargeur lit les pilotes enregistrés dans
  `/usr/share/vulkan/icd.d/` (Mesa RADV pour AMD, ANV pour Intel, le pilote NVIDIA), et ceux-ci chargent
  leurs propres bibliothèques système, car ils doivent correspondre au noyau et au système. Compilez sur
  la version de Debian qui exécutera le binaire.
- L'utilisateur qui exécute `mgs` a besoin du nœud de rendu du GPU (`/dev/dri/renderD*`). Un utilisateur
  connecté au bureau en dispose déjà ; les autres utilisateurs ont besoin du groupe `render` (et `video`).
  `configure-web.sh` ajoute le compte de service aux deux.
- MTP en VRAM, le cache d'experts MoE, le préchargement des experts, `--fit` et la page Supervision
  fonctionnent comme avec CUDA. Sur AMD, la page Supervision lit l'utilisation, la VRAM et la température
  depuis sysfs.
- `mgs` ne compte comme mémoire GPU que la VRAM réelle (les tas de mémoire « device-local »), jamais la
  GTT. La GTT est la partie de la RAM système (normalement la moitié) qu'un GPU AMD ou Intel peut mapper :
  ce n'est pas de la mémoire supplémentaire, et le GPU la lit via PCIe, plusieurs fois plus lentement que
  sa VRAM. Les couches MTP, les poids denses, le cache KV et le cache d'experts sont alloués en VRAM ; les
  experts qui n'y tiennent pas restent en RAM normale, où le CPU les calcule.
- Un GPU intégré (la partie graphique d'un processeur AMD ou Intel) n'a pas de VRAM propre : chaque tas de
  mémoire qu'il signale est de la RAM système. `mgs` compte cette mémoire et la RAM comme un budget unique et
  n'y crée pas le cache d'experts ; voir [GPU intégrés](#integrated-gpu).
- `mgs` vérifie où le noyau conserve réellement cette mémoire (`/proc/self/fdinfo`, publié par amdgpu,
  i915 et xe). Si un autre programme remplit la VRAM et que le pilote en déplace une partie vers la GTT,
  le journal et la page Supervision le signalent, et la page Supervision affiche l'emplacement réel (VRAM
  et GTT) de chaque GPU AMD ou Intel dédié (un GPU intégré n'a pas de VRAM propre, il n'est donc pas
  vérifié). Le cache d'experts est alloué avec une priorité mémoire plus faible (`VK_EXT_memory_priority`,
  utilisée chaque fois que le GPU la propose, sans variable pour l'activer), de sorte que le pilote le
  déplace avant les couches MTP, les poids ou le cache KV. Sans BAR redimensionnable (Resizable BAR), les
  tampons de périphérique évitent aussi la petite fenêtre de VRAM que le CPU peut mapper (256 Mio sur une
  RX 580), que le bureau partage et que le pilote vide en premier vers la GTT, et cette fenêtre ne compte
  pas comme mémoire libre pour eux.
- Seul CUDA peut épingler les pages du modèle qui restent en RAM ; avec Vulkan, les experts atteignent
  donc le GPU via de la mémoire paginable. `mgs` les copie via un petit anneau de transit et met en file
  la copie vers le GPU sans l'attendre, de sorte que le GPU et le CPU continuent de travailler pendant le
  transfert des experts. Sur une RTX 4060 Ti avec Qwen3.6-35B-A3B Q8_0, cela a fait passer le traitement
  de l'invite de 455 à 515 jetons/s et la génération de 20,6 à 23,2 jetons/s. `--load-mode none` copie
  plutôt les experts dans de la mémoire hôte épinglée. Avant l'anneau de transit, sur la même carte avec
  le modèle Q4_K, il a fait passer les transferts d'experts de 2,7 à 13,1 Go/s et la génération de 27,5 à
  36,6 jetons/s. Sur AMD et Intel, cette mémoire
  épinglée est décomptée de la GTT : lorsque les experts n'y tiennent pas, le chargement échoue avec
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), comme ce fut le cas pour
  les 18,9 Go d'experts de ce modèle avec 16,8 Go de GTT sur une RX 580. Ne l'utilisez que lorsque les
  experts tiennent dans la GTT et dans la RAM libre.
- Avec un GPU NVIDIA via Vulkan, et avec des experts BF16 sur n'importe quel GPU, les produits d'experts
  convertissent d'abord leur entrée dans un tampon de travail. Ce tampon prenait 512 Kio par jeton de
  l'invite, si bien que les invites de quelques milliers de jetons échouaient avec
  `ErrorOutOfDeviceMemory` dès que le cache d'experts avait rempli la VRAM, sauf à augmenter
  `--moe-cache-reserve`. Il ne prend plus que quelques Kio par jeton (32 Mio pour un micro-lot de 4096
  jetons de Qwen3.6-35B-A3B), bien en deçà de la réserve par défaut. Sur une RTX 4060 Ti avec
  Qwen3.6-35B-A3B Q8_0, cela a aussi accéléré le traitement de l'invite, de 515 à 549 jetons/s.
- Dimensionnez le modèle en fonction de la machine : le GGUF entier doit tenir dans la RAM libre plus la
  VRAM, moins le cache KV et les tampons de calcul. Avec 8 Go de VRAM et 16 Go de RAM, cela représente
  environ 18 à 20 Go de poids ; les fichiers plus gros sont relus depuis le disque encore et encore, et la
  génération devient très lente. Le Q4_K de 21,6 Go ci-dessus a atteint 7,8 jetons/s sur une RX 580 avec
  15 Go de RAM et 15,9 jetons/s une fois que la machine a disposé de 31 Go (10,0 avec `--spec-type none`).

<a id="rocm"></a>
## ROCm (GPU AMD)

`build/build-local.sh` comme l'installateur Debian compilent le backend ROCm (HIP) avec `--rocm`, pour les
GPU AMD que prend en charge la version de ROCm installée. Aucun des deux scripts n'installe ROCm : installez
vous-même ROCm 6.1 ou ultérieur en tant que root, avec HIP, hipBLAS et rocBLAS et leurs fichiers de
développement. Pour la compilation locale, installez aussi en tant que root les dépendances de compilation
communes (l'installateur Debian les installe lui-même) :

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Ensuite, avec votre utilisateur habituel :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

L'installateur Debian accepte la même option :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --rocm
```

- Les scripts trouvent ROCm via `hipconfig` (dans le `PATH` ou `/opt/rocm/bin/hipconfig`) et utilisent le
  compilateur clang qu'il indique ; `HIPCXX` en choisit un autre. Une configuration qui échoue indique ce qui
  manque : HIP 6.1 ou ultérieur avec les fichiers de développement de hipBLAS et rocBLAS.
- Les cibles GPU sont celles des GPU présents au moment de la compilation, lues avec `amdgpu-arch` ou
  `rocminfo`. Sans GPU détecté, ou pour compiler pour d'autres GPU, passez `--rocm-targets` avec une liste
  séparée par des virgules, par exemple `--rocm-targets gfx1100,gfx1030`.
- Les versions actuelles de ROCm ne prennent pas en charge les GPU GCN jusqu'à Polaris (`gfx6xx` à `gfx8xx`,
  comme la Radeon RX 580, `gfx803`) : les scripts avertissent au sujet de ces cibles ; utilisez `--vulkan`
  pour elles.
- L'installation locale copie toutes les bibliothèques ROCm (HIP, hipBLAS, rocBLAS, hipBLASLt…) dans
  `lib/mgs/`, ainsi que les dossiers de noyaux GPU que rocBLAS et hipBLASLt chargent à l'exécution
  (`rocblas/library`, `hipblaslt/library`) ; le lanceur oriente les deux bibliothèques vers ces copies
  (`ROCBLAS_TENSILE_LIBPATH`, `HIPBLASLT_TENSILE_LIBPATH`), de sorte qu'elles ne lisent jamais les noyaux
  d'une autre version de ROCm. Seul le pilote noyau amdgpu provient du système. Les dossiers de noyaux
  occupent des gigaoctets.
- Le `/usr/local/bin/mgs` de l'installateur Debian charge les bibliothèques ROCm du système, via le chemin
  d'exécution (run path) enregistré lors de la compilation ; lorsque certaines sont introuvables,
  l'installateur s'arrête et demande d'ajouter le dossier `lib` de ROCm dans `/etc/ld.so.conf.d/` et
  d'exécuter `ldconfig`.
- L'utilisateur qui exécute `mgs` a besoin de `/dev/kfd` et de `/dev/dri`, c'est-à-dire des groupes `render`
  et `video` ; l'installateur ajoute le compte de service aux deux.

<a id="deployment"></a>
## Déploiement en production

L'installateur ([installation](#installation)) compile, installe et configure les services web en une
seule exécution :

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com
```

Sans `--domain`, le domaine est la sortie de `hostname -f` (ou celle de `hostname` ou de `uname -n`
lorsque ce n'est pas un nom valide). Le domaine doit pointer vers l'hôte de production. Pour modifier plus
tard la configuration web sans recompiler, exécutez en tant que root, depuis le répertoire des sources
(`/opt/moe-gguf-server-source` pour les installations distantes) :

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

`configure-web.sh` accepte les mêmes options web que l'installateur (`--domain`, `--certificate`,
`--private-key`, `--proxy-ip`), plus `--binary PATH` pour un emplacement d'exécutable autre que celui par
défaut, `--no-system-deps` et `--check`, qui ne fait que valider les options. Chaque exécution réécrit
toute la configuration à partir des options qu'elle reçoit : la relancer sans `--proxy-ip` supprime le
proxy, et sans `--domain` le domaine revient à celui de `hostname -f`.

Le service `moe-gguf-server-web`
exécute une instance Apache dédiée sous l'identité `moe-gguf-server`, sans root.
Chaque démarrage sélectionne le premier port HTTP disponible à partir de 11080 et le premier port HTTPS
disponible à partir de 11443. Chaque séquence avance indépendamment jusqu'à 65535, sans attribuer le
même port aux deux protocoles. Le service inspecte les sockets TCP IPv4/IPv6 en écoute et réessaie si
Apache détecte qu'un port est devenu occupé après sa sélection. Les autres échecs de configuration
arrêtent le démarrage au lieu d'être pris pour des conflits de ports.

HTTP redirige de façon permanente vers le port HTTPS sélectionné, en conservant le chemin et la chaîne
de requête. La redirection n'est pas mise en cache. Le service d'inférence (`moe-gguf-server`, qui s'exécute lui aussi
sous `moe-gguf-server` et démarre avec `--http`) n'écoute sur aucun port TCP : il écoute sur le socket
Unix `/opt/moe-gguf-server/_/temp/web/mgs.sock`, que seul ce compte peut ouvrir, et Apache lui transmet les
requêtes. Avant d'écouter, `mgs` supprime un fichier de socket laissé par un serveur qui ne s'exécute plus,
mais jamais le socket d'un serveur qui répond encore. L'installation requiert root ;
aucun des deux services n'a besoin de s'exécuter en root. Aucun HAProxy n'est configuré et aucun site
Apache partagé n'est importé ni modifié.

Arrêter le service d'inférence (`systemctl stop moe-gguf-server` ou `systemctl restart moe-gguf-server`), comme Ctrl+C ou un `SIGTERM` quand vous lancez `mgs` vous-même, le termine proprement en une ou deux secondes, même au milieu d'un long prompt : les requêtes en cours reçoivent une erreur 503 (`The server is shutting down.`) ou, en streaming, leur flux se termine. Un second Ctrl+C ou `SIGTERM` termine le processus immédiatement, et pendant le chargement du modèle le premier suffit déjà.

Les URL effectives sont publiées une fois les deux sockets d'écoute ouverts :

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

Par exemple, HTTP 11081 et HTTPS 11445 redirigent vers
`https://ai.example.com:11445/`. Le fichier est supprimé à l'arrêt et régénéré au démarrage ; un
redémarrage peut choisir des ports différents. Les clients, les pare-feu et tout proxy externe doivent
utiliser les valeurs publiées. Le port fait partie de
l'[origine du navigateur](https://developer.mozilla.org/en-US/docs/Glossary/Origin) : un port HTTPS
différent a donc un historique et des paramètres locaux distincts. Les conversations existantes restent
dans leur origine initiale ; l'export/import permet de les transférer.

Le compte système n'a pas de mot de passe interactif. Les modèles sont stockés dans
`/opt/moe-gguf-server/models`, et la configuration du programme se trouve dans le répertoire personnel du
compte de service, `/opt/moe-gguf-server/.mgs/config/` : la clé API (`api.key`), le domaine
web (`web-domain`), le proxy de confiance (`web-proxy-ip`, uniquement avec `--proxy-ip`), le gabarit
Apache (`apache2.conf.in`) et les paramètres de
l'interface (`ui-settings.json`), tous appartenant à `moe-gguf-server` avec le mode 0600.
La sortie de l'installation est consignée dans `/root/webapp-install.log`, et
`/root/webapp-credentials.txt` contient chaque identifiant que crée le déploiement : le compte de service
(`moe-gguf-server`, sans mot de passe et avec la connexion désactivée) et la clé API. Les deux fichiers
sont en mode 600. La reconfiguration préserve les clés, les modèles et les certificats existants.

Utilisez votre propre certificat TLS avec :

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

Le certificat se trouve dans `/opt/moe-gguf-server/certificates/` (`fullchain.pem` et
`privkey.pem`, appartenant au compte `moe-gguf-server`, la clé n'étant lisible que par lui).
Sans fichiers fournis, le script conserve le certificat déjà présent ou, la première fois, en crée un
autosigné, que chaque réinstallation ultérieure réutilise, de sorte que les navigateurs conservent leur
exception de confiance. Faites-lui confiance sur votre client ou fournissez un certificat auquel vos
clients font déjà confiance. Les fichiers fournis sont copiés dans ce dossier ; le renouvellement du
certificat doit actualiser ces copies et redémarrer `moe-gguf-server-web`. Le script n'obtient ni ne
renouvelle de certificats Let's Encrypt.
Utilisez la même commande après avoir renouvelé les certificats sources.

Lorsqu'un proxy inverse (HAProxy, nginx…) se trouve devant Apache, indiquez son adresse avec
`--proxy-ip` (IPv4 ou IPv6), à l'installateur ou à `configure-web.sh` :

```bash
bash deploy/configure-web.sh --domain ai.example.com --proxy-ip 10.7.10.1
```

Apache charge alors `mod_remoteip`. En HTTPS, il n'exige l'en-tête du protocole PROXY que de cette adresse
(`RemoteIPProxyProtocol On`, toutes les autres adresses étant listées dans
`RemoteIPProxyProtocolExceptions`), de sorte que les autres machines continuent de se connecter
directement ; en HTTP, il fait confiance à l'en-tête `X-Forwarded-For` de cette adresse (`RemoteIPHeader`,
`RemoteIPTrustedProxy`). Les journaux d'accès enregistrent l'adresse réelle du client. Sans `--proxy-ip`,
aucun proxy n'est configuré.

Le service utilise le mode routeur. Placez des modèles GGUF lisibles dans
`/opt/moe-gguf-server/models`, ou utilisez la gestion des modèles dans l'interface. Pour des options
de ligne de commande spécifiques, utilisez une surcharge systemd sous Debian/Ubuntu ou la configuration
du service OpenRC sous Alpine. Gardez le service d'inférence sur son socket
Unix.
Le configurateur web régénère le fichier de service qu'il gère lorsqu'il est relancé.

Redémarrez le service web pour sélectionner de nouveau les ports :

```bash
systemctl restart moe-gguf-server-web
```

Sous Alpine, utilisez `rc-service moe-gguf-server-web restart`. Les journaux de démarrage se trouvent dans
`journalctl -u moe-gguf-server-web` sous Debian ou dans `/var/log/moe-gguf-server-web.log`
sous Alpine. Les diagnostics d'Apache se trouvent dans `/opt/moe-gguf-server/_/temp/web/startup.log`
et `error.log` ; les journaux d'accès du domaine restent dans `/var/www/DOMAIN-logs/`. Le gabarit
autonome est `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. Ne modifiez pas la configuration
générée dans `_/temp/web/` ; elle est reconstruite à chaque démarrage.

<a id="chat"></a>
## Utilisation de l'interface

1. Ouvrez l'URL du serveur et saisissez sa clé API lorsqu'elle vous est demandée, ou dans Paramètres → Général.
2. Sélectionnez un modèle. En mode routeur, chargez un modèle disponible si aucun n'est chargé ; tant qu'il
   n'y en a aucun, le sélecteur de modèle renvoie vers Paramètres → Modèles, où vous choisissez le dossier de vos
   modèles ([choisir et charger des modèles](#settings-models)).
3. Saisissez un message et envoyez-le. Entrée envoie par défaut ; Maj+Entrée insère un saut
   de ligne. Ces raccourcis peuvent être modifiés dans les paramètres.
4. Ajoutez des fichiers avec l'action de pièce jointe. Les images, l'audio et la vidéo nécessitent des
   modalités compatibles du modèle. Les fichiers PDF peuvent être extraits sous forme de texte ou convertis
   en images ; les modèles sans vision utilisent la voie texte.
5. Arrêtez la génération, ignorez le raisonnement lorsque c'est pris en charge, modifiez un message,
   régénérez une réponse ou créez une branche d'une conversation à l'aide des actions du message.

Les trois traits en haut de la barre latérale la déplient et la replient. Elle affiche **Nouvelle conversation**,
**Supervision**, **Modèle** et **Paramètres**, suivis des conversations : **Épinglées**, **Conversations récentes**
(les cinq sur lesquelles vous avez travaillé en dernier ; une conversation à laquelle vous envoyez de nouveau un
message y remonte) et **Conversations** (les autres, des plus récentes aux plus anciennes). Chaque conversation a une
bulle avec un numéro de 01 à 99, attribué à sa création et conservé pendant toute son existence (un numéro libéré par
la suppression d'une conversation va à la prochaine nouvelle conversation), afin de pouvoir la distinguer lorsque la
barre latérale est repliée et que seules les icônes du menu et les bulles sont affichées. Les conversations peuvent être
épinglées, renommées, supprimées et sélectionnées en masse. Le champ de recherche situé à côté des trois traits (Ctrl+K,
ou ⌘K sur un Mac, déplie la barre latérale et y place le curseur) trouve, sans tenir compte de la casse ni des accents,
les conversations par leur titre et par le contenu de leurs messages, les pages et leurs parties, chaque paramètre, les
modèles, les outils et les serveurs MCP. Cliquez sur un résultat, ou appuyez sur Entrée pour le premier, pour ouvrir sa
conversation ou sa page, où l'élément trouvé est mis en évidence un instant.
Les conversations sont conservées par le serveur dans `~/.mgs/chats/` de l'utilisateur qui exécute `mgs` (dans le service
déployé, `/opt/moe-gguf-server/.mgs/chats/`), un fichier par conversation nommé d'après sa date de création, en UTC
(`2026-11-03@14-59-59.723.jsonl`), de sorte que tous les navigateurs qui ouvrent l'interface voient les mêmes conversations, et
qu'une page ouverte affiche en quelques secondes les conversations qu'un autre navigateur crée, modifie ou supprime. Les paramètres sont conservés par le serveur
dans le fichier `~/.mgs/config/ui-settings.json` de l'utilisateur qui exécute `mgs` (dans le service
déployé, `/opt/moe-gguf-server/.mgs/config/ui-settings.json`), de sorte que tous les navigateurs qui
ouvrent l'interface les partagent : paramètres de génération, options d'affichage, thème, langue, serveurs
MCP, modèles favoris, outils désactivés et toujours autorisés, effort de raisonnement par défaut et dossier
des modèles de Paramètres → Modèles. Seule la clé API reste dans chaque navigateur, car elle authentifie la
requête qui charge le reste ; saisissez-la
donc une fois dans chaque navigateur. L'import/export est disponible dans les
paramètres. Les exports
peuvent contenir des pièces jointes et, si vous les sélectionnez explicitement, des clés API sensibles ou
des en-têtes d'autorisation MCP.

Choisissez la langue dans Paramètres → Général. Les 15 options sont classées ainsi : `de-DE`,
`en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`,
`ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN` ; la valeur par défaut est `en-US`.
Le portugais du Brésil et le portugais européen disposent de traductions distinctes. L'hébreu utilise une
interface de droite à gauche ; la saisie du chat et les messages simples de l'utilisateur suivent la
direction de leur texte, tandis que le code et la sortie du terminal s'affichent de gauche à droite.
Changer de langue enregistre le choix dans les paramètres du serveur et
recharge l'interface ; le serveur le publie dans `/api/config.js`, de sorte que tous les navigateurs
démarrent dans cette langue. Enregistrez d'abord les autres paramètres en attente. Le paramètre de requête
d'URL `lang` le remplace pour cette page, et conserve le choix lorsque celui-ci ne peut pas être enregistré.
Par exemple, ajoutez `?lang=ja-JP` à l'URL de l'interface pour sélectionner le japonais.

Paramètres → Général → **Toujours répondre dans la langue de l’interface** est désactivé par
défaut. Activez-le et enregistrez les paramètres pour demander au modèle d'utiliser la langue et la
variante régionale de l'interface pour chaque nouvelle réponse, y compris dans les conversations
existantes, les réponses régénérées, les continuations, les tours qui suivent des appels d'outils et les
titres générés par le modèle. Pour `es-AR`, l'instruction demande explicitement le voseo argentin :
« Seguí estos pasos », « tenés » et « podés » ; `es-ES` demande l'usage espagnol d'Espagne. L'anglais
britannique/américain et le portugais du Brésil/européen reçoivent également des instructions régionales
distinctes.

L'instruction accompagne chaque nouvelle requête d'inférence et chaque préremplissage du cache KV ;
elle est combinée au message système sortant sans modifier votre invite système enregistrée ni la
conversation. Changer la langue de l'interface affecte les nouvelles requêtes après le rechargement ; une
réponse déjà en cours de génération conserve l'instruction qu'elle a reçue. Désactiver le paramètre cesse
de l'ajouter aux requêtes suivantes. Les réponses existantes ne sont pas traduites. Le code, les commandes,
les noms, les citations et les traductions explicitement demandées conservent la langue requise ; les
explications qui les entourent suivent la langue de l'interface. La cohérence dépend de la capacité du
modèle à suivre l'instruction ; il ne s'agit pas d'une traduction automatique ni d'une garantie de respect
parfait. Les clients externes de l'API ne sont pas concernés par ce paramètre du navigateur.

Choisissez Système, Clair ou Sombre pour le thème. Comme pour la langue, le serveur publie le thème
enregistré dans `/api/config.js`, de sorte que chaque page s'ouvre avec ce thème avant d'être dessinée,
sans rien conserver dans le navigateur ; avec Système, l'interface suit immédiatement le mode clair ou
sombre du système d'exploitation, y compris pendant qu'elle est ouverte. Les paramètres de génération comprennent l'utilisation
du contexte, la température, les échantillonneurs, les pénalités et les limites de tours agentiques.
Certaines fonctionnalités dépendent d'options du serveur ou du modèle sélectionné.

Après chaque réponse, l'interface renvoie la conversation, lorsque le serveur est inactif, afin qu'elle
soit déjà encodée dans le cache KV pendant que vous lisez et que le tour suivant démarre plus tôt. C'est
activé par défaut ; désactivez Paramètres → Développement → **Préremplir le cache KV après chaque réponse**
pour l'arrêter.

Pendant que le serveur traite une autre requête, un message que vous envoyez attend dans la
[file d'attente](#request-queue), et sa réponse affiche un avis à la place du texte de progression
jusqu'à ce qu'elle démarre : « En file d'attente : une autre requête envoyée directement à l'API est traitée en premier (requêtes avant la vôtre : N). » tant qu'une requête qu'un programme a envoyée directement à
l'API est en cours d'exécution, ou sinon « En file d'attente : une autre requête de l'interface web est traitée en premier (requêtes avant la vôtre : N). » (un autre onglet ou navigateur, un titre généré ou le
préremplissage du cache KV). N compte les requêtes en cours d'exécution et en attente devant la vôtre.
L'avis disparaît dès que votre requête démarre.

<a id="monitor"></a>
## Supervision et barre d'état

**Supervision** se trouve sous **Nouvelle conversation** dans la barre latérale (adresse `#/monitor`). Tant que la page
est ouverte, elle s'actualise toutes les deux secondes et affiche :

- **Utilisation** : d'abord une carte par GPU de la machine (NVIDIA, AMD ou Intel, même si aucun modèle ne
  l'utilise), intitulée avec son étiquette en gras et son nom (par exemple **gpu-1** : NVIDIA GeForce RTX 4060 Ti ;
  avec plusieurs GPU, gpu-1, gpu-2, etc.), avec son utilisation, l'utilisation du contrôleur mémoire (bande
  passante), la VRAM utilisée, la température et la consommation électrique rapportée à sa limite ; puis une carte
  par processeur, intitulée de la même manière avec son modèle (par exemple **cpu-1** : AMD Ryzen 9 5950X
  16-Core Processor ; avec plusieurs processeurs, cpu-1, cpu-2, etc.), avec son utilisation, une petite barre
  verticale par CPU logique de ce processeur et la charge moyenne du système. Chaque barre d'utilisation est verte
  et se remplit de rouge à mesure que l'utilisation augmente ; une barre dont la valeur est inconnue reste grise.
  Les GPU NVIDIA sont lus au moyen de la bibliothèque NVML du pilote et les GPU AMD au moyen de
  `gpu_busy_percent` dans sysfs ; tout autre GPU (Intel, ou NVIDIA sans NVML) n'affiche aucune utilisation, et son
  nom et sa mémoire proviennent du backend d'inférence dès qu'un modèle l'utilise. Sur un écran large, les cartes
  s'affichent par deux sur chaque rangée, chacune sur la moitié de la largeur comme les barres de mémoire ; avec un
  nombre impair de cartes, la dernière rangée en contient trois.
- **Mémoire** : une barre par GPU (VRAM) et une pour la RAM ; un GPU sans données de mémoire, comme un GPU Intel
  qu'aucun modèle n'utilise encore, n'a pas de barre. Chaque barre est répartie par usage, chaque usage ayant
  toujours la même couleur, et sa légende liste chaque segment avec sa taille et sa part du total, un par ligne et
  dans le même ordre que dans la barre. Survolez un segment pour voir sa taille.

  | Segment | Contenu |
  | --- | --- |
  | Poids MTP / du modèle brouillon | Poids des couches MTP (nextn) du modèle, d'un GGUF MTP séparé passé avec `--spec-draft-model`, ou d'un modèle brouillon. |
  | Poids des experts | Tenseurs des experts routés du modèle. |
  | Poids denses | Tous les autres poids : attention, experts partagés, embeddings et sortie. |
  | Cache KV | Mémoire de contexte du modèle (cache KV ou état récurrent). |
  | Cache KV MTP / du modèle brouillon | Mémoire de contexte du contexte MTP ou du modèle brouillon. |
  | Tampons de calcul | Tampons temporaires des graphes de calcul. |
  | Cache des experts et préchargement | Cache d'experts MoE (pool, identifiants et zone de transit) et emplacements de préchargement des experts. |
  | Cache des invites et points de contrôle | Invites enregistrées avec `--cache-ram` et points de contrôle de contexte des emplacements (RAM). |
  | Autre | Mémoire utilisée qu'aucune des catégories ci-dessus n'explique : autres processus, le bureau, contextes du pilote, projecteur multimodal. |
  | Libre | Le reste du périphérique ou de la RAM. |

  Les poids mappés depuis le fichier du modèle comptent comme RAM même lorsque le noyau signale ce cache de pages
  comme disponible ; les segments peuvent donc totaliser plus que la mémoire « utilisée » ; la barre n'affiche
  alors aucun segment « Autre » et moins de mémoire libre. Le cache d'experts apparaît après la première requête,
  lorsqu'il est créé. En mode routeur, les barres incluent tous les modèles chargés, et la page Supervision ne
  charge ni ne réveille jamais un modèle.

La **barre d'état** en bas de chaque page affiche d'abord l'utilisation de chaque GPU, puis celle de chaque
processeur physique, chacun dans sa propre barre, avec les mêmes étiquettes que les cartes de la page
Supervision : `gpu-1`, `gpu-2`... et `cpu-1`, `cpu-2`... Les barres ont la même taille, s'actualisent toutes
les deux secondes et, comme celles de la page Supervision, sont vertes et se remplissent de rouge à mesure que
l'utilisation augmente ; le texte de survol de chacune donne le nom du périphérique. Sans GPU, elle affiche une
barre `GPU` vide, et lorsque les barres ne tiennent pas sur un écran étroit, la barre d'état défile
latéralement. Cliquez dessus pour ouvrir la page Supervision. Elle continue de fonctionner pendant le chargement
du modèle. Elle est affichée par défaut ; désactivez Paramètres → Affichage → **Afficher la barre d'état** pour
la masquer.

<a id="models"></a>
## Modèles, outils et MCP

`mgs` ne charge que des modèles MoE. Au démarrage, il lit l'en-tête du GGUF passé avec `--model` (et de celui passé avec `--spec-draft-model`) ; s'il s'agit d'un modèle dense, comme GPT-2 ou Phi-2, `mgs` journalise une erreur comme celle-ci et se termine avec le code de sortie 1 avant de charger quoi que ce soit :

```text
srv    mgs_server: the model '/path/to/model.gguf' is not a Mixture of Experts (MoE) model
srv    mgs_server: mgs only loads MoE models; dense models are not supported
srv    mgs_server: exiting due to a dense model
```

En mode routeur, seul le processus de ce modèle se termine : `/api/v1/models` le marque comme en échec et les autres modèles continuent de fonctionner. Les implémentations de décodeurs denses ne sont pas compilées, mais les couches denses à l'intérieur d'un modèle MoE et une tête uniquement MTP compatible, fournie explicitement (`--spec-draft-model`), sont conservées.

<a id="settings-models"></a>
### Choisir et charger des modèles dans Paramètres → Modèles

Démarré sans `--model`, `mgs` fonctionne en mode routeur : il ne charge aucun modèle de lui-même et liste les modèles
qu'il trouve, afin que vous puissiez les charger depuis l'interface. Paramètres → Modèles comporte deux parties :

- **Dossier des modèles.** Appuyez sur **Choisir un dossier** : un explorateur des dossiers de l'ordinateur où s'exécute
  `mgs` s'ouvre sur le dossier des modèles actuel (ou sur votre dossier personnel). Cliquez sur un dossier pour
  l'ouvrir, utilisez la flèche pour remonter et la maison pour aller à votre dossier personnel, ou saisissez un chemin
  dans son champ et appuyez sur Entrée (un `~` initial désigne votre dossier personnel) ; **Afficher les dossiers
  cachés** ajoute les dossiers dont le nom commence par un point, et l'explorateur compte les fichiers `.gguf` du
  dossier qu'il affiche. Appuyez sur **Choisir** : le chemin absolu de ce dossier remplit le champ et est utilisé
  aussitôt. Le dossier peut contenir un fichier `.gguf` par modèle, ou un sous-dossier par modèle avec les parties d'un
  modèle découpé (`NAME-00001-of-0000N.gguf`) et son projecteur multimodal (`mmproj*.gguf`). `mgs` vérifie qu'il peut
  lire le dossier, l'enregistre avec les autres paramètres et l'analyse aussitôt ; chaque démarrage ultérieur analyse le
  même dossier. Pour le relire après avoir ajouté ou supprimé des fichiers, choisissez de nouveau le même dossier. Les
  fichiers GGUF denses, y compris les têtes MTP séparées, sont exclus de la liste, car `mgs` ne charge que des modèles
  MoE. Choisir un autre dossier décharge les modèles en cours d'exécution du précédent. Les modèles téléchargés dans
  `~/.cache/mgs/` sont également listés. Lorsque `mgs` a été démarré avec `--models-dir`, ce dossier prévaut : **Choisir
  un dossier** est désactivé et la page l'indique.
- **Modèles disponibles.** Chaque modèle avec son état (non chargé, en cours de chargement avec son pourcentage, chargé,
  en veille ou échec du chargement), son fichier et sa taille sur le disque, ainsi qu'un bouton pour le charger ou le
  décharger. Charger un modèle ici le sélectionne aussi dans le chat.

Les modèles chargés et la façon dont ils ont été chargés figurent sur la [page Modèle](#model-page), vers laquelle
renvoie la fin de Paramètres → Modèles. Le sélecteur de modèle du chat liste les mêmes modèles et charge celui que vous
choisissez ; tant que la liste est vide, il renvoie vers Paramètres → Modèles. Lorsque `mgs` est démarré avec `--model`,
Paramètres → Modèles indique seulement que le modèle ne peut pas être changé sans redémarrer `mgs`.

Un seul modèle est chargé à la fois : en charger un autre, ici, depuis le sélecteur de modèle du chat ou au moyen de
l'API, décharge celui qui était chargé (`--models-max N` en autorise N à la fois). Une nouvelle conversation utilise le
modèle chargé à ce moment-là. Une conversation conserve son propre modèle, celui de sa dernière réponse : lorsque vous
lui envoyez un message alors qu'un autre modèle est chargé, celui-ci est déchargé et le modèle de la conversation est
chargé en premier. Si le modèle de la conversation n'est plus là où il était (il a quitté la liste, ou c'est désormais
un autre fichier), une boîte de dialogue le signale et vous permet d'annuler le message, qui reste dans le champ, ou de
l'envoyer avec le modèle chargé.

Le dossier est conservé dans `~/.mgs/config/ui-settings.json` avec les autres paramètres de l'interface, en tant que
section `modelsFolder`. `mgs` l'écrit lorsque vous choisissez le dossier et le lit à chaque démarrage :

```json
{
  "modelsFolder": "/home/user/models"
}
```

Le service déployé (`deploy/`) démarre `mgs` avec `--models-dir /opt/moe-gguf-server/models` ; c'est donc ce dossier
qu'il liste.

<a id="model-page"></a>
### Page Modèle

**Modèle**, dans la barre latérale après **Supervision** (adresse `#/model`), affiche le modèle chargé dans deux
encadrés : **Fichier du modèle**, avec ce que contient son fichier GGUF (description, architecture, fichier, taille sur
le disque, nombre de paramètres, couches, couches MTP, contexte d'entraînement et projecteur multimodal), et
**Paramètres de chargement**, avec la façon dont il a été chargé : les paramètres que `mgs` a choisis de lui-même ou
que vous avez passés (taille du contexte, emplacements parallèles et contexte par emplacement, lot et micro-lot,
threads de génération et de lots, flash attention, types du cache KV, `--fit`, MTP, décodage spéculatif avec son
nombre maximal de jetons de brouillon, cache d'experts MoE et périphériques) et, en mode routeur, la ligne de commande
de son processus. Chaque état activé (flash attention, `--fit`, MTP, décodage spéculatif
et cache d'experts MoE) s'affiche en vert. Flash attention indique si `auto` l'a activée ; MTP indique combien de jetons
de brouillon il demande à chaque étape, ou qu'il est encore en train de les mesurer (0 à 3), ou que le brouillon n'est
pas rentable sur cet ordinateur ; et le cache d'experts MoE indique s'il a été créé, avec sa taille automatique ou
demandée : c'est la première requête après le chargement du modèle qui en décide, il l'indique donc jusque-là, et il
reste désactivé sur un GPU intégré ou lorsque tous les experts sont déjà en VRAM. Ces états en direct proviennent du
rapport de [Supervision](#monitor), qui contient aussi la mémoire utilisée par chaque périphérique, par usage. Tant
qu'aucun modèle n'est chargé, la page renvoie vers Paramètres → Modèles.

<a id="supported-models"></a>
### Familles de modèles prises en charge

| Famille | Architecture GGUF | Remarques |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP lorsque le GGUF le contient. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP lorsque le GGUF le contient. Sa table d'embeddings par couche (environ 29 Go) est lue à la demande depuis le fichier du modèle ; gardez donc le fichier sur un disque rapide. Les contextes longs sont calculés avec une attention dense sous la sélection parcimonieuse : le résultat est correct, mais sans l'accélération des noyaux de calcul parcimonieux. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP ; images avec son mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Les GGUF publiés ne contiennent pas de MTP. |
| DeepSeek-V4.1-Flash | `deepseek41` | Charge les deux formats publiés : les fichiers DwarfStar d'antirez (comme `antirez/deepseek-v4.1-flash-gguf`) et les fichiers de l'écosystème llama.cpp. Sa table Engram (environ 188 Gio dans le fichier) n'est jamais chargée : chaque jeton lit ses lignes depuis le fichier du modèle ; le fichier doit donc rester disponible pendant l'exécution de `mgs`, de préférence sur un SSD, et Linux ou un autre système POSIX est requis. Les fichiers sans modèle de conversation en utilisent un intégré, et `reasoning_effort` accepte un nombre de 1 à 100. Ni MTP ni images. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |
| Flint | `flint` | Les GGUF publiés ne contiennent pas de MTP. |

Gemma 4 et les modèles DeepSeek antérieurs à V4.1 ne sont plus pris en charge (voir [compatibilité supprimée](#removed-compatibility)). Pas plus que les autres architectures MoE héritées de llama.cpp, comme gpt-oss, Qwen3-MoE, Qwen3-Next, GLM-4.x, Granite 4, MiniMax-M2 ou Kimi-Linear.

Avec plusieurs GPU sélectionnés, MTP les vérifie tous avant de se replier sur la RAM. Son estimation de mémoire inclut la tête de sortie. Les tampons MTP intégrés se chargent avant les autres poids ; un GGUF MTP externe passé avec `--spec-draft-model` se charge avant le modèle principal, y compris avec `--fit off`, et active MTP de lui-même (voir [MTP automatique](#performance)) ; un MTP qui ne se trouve que dans un autre GGUF n'est jamais utilisé s'il n'est pas passé de cette façon. La sélection explicite actuelle des périphériques détermine toujours quels GPU sont disponibles.

Utilisez `--reasoning on` ou `--reasoning off` pour contrôler la réflexion. Quand la réflexion est désactivée (par cette option, par l'interrupteur de raisonnement du chat ou, dans une requête, par `enable_thinking: false` dans `chat_template_kwargs` ou `reasoning_effort: "none"`), `mgs` ferme aussi tout bloc de réflexion dès que le modèle l'ouvre : les modèles dont le modèle de conversation ignore ce réglage, comme Flint, répondent donc eux aussi sans réfléchir. Passer `enable_thinking` via `--chat-template-kwargs` est refusé. Utilisez les noms d'options documentés, avec des traits d'union ; les graphies avec des traits de soulignement sont refusées. Passez les listes sous la forme d'une valeur séparée par des virgules au lieu de répéter une option. Seul `--spec-type` reste répétable pour composer des modes spéculatifs.

Le sélecteur de modèle affiche les modèles disponibles, chargés et favoris. Les détails du modèle
comprennent la taille du contexte, les modalités, la quantification et le modèle de conversation. Le mode
routeur gère des processus enfants distincts pour les modèles chargés ; les requêtes identifient le modèle
par le champ ou le paramètre de requête `model`.

Les serveurs MCP sont configurés dans Paramètres → Serveurs MCP, par URL, avec des en-têtes d'autorisation facultatifs.
Vous pouvez inspecter les journaux de connexion, ainsi que les invites, les ressources et les outils disponibles. Le
proxy CORS facultatif du serveur doit être activé explicitement au moyen de l'option correspondante du
moteur ; il ne renvoie que du contenu inerte et refuse d'être ouvert comme une page (voir
[limites de sécurité](#security-limits)). Les restrictions du navigateur sur les ressources HTTP dans les pages HTTPS s'appliquent toujours.
Lorsque deux serveurs MCP activés proposent un outil portant le même nom, aucune des deux copies n'est
proposée au modèle et l'interface affiche un avertissement : désactivez tous ces serveurs sauf un pour
utiliser l'outil. Les outils intégrés du serveur et le bac à sable JavaScript du navigateur sont facultatifs.
Les approbations d'outils et la limite de tours agentiques restent disponibles dans l'interface. La limite de
tours (Paramètres → Agent → **Tours de l'agent**, le nombre maximal de cycles d'exécution d'outils
avant l'arrêt, pour éviter les boucles infinies) est de 50 par défaut.

Les flux peuvent se reconnecter à l'aide de l'identité et du curseur de la conversation enregistrés, tant
que le serveur conserve le flux. Un serveur arrêté ou un flux expiré ne peut pas garantir une relecture.
L'interface signale l'échec d'une reconnexion plutôt que de créer silencieusement une seconde requête de
génération.

### Serveur MCP de shell local

Activez **Serveur MCP local de shells** dans Paramètres → Serveurs MCP, ou démarrez `mgs` avec
`--local-shell-mcp-server`, pour permettre aux modèles du chat web d'utiliser le terminal de la machine où `mgs`
s'exécute :

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- L'interrupteur agit immédiatement, sans redémarrer `mgs`, et il est enregistré dans la section `localShellMcpServer`
  de `~/.mgs/config/ui-settings.json`, de sorte que les démarrages suivants le réactivent (`POST /api/mcp/shell/enabled`
  avec `{"enabled": true}` fait de même). Avec `--local-shell-mcp-server`, il reste activé et l'interrupteur est
  désactivé. Sans clé API, seul un navigateur de l'ordinateur où `mgs` s'exécute peut utiliser l'interrupteur, de
  sorte que personne d'autre sur le réseau ne peut y activer un shell ; avec une clé API, toute personne qui possède
  la clé le peut.

- Au démarrage, `mgs` vérifie quels shells sont installés et propose un outil pour chacun d'eux : `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) et `psh.run` (PowerShell, `pwsh`). Le journal liste les shells trouvés ; un shell
  qui n'est pas installé n'obtient aucun outil.
- L'interface le détecte, l'ajoute aux serveurs MCP sous le nom **local shell mcp server** et l'active. Vous pouvez
  l'y désactiver, ou le désactiver pour une seule conversation, et votre choix est conservé. Lorsque le serveur est
  désactivé, l'entrée disparaît.
- Chaque outil prend `command` et, facultativement, `cwd` (répertoire de travail ; par défaut, le répertoire
  personnel de l'utilisateur qui exécute `mgs`) et `timeout` (en secondes ; 60 par défaut, 3600 au maximum). La
  réponse est un objet JSON avec `exit_code`, `stdout` et `stderr`, plus `signal`, `timed_out`, `cancelled`,
  `timeout`, `seconds`, `shell` et `cwd` ; les sorties longues ne conservent que leur début et leur fin.
- Chaque commande s'exécute dans un nouveau processus, sans terminal et avec une entrée vide : `cd` et les
  variables ne sont pas conservés d'un appel à l'autre, et les programmes interactifs (éditeurs, questions) ne
  peuvent pas être utilisés. Lorsque le délai expire ou que la requête est annulée, tous les processus lancés par
  la commande sont tués. Les processus laissés en arrière-plan doivent rediriger leur sortie.
- L'interface demande l'autorisation avant chaque appel, sauf si vous choisissez de toujours autoriser l'outil.
- `sudo` fonctionne sans terminal (Linux) : lorsqu'il a besoin d'un mot de passe, une fenêtre s'ouvre au centre de
  la page, quelle que soit la page où vous vous trouvez, avec l'invite de sudo (`[sudo] password for user:`),
  l'outil et la commande qui le demande, un champ de mot de passe et, en dessous, les boutons **Accepter** et
  **Annuler**. Le curseur est déjà dans le champ du mot de passe : vous pouvez donc saisir le mot de passe
  immédiatement ; **Entrée** accepte et **Échap** annule ; un clic en dehors de la fenêtre ne la ferme pas. Le mot
  de passe va uniquement à sudo sur la machine où `mgs` s'exécute : il n'est pas enregistré dans la conversation,
  il n'est pas écrit dans les journaux et le modèle ne le voit jamais. Pendant que la fenêtre attend, le délai de la
  commande cesse de courir ; si vous appuyez sur **Annuler**, ou si personne ne répond dans les 5 minutes, sudo
  échoue. sudo redemande le mot de passe à chaque appel, car chacun est un nouveau processus, et jusqu'à trois fois
  lorsque le mot de passe est erroné ; la fenêtre avertit alors que le mot de passe précédent est peut-être erroné.
  Les commandes s'exécutent dans une session qui leur est propre, sans le terminal depuis lequel `mgs` a été
  démarré, de sorte que sudo ne demande jamais le mot de passe dans ce terminal. Seule l'interface web de `mgs` peut
  répondre : avec d'autres clients MCP, sudo échoue comme auparavant, faute de terminal. Le service déployé
  s'exécute sous le compte `moe-gguf-server`, qui n'a normalement ni mot de passe ni droit sudo ; cette fonction
  est donc destinée à `mgs` exécuté avec votre propre utilisateur (par exemple, la compilation dans `~/.local`).
  Lorsque `mgs` n'a pas de variable `DISPLAY`, les commandes reçoivent `DISPLAY=:mgs-askpass`, car sudo n'utilise
  son assistant de mot de passe sans terminal que lorsque `DISPLAY` existe ; les programmes qui tentent d'ouvrir
  une fenêtre échouent aussitôt.
- Dans le chat, chaque appel est affiché avec le nom de son outil, comme `bash.run`. Une fois déplié, il affiche
  jusqu'à trois encadrés gris en police à chasse fixe : **stdin** avec la commande seule, **stdout** avec la sortie
  et **stderr** avec le texte de l'erreur. stdout n'apparaît que lorsque la commande a affiché quelque chose et
  stderr uniquement en cas d'erreur ; le code de sortie figure à côté de leur titre, par exemple
  `stdout (code 2):` (à côté de stdin lorsqu'il n'y a pas de sortie). L'encadré stdin a une bordure blanche (gris
  foncé avec le thème clair), l'encadré stdout une bordure verte et l'encadré stderr une bordure rouge, avec tous
  les outils de shell (`bash.run`, `zsh.run`, `sh.run` et `psh.run`).
- Activez **Paramètres → Affichage → Développer automatiquement les commandes du shell local** pour ouvrir chaque
  appel de lui-même : chaque encadré apparaît dès que son contenu est connu, stdin pendant que le modèle écrit la
  commande, et stdout et stderr lorsque la commande se termine.
- Chaque appel à un outil de shell laisse son propre fichier dans `~/.mgs/local-shell-mcp-server/` (dans le
  répertoire personnel de l'utilisateur qui exécute `mgs` ; dans le service déployé,
  `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), nommé d'après la date et l'heure locales de l'appel à la
  milliseconde près, par exemple `y2026m10d12h23m59s59ms731.jsonl` ; `-2`, `-3`… sont ajoutés lorsque plusieurs
  appels démarrent dans la même milliseconde. Il s'agit d'un fichier JSONL, un objet JSON par ligne : d'abord la
  commande (`"event": "command"`, avec l'outil, le shell, le répertoire de travail et le délai), écrite avant son
  exécution ; puis toute la sortie, sans coupure, au fur et à mesure qu'elle arrive, une ligne par fragment
  (`"event": "stdout"` ou `"stderr"`, avec le texte dans `data`) dans l'ordre où la commande l'a écrite ; et enfin
  le résultat (`"event": "result"` : code de sortie, signal, si le délai a expiré ou si l'appel a été annulé, les
  secondes écoulées et les octets de stdout et de stderr) ou la raison pour laquelle la commande n'a pas pu démarrer
  (`"event": "error"`). Un appel refusé avant son exécution (un outil inconnu, pas de commande, un répertoire de travail qui n'existe
  pas, un délai non valide) laisse aussi son fichier, avec la commande telle qu'elle a été demandée et l'erreur. Le
  chat ne conserve que le début et la fin des sorties très longues, mais le fichier contient tout, et il contient
  déjà la sortie si `mgs` s'arrête pendant l'exécution de la commande. Pour lire tout le stdout d'un appel,
  exécutez `jq -j 'select(.event == "stdout") | .data'
  FILE`. JSON ne contient que du texte ; les octets qui ne sont pas de l'UTF-8 (sortie binaire) sont donc remplacés
  par `�` ; une commande qui affiche des gigaoctets laisse un fichier de cette taille. Seul cet utilisateur peut
  lire le dossier et les fichiers, car la sortie d'une commande peut contenir des secrets ; rien ne les supprime,
  alors effacez les anciens lorsque vous n'en avez plus besoin.

Dans le service déployé, ajoutez `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` à la section `[Service]` d'une
surcharge systemd (`systemctl edit moe-gguf-server`) et redémarrez `moe-gguf-server` ; sous Alpine, exportez la
même variable dans la configuration du service OpenRC.

**Sécurité.** Les commandes s'exécutent avec les droits de l'utilisateur qui exécute `mgs` (dans le service
déployé, le compte `moe-gguf-server`, jamais `root`). Toute personne qui peut ouvrir l'interface ou appeler l'API
peut les exécuter ; n'activez donc l'option que dans des environnements de confiance et protégez le serveur par une
clé API. Les requêtes provenant de pages d'autres sites sont refusées (l'origine CORS par défaut est localhost)
et le journal consigne chaque commande avec son code de sortie. Lisez la commande affichée dans la fenêtre du mot
de passe avant de saisir un mot de passe sudo : le mot de passe autorise cette commande. Le serveur ne le remet
qu'à un assistant de mot de passe dont le processus parent est sudo s'exécutant en tant que root pour votre
utilisateur, de sorte qu'une commande ne peut pas démarrer l'assistant elle-même pour lire le mot de passe.

<a id="api"></a>
## API HTTP

Les exemples utilisent 11443 ; remplacez-le par le port HTTPS de `ports.json` s'il est différent.

La racine par défaut de l'API est `/api`. L'interface reste à `/`. Les routes `/v1/...` à la racine du
serveur d'origine ne sont plus enregistrées. Les clients compatibles OpenAI doivent
utiliser `https://ai.example.com:11443/api/v1` comme URL de base : les routes OpenAI (`/api/v1/models`,
`/api/v1/chat/completions`, `/api/v1/responses`, `/api/v1/audio/transcriptions` et leurs variantes
`input_tokens`) n'existent que sous `/api/v1`, et la vérification de l'état de santé uniquement à `/api/health`.
`/api/rerank` et `/api/v1/rerank` restent tous deux pour les clients TEI/Jina, et les routes natives
(`/api/completions`, `/api/embeddings`…) sont des gestionnaires différents de leurs homonymes de `/api/v1`.
Les noms des champs des requêtes et des
réponses conservent la compatibilité avec l'amont ; les interfaces C/C++ du moteur conservent leurs fonctions
actuelles, mais pas celles qui étaient marquées comme obsolètes ou qui n'avaient aucune utilité (voir [compatibilité supprimée](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Définissez `MGS_API_KEY` avec la clé installée, dans votre propre shell (seul `curl` l'utilise ; `mgs` lit sa clé
dans `--api-key`, `MGS_ARG_API_KEY` ou `--api-key-file`). Les opérations natives, les
embeddings, la tokenisation, le reclassement (reranking), la gestion des modèles et les flux reprenables sont
documentés dans le Swagger UI fourni, à l'adresse `/api/doc/` ; son JSON OpenAPI se trouve à
`/api/doc/openapi.json`, généré par le serveur à chaque requête, sans fichier JSON statique. L'inventaire
reflète les routes enregistrées par le processus en cours d'exécution, y compris les routes du routeur et les
routes de compatibilité facultatives. Les fonctionnalités désactivées peuvent renvoyer 403. Une clé manquante
ou non valide renvoie 401 et le chargement d'un modèle renvoie 503. `/api/health` et
`/api/v1/models` restent publiques pour `GET` et `HEAD` ; voir
[limites de sécurité](#security-limits) pour les routes qui exigent toujours la clé.

`--api-prefix /api/custom` déplace les routes d'inférence sous `/api/custom`.
L'interface à la racine reçoit ce préfixe depuis `/api/config.js` ; la documentation reste à `/api/doc/`.
L'espace de noms de la documentation est réservé et ne peut pas servir de préfixe d'inférence. Les processus
enfants du routeur utilisent `/api` en interne. Le serveur valide les préfixes ; les valeurs hors de `/api/`
et les chemins terminés par une barre oblique sont refusés.

La page Supervision utilise `GET /api/monitor` (rapport complet), `GET /api/monitor/memory` (instantanés de la
mémoire des modèles chargés) et `GET /api/monitor/usage` (uniquement l'utilisation du CPU et du GPU, disponible
aussi pendant le chargement du modèle). Ces routes lisent des compteurs et des instantanés publiés par la boucle
d'inférence, jamais la file des tâches ; les interroger ne réveille donc pas un modèle en veille et ne
réinitialise pas son minuteur d'inactivité. Chaque modèle du rapport contient `expert_cache` (avec `active` et
`pending`) et, avec MTP, `mtp` : `draft_n` (les jetons de brouillon demandés à chaque étape, 0 lorsque le brouillon
n'a pas été rentable, -1 pendant leur mesure), `draft_n_max` et `measuring`. Leurs champs sont décrits dans Swagger.

L'interface lit ses paramètres avec `GET /api/ui/settings` (`{"settings": DOCUMENT}`, ou
`{"settings": null}` avant le premier enregistrement) et les enregistre avec `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, qui remplace entièrement `~/.mgs/config/ui-settings.json`, avec le mode 0600). Les
deux exigent la clé API et répondent aussi pendant le chargement du modèle.

Les conversations utilisent `GET /api/chats` (les conversations et leur `version`), `GET /api/chats/version`,
`GET /api/chats/search?q=WORDS&limit=N` (le message le plus récent de l'utilisateur ou de l'assistant de chaque conversation qui
contient chacun des mots, sans tenir compte de la casse ni des accents) et `GET`, `POST` et `DELETE /api/chats/{id_chat}`. Chaque
fichier commence par la conversation et continue avec un message par ligne ; les fichiers conservent les dates au format natif
`2026-11-03@14-59-59.723` (UTC) et l'API en millisecondes depuis l'époque Unix. Une conversation enregistrée sans numéro valide
reçoit le plus petit numéro libre. Elles exigent la clé API et répondent aussi pendant le chargement du modèle.

En mode routeur, `GET /api/models/folder` renvoie le dossier des modèles : `folder`, `source` (`cli` pour
`--models-dir`, `settings` pour la section `modelsFolder` de `~/.mgs/config/ui-settings.json`, ou `none`), `exists`,
`model_count`, `settings_file` et `cache_dir`. `POST /api/models/folder` avec `{"folder": "/absolute/path"}` (un `~`
initial désigne le dossier personnel ; `""` le supprime) vérifie que le dossier peut être lu, l'enregistre dans cette
section, analyse à nouveau les modèles et
renvoie le même objet ; il répond 400 lorsque le dossier a été défini avec `--models-dir` ou ne peut pas être lu.
`GET /api/models/folder/browse?path=/absolute/path` alimente l'explorateur de dossiers : il renvoie `path`, `parent`
(vide pour `/`), `home`, `folders` (les sous-dossiers triés par nom, chacun avec `name`, `path` et `hidden`) et
`gguf_files` (le nombre de fichiers `.gguf` situés directement dans le dossier) ; sans `path`, il liste le dossier des
modèles, sinon le dossier personnel, et il répond 400 lorsque le chemin n'est pas un dossier qui peut être lu. Les trois
exigent la clé API. En mode routeur, chaque entrée de `GET /api/v1/models` contient `path`, `file_size` (en octets,
toutes les parties d'un modèle découpé) et `mmproj`, et un modèle en cours d'exécution ajoute `meta` et `load`, la
configuration avec laquelle il a été chargé (`architecture`, `description`, `n_layer`, `n_layer_mtp`, `n_ctx`,
`n_ctx_slot`, `n_parallel`, `n_batch`, `n_ubatch`, `n_threads`, `n_threads_batch`, `flash_attn`,
`flash_attn_enabled` (si le contexte l'utilise, y compris avec `auto`), `cache_type_k`, `cache_type_v`, `fit`, `mtp`,
`speculative`, `draft_n_max`, `moe_cache_mib`, `devices` et `mmproj`). Avec `--model`,
l'unique entrée contient aussi `load`.

<a id="request-queue"></a>
### File d'attente des requêtes et requêtes parallèles

Par défaut, `mgs` traite une requête à la fois (`-np 1`) et lui donne tout le contexte, de sorte que chaque
requête peut utiliser tous les jetons du contexte, qu'elle provienne de l'interface web ou directement de l'API.
Les requêtes qui arrivent entre-temps attendent dans une file unique et sont traitées dans leur ordre d'arrivée :
les requêtes de l'interface ne passent pas devant celles de l'API, ni l'inverse, et une requête qui doit
attendre de nouveau conserve sa place. L'API se comporte de la même
manière quels que soient les paramètres de l'interface.

Pour traiter plusieurs requêtes en même temps, démarrez `mgs` avec `-np N` (`--parallel N`, `MGS_ARG_PARALLEL`) :
jusqu'à N requêtes s'exécutent en parallèle et le contexte est réparti entre elles, chacune en recevant une part
(`n_ctx_slot` dans la ligne `initializing` du journal) ; avec `--kv-unified`, elles partagent plutôt un cache KV
unique couvrant tout le contexte. Les requêtes au-delà de N attendent toujours dans la file.

L'interface web ajoute `"mgs_client": "webui"` à ses requêtes. Tant que l'une de ses requêtes en flux attend, le
flux commence par des événements tels que

```text
data: {"mgs_queue":{"ahead":2,"running_api":1,"running_webui":0,"waiting_api":0,"waiting_webui":1}}
```

(`ahead` : requêtes en cours d'exécution ou en attente devant elle ; `running_api` et `running_webui` : combien
d'entre elles sont en cours d'exécution et sont arrivées directement à l'API ou depuis l'interface web ;
`waiting_api` et `waiting_webui` : la même chose pour les requêtes qui attendent devant elle dans la file), puis un dernier événement avec tous
les compteurs à 0 lorsqu'elle démarre. Les clients qui n'envoient pas `mgs_client` ne reçoivent jamais ces
événements ; les clients compatibles OpenAI obtiennent donc le flux habituel. Swagger décrit le champ avec les
autres paramètres de complétion.

<a id="security-limits"></a>
### Limites de sécurité

Ces limites protègent le serveur et l'interface contre les requêtes, les fichiers et les modèles hostiles. Elles
sont assez généreuses pour un usage normal : ce qui les dépasse échoue avec une erreur claire au lieu d'arrêter
le serveur.

- **Clé API.** Les routes d'état de santé et de liste des modèles (`/api/health` et `/api/v1/models`) ne sont
  publiques que pour `GET` et `HEAD`. Le rechargement de la liste des modèles (`GET /api/v1/models?reload=1`) et l'ajout ou la suppression de modèles en
  mode routeur (`POST` et `DELETE /api/models`) exigent la clé dès qu'une clé est configurée.
- **Pages d'autres sites.** Par défaut (`--cors-origins` non défini, ce qui équivaut à `localhost`), seules les
  pages de `localhost` et de ce même serveur peuvent appeler l'API : le serveur répond 403 à toute requête autre
  que `GET`, `HEAD` ou `OPTIONS` dont l'en-tête `Origin` désigne un autre site, et le navigateur ne laisse pas les
  autres sites lire ses réponses. L'interface (y compris derrière Apache) et les clients qui n'envoient pas d'en-tête
  `Origin`, comme curl ou les SDK, ne sont pas concernés. Pour permettre aux pages d'autres sites d'appeler l'API,
  indiquez leurs origines dans `--cors-origins` (`*` admet n'importe quel site : configurez alors une clé API).
  `POST /api/tools` n'accepte que `Content-Type: application/json` (415 sinon).
- **DNS rebinding.** En HTTP simple (`--http`) sur un port TCP et sans clé API, le serveur répond 403
  (`Host not allowed`) à toute requête dont l'en-tête `Host` n'est ni une adresse IP, ni `localhost` (ou un nom se
  terminant par `.localhost`), ni le nom de la machine (y compris avec `.local`), ni le nom indiqué avec `--host`,
  de sorte qu'une page d'un autre site qui fait pointer son propre nom vers cette machine ne peut pas utiliser le
  serveur. HTTPS, une clé API, le socket Unix du déploiement et Vertex AI (`AIP_MODE=PREDICTION`) ne sont pas
  concernés. Pour joindre un serveur en HTTP simple par un autre nom, configurez une clé API.
- **Proxy CORS.** `/api/cors-proxy` refuse les requêtes qu'un navigateur ouvrirait comme page, cadre ou worker
  (403) et renvoie chaque réponse sous forme de contenu inerte : il ne peut ni exécuter de scripts ni définir de
  cookies sur l'adresse du serveur.
- **Listes du routeur.** En mode routeur, `status.args` et `status.preset` de `/api/v1/models` affichent
  `<redacted>` au lieu des valeurs de `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`,
  `--mcp-servers-config` et `--mcp-servers-json` ; le journal fait de même. Les processus des modèles reçoivent
  toujours les valeurs réelles.
- **Vertex AI.** Avec `AIP_MODE=PREDICTION`, le `@requestFormat` de chaque instance n'atteint que les routes
  d'inférence (completions, chat completions, responses, messages, infill, embeddings, rerank, tokenize,
  detokenize, apply-template et comptage de jetons) ; toute autre route, y compris la route de prédiction
  elle-même, répond par une erreur pour cette instance.
- **Téléchargement de modèles.** Le jeton Hugging Face (`--hf-token`, `HF_TOKEN`) n'est envoyé qu'au point de
  terminaison Hugging Face (`MGS_MODEL_ENDPOINT`, `https://huggingface.co/` par défaut) ; un modèle, un projecteur
  ou un modèle brouillon téléchargé depuis toute autre URL est demandé sans lui.
- **Clients lents.** Pendant que le serveur reçoit une requête, chaque lecture attend 30 s au maximum (`--timeout`
  régit toujours les écritures, de sorte que les longues réponses ne sont pas coupées), et au plus 256 connexions
  acceptées attendent un thread libre.
- **Corps des requêtes.** Les corps JSON imbriqués sur plus de 1024 niveaux de tableaux ou d'objets sont refusés
  avec 400. Les schémas JSON (`response_format`, `json_schema`) peuvent s'imbriquer jusqu'à 256 niveaux ; les
  valeurs `pattern` vides et les quantificateurs sans rien à répéter sont refusés.
- **Médias distants.** Les URL dans `image_url`, `input_audio` et `input_video` ne peuvent pointer que vers des
  adresses publiques d'Internet. Le bouclage, les réseaux privés (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16,
  100.64.0.0/10), les adresses link-local (y compris l'adresse de métadonnées cloud 169.254.169.254), les adresses
  IPv6 locales uniques et les autres adresses non publiques sont refusés avec 400, y compris lorsqu'une
  redirection y mène. Pour les autoriser, par exemple pour lire des images depuis un serveur de votre réseau,
  démarrez `mgs` avec `--media-allow-private-urls`
  (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`) ; ne le faites que si tous les clients de l'API sont de confiance. Chaque
  téléchargement dispose de 10 s pour se connecter et de 30 s au total, avec au plus 5 redirections et 10 Mio, et
  le serveur exécute au plus 8 téléchargements à la fois. Tous les médias d'une même requête peuvent contenir
  ensemble au plus 256 éléments et 512 Mio, et leurs téléchargements distants disposent de 120 s au total ; les
  fichiers lus avec `--media-path` comptent dans les mêmes limites et sont vérifiés avant d'être lus.
- **Médias décodés.** Images jusqu'à 32768 pixels de côté et 100 mégapixels, audio jusqu'à une heure, images vidéo
  jusqu'à 8192 pixels de côté et 2 Gio d'images décodées par vidéo, et au plus 2 Gio d'images, d'audio et d'images
  vidéo décodés par requête (y compris les images de toutes ses vidéos).
- **Projecteurs.** Un projecteur multimodal dont les métadonnées ne correspondent pas à son graphe (par exemple un
  facteur de fusion des patchs inférieur à 2 dans un projecteur qui fusionne les patchs) échoue au chargement avec
  une erreur au lieu d'arrêter le serveur.
- **Outils du serveur.** `read_file` avec une plage de lignes lit le fichier par blocs et ne conserve que ces
  lignes, en parcourant au plus 256 Mio du fichier ; sans `end_line`, les fichiers de plus de 16 Ko sont refusés.
- **Modèles de conversation du modèle.** `range()` produit au plus un million d'éléments, les chaînes construites
  restent sous 256 Mio, l'indentation de `indent()` et `tojson()` sous 256 espaces et l'imbrication sous 1024
  instructions ; un modèle de conversation qui les dépasse échoue avec une erreur de modèle de conversation.
- **Interface.** Les liens dans les messages ne conservent que les schémas http, https, mailto et tel (ou les
  adresses relatives), et les images jointes doivent être des images intégrées. L'import de conversations accepte
  des fichiers jusqu'à 512 Mo, des archives ZIP jusqu'à 10 000 entrées, 256 Mo par fichier JSONL et 1 Go extrait
  au total. Un fichier contenant des enregistrements mal formés, ou un message qui appartient à une autre
  conversation, est entièrement refusé ; une conversation dont les messages existent déjà,
  ou dont les messages se relient entre eux en boucle, est ignorée et signalée. Les images SVG écrites par le modèle
  restent dans leur propre cadre et ne peuvent pas recouvrir le reste de la page. La liste des ressources ou des
  templates de ressources d'un serveur MCP s'arrête après 100 pages, 10 000 éléments ou 60 s, ou lorsqu'un curseur se
  répète. Une ressource MCP est toujours lue, mise en cache et jointe depuis le serveur à partir duquel elle a été
  choisie, même lorsqu'un autre serveur propose la même URI. Une approbation **Toujours autoriser** d'un outil MCP
  s'applique tant que son serveur conserve la même adresse, le même transport, le même usage du proxy et les mêmes
  en-têtes : si l'un d'eux change, l'interface pose de nouveau la question.

Les commandes d'installation téléchargent toujours la branche `main` actuelle via HTTPS ; elles ne vérifient ni
signature ni révision fixe.

<a id="performance"></a>
## Mémoire et performances

L'enregistrement de la mémoire hôte et le préchargement des experts sont activés par défaut. Le préchargement
fait se chevaucher les transferts d'experts et le calcul, en particulier pour les grandes invites.
Les résultats dépendent du modèle, de la capacité mémoire et de la bande passante RAM/PCIe.

Pour les modèles dont les experts sont déchargés sur le CPU, `--cpu-moe` ou `--n-cpu-moe N` peuvent réduire
l'utilisation de la VRAM. L'enregistrement hôte épingle les pages de poids mappées pour accélérer les copies de
l'hôte vers le périphérique. La mémoire épinglée ne peut pas être évincée vers l'espace d'échange ; laissez donc
assez de RAM au système d'exploitation. Cette voie d'enregistrement est propre à POSIX ; Windows conserve son
comportement de chargement antérieur. `--load-mode none` n'est recommandé que lorsque l'enregistrement hôte
échoue totalement ou partiellement, et non comme une exigence normale.

`mgs` tient compte de la limite de mémoire de son cgroup (un service systemd avec `MemoryMax`, un conteneur)
en plus de la RAM libre. L'ajustement automatique ne garde les experts MoE en RAM que s'ils tiennent dans
cette limite, et place sinon des couches d'experts entières sur le GPU. Lorsque les experts restent en RAM,
le cache des invites (`--cache-ram`, 8 Gio par défaut) prend au plus la mémoire qui reste après eux et la
marge, ce que le journal signale par `prompt cache limited to N MiB`, et `--cache-ram` le fixe explicitement.
Seules les pages du modèle qui restent en RAM sont épinglées, et seulement tant que la mémoire encore disponible garde une marge (4 Gio ou 10 % de la RAM) ; au-delà, le journal affiche `pinning X of Y GiB of mapped weights` et le reste utilise de la mémoire paginable. Dans un cgroup de 30 Go avec le Qwen3.6-35B-A3B Q8_0 de 37,6 Go, la
version 0.1.17 restait bloquée pendant le chargement jusqu'à ce que le système la tue, désormais elle démarre
en quelques minutes et génère. Jusqu'à la version 0.1.22, avec des couches d'experts entières sur le GPU, `mgs` épinglait aussi les experts déjà copiés sur le GPU : sur un poste de travail avec peu de RAM libre et sans swap, tout le système pouvait cesser de répondre pendant le chargement du modèle.

Lorsque les fichiers du modèle sont plus grands que la mémoire encore disponible (ou que la limite du cgroup dans lequel il s'exécute), `mgs` ne les lit pas entièrement au chargement, ce qui ne ferait qu'évincer leurs propres premières pages et le cache de pages de tous les autres programmes : il lit ce qui va au GPU et ce qu'il épingle, et le reste à la première utilisation. Avec Qwen3.8-Flash-Next (113 Go) dans 40 Gio de mémoire et un disque SATA, le chargement est passé de plus de 5 minutes à environ une minute et demie.

Le préchargement des experts utilise par défaut trois emplacements en VRAM. Chaque emplacement doit pouvoir
contenir le plus grand tenseur d'experts transféré. Les valeurs 2 à 8 dans `GGML_SCHED_PREFETCH_EXPERTS`
choisissent explicitement le nombre d'emplacements. Si l'allocation mémoire échoue, au moins deux emplacements
alloués sont conservés lorsque c'est possible ; sinon, les transferts ordinaires sont utilisés. Le préchargement
ne démarre que lorsque le lot comporte au moins deux fois plus de sélections d'experts que d'experts. Son pool
appartient au premier GPU qui l'active ; les autres GPU conservent des copies sélectives.
Des lots plus grands peuvent aider le préremplissage, mais nécessitent plus de mémoire. Sur un [GPU intégré](#integrated-gpu), il n'y a pas de préchargement : remplir un emplacement serait une copie de RAM à RAM faite par le CPU, et les emplacements prendraient de la RAM au cache de pages des experts ; les experts utilisés sont donc copiés au besoin (avec Qwen3.8-Flash-Next sur une Radeon 860M, le prompt est passé de 15 à 33 tokens/s et la génération de 4,3 à 5,0 tokens/s).

Chaque emplacement prend la taille que le GPU alloue pour le plus grand tenseur d'experts, y compris le remplissage que CUDA ajoute aux lignes dont la longueur n'est pas un multiple de 512 éléments, et laisse `--moe-cache-reserve` (1024 MiB par défaut) libre pour la mémoire que le calcul occupe pendant son exécution ; si un emplacement ne tient pas, le préchargement continue avec ceux qui tiennent s'il y en a au moins deux, ou copie à la place les experts utilisés. Jusqu'à la version 0.1.31, les emplacements omettaient ce remplissage, si bien que Qwen3.8-Flash-Next, dont les experts ont des lignes de 640 éléments, pouvait s'arrêter à la première longue invite avec `CUDA error: an illegal memory access was encountered` (ou `an internal operation failed`) dès que le GPU avait de la place pour deux emplacements, par exemple avec MTP sur une RTX 4060 Ti. Avec Qwen3.8-Flash-Next sur la RTX 4060 Ti, MTP de 2 tokens traite désormais l'invite à 12 tokens/s et génère à 1,31 tokens/s.

Pour le diagnostic, définissez `GGML_CUDA_REGISTER_HOST=0`, `GGML_SCHED_PREFETCH_EXPERTS=0` ou
`GGML_VK_STAGING_RING=0` (transferts Vulkan via l'anneau de transit) pour désactiver l'optimisation correspondante.
Ne définissez pas ces variables en fonctionnement normal, sauf si vous choisissez délibérément de vous en passer.

Le MTP automatique examine les métadonnées GGUF et les noms des tenseurs avant l'allocation des poids,
y compris dans chaque fragment (shard). Pour le MTP intégré à `--model`, il nécessite une architecture prise en
charge, un tronc de modèle et toutes les têtes intégrées. Un GGUF MTP séparé passé avec `--spec-draft-model`
l'active aussi lorsqu'il déclare des têtes MTP complètes, sans tronc ; un modèle brouillon sans têtes MTP
conserve le type spéculatif que vous choisissez. Un MTP qui ne se trouve que dans un autre GGUF n'est jamais
recherché, et une architecture non prise en charge ne l'active pas. La limite de brouillon par défaut est de
deux jetons : avec MTP, deux jetons ont généré aussi vite que trois sur une RTX 4060 Ti et sur un GPU intégré
Radeon 860M, et un brouillon rejeté gaspille moins de travail de vérification. Toute sélection explicite avec `--spec-type` est prioritaire et désactive la sélection
automatique ; `--spec-type none` désactive MTP. `--spec-draft-n-max` change la limite. MTP agit sur la génération,
pas directement sur le préremplissage ni sur le temps jusqu'au premier jeton. L'amélioration dépend du taux
d'acceptation et du matériel ; augmenter la limite ne garantit pas plus de vitesse.

Que MTP accélère la génération dépend de la machine : chaque étape vérifie les tokens du brouillon avec le modèle entier, ce qui coûte peu quand les experts sont dans le GPU et beaucoup quand la plupart viennent de la RAM ou du disque. Quand MTP est activé automatiquement, `mgs` le mesure donc pendant les premières réponses : après une courte mise en température, il alterne étape par étape 0 (sans brouillon), 1, 2 et 3 tokens de brouillon jusqu'à ce que chacun ait généré 64 tokens, environ 500 tokens au total, garde le plus rapide si le brouillon est plus de 5 % plus rapide que sans brouillon, et l'indique dans le journal par une ligne telle que `MTP drafts 2 tokens from now on, the fastest on this machine`. Le résultat est enregistré dans `~/.mgs/tuning/mtp.json` par modèle, machine et version de `mgs`, et les démarrages suivants l'utilisent directement ; quand le brouillon n'était pas rentable, le journal indique `MTP drafting is not worth it on this machine` et, dès le démarrage suivant, MTP n'est pas chargé du tout, ce qui laisse aussi sa VRAM aux experts, au contexte et au micro-lot. Supprimer le fichier (ou son entrée) relance la mesure. `--spec-draft-n-max N` fixe le nombre de tokens de brouillon et désactive la mesure, `--spec-type draft-mtp` garde MTP activé, et un GGUF MTP séparé passé avec `--spec-draft-model` est toujours utilisé. Sur la Radeon 860M, Qwen3.8-Flash-Next a généré 5,12 / 7,09 / 7,34 / 6,97 tokens/s avec 0 / 1 / 2 / 3 tokens de brouillon et garde 2 ; sur la RTX 4060 Ti, il a généré 1,81 / 1,79 / 1,30 / 1,59, il fonctionne donc sans MTP dès le démarrage suivant, avec 23-24 tokens/s pour le prompt et 1,69-1,70 pour la génération au lieu de 12 et 1,4, tandis que Qwen3.6-35B-A3B y a généré 26,38 / 33,92 / 36,21 / 35,41 et garde 2.

<a id="mtp-vram"></a>
Les poids MTP restent toujours en mémoire GPU lorsque le GPU peut les contenir : les couches MTP greffées sur le
modèle, un GGUF MTP séparé passé avec `--spec-draft-model`, et la tête de sortie qu'ils utilisent. Les options qui
déplacent des poids vers la mémoire système (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, leurs équivalents
`--spec-draft-*` et l'ajustement automatique) ne s'appliquent pas à eux, et ils restent sur le GPU même avec une
valeur faible de `--gpu-layers`. L'ajustement automatique place le reste du modèle autour d'eux, et le
[cache d'experts MoE](#expert-cache) utilise la mémoire qui reste ensuite. Lorsqu'une couche MTP ne tient dans
la mémoire libre d'aucun GPU sélectionné, le journal affiche un avertissement (`no selected GPU has enough free
memory for MTP layer 0; using system memory`) et seule cette couche reste en mémoire système ; un tenseur dont le
GPU ne peut pas contenir le type y reste aussi. Sur un [GPU intégré](#integrated-gpu), la mémoire libre comptée
pour MTP est au plus ce que la RAM système peut encore fournir. Avec un GGUF MTP séparé, les couches MTP greffées
sur le modèle principal ne sont pas chargées, afin de ne pas occuper la mémoire deux fois ; avec
`--spec-type none`, aucun poids MTP n'est chargé. Le journal de chargement confirme le placement par des lignes
telles que `MTP layer 0 assigned to CUDA0 (856.36 MiB)`, visibles avec `-lv 4`.

<a id="expert-cache"></a>
### Cache d'experts MoE

Lorsque les experts MoE restent en mémoire système et que le reste de la couche s'exécute sur un GPU, le serveur
conserve les experts les plus récemment utilisés de toutes les couches dans un cache qui occupe la VRAM libre ;
avec plusieurs GPU, chaque GPU met en cache dans sa propre VRAM les couches qui lui sont attribuées. Pour chaque
jeton généré (et pour tout lot de moins de 32 jetons), les experts trouvés dans le cache s'exécutent sur le GPU,
une partie des experts manquants est transférée vers le cache et le reste s'exécute en même temps sur le CPU ;
cette part suit les bandes passantes PCIe et CPU mesurées. Les longues invites conservent le préchargement des
experts, qui copie désormais les experts en cache à l'intérieur du GPU et ne transfère que les autres. La sortie
correspond, aux arrondis près, à celle d'une exécution sans cache. Le GPU et le CPU arrondissent différemment et
les experts que chacun calcule dépendent du contenu du cache ; une longue génération gloutonne (greedy) peut donc,
d'une exécution à l'autre, dériver au bout de quelques jetons vers un texte différent mais tout aussi valide ; le
cache du llama.cpp officiel se comporte de la même manière.

Le cache est activé par défaut et il est créé par la première requête, une fois le modèle, le contexte MTP et le
projecteur multimodal chargés. Sans `--gpu-layers`, `--n-cpu-moe` ni `--override-tensor`, l'ajustement
automatique garde tous les experts en mémoire système lorsqu'ils y tiennent avec une marge (4 Gio ou 10 % de la
RAM) et laisse la VRAM libre au cache (avec plusieurs GPU, il répartit les couches entre eux selon leur mémoire
libre, afin que chaque GPU dispose d'un cache) ; sinon, il place des couches d'experts entières sur le GPU comme
auparavant, et le cache utilise ce qui reste. Les options de placement explicites sont respectées : `--cpu-moe`
donne au cache le plus de VRAM et `--n-cpu-moe N` garde certaines couches statiques sur le GPU. Les experts gardés en RAM sont épinglés tant que la mémoire encore disponible garde cette marge ; le surplus reste paginable, si bien que le système continue de répondre même quand la RAM manque.

Chaque couche garde sa propre part du cache, égale pour toutes, si bien qu'une rafale de défauts dans certaines
couches ne vide jamais le cache des autres, et dans chaque part les experts réutilisés sont protégés de ceux
qui ne servent qu'une fois. Un expert transféré dans le cache entre dans une zone d'essai et passe dans une
zone protégée (au plus la moitié de la part) lorsqu'il est réutilisé, et les nouveaux experts remplacent
d'abord ceux de la zone d'essai. Lorsque seule une partie des experts manquants peut être transférée, ceux
utilisés le plus récemment et le plus souvent passent en premier. Avec Qwen3.6-35B-A3B Q8_0 sur une RTX 4060
Ti, la génération est passée de 37,4 à 38,1 jetons/s (de 45,2 à 47,1 avec MTP) et les succès du cache de 72,3 à
74,1 %. Le partage garde aussi le cache utile lorsqu'une étape amène plus d'experts que tout le cache ne peut
en contenir, comme avec MTP sur des GPU avec moins de VRAM, où une seule liste partagée par toutes les couches
ne gardait presque rien.

<a id="integrated-gpu"></a>
Sur un GPU intégré (la partie graphique d'un processeur AMD ou Intel via Vulkan, ou NVIDIA Jetson et DGX Spark via
CUDA), la mémoire que signale le GPU fait partie de la RAM système. L'ajustement automatique compte les deux comme
un budget unique : le GPU peut utiliser au plus ce que la RAM peut encore fournir après les tampons qui restent sur
le CPU. Si le modèle entier y tient (en réduisant le contexte si nécessaire, comme sur un GPU dédié), tout
s'exécute sur le GPU ; sinon, des couches d'experts entières vont sur le GPU jusqu'à cette limite et le CPU
calcule le reste. Il n'y a pas de cache d'experts sur un GPU intégré, pas même avec `--cpu-moe` ou `--n-cpu-moe`,
car il copierait des experts de la RAM vers la même RAM et les conserverait en double. Le journal affiche le
budget dans une ligne telle que
`Vulkan0 is an integrated GPU that shares the system memory: N MiB usable` et la décision dans `no MoE expert cache on an integrated GPU`. Avec le modèle mappé (`--load-mode mmap`, la valeur par défaut), les poids qui restent sur le CPU quand toutes les couches sont sur le GPU (les tables d'embeddings des tokens et par couche, lues ligne par ligne) ne comptent pas dans ce budget, car seules les lignes lues restent en mémoire : Qwen3.8-Flash-Next, dont les embeddings par couche occupent 28 Go, reçoit toute la mémoire que déclare le GPU au lieu de ce qu'il en reste après eux. Sur un tel GPU, `mgs` se relance aussi une fois au démarrage avec `OMP_WAIT_POLICY=passive`, sauf si la variable est déjà définie, pour que les threads CPU dorment au lieu de tourner pendant que le GPU travaille : ils partagent avec lui la puissance du processeur, et avec des threads en rotation le GPU d'une Radeon 860M tournait à environ la moitié de sa fréquence. Le journal l'indique dans la ligne `the OpenMP threads of the CPU wait passively` ; `OMP_WAIT_POLICY=active` garde les threads en rotation.

<a id="context-size"></a>
Lorsque le modèle ne tient pas en VRAM et que `-c` n'est pas fourni, l'ajustement automatique choisit aussi la
taille du contexte : le cache KV occupe au plus un quart de la VRAM restante après les poids denses, jusqu'au
contexte d'entraînement du modèle, et les experts (le cache ou des couches d'experts entières) conservent le
reste. Auparavant, ces modèles obtenaient toujours le minimum de 4096 jetons partagés entre tous les emplacements,
ce qui tronquait les longues réponses (`truncated = 1` dans le journal). Le journal affiche le résultat dans une
ligne telle que `context size raised from 4096 to N tokens` et dans `n_ctx_slot` de la ligne
`initializing`. `-c N` définit explicitement le contexte (plus de contexte laisse moins de VRAM au cache et
réduit la vitesse de génération : 35,8 jetons/s avec 4096 jetons et 33,9 avec 122624 sur la machine de
référence) et `--fit-ctx N` relève le minimum. Le contexte MTP prend la même taille que le contexte du modèle au
lieu de réserver de la VRAM pour tout le contexte d'entraînement. Lorsque des couches d'experts entières remplissent le GPU parce que les experts ne restent pas en RAM pour le cache (toujours sur un [GPU intégré](#integrated-gpu)), le cache KV prend au plus un seizième, car chaque couche d'experts laissée hors du GPU est lue à chaque token dans la RAM, ou sur le disque quand le modèle dépasse la RAM : Qwen3.8-Flash-Next sur une Radeon 860M avec 46 Go de RAM reçoit 37376 tokens et génère 5,3 tokens/s, là où avec un quart il recevait 137216 tokens et générait 4,9 tokens/s.

L'ajustement automatique compte la mémoire du contexte MTP pour la taille de contexte qu'il choisit, et non pour le contexte d'entraînement du modèle : jusqu'à la version 0.1.32, il lui comptait 1300 MiB de VRAM avec Qwen3.8-Flash-Next sur une RTX 4060 Ti, où il occupe environ 160 MiB, et choisissait un contexte plus court pour le modèle. Avec Qwen3.8-Flash-Next et un MTP de 2 tokens sur la RTX 4060 Ti, l'ajustement compte désormais 130 MiB plus 4,57 KiB par token de contexte pour le contexte MTP et donne au modèle un contexte de 7936 tokens au lieu de 6912.

Avant le contexte, l'ajustement automatique choisit le micro-lot (`-ub`), le nombre de jetons de l'invite
traités en une passe. Avec les experts en RAM, chaque passe transfère vers le GPU les experts qui ne sont pas dans
le cache ; des passes moins nombreuses et plus grandes traitent donc les longues invites beaucoup plus vite,
tandis que la génération ne change pas. Sauf si `-b` ou `-ub` est fourni, il prend la plus grande des valeurs
4096, 2048 et 1024 jetons dont les tampons supplémentaires tiennent dans un huitième de la VRAM restante après
les poids denses, et relève le lot (`-b`) en conséquence. Sur la machine de référence, une invite de 2514 jetons
est passée de 214 à 755 jetons/s avec 4096, à vitesse de génération égale ; les tampons plus grands réduisent le
contexte par défaut (environ 90000 jetons sans MTP et 61000 avec MTP sur 16 Go). Le journal affiche le choix
dans une ligne telle que `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

Quand des couches d'experts entières remplissent le GPU, elles laissent sous la marge moins de mémoire que n'en demande le tenseur d'experts suivant, et rien d'autre ne l'utilise ; l'ajustement automatique augmente alors le micro-lot avec cette mémoire, si bien qu'aucun expert ne quitte le GPU. Le journal l'indique par une ligne telle que `micro-batch size raised from 2048 to 4096 tokens (batch 4096) with the device memory that the whole expert layers leave unused`. Que cela suffise dépend de la VRAM libre : avec Qwen3.8-Flash-Next et MTP sur une RTX 4060 Ti, ce n'était pas le cas avec la marge par défaut, mais ce l'était avec `--fit-target 800`. Lors de cette exécution, le prompt a été traité à 21-22 tokens/s au lieu de 12, avec une génération à 1,41-1,46 tokens/s (1,39 auparavant).

Avec un placement explicite (`--gpu-layers`, `--tensor-split`, `--override-tensor`, `--cpu-moe` ou
`--n-cpu-moe`), l'ajustement automatique le conserve tel quel et n'ajuste autour de lui que le contexte et le
micro-lot que vous n'avez pas définis, au lieu de s'arrêter ; le journal indique
`the device placement was set explicitly`.

| Option | Variable | Effet |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_ARG_MOE_CACHE` (`0` ou `false` désactive) | Active ou désactive le cache (activé par défaut). |
| `--moe-cache-size MiB` | `MGS_ARG_MOE_CACHE_SIZE` | Taille totale fixe du cache, répartie entre les GPU selon les experts que chacun met en cache ; `-1` (par défaut) utilise la VRAM libre de chaque GPU. |
| `--moe-cache-reserve MiB` | `MGS_ARG_MOE_CACHE_RESERVE` | VRAM laissée libre sur chaque GPU en plus du cache et des emplacements de préchargement (1024 par défaut). Augmentez-la lorsque d'autres programmes utilisent le GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnostic : part fixe (0–1) des experts manquants à transférer, à la place de l'équilibre mesuré. |
| — | `MGS_MOE_CACHE_STATS` | Diagnostic : consigne les compteurs du cache toutes les N étapes de décodage (visible avec `-lv 4`). |
| — | `MGS_MOE_CACHE_PROTECTED` | Diagnostic : part (0–1) de la part de chaque couche réservée aux experts réutilisés (0,5 par défaut), `0` donne un ordre LRU simple par couche. |

`GET /api/expert-cache` indique si le cache existe (`active`) ou si la requête suivante doit encore le créer ou
appliquer une nouvelle taille (`pending`), et renvoie la taille, les couches et les emplacements en cache, les succès et les échecs,
les transferts, les experts CPU, la part des octets de préchargement servis depuis le cache et les bandes
passantes mesurées (avec plusieurs GPU, les tailles et les compteurs s'additionnent et la bande passante de
transfert est leur moyenne). `POST /api/expert-cache` avec
`{"size_mib": N}` change la taille sans redémarrage (`-1` automatique, `0` libère le cache) ; la requête suivante
l'applique et démarre avec un cache vide :

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Avec `--metrics`, `/api/metrics` ajoute des compteurs et des jauges `mgs:moe_cache_*` (étapes, succès, échecs,
transferts, experts CPU, octets de préchargement, taille, emplacements et bandes passantes).

Mesure de référence sur une RTX 4060 Ti 16 Go avec Qwen3.6-35B-A3B Q8_0 (35 Gio, 8 threads CPU), comparée à
`--n-cpu-moe 28` sans cache : génération 26,9 → 35,7 jetons/s (+33 %), traitement de l'invite 767–778 →
718–730 jetons/s (−6 %), et avec MTP activé 33,6 → 47,0 jetons/s (+40 %). Sur la même machine, le llama.cpp
officiel avec son propre cache (`-cmoe --moe-cache-mib 10240`) a atteint 26,7 jetons/s, et 27,8 avec MTP. Les
gains dépendent de la part des experts qui tient en VRAM, de la localité du routage du modèle et des bandes
passantes PCIe et RAM.

<a id="threads"></a>
### Threads CPU

Le CPU calcule les experts qui ne sont pas dans le cache. Ce travail est limité par la bande passante mémoire et
sature avec peu de threads, tandis qu'utiliser tous les CPU logiques met les threads en concurrence avec celui qui
pilote le GPU. Sans `-t`, `mgs` utilise les cœurs utiles au calcul, mais jamais plus de la moitié des CPU logiques
sur lesquels il peut s'exécuter, sauf sur un CPU d'au plus 8 CPU logiques sans SMT, où il utilise tous les cœurs : avec
si peu de cœurs, c'est le CPU qui limite (sur un i7-7700K à 4 cœurs avec une RX 580, où le CPU calcule 40 des 41 couches
d'experts, 4 threads ont généré 17 % plus vite que 2). Sur une machine avec SMT (deux threads par cœur), le plafond
correspond au nombre de cœurs physiques, comme auparavant ; dans une machine virtuelle plus grande qui n'expose pas
le SMT, c'est la moitié des CPU virtuels. Sur la machine de référence (16 cœurs virtuels avec une session de bureau, Qwen3.6-35B-A3B Q8_0, sans
MTP), 4 à 12 threads ont généré à la même vitesse (32,9–33,9 jetons/s) et 16 threads à 28,7 jetons/s. Le journal
de démarrage affiche le choix dans une ligne telle que
`using 8 CPU threads for generation and 8 for batches` ; `-t` et `-tb` les définissent explicitement.

<a id="kv-cache-type"></a>
### Type du cache KV

Sauf si `-ctk` (`--cache-type-k`) ou `-ctv` (`--cache-type-v`) est indiqué, `mgs` donne au cache KV la précision des poids du modèle, d'après le type qui contient la plupart des poids de ses experts : les modèles non quantifiés (F16, BF16, F32) gardent un cache `f16`, ceux de plus de 5 bits par poids (Q5_K_M, Q6_K, Q8_0) reçoivent `q8_0` et les autres (Q4_K_M, IQ4_XS, MXFP4, Q3_K, Q2_K…) `q4_0`, les deux types quantifiés que flash attention lit sur tous les backends. La règle est la même sur tous les backends (CPU, CUDA, ROCm et Vulkan). Un cache quantifié exige flash attention : le cache reste donc en `f16` quand flash attention est désactivé (`-fa off`) ou ne peut pas fonctionner avec ce type sur le GPU. Le journal de démarrage indique le choix dans une ligne comme `KV cache type chosen from the weights of the model: q8_0`, et la page Modèle affiche les types utilisés.

Un cache `q8_0` occupe environ la moitié de la mémoire d'un cache `f16`, et un cache `q4_0` environ 28 % : l'ajustement automatique accorde ainsi beaucoup plus de contexte dans la même mémoire, ou laisse cette mémoire au cache d'experts, au prix d'une attention un peu moins précise et de quelques pour cent de vitesse. La lecture d'un cache quantifié demande un peu de travail supplémentaire sur tous les backends, et CUDA (GPU NVIDIA) et ROCm (GPU AMD avec une compilation `--rocm`), qui partagent les mêmes noyaux flash attention, ne le lisent directement que lorsque très peu de tokens sont traités à la fois (la génération normale, ni le prompt ni la vérification de plusieurs tokens de MTP) avec des têtes d'attention de 64, 128 ou 256 valeurs, et le convertissent sinon en `f16` à chaque étape. Sur une Radeon 860M (Vulkan) avec Qwen3.6-35B-A3B Q6_K et sans MTP, `q8_0` a généré 3,8 % de tokens par seconde en moins que `f16` après un prompt court et 2,7 % en moins après un prompt de 8060 tokens, et a traité ce prompt 1,3 % plus lentement. Si une seule des deux options est indiquée, l'autre reste en `f16`, et `-ctk f16 -ctv f16` rétablit le cache `f16` quand la mémoire suffit et que ces quelques pour cent comptent plus que le contexte. Le contexte MTP et le modèle brouillon de `--spec-draft-model` suivent la même règle ; `--spec-draft-type-k` et `--spec-draft-type-v` fixent leurs types.

<a id="checkpoints"></a>
### Points de contrôle du contexte

Les modèles dotés de couches récurrentes ou à fenêtre glissante (par exemple les hybrides Qwen3.5/3.6/3.8 ou
Nemotron-H) ne peuvent pas ramener leur état en arrière ; lorsqu'une requête modifie une partie
antérieure de la conversation, le serveur reprend donc à partir du point de contrôle le plus récent antérieur à
la modification. Les points de contrôle sont placés au début des messages de l'utilisateur, de l'assistant et des
outils (là où les agents de programmation suppriment l'ancien raisonnement ou raccourcissent d'anciens résultats
d'outils) et au moins tous les `--checkpoint-min-step` jetons (1024 par défaut, auparavant 8192), sans passe
supplémentaire sur l'invite. Lorsque `--ctx-checkpoints` (32) est atteint, le point de contrôle le plus proche de
ses voisins est supprimé, de sorte que les autres restent répartis sur toute la conversation. Dans le scénario
d'agent de référence (14 000 jetons, huit résultats d'outils), raccourcir un résultat d'outil tardif est passé de
19,3 s à 5,5 s, et un résultat précoce de 18,0 s à 13,6 s. Chaque point de contrôle conserve l'état récurrent ou
de la fenêtre glissante en mémoire hôte (63 Mio pour Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Options mises à jour et diagnostics

Conservez le `--load-mode mmap` par défaut lorsque l'enregistrement hôte réussit. N'utilisez `--load-mode none` qu'en solution de repli lorsque l'enregistrement échoue. `--load-mode dio` fait désormais transiter les lectures par au plus 64 Mio de mémoire supplémentaire ; il ne s'agit pas d'un plafond de la RAM totale du modèle.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` a été supprimée, car elle n'avait aucun effet à l'exécution. Les anciennes options de chargement (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) et l'option de compilation `GGML_CUDA_FA_ALL_QUANTS` ne sont plus acceptées ; voir [compatibilité supprimée](#removed-compatibility) pour leurs remplacements. Pour une compilation CMake manuelle, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` ajoute cette combinaison K/V de noyaux vectoriels ; les listes séparées par des virgules, ou par des points-virgules entre guillemets, sont acceptées. La liste par défaut est `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16` ; F16 est toujours incluse. `all` augmente le travail de compilation. Les combinaisons de noyaux vectoriels manquantes se replient sur une conversion F16 et peuvent être plus lentes.

MTP sépare les graphes avec et sans sorties, préserve l'ordre original des jetons entre séquences simultanées et utilise les positions réelles après les images.

L'ajustement automatique compte les couches nextn (MTP), de sorte que la première couche ne reste plus sur le CPU lorsque MTP est désactivé. Les poids MTP restent désormais toujours en mémoire GPU ; voir [placement de MTP](#mtp-vram). La valeur par défaut de `--checkpoint-min-step` est désormais 1024 au lieu de 8192 ; voir [points de contrôle du contexte](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` et `--moe-cache-reserve` contrôlent le [cache d'experts MoE](#expert-cache). L'ajustement automatique ne laisse plus les modèles plus grands que la VRAM avec un contexte de 4096 jetons et choisit désormais aussi le micro-lot ; voir [taille du contexte](#context-size). Le nombre par défaut de threads CPU est désormais d'au plus la moitié des CPU logiques ; voir [threads CPU](#threads). `-np` utilise désormais par défaut un seul emplacement disposant de tout le contexte, au lieu du mode automatique à quatre emplacements partageant un cache KV unifié, et `--kv-unified` est désactivé par défaut ; voir [file d'attente des requêtes](#request-queue). Sur un GPU intégré, l'ajustement automatique compte désormais la mémoire du GPU et la RAM comme un budget unique et ne crée plus le cache d'experts MoE ; voir [GPU intégrés](#integrated-gpu). Depuis la version 0.1.19, l'ajustement automatique ne s'arrête plus lorsque le placement est indiqué explicitement : il conserve ce placement et ajuste le contexte et le micro-lot autour de lui ; voir [taille du contexte](#context-size).

Avec `--metrics`, la route authentifiée `GET /api/metrics` inclut `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` et `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Les indices de position commencent à zéro ; cette série apparaît après la fin d'une requête spéculative. Comparez les deltas des compteurs de jetons acceptés et proposés pour évaluer l'acceptation, et mesurez la latence séparément ; l'acceptation seule ne prouve pas une accélération. En mode routeur, sélectionnez le modèle avec `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, désactivé par défaut) permet aux URL de médias des requêtes d'atteindre le bouclage, les adresses privées et les autres adresses non publiques ; voir [limites de sécurité](#security-limits).

Les exports de conversations lisent l'arbre complet des messages et les métadonnées persistés. Si les outils du serveur renvoient 403, les nouveaux messages cessent de répéter cette recherche échouée ; rouvrir le panneau des outils la relance. Les valeurs par défaut du serveur appliquées lors de la première visite préservent les paramètres déjà modifiés par l'utilisateur, y compris la clé API.

[Implémentation et sources amont](CODE.fr-FR.md#upstream-review).

<a id="removed-compatibility"></a>
## Compatibilité supprimée

Les couches de rétrocompatibilité ont été supprimées. Les anciens noms ne sont plus
traduits ni signalés par un avertissement : le serveur refuse les options de ligne de commande supprimées
comme arguments non valides, ignore les variables d'environnement supprimées et répond 404 pour les
routes supprimées. Mettez à jour les scripts, les surcharges de service, les préréglages et les clients avec
les remplacements ci-dessous.

| Supprimé | À utiliser à la place |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (ou `mmap+mlock` pour combiner les deux) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Rien ; l'option n'avait aucun effet. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Anciens alias de modèle brouillon tels que `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` ou `--draft-p-min` | L'option `--spec-draft-*` correspondante, par exemple `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` ou `--spec-draft-p-min`. `mgs --help` les liste toutes. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` ou `--spec-ngram-mod-n-max` / `--spec-draft-n-min` ou `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | L'option `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` et `--spec-ngram-*-min-hits` du mode n-gramme choisi |
| Variables d'environnement, options CMake et symboles de l'API C portant l'ancien préfixe amont, y compris les alias que des versions antérieures avaient déjà abandonnés | Le nom actuel avec le préfixe `MGS_` (`mgs_` dans l'API C), par exemple `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` au lieu d'une variante `NO_`, `MGS_ARG_LOAD_MODE` au lieu des variables de mappage mémoire, de verrouillage et d'E/S directes, `MGS_ARG_SPEC_DRAFT_N_MAX` et `MGS_ARG_SPEC_DRAFT_N_MIN` pour les limites du brouillon, ou `MGS_BUILD_UI` ; pour les options des backends, l'option `GGML_*` correspondante, par exemple `GGML_CUDA` ou `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `HF_HOME` et `MGS_CACHE` | Rien ; tous les téléchargements, y compris depuis Hugging Face, vont dans `~/.cache/mgs/` (sous `XDG_CACHE_HOME` lorsqu'elle est définie). |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` ou `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valeur `deepseek-legacy` de `reasoning_format` (champ de requête et `--reasoning-format`) | `deepseek` ou `auto` ; le raisonnement est renvoyé dans `reasoning_content` |
| Champ de requête `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (tableau de chaînes) |
| `reasoning_in_content` dans `generation_settings` | Rien ; le champ a été supprimé. |
| Téléchargement d'une interface amont précompilée depuis un bucket Hugging Face (les options CMake `USE_PREBUILT_UI` et `UI_HF_BUCKET` et la variable d'environnement `HF_UI_VERSION`) | Rien ; l'interface est toujours compilée depuis `frontend/` (`MGS_BUILD_UI=ON`) ou omise. |
| `GGML_CUDA_FA_ALL_QUANTS=ON` de CMake | `GGML_CUDA_FA_QUANTS=all` |
| Numéro de build et commit : les variables CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (aussi dans `mgs-config.cmake`) et la chaîne `b0-unknown` de `build_info`, de `system_fingerprint` et du `User-Agent` | `MGS_VERSION`, défini dans `backend/cmake/build-info.cmake` ; ces champs contiennent désormais la version, par exemple `0.1.19` |
| Modèles Meta Llama (les architectures GGUF `llama`, `llama4` et `llama-embed` et le projecteur de vision de Llama 4) et les modèles de conversation intégrés `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` et `llama4` | Rien ; utilisez l'un des modèles MoE pris en charge. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Option d'installation `--no-sudo` | Rien ; les installateurs s'exécutent toujours en root, sans sudo. |
| Fonctions et champs que les API C avaient marqués comme obsolètes : dans `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, les alias `mgs_token_*` (par exemple `mgs_token_bos`, `mgs_token_get_text` ou `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, le champ `defrag_thold` de `mgs_context_params` et `MGS_STATE_SEQ_FLAGS_SWA_ONLY` ; dans `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` et `ggml_upscale_ext` ; dans `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` et le champ `image_marker` | La fonction actuelle : `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` au lieu de `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` et `media_marker`. Les exécutions de préchauffage sont faites par l'appelant, et `defrag_thold` n'avait aucun effet. |
| Gemma 4 : les architectures `gemma4` et `gemma4-assistant`, les projecteurs `gemma4v`, `gemma4uv`, `gemma4a` et `gemma4ua`, le format de chat `peg-gemma4`, le tokenizer et le pré-tokenizer `gemma4`, et les préréglages `--vision-gemma-4b-default` et `--vision-gemma-12b-default` | Rien ; utilisez l'une des [familles prises en charge](#supported-models). |
| Modèles DeepSeek antérieurs à V4.1 : les architectures `deepseek`, `deepseek2` (y compris les fichiers Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 et Youtu convertis avec elle), `deepseek2-ocr` (avec les projecteurs `deepseekocr` et `deepseekocr2`), `deepseek32`, `mistral4` et `deepseek4`, les pré-tokenizers `deepseek-llm` et `deepseek-coder`, la dorsale de brouillon DSpark de DeepSeek-V4 et la détection du modèle de conversation de DeepSeek-V3.2 (DSML avec `function_calls`) | DeepSeek-V4.1-Flash (`deepseek41`). |
| Les autres architectures MoE héritées de llama.cpp : `grok`, `nomic-bert-moe`, `qwen2moe`, `qwen3moe`, `qwen3next`, `qwen3vlmoe`, `phimoe`, `minicpm`, `jamba`, `cohere2moe`, `dbrx`, `olmoe`, `arctic`, `glm4moe`, `exaone-moe`, `granitemoe`, `granitehybrid`, `bailingmoe`, `bailingmoe2`, `dots1`, `afmoe`, `laguna`, `ernie4_5-moe`, `hunyuan-moe`, `hy_v3`, `gpt-oss`, `lfm2moe`, `smallthinker`, `llada-moe`, `grovemoe`, `minimax-m2`, `rnd1`, `mistral3`, `step35`, `kimi-linear` et `mellum`, et les formats de chat de Ministral 3, gpt-oss, Functionary v3.2, Kimi-K2, Cohere2-MoE, LFM2, GigaChat3 et MiniCPM5 | Rien ; utilisez l'une des [familles prises en charge](#supported-models). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) et les modèles intégrés du moteur non Jinja (`--chat-template` avec un nom tel que `deepseek3`, `chatglm4` ou `mistral-v7-tekken`) | Rien ; les modèles de conversation passent toujours par Jinja. `--chat-template chatml` sélectionne toujours ChatML ; pour un autre modèle de conversation, passez sa source Jinja avec `--chat-template` ou `--chat-template-file`. Les requêtes avec `tools` n'ont plus besoin d'aucune option. |
| `-hfr` | `-hf` |
| Seconds noms longs d'une option : `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe` (les clés des préréglages du routeur suivent les mêmes noms) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, nommées d'après leurs options |
| Champ de requête `thinking_budget_tokens` | `reasoning_budget_tokens` (les clients Anthropic continuent d'envoyer `thinking.budget_tokens`) |
| `content` dans `POST /api/embeddings` et `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, la structure `mgs_chat_message` et `MGS_FILE_MAGIC_GGLA` dans `mgs.h` ; `mtmd_image_tokens_get_id` dans `mtmd.h` | Rien ; le formatage du chat est assuré par les modèles Jinja de `common/chat.h`, et les autres n'avaient aucune utilité. |
| `MGS_API_KEY`, `MGS_ARG_BATCH`, `MGS_ARG_UBATCH`, `MGS_ARG_N_PREDICT`, `MGS_ARG_N_PARALLEL`, `MGS_ARG_N_GPU_LAYERS`, `MGS_ARG_EMBEDDINGS`, `MGS_ARG_RERANKING`, `MGS_ARG_LOG_VERBOSITY`, `MGS_ARG_ROPE_SCALING_TYPE`, `MGS_ARG_SPEC_DRAFT_HF_REPO`, `MGS_ARG_STATIC_PATH`, `MGS_ARG_ENDPOINT_METRICS`, `MGS_ARG_ENDPOINT_PROPS`, `MGS_ARG_ENDPOINT_SLOTS` | `MGS_ARG_API_KEY`, `MGS_ARG_BATCH_SIZE`, `MGS_ARG_UBATCH_SIZE`, `MGS_ARG_PREDICT`, `MGS_ARG_PARALLEL`, `MGS_ARG_GPU_LAYERS`, `MGS_ARG_EMBEDDING`, `MGS_ARG_RERANK`, `MGS_ARG_VERBOSITY`, `MGS_ARG_ROPE_SCALING`, `MGS_ARG_SPEC_DRAFT_HF`, `MGS_ARG_PATH`, `MGS_ARG_METRICS`, `MGS_ARG_PROPS`, `MGS_ARG_SLOTS` : chaque variable est `MGS_ARG_` suivi du nom long de son option en majuscules, avec `_` au lieu de `-` |
| `MGS_MOE_CACHE`, `MGS_MOE_CACHE_MIB`, `MGS_MOE_CACHE_RESERVE_MIB` | `--moe-cache`/`--no-moe-cache`, `--moe-cache-size` et `--moe-cache-reserve`, ou `MGS_ARG_MOE_CACHE`, `MGS_ARG_MOE_CACHE_SIZE` et `MGS_ARG_MOE_CACHE_RESERVE` |
| `MTMD_BACKEND_DEVICE`, `MTMD_DEBUG_EMBEDDINGS` et l'option CMake `MTMD_VIDEO` | `MGS_MTMD_BACKEND_DEVICE`, `MGS_MTMD_DEBUG_EMBEDDINGS` et `MGS_MTMD_VIDEO` |
| `--spec-mtp-auto`, `--no-spec-mtp-auto` (`MGS_ARG_SPEC_MTP_AUTO`) | Rien : MTP s'active de lui-même ; `--spec-type none` le désactive et tout autre `--spec-type` le remplace. |
| `--spec-default` | `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, les valeurs qu'elle définissait |
| Les préréglages de modèles `--tts-oute-default`, `--embd-gemma-default`, `--fim-qwen-1.5b-default`, `--fim-qwen-3b-default`, `--fim-qwen-7b-default`, `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec`, `--fim-qwen-30b-default`, `--gpt-oss-20b-default` et `--gpt-oss-120b-default` | Rien ; passez le modèle avec `--model` ou `-hf`. |
| Les options des outils en ligne de commande que `mgs` n'inclut pas, et leurs variables, telles que `-p`/`--prompt`, `-f`/`--file`, `-sys`/`--system-prompt`, `-i`/`--interactive`, `-cnv`/`--conversation`, `--hellaswag`, `--winogrande`, `--kl-divergence`, `--save-all-logits`, `--ppl-stride`, `-o`/`--output`, `--chunk`, `--image`, `--audio`, `--video`, `-pps`, `-tgs`, `-npp`, `--diffusion-steps`, `-lr`/`--learning-rate`, `-mv`/`--model-vocoder`, `-hfv`/`--hf-repo-v`, `--tts-speaker-file`, `-fitp`/`--fit-print`, `--show-timings`, `--prompt-cache`, `--tensor-filter`, `-gan`/`--grp-attn-n` et les options de téléchargement `--mtp`, `--dflash` et `--eagle3` | Rien ; le serveur ne les a jamais utilisées. |
| Les valeurs `draft-eagle3`, `draft-dflash` et `draft-dspark` de `--spec-type`, avec les modèles brouillons Eagle3, DFlash et DSpark | `draft-mtp`, `draft-simple` ou l'un des types n-gramme |
| Les backends ggml autres que CPU, CUDA, Vulkan et ROCm (HIP) : BLAS, CANN, ET, Hexagon, Metal, MUSA, OpenCL, OpenVINO, RPC, SYCL, virtgpu, WebGPU, zDNN et ZenDNN, avec leurs options CMake (comme `GGML_BLAS`, `GGML_METAL`, `GGML_RPC` ou `GGML_SYCL`) et `--rpc` | CPU, CUDA (`--cuda`), ROCm (`--rocm`) ou Vulkan (`--vulkan`) |
| Dans `mgs.h`, `mgs_get_logits`, `mgs_get_embeddings`, `mgs_encode`, `mgs_set_causal_attn`, `mgs_model_has_encoder`, `mgs_model_has_decoder`, `mgs_model_decoder_start_token`, `mgs_model_is_recurrent`, `mgs_get_ctx_other` et le champ `ctx_other` de `mgs_context_params` ; dans `mgs-ext.h`, `mgs_set_embeddings_layer_inp`, `mgs_get_embeddings_layer_inp`, `mgs_model_target_layer_ids` et `mgs_model_target_layer_ids_n` ; dans `mtmd.h`, `mtmd_decode_use_non_causal`, `mtmd_image_tokens_get_n_tokens` et `mtmd_image_tokens_get_n_pos` | `mgs_get_logits_ith`, `mgs_get_embeddings_ith` ou `mgs_get_embeddings_seq`, `mgs_decode` (qui exécute aussi les modèles sans cache KV), `mtmd_input_chunk_get_n_tokens` et `mtmd_input_chunk_get_n_pos` ; les autres servaient aux modèles encodeur-décodeur, uniquement de plongements ou denses, ou au cache KV partagé entre contextes, et n'ont pas de remplacement. |
| Les projecteurs multimodaux de Gemma 3 (`gemma3`, `gemma3nv`, `gemma3na` et MobileNetV5) | Rien ; utilisez le projecteur de l'une des [familles prises en charge](#supported-models). |
| Les routes OpenAI sans `/v1` (`GET /api/models`, `POST /api/chat/completions`, `POST /api/responses`, `POST /api/audio/transcriptions`, `POST /api/chat/completions/input_tokens`, `POST /api/responses/input_tokens`) et `GET /api/v1/health` | La même route sous `/api/v1` (`GET /api/v1/models`, `POST /api/v1/chat/completions`…) et `GET /api/health`. `POST /api/models` et `DELETE /api/models` du mode routeur restent. |
| Les champs `ahead_api` et `ahead_webui` des événements `mgs_queue` | `running_api`, `running_webui`, `waiting_api` et `waiting_webui` ([file d'attente des requêtes](#request-queue)) |
| Les variables d'installation `MGS_INSTALL_SOURCE_DIR`, `MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`, `MGS_BUILD_JOBS` et `MGS_BUILD_BACKEND`, et l'option `--build-ui` des installateurs et de `build/build-local.sh` | `--build-dir`, `--prefix`, `--jobs` et `--cuda`, `--rocm`, `--vulkan` ou `--cpu` ; les sources vont toujours dans `/opt/moe-gguf-server-source` et l'interface est toujours compilée, sauf si `--no-ui` est indiqué. |
| La configuration des services web comme étape distincte après l'installation | L'installateur les configure dans la même exécution (`--domain`, `--certificate`, `--private-key`, `--proxy-ip`) ; `configure-web.sh` reste disponible pour reconfigurer sans recompiler. |
| Le service d'inférence sur `127.0.0.1:8080` (`--port 8080`) derrière Apache | Le socket Unix `/opt/moe-gguf-server/_/temp/web/mgs.sock` ([déploiement](#deployment)) |

CMake signale les variables `-D` supprimées comme inutilisées au lieu de les appliquer ; vérifiez donc
la sortie de configuration des compilations manuelles.

L'interface web conserve ses paramètres (`~/.mgs/config/ui-settings.json`) et ses conversations (`~/.mgs/chats/`) sur le serveur,
et dans le navigateur uniquement sa clé API (`MgsUi.apiKey` dans `localStorage`). Elle ne lit ni ne migre les
paramètres que les anciennes versions gardaient dans `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, les modèles favoris, les listes d'outils, l'effort de raisonnement,
les recommandations MCP ignorées, la langue et le mode de couleur de l'ancienne bibliothèque de thèmes) ; les paramètres repartent donc de
leurs valeurs par défaut et la clé API doit être saisie de nouveau une fois. Elle ne lit ni ne
migre pas davantage les données enregistrées par les anciennes versions sous d'autres noms, la clé autonome
`theme`, les marqueurs de raisonnement et d'appels d'outils intégrés aux anciens messages, les pièces
jointes de texte collé de l'ancien type `context`, l'ancien format de liste des recommandations MCP
ignorées ni l'ancien champ `webui_settings` du serveur. Elle ne lit pas non plus les conversations que les anciennes versions gardaient dans le navigateur (la base de données IndexedDB `MgsUi`) :
pour les conserver, exportez-les depuis l'ancienne version avant la mise à niveau et importez l'export ensuite, ce qui les
enregistre sur le serveur. L'import de conversations n'accepte que les exports JSONL et ZIP, dans les limites
décrites dans [limites de sécurité](#security-limits) ; l'ancien format d'export JSON est refusé. Les anciens
paramètres restent dans le navigateur jusqu'à ce qu'ils soient supprimés : pour les
effacer, ouvrez la console du navigateur (F12) sur la page de l'interface et exécutez
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
une fois pour chaque adresse et chaque port avec lesquels l'interface a été ouverte.

Tous les téléchargements sont mis en cache dans `~/.cache/mgs/` (sous `XDG_CACHE_HOME` lorsqu'elle est définie) :
les fichiers téléchargés avec `-mu`/`--model-url` et, en conservant leur propre structure
(`~/.cache/mgs/hub/models--<organization>--<repo>/`), les téléchargements depuis Hugging Face. Aucune
variable ne change ce dossier. Les fichiers que les anciennes versions enregistraient dans un autre dossier de
cache, y compris le cache de Hugging Face (`~/.cache/huggingface/hub`), sont téléchargés de nouveau.

### Anciens fichiers de modèles

Le code de compatibilité pour les anciennes conversions suivantes a également été supprimé.
Reconvertissez le modèle d'origine, ou téléchargez un GGUF actuel, pour les utiliser.

| Ancien fichier | Comportement actuel |
| --- | --- |
| Grok-1 converti avant août 2025 (sans `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` ni `grok.attn_logit_softcapping`) | Ne se charge pas : clé manquante. |
| Kimi-Linear avec le tenseur `blk.N.attn_kv_b` non scindé | Ne se charge pas : `attn_k_b`/`attn_v_b` sont absents. Cette voie échouait déjà lors de la construction du graphe. |
| Fichiers Qwen3-Next contenant uniquement le tenseur fusionné `ssm_in` | Ne se charge pas : tenseur `blk.0.attn_qkv.weight` manquant. |
| Fichiers sans `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) ou `full_attention_interval` (Qwen3-Next), et fichiers Cohere2-MoE avec une seule période SWA au lieu du tableau par couche | Ne se charge pas : clé manquante. |
| Vocabulaires BPE sans `tokenizer.ggml.pre` ou avec les noms de pré-tokenizer `llama3`, `llama-v3`, `jina-es`, `jina-de` ou `llada-moe`, et tokenizers Kimi-K2 sans fusions (merges) | Ne se charge pas : clé manquante, type de pré-tokenizer inconnu ou fusions manquantes. |
| Fichiers GGUF de version 1 ou 2 | Ne se charge pas : seule la version 3 de GGUF est acceptée. |
| Fichiers mmproj des anciens scripts ponctuels : types de projecteur `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x à 4.5, MiniCPM-o 2.6) et `adapter` (GLM-Edge-V), et fichiers sans type de projecteur | Le projecteur ne se charge pas : type de projecteur inconnu ou manquant. |
| Fichiers mmproj d'anciennes conversions avec `ffn_up` et `ffn_down` inversés | Ils ne sont plus corrigés ; le projecteur échoue. |
| mmproj de Kimi-K2.5 sans `image_min_pixels`/`image_max_pixels` et Step3-VL sans `preproc_image_size` | Le projecteur ne se charge pas : clé ou tenseur manquant. |
| Fichiers ne contenant que les anciennes clés `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` et `tokenizer.ggml.middle_token_id` au lieu des clés `tokenizer.ggml.fim_*_token_id` | Se charge ; les jetons d'infill sont reconnus par leur texte (par exemple `<PRE>` ou `<\|fim_prefix\|>`), et l'infill ne fonctionne pas s'ils ne sont pas trouvés. |

Les solutions de repli dont dépendent encore les conversions actuelles ou les GGUF publiés les plus
téléchargés des familles prises en charge ont été conservées, après vérification de leurs en-têtes sur
Hugging Face : la valeur par défaut de `add_bos` des pré-tokenizers `llama-bpe`, Tekken et
similaires ; la disposition des couches récurrentes de Qwen3.6 dérivée de
`full_attention_interval` (les GGUF de Qwen3.6 les plus téléchargés n'ont pas de
`recurrent_layers`) ; et la disposition intégrée de l'indexeur de GLM-5.2 pour les fichiers GLM-5.2/5.3
sans `attention.indexer.types`.

<a id="maintenance"></a>
## Maintenance et dépannage

- Vérifiez quelle version est installée avec `mgs --version` (la commande affiche, par exemple, `version: 0.1.57`) ;
  la compilation locale l'exécute à la fin et CMake affiche `mgs version: 0.1.57` pendant la configuration.
  L'interface affiche la même version dans Paramètres → Affichage →
  **Afficher les informations sur la version de build** (coin inférieur droit) et sous
  **Informations de compilation** dans la boîte de dialogue des informations sur le modèle. Chaque modification du projet incrémente le dernier nombre : 0.1.0, 0.1.1, 0.1.2, etc.
- Relancez `curl | bash` pour mettre à jour depuis `main`, ou la commande d'installation locale appropriée pour recompiler et remplacer le binaire.
  Passez de nouveau les mêmes options web (`--domain`, `--proxy-ip`…), car chaque exécution réécrit la
  configuration web. L'installateur redémarre les deux services à la fin ; pour redémarrer le service
  d'inférence à la main, exécutez `systemctl restart moe-gguf-server` sous Debian,
  ou `rc-service moe-gguf-server restart` sous Alpine.
- Consultez `/root/webapp-install.log`, `journalctl -u moe-gguf-server` sous Debian,
  ou `/var/log/moe-gguf-server.log` sous Alpine. Les journaux d'Apache se trouvent sous
  `/var/www/DOMAIN-logs/`.
- Si la compilation CUDA n'est pas disponible, installez le Toolkit séparément ou utilisez
  `--rocm`, `--vulkan` ou `--cpu`. Si NCCL est introuvable, vérifiez la disponibilité du dépôt et la version de CUDA,
  ou choisissez explicitement `--no-nccl`. Si ROCm est introuvable, installez-le avec ses fichiers de
  développement ou définissez `HIPCXX` ([ROCm](#rocm)).
- Si la compilation de l'interface échoue, vérifiez les versions de Node.js et de npm, les téléchargements des
  dépendances et le journal. Ne la remplacez pas par une interface précompilée d'amont.
- Pour les compilations de déploiement, utilisez un répertoire neuf dans `_/temp/` après une modification de
  l'organisation des sources. Une copie du projet sur SSHFS peut interdire les exécutables ; utilisez un disque
  local en production.
- Pour la compilation locale, relancez la commande d'une ligne ci-dessus pour télécharger la version actuelle de
  `main`, ou `bash build/build-local.sh` pour utiliser vos sources locales, puis redémarrez la copie que vous avez
  lancée à la main. Lisez la sortie du terminal si elle échoue ou signale des bibliothèques
  hors de `lib/`. Le fichier temporaire `build.log` est supprimé avec le répertoire de compilation.
- Les erreurs 404 de l'API dans les anciens clients nécessitent généralement de changer l'URL de base en
  `/api/v1`. La clé API installée doit figurer dans l'en-tête Authorization.

Voir [CODE.fr-FR.md](CODE.fr-FR.md) pour la carte de l'implémentation et les points d'extension.

<a id="license"></a>
## Licence

`mgs` est distribué sous la [PolyForm NonCommercial License 1.0.0](../LICENSE). Vous pouvez l'utiliser,
le modifier et le partager à toute fin non commerciale (NonCommercial) : usage personnel, études, recherche, projets de loisir,
ainsi qu'utilisation par des organisations caritatives, des établissements d'enseignement, des organismes publics
de recherche et des institutions gouvernementales. Un usage commercial, comme celui d'une entreprise qui l'utilise
pour son activité, nécessite une licence distincte accordée par nipegun.

Les parties qui proviennent de llama.cpp (MIT, The ggml authors), de FreeToken (Apache 2.0) et des
bibliothèques incluses conservent leurs propres licences ; leurs avis se trouvent dans `build/licenses/` et la
compilation locale les installe, avec `LICENSE`, dans `~/.local/share/doc/mgs/licenses/`.
