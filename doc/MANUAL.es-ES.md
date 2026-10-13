# Manual de uso

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Índice: [instalación](#installation), [compilación local](#local-build),
[despliegue](#deployment), [chat](#chat), [monitor](#monitor),
[modelos y herramientas](#models), [modelos admitidos](#supported-models), [API](#api), [cola de peticiones](#request-queue), [límites de seguridad](#security-limits), [Vulkan](#vulkan),
[rendimiento](#performance),
[compatibilidad eliminada](#removed-compatibility),
[mantenimiento](#maintenance), [licencia](#license).

<a id="installation"></a>
## Instalación y primer arranque

La instalación se realiza como root en el servidor de producción. Debian/Ubuntu
admite un script recibido directamente por la entrada estándar:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

No hace falta descargar primero los demás archivos. El script obtiene el archivo
de fuentes de `main` de [este repositorio](https://github.com/nipegun/moe-gguf-server/), valida que esté
completo y actualiza la copia gestionada en `/opt/moe-gguf-server-source`.
Las descargas temporales y las cachés se mantienen bajo `_/temp/` del proyecto.
La instalación inicial necesita curl y certificados CA para obtener el script;
las demás dependencias de descarga se instalan automáticamente si faltan.

Para elegir otra carpeta y pasar opciones:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

La carpeta remota debe ser absoluta y estar en un disco local. El instalador
rechaza sobrescribir una carpeta ajena: solo adopta una vacía o reutiliza una que
tenga su marcador de origen. Un bloqueo impide instalaciones simultáneas sobre
esa copia. Los siguientes usos de `curl | bash` sincronizan los fuentes, incluidos
los archivos eliminados de `main`, conservando `_/` y la caché de compilación.
La clave API, los modelos y los servicios desplegados viven fuera de esa copia
y no se modifican. Si falla una descarga o compilación, no se instala el nuevo
ejecutable; se puede repetir el mismo comando. No se necesitan operaciones Git.

Un script guardado sin sus archivos vecinos usa el mismo modo remoto. Si se
ejecuta `bash deploy/install-update-reinstall-debian.sh` desde un proyecto local
completo, instala sus fuentes actuales sin descargarlos. Las opciones después de
`bash -s --` llegan al instalador de compilación. `--help` no requiere root ni
descarga fuentes. `--no-system-deps` también impide instalar las dependencias de
descarga: curl, certificados CA, tar, gzip, rsync, util-linux y coreutils deben
estar disponibles previamente. Los procesos de instalación no consumen el script
que llega por la tubería y todo el registro utiliza `/root/webapp-install.log`.

La configuración web se realiza por separado, usando
`/opt/moe-gguf-server-source/deploy/configure-web.sh` o la ruta equivalente de
la carpeta elegida.

Alpine sigue usando una copia completa local y
`bash deploy/install-update-reinstall-alpine.sh`; necesita Bash instalado.
Debian usa CUDA y NCCL por defecto; Alpine utiliza CPU y no admite la instalación
CUDA/NCCL. `--cpu` selecciona CPU en Debian y `--vulkan`, GPU de AMD, Intel o NVIDIA mediante Vulkan
([Vulkan](#vulkan)). CUDA Toolkit y los controladores de NVIDIA deben estar instalados previamente. `--no-nccl` desactiva NCCL y
`--no-system-deps` deja la gestión de paquetes en manos del administrador.

La interfaz local necesita Node.js 20.19 o posterior en 20.x, o 22.12 o posterior.
`--no-ui` la omite. Otras opciones: `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST`, `--no-openssl` y `--build-ui`.
Las variables `MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`,
`MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND` y `CUDACXX` permiten ajustar estos valores.
Los directorios de compilación y copia de fuentes deben ser diferentes, no pueden
contenerse entre sí y deben permanecer dentro de `_/temp/`.

El instalador añade los paquetes NCCL compatibles y el repositorio de NVIDIA
si hace falta, pero nunca instala ni actualiza CUDA Toolkit ni los controladores
GPU. Enlaza estáticamente las bibliotecas internas e instala
`/usr/local/bin/mgs`. Ningún instalador usa ni instala sudo.

En Debian/Ubuntu, `libnccl2` y `libnccl-dev` se instalan juntos con la misma versión
exacta seleccionada para la versión de CUDA detectada. Esta orden de APT usa
`--allow-downgrades` para que un paquete NCCL instalado más reciente no bloquee
la instalación de la versión seleccionada. La opción solo se pasa a la orden
de instalación de NCCL.

Si la distribución ofrece un Node.js antiguo, instala una versión compatible
primero. La interfaz siempre se compila localmente; la compilación no puede descargar
los recursos precompilados originales, que no incluyen las traducciones ni las
rutas de API de este proyecto. Si falla la compilación de la interfaz, se detiene la
instalación.

Desde una copia local completa, elige CPU o consulta las opciones de instalación:

```bash
bash deploy/install-update-reinstall-debian.sh --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

En Alpine, instala Bash con `apk add --no-cache bash` antes de ejecutar su instalador.

Para ejecutar un modelo:

```bash
/usr/local/bin/mgs --model /ruta/al/modelo.gguf
```

No hace falta ninguna otra opción: `mgs` elige los ajustes que dan la mayor velocidad de inferencia en la máquina
en la que se ejecuta. El ajuste automático coloca el modelo en la GPU y, si la memoria del sistema lo permite,
mantiene los expertos MoE en la RAM para que la VRAM libre se convierta en la
[caché de expertos MoE](#expert-cache); también elige el [micro-lote y el tamaño del contexto](#context-size).
El número de hilos de CPU depende del procesador ([Hilos de CPU](#threads)), y las capas MTP que haya en el GGUF
se activan y se mantienen en la VRAM ([MTP](#mtp-vram)). Cualquier opción indicada de forma explícita (`-t`, `-b`,
`-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`, `--override-tensor`…) prevalece sobre la elección automática. El
registro de arranque muestra lo que se ha elegido.

Abre la URL HTTPS mostrada al arrancar, normalmente `https://127.0.0.1:11443/`.
Un modelo grande puede tardar en cargar. La interfaz
y la documentación siguen disponibles mientras las peticiones del modelo indican
que está cargando. `--help` enumera las opciones del motor. El GGUF, la memoria GPU,
el contexto y los proyectores necesarios deben ser adecuados para la tarea.

El código del servidor activa HTTPS por defecto con cualquier método de compilación
o instalación. Reserva el primer puerto HTTPS libre desde 11443 y después uno
HTTP desde 11080, excluyendo el HTTPS elegido. HTTP responde con una redirección
permanente 308 al puerto HTTPS real, conservando el host, la ruta y los parámetros.
El registro muestra las URL elegidas; cada serie avanza de forma independiente.
`--port N` fija el puerto principal y falla si está ocupado; `--port 0` deja que
lo elija el sistema. `--http-port N` cambia el inicio de la búsqueda del puerto
de redirección. Las variables equivalentes son `MGS_ARG_PORT` y `MGS_ARG_HTTP_PORT`.

Sin archivos de certificado, el ejecutable crea un certificado autofirmado y su
clave privada la primera vez que arranca, sin necesitar el comando externo
`openssl`, y los guarda en `~/.mgs/certificates/` (`fullchain.pem` y `privkey.pem`;
la clave solo la puedes leer tú). Las ejecuciones siguientes, y cada nueva
compilación, los reutilizan, así que el navegador conserva su excepción de confianza
y no hace falta recargar; solo se crea una pareja nueva si faltan esos archivos, no
coinciden o el certificado ha caducado (a los 365 días). El certificado cubre
`localhost`, las direcciones de loopback, el host indicado con `--host` y el nombre
y las direcciones de red que tenía el equipo al crearlo; para que el navegador deje
de avisar, importa `~/.mgs/certificates/fullchain.pem` como de confianza, y borra
los dos archivos para crear uno nuevo. Si esa carpeta no se puede escribir, cada
ejecución genera su propio certificado en memoria. Indica juntos `--ssl-cert-file
/ruta/fullchain.pem` y `--ssl-key-file /ruta/privkey.pem` para usar un certificado
en el que confíen los clientes. Una pareja
incompleta o inválida detiene el arranque. Una compilación con `--no-openssl` no
puede servir HTTPS y requiere `--http` explícito (o `MGS_ARG_HTTPS=false`);
HTTP sin TLS busca puerto desde 11080. Los hijos del router usan HTTP explícito
en puertos privados de loopback. Los sockets Unix y el modo Vertex AI
`AIP_MODE=PREDICTION` conservan su transporte HTTP.

La configuración de CMake está en `build/CMakeLists.txt`. Tanto los instaladores
como el script de compilación local incluyen `build/` al copiar los fuentes y lo
seleccionan automáticamente. Los instaladores regeneran la configuración de CMake
si una caché existente apunta a otro directorio de fuentes. La compilación local
siempre comienza en una carpeta nueva de `/tmp/` que se elimina al salir. Para
usar CMake directamente desde la raíz, ejecuta `cmake -S build -B _/temp/cmake`;
los comandos habituales de instalación y compilación local siguen siendo los mismos.
HTTPS usa por defecto el OpenSSL del sistema (`MGS_OPENSSL=ON`); en una compilación
directa con CMake, `-DMGS_BUILD_BORINGSSL=ON` o `-DMGS_BUILD_LIBRESSL=ON` descarga y
compila BoringSSL o LibreSSL en su lugar, y `-DMGS_OPENSSL=OFF` sin ninguna de las
dos compila el servidor sin HTTPS.

<a id="local-build"></a>
## Compilación local para uso manual

`build/build-local.sh` compila el proyecto en tu propio ordenador e instala una copia
autónoma para tu usuario que lanzas a mano. No instala servicios, no arranca nada en segundo plano,
nunca usa root ni sudo y nunca instala paquetes. Ejecútalo con tu usuario normal
desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

El comando descarga el archivo de fuentes de `main` sin Git en la carpeta privada
de compilación de `/tmp`. El archivo descargado y los fuentes extraídos se eliminan
con los demás temporales al salir. Una copia suelta del script funciona igual.
Desde una copia local completa del proyecto, compila tus fuentes locales con:

```bash
bash build/build-local.sh
```

Ejecutar este script mediante su ruta absoluta también utiliza el proyecto
completo que lo contiene. El mensaje `Using local sources: ...` confirma qué
carpeta se compila, incluidas sus traducciones de la interfaz. Si termina con
`Error: Compilation failed.`, no se ha instalado un ejecutable nuevo; la
instalación existente conserva su interfaz anterior. Lee el primer error del
compilador que aparece antes de `ninja: build stopped`, corrige su causa y
repite el mismo script. Cuando la compilación y la instalación terminen
correctamente, reinicia `mgs` y recarga la interfaz web.

Errores como que `std::vector` no tenga un miembro llamado `mPushBack` o
`no declaration matches` en `subproc.cpp` indican nombres inconsistentes en
los fuentes. Utiliza los archivos corregidos y repite la compilación. La
compilación de la interfaz también requiere que sus importaciones coincidan
con los nombres exportados por los módulos; npm/Vite avisa si falta alguno.

Si SvelteKit muestra `Page options are ignored when router.type === 'hash'`
y señala `fLoad`, las rutas del chat deben exportar el nombre exacto `load`
del framework. Las rutas corregidas conservan la validación de la clave de
API. La compilación ejecuta `svelte-kit sync` antes de generar los recursos
y compilar la interfaz, y se detiene si falla la sincronización. Un fallo en
`Provisioning UI assets` impide la instalación aunque las bibliotecas C++
ya se hayan compilado.

Para pasar opciones en el comando de una línea, utiliza `bash -s --`; por ejemplo,
para compilar para CPU:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` muestra las opciones sin descargar el archivo de fuentes ni compilar.

El resultado se instala para tu usuario en el prefijo `~/.local`:

| ruta | contenido |
| --- | --- |
| `~/.local/bin/mgs` | El programa que ejecutas: un pequeño lanzador estático que localiza `../lib/mgs/` en relación con su ubicación y arranca el servidor mediante el cargador incluido. |
| `~/.local/lib/mgs/` | El binario del servidor (`mgs`, con la interfaz web y la documentación de la API incorporadas) y todas las bibliotecas compartidas que necesita: el cargador dinámico y la biblioteca C de glibc, el runtime de C++, OpenMP, OpenSSL, los módulos de resolución de nombres de glibc y, en compilaciones CUDA, el runtime de CUDA, cuBLAS y NCCL. |
| `~/.local/share/doc/mgs/licenses/` | La licencia PolyForm NonCommercial 1.0.0 del proyecto (`LICENSE`) y los avisos de licencia de terceros copiados desde `build/licenses/` del proyecto (llama.cpp, FreeToken, Swagger UI, cpp-httplib y nlohmann/json). |

Sólo el usuario que ejecuta el script puede usar esta instalación. Todo se resuelve en relación con la ubicación del
lanzador: este ejecuta `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` desde su prefijo, y el servidor y sus
bibliotecas buscan en `lib/mgs/` antes que en cualquier otro sitio, incluido `LD_LIBRARY_PATH`. La única biblioteca
que se sigue tomando del sistema es la del controlador NVIDIA (`libcuda.so.1` y las bibliotecas `libnvidia-*` que
carga), porque debe coincidir con el módulo del kernel en ejecución; con `--vulkan`, el controlador de Vulkan del
sistema ([Vulkan](#vulkan)). Al terminar, el script enumera, mediante el
cargador incluido, las bibliotecas que cargaría el servidor, falla si alguna procede de fuera de `lib/mgs/` y ejecuta
`--version` a través del lanzador.

Debian añade `~/.local/bin` al `PATH` al iniciar sesión una vez que esa carpeta existe (mediante `~/.profile`), así
que tras volver a iniciar sesión puedes ejecutar `mgs` por su nombre; el script muestra un recordatorio mientras no
esté en el `PATH`. Lánzalo cuando quieras y detenlo con Ctrl+C:

```bash
mgs --model /ruta/modelo.gguf
```

También funciona con la ruta completa: `~/.local/bin/mgs --model /ruta/modelo.gguf`.

Abre `https://127.0.0.1:11443/` para el chat y `https://127.0.0.1:11443/api/doc/` para la
documentación de la API. Todas las opciones de este manual son válidas; por ejemplo,
`--models-dir` activa el router de varios modelos.

Requisitos, que debes instalar tú como root porque el script solo los comprueba:
`cmake`, un compilador C/C++ y la biblioteca C estática (`build-essential`, que
incluye `libc6-dev`), `patchelf`, `libssl-dev` (o
`--no-openssl`), Node.js 20.19+/22.12+ con npm (o `--no-ui`), un CUDA Toolkit ya
instalado (o `--cpu`) y NCCL con sus archivos de desarrollo (`libnccl2` y
`libnccl-dev`, o `--no-nccl`). `nvcc` se toma de `CUDACXX`, del `PATH` o de
`/usr/local/cuda/bin/nvcc`. Si Ninja está instalado, se utiliza.
Las utilidades básicas incluyen `tar`, `coreutils` y `util-linux`. La descarga de
fuentes también requiere `curl`, `ca-certificates` y `gzip`; ninguna dependencia
se instala automáticamente.

Opciones: `--prefix PATH` (otro prefijo de instalación), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`
y `--help`.

Conviene saber que:

- La compilación se optimiza para la CPU de este ordenador y para las GPU presentes
  al compilar. Recompila en cada ordenador en lugar de copiar la instalación.
- La instalación es reubicable como prefijo: mueve juntos `bin/mgs` y `lib/mgs/`, conservando sus niveles
  `bin/` y `lib/`. Arranca siempre `bin/mgs`; `lib/mgs/mgs` es interno y no debe ejecutarse directamente, porque se
  saltaría el cargador incluido.
- Repetir el script actualiza la instalación. Solo sustituye `bin/mgs`, `lib/mgs/` y
  `share/doc/mgs/licenses/`. Una copia que ya esté en ejecución sigue funcionando; reiníciala para usar la nueva
  compilación.
- El script rechaza un prefijo en el que ya existan `bin/mgs`, `lib/mgs/` o esa carpeta de licencias si no los creó
  él; deja un marcador oculto `.moe-gguf-server-bundle` en `lib/mgs/`.
- `--prefix PATH` instala en otro prefijo en el que tu usuario pueda escribir; no puede ser `/` ni una carpeta dentro
  del proyecto.
- Los fuentes copiados, los archivos de compilación, las cachés, la instalación
  preparada y el registro usan una carpeta privada
  `/tmp/moe-gguf-server-build.XXXXXXXX/`. Tras copiar el resultado al prefijo,
  el script elimina esa carpeta. También la limpia ante errores e interrupciones
  gestionadas. Cada ejecución compila desde cero, sin escribir en los fuentes ni
  en `/opt/`. La compilación local no admite `--build-dir`.
- El prefijo se bloquea durante la copia y sustitución de los archivos terminados
  para impedir actualizaciones simultáneas de la misma instalación.
- Una compilación CUDA necesita el controlador NVIDIA para arrancar; usa `--vulkan` en ordenadores con otra GPU
  y `--cpu` en ordenadores sin GPU.

<a id="vulkan"></a>
## Vulkan (GPU de AMD, Intel y NVIDIA)

Además de CUDA (la opción predeterminada) y CPU, tanto `build/build-local.sh` como el instalador de Debian compilan
el backend Vulkan con `--vulkan`, para GPU sin CUDA como las Radeon de AMD. Instala primero como root las
dependencias de compilación:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Después, con tu usuario normal:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

El instalador de Debian admite la misma opción e instala por sí mismo esos paquetes de Vulkan:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` muestra la GPU, por ejemplo `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- La instalación local incluye el cargador de Vulkan (`libvulkan.so.1`) como cualquier otra biblioteca, pero el
  controlador de Vulkan procede del sistema: el cargador lee los controladores registrados en
  `/usr/share/vulkan/icd.d/` (Mesa RADV para AMD, ANV para Intel, el controlador de NVIDIA) y estos cargan sus
  propias bibliotecas del sistema, porque deben coincidir con el núcleo y con el sistema. Compila en la misma versión
  de Debian que va a ejecutar el binario.
- El usuario que ejecuta `mgs` necesita el nodo de render de la GPU (`/dev/dri/renderD*`). Un usuario con sesión en
  el escritorio ya lo tiene; los demás necesitan el grupo `render` (y `video`). `configure-web.sh` añade a ambos la
  cuenta del servicio.
- MTP en VRAM, la caché de expertos MoE, la precarga de expertos, `--fit` y el monitor funcionan igual que con CUDA.
  En AMD, el monitor lee de sysfs el uso, la VRAM y la temperatura.
- `mgs` solo cuenta como memoria de GPU la VRAM real (los heaps de memoria «device-local»), nunca GTT. GTT es la
  parte de la RAM del sistema (normalmente la mitad) que una GPU de AMD o Intel puede mapear: no es memoria
  adicional, y la GPU la lee por PCIe, muchas veces más despacio que su VRAM. Las capas MTP, los pesos densos, la
  caché KV y la caché de expertos se reservan en VRAM; los expertos que no caben se quedan en la RAM normal, donde
  los calcula la CPU.
- `mgs` comprueba dónde guarda de verdad el núcleo esa memoria (`/proc/self/fdinfo`, que publican amdgpu, i915 y xe).
  Si otro programa llena la VRAM y el controlador mueve parte de ella a GTT, el registro y el monitor lo avisan, y el
  monitor muestra la ubicación real (VRAM y GTT) de cada GPU de AMD o Intel. La caché de expertos se reserva con una
  prioridad de memoria menor (`VK_EXT_memory_priority`), así que el controlador la mueve antes que las capas MTP, los
  pesos o la caché KV. Sin ReBAR, los búferes de dispositivo tampoco usan la pequeña ventana de VRAM que la CPU puede
  mapear (256 MiB en una RX 580), que comparte el escritorio y que el controlador vacía hacia GTT primero.
- Solo CUDA puede fijar en memoria las páginas del modelo que se quedan en la RAM, así que con Vulkan los expertos
  llegan a la GPU a través de memoria paginable. `--load-mode none` los copia a memoria fijada del host, lo que en una
  RTX 4060 Ti mediante Vulkan (Qwen3.6-35B-A3B Q4_K) subió la carga de expertos de 2,7 a 13,1 GB/s y la generación
  de 27,5 a 36,6 tokens/s. En AMD e Intel esa memoria fijada se descuenta de GTT: si los expertos no caben, la carga
  falla con `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), como les pasó a los 18,9 GB
  de expertos de ese modelo con 16,8 GB de GTT en una RX 580. Úsalo solo cuando los expertos quepan en GTT y en la
  RAM libre.
- Elige el modelo según la máquina: el GGUF entero debería caber en la RAM libre más la VRAM, descontando la caché
  KV y los búferes de cálculo. Con 8 GB de VRAM y 16 GB de RAM son unos 18–20 GB de pesos; los archivos mayores se
  leen del disco una y otra vez y la generación se vuelve muy lenta. El Q4_K de 21,6 GB anterior llegó a
  7,8 tokens/s en una RX 580 con 15 GB de RAM y a 15,9 tokens/s cuando la máquina tuvo 31 GB (10,0 con
  `--spec-type none`).

<a id="deployment"></a>
## Despliegue en producción

Desde la carpeta de fuentes (`/opt/moe-gguf-server-source` en modo remoto),
después de instalar el ejecutable:

```bash
bash deploy/configure-web.sh --domain ia.ejemplo.com
```

El dominio debe resolver al servidor de producción. El servicio
`moe-gguf-server-web` ejecuta Apache con la cuenta `moe-gguf-server`, sin root,
en una instancia independiente. En cada arranque selecciona el primer puerto
disponible desde 11080 para HTTP y desde 11443 para HTTPS. Si están ocupados,
incrementa cada serie por separado hasta encontrar un puerto libre, sin superar
65535 ni asignar el mismo puerto a ambos protocolos. Comprueba los listeners TCP
IPv4/IPv6 y reintenta si Apache detecta que un puerto se ha ocupado entre la
selección y la apertura real. Otros errores de configuración detienen el arranque.

HTTP redirige permanentemente al HTTPS seleccionado, conservando ruta y consulta.
La redirección no se almacena en caché. Apache reenvía las peticiones a la
inferencia en `127.0.0.1:8080`, arrancada explícitamente con `--http --port 8080`,
cuyo usuario también es `moe-gguf-server`.
La instalación requiere root; ninguno de esos servicios necesita ejecutarse como
root. No se configura HAProxy ni se importan ni modifican los sitios del Apache
compartido.

Las URL efectivas se publican después de abrir ambos listeners:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

Por ejemplo, HTTP 11081 y HTTPS 11445 producen una redirección a
`https://ia.ejemplo.com:11445/`. El archivo desaparece al detenerse el servicio
y se regenera al arrancar; un reinicio puede elegir otros puertos. Cualquier
cliente, cortafuegos o proxy externo debe utilizar los valores publicados.
El puerto forma parte del [origen del navegador](https://developer.mozilla.org/en-US/docs/Glossary/Origin):
si cambia HTTPS, el historial y los ajustes locales del origen anterior siguen
separados. La exportación/importación permite trasladar las conversaciones.

La cuenta de sistema no tiene contraseña interactiva. Los modelos están en
`/opt/moe-gguf-server/models`, y la configuración del programa vive en la carpeta personal de la
cuenta del servicio, `/opt/moe-gguf-server/.mgs/config/`: la clave API (`api.key`), el dominio
web (`web-domain`), la plantilla de Apache (`apache2.conf.in`) y los ajustes de la interfaz
(`ui-settings.json`), todos propiedad de `moe-gguf-server` y con modo 0600.
Las credenciales están en `/root/webapp-credentials.txt` y el registro de
instalación en `/root/webapp-install.log`, ambos con permisos 600. La
reconfiguración conserva claves, modelos y certificados. La clave TLS tiene
permisos 640, con propietario root y grupo moe-gguf-server, para permitir
el arranque sin privilegios.

Para utilizar un certificado propio:

```bash
bash deploy/configure-web.sh --domain ia.ejemplo.com \
  --certificate /ruta/fullchain.pem --private-key /ruta/privkey.pem
```

El certificado se guarda en `/opt/moe-gguf-server/certificates/` (`fullchain.pem` y
`privkey.pem`, propiedad de la cuenta `moe-gguf-server`, que es la única que puede
leer la clave). Si no aportas esos archivos, se conserva el certificado que ya está
ahí o, la primera vez, se crea uno autofirmado, que reutilizan todas las
reinstalaciones siguientes, así que el navegador conserva su excepción de confianza.
Configura su confianza en el cliente o proporciona un certificado que ya sea de
confianza. Los archivos aportados se copian a esa carpeta; al renovarlos hay que
actualizar las copias y reiniciar `moe-gguf-server-web`. El script no obtiene ni renueva certificados Let's Encrypt.
Repite el comando después de renovar los archivos originales. `--binary` permite
indicar otra ubicación del ejecutable.

El servicio funciona en modo router. Guarda GGUF legibles en
`/opt/moe-gguf-server/models` o utiliza la gestión de modelos de la interfaz.
Para opciones específicas, utiliza un override de systemd en Debian/Ubuntu o la
configuración OpenRC en Alpine. Mantén el backend en la interfaz de loopback.
El configurador web regenera su archivo de servicio al volver a ejecutarse.

Para reiniciar la web y volver a buscar puertos:

```bash
systemctl restart moe-gguf-server-web
```

En Alpine: `rc-service moe-gguf-server-web restart`. Para los errores de arranque,
consultar `journalctl -u moe-gguf-server-web` en Debian o
`/var/log/moe-gguf-server-web.log` en Alpine. El diagnóstico de Apache está en
`/opt/moe-gguf-server/_/temp/web/startup.log` y `error.log`; los registros de
acceso del dominio siguen en `/var/www/DOMINIO-logs/`. La plantilla independiente
está en `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. No se debe editar el archivo
generado dentro de `_/temp/web/`, porque se reconstruye en cada arranque.

<a id="chat"></a>
## Uso de la interfaz

1. Abre la URL e introduce la clave API cuando se solicite o en Ajustes → General.
2. Selecciona un modelo. En modo router, carga uno disponible si no hay ninguno cargado.
3. Escribe un mensaje y envíalo. Intro envía y Mayús+Intro inserta un salto de línea
   de forma predeterminada; puedes cambiarlo en los ajustes.
4. Adjunta archivos. Imágenes, audio y vídeo requieren modalidades compatibles.
   Los PDF pueden extraerse como texto o convertirse en imágenes; los modelos
   sin visión utilizan texto.
5. Puedes detener la generación, omitir el razonamiento cuando sea compatible,
   editar mensajes, regenerar respuestas o bifurcar conversaciones.

La barra lateral permite buscar, fijar, renombrar, eliminar y seleccionar varias
conversaciones. Las conversaciones se guardan en el navegador mediante IndexedDB;
expórtalas antes de borrar los datos del navegador o cambiar de perfil. Los ajustes
los guarda el servidor en `~/.mgs/config/ui-settings.json` del usuario que ejecuta
`mgs` (en el servicio desplegado, `/opt/moe-gguf-server/.mgs/config/ui-settings.json`),
así que todos los navegadores que abren la interfaz los comparten: parámetros de
generación, opciones de visualización, tema, idioma, servidores MCP, modelos
favoritos, herramientas desactivadas y siempre permitidas y esfuerzo de razonamiento
predeterminado. Solo la clave API se queda en cada navegador, porque autentica la
petición que carga el resto, así que hay que introducirla una vez en cada navegador.
Los ajustes incluyen importación y exportación. Las exportaciones pueden contener adjuntos y, si lo
seleccionas expresamente, claves API o cabeceras de autorización MCP.

El idioma se elige en Ajustes → General. Los 15 idiomas aparecen en este orden:
`de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`,
`ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; el predeterminado es `en-US`.
El portugués de Brasil y el de Portugal tienen traducciones independientes.
En hebreo la interfaz se muestra de derecha a izquierda; la entrada del chat y
los mensajes de usuario sin Markdown siguen la dirección de su propio texto,
mientras que el código y las salidas de terminal se muestran de izquierda a derecha.
El cambio guarda la selección en los ajustes del servidor y recarga la
interfaz; el servidor la publica en `/api/config.js`, así que todos los navegadores
arrancan en ese idioma. Guarda primero cualquier otro ajuste pendiente. El parámetro
`lang` de la URL lo sustituye en esa página y conserva la elección cuando no se puede
guardar.
Por ejemplo, añade `?lang=ja-JP` a la URL de la interfaz para elegir japonés.

Ajustes → General → **Responder siempre en el idioma de la interfaz** viene
desactivado por defecto. Actívalo y guarda los ajustes para indicar al modelo
que use el idioma y la variante regional de la interfaz en cada nueva
respuesta, también en conversaciones existentes, regeneraciones,
continuaciones, turnos después de usar herramientas y títulos generados por
el modelo. Para `es-AR`, la instrucción exige voseo argentino: «Seguí estos
pasos», «tenés» y «podés»; `es-ES` pide los usos del español de España. También
se distinguen el inglés británico y estadounidense y el portugués de Brasil
y de Portugal.

La instrucción se incluye en cada nueva petición de inferencia y preparación
de la caché KV; se combina con el mensaje de sistema enviado sin editar el
prompt ni la conversación guardados. Cambiar el idioma de la interfaz afecta
a las nuevas peticiones tras la recarga; una respuesta que ya se esté
generando conserva la instrucción que recibió. Al desactivar el ajuste deja
de añadirse a las peticiones siguientes. Las respuestas existentes no se
traducen. El código, los comandos, los nombres, las citas y las traducciones
pedidas expresamente conservan el idioma necesario; las explicaciones que
los acompañan siguen el idioma de la interfaz. El grado de cumplimiento
depende del modelo: no es una traducción automática ni una garantía de
obediencia perfecta. Este ajuste del navegador no afecta a clientes externos
de la API.

Los temas son sistema, claro y oscuro. Los ajustes incluyen temperatura,
muestreadores, penalizaciones y límites de turnos del agente. Algunas funciones
dependen de opciones del servidor o del modelo seleccionado.

Después de cada respuesta, la interfaz vuelve a enviar la conversación, si el
servidor está libre, para que ya esté codificada en la caché KV mientras la lees y
el siguiente turno empiece antes. Está activado por defecto; desactiva Ajustes →
Desarrollo → **Preparar la caché KV después de cada respuesta** para evitarlo.

Mientras el servidor resuelve otra consulta, el mensaje que envíes espera en la
[cola](#request-queue), y su respuesta muestra un aviso en lugar del texto de progreso hasta
que empieza: «En cola: antes se está resolviendo otra consulta enviada directamente a la API
(consultas por delante: N).» cuando un programa usa la API al mismo tiempo, o el mismo aviso
para una consulta de la propia interfaz web (otra pestaña u otro navegador, un título generado
o la preparación de la caché KV). El aviso desaparece en cuanto empieza tu consulta.

<a id="monitor"></a>
## Monitor y barra de estado

**Monitor** está debajo de **Ajustes** en la barra lateral (dirección `#/monitor`). Mientras está abierto se
actualiza cada dos segundos y muestra:

- **Uso**: primero una tarjeta por cada GPU del equipo (NVIDIA, AMD o Intel, aunque ningún modelo la use), titulada
  con su etiqueta en negrita y su nombre (por ejemplo **gpu-1**: NVIDIA GeForce RTX 4060 Ti; con varias GPU, gpu-1,
  gpu-2, etc.), con su uso, el uso del controlador de memoria (ancho de banda), la VRAM en uso, la temperatura y el
  consumo frente a su límite, y después una tarjeta por procesador, titulada de la misma forma con su modelo (por
  ejemplo **cpu-1**: AMD Ryzen 9 5950X 16-Core Processor; con varios procesadores, cpu-1, cpu-2, etc.), con su uso,
  una barra vertical pequeña por cada CPU lógica de ese procesador y la carga media del sistema. Todas las barras de
  uso son verdes y se llenan de rojo a medida que aumenta el uso; una barra cuyo valor se desconoce se queda gris.
  Las GPU de NVIDIA se leen mediante la biblioteca NVML del controlador y las de AMD mediante `gpu_busy_percent` en
  sysfs; cualquier otra GPU (Intel, o NVIDIA sin NVML) no muestra su uso, y su nombre y su memoria llegan del
  backend de inferencia en cuanto un modelo la usa.
- **Memoria**: una barra por GPU (VRAM) y otra para la RAM; una GPU sin datos de memoria, como una GPU de Intel que
  todavía no usa ningún modelo, no tiene barra. Cada barra se divide según el uso de la memoria, cada uso siempre
  con el mismo color, y su leyenda enumera todos los segmentos con su tamaño y su proporción del total, uno por
  línea y en el mismo orden que en la barra. Pasa el puntero por un segmento para ver su tamaño.

  | Segmento | Contenido |
  | --- | --- |
  | Pesos MTP / borrador | Pesos de las capas MTP (nextn) del modelo, de un GGUF de MTP separado indicado con `--spec-draft-model` o de un modelo borrador. |
  | Pesos de los expertos | Tensores de los expertos enrutados del modelo. |
  | Pesos densos | El resto de los pesos: atención, expertos compartidos, embeddings y salida. |
  | Caché KV | Memoria de contexto del modelo (caché KV o estado recurrente). |
  | Caché KV de MTP / borrador | Memoria de contexto del contexto MTP o del borrador. |
  | Búferes de cálculo | Búferes temporales de los grafos de cálculo. |
  | Caché y precarga de expertos | Caché de expertos MoE (depósito, identificadores y staging) y los slots de precarga de expertos. |
  | Caché de prompts y checkpoints | Prompts guardados con `--cache-ram` y los checkpoints de contexto de los slots (RAM). |
  | Otros | Memoria en uso que no explica nada de lo anterior: otros procesos, el escritorio, contextos del controlador, el proyector multimodal. |
  | Libre | El resto del dispositivo o de la RAM. |

  Los pesos mapeados desde el archivo del modelo cuentan como RAM aunque el núcleo considere disponible esa caché de
  páginas, así que los segmentos pueden sumar más que la memoria «en uso»; entonces la barra no muestra «Otros» y
  muestra menos memoria libre. La caché de expertos aparece tras la primera petición, que es cuando se crea. En modo
  router las barras incluyen todos los modelos cargados, y el monitor nunca carga ni despierta un modelo.

La **barra de estado**, en la parte inferior de todas las páginas, muestra primero el uso de cada GPU y después el
de cada procesador físico, cada uno en su propia barra y con las mismas etiquetas que las tarjetas del monitor:
`gpu-1`, `gpu-2`... y `cpu-1`, `cpu-2`... Las barras tienen el mismo tamaño, se actualizan cada dos segundos y, como
las del monitor, son verdes y se llenan de rojo a medida que aumenta el uso; el texto emergente de cada una indica el
nombre del dispositivo. Sin GPU muestra una barra `GPU` vacía y, cuando las barras no caben en una pantalla estrecha,
la barra de estado se desplaza en horizontal. Haz clic en ella para abrir el monitor. Sigue funcionando mientras carga
el modelo. Se muestra por defecto; desactiva Ajustes → Visualización → **Mostrar la barra de estado** para ocultarla.

<a id="models"></a>
## Modelos, herramientas y MCP

`mgs` admite únicamente modelos principales MoE. Los modelos densos, como GPT-2 o Phi-2, se rechazan antes de cargar sus pesos y sus decodificadores no se compilan. Se mantienen las capas densas que forman parte de un MoE y las cabezas MTP compatibles que se pasan explícitamente en un GGUF separado.

<a id="supported-models"></a>
### Familias de modelos admitidas

| Familia | Arquitectura GGUF | Notas |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP cuando el GGUF lo trae. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP cuando el GGUF lo trae. Su tabla de embeddings por capa (unos 29 GB) se lee del archivo del modelo bajo demanda, así que conviene tener el archivo en un disco rápido. Los contextos largos se calculan con atención densa bajo la selección dispersa: el resultado es correcto, pero sin la mejora de velocidad de los kernels dispersos. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; imágenes con su mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Los GGUF publicados no traen MTP. Con Vulkan, su mezcla residual se ejecuta en la CPU. |
| DeepSeek-V4.1-Flash | `deepseek41` | Carga las dos disposiciones publicadas: los archivos DwarfStar de antirez (como `antirez/deepseek-v4.1-flash-gguf`) y los del ecosistema llama.cpp. Su tabla Engram (unos 188 GiB dentro del archivo) nunca se carga: cada token lee sus filas del archivo del modelo, así que el archivo debe seguir disponible mientras `mgs` se ejecuta, mejor en un SSD, y hace falta Linux u otro sistema POSIX. Los archivos sin plantilla de chat usan una integrada, y `reasoning_effort` acepta un número del 1 al 100. Sin MTP ni imágenes. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

Gemma 4 y los modelos DeepSeek anteriores a la V4.1 ya no se admiten (consulta la [compatibilidad eliminada](#removed-compatibility)).

Con varias GPU seleccionadas, MTP comprueba todas antes de recurrir a la RAM. Su estimación de memoria incluye la cabeza de salida. Los búferes MTP integrados se cargan antes de los demás pesos; un GGUF MTP externo pasado con `--spec-draft-model` se carga antes del modelo principal, también con `--fit off`. La selección explícita de dispositivos sigue determinando qué GPU están disponibles.

Usa `--reasoning on` o `--reasoning off` para controlar el razonamiento. Se rechaza `enable_thinking` dentro de `--chat-template-kwargs`. Usa los nombres documentados de las opciones, con guiones; las variantes con guiones bajos se rechazan. Pasa las listas como un valor separado por comas, en vez de repetir la opción. Solo `--spec-type` admite repetición para combinar modos especulativos.

El selector distingue modelos disponibles, cargados y favoritos. Los detalles
incluyen contexto, modalidades, cuantización y plantilla de chat. El modo router
mantiene procesos separados para los modelos cargados; las peticiones identifican
el modelo mediante el campo o parámetro `model`.

Configura servidores MCP por URL y, si hace falta, cabeceras de autorización.
Puedes consultar registros de conexión, instrucciones, recursos y herramientas.
El proxy CORS opcional debe activarse mediante la opción correspondiente del motor; sólo
devuelve contenido inerte y no se deja abrir como página (consulta [límites de seguridad](#security-limits)).
Si dos servidores MCP activos ofrecen una herramienta con el mismo nombre, no se ofrece al modelo
ninguna de las dos y la interfaz muestra un aviso: desactiva todos esos servidores menos uno para usarla.
Se siguen aplicando las restricciones del navegador para recursos HTTP dentro
de páginas HTTPS. Las herramientas integradas del servidor y el entorno JavaScript
aislado son opcionales. La interfaz conserva las autorizaciones de herramientas
y el límite de turnos del agente. Ese límite (Ajustes → Agentes → **Turnos del agente**,
el número máximo de ciclos de ejecución de herramientas antes de detenerse, para evitar
bucles infinitos) es 50 por defecto.

Los flujos pueden reconectarse mediante la identidad y el cursor guardados
mientras el servidor conserve la sesión. Un servidor detenido o un flujo caducado
no garantizan la recuperación. La interfaz informa del fallo de reconexión y no
genera otra respuesta silenciosamente.

### Servidor MCP local de shells

Arranca `mgs` con `--local-shell-mcp-server` para que los modelos del chat web puedan usar la terminal del equipo en el
que se ejecuta `mgs`:

```bash
mgs --model /ruta/al/modelo.gguf --local-shell-mcp-server
```

- Al arrancar, `mgs` comprueba qué shells están instalados y ofrece una herramienta para cada uno: `bash.run`
  (Bash), `zsh.run` (Zsh), `sh.run` (sh) y `psh.run` (PowerShell, `pwsh`). El registro muestra los shells
  encontrados; un shell que no está instalado no tiene herramienta.
- La interfaz lo detecta, lo añade a los servidores MCP como **local shell mcp server** y lo activa. Puedes desactivarlo ahí, o
  solo para una conversación, y se conserva tu elección. Cuando `mgs` arranca sin la opción, la entrada
  desaparece.
- Cada herramienta recibe `command` y, de forma opcional, `cwd` (directorio de trabajo; por defecto, el directorio
  personal del usuario que ejecuta `mgs`) y `timeout` (segundos; 60 por defecto y 3600 como máximo). La respuesta
  es un objeto JSON con `exit_code`, `stdout` y `stderr`, además de `signal`, `timed_out`, `cancelled`, `timeout`,
  `seconds`, `shell` y `cwd`; de las salidas largas solo se conservan el principio y el final.
- Cada comando se ejecuta en un proceso nuevo, sin terminal y con la entrada vacía: `cd` y las variables no pasan a
  la llamada siguiente, y no se pueden usar programas interactivos (editores, preguntas). Cuando se agota el tiempo
  límite o se cancela la petición, se matan todos los procesos que haya iniciado el comando. Los procesos que se
  queden en segundo plano tienen que redirigir su salida.
- La interfaz pide permiso antes de cada llamada, salvo que elijas permitir siempre la herramienta.
- `sudo` funciona sin terminal (Linux): cuando necesita una contraseña, se abre una ventana en el centro de la
  página, estés en la página que estés, con el mensaje de sudo (`[sudo] password for usuario:`), la herramienta y el
  comando que la pide, un campo de contraseña y, debajo, los botones **Aceptar** y **Cancelar**. El cursor ya está
  en el campo de contraseña, así que puedes escribirla directamente; **Intro** acepta y **Escape** cancela; un clic
  fuera de la ventana no la cierra. La contraseña solo llega a sudo en el equipo donde se
  ejecuta `mgs`: no se guarda en la conversación, no se escribe en los registros y el modelo nunca la ve. Mientras la
  ventana espera, el tiempo límite del comando deja de contar; si pulsas **Cancelar**, o nadie responde en 5
  minutos, sudo falla. sudo vuelve a pedirla en cada llamada, porque cada una es un proceso nuevo, y hasta tres veces
  si la contraseña es incorrecta; entonces la ventana avisa de que puede que la contraseña anterior fuera
  incorrecta. Los comandos se ejecutan en una sesión propia, sin la terminal desde la que se arrancó `mgs`, así que
  sudo nunca pide la contraseña en esa terminal. Solo puede responder la interfaz web de `mgs`: con otros clientes MCP, sudo falla como antes por
  no tener terminal. El servicio desplegado se ejecuta con la cuenta `moe-gguf-server`, que normalmente no tiene
  contraseña ni permiso de sudo, así que está pensado para `mgs` ejecutado con tu propio usuario (por ejemplo, la
  compilación de `~/.local`). Cuando `mgs` no tiene la variable `DISPLAY`, los comandos reciben
  `DISPLAY=:mgs-askpass`, porque sudo solo usa su programa de contraseña sin terminal cuando existe `DISPLAY`; los
  programas que intentan abrir una ventana fallan al instante.
- En el chat, cada llamada se muestra con el nombre de su herramienta, como `bash.run`. Al desplegarla aparecen
  hasta tres cajas grises con letra monoespaciada: **stdin** con el comando tal cual, **stdout** con la salida y
  **stderr** con el texto del error. stdout solo aparece si el comando escribió algo y stderr solo si hubo un error;
  el código de salida va junto a su título, por ejemplo `stdout (código 2):` (junto a stdin cuando no hay salida).
  La caja stdin tiene el borde blanco (gris oscuro en el tema claro); la caja stdout, verde, y la caja stderr,
  rojo, con todas las herramientas de shell (`bash.run`, `zsh.run`, `sh.run` y `psh.run`).
- Activa **Ajustes → Visualización → Desplegar automáticamente los comandos de la shell local** para que cada
  llamada se abra sola: cada caja aparece en cuanto se conoce su contenido, stdin mientras el modelo escribe el
  comando y stdout y stderr cuando el comando termina.
- Cada llamada a una herramienta de shell deja su propio archivo en `~/.mgs/local-shell-mcp-server/` (en el
  directorio personal del usuario que ejecuta `mgs`; en el servicio desplegado,
  `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), con el nombre de la fecha y la hora locales de la llamada,
  con sus milisegundos, por ejemplo `y2026m10d12h23m59s59ms731.jsonl`; si empiezan varias llamadas en el mismo
  milisegundo se añade `-2`, `-3`… Es
  un archivo JSONL, con un objeto JSON por línea: primero el comando (`"event": "command"`, con la herramienta, el
  shell, el directorio de trabajo y el tiempo límite), que se escribe antes de ejecutarlo; después toda la salida,
  sin recortes, a medida que llega, una línea por fragmento (`"event": "stdout"` o `"stderr"`, con el texto en
  `data`) en el orden en que la escribió el comando, y por último el resultado (`"event": "result"`: código de
  salida, señal, si se agotó el tiempo o se canceló, los segundos que tardó y los bytes de stdout y de stderr) o el
  motivo por el que no pudo arrancar (`"event": "error"`). Una llamada rechazada antes de ejecutarse (sin comando,
  con un directorio de trabajo que no existe o con un tiempo límite no válido) también deja su archivo, con el
  comando tal como se pidió y el error. El chat solo conserva el principio y el final de las salidas muy largas,
  pero el archivo lo tiene todo, y ya contiene la salida aunque `mgs` se detenga mientras se ejecuta el comando.
  Para leer todo el stdout de una llamada, ejecuta `jq -j 'select(.event == "stdout") | .data' ARCHIVO`. JSON solo
  admite texto, así que los bytes que no son UTF-8 (una salida binaria) se sustituyen por `�`; un comando que
  escribe gigabytes deja un archivo de ese tamaño. Solo ese usuario puede leer la carpeta y los archivos, porque la
  salida de un comando puede contener secretos; nada los borra, así que elimina los antiguos cuando ya no los
  necesites.

En el servicio desplegado, añade `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` a la sección `[Service]` de un override
de systemd (`systemctl edit moe-gguf-server`) y reinicia `moe-gguf-server`; en Alpine, exporta la misma variable
en la configuración del servicio OpenRC.

**Seguridad.** Los comandos se ejecutan con los permisos del usuario que ejecuta `mgs` (en el servicio desplegado,
la cuenta `moe-gguf-server`, nunca `root`). Cualquiera que pueda abrir la interfaz o llamar a la API puede
ejecutarlos, así que activa la opción solo en entornos de confianza y protege el servidor con una clave de API. Se
rechazan las peticiones de páginas de otros sitios, el origen CORS predeterminado pasa a ser localhost y el registro
anota cada comando con su código de salida. Lee el comando que muestra la ventana de la contraseña antes de escribir
una contraseña de sudo: la contraseña autoriza ese comando. El servidor solo se la entrega a un programa de
contraseña cuyo proceso padre sea sudo ejecutándose como root para tu usuario, así que un comando no puede lanzar
ese programa por su cuenta para leerla.

<a id="api"></a>
## API HTTP

Los ejemplos usan 11443; si se eligió otro puerto, utilizar el valor de `ports.json`.

La raíz predeterminada es `/api` y la interfaz sigue en `/`. Las rutas originales
`/v1/...` de la raíz ya no están registradas. Los clientes compatibles con OpenAI
deben usar `https://ia.ejemplo.com:11443/api/v1`. Los nombres de campos mantienen su
compatibilidad original; las interfaces C/C++ del motor conservan sus funciones actuales,
pero no las que estaban marcadas como obsoletas (consulta [compatibilidad eliminada](#removed-compatibility)).

```bash
curl https://ia.ejemplo.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hola"}]}'
```

Asigna la clave instalada a `MGS_API_KEY` en tu propia consola. Swagger está en
`/api/doc/` y su especificación en `/api/doc/openapi.json`, generada por el servidor
en cada solicitud sin un archivo JSON estático. Incluye las operaciones
nativas, embeddings, tokenización, reranking, modelos y flujos reanudables.
El inventario refleja las rutas registradas por el proceso, incluidas las del
router y las de compatibilidad opcional. Las funciones desactivadas pueden
responder con 403; una clave no válida produce 401 y un modelo cargando, 503.
Las rutas de salud y listado de modelos mantienen su acceso público con `GET` y `HEAD`;
consulta en [límites de seguridad](#security-limits) las rutas que siempre exigen la clave.

`--api-prefix /api/personalizado` cambia el prefijo de inferencia. La interfaz
lo obtiene de `/api/config.js`; Swagger permanece en `/api/doc/`, espacio reservado
que no se puede utilizar como prefijo de inferencia. Los procesos hijos del
router usan `/api` internamente. Se rechazan los prefijos fuera de `/api/` y los
que terminan con barra.

El monitor usa `GET /api/monitor` (informe completo), `GET /api/monitor/memory` (instantáneas de memoria de los
modelos cargados) y `GET /api/monitor/usage` (solo el uso de CPU y GPU, disponible también mientras carga el modelo).
Leen contadores e instantáneas que publica el bucle de inferencia, nunca la cola de tareas, así que consultarlos no
despierta un modelo en reposo ni reinicia su temporizador de inactividad. Sus campos se describen en Swagger.

La interfaz lee sus ajustes con `GET /api/ui/settings` (`{"settings": DOCUMENTO}`, o `{"settings": null}` antes del
primer guardado) y los guarda con `POST /api/ui/settings` (`{"settings": DOCUMENTO}`, que sustituye entero
`~/.mgs/config/ui-settings.json`, con modo 0600). Las dos piden la clave API y responden también mientras carga el
modelo.

<a id="request-queue"></a>
### Cola de peticiones y peticiones en paralelo

Por defecto, `mgs` resuelve una petición cada vez (`-np 1`) y le da el contexto completo, así que cada petición
puede usar todos los tokens de contexto, tanto si llega desde la interfaz web como si llega directamente a la API.
Las peticiones que llegan mientras tanto esperan en una única cola y se resuelven en el orden en que llegaron: las
de la interfaz no adelantan a las de la API, ni al revés. La API se comporta igual sea cual sea la configuración de
la interfaz.

Para resolver varias peticiones a la vez, arranca `mgs` con `-np N` (`--parallel N`, `MGS_ARG_N_PARALLEL`): se
ejecutan hasta N peticiones en paralelo y el contexto se reparte entre ellas, así que cada una recibe una parte
(`n_ctx_slot` en la línea `initializing` del registro); con `--kv-unified`, en cambio, comparten una sola caché KV
del contexto completo. Las peticiones que pasen de N siguen esperando en la cola.

La interfaz web añade `"mgs_client": "webui"` a sus peticiones. Mientras una de sus peticiones en streaming
espera, el stream empieza con eventos como

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: peticiones que se ejecutan o esperan por delante; `ahead_api` y `ahead_webui`: cuántas de ellas llegaron
directamente a la API o desde la interfaz web) y, cuando empieza, recibe uno último con todos los contadores a 0. Los
clientes que no envían `mgs_client` nunca reciben estos eventos, así que los clientes compatibles con OpenAI
reciben el stream habitual. Swagger describe el campo junto a los demás parámetros de completion.

<a id="security-limits"></a>
### Límites de seguridad

Estos límites protegen el servidor y la interfaz de peticiones, archivos y modelos hostiles. Son holgados para el
uso normal: lo que los supera falla con un error claro en lugar de detener el servidor.

- **Clave API.** Las rutas de salud y de listado de modelos sólo son públicas con `GET` y `HEAD`. Recargar la
  lista de modelos (`GET /api/models?reload=1`) y añadir o borrar modelos en modo router (`POST` y
  `DELETE /api/models`) exigen la clave siempre que haya una configurada.
- **Páginas de otros sitios.** Cuando hay herramientas del servidor o servidores MCP activos (`--tools`, `--agent`,
  `--mcp-servers-config`, `--mcp-servers-json` o `--local-shell-mcp-server`) y no se ha indicado
  `--cors-origins`, el servidor responde 403 a toda petición que no sea `GET`, `HEAD` ni `OPTIONS` cuya cabecera
  `Origin` nombre otro sitio. No afecta a la interfaz (tampoco detrás de Apache) ni a los clientes que no envían
  `Origin`, como curl o los SDK. `POST /api/tools` sólo acepta `Content-Type: application/json` (si no, 415).
- **Proxy CORS.** `/api/cors-proxy` rechaza las peticiones que el navegador abriría como página, marco o worker
  (403) y devuelve todas las respuestas como contenido inerte: no puede ejecutar scripts ni crear cookies en la
  dirección del servidor.
- **Listados del router.** En modo router, `status.args` y `status.preset` de `/api/models` muestran
  `<redacted>` en lugar de los valores de `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`,
  `--mcp-servers-config` y `--mcp-servers-json`; el registro hace lo mismo. Los procesos de los modelos siguen
  recibiendo los valores reales.
- **Clientes lentos.** Mientras el servidor recibe una petición, cada lectura espera como mucho 30 s (`--timeout`
  sigue rigiendo las escrituras, así que las respuestas largas no se cortan), y como mucho 256 conexiones
  aceptadas esperan a que quede un hilo libre.
- **Cuerpos de las peticiones.** Los cuerpos JSON con más de 1024 niveles de arrays u objetos se rechazan con 400.
  Los esquemas JSON (`response_format`, `json_schema`) pueden anidarse hasta 256 niveles; se rechazan los
  `pattern` vacíos y los cuantificadores sin nada que repetir.
- **Medios remotos.** Las URL de `image_url`, `input_audio` e `input_video` sólo pueden apuntar a direcciones
  públicas de Internet. Loopback, las redes privadas (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10),
  las direcciones link-local (incluida la de metadatos de la nube, 169.254.169.254), las direcciones locales
  únicas de IPv6 y las demás direcciones no públicas se rechazan con 400, también cuando se llega a ellas por una
  redirección. Para permitirlas, por ejemplo para leer imágenes de un servidor de tu red, arranca `mgs` con
  `--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); hazlo sólo si confías en todos los clientes
  de la API. Cada descarga tiene 10 s para conectar, 30 s en total, hasta 5 redirecciones y 10 MiB, y el servidor
  hace como mucho 8 descargas a la vez.
- **Medios decodificados.** Imágenes de hasta 32768 píxeles por lado y 100 megapíxeles, audio de hasta una hora,
  fotogramas de vídeo de hasta 8192 píxeles por lado y 2 GiB de fotogramas decodificados por vídeo, y como mucho
  2 GiB de imágenes y audio decodificados por petición.
- **Plantillas de chat del modelo.** `range()` produce como mucho un millón de elementos, las cadenas construidas
  no pasan de 256 MiB, la sangría de `indent()` y `tojson()` de 256 espacios y la anidación de 1024 sentencias;
  una plantilla que los supera falla con un error de plantilla.
- **RPC.** En una compilación con el backend RPC, el servidor RPC comprueba los tamaños de trama y los límites de
  los tensores que le envían sus pares y cierra sólo la conexión culpable. Exponlo sólo a equipos de confianza:
  sigue atendiendo a un cliente cada vez.
- **Interfaz.** Los enlaces de los mensajes sólo conservan los esquemas http, https, mailto y tel (o direcciones
  relativas), y los adjuntos de imagen tienen que ser imágenes incrustadas. La importación de conversaciones
  acepta archivos de hasta 512 MB, archivos ZIP con hasta 10 000 entradas, 256 MB por archivo JSONL y 1 GB
  extraído en total. Un archivo con registros mal formados, o con un mensaje que pertenece a otra conversación,
  se rechaza entero; una conversación cuyos mensajes ya existen se omite y se indica en el aviso.

Los comandos de instalación siguen descargando la rama `main` actual por HTTPS; no comprueban una firma ni una
revisión fija.

<a id="performance"></a>
## Memoria y rendimiento

El registro de memoria de host y la precarga de expertos están activos por
defecto. La precarga solapa las transferencias con el cálculo, especialmente
con prompts grandes. El resultado depende del modelo, la capacidad de memoria
y el ancho de banda RAM/PCIe.

`--cpu-moe` o `--n-cpu-moe N` permiten mantener expertos en RAM para reducir el
uso de VRAM. El registro de memoria fija los pesos mapeados para agilizar sus
transferencias. La RAM fijada no se puede paginar; deja memoria suficiente para
el sistema operativo. Esta ruta es específica de POSIX; Windows mantiene el
comportamiento anterior. Solo se recomienda `--load-mode none` si el registro falla
total o parcialmente.

La precarga usa tres slots de VRAM por defecto. Cada uno debe alojar el mayor
tensor de expertos transferido. Los valores 2–8 de `GGML_SCHED_PREFETCH_EXPERTS`
eligen el número de slots. Ante un fallo de reserva se conservan los que caben
si son al menos dos; de lo contrario se usan transferencias normales. Se activa
cuando el lote contiene al menos el doble de selecciones que expertos. El grupo
pertenece a la primera GPU que lo activa; las demás mantienen la copia selectiva.
Un lote mayor puede mejorar el prefill, pero también consume más memoria.

Para diagnóstico, `GGML_CUDA_REGISTER_HOST=0` desactiva el registro y
`GGML_SCHED_PREFETCH_EXPERTS=0` desactiva la precarga. No se necesitan esas
variables durante el funcionamiento normal.

MTP automático inspecciona metadatos y nombres de tensores antes de reservar
pesos, incluidas todas las partes del GGUF. Requiere arquitectura compatible,
tronco y todos los cabezales integrados; un cabezal externo no lo activa.
El máximo predeterminado es de tres tokens de borrador. `--spec-type` explícito
prevalece; `--no-spec-mtp-auto` y `--spec-type none` desactivan la selección
automática. `--spec-draft-n-max` ajusta el máximo. MTP afecta a la generación, no
directamente al prefill ni al tiempo hasta el primer token. La mejora depende
de la aceptación y del hardware; aumentar el máximo no garantiza más velocidad.

<a id="mtp-vram"></a>
Los pesos MTP se quedan siempre en la memoria de la GPU cuando caben en ella: las capas MTP injertadas en el
modelo, un GGUF de MTP separado pasado con `--spec-draft-model` y la cabeza de salida que usan. Las opciones que
llevan pesos a la memoria del sistema (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, sus equivalentes
`--spec-draft-*` y el ajuste automático) no se les aplican, y siguen en la GPU aunque `--gpu-layers` sea bajo. El
ajuste automático coloca el resto del modelo a su alrededor y la [caché de expertos MoE](#expert-cache) usa la
memoria que queda después. Si las capas MTP no caben en la memoria libre de la GPU con un margen de 512 MiB, el
registro muestra un aviso y se colocan como cualquier otra capa. Con un GGUF de MTP separado no se cargan las capas
MTP injertadas en el modelo principal, para que no ocupen memoria dos veces; con `--spec-type none` no se carga
ningún peso MTP. El registro de carga confirma la ubicación con una línea como
`keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)`, visible con `-lv 4`.

<a id="expert-cache"></a>
### Caché de expertos MoE

Cuando los expertos MoE están en la memoria del sistema y el resto de la capa se ejecuta en una GPU, el
servidor mantiene los expertos usados más recientemente de todas las capas en una caché que ocupa la VRAM libre; con
varias GPU, cada una guarda en su VRAM las capas que tiene asignadas. En cada token generado (y en cualquier lote de menos de 32 tokens), los expertos que están en la caché
se calculan en la GPU, una parte de los que faltan se sube a la caché y el resto se calcula en la CPU al mismo
tiempo; esa parte se ajusta a los anchos de banda medidos de PCIe y de la CPU. Los prompts largos conservan la
precarga de expertos, que ahora copia dentro de la GPU los expertos cacheados y solo sube los demás. El resultado coincide con el de la ejecución sin caché salvo redondeo. La GPU y la CPU redondean de
forma distinta y los expertos que calcula cada una dependen del contenido de la caché, así que una generación
voraz larga puede derivar, de una ejecución a otra, hacia un texto distinto pero igual de válido tras algunos
tokens; la caché de llama.cpp oficial se comporta igual.

La caché está activa por defecto y se crea en la primera petición, cuando ya están cargados el modelo, el
contexto MTP y el proyector multimodal. Sin `--gpu-layers`, `--n-cpu-moe` ni `--override-tensor`, el ajuste
automático mantiene todos los expertos en la memoria del sistema si caben con margen (4 GiB o el 10 % de la RAM)
y deja la VRAM libre a la caché (con varias GPU, reparte las capas entre ellas según su memoria libre
para que todas tengan caché); si no, coloca capas de expertos completas en la GPU como antes y la caché usa
lo que queda. Las opciones explícitas de colocación se respetan: `--cpu-moe` deja a la caché la mayor parte de
la VRAM y `--n-cpu-moe N` mantiene algunas capas fijas en la GPU. Los expertos que se quedan en la RAM se fijan
en memoria, así que la RAM debe poder alojarlos todos además del sistema operativo.

<a id="context-size"></a>
Cuando el modelo no cabe en la VRAM y no se indica `-c`, el ajuste automático elige también el tamaño del
contexto: la caché KV ocupa como mucho una cuarta parte de la VRAM que queda tras los pesos densos, hasta el
contexto de entrenamiento del modelo, y los expertos (la caché o las capas de expertos completas) se quedan con el
resto. Antes, esos modelos recibían siempre el mínimo de 4096 tokens compartidos por todos los slots, lo que
cortaba las respuestas largas (`truncated = 1` en el registro). El registro muestra el resultado en una línea como
`context size raised from 4096 to N tokens` y en `n_ctx_slot` de la línea `initializing`. `-c N` fija el
contexto de forma explícita (más contexto deja menos VRAM a la caché y reduce la velocidad de generación: 35,8
tokens/s con 4096 tokens y 33,9 con 122624 en la máquina de referencia) y `--fit-ctx N` sube el mínimo. El
contexto MTP toma el mismo tamaño que el contexto del modelo en lugar de reservar VRAM para todo el contexto de
entrenamiento.

Antes que el contexto, el ajuste automático elige el micro-lote (`-ub`), el número de tokens del prompt que se
procesan en cada pasada. Con los expertos en la RAM, cada pasada sube a la GPU los expertos que no están en la
caché, así que menos pasadas y más grandes procesan los prompts largos mucho más rápido, sin cambiar la
generación. Si no se indica `-b` ni `-ub`, toma el mayor de 4096, 2048 y 1024 tokens cuyos búferes extra quepan en
una octava parte de la VRAM que queda tras los pesos densos, y sube el lote (`-b`) con él. En la máquina de
referencia, un prompt de 2514 tokens pasó de 214 a 755 tokens/s con 4096, con la misma velocidad de generación; los
búferes más grandes acortan el contexto predeterminado (unos 90000 tokens sin MTP y 61000 con MTP en 16 GB). El
registro muestra la elección en una línea como `micro-batch size raised from 512 to 4096 tokens (batch 4096)`.

| Opción | Variable | Efecto |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE` (`0` la desactiva) | Activa o desactiva la caché (activa por defecto). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | Tamaño total fijo de la caché, repartido entre las GPU según los expertos que guarda cada una; `-1` (por defecto) usa la VRAM libre de cada GPU. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | VRAM que se deja libre en cada GPU además de la caché y de los huecos de precarga (1024 por defecto). Auméntala si otros programas usan la GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnóstico: proporción fija (0–1) de expertos que faltan y se suben, en lugar del equilibrio medido. |
| — | `MGS_MOE_CACHE_STATS` | Diagnóstico: registra los contadores de la caché cada N pasos de generación (visibles con `-lv 4`). |

`GET /api/expert-cache` devuelve el tamaño, las capas y huecos cacheados, los aciertos y fallos, las subidas,
los expertos calculados en la CPU, la proporción de bytes de precarga servidos desde la caché y los anchos de banda medidos (con varias GPU, los
tamaños y contadores se suman y el ancho de banda de subida es su media). `POST /api/expert-cache` con `{"size_mib": N}` cambia el tamaño sin reiniciar (`-1` automático,
`0` libera la caché); la siguiente petición lo aplica y empieza con la caché vacía:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Con `--metrics`, `/api/metrics` añade contadores e indicadores `mgs:moe_cache_*` (pasos, aciertos, fallos,
subidas, expertos en CPU, bytes de precarga, tamaño, huecos y anchos de banda).

Medición de referencia en una RTX 4060 Ti de 16 GB con Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 hilos de CPU), frente a
`--n-cpu-moe 28` sin caché: generación de 26,9 a 35,7 tokens/s (+33 %), procesado del prompt de 767–778 a
718–730 tokens/s (−6 %) y, con MTP activo, de 33,6 a 47,0 tokens/s (+40 %). En la misma máquina, llama.cpp oficial
con su propia caché (`-cmoe --moe-cache-mib 10240`) llegó a 26,7 tokens/s, y a 27,8 con MTP. La ganancia depende de la parte de
los expertos que cabe en la VRAM, de la localidad del enrutado del modelo y de los anchos de banda de PCIe y de
la RAM.

<a id="threads"></a>
### Hilos de CPU

La CPU calcula los expertos que no están en la caché. Ese trabajo está limitado por el ancho de banda de la
memoria y se satura con pocos hilos, mientras que usar todos los CPU lógicos hace que los hilos compitan con el
que maneja la GPU. Sin `-t`, `mgs` usa los núcleos útiles para cálculo, pero nunca más de la mitad de los CPU
lógicos en los que puede ejecutarse. En una máquina con SMT (dos hilos por núcleo) eso es el número de núcleos
físicos, como antes; en una máquina virtual que no muestra SMT es la mitad de los CPU virtuales. En la máquina de
referencia (16 núcleos virtuales con una sesión de escritorio, Qwen3.6-35B-A3B Q8_0, sin MTP), de 4 a 12 hilos
generaron a la misma velocidad (32,9–33,9 tokens/s) y 16 hilos a 28,7 tokens/s. El registro de arranque muestra la
elección en una línea como `using 8 CPU threads for generation and 8 for batches`; `-t` y `-tb` la fijan de forma
explícita.

<a id="checkpoints"></a>
### Checkpoints de contexto

Los modelos con capas recurrentes o de ventana deslizante (por ejemplo los híbridos Qwen3.5/3.6/3.8,
Nemotron-H o Granite 4) no pueden deshacer su estado, así que cuando una petición cambia una
parte anterior de la conversación, el servidor continúa desde el checkpoint más reciente anterior al cambio.
Los checkpoints se colocan al comienzo de los mensajes de usuario, asistente y herramienta (donde los agentes
de programación eliminan razonamientos antiguos o recortan resultados de herramientas) y al menos cada
`--checkpoint-min-step` tokens (1024 por defecto, antes 8192), sin pasadas adicionales por el prompt. Al
alcanzar `--ctx-checkpoints` (32), se elimina el checkpoint más próximo a sus vecinos, de modo que los demás
siguen repartidos por la conversación. En el escenario de agente de referencia (14 000 tokens, ocho resultados
de herramientas), recortar un resultado tardío pasó de 19,3 s a 5,5 s y uno temprano de 18,0 s a 13,6 s. Cada
checkpoint guarda en la memoria del host el estado recurrente o de ventana deslizante (63 MiB en
Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Opciones actualizadas y diagnóstico

Conviene mantener `--load-mode mmap`, predeterminado, cuando funciona el registro de memoria. `--load-mode none` es una alternativa si ese registro falla. `--load-mode dio` utiliza ahora un buffer auxiliar de hasta 64 MiB; no es un límite de RAM total del modelo.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` se retiró porque no tenía efecto durante la ejecución. Las antiguas opciones de carga (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) y el flag de compilación `GGML_CUDA_FA_ALL_QUANTS` ya no se aceptan; sus sustitutos están en [compatibilidad eliminada](#removed-compatibility). En una compilación CMake manual, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` añade esa combinación K/V de kernels vectoriales; se admiten listas con comas o con punto y coma entre comillas. La lista predeterminada es `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 se incluye siempre. `all` aumenta el trabajo de compilación. Si falta una combinación vectorial, se convierte a F16, lo que puede ser más lento.

Los GGUF completos de GLM-4.5-Air (`glm4moe`) con un único cabezal pueden activar MTP automático. Se mantienen las demás comprobaciones y la prioridad de las opciones explícitas. MTP separa grafos con y sin salidas, conserva el orden original de tokens de secuencias concurrentes y usa las posiciones reales después de imágenes.

El ajuste automático cuenta las capas nextn (MTP), así que la primera capa ya no se queda en la CPU cuando MTP está desactivado. Los pesos MTP se quedan ahora siempre en la memoria de la GPU; consulta [ubicación de MTP](#mtp-vram). El valor por defecto de `--checkpoint-min-step` es ahora 1024 en lugar de 8192; consulta [checkpoints de contexto](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` y `--moe-cache-reserve` controlan la [caché de expertos MoE](#expert-cache). El ajuste automático ya no deja con un contexto de 4096 tokens a los modelos más grandes que la VRAM y ahora también elige el micro-lote; consulta [tamaño del contexto](#context-size). El número de hilos de CPU por defecto es ahora, como mucho, la mitad de los CPU lógicos; consulta [Hilos de CPU](#threads). `-np` usa ahora por defecto un solo slot con el contexto completo, en lugar del modo automático con cuatro slots que compartían una caché KV unificada, y `--kv-unified` está desactivado por defecto; consulta [Cola de peticiones](#request-queue).

Con `--metrics`, `GET /api/metrics` autenticado incluye `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` y `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Las posiciones empiezan en cero; esa serie aparece al completar una solicitud especulativa. La aceptación se puede evaluar con las diferencias de contadores aceptados/propuestos; la latencia debe medirse por separado, porque una aceptación alta no demuestra por sí sola más velocidad. En modo router se elige el modelo con `?model=ID_MODELO`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, desactivada por defecto) permite que las URL de medios de las peticiones lleguen a direcciones de loopback, privadas y otras no públicas; consulta [límites de seguridad](#security-limits).

Las exportaciones leen el árbol completo de mensajes y los metadatos persistidos. Si las herramientas del servidor responden con 403, los mensajes nuevos dejan de repetir la consulta fallida; abrir de nuevo el panel de herramientas permite reintentar. Los valores del servidor en la primera visita conservan los ajustes ya modificados por el usuario, incluida la clave de API.

[Implementación y fuentes de upstream](CODE.es-ES.md#upstream-review).

<a id="removed-compatibility"></a>
## Compatibilidad eliminada

Se ha eliminado el código de retrocompatibilidad. Los nombres antiguos ya no se traducen ni
generan avisos: el servidor rechaza como argumentos no válidos las opciones de
línea de comandos eliminadas, ignora las variables de entorno eliminadas y
responde 404 en las rutas eliminadas. Los scripts, las personalizaciones de los
servicios, los presets y los clientes deben usar los sustitutos de esta tabla.

| Eliminado | Sustituto |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (o `mmap+mlock` para combinar ambos) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Nada; no tenía efecto. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Alias antiguos del modelo de borrador, como `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` o `--draft-p-min` | La opción `--spec-draft-*` equivalente, por ejemplo `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` o `--spec-draft-p-min`. `mgs --help` las muestra todas. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` o `--spec-ngram-mod-n-max` / `--spec-draft-n-min` o `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | La opción `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` o `--spec-ngram-*-min-hits` del modo n-gram elegido |
| Variables de entorno, opciones de CMake y símbolos de la API C con el antiguo prefijo heredado del proyecto original, incluidos los alias que versiones anteriores ya habían retirado | El nombre actual con el prefijo `MGS_` (`mgs_` en la API C), por ejemplo `MGS_ARG_PORT`, `MGS_ARG_<NOMBRE>=false` en lugar de una variante `NO_`, `MGS_ARG_LOAD_MODE` en lugar de las variables de mapeo en memoria, bloqueo y Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` y `MGS_ARG_SPEC_DRAFT_N_MIN` para los límites del borrador, `MGS_CACHE` o `MGS_BUILD_UI`; para las opciones de los backends, la opción `GGML_*` equivalente, por ejemplo `GGML_CUDA` o `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` o `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` o `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valor `deepseek-legacy` de `reasoning_format` (campo de petición y `--reasoning-format`) | `deepseek` o `auto`; el razonamiento se devuelve en `reasoning_content` |
| Campo de petición `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (array de cadenas) |
| `reasoning_in_content` en `generation_settings` | Nada; el campo se ha eliminado. |
| Descarga de una interfaz precompilada del proyecto original desde un bucket de Hugging Face (las opciones de CMake `USE_PREBUILT_UI` y `UI_HF_BUCKET` y la variable de entorno `HF_UI_VERSION`) | Nada; la interfaz siempre se compila desde `frontend/` (`MGS_BUILD_UI=ON`) o se omite. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Número de compilación y commit: las variables de CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (también en `mgs-config.cmake`) y la cadena `b0-unknown` de `build_info`, `system_fingerprint` y el `User-Agent` | `MGS_VERSION`, definida en `backend/cmake/build-info.cmake`; esos campos llevan ahora la versión, como `0.1.0` |
| Modelos Llama de Meta (las arquitecturas GGUF `llama`, `llama4` y `llama-embed` y el proyector de visión de Llama 4) y las plantillas de chat integradas `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` y `llama4` | Nada; usa uno de los modelos MoE compatibles. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Opción `--no-sudo` de los instaladores | Nada; los instaladores siempre se ejecutan como root y sin sudo. |
| Funciones y campos que las API C tenían marcados como obsoletos: en `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, los alias `mgs_token_*` (por ejemplo `mgs_token_bos`, `mgs_token_get_text` o `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, el campo `defrag_thold` de `mgs_context_params` y `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; en `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` y `ggml_upscale_ext`; en `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` y el campo `image_marker` | La función actual: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` en lugar de `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` y `media_marker`. Las ejecuciones de calentamiento las hace quien llama, y `defrag_thold` no tenía ningún efecto. |
| Gemma 4: las arquitecturas `gemma4` y `gemma4-assistant`, los proyectores `gemma4v`, `gemma4uv`, `gemma4a` y `gemma4ua`, el formato de chat `peg-gemma4`, el tokenizador y el pretokenizador `gemma4`, y los presets `--vision-gemma-4b-default` y `--vision-gemma-12b-default` | Nada; usa una de las [familias admitidas](#supported-models). |
| Modelos DeepSeek anteriores a la V4.1: las arquitecturas `deepseek`, `deepseek2` (también los archivos de Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 y Youtu convertidos con ella), `deepseek2-ocr` (con los proyectores `deepseekocr` y `deepseekocr2`), `deepseek32`, `mistral4` y `deepseek4`, los pretokenizadores `deepseek-llm` y `deepseek-coder` y la base de borrador DSpark de DeepSeek-V4 | DeepSeek-V4.1-Flash (`deepseek41`). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) y las plantillas integradas del motor sin Jinja (`--chat-template` con un nombre como `deepseek3`, `chatglm4` o `mistral-v7-tekken`) | Nada; las plantillas de chat pasan siempre por Jinja. `--chat-template chatml` sigue eligiendo ChatML; para otra plantilla, pasa su código Jinja con `--chat-template` o `--chat-template-file`. Las peticiones con `tools` ya no necesitan ninguna opción. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| Segundos nombres largos de una opción: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk` (las claves de los presets del router siguen los mismos nombres) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, con el nombre de su opción |
| Campo de petición `thinking_budget_tokens` | `reasoning_budget_tokens` (los clientes de Anthropic siguen enviando `thinking.budget_tokens`) |
| `content` en `POST /api/embeddings` y `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, la estructura `mgs_chat_message` y `MGS_FILE_MAGIC_GGLA` en `mgs.h`; `mtmd_image_tokens_get_id` en `mtmd.h` | Nada; el formato del chat lo hacen las plantillas Jinja de `common/chat.h`, y los demás no tenían uso. |

CMake avisa de que las variables `-D` eliminadas no se utilizan, en lugar de
aplicarlas, así que conviene revisar la salida de configuración de las
compilaciones manuales.

La interfaz web guarda sus ajustes en el servidor (`~/.mgs/config/ui-settings.json`),
y en el navegador solo su clave API (`MgsUi.apiKey` en `localStorage`) y sus
conversaciones (la base de datos IndexedDB `MgsUi`). No lee ni migra los ajustes que
versiones anteriores guardaban en `localStorage` (`MgsUi.config`, `MgsUi.userOverrides`,
los modelos favoritos, las listas de herramientas, el esfuerzo de razonamiento, las
recomendaciones MCP descartadas y el idioma), así que los ajustes empiezan con sus
valores predeterminados y hay que volver a introducir la clave API una vez. Tampoco
lee ni migra los datos que versiones anteriores guardaron con otros nombres, la clave
independiente `theme`,
los marcadores de razonamiento y de llamadas a herramientas incrustados en mensajes
antiguos, los adjuntos de texto pegado del antiguo tipo `context`, el antiguo formato
de lista de recomendaciones MCP descartadas o el antiguo campo `webui_settings` del
servidor. Para conservar las conversaciones de una versión anterior, expórtalas desde
ella antes de actualizar e importa la exportación después. La importación de
conversaciones solo acepta exportaciones JSONL y ZIP, dentro de los límites que se indican en
[límites de seguridad](#security-limits); el antiguo formato de exportación JSON se rechaza. Los ajustes antiguos se quedan en el navegador hasta que se borran:
para eliminarlos, abre la consola del navegador (F12) en la página de la interfaz y ejecuta
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
una vez por cada dirección y puerto con los que se haya abierto la interfaz.

Los archivos descargados con `-mu`/`--model-url` se guardan en `~/.cache/mgs/`, así
que los que versiones anteriores guardaron en su antigua carpeta de caché se vuelven
a descargar. Las descargas de Hugging Face siguen usando la caché de Hugging Face
(`~/.cache/huggingface/hub`, o la carpeta de `MGS_CACHE`, `HF_HUB_CACHE` o `HF_HOME`).

### Archivos de modelo antiguos

También se ha eliminado el código de compatibilidad con estas conversiones antiguas.
Para usarlas hay que volver a convertir el modelo original o descargar un GGUF
actual.

| Archivo antiguo | Comportamiento actual |
| --- | --- |
| Grok-1 convertido antes de agosto de 2025 (sin `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` o `grok.attn_logit_softcapping`) | No carga: falta una clave. |
| Kimi-Linear con el tensor sin dividir `blk.N.attn_kv_b` | No carga: faltan `attn_k_b`/`attn_v_b`. Esa ruta ya fallaba al construir el grafo. |
| Archivos de Qwen3-Next solo con el tensor combinado `ssm_in` | No carga: falta el tensor `blk.0.attn_qkv.weight`. |
| Archivos sin `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) o `full_attention_interval` (Qwen3-Next), y archivos de Cohere2-MoE con un único periodo SWA en lugar de la lista por capa | No carga: falta una clave. |
| Vocabularios BPE sin `tokenizer.ggml.pre` o con los nombres de pretokenizador `llama3`, `llama-v3`, `jina-es`, `jina-de` o `llada-moe`, y tokenizadores de Kimi-K2 sin fusiones (merges) | No carga: falta una clave, el pretokenizador es desconocido o faltan las fusiones. |
| Archivos GGUF de las versiones 1 o 2 | No carga: solo se acepta la versión 3 de GGUF. |
| mmproj de los scripts antiguos de un solo uso: tipos de proyector `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x a 4.5, MiniCPM-o 2.6) y `adapter` (GLM-Edge-V), y archivos sin tipo de proyector | El proyector no carga: el tipo de proyector es desconocido o falta. |
| mmproj de conversiones antiguas con `ffn_up` y `ffn_down` intercambiados | Ya no se corrigen; el proyector falla. |
| mmproj de Kimi-K2.5 sin `image_min_pixels`/`image_max_pixels` y de Step3-VL sin `preproc_image_size` | El proyector no carga: falta una clave o un tensor. |
| Archivos que solo llevan las claves antiguas `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` y `tokenizer.ggml.middle_token_id` en lugar de las `tokenizer.ggml.fim_*_token_id` | Carga; los tokens de relleno (infill) se reconocen por su texto (por ejemplo `<PRE>` o `<\|fim_prefix\|>`), y el relleno no funciona si no se encuentran. |

Se conservan los respaldos de los que todavía dependen las conversiones actuales o
los GGUF publicados más descargados de las familias admitidas, comprobados en sus
cabeceras de Hugging Face: el `add_bos` predeterminado de los pretokenizadores
`llama-bpe`, Tekken y similares; la disposición de capas recurrentes de Qwen3.6
deducida de `full_attention_interval` (los GGUF de Qwen3.6 más descargados no
tienen `recurrent_layers`), y la disposición integrada del indexador de GLM-5.2
para los GGUF de GLM-5.2/5.3 sin `attention.indexer.types`.

<a id="maintenance"></a>
## Mantenimiento y resolución de problemas

- Comprueba qué versión está instalada con `mgs --version` (muestra, por ejemplo, `version: 0.1.0`); la compilación
  local lo ejecuta al terminar y CMake muestra `mgs version: 0.1.0` durante la configuración. La interfaz muestra la
  misma versión con Ajustes → Visualización → **Mostrar la versión de compilación** (esquina inferior derecha) y en
  **Información de compilación** del cuadro de información del modelo. Cada cambio en el proyecto sube el último
  número: 0.1.0, 0.1.1, 0.1.2, etc.
- Repite el comando `curl | bash` para actualizar desde `main`, o el instalador local correspondiente para recompilar y sustituir el ejecutable.
  Después reinicia el servicio con `systemctl restart moe-gguf-server` en Debian
  o `rc-service moe-gguf-server restart` en Alpine.
- Consulta `/root/webapp-install.log`, `journalctl -u moe-gguf-server` en Debian
  o `/var/log/moe-gguf-server.log` en Alpine. Los registros Apache están en
  `/var/www/DOMINIO-logs/`.
- Si no puedes compilar CUDA, instala el Toolkit por separado o utiliza `--cpu`.
  Si falta NCCL, revisa los repositorios y la versión de CUDA, o elige `--no-nccl`.
- Si falla la interfaz, comprueba Node.js, npm, las descargas y el registro.
  No sustituyas la interfaz por una precompilada del proyecto original.
- Para deploy, tras reorganizar los fuentes, utiliza una compilación nueva en `_/temp/`.
  SSHFS puede impedir ejecutar archivos; usa un disco local en producción.
- Para la compilación local, repite el comando de una línea anterior para descargar
  el `main` actual, o `bash build/build-local.sh` para usar tus fuentes locales, y reinicia la copia
  que lanzaste a mano. Consulta la salida de la terminal si falla o indica
  bibliotecas fuera de `lib/`. El registro temporal `build.log` se elimina junto
  con la carpeta de compilación.
- Un 404 en clientes antiguos suele requerir cambiar la URL base a `/api/v1`.
  La clave API debe enviarse en la cabecera Authorization.

Las reglas impiden que el asistente ejecute pruebas, despliegues de prueba,
mediciones GPU o comprobaciones automatizadas en navegadores móviles. La revisión
estática no garantiza el funcionamiento en ejecución. Consulta
[CODE.es-ES.md](CODE.es-ES.md) para ver el mapa de implementación.

<a id="license"></a>
## Licencia

`mgs` se distribuye bajo la [licencia PolyForm NonCommercial 1.0.0](../LICENSE). Puedes usarlo,
modificarlo y compartirlo con cualquier fin no comercial: uso personal, estudio, investigación,
proyectos de afición y uso por parte de organizaciones benéficas, centros educativos, organismos
públicos de investigación e instituciones públicas. El uso comercial, como el de una empresa
para su actividad, necesita una licencia aparte de nipegun.

La licencia solo cubre las contribuciones de nipegun. Las partes que proceden de llama.cpp (MIT,
The ggml authors), de FreeToken (Apache 2.0) y de las bibliotecas incluidas conservan sus propias
licencias; sus avisos están en `build/licenses/` y la compilación local los instala en
`~/.local/share/doc/mgs/licenses/`.
