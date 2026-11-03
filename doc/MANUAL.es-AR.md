# Manual de uso

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Índice: [instalación](#installation), [compilación local](#local-build),
[despliegue](#deployment), [chat](#chat), [monitor](#monitor),
[modelos y herramientas](#models), [elegir modelos](#settings-models), [página Modelo](#model-page), [modelos admitidos](#supported-models), [API](#api), [cola de pedidos](#request-queue), [límites de seguridad](#security-limits), [Vulkan](#vulkan), [ROCm](#rocm),
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

El instalador despliega en un solo paso: compila `mgs`, lo instala en
`/usr/local/bin/mgs` y configura los servicios de inferencia y web
([despliegue](#deployment)). No hace falta descargar primero los demás archivos. El script obtiene el archivo
de fuentes de `main` de [este repositorio](https://github.com/nipegun/moe-gguf-server/), valida que esté
completo y actualiza la copia gestionada en `/opt/moe-gguf-server-source`.
Las descargas temporales y las cachés se mantienen bajo `_/temp/` del proyecto.
La instalación inicial necesita curl y certificados CA para obtener el script;
las demás dependencias de descarga se instalan automáticamente si faltan.

Para pasar opciones de compilación y de la web:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ia.ejemplo.com --cpu --jobs 4
```

La copia gestionada vive siempre en `/opt/moe-gguf-server-source`, propiedad de root y
con modo 700. El instalador solo la adopta mientras está vacía o la reutiliza cuando
tiene su marcador de origen; rechaza cualquier otro contenido. Un bloqueo impide instalaciones simultáneas sobre
esa copia. Los siguientes usos de `curl | bash` sincronizan los fuentes, incluidos
los archivos eliminados de `main`, conservando `_/` y la caché de compilación.
La clave API, los modelos y los certificados viven fuera de esa copia
y se conservan. Si falla una descarga, la compilación o la comprobación de las opciones, no se instala el nuevo
ejecutable; se puede repetir el mismo comando. No se necesitan operaciones Git.

Un script guardado sin sus archivos vecinos usa el mismo modo remoto. Si se
ejecuta `bash deploy/install-update-reinstall-debian.sh` desde un proyecto local
completo, instala sus fuentes actuales sin descargarlos. Las opciones después de
`bash -s --` llegan al instalador de compilación. `--help` no requiere root ni
descarga fuentes. `--no-system-deps` también impide instalar las dependencias de
descarga: curl, certificados CA, tar, gzip, rsync, util-linux y coreutils deben
estar disponibles previamente. Los procesos de instalación no consumen el script
que llega por la tubería y todo el registro utiliza `/root/webapp-install.log`.

Las opciones de la web se comprueban antes de compilar, para que un error no aparezca
después de una compilación larga: `--domain NAME` (por defecto, la salida de `hostname -f`,
o la de `hostname` o `uname -n` cuando esa no es un nombre válido), `--certificate PATH` y
`--private-key PATH` (siempre juntas) y `--proxy-ip IP` (consultá
[despliegue](#deployment)). `/opt/moe-gguf-server-source/deploy/configure-web.sh`
sigue disponible para cambiar más adelante la configuración web sin recompilar.

Alpine sigue usando una copia completa local y
`bash deploy/install-update-reinstall-alpine.sh`; necesita Bash instalado y solo
compila para la CPU. En Debian, elegí el backend:

- CUDA (`--cuda`, la opción predeterminada) para GPU de NVIDIA. CUDA Toolkit y el
  driver de NVIDIA tienen que estar instalados previamente; el instalador agrega los paquetes
  NCCL que corresponden a esa versión de CUDA (más abajo). `--no-nccl` desactiva NCCL de forma explícita.
- ROCm (`--rocm`) para GPU de AMD compatibles con ROCm; `--rocm-targets LIST` elige los
  objetivos de GPU. ROCm tiene que estar instalado previamente ([ROCm](#rocm)).
- Vulkan (`--vulkan`) para GPU de AMD, Intel o NVIDIA ([Vulkan](#vulkan)).
- Solo CPU (`--cpu`), para máquinas sin GPU.

`--no-system-deps` deja la gestión de paquetes en manos del administrador.

La interfaz local necesita Node.js 20.19 o posterior en 20.x, o 22.12 o posterior.
`--no-ui` la omite. Otras opciones: `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST` y `--no-openssl`. `CUDACXX` selecciona el compilador de CUDA
y `HIPCXX`, el compilador clang de ROCm.
Los directorios de compilación y copia de fuentes deben ser diferentes, no pueden
contenerse entre sí y deben permanecer dentro de `_/temp/`.

El instalador añade los paquetes NCCL compatibles y el repositorio de NVIDIA
si hace falta, pero nunca instala ni actualiza CUDA Toolkit, ROCm ni los controladores
GPU. Enlaza estáticamente las bibliotecas internas e instala
`/usr/local/bin/mgs`. Ningún instalador usa ni instala sudo.

En Debian/Ubuntu, `libnccl2` y `libnccl-dev` se instalan juntos con la misma versión
exacta seleccionada para la versión de CUDA detectada. Esta orden de APT usa
`--allow-downgrades` para que un paquete NCCL instalado más reciente no bloquee
la instalación de la versión seleccionada. La opción solo se pasa a la orden
de instalación de NCCL.

Si la distribución ofrece un Node.js antiguo, instalá una versión compatible
primero. La interfaz siempre se compila localmente; la compilación no puede descargar
los recursos precompilados originales, que no incluyen las traducciones ni las
rutas de API de este proyecto. Si falla la compilación de la interfaz, se detiene la
instalación.

Desde una copia local completa, elegí CPU o consultá las opciones de instalación:

```bash
bash deploy/install-update-reinstall-debian.sh --domain ia.ejemplo.com --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

En Alpine, instalá Bash con `apk add --no-cache bash` antes de ejecutar su instalador.

Para ejecutar un modelo:

```bash
/usr/local/bin/mgs --model /ruta/al/modelo.gguf
```

No hace falta ninguna otra opción: `mgs` elige los ajustes que dan la mayor velocidad de inferencia en la máquina
donde se ejecuta. El ajuste automático ubica el modelo en la GPU y, si la memoria del sistema alcanza, mantiene
los expertos MoE en la RAM para que la VRAM libre se convierta en la [caché de expertos MoE](#expert-cache);
también elige el [micro-lote y el tamaño del contexto](#context-size). La cantidad de hilos de CPU depende del
procesador ([Hilos de CPU](#threads)), y las capas MTP que haya en el GGUF se activan y se mantienen en la VRAM
([MTP](#mtp-vram)). Cualquier opción indicada de forma explícita (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`,
`--n-cpu-moe`, `--override-tensor`…) prevalece sobre la elección automática. El registro de arranque muestra lo
que se eligió.

Ejecutá `mgs` con tu propio usuario: nunca necesita root. Si se arranca como root, registra el
aviso `mgs is running as root: run it with an unprivileged user instead` (el
servidor MCP local de shells le daría al modelo una shell de root) y sigue arrancando.

Abrí la URL HTTPS mostrada al arrancar, normalmente `https://127.0.0.1:11443/`.
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

Hasta la versión 0.1.34 la búsqueda se detenía en el segundo puerto: un `mgs` arrancado mientras el 11443 estaba ocupado (por ejemplo, por otro `mgs`) terminaba con `couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 11444` en lugar de tomar el siguiente puerto libre.

Sin archivos de certificado, el ejecutable crea un certificado autofirmado y su
clave privada la primera vez que arranca, sin necesitar el comando externo
`openssl`, y los guarda en `~/.mgs/certificates/` (`fullchain.pem` y `privkey.pem`;
la clave solo la podés leer vos). Las ejecuciones siguientes, y cada nueva
compilación, los reutilizan, así que el navegador conserva su excepción de confianza
y no hace falta recargar; solo se crea un par nuevo si faltan esos archivos, no
coinciden o el certificado venció (a los 365 días). El certificado cubre
`localhost`, las direcciones de loopback, el host indicado con `--host` y el nombre
y las direcciones de red que tenía el equipo al crearlo; para que el navegador deje
de avisar, importá `~/.mgs/certificates/fullchain.pem` como de confianza, y borrá
los dos archivos para crear uno nuevo. Si esa carpeta no se puede escribir, cada
ejecución genera su propio certificado en memoria. Indicá juntos `--ssl-cert-file
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
usar CMake directamente desde la raíz, ejecutá `cmake -S build -B _/temp/cmake`;
los comandos habituales de instalación y compilación local siguen siendo los mismos.
HTTPS usa por defecto el OpenSSL del sistema (`MGS_OPENSSL=ON`); en una compilación
directa con CMake, `-DMGS_BUILD_BORINGSSL=ON` o `-DMGS_BUILD_LIBRESSL=ON` descarga y
compila BoringSSL o LibreSSL en su lugar, y `-DMGS_OPENSSL=OFF` sin ninguna de las
dos compila el servidor sin HTTPS.

<a id="local-build"></a>
## Compilación local para uso manual

`build/build-local.sh` compila el proyecto en tu propia computadora e instala una
copia autónoma para tu usuario que lanzás a mano. No instala servicios, no arranca nada en
segundo plano, nunca usa root ni sudo y nunca instala paquetes. Ejecutalo con tu
usuario normal desde cualquier carpeta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

El comando descarga el archivo de fuentes de `main` sin Git en la carpeta privada
de compilación de `/tmp`. El archivo descargado y los fuentes extraídos se eliminan
con los demás temporales al salir. Una copia suelta del script funciona igual.
Desde una copia local completa del proyecto, compilá tus fuentes locales con:

```bash
bash build/build-local.sh
```

Ejecutar este script mediante su ruta absoluta también utiliza el proyecto
completo que lo contiene. El mensaje `Using local sources: ...` confirma qué
carpeta se compila, incluidas sus traducciones de la interfaz. Si termina con
`Error: Compilation failed.`, no se instaló un ejecutable nuevo; la instalación
existente conserva su interfaz anterior. Leé el primer error del compilador
que aparece antes de `ninja: build stopped`, corregí su causa y volvé a ejecutar
el mismo script. Cuando la compilación y la instalación terminen correctamente,
reiniciá `mgs` y recargá la interfaz web.

Errores como que `std::vector` no tenga un miembro llamado `mPushBack` o
`no declaration matches` en `subproc.cpp` indican nombres inconsistentes en
los fuentes. Usá los archivos corregidos y volvé a compilar. La compilación
de la interfaz también requiere que sus importaciones coincidan con los
nombres exportados por los módulos; npm/Vite avisa si falta alguno.

Si SvelteKit muestra `Page options are ignored when router.type === 'hash'`
y señala `fLoad`, las rutas del chat deben exportar el nombre exacto `load`
del framework. Las rutas corregidas conservan la validación de la clave de
API. La compilación ejecuta `svelte-kit sync` antes de generar los recursos
y compilar la interfaz, y se detiene si falla la sincronización. Un fallo en
`Provisioning UI assets` impide la instalación aunque las bibliotecas C++
ya se hayan compilado.

Para pasar opciones en el comando de una línea, usá `bash -s --`; por ejemplo,
para compilar para CPU:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` muestra las opciones sin descargar el archivo de fuentes ni compilar.

El resultado se instala para tu usuario en el prefijo `~/.local`:

| ruta | contenido |
| --- | --- |
| `~/.local/bin/mgs` | El programa que ejecutás: un pequeño lanzador estático que localiza `../lib/mgs/` en relación con su ubicación y arranca el servidor mediante el cargador incluido. |
| `~/.local/lib/mgs/` | El binario del servidor (`mgs`, con la interfaz web y la documentación de la API incorporadas) y todas las bibliotecas compartidas que necesita: el cargador dinámico y la biblioteca C de glibc, el runtime de C++, OpenMP, OpenSSL, los módulos de resolución de nombres de glibc y, en compilaciones CUDA, el runtime de CUDA, cuBLAS y NCCL; en compilaciones ROCm, las bibliotecas de ROCm (HIP, hipBLAS, rocBLAS, hipBLASLt…) y las carpetas de kernels de GPU de rocBLAS y hipBLASLt (`rocblas/library`, `hipblaslt/library`). |
| `~/.local/share/doc/mgs/licenses/` | La licencia PolyForm NonCommercial 1.0.0 del proyecto (`LICENSE`) y los avisos de licencia de terceros copiados desde `build/licenses/` del proyecto (llama.cpp, FreeToken, Swagger UI, cpp-httplib y nlohmann/json). |

Sólo el usuario que ejecuta el script puede usar esta instalación. Todo se resuelve en relación con la ubicación del
lanzador: este ejecuta `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` desde su prefijo, y el servidor y sus
bibliotecas buscan en `lib/mgs/` antes que en cualquier otro sitio, incluido `LD_LIBRARY_PATH`. La única biblioteca
que se sigue tomando del sistema es la del controlador NVIDIA (`libcuda.so.1` y las bibliotecas `libnvidia-*` que
carga), porque debe coincidir con el módulo del kernel en ejecución; con `--vulkan`, el driver de Vulkan del
sistema ([Vulkan](#vulkan)); con `--rocm`, del sistema solo se usa el driver amdgpu del kernel (`/dev/kfd`,
`/dev/dri`), y el lanzador les indica a rocBLAS y hipBLASLt las carpetas de kernels incluidas en `lib/mgs/`
([ROCm](#rocm)). Al terminar, el script enumera, mediante el
cargador incluido, las bibliotecas que cargaría el servidor, falla si alguna procede de fuera de `lib/mgs/` y ejecuta
`--version` a través del lanzador.

Debian añade `~/.local/bin` al `PATH` al iniciar sesión una vez que esa carpeta existe (mediante `~/.profile`), así
que tras volver a iniciar sesión podés ejecutar `mgs` por su nombre; el script muestra un recordatorio mientras no
esté en el `PATH`. Lanzalo cuando quieras y detenlo con Ctrl+C:

```bash
mgs --model /ruta/modelo.gguf
```

También funciona con la ruta completa: `~/.local/bin/mgs --model /ruta/modelo.gguf`.

Abrí `https://127.0.0.1:11443/` para el chat y `https://127.0.0.1:11443/api/doc/` para la
documentación de la API. Todas las opciones de este manual son válidas. Ejecutá `mgs` sin `--model` para arrancar
el router de varios modelos y elegí la carpeta de modelos en Ajustes → Modelos
([elegir y cargar modelos](#settings-models)).

Requisitos, que tenés que instalar vos como root porque el script solo los
comprueba: `cmake`, un compilador C/C++ y la biblioteca C estática
(`build-essential`, que incluye `libc6-dev`), `patchelf`,
`libssl-dev` (o `--no-openssl`), Node.js 20.19+/22.12+ con npm (o `--no-ui`) y las
herramientas del backend elegido: para CUDA (la opción predeterminada), un CUDA Toolkit ya
instalado y NCCL con sus archivos de desarrollo (`libnccl2` y `libnccl-dev`, o `--no-nccl`);
para `--rocm`, un ROCm ya instalado ([ROCm](#rocm)); para `--vulkan`, los paquetes de Vulkan
([Vulkan](#vulkan)); `--cpu` no necesita ninguna. `nvcc` se toma de `CUDACXX`, del
`PATH` o de `/usr/local/cuda/bin/nvcc`. Si Ninja está instalado, se usa.
Las utilidades básicas incluyen `tar`, `coreutils` y `util-linux`. La descarga de
fuentes también requiere `curl`, `ca-certificates` y `gzip`; ninguna dependencia
se instala automáticamente.

Opciones: `--prefix PATH` (otro prefijo de instalación), `--jobs N`, `--cuda`, `--rocm`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--rocm-targets LIST`, `--no-ui`, `--no-openssl`, `--no-nccl`
y `--help`. `CUDACXX` selecciona el compilador de CUDA y `HIPCXX`, el compilador clang de ROCm.

Conviene saber que:

- La compilación se optimiza para la CPU de esta computadora y para las GPU
  presentes al compilar. Recompilá en cada computadora en lugar de copiar la instalación.
- La instalación es reubicable como prefijo: mové juntos `bin/mgs` y `lib/mgs/`, conservando sus niveles
  `bin/` y `lib/`. Arrancá siempre `bin/mgs`; `lib/mgs/mgs` es interno y no debe ejecutarse directamente, porque se
  saltaría el cargador incluido.
- Repetir el script actualiza la instalación. Solo reemplaza `bin/mgs`, `lib/mgs/` y
  `share/doc/mgs/licenses/`. Una copia que ya esté en ejecución sigue funcionando; reiniciala para usar la nueva
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
- Una compilación CUDA necesita el controlador NVIDIA para arrancar; usá `--rocm` o `--vulkan` en computadoras
  con otra GPU y `--cpu` en computadoras sin GPU.

<a id="vulkan"></a>
## Vulkan (GPU de AMD, Intel y NVIDIA)

Además de CUDA (la opción predeterminada) y CPU, tanto `build/build-local.sh` como el instalador de Debian compilan
el backend Vulkan con `--vulkan`, para GPU sin CUDA como las Radeon de AMD. Instalá primero como root las
dependencias de compilación:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Después, con tu usuario normal:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

El instalador de Debian admite la misma opción e instala por su cuenta esos paquetes de Vulkan:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` muestra la GPU, por ejemplo `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- La instalación local incluye el cargador de Vulkan (`libvulkan.so.1`) como cualquier otra biblioteca, pero el
  driver de Vulkan viene del sistema: el cargador lee los drivers registrados en `/usr/share/vulkan/icd.d/` (Mesa
  RADV para AMD, ANV para Intel, el driver de NVIDIA) y estos cargan sus propias bibliotecas del sistema, porque
  tienen que coincidir con el kernel y con el sistema. Compilá en la misma versión de Debian que va a ejecutar el
  binario.
- El usuario que ejecuta `mgs` necesita el nodo de render de la GPU (`/dev/dri/renderD*`). Un usuario con sesión en
  el escritorio ya lo tiene; los demás necesitan el grupo `render` (y `video`). `configure-web.sh` agrega a ambos la
  cuenta del servicio.
- MTP en VRAM, la caché de expertos MoE, la precarga de expertos, `--fit` y el monitor funcionan igual que con CUDA.
  En AMD, el monitor lee de sysfs el uso, la VRAM y la temperatura.
- `mgs` solo cuenta como memoria de GPU la VRAM real (los heaps de memoria «device-local»), nunca GTT. GTT es la
  parte de la RAM del sistema (normalmente la mitad) que una GPU de AMD o Intel puede mapear: no es memoria
  adicional, y la GPU la lee por PCIe, muchas veces más despacio que su VRAM. Las capas MTP, los pesos densos, la
  caché KV y la caché de expertos se reservan en VRAM; los expertos que no entran quedan en la RAM normal, donde los
  calcula la CPU.
- Una GPU integrada (los gráficos de un procesador AMD o Intel) no tiene VRAM propia: todos los heaps de memoria que
  informa son RAM del sistema. `mgs` cuenta esa memoria y la RAM como un único presupuesto y no crea la caché de
  expertos en ella; consultá [GPU integradas](#integrated-gpu).
- `mgs` comprueba dónde guarda de verdad el kernel esa memoria (`/proc/self/fdinfo`, que publican amdgpu, i915 y xe).
  Si otro programa llena la VRAM y el driver mueve parte de ella a GTT, el registro y el monitor lo avisan, y el
  monitor muestra la ubicación real (VRAM y GTT) de cada GPU dedicada de AMD o Intel (una GPU integrada no tiene
  VRAM propia, así que no se comprueba). La caché de expertos se reserva con una prioridad de memoria menor
  (`VK_EXT_memory_priority`, que se usa siempre que la GPU lo ofrece, sin ninguna variable para activarlo), así que el
  driver la mueve antes que las capas MTP, los pesos o la caché KV. Sin ReBAR, los búferes de dispositivo tampoco
  usan la pequeña ventana de VRAM que la CPU puede mapear (256 MiB en una RX 580), que comparte el escritorio y que el
  driver vacía hacia GTT primero, y esa ventana no cuenta como memoria libre para ellos.
- Solo CUDA puede fijar en memoria las páginas del modelo que quedan en la RAM, así que con Vulkan los expertos
  llegan a la GPU a través de memoria paginable. `mgs` los copia mediante un pequeño anillo de staging y pone en
  cola la copia hacia la GPU sin esperar a que termine, así que la GPU y la CPU siguen trabajando mientras se
  transfieren los expertos: en una RTX 4060 Ti con Qwen3.6-35B-A3B Q8_0, eso aumentó el procesamiento del prompt de
  455 a 515 tokens/s y la generación de 20,6 a 23,2 tokens/s. `--load-mode none`, en cambio, copia los expertos a
  memoria fijada del host; antes del anillo de staging, en la misma GPU y con el modelo Q4_K, subió la carga de
  expertos de 2,7 a 13,1 GB/s y la generación de 27,5 a 36,6 tokens/s. En AMD e Intel esa memoria fijada se
  descuenta de GTT: si los expertos no entran, la carga
  falla con `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), como les pasó a los 18,9 GB
  de expertos de ese modelo con 16,8 GB de GTT en una RX 580. Usalo solo cuando los expertos entren en GTT y en la
  RAM libre.
- Con una GPU de NVIDIA a través de Vulkan, y con expertos BF16 en cualquier GPU, los productos de expertos
  convierten primero su entrada en un búfer de trabajo. Ese búfer ocupaba antes 512 KiB por cada token del prompt,
  así que los prompts de unos pocos miles de tokens fallaban con `ErrorOutOfDeviceMemory` en cuanto la caché de
  expertos había llenado la VRAM, salvo que se subiera `--moe-cache-reserve`. Ahora ocupa unos pocos KiB por token
  (32 MiB para un micro-lote de 4096 tokens de Qwen3.6-35B-A3B), muy por debajo de la reserva por defecto. En una
  RTX 4060 Ti con Qwen3.6-35B-A3B Q8_0, eso también aceleró el procesamiento del prompt de 515 a 549 tokens/s.
- Elegí el modelo según la máquina: el GGUF entero debería entrar en la RAM libre más la VRAM, descontando la caché
  KV y los búferes de cálculo. Con 8 GB de VRAM y 16 GB de RAM son unos 18–20 GB de pesos; los archivos más grandes
  se leen del disco una y otra vez y la generación se vuelve muy lenta. El Q4_K de 21,6 GB anterior llegó a
  7,8 tokens/s en una RX 580 con 15 GB de RAM y a 15,9 tokens/s cuando la máquina tuvo 31 GB (10,0 con
  `--spec-type none`).

<a id="rocm"></a>
## ROCm (GPU de AMD)

Tanto `build/build-local.sh` como el instalador de Debian compilan el backend ROCm (HIP) con `--rocm`, para las GPU
de AMD que admite la versión de ROCm instalada. Ninguno de los dos scripts instala ROCm: instalá vos mismo, como
root, ROCm 6.1 o posterior, con HIP, hipBLAS y rocBLAS y sus archivos de desarrollo. Para la compilación local,
instalá también como root las dependencias de compilación comunes (el instalador de Debian las instala por su
cuenta):

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Después, con tu usuario normal:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

El instalador de Debian admite la misma opción:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --rocm
```

- Los scripts encuentran ROCm mediante `hipconfig` (en el `PATH` o en `/opt/rocm/bin/hipconfig`) y usan el
  compilador clang que este indica; `HIPCXX` selecciona otro. Si la configuración falla, indica lo que falta: HIP 6.1
  o posterior con los archivos de desarrollo de hipBLAS y rocBLAS.
- Los objetivos de GPU son los de las GPU presentes al compilar, que se leen con `amdgpu-arch` o `rocminfo`. Si no se
  detecta ninguna GPU, o para compilar para otras, pasá `--rocm-targets` con una lista separada por comas, por
  ejemplo `--rocm-targets gfx1100,gfx1030`.
- Las versiones actuales de ROCm no admiten las GPU GCN hasta Polaris (de `gfx6xx` a `gfx8xx`, como la Radeon RX 580,
  `gfx803`): los scripts avisan de esos objetivos; para ellas, usá `--vulkan`.
- La instalación local copia todas las bibliotecas de ROCm (HIP, hipBLAS, rocBLAS, hipBLASLt…) a `lib/mgs/`, junto con
  las carpetas de kernels de GPU que rocBLAS y hipBLASLt cargan en tiempo de ejecución (`rocblas/library`,
  `hipblaslt/library`); el lanzador les indica a ambas bibliotecas esas copias (`ROCBLAS_TENSILE_LIBPATH`,
  `HIPBLASLT_TENSILE_LIBPATH`), así que nunca leen los kernels de otra versión de ROCm. Del sistema solo viene el
  driver amdgpu del kernel. Las carpetas de kernels ocupan varios gigabytes.
- El `/usr/local/bin/mgs` del instalador de Debian carga las bibliotecas de ROCm del sistema, mediante la ruta de
  ejecución (run path) registrada al compilar; si no encuentra alguna, el instalador se detiene y pide agregar la
  carpeta `lib` de ROCm a `/etc/ld.so.conf.d/` y ejecutar `ldconfig`.
- El usuario que ejecuta `mgs` necesita `/dev/kfd` y `/dev/dri`, es decir, los grupos `render` y `video`; el
  instalador agrega a ambos la cuenta del servicio.

<a id="deployment"></a>
## Despliegue en producción

El instalador ([instalación](#installation)) compila, instala y configura los
servicios web en una sola ejecución:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ia.ejemplo.com
```

Sin `--domain`, el dominio es la salida de `hostname -f` (o la de `hostname` o
`uname -n` cuando esa no es un nombre válido). El dominio tiene que resolver al servidor
de producción. Para cambiar más adelante la configuración web sin recompilar, ejecutá
como root, desde la carpeta de fuentes (`/opt/moe-gguf-server-source` en modo remoto):

```bash
bash deploy/configure-web.sh --domain ia.ejemplo.com
```

`configure-web.sh` admite las mismas opciones de la web que el instalador (`--domain`,
`--certificate`, `--private-key`, `--proxy-ip`), además de `--binary PATH` para indicar otra
ubicación del ejecutable, `--no-system-deps` y `--check`, que solo valida las
opciones. Cada ejecución reescribe toda la configuración a partir de las opciones que recibe:
si se vuelve a ejecutar sin `--proxy-ip`, se elimina el proxy, y sin `--domain`, el
dominio vuelve a ser el de `hostname -f`.

El servicio
`moe-gguf-server-web` ejecuta Apache con la cuenta `moe-gguf-server`, sin root,
en una instancia independiente. En cada arranque selecciona el primer puerto
disponible desde 11080 para HTTP y desde 11443 para HTTPS. Si están ocupados,
incrementa cada serie por separado hasta encontrar un puerto libre, sin superar
65535 ni asignar el mismo puerto a ambos protocolos. Comprueba los listeners TCP
IPv4/IPv6 y reintenta si Apache detecta que un puerto se ha ocupado entre la
selección y la apertura real. Otros errores de configuración detienen el arranque.

HTTP redirige permanentemente al HTTPS seleccionado, conservando ruta y consulta.
La redirección no se almacena en caché. El servicio de inferencia (`moe-gguf-server`,
que también se ejecuta como `moe-gguf-server` y se arranca con `--http`) no escucha en ningún
puerto TCP: escucha en el socket Unix `/opt/moe-gguf-server/_/temp/web/mgs.sock`, que solo
esa cuenta puede abrir, y Apache le reenvía los pedidos por él. Antes de escuchar, `mgs`
borra el archivo de socket que haya dejado un servidor que ya no está en marcha, pero nunca
el socket de un servidor que todavía responde.
La instalación requiere root; ninguno de esos servicios necesita ejecutarse como
root. No se configura HAProxy ni se importan ni modifican los sitios del Apache
compartido.

Al detener el servicio de inferencia (`systemctl stop moe-gguf-server` o `systemctl restart moe-gguf-server`), igual que con Ctrl+C o un `SIGTERM` cuando ejecutás `mgs` vos mismo, se cierra de forma ordenada en un par de segundos, también en mitad de un prompt largo: las peticiones en curso reciben un error 503 (`The server is shutting down.`) o, si van en streaming, su stream termina. Un segundo Ctrl+C o `SIGTERM` termina el proceso en el acto, y mientras el modelo se está cargando ya lo hace el primero.

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
web (`web-domain`), el proxy de confianza (`web-proxy-ip`, solo con `--proxy-ip`), la
plantilla de Apache (`apache2.conf.in`) y los ajustes de la interfaz
(`ui-settings.json`), todos propiedad de `moe-gguf-server` y con modo 0600.
El registro de instalación se guarda en `/root/webapp-install.log`, y
`/root/webapp-credentials.txt` contiene todas las credenciales que crea el despliegue: la
cuenta del servicio (`moe-gguf-server`, sin contraseña y con el inicio de sesión desactivado) y
la clave API. Ambos archivos tienen permisos 600. La
reconfiguración conserva claves, modelos y certificados.

Para usar un certificado propio:

```bash
bash deploy/configure-web.sh --domain ia.ejemplo.com \
  --certificate /ruta/fullchain.pem --private-key /ruta/privkey.pem
```

El certificado se guarda en `/opt/moe-gguf-server/certificates/` (`fullchain.pem` y
`privkey.pem`, propiedad de la cuenta `moe-gguf-server`, que es la única que puede
leer la clave). Si no aportás esos archivos, se conserva el certificado que ya está
ahí o, la primera vez, se crea uno autofirmado, que reutilizan todas las
reinstalaciones siguientes, así que el navegador conserva su excepción de confianza.
Configurá su confianza en el cliente o proporcioná un certificado que ya sea de
confianza. Los archivos aportados se copian a esa carpeta; al renovarlos hay que
actualizar las copias y reiniciar `moe-gguf-server-web`. El script no obtiene ni renueva certificados Let's Encrypt.
Repetí el comando después de renovar los archivos originales.

Cuando hay un proxy inverso (HAProxy, nginx…) delante de Apache, indicá su dirección
con `--proxy-ip` (IPv4 o IPv6), al instalador o a `configure-web.sh`:

```bash
bash deploy/configure-web.sh --domain ia.ejemplo.com --proxy-ip 10.7.10.1
```

Apache carga entonces `mod_remoteip`. En HTTPS exige la cabecera del protocolo PROXY
solo a esa dirección (`RemoteIPProxyProtocol On`, con todas las demás direcciones en
`RemoteIPProxyProtocolExceptions`), así que las demás computadoras siguen conectándose directamente; en
HTTP confía en la cabecera `X-Forwarded-For` de esa dirección (`RemoteIPHeader`,
`RemoteIPTrustedProxy`). Los registros de acceso guardan la dirección real del cliente. Sin
`--proxy-ip`, no se configura ningún proxy.

El servicio funciona en modo router. Guardá GGUF legibles en
`/opt/moe-gguf-server/models` o utilizá la gestión de modelos de la interfaz.
Para opciones específicas, utilizá un override de systemd en Debian/Ubuntu o la
configuración OpenRC en Alpine. Mantené el servicio de inferencia en su socket Unix.
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

1. Abrí la URL e ingresá la clave API cuando se solicite o en Ajustes → General.
2. Seleccioná un modelo. En modo router, cargá uno disponible si no hay ninguno cargado; mientras no haya ninguno, el
   selector de modelos enlaza a Ajustes → Modelos, donde elegís la carpeta de tus modelos
   ([elegir y cargar modelos](#settings-models)).
3. Escribí un mensaje y envialo. Enter envía y Mayús+Enter inserta un salto de línea
   de forma predeterminada; podés cambiarlo en los ajustes.
4. Adjuntá archivos. Imágenes, audio y video requieren modalidades compatibles.
   Los PDF pueden extraerse como texto o convertirse en imágenes; los modelos
   sin visión utilizan texto.
5. Podés detener la generación, omitir el razonamiento cuando sea compatible,
   editar mensajes, regenerar respuestas o bifurcar conversaciones.

Las tres líneas de la parte superior de la barra lateral la expanden y la contraen. Muestra **Chat nuevo**,
**Monitor**, **Modelo** y **Ajustes**, seguidos de las conversaciones: **Fijadas**, **Chats recientes** (los cinco
últimos con los que trabajaste; un chat al que le volvés a enviar un mensaje sube ahí) y **Chats** (el resto, primero
los más nuevos). Cada chat tiene un globo de diálogo con un número del 01 al 99, que recibe al crearse y conserva
durante toda su vida (el número que queda libre al eliminar un chat pasa al siguiente chat nuevo), para poder
distinguirlo mientras la barra lateral está contraída, cuando solo se muestran los íconos del menú y los globos. Los
chats se pueden fijar, renombrar, eliminar y seleccionar varios a la vez. El campo de búsqueda al lado de las tres
líneas (Ctrl+K, o ⌘K en una Mac, expande la barra lateral y le da el foco) encuentra, sin distinguir mayúsculas ni
acentos, los chats por su título y por el contenido de sus mensajes, las páginas y sus partes, todos los ajustes, los
modelos, las herramientas y los servidores MCP. Hacé clic en un resultado, o presioná Enter para ir al primero, y se
abre su chat o su página, con lo encontrado resaltado por un momento. Las conversaciones las guarda el servidor en
`~/.mgs/chats/` del usuario que ejecuta `mgs` (en el servicio desplegado, `/opt/moe-gguf-server/.mgs/chats/`), un
archivo por chat con el nombre de la fecha en que se creó, en UTC (`2026-11-03@14-59-59.723.jsonl`), así que todos los
navegadores que abren la interfaz ven los mismos chats, y una página abierta muestra en pocos segundos los chats que
otro navegador crea, modifica o elimina. Los ajustes
los guarda el servidor en `~/.mgs/config/ui-settings.json` del usuario que ejecuta
`mgs` (en el servicio desplegado, `/opt/moe-gguf-server/.mgs/config/ui-settings.json`),
así que todos los navegadores que abren la interfaz los comparten: parámetros de
generación, opciones de visualización, tema, idioma, servidores MCP, modelos
favoritos, herramientas desactivadas y siempre permitidas, esfuerzo de razonamiento
predeterminado y la carpeta de modelos de Ajustes → Modelos. Solo la clave API se queda en cada navegador, porque
autentica el pedido que carga el resto, así que hay que ingresarla una vez en cada navegador.
Los ajustes incluyen importación y exportación. Las exportaciones pueden contener adjuntos y, si lo
seleccionás expresamente, claves API o cabeceras de autorización MCP.

El idioma se elige en Ajustes → General. Los 15 idiomas aparecen en este orden:
`de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`,
`ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; el predeterminado es `en-US`.
El portugués de Brasil y el de Portugal tienen traducciones independientes.
En hebreo la interfaz se muestra de derecha a izquierda; la entrada del chat y
los mensajes de usuario sin Markdown siguen la dirección de su propio texto,
mientras que el código y las salidas de terminal se muestran de izquierda a derecha.
El cambio guarda la selección en los ajustes del servidor y recarga la
interfaz; el servidor la publica en `/api/config.js`, así que todos los navegadores
arrancan en ese idioma. Guardá primero cualquier otro ajuste pendiente. El parámetro
`lang` de la URL lo reemplaza en esa página y conserva la elección cuando no se puede
guardar.
Por ejemplo, agregá `?lang=ja-JP` a la URL de la interfaz para elegir japonés.

Ajustes → General → **Responder siempre en el idioma de la interfaz** viene
desactivado por defecto. Activalo y guardá los ajustes para indicarle al
modelo que use el idioma y la variante regional de la interfaz en cada nueva
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
de agregarse a las peticiones siguientes. Las respuestas existentes no se
traducen. El código, los comandos, los nombres, las citas y las traducciones
pedidas expresamente conservan el idioma necesario; las explicaciones que
los acompañan siguen el idioma de la interfaz. El grado de cumplimiento
depende del modelo: no es una traducción automática ni una garantía de
obediencia perfecta. Este ajuste del navegador no afecta a clientes externos
de la API.

Los temas son sistema, claro y oscuro. Igual que el idioma, el servidor publica el tema guardado en
`/api/config.js`, así que cada página se abre con él antes de dibujarse, sin guardar nada en el navegador; con
sistema, la interfaz sigue al instante el modo claro u oscuro del sistema operativo, también mientras está abierta.
Los ajustes incluyen temperatura,
muestreadores, penalizaciones y límites de turnos del agente. Algunas funciones
dependen de opciones del servidor o del modelo seleccionado.

Después de cada respuesta, la interfaz vuelve a enviar la conversación, si el
servidor está libre, para que ya esté codificada en la caché KV mientras la leés y
el siguiente turno empiece antes. Está activado por defecto; desactivá Ajustes →
Desarrollo → **Preparar la caché KV después de cada respuesta** para evitarlo.

Mientras el servidor resuelve otra consulta, el mensaje que envíes espera en la
[cola](#request-queue), y su respuesta muestra un aviso en lugar del texto de progreso hasta
que empieza: «En cola: antes se está resolviendo otra consulta enviada directamente a la API
(consultas adelante: N).» mientras se ejecuta una consulta que un programa envió directamente a
la API, o, si no, «En cola: antes se está resolviendo otra consulta de la interfaz web (consultas
adelante: N).» (otra pestaña u otro navegador, un título generado o la preparación de la caché KV).
N cuenta las consultas que se ejecutan y esperan adelante de la tuya. El aviso desaparece apenas
empieza tu consulta.

<a id="monitor"></a>
## Monitor y barra de estado

**Monitor** está debajo de **Chat nuevo** en la barra lateral (dirección `#/monitor`). Mientras está abierto se
actualiza cada dos segundos y muestra:

- **Uso**: primero una tarjeta por cada GPU del equipo (NVIDIA, AMD o Intel, aunque ningún modelo la use), titulada
  con su etiqueta en negrita y su nombre (por ejemplo **gpu-1**: NVIDIA GeForce RTX 4060 Ti; con varias GPU, gpu-1,
  gpu-2, etc.), con su uso, el uso del controlador de memoria (ancho de banda), la VRAM en uso, la temperatura y el
  consumo frente a su límite, y después una tarjeta por procesador, titulada de la misma forma con su modelo (por
  ejemplo **cpu-1**: AMD Ryzen 9 5950X 16-Core Processor; con varios procesadores, cpu-1, cpu-2, etc.), con su uso,
  una barra vertical chica por cada CPU lógica de ese procesador y la carga promedio del sistema. Todas las barras
  de uso son verdes y se llenan de rojo a medida que aumenta el uso; una barra cuyo valor se desconoce queda gris.
  Las GPU de NVIDIA se leen mediante la biblioteca NVML del controlador y las de AMD mediante `gpu_busy_percent` en
  sysfs; cualquier otra GPU (Intel, o NVIDIA sin NVML) no muestra su uso, y su nombre y su memoria llegan del
  backend de inferencia en cuanto un modelo la usa. En una pantalla ancha, las tarjetas se ubican dos por fila, cada
  una con la mitad del ancho, como las barras de memoria; con una cantidad impar de tarjetas, la última fila tiene
  tres.
- **Memoria**: una barra por GPU (VRAM) y otra para la RAM; una GPU sin datos de memoria, como una GPU de Intel que
  todavía no usa ningún modelo, no tiene barra. Cada barra se divide según el uso de la memoria, cada uso siempre
  con el mismo color, y su leyenda enumera todos los segmentos con su tamaño y su proporción del total, uno por
  línea y en el mismo orden que en la barra. Pasá el puntero por un segmento para ver su tamaño.

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
  | Otros | Memoria en uso que no explica nada de lo anterior: otros procesos, el escritorio, contextos del driver, el proyector multimodal. |
  | Libre | El resto del dispositivo o de la RAM. |

  Los pesos mapeados desde el archivo del modelo cuentan como RAM aunque el kernel considere disponible esa caché de
  páginas, así que los segmentos pueden sumar más que la memoria «en uso»; en ese caso la barra no muestra «Otros» y
  muestra menos memoria libre. La caché de expertos aparece después de la primera solicitud, que es cuando se crea.
  En modo router las barras incluyen todos los modelos cargados, y el monitor nunca carga ni despierta un modelo.

La **barra de estado**, en la parte inferior de todas las páginas, muestra primero el uso de cada GPU y después el
de cada procesador físico, cada uno en su propia barra y con las mismas etiquetas que las tarjetas del monitor:
`gpu-1`, `gpu-2`... y `cpu-1`, `cpu-2`... Las barras tienen el mismo tamaño, se actualizan cada dos segundos y, como
las del monitor, son verdes y se llenan de rojo a medida que aumenta el uso; el texto emergente de cada una indica el
nombre del dispositivo. Sin GPU muestra una barra `GPU` vacía y, cuando las barras no entran en una pantalla angosta,
la barra de estado se desplaza en horizontal. Hacé clic en ella para abrir el monitor. Sigue funcionando mientras
carga el modelo. Se muestra por defecto; desactivá Ajustes → Visualización → **Mostrar la barra de estado** para
ocultarla.

<a id="models"></a>
## Modelos, herramientas y MCP

`mgs` solo carga modelos MoE. Al arrancar, lee la cabecera del GGUF que se pasa con `--model` (y la del que se pasa con `--spec-draft-model`); si es un modelo denso, como GPT-2 o Phi-2, `mgs` registra un error como este y termina con el código de salida 1 antes de cargar nada:

```text
srv    mgs_server: the model '/ruta/al/modelo.gguf' is not a Mixture of Experts (MoE) model
srv    mgs_server: mgs only loads MoE models; dense models are not supported
srv    mgs_server: exiting due to a dense model
```

En modo router solo termina el proceso de ese modelo: `/api/v1/models` lo marca como fallido y los demás modelos siguen funcionando. Las implementaciones de decodificadores densos no se compilan, pero se mantienen las capas densas que forman parte de un MoE y una cabeza exclusivamente MTP compatible que se pasa explícitamente (`--spec-draft-model`).

<a id="settings-models"></a>
### Elegir y cargar modelos en Ajustes → Modelos

Si lo arrancás sin `--model`, `mgs` funciona en modo router: no carga ningún modelo por su cuenta y lista los modelos
que encuentra, para que los cargues desde la interfaz. Ajustes → Modelos tiene dos partes:

- **Carpeta de modelos.** Presioná **Elegir carpeta**: se abre un explorador de las carpetas del equipo en el que se
  ejecuta `mgs`, en la carpeta de modelos actual (o en tu carpeta personal). Hacé clic en una carpeta para abrirla, usá
  la flecha para subir y la casa para ir a tu carpeta personal, o escribí una ruta en su campo y presioná Enter (una `~`
  al principio es tu carpeta personal); **Mostrar carpetas ocultas** agrega las carpetas cuyo nombre empieza con un
  punto, y el explorador cuenta los archivos `.gguf` de la carpeta que muestra. Presioná **Elegir** y la ruta absoluta
  de esa carpeta completa el campo y se usa en el acto. La carpeta puede tener un archivo `.gguf` por modelo, o una
  subcarpeta por modelo con las partes de un modelo dividido (`NOMBRE-00001-of-0000N.gguf`) y su proyector multimodal
  (`mmproj*.gguf`). `mgs` comprueba que puede leer la carpeta, la guarda con los demás ajustes y la analiza en el acto;
  cada arranque posterior analiza la misma carpeta. Para volver a leerla después de agregar o borrar archivos, elegí de
  nuevo la misma carpeta. Los archivos GGUF densos, incluidas las cabezas MTP separadas, quedan fuera de la lista,
  porque `mgs` solo carga modelos MoE. Elegir otra carpeta descarga de la memoria los modelos en ejecución de la
  anterior. También se listan los modelos descargados en `~/.cache/mgs/`. Si `mgs` se arrancó con `--models-dir`, esa
  carpeta prevalece: el botón **Elegir carpeta** está desactivado y la página lo indica.
- **Modelos disponibles.** Cada modelo con su estado (sin cargar, cargando con su porcentaje, cargado, en reposo o
  error al cargar), su archivo y su tamaño en disco, y un botón para cargarlo o descargarlo de la memoria. Cargar un
  modelo acá también lo selecciona en el chat.

Los modelos cargados y cómo se cargaron aparecen en la [página Modelo](#model-page), a la que enlaza el final de
Ajustes → Modelos. El selector de modelos del chat lista los mismos modelos y carga el que elijas; mientras la lista
está vacía, enlaza a Ajustes → Modelos. Si lo arrancás con `--model`, Ajustes → Modelos solo indica que el modelo no
se puede cambiar sin reiniciar `mgs`.

Solo hay un modelo cargado a la vez: cargar otro, acá, desde el selector del chat o mediante la API, descarga el que
estaba cargado (`--models-max N` permite N a la vez). Un chat nuevo usa el modelo cargado en ese momento. Un chat
conserva su propio modelo, el de su última respuesta: cuando le enviás un mensaje mientras hay otro modelo cargado,
primero se descarga ese y se carga el modelo del chat. Si el modelo del chat ya no está donde estaba (salió de la
lista o ahora es otro archivo), un cuadro de diálogo lo indica y te permite cancelar el mensaje, que queda en el
campo, o enviarlo con el modelo cargado.

La carpeta se guarda en `~/.mgs/config/ui-settings.json` con los demás ajustes de la interfaz, como la sección
`modelsFolder`. `mgs` la escribe cuando elegís la carpeta y la lee en cada arranque:

```json
{
  "modelsFolder": "/home/usuario/modelos"
}
```

El servicio desplegado (`deploy/`) arranca `mgs` con `--models-dir /opt/moe-gguf-server/models`, así que esa es la
carpeta que lista.

<a id="model-page"></a>
### Página Modelo

**Modelo**, debajo de **Monitor** en la barra lateral (dirección `#/model`), muestra el modelo cargado en dos cajas:
**Archivo del modelo**, con lo que contiene su archivo GGUF (descripción, arquitectura, archivo, tamaño en disco,
parámetros, capas, capas MTP, contexto de entrenamiento y proyector multimodal), y **Parámetros de carga**, con cómo
se cargó: los parámetros que `mgs` eligió por su cuenta o que pasaste vos (tamaño del contexto, slots en paralelo y
contexto por slot, lote y micro-lote, hilos de generación y de lotes, flash attention, tipos de la caché KV, `--fit`,
MTP, decodificación especulativa con su cantidad máxima de tokens de borrador, caché de expertos MoE y dispositivos)
y, en modo router, la línea de comandos de su proceso. Cada estado activo (flash attention,
`--fit`, MTP, decodificación especulativa y caché de expertos MoE) se muestra en verde. Flash attention indica si
`auto` lo activó; MTP muestra cuántos tokens de borrador pide en cada paso, o que todavía los está midiendo (de 0 a
3), o que el borrador no conviene en este equipo, y la caché de expertos MoE indica si se creó, con su tamaño
automático o pedido: lo decide la primera solicitud después de cargar el modelo, así que hasta entonces lo indica, y
queda desactivada en una GPU integrada o cuando todos los expertos ya están en la VRAM. Estos estados en vivo vienen
del informe del [Monitor](#monitor), que también tiene la memoria que usa cada dispositivo, por propósito. Mientras no
hay ningún modelo cargado, la página enlaza a Ajustes → Modelos.

<a id="supported-models"></a>
### Familias de modelos admitidas

| Familia | Arquitectura GGUF | Notas |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP cuando el GGUF lo trae. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP cuando el GGUF lo trae. Su tabla de embeddings por capa (unos 29 GB) se lee del archivo del modelo bajo demanda, así que conviene tener el archivo en un disco rápido. Los contextos largos se calculan con atención densa bajo la selección dispersa: el resultado es correcto, pero sin la mejora de velocidad de los kernels dispersos. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; imágenes con su mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Los GGUF publicados no traen MTP. |
| DeepSeek-V4.1-Flash | `deepseek41` | Carga las dos disposiciones publicadas: los archivos DwarfStar de antirez (como `antirez/deepseek-v4.1-flash-gguf`) y los del ecosistema llama.cpp. Su tabla Engram (unos 188 GiB dentro del archivo) nunca se carga: cada token lee sus filas del archivo del modelo, así que el archivo debe seguir disponible mientras `mgs` se ejecuta, mejor en un SSD, y hace falta Linux u otro sistema POSIX. Los archivos sin plantilla de chat usan una integrada, y `reasoning_effort` acepta un número del 1 al 100. Sin MTP ni imágenes. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |
| Flint | `flint` | Los GGUF publicados no traen MTP. |

Gemma 4 y los modelos DeepSeek anteriores a la V4.1 ya no se admiten (consultá la [compatibilidad eliminada](#removed-compatibility)). Tampoco las demás arquitecturas MoE heredadas de llama.cpp, como gpt-oss, Qwen3-MoE, Qwen3-Next, GLM-4.x, Granite 4, MiniMax-M2 o Kimi-Linear.

Con varias GPU seleccionadas, MTP comprueba todas antes de recurrir a la RAM. Su estimación de memoria incluye la cabeza de salida. Los búferes MTP integrados se cargan antes de los demás pesos; un GGUF MTP externo pasado con `--spec-draft-model` se carga antes del modelo principal, también con `--fit off`, y activa MTP por su cuenta (consultá [MTP automático](#performance)); un MTP que solo está en otro GGUF nunca se usa si no se pasa de esa forma. La selección explícita de dispositivos sigue determinando qué GPU están disponibles.

Usá `--reasoning on` o `--reasoning off` para controlar el razonamiento. Cuando el razonamiento está desactivado (con esa opción, con el interruptor de razonamiento del chat o, en un pedido, con `enable_thinking: false` en `chat_template_kwargs` o `reasoning_effort: "none"`), `mgs` también cierra cualquier bloque de razonamiento en cuanto el modelo lo abre, así que los modelos cuya plantilla de chat ignora ese ajuste, como Flint, también responden sin razonar. Se rechaza `enable_thinking` dentro de `--chat-template-kwargs`. Usá los nombres documentados de las opciones, con guiones; las variantes con guiones bajos se rechazan. Pasá las listas como un valor separado por comas, en vez de repetir la opción. Solo `--spec-type` admite repetición para combinar modos especulativos.

El selector distingue modelos disponibles, cargados y favoritos. Los detalles
incluyen contexto, modalidades, cuantización y plantilla de chat. El modo router
mantiene procesos separados para los modelos cargados; las peticiones identifican
el modelo mediante el campo o parámetro `model`.

Configurá los servidores MCP en Ajustes → Servidores MCP, por URL y, si hace falta, con cabeceras de autorización.
Podés consultar registros de conexión, instrucciones, recursos y herramientas.
El proxy CORS opcional debe activarse mediante la opción correspondiente del motor; sólo
devuelve contenido inerte y no se deja abrir como página (consultá [límites de seguridad](#security-limits)).
Si dos servidores MCP activos ofrecen una herramienta con el mismo nombre, no se le ofrece al modelo
ninguna de las dos y la interfaz muestra un aviso: desactivá todos esos servidores menos uno para usarla.
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

Activá **Servidor MCP local de shells** en Ajustes → Servidores MCP, o arrancá `mgs` con `--local-shell-mcp-server`,
para que los modelos del chat web puedan usar la terminal del equipo en el que se ejecuta `mgs`:

```bash
mgs --model /ruta/al/modelo.gguf --local-shell-mcp-server
```

- El interruptor tiene efecto inmediato, sin reiniciar `mgs`, y se guarda en la sección `localShellMcpServer` de
  `~/.mgs/config/ui-settings.json`, así que los arranques siguientes lo vuelven a activar
  (`POST /api/mcp/shell/enabled` con `{"enabled": true}` hace lo mismo). Con `--local-shell-mcp-server` queda activado
  y el interruptor está deshabilitado. Sin clave API, solo un navegador del equipo donde se ejecuta `mgs` puede usar
  el interruptor, así que nadie más en la red puede activar un shell ahí; con clave API, puede hacerlo cualquiera que
  tenga la clave.

- Al arrancar, `mgs` comprueba qué shells están instalados y ofrece una herramienta para cada uno: `bash.run`
  (Bash), `zsh.run` (Zsh), `sh.run` (sh) y `psh.run` (PowerShell, `pwsh`). El registro muestra los shells
  encontrados; un shell que no está instalado no tiene herramienta.
- La interfaz lo detecta, lo agrega a los servidores MCP como **local shell mcp server** y lo activa. Podés desactivarlo ahí, o
  solo para una conversación, y se conserva tu elección. Cuando se desactiva el servidor, la entrada desaparece.
- Cada herramienta recibe `command` y, de forma opcional, `cwd` (directorio de trabajo; por defecto, el directorio
  personal del usuario que ejecuta `mgs`) y `timeout` (segundos; 60 por defecto y 3600 como máximo). La respuesta
  es un objeto JSON con `exit_code`, `stdout` y `stderr`, además de `signal`, `timed_out`, `cancelled`, `timeout`,
  `seconds`, `shell` y `cwd`; de las salidas largas solo se conservan el principio y el final.
- Cada comando se ejecuta en un proceso nuevo, sin terminal y con la entrada vacía: `cd` y las variables no pasan a
  la llamada siguiente, y no se pueden usar programas interactivos (editores, preguntas). Cuando se agota el tiempo
  límite o se cancela la petición, se matan todos los procesos que haya iniciado el comando. Los procesos que
  queden en segundo plano tienen que redirigir su salida.
- La interfaz pide permiso antes de cada llamada, salvo que elijas permitir siempre la herramienta.
- `sudo` funciona sin terminal (Linux): cuando necesita una contraseña, se abre una ventana en el centro de la
  página, estés en la página que estés, con el mensaje de sudo (`[sudo] password for usuario:`), la herramienta y el
  comando que la pide, un campo de contraseña y, debajo, los botones **Aceptar** y **Cancelar**. El cursor ya está
  en el campo de contraseña, así que podés escribirla directamente; **Enter** acepta y **Escape** cancela; un clic
  afuera de la ventana no la cierra. La contraseña solo llega a sudo en el equipo donde
  se ejecuta `mgs`: no se guarda en la conversación, no se escribe en los registros y el modelo nunca la ve.
  Mientras la ventana espera, el tiempo límite del comando deja de contar; si apretás **Cancelar**, o nadie responde
  en 5 minutos, sudo falla. sudo la vuelve a pedir en cada llamada, porque cada una es un proceso nuevo, y hasta
  tres veces si la contraseña es incorrecta; en ese caso la ventana avisa que puede que la contraseña anterior fuera
  incorrecta. Los comandos se ejecutan en una sesión propia, sin la terminal desde la que se arrancó `mgs`, así que
  sudo nunca pide la contraseña en esa terminal. Solo puede responder la interfaz web de `mgs`: con otros clientes MCP, sudo
  falla como antes por no tener terminal. El servicio desplegado se ejecuta con la cuenta `moe-gguf-server`, que
  normalmente no tiene contraseña ni permiso de sudo, así que está pensado para `mgs` ejecutado con tu propio
  usuario (por ejemplo, la compilación de `~/.local`). Cuando `mgs` no tiene la variable `DISPLAY`, los comandos
  reciben `DISPLAY=:mgs-askpass`, porque sudo solo usa su programa de contraseña sin terminal cuando existe
  `DISPLAY`; los programas que intentan abrir una ventana fallan al instante.
- En el chat, cada llamada se muestra con el nombre de su herramienta, como `bash.run`. Al desplegarla aparecen
  hasta tres cajas grises con letra monoespaciada: **stdin** con el comando tal cual, **stdout** con la salida y
  **stderr** con el texto del error. stdout solo aparece si el comando escribió algo y stderr solo si hubo un error;
  el código de salida va junto a su título, por ejemplo `stdout (código 2):` (junto a stdin cuando no hay salida).
  La caja stdin tiene el borde blanco (gris oscuro en el tema claro); la caja stdout, verde, y la caja stderr,
  rojo, con todas las herramientas de shell (`bash.run`, `zsh.run`, `sh.run` y `psh.run`).
- Activá **Ajustes → Visualización → Desplegar automáticamente los comandos de la shell local** para que cada
  llamada se abra sola: cada caja aparece en cuanto se conoce su contenido, stdin mientras el modelo escribe el
  comando y stdout y stderr cuando el comando termina.
- Cada llamada a una herramienta de shell deja su propio archivo en `~/.mgs/local-shell-mcp-server/` (en el
  directorio personal del usuario que ejecuta `mgs`; en el servicio desplegado,
  `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), con el nombre de la fecha y la hora locales de la llamada,
  con sus milisegundos, por ejemplo `y2026m10d12h23m59s59ms731.jsonl`; si empiezan varias llamadas en el mismo
  milisegundo se agrega `-2`, `-3`… Es
  un archivo JSONL, con un objeto JSON por línea: primero el comando (`"event": "command"`, con la herramienta, el
  shell, el directorio de trabajo y el tiempo límite), que se escribe antes de ejecutarlo; después toda la salida,
  sin recortes, a medida que llega, una línea por fragmento (`"event": "stdout"` o `"stderr"`, con el texto en
  `data`) en el orden en que la escribió el comando, y por último el resultado (`"event": "result"`: código de
  salida, señal, si se agotó el tiempo o se canceló, los segundos que tardó y los bytes de stdout y de stderr) o el
  motivo por el que no pudo arrancar (`"event": "error"`). Una llamada rechazada antes de ejecutarse (con una herramienta desconocida, sin comando,
  con un directorio de trabajo que no existe o con un tiempo límite no válido) también deja su archivo, con el
  comando tal como se pidió y el error. El chat solo conserva el principio y el final de las salidas muy largas,
  pero el archivo tiene todo, y ya contiene la salida aunque `mgs` se detenga mientras se ejecuta el comando. Para
  leer todo el stdout de una llamada, ejecutá `jq -j 'select(.event == "stdout") | .data' ARCHIVO`. JSON solo admite
  texto, así que los bytes que no son UTF-8 (una salida binaria) se reemplazan por `�`; un comando que escribe
  gigabytes deja un archivo de ese tamaño. Solo ese usuario puede leer la carpeta y los archivos, porque la salida
  de un comando puede contener secretos; nada los borra, así que eliminá los viejos cuando ya no los necesites.

En el servicio desplegado, agregá `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` a la sección `[Service]` de un override
de systemd (`systemctl edit moe-gguf-server`) y reiniciá `moe-gguf-server`; en Alpine, exportá la misma variable
en la configuración del servicio OpenRC.

**Seguridad.** Los comandos se ejecutan con los permisos del usuario que ejecuta `mgs` (en el servicio desplegado,
la cuenta `moe-gguf-server`, nunca `root`). Cualquiera que pueda abrir la interfaz o llamar a la API puede
ejecutarlos, así que activá la opción solo en entornos de confianza y protegé el servidor con una clave de API. Se
rechazan las peticiones de páginas de otros sitios (el origen CORS predeterminado es localhost) y el registro
anota cada comando con su código de salida. Leé el comando que muestra la ventana de la contraseña antes de escribir
una contraseña de sudo: la contraseña autoriza ese comando. El servidor solo se la entrega a un programa de
contraseña cuyo proceso padre sea sudo ejecutándose como root para tu usuario, así que un comando no puede lanzar
ese programa por su cuenta para leerla.

<a id="api"></a>
## API HTTP

Los ejemplos usan 11443; si se eligió otro puerto, utilizar el valor de `ports.json`.

La raíz predeterminada es `/api` y la interfaz sigue en `/`. Las rutas originales
`/v1/...` de la raíz ya no están registradas. Los clientes compatibles con OpenAI
deben usar `https://ia.ejemplo.com:11443/api/v1` como URL base: las rutas de OpenAI (`/api/v1/models`,
`/api/v1/chat/completions`, `/api/v1/responses`, `/api/v1/audio/transcriptions` y sus variantes
`input_tokens`) solo existen bajo `/api/v1`, y la comprobación de salud, solo en `/api/health`.
`/api/rerank` y `/api/v1/rerank` se mantienen las dos para los clientes TEI/Jina, y las rutas nativas
(`/api/completions`, `/api/embeddings`…) tienen manejadores distintos de sus homónimas de `/api/v1`.
Los nombres de campos mantienen su
compatibilidad original; las interfaces C/C++ del motor conservan sus funciones actuales,
pero no las que estaban marcadas como obsoletas o no tenían uso (consultá [compatibilidad eliminada](#removed-compatibility)).

```bash
curl https://ia.ejemplo.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hola"}]}'
```

Asigná la clave instalada a `MGS_API_KEY` en tu propia consola (solo la usa `curl`; `mgs` lee su clave de
`--api-key`, `MGS_ARG_API_KEY` o `--api-key-file`). Swagger está en
`/api/doc/` y su especificación en `/api/doc/openapi.json`, generada por el servidor
en cada solicitud sin un archivo JSON estático. Incluye las operaciones
nativas, embeddings, tokenización, reranking, modelos y flujos reanudables.
El inventario refleja las rutas registradas por el proceso, incluidas las del
router y las de compatibilidad opcional. Las funciones desactivadas pueden
responder con 403; una clave no válida produce 401 y un modelo cargando, 503.
`/api/health` y `/api/v1/models` mantienen su acceso público con `GET` y `HEAD`;
consultá en [límites de seguridad](#security-limits) las rutas que siempre exigen la clave.

`--api-prefix /api/personalizado` cambia el prefijo de inferencia. La interfaz
lo obtiene de `/api/config.js`; Swagger permanece en `/api/doc/`, espacio reservado
que no se puede utilizar como prefijo de inferencia. Los procesos hijos del
router usan `/api` internamente. Se rechazan los prefijos fuera de `/api/` y los
que terminan con barra.

El monitor usa `GET /api/monitor` (informe completo), `GET /api/monitor/memory` (instantáneas de memoria de los
modelos cargados) y `GET /api/monitor/usage` (solo el uso de CPU y GPU, disponible también mientras carga el modelo).
Leen contadores e instantáneas que publica el bucle de inferencia, nunca la cola de tareas, así que consultarlos no
despierta un modelo en reposo ni reinicia su temporizador de inactividad. Cada modelo del informe lleva
`expert_cache` (con `active` y `pending`) y, con MTP, `mtp`: `draft_n` (los tokens de borrador que pide en cada
paso, 0 cuando el borrador no conviene y -1 mientras se miden), `draft_n_max` y `measuring`. Sus campos se
describen en Swagger.

La interfaz lee sus ajustes con `GET /api/ui/settings` (`{"settings": DOCUMENTO}`, o `{"settings": null}` antes del
primer guardado) y los guarda con `POST /api/ui/settings` (`{"settings": DOCUMENTO}`, que reemplaza entero
`~/.mgs/config/ui-settings.json`, con modo 0600). Las dos piden la clave API y responden también mientras carga el
modelo.

Los chats usan `GET /api/chats` (las conversaciones y su `version`), `GET /api/chats/version`,
`GET /api/chats/search?q=PALABRAS&limit=N` (el mensaje más reciente del usuario o del asistente de cada chat que
contiene todas las palabras, sin distinguir mayúsculas ni acentos) y `GET`, `POST` y `DELETE /api/chats/{id_chat}`.
Cada archivo empieza con la conversación y sigue con un mensaje por línea; los archivos guardan las fechas en el
formato nativo `2026-11-03@14-59-59.723` (UTC) y la API las da en milisegundos desde la época Unix. Un chat que se
guarda sin un número válido recibe el número libre más bajo. Piden la clave API y responden también mientras carga
el modelo.

En modo router, `GET /api/models/folder` devuelve la carpeta de modelos: `folder`, `source` (`cli` para `--models-dir`,
`settings` para la sección `modelsFolder` de `~/.mgs/config/ui-settings.json` o `none`), `exists`, `model_count`,
`settings_file` y `cache_dir`. `POST /api/models/folder` con `{"folder": "/ruta/absoluta"}` (una `~` al principio es la
carpeta personal; `""` la quita) comprueba que se puede leer la carpeta, la guarda en esa sección, vuelve a buscar los
modelos y devuelve el mismo objeto; responde 400 si la carpeta se fijó con `--models-dir` o no se puede leer.
`GET /api/models/folder/browse?path=/ruta/absoluta` alimenta el explorador de carpetas: devuelve `path`, `parent` (vacío
en `/`), `home`, `folders` (las subcarpetas ordenadas por nombre, cada una con `name`, `path` y `hidden`) y `gguf_files`
(cuántos archivos `.gguf` hay directamente en la carpeta); sin `path` lista la carpeta de modelos o, si no la hay, la
carpeta personal, y responde 400 si la ruta no es una carpeta que se pueda leer. Las tres piden la clave API. En modo
router, cada entrada de `GET /api/v1/models` lleva `path`, `file_size` (bytes, todas las partes de un modelo dividido) y
`mmproj`, y un modelo en ejecución agrega `meta` y `load`, la configuración con la que se cargó (`architecture`,
`description`, `n_layer`, `n_layer_mtp`, `n_ctx`, `n_ctx_slot`, `n_parallel`, `n_batch`, `n_ubatch`, `n_threads`,
`n_threads_batch`, `flash_attn`, `flash_attn_enabled` (si el contexto lo usa, también con `auto`), `cache_type_k`,
`cache_type_v`, `fit`, `mtp`, `speculative`, `draft_n_max`, `moe_cache_mib`, `devices` y `mmproj`). Con `--model`, la única entrada también lleva `load`.

<a id="request-queue"></a>
### Cola de pedidos y pedidos en paralelo

Por defecto, `mgs` resuelve un pedido por vez (`-np 1`) y le da el contexto completo, así que cada pedido puede
usar todos los tokens de contexto, tanto si llega desde la interfaz web como si llega directamente a la API. Los
pedidos que llegan mientras tanto esperan en una única cola y se resuelven en el orden en que llegaron: los de la
interfaz no se adelantan a los de la API, ni al revés, y un pedido que tiene que volver a esperar conserva su
lugar. La API se comporta igual sin importar cómo esté
configurada la interfaz.

Para resolver varios pedidos al mismo tiempo, arrancá `mgs` con `-np N` (`--parallel N`, `MGS_ARG_PARALLEL`):
se ejecutan hasta N pedidos en paralelo y el contexto se reparte entre ellos, así que cada uno recibe una parte
(`n_ctx_slot` en la línea `initializing` del registro); con `--kv-unified`, en cambio, comparten una sola caché KV
del contexto completo. Los pedidos que pasen de N siguen esperando en la cola.

La interfaz web agrega `"mgs_client": "webui"` a sus pedidos. Mientras uno de sus pedidos en streaming espera, el
stream empieza con eventos como

```text
data: {"mgs_queue":{"ahead":2,"running_api":1,"running_webui":0,"waiting_api":0,"waiting_webui":1}}
```

(`ahead`: pedidos que se ejecutan o esperan adelante; `running_api` y `running_webui`: cuántos de ellos se están
ejecutando ahora y llegaron directamente a la API o desde la interfaz web; `waiting_api` y `waiting_webui`: lo mismo
para los pedidos que esperan adelante en la cola) y, cuando empieza, recibe uno último con todos los contadores en 0. Los
clientes que no envían `mgs_client` nunca reciben estos eventos, así que los clientes compatibles con OpenAI
reciben el stream de siempre. Swagger describe el campo junto con los demás parámetros de completion.

<a id="security-limits"></a>
### Límites de seguridad

Estos límites protegen el servidor y la interfaz de pedidos, archivos y modelos hostiles. Son holgados para el
uso normal: lo que los supera falla con un error claro en lugar de detener el servidor.

- **Clave API.** Las rutas de salud y de listado de modelos (`/api/health` y `/api/v1/models`) sólo son públicas
  con `GET` y `HEAD`. Recargar la lista de modelos (`GET /api/v1/models?reload=1`) y agregar o borrar modelos en modo router (`POST` y
  `DELETE /api/models`) exigen la clave siempre que haya una configurada.
- **Páginas de otros sitios.** Por defecto (sin `--cors-origins`, que equivale a `localhost`) sólo las páginas de
  `localhost` y de este mismo servidor pueden llamar a la API: el servidor responde 403 a todo pedido que no sea
  `GET`, `HEAD` ni `OPTIONS` cuya cabecera `Origin` nombre otro sitio, y el navegador no deja que otros sitios lean
  sus respuestas. No afecta a la interfaz (tampoco detrás de Apache) ni a los clientes que no envían `Origin`, como
  curl o los SDK. Para que las páginas de otros sitios puedan llamar a la API, indicá sus orígenes en
  `--cors-origins` (`*` admite cualquier sitio: en ese caso, configurá una clave API). `POST /api/tools` sólo
  acepta `Content-Type: application/json` (si no, 415).
- **DNS rebinding.** Con HTTP sin TLS (`--http`) en un puerto TCP y sin clave API, el servidor responde 403
  (`Host not allowed`) a todo pedido cuya cabecera `Host` no sea una dirección IP, `localhost` (o un nombre que
  termine en `.localhost`), el nombre del equipo (también con `.local`) o el nombre indicado con `--host`, para que
  una página de otro sitio que haga apuntar su propio nombre a este equipo no pueda usar el servidor. No afecta a
  HTTPS, a los servidores con clave API, al socket Unix del despliegue ni a Vertex AI (`AIP_MODE=PREDICTION`).
  Para llegar a un servidor HTTP sin TLS a través de otro nombre, configurá una clave API.
- **Proxy CORS.** `/api/cors-proxy` rechaza los pedidos que el navegador abriría como página, marco o worker
  (403) y devuelve todas las respuestas como contenido inerte: no puede ejecutar scripts ni crear cookies en la
  dirección del servidor.
- **Listados del router.** En modo router, `status.args` y `status.preset` de `/api/v1/models` muestran
  `<redacted>` en lugar de los valores de `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`,
  `--mcp-servers-config` y `--mcp-servers-json`; el registro hace lo mismo. Los procesos de los modelos siguen
  recibiendo los valores reales.
- **Vertex AI.** Con `AIP_MODE=PREDICTION`, el `@requestFormat` de cada instancia sólo llega a las rutas de
  inferencia (completions, chat completions, responses, messages, infill, embeddings, rerank, tokenize,
  detokenize, apply-template y conteo de tokens); cualquier otra ruta, incluida la propia ruta de predicción,
  responde con un error para esa instancia.
- **Descargas de modelos.** El token de Hugging Face (`--hf-token`, `HF_TOKEN`) sólo se envía al endpoint de
  Hugging Face (`MGS_MODEL_ENDPOINT`, `https://huggingface.co/` por defecto); un modelo, un proyector o un modelo
  de borrador descargado de cualquier otra URL se pide sin él.
- **Clientes lentos.** Mientras el servidor recibe un pedido, cada lectura espera como mucho 30 s (`--timeout`
  sigue rigiendo las escrituras, así que las respuestas largas no se cortan), y como mucho 256 conexiones
  aceptadas esperan a que quede un hilo libre.
- **Cuerpos de los pedidos.** Los cuerpos JSON con más de 1024 niveles de arrays u objetos se rechazan con 400.
  Los esquemas JSON (`response_format`, `json_schema`) pueden anidarse hasta 256 niveles; se rechazan los
  `pattern` vacíos y los cuantificadores sin nada que repetir.
- **Medios remotos.** Las URL de `image_url`, `input_audio` e `input_video` sólo pueden apuntar a direcciones
  públicas de Internet. Loopback, las redes privadas (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10),
  las direcciones link-local (incluida la de metadatos de la nube, 169.254.169.254), las direcciones locales
  únicas de IPv6 y las demás direcciones no públicas se rechazan con 400, también cuando se llega a ellas por una
  redirección. Para permitirlas, por ejemplo para leer imágenes de un servidor de tu red, arrancá `mgs` con
  `--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); hacelo sólo si confiás en todos los
  clientes de la API. Cada descarga tiene 10 s para conectar, 30 s en total, hasta 5 redirecciones y 10 MiB, y el
  servidor hace como mucho 8 descargas a la vez. Todos los medios de un pedido juntos pueden sumar como mucho 256
  elementos y 512 MiB, y sus descargas remotas tienen 120 s en total; los archivos leídos con `--media-path`
  cuentan para esos mismos límites y se comprueban antes de leerlos.
- **Medios decodificados.** Imágenes de hasta 32768 píxeles por lado y 100 megapíxeles, audio de hasta una hora,
  fotogramas de video de hasta 8192 píxeles por lado y 2 GiB de fotogramas decodificados por video, y como mucho
  2 GiB de imágenes, audio y fotogramas de video decodificados por pedido (incluidos los fotogramas de todos sus
  videos).
- **Proyectores.** Un proyector multimodal cuyos metadatos no encajan con su grafo (por ejemplo, un factor de
  fusión de parches menor que 2 en un proyector que fusiona parches) falla al cargar con un error en lugar de
  detener el servidor.
- **Herramientas del servidor.** `read_file` con un rango de líneas lee el archivo por bloques y sólo conserva esas
  líneas, recorriendo como mucho 256 MiB del archivo; sin `end_line`, se rechazan los archivos de más de 16 KB.
- **Plantillas de chat del modelo.** `range()` produce como mucho un millón de elementos, las cadenas construidas
  no pasan de 256 MiB, la sangría de `indent()` y `tojson()` de 256 espacios y la anidación de 1024 sentencias;
  una plantilla que los supera falla con un error de plantilla.
- **Interfaz.** Los enlaces de los mensajes sólo conservan los esquemas http, https, mailto y tel (o direcciones
  relativas), y los adjuntos de imagen tienen que ser imágenes incrustadas. La importación de conversaciones
  acepta archivos de hasta 512 MB, archivos ZIP con hasta 10 000 entradas, 256 MB por archivo JSONL y 1 GB
  extraído en total. Un archivo con registros mal formados, o con un mensaje que pertenece a otra conversación,
  se rechaza entero; una conversación cuyos mensajes ya existen, o cuyos mensajes se enlazan entre sí en un bucle,
  se saltea y se indica en el aviso. Las imágenes SVG que escribe el modelo se quedan dentro de su propia caja y no
  pueden tapar el resto de la página. El listado de los recursos o de las plantillas de recursos de un servidor MCP
  se detiene después de 100 páginas, 10 000 elementos, 60 s o un cursor repetido. Un recurso MCP siempre se lee, se
  guarda en caché y se adjunta desde el servidor en el que se eligió, aunque otro servidor ofrezca el mismo URI.
  Una aprobación **Permitir siempre** de una herramienta MCP se aplica mientras su servidor mantenga la misma
  dirección, el mismo transporte, el mismo uso del proxy y las mismas cabeceras: si cambia cualquiera de ellos, la
  interfaz vuelve a preguntar.

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
transferencias. La RAM fijada no se puede paginar; dejá memoria suficiente para
el sistema operativo. Esta ruta es específica de POSIX; Windows mantiene el
comportamiento anterior. Solo se recomienda `--load-mode none` si el registro falla
total o parcialmente.

`mgs` tiene en cuenta el límite de memoria de su cgroup (un servicio de systemd
con `MemoryMax`, un contenedor) además de la RAM libre. El ajuste automático
solo deja los expertos MoE en la RAM cuando entran en ese límite y, si no, pone
capas de expertos enteras en la GPU. Cuando los expertos quedan en la RAM, la
caché de prompts (`--cache-ram`, 8 GiB por defecto) ocupa como máximo la memoria
que queda después de ellos y del margen, y el registro lo indica con `prompt cache limited to N MiB`;
`--cache-ram` la fija a mano. Solo se fijan las páginas del modelo que quedan en la RAM, y solo tantas como dejen un margen (4 GiB o el 10 % de la RAM) de la memoria que sigue disponible; a partir de ahí el registro muestra `pinning X of Y GiB of mapped weights` y el resto usa memoria paginable. En un cgroup de 30 GB con el Qwen3.6-35B-A3B Q8_0 de
37,6 GB, la versión 0.1.17 se quedaba colgada durante la carga hasta que el
sistema la mataba; ahora arranca en unos minutos y genera. Hasta la versión 0.1.22, con capas de expertos enteras en la GPU, `mgs` también fijaba los expertos ya copiados a la GPU, así que en un escritorio con poca RAM libre y sin swap todo el sistema podía dejar de responder mientras se cargaba el modelo.

Cuando los archivos del modelo son mayores que la memoria que sigue disponible (o que el límite del cgroup en el que se ejecuta), `mgs` no los lee enteros al cargar, lo que solo desalojaría sus propias primeras páginas y la caché de páginas de los demás programas: lee lo que va a la GPU y lo que fija, y el resto la primera vez que se usa. Con Qwen3.8-Flash-Next (113 GB) en 40 GiB de memoria y un disco SATA, la carga pasó de más de 5 minutos a un minuto y medio, aproximadamente.

La precarga usa tres slots de VRAM por defecto. Cada uno debe alojar el mayor
tensor de expertos transferido. Los valores 2–8 de `GGML_SCHED_PREFETCH_EXPERTS`
eligen el número de slots. Ante un fallo de reserva se conservan los que caben
si son al menos dos; de lo contrario se usan transferencias normales. Se activa
cuando el lote contiene al menos el doble de selecciones que expertos. El grupo
pertenece a la primera GPU que lo activa; las demás mantienen la copia selectiva.
Un lote mayor puede mejorar el prefill, pero también consume más memoria. En una [GPU integrada](#integrated-gpu) no hay precarga: llenar una ranura sería una copia de RAM a RAM hecha por la CPU, y las ranuras quitarían RAM a la caché de páginas de los expertos, así que los expertos que se usan se copian cuando hacen falta (con Qwen3.8-Flash-Next en una Radeon 860M el prompt pasó de 15 a 33 tokens/s y la generación, de 4,3 a 5,0 tokens/s).

Cada slot ocupa lo que la GPU reserva para el mayor tensor de expertos, incluido el relleno que CUDA añade a las filas cuya longitud no es múltiplo de 512 elementos, y deja libre `--moe-cache-reserve` (1024 MiB por defecto) para la memoria que el cálculo ocupa mientras se ejecuta; si un slot no cabe, la precarga sigue con los que caben si son al menos dos, o copia en su lugar los expertos que se usan. Hasta la versión 0.1.31 los slots no incluían ese relleno, así que Qwen3.8-Flash-Next, cuyos expertos tienen filas de 640 elementos, podía pararse en el primer prompt largo con `CUDA error: an illegal memory access was encountered` (o `an internal operation failed`) siempre que la GPU tenía sitio para dos slots, por ejemplo con MTP en una RTX 4060 Ti. Con Qwen3.8-Flash-Next en la RTX 4060 Ti, MTP de 2 tokens procesa ahora el prompt a 12 tokens/s y genera a 1,31 tokens/s.

Para diagnóstico, `GGML_CUDA_REGISTER_HOST=0` desactiva el registro,
`GGML_SCHED_PREFETCH_EXPERTS=0` desactiva la precarga y `GGML_VK_STAGING_RING=0`
desactiva el anillo de staging de las subidas con Vulkan. No se necesitan esas
variables durante el funcionamiento normal.

MTP automático inspecciona metadatos y nombres de tensores antes de reservar
pesos, incluidas todas las partes del GGUF. Para el MTP integrado en `--model`, requiere
arquitectura compatible, tronco y todos los cabezales integrados. Un GGUF de MTP separado pasado
con `--spec-draft-model` también lo activa cuando declara cabezales MTP completos, sin tronco; un
modelo de borrador sin cabezales MTP mantiene el tipo especulativo que elijas. Un MTP que solo
está en otro GGUF nunca se busca, y una arquitectura no compatible no lo activa.
El máximo predeterminado es de dos tokens de borrador: con MTP, dos generaron tan rápido como tres en una
RTX 4060 Ti y en una Radeon 860M integrada, y un borrador rechazado desperdicia menos trabajo de verificación. Cualquier `--spec-type` explícito
prevalece y desactiva la selección automática; `--spec-type none` desactiva MTP.
`--spec-draft-n-max` ajusta el máximo. MTP afecta a la generación, no
directamente al prefill ni al tiempo hasta el primer token. La mejora depende
de la aceptación y del hardware; aumentar el máximo no garantiza más velocidad.

Que MTP acelere la generación depende del equipo: cada paso comprueba los tokens del borrador con el modelo entero, lo que cuesta poco cuando los expertos están en la GPU y mucho cuando la mayoría vienen de la RAM o del disco. Por eso, cuando MTP se activa automáticamente, `mgs` lo mide durante las primeras respuestas: tras un breve calentamiento, se turna paso a paso con 0 (sin borrador), 1, 2 y 3 tokens de borrador hasta que cada uno ha generado 64 tokens, unos 500 tokens en total, se queda con el más rápido si el borrador es más de un 5 % más rápido que sin él y lo indica en el registro con una línea como `MTP drafts 2 tokens from now on, the fastest on this machine`. El resultado se guarda en `~/.mgs/tuning/mtp.json` por modelo, equipo y versión de `mgs`, y los arranques siguientes lo usan directamente; cuando el borrador no compensó, el registro dice `MTP drafting is not worth it on this machine` y desde el siguiente arranque MTP no se carga, lo que además deja su VRAM a los expertos, al contexto y al micro-lote. Borrar el archivo (o su entrada) hace que se vuelva a medir. `--spec-draft-n-max N` fija el número de tokens de borrador y desactiva la medición, `--spec-type draft-mtp` mantiene MTP activado y un GGUF de MTP aparte pasado con `--spec-draft-model` se usa siempre. En la Radeon 860M, Qwen3.8-Flash-Next generó 5,12 / 7,09 / 7,34 / 6,97 tokens/s con 0 / 1 / 2 / 3 tokens de borrador y se queda con 2; en la RTX 4060 Ti generó 1,81 / 1,79 / 1,30 / 1,59, así que desde el siguiente arranque funciona sin MTP, con 23-24 tokens/s de prompt y 1,69-1,70 de generación en lugar de 12 y 1,4, mientras que Qwen3.6-35B-A3B generó allí 26,38 / 33,92 / 36,21 / 35,41 y se queda con 2.

<a id="mtp-vram"></a>
Los pesos MTP se quedan siempre en la memoria de la GPU cuando caben en ella: las capas MTP injertadas en el
modelo, un GGUF de MTP separado pasado con `--spec-draft-model` y la cabeza de salida que usan. Las opciones que
llevan pesos a la memoria del sistema (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, sus equivalentes
`--spec-draft-*` y el ajuste automático) no se les aplican, y siguen en la GPU aunque `--gpu-layers` sea bajo. El
ajuste automático coloca el resto del modelo a su alrededor y la [caché de expertos MoE](#expert-cache) usa la
memoria que queda después. Si una capa MTP no entra en la memoria libre de ninguna GPU seleccionada, el registro
muestra un aviso (`no selected GPU has enough free memory for MTP layer 0; using system memory`) y solo esa capa queda
en la memoria del sistema; un tensor de un tipo que la GPU no admite también queda ahí. En una
[GPU integrada](#integrated-gpu), la memoria libre que se cuenta para MTP es como máximo lo que la RAM del sistema
todavía puede dar. Con un GGUF de MTP separado no se cargan las capas MTP injertadas en el modelo principal, para
que no ocupen memoria dos veces; con `--spec-type none` no se carga ningún peso MTP. El registro de carga confirma
la ubicación con líneas como `MTP layer 0 assigned to CUDA0 (856.36 MiB)`, visibles con `-lv 4`.

<a id="expert-cache"></a>
### Caché de expertos MoE

Cuando los expertos MoE están en la memoria del sistema y el resto de la capa corre en una GPU, el servidor
mantiene los expertos usados más recientemente de todas las capas en una caché que ocupa la VRAM libre; con
varias GPU, cada una guarda en su VRAM las capas que tiene asignadas. En
cada token generado (y en cualquier lote de menos de 32 tokens), los expertos que están en la caché se calculan
en la GPU, una parte de los que faltan se sube a la caché y el resto se calcula en la CPU al mismo tiempo; esa
parte se ajusta a los anchos de banda medidos de PCIe y de la CPU. Los prompts largos conservan la precarga de
expertos, que ahora copia dentro de la GPU los expertos cacheados y solo sube los demás. El resultado coincide con el de la ejecución sin caché salvo redondeo. La GPU y la CPU redondean de
forma distinta y los expertos que calcula cada una dependen del contenido de la caché, así que una generación
voraz larga puede derivar, de una ejecución a otra, hacia un texto distinto pero igual de válido tras algunos
tokens; la caché de llama.cpp oficial se comporta igual.

La caché está activa por defecto y se crea en la primera solicitud, cuando ya están cargados el modelo, el
contexto MTP y el proyector multimodal. Sin `--gpu-layers`, `--n-cpu-moe` ni `--override-tensor`, el ajuste
automático mantiene todos los expertos en la memoria del sistema si entran con margen (4 GiB o el 10 % de la
RAM) y deja la VRAM libre a la caché (con varias GPU, reparte las capas entre ellas según su memoria libre
para que todas tengan caché); si no, ubica capas de expertos completas en la GPU como antes y la caché
usa lo que queda. Las opciones explícitas de ubicación se respetan: `--cpu-moe` deja a la caché la mayor parte
de la VRAM y `--n-cpu-moe N` mantiene algunas capas fijas en la GPU. Los expertos que quedan en la RAM se fijan en memoria mientras la memoria que sigue disponible conserve ese margen; el resto sigue siendo paginable, así que el sistema sigue respondiendo aunque falte RAM.

Cada capa conserva su propia parte de la caché, igual para todas, así que una racha de fallos en algunas
capas nunca vacía la caché de las demás, y dentro de cada parte los expertos que se vuelven a usar quedan
protegidos frente a los que se usan una sola vez: un experto que se sube a la caché entra en una zona de
prueba y pasa a una zona protegida (como máximo la mitad de la parte) cuando se vuelve a usar, y los
expertos nuevos reemplazan primero a los de la zona de prueba. Cuando solo se puede subir una parte de los
expertos que faltan, van primero los usados más reciente y frecuentemente. Con Qwen3.6-35B-A3B Q8_0 en una
RTX 4060 Ti, la generación pasó de 37,4 a 38,1 tokens/s (de 45,2 a 47,1 con MTP) y los aciertos de la caché
del 72,3 al 74,1 %. El reparto también mantiene útil la caché cuando un paso trae más expertos de los que
entran en toda ella, como con MTP en GPU con menos VRAM, donde una sola lista compartida por todas las
capas no conservaba casi nada.

<a id="integrated-gpu"></a>
En una GPU integrada (los gráficos de un procesador AMD o Intel mediante Vulkan, o NVIDIA Jetson y DGX Spark
mediante CUDA), la memoria que informa la GPU es parte de la RAM del sistema. El ajuste automático cuenta las dos
como un único presupuesto: la GPU puede usar como máximo lo que la RAM todavía puede dar después de los búferes que
quedan en la CPU. Si el modelo entero entra ahí (reduciendo el contexto si hace falta, como en una GPU dedicada),
todo corre en la GPU; si no, capas de expertos completas van a la GPU hasta ese límite y la CPU calcula el resto.
No hay caché de expertos en una GPU integrada, ni siquiera con `--cpu-moe` o `--n-cpu-moe`, porque copiaría
expertos de la RAM a la misma RAM y los guardaría dos veces. El registro muestra el presupuesto en una línea como
`Vulkan0 is an integrated GPU that shares the system memory: N MiB usable` y la decisión en
`no MoE expert cache on an integrated GPU`. Con el modelo mapeado (`--load-mode mmap`, el valor por defecto), los pesos que se quedan en la CPU cuando todas las capas están en la GPU (las tablas de embeddings de tokens y por capa, que se leen fila a fila) no cuentan en ese presupuesto, porque sólo las filas leídas quedan en memoria: Qwen3.8-Flash-Next, cuyos embeddings por capa ocupan 28 GB, recibe toda la memoria que declara la GPU en lugar de lo que queda después de ellos. En una GPU así, `mgs` además se vuelve a ejecutar una vez al arrancar con `OMP_WAIT_POLICY=passive`, salvo que la variable ya esté definida, para que los hilos de la CPU duerman en lugar de girar mientras trabaja la GPU: comparten con ella la potencia del procesador, y con los hilos girando la GPU de una Radeon 860M iba a la mitad de su frecuencia. El registro lo indica en la línea `the OpenMP threads of the CPU wait passively`; `OMP_WAIT_POLICY=active` mantiene los hilos girando.

<a id="context-size"></a>
Cuando el modelo no entra en la VRAM y no se indica `-c`, el ajuste automático elige también el tamaño del
contexto: la caché KV ocupa como máximo una cuarta parte de la VRAM que queda después de los pesos densos, hasta
el contexto de entrenamiento del modelo, y los expertos (la caché o las capas de expertos completas) se quedan con
el resto. Antes, esos modelos recibían siempre el mínimo de 4096 tokens compartidos por todos los slots, lo que
cortaba las respuestas largas (`truncated = 1` en el registro). El registro muestra el resultado en una línea como
`context size raised from 4096 to N tokens` y en `n_ctx_slot` de la línea `initializing`. `-c N` fija el
contexto de forma explícita (más contexto le deja menos VRAM a la caché y baja la velocidad de generación: 35,8
tokens/s con 4096 tokens y 33,9 con 122624 en la máquina de referencia) y `--fit-ctx N` sube el mínimo. El
contexto MTP toma el mismo tamaño que el contexto del modelo en lugar de reservar VRAM para todo el contexto de
entrenamiento. Cuando las capas de expertos completas llenan la GPU porque los expertos no se quedan en la RAM para la caché (siempre en una [GPU integrada](#integrated-gpu)), la caché KV ocupa como mucho un dieciseisavo, porque cada capa de expertos que queda fuera de la GPU se lee de la RAM, o del disco cuando el modelo es mayor que la RAM, en cada token: Qwen3.8-Flash-Next en una Radeon 860M con 46 GB de RAM recibe 37376 tokens y genera 5,3 tokens/s, donde con una cuarta parte recibía 137216 tokens y generaba 4,9 tokens/s.

El ajuste automático cuenta la memoria del contexto MTP para el tamaño de contexto que elige, no para el contexto de entrenamiento del modelo: hasta la versión 0.1.32 le contaba 1300 MiB de VRAM con Qwen3.8-Flash-Next en una RTX 4060 Ti, donde ocupa unos 160 MiB, y elegía un contexto más corto para el modelo. Con Qwen3.8-Flash-Next y MTP de 2 tokens en la RTX 4060 Ti, el ajuste cuenta ahora 130 MiB más 4,57 KiB por token de contexto para el contexto MTP y da al modelo un contexto de 7936 tokens en lugar de 6912.

Antes que el contexto, el ajuste automático elige el micro-lote (`-ub`), la cantidad de tokens del prompt que se
procesan en cada pasada. Con los expertos en la RAM, cada pasada sube a la GPU los expertos que no están en la
caché, así que menos pasadas y más grandes procesan los prompts largos mucho más rápido, sin cambiar la
generación. Si no se indica `-b` ni `-ub`, toma el mayor de 4096, 2048 y 1024 tokens cuyos búferes extra entren en
una octava parte de la VRAM que queda después de los pesos densos, y sube el lote (`-b`) con él. En la máquina de
referencia, un prompt de 2514 tokens pasó de 214 a 755 tokens/s con 4096, con la misma velocidad de generación; los
búferes más grandes acortan el contexto predeterminado (unos 90000 tokens sin MTP y 61000 con MTP en 16 GB). El
registro muestra la elección en una línea como `micro-batch size raised from 512 to 4096 tokens (batch 4096)`.

Cuando las capas de expertos enteras llenan la GPU, dejan por debajo del margen menos memoria de la que necesita el siguiente tensor de expertos, y nada más la usa; el ajuste automático amplía entonces el micro-lote con esa memoria, así que ningún experto sale de la GPU. El registro lo muestra en una línea como `micro-batch size raised from 2048 to 4096 tokens (batch 4096) with the device memory that the whole expert layers leave unused`. Que alcance depende de la VRAM libre: con Qwen3.8-Flash-Next y MTP en una RTX 4060 Ti no alcanzaba con el margen por defecto y sí con `--fit-target 800`. En esa ejecución el prompt se procesó a 21-22 tokens/s en lugar de 12, con la generación a 1,41-1,46 tokens/s (1,39 antes).

Con una ubicación explícita (`--gpu-layers`, `--tensor-split`, `--override-tensor`, `--cpu-moe` o `--n-cpu-moe`),
el ajuste automático la respeta tal como se indicó y solo ajusta a su alrededor el contexto y el micro-lote que no
hayas fijado, en lugar de detenerse; el registro indica `the device placement was set explicitly`.

| Opción | Variable | Efecto |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_ARG_MOE_CACHE` (`0` o `false` la desactiva) | Activa o desactiva la caché (activa por defecto). |
| `--moe-cache-size MiB` | `MGS_ARG_MOE_CACHE_SIZE` | Tamaño total fijo de la caché, repartido entre las GPU según los expertos que guarda cada una; `-1` (por defecto) usa la VRAM libre de cada GPU. |
| `--moe-cache-reserve MiB` | `MGS_ARG_MOE_CACHE_RESERVE` | VRAM que se deja libre en cada GPU además de la caché y de los huecos de precarga (1024 por defecto). Aumentala si otros programas usan la GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnóstico: proporción fija (0–1) de expertos que faltan y se suben, en lugar del equilibrio medido. |
| — | `MGS_MOE_CACHE_STATS` | Diagnóstico: registra los contadores de la caché cada N pasos de generación (visibles con `-lv 4`). |
| — | `MGS_MOE_CACHE_PROTECTED` | Diagnóstico: proporción (0–1) de la parte de cada capa reservada a los expertos que se vuelven a usar (0,5 por defecto); `0` deja un orden LRU simple por capa. |

`GET /api/expert-cache` devuelve si la caché existe (`active`) o si la próxima solicitud todavía tiene que crearla o
aplicar un tamaño nuevo (`pending`), el tamaño, las capas y huecos cacheados, los aciertos y fallos, las subidas,
los expertos calculados en la CPU, la proporción de bytes de precarga servidos desde la caché y los anchos de banda medidos (con varias GPU, los
tamaños y contadores se suman y el ancho de banda de subida es su media). `POST /api/expert-cache` con `{"size_mib": N}` cambia el tamaño sin reiniciar (`-1` automático,
`0` libera la caché); la siguiente solicitud lo aplica y arranca con la caché vacía:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Con `--metrics`, `/api/metrics` agrega contadores e indicadores `mgs:moe_cache_*` (pasos, aciertos, fallos,
subidas, expertos en CPU, bytes de precarga, tamaño, huecos y anchos de banda).

Medición de referencia en una RTX 4060 Ti de 16 GB con Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 hilos de CPU), frente a
`--n-cpu-moe 28` sin caché: generación de 26,9 a 35,7 tokens/s (+33 %), procesamiento del prompt de 767–778 a
718–730 tokens/s (−6 %) y, con MTP activo, de 33,6 a 47,0 tokens/s (+40 %). En la misma máquina, llama.cpp oficial
con su propia caché (`-cmoe --moe-cache-mib 10240`) llegó a 26,7 tokens/s, y a 27,8 con MTP. La ganancia depende de la parte de
los expertos que entra en la VRAM, de la localidad del enrutado del modelo y de los anchos de banda de PCIe y de
la RAM.

<a id="threads"></a>
### Hilos de CPU

La CPU calcula los expertos que no están en la caché. Ese trabajo está limitado por el ancho de banda de la
memoria y se satura con pocos hilos, mientras que usar todos los CPU lógicos hace que los hilos compitan con el
que maneja la GPU. Sin `-t`, `mgs` usa los núcleos útiles para cálculo, pero nunca más de la mitad de los CPU
lógicos en los que puede ejecutarse, salvo en una CPU con hasta 8 CPU lógicos sin SMT, donde usa todos los núcleos:
con tan pocos núcleos el límite es la CPU (en un i7-7700K de 4 núcleos con una RX 580, donde la CPU calcula 40 de las
41 capas de expertos, 4 hilos generaron un 17 % más rápido que 2). En una máquina con SMT (dos hilos por núcleo) el
tope es la cantidad de núcleos físicos, como antes; en una máquina virtual más grande que no muestra SMT es la mitad
de los CPU virtuales. En la máquina de
referencia (16 núcleos virtuales con una sesión de escritorio, Qwen3.6-35B-A3B Q8_0, sin MTP), de 4 a 12 hilos
generaron a la misma velocidad (32,9–33,9 tokens/s) y 16 hilos a 28,7 tokens/s. El registro de arranque muestra la
elección en una línea como `using 8 CPU threads for generation and 8 for batches`; `-t` y `-tb` la fijan de forma
explícita.

<a id="kv-cache-type"></a>
### Tipo de la caché KV

Si no se pasa `-ctk` (`--cache-type-k`) ni `-ctv` (`--cache-type-v`), `mgs` da a la caché KV la precisión de los pesos del modelo, según el tipo que contiene la mayor parte de los pesos de sus expertos: los modelos sin cuantizar (F16, BF16, F32) mantienen una caché `f16`, los de más de 5 bits por peso (Q5_K_M, Q6_K, Q8_0) usan `q8_0` y el resto (Q4_K_M, IQ4_XS, MXFP4, Q3_K, Q2_K…) usan `q4_0`, los dos tipos cuantizados que flash attention lee en todos los backends. La regla es la misma en todos los backends (CPU, CUDA, ROCm y Vulkan). Una caché cuantizada necesita flash attention, así que la caché se queda en `f16` cuando flash attention está desactivado (`-fa off`) o no puede funcionar con ese tipo en la GPU. El registro de arranque muestra la elección en una línea como `KV cache type chosen from the weights of the model: q8_0`, y la página Modelo muestra los tipos en uso.

Una caché `q8_0` ocupa aproximadamente la mitad de memoria que una `f16`, y una `q4_0`, alrededor del 28 %, así que el ajuste automático da mucho más contexto en la misma memoria, o deja esa memoria a la caché de expertos, a cambio de una atención algo menos precisa y de un pequeño porcentaje de velocidad. Leer una caché cuantizada requiere algo de trabajo adicional en todos los backends, y CUDA (GPU de NVIDIA) y ROCm (GPU de AMD con una compilación `--rocm`), que comparten los mismos kernels de flash attention, solo la leen directamente cuando se procesan muy pocos tokens a la vez (la generación normal, no el prompt ni la verificación de varios tokens de MTP) con cabezas de atención de 64, 128 o 256 valores, y en los demás casos la convierten a `f16` en cada paso. En una Radeon 860M (Vulkan) con Qwen3.6-35B-A3B Q6_K y sin MTP, `q8_0` generó un 3,8 % menos de tokens por segundo que `f16` después de un prompt corto y un 2,7 % menos después de un prompt de 8060 tokens, y procesó ese prompt un 1,3 % más despacio. Si se pasa solo una de las dos opciones, la otra se queda en `f16`, y `-ctk f16 -ctv f16` recupera la caché `f16` cuando la memoria alcanza y ese pequeño porcentaje importa más que el contexto. El contexto de MTP y el modelo borrador de `--spec-draft-model` siguen la misma regla; `--spec-draft-type-k` y `--spec-draft-type-v` fijan sus tipos.

<a id="checkpoints"></a>
### Checkpoints de contexto

Los modelos con capas recurrentes o de ventana deslizante (por ejemplo los híbridos Qwen3.5/3.6/3.8 o
Nemotron-H) no pueden deshacer su estado, así que cuando una solicitud cambia una
parte anterior de la conversación, el servidor continúa desde el checkpoint más reciente anterior al cambio.
Los checkpoints se ubican al comienzo de los mensajes de usuario, asistente y herramienta (donde los agentes de
programación eliminan razonamientos viejos o recortan resultados de herramientas) y al menos cada
`--checkpoint-min-step` tokens (1024 por defecto, antes 8192), sin pasadas adicionales por el prompt. Al llegar
a `--ctx-checkpoints` (32), se elimina el checkpoint más próximo a sus vecinos, así los demás siguen repartidos
por la conversación. En el escenario de agente de referencia (14 000 tokens, ocho resultados de herramientas),
recortar un resultado tardío pasó de 19,3 s a 5,5 s y uno temprano de 18,0 s a 13,6 s. Cada checkpoint guarda en
la memoria del host el estado recurrente o de ventana deslizante (63 MiB en Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Opciones actualizadas y diagnóstico

Conviene mantener `--load-mode mmap`, predeterminado, cuando funciona el registro de memoria. `--load-mode none` es una alternativa si ese registro falla. `--load-mode dio` utiliza ahora un buffer auxiliar de hasta 64 MiB; no es un límite de RAM total del modelo.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` se retiró porque no tenía efecto durante la ejecución. Las antiguas opciones de carga (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) y el flag de compilación `GGML_CUDA_FA_ALL_QUANTS` ya no se aceptan; sus sustitutos están en [compatibilidad eliminada](#removed-compatibility). En una compilación CMake manual, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` añade esa combinación K/V de kernels vectoriales; se admiten listas con comas o con punto y coma entre comillas. La lista predeterminada es `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 se incluye siempre. `all` aumenta el trabajo de compilación. Si falta una combinación vectorial, se convierte a F16, lo que puede ser más lento.

MTP separa grafos con y sin salidas, conserva el orden original de tokens de secuencias concurrentes y usa las posiciones reales después de imágenes.

El ajuste automático cuenta las capas nextn (MTP), así que la primera capa ya no queda en la CPU cuando MTP está desactivado. Los pesos MTP ahora quedan siempre en la memoria de la GPU; consultá [ubicación de MTP](#mtp-vram). El valor por defecto de `--checkpoint-min-step` ahora es 1024 en lugar de 8192; consultá [checkpoints de contexto](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` y `--moe-cache-reserve` controlan la [caché de expertos MoE](#expert-cache). El ajuste automático ya no deja con un contexto de 4096 tokens a los modelos más grandes que la VRAM y ahora también elige el micro-lote; consultá [tamaño del contexto](#context-size). La cantidad de hilos de CPU por defecto ahora es, como máximo, la mitad de los CPU lógicos; consultá [Hilos de CPU](#threads). `-np` ahora usa por defecto un solo slot con el contexto completo, en lugar del modo automático con cuatro slots que compartían una caché KV unificada, y `--kv-unified` está desactivado por defecto; consultá [Cola de pedidos](#request-queue). En una GPU integrada, el ajuste automático ahora cuenta la memoria de la GPU y la RAM como un único presupuesto y ya no crea la caché de expertos MoE; consultá [GPU integradas](#integrated-gpu). Desde la 0.1.19, el ajuste automático ya no se detiene cuando la ubicación se indica de forma explícita: la respeta y ajusta a su alrededor el contexto y el micro-lote; consultá [tamaño del contexto](#context-size).

Con `--metrics`, `GET /api/metrics` autenticado incluye `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` y `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Las posiciones empiezan en cero; esa serie aparece al completar una solicitud especulativa. La aceptación se puede evaluar con las diferencias de contadores aceptados/propuestos; la latencia debe medirse por separado, porque una aceptación alta no demuestra por sí sola más velocidad. En modo router se elige el modelo con `?model=ID_MODELO`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, desactivada por defecto) permite que las URL de medios de los pedidos lleguen a direcciones de loopback, privadas y otras no públicas; consultá [límites de seguridad](#security-limits).

Las exportaciones leen el árbol completo de mensajes y los metadatos persistidos. Si las herramientas del servidor responden con 403, los mensajes nuevos dejan de repetir la consulta fallida; abrir de nuevo el panel de herramientas permite reintentar. Los valores del servidor en la primera visita conservan los ajustes ya modificados por el usuario, incluida la clave de API.

[Implementación y fuentes de upstream](CODE.es-AR.md#upstream-review).

<a id="removed-compatibility"></a>
## Compatibilidad eliminada

Se eliminó el código de retrocompatibilidad. Los nombres antiguos ya no se traducen ni
generan avisos: el servidor rechaza como argumentos no válidos las opciones de
línea de comandos eliminadas, ignora las variables de entorno eliminadas y
responde 404 en las rutas eliminadas. Los scripts, las personalizaciones de los
servicios, los presets y los clientes tienen que usar los sustitutos de esta tabla.

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
| Variables de entorno, opciones de CMake y símbolos de la API C con el antiguo prefijo heredado del proyecto original, incluidos los alias que versiones anteriores ya habían retirado | El nombre actual con el prefijo `MGS_` (`mgs_` en la API C), por ejemplo `MGS_ARG_PORT`, `MGS_ARG_<NOMBRE>=false` en lugar de una variante `NO_`, `MGS_ARG_LOAD_MODE` en lugar de las variables de mapeo en memoria, bloqueo y Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` y `MGS_ARG_SPEC_DRAFT_N_MIN` para los límites del borrador, o `MGS_BUILD_UI`; para las opciones de los backends, la opción `GGML_*` equivalente, por ejemplo `GGML_CUDA` o `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `HF_HOME` y `MGS_CACHE` | Nada; todas las descargas, también las de Hugging Face, van a `~/.cache/mgs/` (bajo `XDG_CACHE_HOME` cuando está definida). |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` o `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valor `deepseek-legacy` de `reasoning_format` (campo de petición y `--reasoning-format`) | `deepseek` o `auto`; el razonamiento se devuelve en `reasoning_content` |
| Campo de petición `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (array de cadenas) |
| `reasoning_in_content` en `generation_settings` | Nada; el campo se eliminó. |
| Descarga de una interfaz precompilada del proyecto original desde un bucket de Hugging Face (las opciones de CMake `USE_PREBUILT_UI` y `UI_HF_BUCKET` y la variable de entorno `HF_UI_VERSION`) | Nada; la interfaz siempre se compila desde `frontend/` (`MGS_BUILD_UI=ON`) o se omite. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Número de compilación y commit: las variables de CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (también en `mgs-config.cmake`) y la cadena `b0-unknown` de `build_info`, `system_fingerprint` y el `User-Agent` | `MGS_VERSION`, definida en `backend/cmake/build-info.cmake`; esos campos llevan ahora la versión, como `0.1.19` |
| Modelos Llama de Meta (las arquitecturas GGUF `llama`, `llama4` y `llama-embed` y el proyector de visión de Llama 4) y las plantillas de chat integradas `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` y `llama4` | Nada; usá uno de los modelos MoE compatibles. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Opción `--no-sudo` de los instaladores | Nada; los instaladores siempre se ejecutan como root y sin sudo. |
| Funciones y campos que las API C tenían marcados como obsoletos: en `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, los alias `mgs_token_*` (por ejemplo `mgs_token_bos`, `mgs_token_get_text` o `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, el campo `defrag_thold` de `mgs_context_params` y `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; en `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` y `ggml_upscale_ext`; en `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` y el campo `image_marker` | La función actual: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` en lugar de `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` y `media_marker`. Las ejecuciones de calentamiento las hace quien llama, y `defrag_thold` no tenía ningún efecto. |
| Gemma 4: las arquitecturas `gemma4` y `gemma4-assistant`, los proyectores `gemma4v`, `gemma4uv`, `gemma4a` y `gemma4ua`, el formato de chat `peg-gemma4`, el tokenizador y el pretokenizador `gemma4`, y los presets `--vision-gemma-4b-default` y `--vision-gemma-12b-default` | Nada; usá una de las [familias admitidas](#supported-models). |
| Modelos DeepSeek anteriores a la V4.1: las arquitecturas `deepseek`, `deepseek2` (también los archivos de Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 y Youtu convertidos con ella), `deepseek2-ocr` (con los proyectores `deepseekocr` y `deepseekocr2`), `deepseek32`, `mistral4` y `deepseek4`, los pretokenizadores `deepseek-llm` y `deepseek-coder`, la base de borrador DSpark de DeepSeek-V4 y la detección de la plantilla de chat de DeepSeek-V3.2 (DSML con `function_calls`) | DeepSeek-V4.1-Flash (`deepseek41`). |
| Las demás arquitecturas MoE heredadas de llama.cpp: `grok`, `nomic-bert-moe`, `qwen2moe`, `qwen3moe`, `qwen3next`, `qwen3vlmoe`, `phimoe`, `minicpm`, `jamba`, `cohere2moe`, `dbrx`, `olmoe`, `arctic`, `glm4moe`, `exaone-moe`, `granitemoe`, `granitehybrid`, `bailingmoe`, `bailingmoe2`, `dots1`, `afmoe`, `laguna`, `ernie4_5-moe`, `hunyuan-moe`, `hy_v3`, `gpt-oss`, `lfm2moe`, `smallthinker`, `llada-moe`, `grovemoe`, `minimax-m2`, `rnd1`, `mistral3`, `step35`, `kimi-linear` y `mellum`, y los formatos de chat de Ministral 3, gpt-oss, Functionary v3.2, Kimi-K2, Cohere2-MoE, LFM2, GigaChat3 y MiniCPM5 | Nada; usá una de las [familias admitidas](#supported-models). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) y las plantillas integradas del motor sin Jinja (`--chat-template` con un nombre como `deepseek3`, `chatglm4` o `mistral-v7-tekken`) | Nada; las plantillas de chat pasan siempre por Jinja. `--chat-template chatml` sigue eligiendo ChatML; para otra plantilla, pasá su código Jinja con `--chat-template` o `--chat-template-file`. Los pedidos con `tools` ya no necesitan ninguna opción. |
| `-hfr` | `-hf` |
| Segundos nombres largos de una opción: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe` (las claves de los presets del router siguen los mismos nombres) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, con el nombre de su opción |
| Campo de pedido `thinking_budget_tokens` | `reasoning_budget_tokens` (los clientes de Anthropic siguen enviando `thinking.budget_tokens`) |
| `content` en `POST /api/embeddings` y `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, la estructura `mgs_chat_message` y `MGS_FILE_MAGIC_GGLA` en `mgs.h`; `mtmd_image_tokens_get_id` en `mtmd.h` | Nada; el formato del chat lo hacen las plantillas Jinja de `common/chat.h`, y los demás no tenían uso. |
| `MGS_API_KEY`, `MGS_ARG_BATCH`, `MGS_ARG_UBATCH`, `MGS_ARG_N_PREDICT`, `MGS_ARG_N_PARALLEL`, `MGS_ARG_N_GPU_LAYERS`, `MGS_ARG_EMBEDDINGS`, `MGS_ARG_RERANKING`, `MGS_ARG_LOG_VERBOSITY`, `MGS_ARG_ROPE_SCALING_TYPE`, `MGS_ARG_SPEC_DRAFT_HF_REPO`, `MGS_ARG_STATIC_PATH`, `MGS_ARG_ENDPOINT_METRICS`, `MGS_ARG_ENDPOINT_PROPS`, `MGS_ARG_ENDPOINT_SLOTS` | `MGS_ARG_API_KEY`, `MGS_ARG_BATCH_SIZE`, `MGS_ARG_UBATCH_SIZE`, `MGS_ARG_PREDICT`, `MGS_ARG_PARALLEL`, `MGS_ARG_GPU_LAYERS`, `MGS_ARG_EMBEDDING`, `MGS_ARG_RERANK`, `MGS_ARG_VERBOSITY`, `MGS_ARG_ROPE_SCALING`, `MGS_ARG_SPEC_DRAFT_HF`, `MGS_ARG_PATH`, `MGS_ARG_METRICS`, `MGS_ARG_PROPS`, `MGS_ARG_SLOTS`: cada variable es `MGS_ARG_` seguido del nombre largo de su opción en mayúsculas, con `_` en lugar de `-` |
| `MGS_MOE_CACHE`, `MGS_MOE_CACHE_MIB`, `MGS_MOE_CACHE_RESERVE_MIB` | `--moe-cache`/`--no-moe-cache`, `--moe-cache-size` y `--moe-cache-reserve`, o `MGS_ARG_MOE_CACHE`, `MGS_ARG_MOE_CACHE_SIZE` y `MGS_ARG_MOE_CACHE_RESERVE` |
| `MTMD_BACKEND_DEVICE`, `MTMD_DEBUG_EMBEDDINGS` y la opción de CMake `MTMD_VIDEO` | `MGS_MTMD_BACKEND_DEVICE`, `MGS_MTMD_DEBUG_EMBEDDINGS` y `MGS_MTMD_VIDEO` |
| `--spec-mtp-auto`, `--no-spec-mtp-auto` (`MGS_ARG_SPEC_MTP_AUTO`) | Nada: MTP se activa por sí solo; `--spec-type none` lo desactiva y cualquier otro `--spec-type` lo sustituye. |
| `--spec-default` | `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, los valores que fijaba |
| Los presets de modelo `--tts-oute-default`, `--embd-gemma-default`, `--fim-qwen-1.5b-default`, `--fim-qwen-3b-default`, `--fim-qwen-7b-default`, `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec`, `--fim-qwen-30b-default`, `--gpt-oss-20b-default` y `--gpt-oss-120b-default` | Nada; pasá el modelo con `--model` o `-hf`. |
| Las opciones de las herramientas de línea de comandos que `mgs` no incluye, y sus variables, como `-p`/`--prompt`, `-f`/`--file`, `-sys`/`--system-prompt`, `-i`/`--interactive`, `-cnv`/`--conversation`, `--hellaswag`, `--winogrande`, `--kl-divergence`, `--save-all-logits`, `--ppl-stride`, `-o`/`--output`, `--chunk`, `--image`, `--audio`, `--video`, `-pps`, `-tgs`, `-npp`, `--diffusion-steps`, `-lr`/`--learning-rate`, `-mv`/`--model-vocoder`, `-hfv`/`--hf-repo-v`, `--tts-speaker-file`, `-fitp`/`--fit-print`, `--show-timings`, `--prompt-cache`, `--tensor-filter`, `-gan`/`--grp-attn-n` y las opciones de descarga `--mtp`, `--dflash` y `--eagle3` | Nada; el servidor nunca las usó. |
| Los valores `draft-eagle3`, `draft-dflash` y `draft-dspark` de `--spec-type`, con los modelos de borrador Eagle3, DFlash y DSpark | `draft-mtp`, `draft-simple` o uno de los tipos n-gram |
| Los backends de ggml distintos de CPU, CUDA, Vulkan y ROCm (HIP): BLAS, CANN, ET, Hexagon, Metal, MUSA, OpenCL, OpenVINO, RPC, SYCL, virtgpu, WebGPU, zDNN y ZenDNN, con sus opciones de CMake (como `GGML_BLAS`, `GGML_METAL`, `GGML_RPC` o `GGML_SYCL`) y `--rpc` | CPU, CUDA (`--cuda`), ROCm (`--rocm`) o Vulkan (`--vulkan`) |
| En `mgs.h`, `mgs_get_logits`, `mgs_get_embeddings`, `mgs_encode`, `mgs_set_causal_attn`, `mgs_model_has_encoder`, `mgs_model_has_decoder`, `mgs_model_decoder_start_token`, `mgs_model_is_recurrent`, `mgs_get_ctx_other` y el campo `ctx_other` de `mgs_context_params`; en `mgs-ext.h`, `mgs_set_embeddings_layer_inp`, `mgs_get_embeddings_layer_inp`, `mgs_model_target_layer_ids` y `mgs_model_target_layer_ids_n`; en `mtmd.h`, `mtmd_decode_use_non_causal`, `mtmd_image_tokens_get_n_tokens` y `mtmd_image_tokens_get_n_pos` | `mgs_get_logits_ith`, `mgs_get_embeddings_ith` o `mgs_get_embeddings_seq`, `mgs_decode` (que también ejecuta modelos sin caché KV), `mtmd_input_chunk_get_n_tokens` y `mtmd_input_chunk_get_n_pos`; los demás servían a modelos codificador-decodificador, solo de embeddings o densos, o a la caché KV compartida entre contextos, y no tienen sustituto. |
| Los proyectores multimodales de Gemma 3 (`gemma3`, `gemma3nv`, `gemma3na` y MobileNetV5) | Nada; usá el proyector de una de las [familias admitidas](#supported-models). |
| Las rutas de OpenAI sin `/v1` (`GET /api/models`, `POST /api/chat/completions`, `POST /api/responses`, `POST /api/audio/transcriptions`, `POST /api/chat/completions/input_tokens`, `POST /api/responses/input_tokens`) y `GET /api/v1/health` | La misma ruta bajo `/api/v1` (`GET /api/v1/models`, `POST /api/v1/chat/completions`…) y `GET /api/health`. `POST /api/models` y `DELETE /api/models` del modo router se mantienen. |
| Los campos `ahead_api` y `ahead_webui` de los eventos `mgs_queue` | `running_api`, `running_webui`, `waiting_api` y `waiting_webui` ([cola de pedidos](#request-queue)) |
| Las variables de los instaladores `MGS_INSTALL_SOURCE_DIR`, `MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`, `MGS_BUILD_JOBS` y `MGS_BUILD_BACKEND`, y la opción `--build-ui` de los instaladores y de `build/build-local.sh` | `--build-dir`, `--prefix`, `--jobs` y `--cuda`, `--rocm`, `--vulkan` o `--cpu`; los fuentes van siempre a `/opt/moe-gguf-server-source` y la interfaz siempre se compila salvo que se indique `--no-ui`. |
| Configurar los servicios web como un paso aparte después de la instalación | El instalador los configura en la misma ejecución (`--domain`, `--certificate`, `--private-key`, `--proxy-ip`); `configure-web.sh` sigue disponible para reconfigurar sin recompilar. |
| El servicio de inferencia en `127.0.0.1:8080` (`--port 8080`) detrás de Apache | El socket Unix `/opt/moe-gguf-server/_/temp/web/mgs.sock` ([despliegue](#deployment)) |

CMake avisa de que las variables `-D` eliminadas no se utilizan, en lugar de
aplicarlas, así que conviene revisar la salida de configuración de las
compilaciones manuales.

La interfaz web guarda sus ajustes (`~/.mgs/config/ui-settings.json`) y sus chats (`~/.mgs/chats/`) en el servidor,
y en el navegador solo su clave API (`MgsUi.apiKey` en `localStorage`). No lee ni migra los ajustes que
versiones anteriores guardaban en `localStorage` (`MgsUi.config`, `MgsUi.userOverrides`,
los modelos favoritos, las listas de herramientas, el esfuerzo de razonamiento, las
recomendaciones MCP descartadas, el idioma y el modo de color de la antigua biblioteca de temas), así que los ajustes empiezan con sus
valores predeterminados y hay que volver a ingresar la clave API una vez. Tampoco
lee ni migra los datos que versiones anteriores guardaron con otros nombres, la clave
independiente `theme`,
los marcadores de razonamiento y de llamadas a herramientas incrustados en mensajes
antiguos, los adjuntos de texto pegado del antiguo tipo `context`, el antiguo formato
de lista de recomendaciones MCP descartadas o el antiguo campo `webui_settings` del
servidor. Tampoco lee las conversaciones que versiones anteriores guardaban en el navegador (la base de datos
IndexedDB `MgsUi`): para conservarlas, exportalas desde la versión anterior antes de actualizar e importá la
exportación después, y así quedan guardadas en el servidor. La importación de
conversaciones solo acepta exportaciones JSONL y ZIP, dentro de los límites que se indican en
[límites de seguridad](#security-limits); el antiguo formato de exportación JSON se rechaza. Los ajustes antiguos se quedan en el navegador hasta que se borran:
para eliminarlos, abrí la consola del navegador (F12) en la página de la interfaz y ejecutá
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
una vez por cada dirección y puerto con los que se haya abierto la interfaz.

Todas las descargas se guardan en la caché `~/.cache/mgs/` (bajo `XDG_CACHE_HOME` cuando está definida):
los archivos descargados con `-mu`/`--model-url` y, conservando su propia estructura
(`~/.cache/mgs/hub/models--<organización>--<repo>/`), las descargas de Hugging Face. Ninguna
variable cambia esa carpeta. Los archivos que versiones anteriores guardaron en otra carpeta de caché,
incluida la caché de Hugging Face (`~/.cache/huggingface/hub`), se vuelven a descargar.

### Archivos de modelo antiguos

También se eliminó el código de compatibilidad con estas conversiones antiguas.
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

- Comprobá qué versión está instalada con `mgs --version` (muestra, por ejemplo, `version: 0.1.57`); la compilación
  local lo ejecuta al terminar y CMake muestra `mgs version: 0.1.57` durante la configuración. La interfaz muestra la
  misma versión con Ajustes → Visualización → **Mostrar la versión de compilación** (esquina inferior derecha) y en
  **Información de compilación** del cuadro de información del modelo. Cada cambio en el proyecto sube el último
  número: 0.1.0, 0.1.1, 0.1.2, etc.
- Repetí el comando `curl | bash` para actualizar desde `main`, o el instalador local correspondiente para recompilar y sustituir el ejecutable.
  Volvé a pasar las mismas opciones de la web (`--domain`, `--proxy-ip`…), porque cada ejecución reescribe la
  configuración web. El instalador reinicia los dos servicios al terminar; para reiniciar a mano el servicio de
  inferencia, ejecutá `systemctl restart moe-gguf-server` en Debian o `rc-service moe-gguf-server restart` en Alpine.
- Consultá `/root/webapp-install.log`, `journalctl -u moe-gguf-server` en Debian
  o `/var/log/moe-gguf-server.log` en Alpine. Los registros Apache están en
  `/var/www/DOMINIO-logs/`.
- Si no podés compilar CUDA, instalá el Toolkit por separado o usá `--rocm`, `--vulkan` o `--cpu`.
  Si falta NCCL, revisá los repositorios y la versión de CUDA, o elegí `--no-nccl`. Si no se encuentra ROCm,
  instalalo con sus archivos de desarrollo o definí `HIPCXX` ([ROCm](#rocm)).
- Si falla la interfaz, comprobá Node.js, npm, las descargas y el registro.
  No sustituyas la interfaz por una precompilada del proyecto original.
- Para deploy, tras reorganizar los fuentes, utilizá una compilación nueva en `_/temp/`.
  SSHFS puede impedir ejecutar archivos; usá un disco local en producción.
- Para la compilación local, repetí el comando de una línea anterior para descargar
  el `main` actual, o `bash build/build-local.sh` para usar tus fuentes locales, y reiniciá la copia
  que lanzaste a mano. Consultá la salida de la terminal si falla o informa
  bibliotecas fuera de `lib/`. El registro temporal `build.log` se elimina junto
  con la carpeta de compilación.
- Un 404 en clientes antiguos suele requerir cambiar la URL base a `/api/v1`.
  La clave API debe enviarse en la cabecera Authorization.

Consultá [CODE.es-AR.md](CODE.es-AR.md) para ver el mapa de implementación y los puntos de extensión.

<a id="license"></a>
## Licencia

`mgs` se distribuye bajo la [licencia PolyForm NonCommercial 1.0.0](../LICENSE). Podés usarlo,
modificarlo y compartirlo con cualquier fin no comercial: uso personal, estudio, investigación,
proyectos personales por hobby y uso por parte de organizaciones benéficas, instituciones
educativas, organismos públicos de investigación e instituciones públicas. El uso comercial, como
el de una empresa para su actividad, necesita una licencia aparte de nipegun.

Las partes que provienen de llama.cpp (MIT, The ggml authors), de FreeToken (Apache 2.0) y de las
bibliotecas incluidas conservan sus propias licencias; sus avisos están en `build/licenses/` y la
compilación local los instala, junto con `LICENSE`, en `~/.local/share/doc/mgs/licenses/`.
