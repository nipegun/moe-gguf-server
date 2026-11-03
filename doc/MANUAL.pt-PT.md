# Manual do utilizador

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Conteúdo: [instalação](#installation), [compilação local](#local-build), [implementação](#deployment),
[chat](#chat), [monitor](#monitor), [modelos e ferramentas](#models), [escolher modelos](#settings-models), [página do modelo](#model-page), [modelos suportados](#supported-models), [API](#api),
[fila de pedidos](#request-queue), [limites de segurança](#security-limits), [Vulkan](#vulkan), [ROCm](#rocm), [desempenho](#performance), [compatibilidade removida](#removed-compatibility),
[manutenção](#maintenance), [licença](#license).

<a id="installation"></a>
## Instalação e primeiro arranque

Instale como root no servidor de produção. O Debian/Ubuntu aceita o instalador
diretamente pela entrada padrão:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

O instalador implementa tudo num único passo: compila o `mgs`, instala-o em
`/usr/local/bin/mgs` e configura os serviços de inferência e web
([implementação](#deployment)). Não é preciso transferir primeiro nenhum ficheiro adjacente. O script de entrada transfere o
arquivo completo do código-fonte de `main` a partir [deste repositório](https://github.com/nipegun/moe-gguf-server/), verifica-o
e sincroniza uma cópia gerida em `/opt/moe-gguf-server-source`. As transferências
e as caches de compilação ficam em `_/temp/` do projeto. São necessários o curl e os certificados de CA
para obter o script de entrada; as dependências de bootstrap em falta são depois
instaladas automaticamente.

Para passar opções de compilação e web, use:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com --cpu --jobs 4
```

A cópia gerida fica sempre em `/opt/moe-gguf-server-source`, pertence ao root e tem o
modo 700. O instalador só a adota enquanto estiver vazia ou a reutiliza quando tiver a sua marca de origem do código-fonte;
os conteúdos não relacionados são rejeitados. Um bloqueio impede instalações simultâneas
nessa cópia. As execuções seguintes de `curl | bash` sincronizam os ficheiros do código-fonte, incluindo os ficheiros
removidos de `main`, e preservam `_/` e as caches de compilação. As chaves de API, os modelos e os
certificados ficam fora dessa árvore de código-fonte e são conservados. Uma transferência,
uma compilação ou uma verificação de opções falhada não instala um novo executável; volte a executar o mesmo comando para
tentar novamente. Não é necessária nenhuma operação Git.

Um script transferido isoladamente usa o mesmo modo remoto. Executar
`bash deploy/install-update-reinstall-debian.sh` a partir de uma cópia local completa
compila essa cópia sem transferir o código-fonte. Os argumentos a seguir a `bash -s --`
são reencaminhados para o instalador de compilação. `--help` não requer root nem a transferência
do código-fonte. `--no-system-deps` desativa também a instalação dos pacotes de bootstrap: o curl,
os certificados de CA, o tar, o gzip, o rsync, o util-linux e o coreutils já têm de existir.
Os subprocessos do instalador não podem consumir o script a partir do pipe; todas as fases partilham
um único fluxo de saída `/root/webapp-install.log`.

As opções web são verificadas antes da compilação, para que um erro não surja apenas depois
de uma compilação longa: `--domain NAME` (por predefinição, a saída de `hostname -f`, ou de
`hostname` ou `uname -n` quando esta não for um nome válido), `--certificate PATH` e
`--private-key PATH` (sempre em conjunto) e `--proxy-ip IP` (consulte
[implementação](#deployment)). `/opt/moe-gguf-server-source/deploy/configure-web.sh`
continua disponível para alterar mais tarde a configuração web sem recompilar.

O Alpine continua a usar uma cópia local completa e
`bash deploy/install-update-reinstall-alpine.sh`, com o Bash instalado previamente; só
compila para a CPU. No Debian, escolha o backend:

- CUDA (`--cuda`, a predefinição) para GPU NVIDIA. O CUDA Toolkit e o controlador da
  NVIDIA já têm de estar instalados; o instalador adiciona os pacotes NCCL que
  correspondem a essa versão do CUDA (ver abaixo). `--no-nccl` desativa explicitamente o NCCL.
- ROCm (`--rocm`) para GPU AMD suportadas pelo ROCm; `--rocm-targets LIST` escolhe os
  alvos de GPU. O ROCm já tem de estar instalado ([ROCm](#rocm)).
- Vulkan (`--vulkan`) para GPU AMD, Intel ou NVIDIA ([Vulkan](#vulkan)).
- Apenas CPU (`--cpu`), para máquinas sem GPU.

`--no-system-deps` deixa a gestão de pacotes a cargo do administrador.

A interface local precisa do Node.js 20.19+ na série 20.x ou 22.12+. `--no-ui` omite-a. As outras
opções são `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST` e `--no-openssl`. `CUDACXX` seleciona o compilador CUDA
e `HIPCXX` o compilador clang do ROCm. Os diretórios de compilação e de preparação têm de ser descendentes distintos
e sem sobreposição do diretório `_/temp/` do projeto.

O instalador adiciona os pacotes NCCL correspondentes e o repositório da NVIDIA quando é necessário,
mas nunca instala nem atualiza o CUDA Toolkit, o ROCm nem os controladores da GPU. Liga as bibliotecas
internas estaticamente e instala `/usr/local/bin/mgs`. Nenhum dos
instaladores usa ou instala o sudo.

No Debian/Ubuntu, `libnccl2` e `libnccl-dev` são instalados em conjunto na mesma
versão exata selecionada para a versão do CUDA detetada. Este comando APT usa
`--allow-downgrades` para que um pacote NCCL mais recente já instalado não bloqueie a instalação
da versão selecionada. A opção só é passada ao comando de instalação do NCCL.

Se a distribuição incluir um Node.js mais antigo, instale primeiro uma versão suportada.
A interface é sempre compilada localmente; a compilação não pode transferir recursos pré-compilados
do upstream, que não têm as traduções nem as rotas de API deste projeto. Uma falha na compilação da interface
interrompe a instalação.

A partir de uma cópia local completa, escolha uma compilação de CPU ou liste as opções de instalação:

```bash
bash deploy/install-update-reinstall-debian.sh --domain ai.example.com --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

No Alpine, instale o Bash com `apk add --no-cache bash` antes de executar o respetivo instalador.

Para executar um único modelo, use:

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

Não é necessária nenhuma outra opção: o `mgs` escolhe as definições que dão a maior velocidade de inferência na máquina em que
é executado. O ajuste automático coloca o modelo na GPU e, quando a memória do sistema o permite, mantém os especialistas MoE na RAM para
que a VRAM livre se torne a [cache de especialistas MoE](#expert-cache); também escolhe o
[microlote e o tamanho do contexto](#context-size). O número de threads da CPU acompanha o processador
([threads da CPU](#threads)), e as camadas MTP encontradas no GGUF são ativadas e mantidas na VRAM
([MTP](#mtp-vram)). Qualquer opção indicada explicitamente (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`,
`--override-tensor`…) prevalece sobre a escolha automática. O registo de arranque mostra o que foi escolhido.

Execute o `mgs` com o seu próprio utilizador: nunca precisa de root. Se for iniciado como root, regista o
aviso `mgs is running as root: run it with an unprivileged user instead` (o
servidor MCP da shell local daria ao modelo uma shell de root) e continua a arrancar.

Abra o URL HTTPS apresentado no arranque, normalmente `https://127.0.0.1:11443/`.
Carregar um modelo grande pode demorar. A interface
web e a documentação da API continuam disponíveis enquanto os pedidos ao modelo indicam
que este está a carregar. `--help` lista as opções herdadas do motor. Os ficheiros GGUF, a memória da GPU,
o comprimento do contexto e os projetores específicos do modelo têm de corresponder à carga de trabalho pretendida.

O código do servidor ativa o HTTPS por predefinição em todos os métodos de compilação e de instalação.
Associa-se à primeira porta HTTPS disponível a partir de 11443 e, depois, a uma porta de redirecionamento HTTP
a partir de 11080, sem contar com a porta HTTPS escolhida. O HTTP responde com um redirecionamento permanente 308
para a porta HTTPS real e mantém o host, o caminho e a query string pedidos.
O registo apresenta os URL selecionados; os conflitos de portas fazem avançar cada sequência de forma independente.
`--port N` fixa a porta principal e falha se esta estiver ocupada; `--port 0` deixa o sistema operativo
escolhê-la. `--http-port N` altera o início da procura da porta de redirecionamento.
As variáveis de ambiente correspondentes são `MGS_ARG_PORT` e `MGS_ARG_HTTP_PORT`.

Até à versão 0.1.34 a pesquisa parava na segunda porta: um `mgs` iniciado enquanto a 11443 estava em uso (por exemplo, por outro `mgs`) terminava com `couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 11444` em vez de usar a porta livre seguinte.

Sem ficheiros de certificado, o executável cria um certificado autoassinado e a respetiva
chave privada da primeira vez que arranca, sem precisar de um comando `openssl`
externo, e guarda-os em `~/.mgs/certificates/` (`fullchain.pem` e
`privkey.pem`, e a chave só pode ser lida por si). Todas as execuções seguintes, e todas as novas compilações,
reutilizam-nos, pelo que o navegador mantém a sua exceção de confiança e não precisa de recarregar a página; só se cria um novo
par quando esses ficheiros faltam, não correspondem um ao outro ou o certificado
expirou (ao fim de 365 dias). O certificado abrange `localhost`, os endereços de
loopback, o host indicado com `--host`, bem como o nome da máquina e os endereços de rede
que esta tinha quando o certificado foi criado; para eliminar por completo o aviso do navegador, importe
`~/.mgs/certificates/fullchain.pem` como fidedigno, e apague ambos os ficheiros para criar
um novo. Se não for possível escrever nessa pasta, cada execução gera o seu próprio certificado em
memória. Indique `--ssl-cert-file /path/fullchain.pem` e `--ssl-key-file
/path/privkey.pem` para usar um certificado em que os seus clientes confiem. Um par em falta ou inválido impede
o arranque. As compilações feitas com `--no-openssl` não podem servir HTTPS e exigem `--http` explícito
(ou `MGS_ARG_HTTPS=false`); o HTTP simples começa a procura de porta em 11080.
Os processos filhos do router usam HTTP explícito em portas de loopback privadas. Os sockets Unix e
o `AIP_MODE=PREDICTION` do Vertex AI mantêm o seu transporte HTTP.

A configuração do CMake está em `build/CMakeLists.txt`. Tanto os scripts de instalação como os de compilação
local incluem `build/` na sua cópia do código-fonte e selecionam-no automaticamente.
Os instaladores repõem a configuração gerada do CMake se uma cache existente apontar
para outro diretório de código-fonte. As compilações locais começam sempre num diretório novo em
`/tmp/` e removem-no ao terminar. Para usar o CMake diretamente, execute
`cmake -S build -B _/temp/cmake` a partir da raiz do projeto; os comandos habituais do instalador e da
compilação local mantêm-se iguais. O HTTPS usa por predefinição o OpenSSL do sistema
(`MGS_OPENSSL=ON`); numa compilação direta com o CMake, `-DMGS_BUILD_BORINGSSL=ON` ou
`-DMGS_BUILD_LIBRESSL=ON` transfere e compila, em alternativa, o BoringSSL ou o LibreSSL, e
`-DMGS_OPENSSL=OFF` sem nenhum deles compila o servidor sem HTTPS.

<a id="local-build"></a>
## Compilação local para utilização manual

O `build/build-local.sh` compila o projeto no seu próprio computador e instala uma
cópia autónoma para o seu utilizador, que inicia manualmente. Não instala serviços, não deixa
nada a correr em segundo plano, nunca usa root nem sudo e nunca instala pacotes.
Execute-o como o seu utilizador habitual, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

O comando de uma linha transfere o arquivo do código-fonte de `main`, sem Git, para a pasta privada
de compilação em `/tmp`. O arquivo e o código-fonte extraído são removidos juntamente com
os restantes ficheiros temporários de compilação ao terminar. Uma cópia autónoma do script funciona
da mesma forma. A partir de uma cópia local completa do projeto, compile o seu código-fonte local com:

```bash
bash build/build-local.sh
```

Executar este script pelo seu caminho absoluto também usa o projeto completo que
o contém. A mensagem `Using local sources: ...` confirma que pasta está
a ser compilada, incluindo as traduções da interface. Se a execução terminar com
`Error: Compilation failed.`, não foi instalado nenhum executável novo; uma
instalação existente mantém a interface anterior. Leia o primeiro erro do compilador acima de
`ninja: build stopped`, corrija a sua causa e volte a executar o mesmo script. Quando a
compilação e a instalação forem bem-sucedidas, reinicie o `mgs` e recarregue a interface web.

Erros como `std::vector` sem nenhum membro chamado `mPushBack` ou
`no declaration matches` em `subproc.cpp` indicam nomes inconsistentes no código-fonte.
Use os ficheiros de código-fonte corrigidos e volte a executar a compilação. A compilação da interface também
exige que as importações da sua configuração correspondam aos nomes exportados pelos módulos
do código-fonte; uma exportação em falta é comunicada pelo passo npm/Vite.

Se o SvelteKit indicar `Page options are ignored when router.type === 'hash'`
e mencionar `fLoad`, as rotas do chat têm de exportar o nome exato `load` exigido pela
framework. As rotas corrigidas mantêm a validação da chave de API. A compilação executa
`svelte-kit sync` antes de gerar os recursos e compilar a interface, e
para se a sincronização falhar. Uma falha em `Provisioning UI assets` continua a
impedir a instalação, mesmo quando as bibliotecas C++ já foram compiladas.

Para passar opções através do comando de uma linha, use `bash -s --`; por exemplo, uma compilação de CPU:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` mostra as opções sem transferir o arquivo do código-fonte nem compilar.

O resultado é instalado para o seu utilizador sob o prefixo `~/.local`:

| caminho | conteúdo |
| --- | --- |
| `~/.local/bin/mgs` | O programa que executa: um pequeno lançador estático que encontra `../lib/mgs/` em relação a si próprio e inicia o servidor através do carregador incluído. |
| `~/.local/lib/mgs/` | O binário do servidor (`mgs`, com a interface web e a documentação da API incorporadas) e todas as bibliotecas partilhadas de que precisa: o carregador dinâmico e a biblioteca C da glibc, o runtime de C++, o OpenMP, o OpenSSL, os módulos de serviço de nomes da glibc e, nas compilações CUDA, o runtime do CUDA, o cuBLAS e o NCCL; nas compilações ROCm, as bibliotecas do ROCm (HIP, hipBLAS, rocBLAS, hipBLASLt…) e as pastas de kernels de GPU do rocBLAS e do hipBLASLt (`rocblas/library`, `hipblaslt/library`). |
| `~/.local/share/doc/mgs/licenses/` | A licença PolyForm NonCommercial 1.0.0 do projeto (`LICENSE`) e os avisos de licença de terceiros copiados de `build/licenses/` da árvore do código-fonte (llama.cpp, FreeToken, Swagger UI, cpp-httplib e nlohmann/json). |

Só o utilizador que executa o script pode usar esta instalação. Tudo é resolvido em relação ao local onde está o
lançador: este executa `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` a partir do seu prefixo, e o servidor e as suas bibliotecas procuram
em `lib/mgs/` antes de qualquer outro sítio, incluindo `LD_LIBRARY_PATH`. A única biblioteca que continua a ser obtida do sistema é o
controlador NVIDIA (`libcuda.so.1` e as bibliotecas `libnvidia-*` que este carrega), porque tem de corresponder ao módulo do kernel
em execução; com `--vulkan`, o controlador Vulkan do sistema ([Vulkan](#vulkan)); com `--rocm`, do sistema só é usado o
controlador do kernel amdgpu (`/dev/kfd`, `/dev/dri`), e o lançador aponta o rocBLAS e o hipBLASLt para as pastas de kernels
incluídas em `lib/mgs/` ([ROCm](#rocm)). No final, o script lista, através do carregador incluído, as bibliotecas que o servidor carregaria, falha se alguma
vier de fora de `lib/mgs/` e executa `--version` através do lançador.

O Debian adiciona `~/.local/bin` ao `PATH` no início de sessão assim que essa pasta existe (através de `~/.profile`), pelo que, depois de voltar
a iniciar sessão, pode executar o `mgs` pelo nome; o script mostra um lembrete enquanto este não estiver no `PATH`. Inicie-o sempre que
quiser e pare-o com Ctrl+C:

```bash
mgs --model /path/model.gguf
```

O caminho completo também funciona: `~/.local/bin/mgs --model /path/model.gguf`.

Abra `https://127.0.0.1:11443/` para o chat e `https://127.0.0.1:11443/api/doc/` para
a documentação da API. Todas as opções deste manual se aplicam. Execute o `mgs` sem `--model` para iniciar o
router multimodelo e escolha a pasta de modelos em Definições → Modelos
([escolher e carregar modelos](#settings-models)).

Requisitos, que instala por si próprio como root, porque o script apenas os
verifica: `cmake`, um compilador C/C++ e a biblioteca C estática (`build-essential`, que
traz `libc6-dev`), `patchelf`, `libssl-dev` (ou
`--no-openssl`), Node.js 20.19+/22.12+ com npm (ou `--no-ui`) e o toolkit do backend
escolhido: para CUDA (a predefinição), um CUDA Toolkit já instalado e o NCCL com os seus ficheiros
de desenvolvimento (`libnccl2` e `libnccl-dev`, ou `--no-nccl`); para `--rocm`, um ROCm já
instalado ([ROCm](#rocm)); para `--vulkan`, os pacotes Vulkan ([Vulkan](#vulkan));
`--cpu` não precisa de nenhum. O `nvcc` é obtido de `CUDACXX`, do `PATH` ou de
`/usr/local/cuda/bin/nvcc`. O Ninja é usado quando está instalado.
Os utilitários básicos incluem `tar`, `coreutils` e `util-linux`. As compilações remotas também
exigem `curl`, `ca-certificates` e `gzip`; não é instalada automaticamente nenhuma dependência.

Opções: `--prefix PATH` (outro prefixo de instalação), `--jobs N`, `--cuda`, `--rocm`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--rocm-targets LIST`, `--no-ui`, `--no-openssl`, `--no-nccl`
e `--help`. `CUDACXX` seleciona o compilador CUDA e `HIPCXX` o compilador clang do ROCm.

Convém saber:

- A compilação é otimizada para a CPU deste computador e para as GPU presentes durante
  a compilação. Volte a compilar em cada computador em vez de copiar a instalação.
- A instalação pode ser deslocada como prefixo: mova `bin/mgs` e `lib/mgs/` em conjunto, sem alterar os seus níveis `bin/` e
  `lib/`. Inicie sempre `bin/mgs`; `lib/mgs/mgs` é interno e não deve ser executado diretamente, porque contornaria
  o carregador incluído.
- Voltar a executar o script atualiza a instalação. Só substitui `bin/mgs`, `lib/mgs/` e
  `share/doc/mgs/licenses/`. Uma cópia que já esteja em execução continua a funcionar; reinicie-a para usar a nova compilação.
- O script recusa um prefixo onde `bin/mgs`, `lib/mgs/` ou essa pasta de licenças já existam, a não ser que tenha sido ele a
  criá-los; deixa uma marca oculta `.moe-gguf-server-bundle` em `lib/mgs/`.
- `--prefix PATH` instala sob outro prefixo em que o seu utilizador possa escrever; não pode ser `/` nem uma pasta dentro do
  projeto.
- O código-fonte preparado, os ficheiros de compilação, as caches, a instalação montada e o registo usam uma
  pasta privada `/tmp/moe-gguf-server-build.XXXXXXXX/`. Depois de copiar o resultado
  para o prefixo, o script remove essa pasta. Também faz a limpeza em caso de erro
  e de interrupções tratadas. Cada execução compila do zero, sem escrever na
  árvore do código-fonte nem em `/opt/`. As compilações locais não aceitam `--build-dir`.
- O prefixo fica bloqueado enquanto se copiam e substituem os artefactos concluídos,
  o que impede atualizações simultâneas da mesma instalação.
- Uma compilação CUDA precisa do controlador NVIDIA para arrancar; use `--rocm` ou `--vulkan` em computadores com outra GPU e `--cpu` em
  computadores sem GPU.

<a id="vulkan"></a>
## Vulkan (GPU AMD, Intel e NVIDIA)

Além do CUDA (a predefinição) e da CPU, tanto o `build/build-local.sh` como o instalador do Debian compilam o backend Vulkan
com `--vulkan`, para GPU sem CUDA, como as placas AMD Radeon. Instale primeiro as dependências de compilação como root:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Depois, como o seu utilizador habitual:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

O instalador do Debian aceita a mesma opção e instala esses pacotes Vulkan por si próprio:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` mostra a GPU, por exemplo `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- A instalação local inclui o carregador Vulkan (`libvulkan.so.1`) como qualquer outra biblioteca, mas o controlador Vulkan
  vem do sistema: o carregador lê os controladores registados em `/usr/share/vulkan/icd.d/` (Mesa RADV para AMD,
  ANV para Intel, o controlador NVIDIA) e estes carregam as suas próprias bibliotecas do sistema, porque têm de corresponder ao kernel
  e ao sistema. Compile na versão do Debian que vai executar o binário.
- O utilizador que executa o `mgs` precisa do nó de renderização da GPU (`/dev/dri/renderD*`). Um utilizador com sessão iniciada no ambiente de trabalho já
  o tem; os outros utilizadores precisam do grupo `render` (e de `video`). O `configure-web.sh` adiciona a conta do serviço a ambos.
- O MTP na VRAM, a cache de especialistas MoE, o pré-carregamento de especialistas, `--fit` e o Monitor funcionam como com o CUDA. Na AMD, o
  Monitor lê a utilização, a VRAM e a temperatura a partir do sysfs.
- O `mgs` só conta como memória da GPU a VRAM real (os heaps de memória device-local), nunca a GTT. A GTT é a parte da
  RAM do sistema (normalmente metade) que uma GPU AMD ou Intel consegue mapear: não é memória adicional, e a GPU lê-a
  através do PCIe, muitas vezes mais devagar do que a sua VRAM. As camadas MTP, os pesos densos, a cache KV e a cache de especialistas são
  reservados na VRAM; os especialistas que não cabem ficam na RAM normal, onde a CPU os calcula.
- Uma GPU integrada (os gráficos de um processador AMD ou Intel) não tem VRAM própria: todos os heaps de memória que
  indica são RAM do sistema. O `mgs` conta essa memória e a RAM como um único orçamento e não cria nela a cache de
  especialistas; consulte [GPU integradas](#integrated-gpu).
- O `mgs` verifica onde o kernel mantém realmente essa memória (`/proc/self/fdinfo`, publicado pelo amdgpu, pelo i915 e pelo xe). Se
  outro programa encher a VRAM e o controlador mover parte dela para a GTT, o registo e o Monitor indicam-no, e o
  Monitor mostra a localização real (VRAM e GTT) de cada GPU AMD ou Intel dedicada (uma GPU integrada não tem VRAM
  própria, pelo que não é verificada). A cache de especialistas é reservada com uma
  prioridade de memória mais baixa (`VK_EXT_memory_priority`, usada sempre que a GPU a oferece, sem nenhuma variável para a
  ativar), pelo que o controlador a move antes das camadas MTP, dos pesos ou da cache KV. Sem resizable BAR, os buffers do
  dispositivo também evitam a pequena janela de VRAM que a CPU consegue mapear (256 MiB numa RX 580), que o ambiente de
  trabalho partilha e que o controlador esvazia primeiro para a GTT, e essa janela não conta para eles como memória livre.
- Só o CUDA consegue fixar as páginas do modelo que ficam na RAM, pelo que, com Vulkan, os especialistas chegam à GPU através de memória
  paginável. O `mgs` copia-os através de um pequeno anel intermédio e coloca na fila a cópia para a GPU sem esperar por ela, pelo que
  a GPU e a CPU continuam a trabalhar enquanto os especialistas são carregados: numa RTX 4060 Ti com o Qwen3.6-35B-A3B Q8_0, isto
  aumentou o processamento do prompt de 455 para 515 tokens/s e a geração de 20,6 para 23,2 tokens/s. `--load-mode none` copia os
  especialistas, em vez disso, para memória do anfitrião fixada; antes do anel intermédio, na mesma placa com o modelo Q4_K, aumentou
  os carregamentos de especialistas de 2,7 para 13,1 GB/s e a geração de 27,5 para 36,6 tokens/s. Na
  AMD e na Intel, essa memória fixada conta para a GTT: quando os especialistas não cabem nela, o carregamento falha com
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), como aconteceu com os 18,9 GB de especialistas desse
  modelo com 16,8 GB de GTT numa RX 580. Use-o apenas quando os especialistas couberem na GTT e na RAM livre.
- Com uma GPU NVIDIA através de Vulkan, e com especialistas BF16 em qualquer GPU, os produtos de especialistas convertem primeiro a sua
  entrada num buffer de trabalho. Esse buffer ocupava 512 KiB por cada token do prompt, pelo que os prompts de alguns milhares de tokens
  falhavam com `ErrorOutOfDeviceMemory` assim que a cache de especialistas tinha enchido a VRAM, a menos que se aumentasse
  `--moe-cache-reserve`. Agora ocupa poucos KiB por token (32 MiB para um micro-lote de 4096 tokens do Qwen3.6-35B-A3B), bem dentro da
  reserva predefinida. Numa RTX 4060 Ti com o Qwen3.6-35B-A3B Q8_0, isto também acelerou o processamento do prompt de 515 para 549
  tokens/s.
- Dimensione o modelo de acordo com a máquina: o GGUF inteiro deve caber na RAM livre mais a VRAM, depois de descontar a cache KV e os
  buffers de cálculo. Com 8 GB de VRAM e 16 GB de RAM, isso significa cerca de 18–20 GB de pesos; os ficheiros maiores são lidos
  do disco uma e outra vez e a geração torna-se muito lenta. O Q4_K de 21,6 GB acima atingiu 7,8 tokens/s numa
  RX 580 com 15 GB de RAM e 15,9 tokens/s quando a máquina passou a ter 31 GB (10,0 com `--spec-type none`).

<a id="rocm"></a>
## ROCm (GPU AMD)

Tanto o `build/build-local.sh` como o instalador do Debian compilam o backend ROCm (HIP) com `--rocm`, para GPU AMD
suportadas pela versão do ROCm instalada. Nenhum dos scripts instala o ROCm: instale por si próprio, como root, o ROCm
6.1 ou posterior, com o HIP, o hipBLAS e o rocBLAS e os respetivos ficheiros de desenvolvimento. Para a compilação local,
instale também como root as dependências de compilação comuns (o instalador do Debian instala-as por si próprio):

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates
```

Depois, como o seu utilizador habitual:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --rocm
```

O instalador do Debian aceita a mesma opção:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --rocm
```

- Os scripts encontram o ROCm através do `hipconfig` (no `PATH` ou em `/opt/rocm/bin/hipconfig`) e usam o compilador
  clang que este indica; `HIPCXX` seleciona outro. Uma configuração falhada indica o que falta: HIP 6.1 ou posterior
  com os ficheiros de desenvolvimento do hipBLAS e do rocBLAS.
- Os alvos de GPU são os das GPU presentes durante a compilação, lidos com `amdgpu-arch` ou `rocminfo`. Sem nenhuma GPU
  detetada, ou para compilar para outras, indique `--rocm-targets` com uma lista separada por vírgulas, por exemplo
  `--rocm-targets gfx1100,gfx1030`.
- As versões atuais do ROCm não suportam as GPU GCN até Polaris (`gfx6xx` a `gfx8xx`, como a Radeon RX 580,
  `gfx803`): os scripts avisam sobre esses alvos; use `--vulkan` para elas.
- A instalação local copia todas as bibliotecas do ROCm (HIP, hipBLAS, rocBLAS, hipBLASLt…) para `lib/mgs/`, juntamente
  com as pastas de kernels de GPU que o rocBLAS e o hipBLASLt carregam em tempo de execução (`rocblas/library`,
  `hipblaslt/library`); o lançador aponta as duas bibliotecas para essas cópias (`ROCBLAS_TENSILE_LIBPATH`,
  `HIPBLASLT_TENSILE_LIBPATH`), pelo que nunca leem os kernels de outra versão do ROCm. Só o controlador do kernel
  amdgpu vem do sistema. As pastas de kernels ocupam gigabytes.
- O `/usr/local/bin/mgs` do instalador do Debian carrega as bibliotecas do ROCm do sistema, através do caminho de
  execução registado durante a compilação; quando não encontra alguma, o instalador para e pede para adicionar a pasta
  `lib` do ROCm a `/etc/ld.so.conf.d/` e executar `ldconfig`.
- O utilizador que executa o `mgs` precisa de `/dev/kfd` e `/dev/dri`, ou seja, dos grupos `render` e `video`; o
  instalador adiciona a conta do serviço a ambos.

<a id="deployment"></a>
## Implementação em produção

O instalador ([instalação](#installation)) compila, instala e configura os serviços
web numa única execução:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --domain ai.example.com
```

Sem `--domain`, o domínio é a saída de `hostname -f` (ou de `hostname` ou
`uname -n` quando esta não for um nome válido). O domínio tem de resolver para o servidor
de produção. Para alterar mais tarde a configuração web sem recompilar, execute como root, a
partir do diretório do código-fonte (`/opt/moe-gguf-server-source` nas instalações remotas):

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

O `configure-web.sh` aceita as mesmas opções web que o instalador (`--domain`,
`--certificate`, `--private-key`, `--proxy-ip`), além de `--binary PATH` para uma localização
do executável diferente da predefinida, `--no-system-deps` e `--check`, que apenas valida as
opções. Cada execução reescreve toda a configuração a partir das opções que recebe:
voltar a executá-lo sem `--proxy-ip` remove o proxy e, sem `--domain`, o domínio volta a
ser o de `hostname -f`.

O serviço `moe-gguf-server-web`
executa uma instância dedicada do Apache como `moe-gguf-server`, sem root.
Cada arranque seleciona a primeira porta HTTP disponível a partir de 11080 e a primeira porta HTTPS
disponível a partir de 11443. Cada sequência avança de forma independente até 65535, sem
atribuir a mesma porta aos dois protocolos. Inspeciona os sockets TCP IPv4/IPv6 em escuta
e tenta novamente se o Apache detetar que uma porta ficou ocupada depois de ter sido selecionada. As outras
falhas de configuração interrompem o arranque, em vez de serem confundidas com conflitos de portas.

O HTTP redireciona de forma permanente para a porta HTTPS selecionada e mantém o caminho e a query string.
O redirecionamento não é guardado em cache. O serviço de inferência (`moe-gguf-server`, também
executado como `moe-gguf-server` e iniciado com `--http`) não escuta em nenhuma porta TCP: escuta no
socket Unix `/opt/moe-gguf-server/_/temp/web/mgs.sock`, que só essa conta pode abrir, e o Apache
reencaminha-lhe os pedidos. Antes de escutar, o `mgs` apaga um ficheiro de socket deixado por um
servidor que já não está em execução, mas nunca o socket de um servidor que ainda responde. A instalação requer root;
nenhum dos serviços precisa de ser executado como root. Não é configurado nenhum HAProxy e não são importados nem modificados
sites partilhados do Apache.

Parar o serviço de inferência (`systemctl stop moe-gguf-server` ou `systemctl restart moe-gguf-server`), tal como Ctrl+C ou um `SIGTERM` quando executa o `mgs` por si, encerra-o de forma ordenada em um ou dois segundos, também a meio de um prompt longo: os pedidos em curso recebem um erro 503 (`The server is shutting down.`) ou, em streaming, o seu stream termina. Um segundo Ctrl+C ou `SIGTERM` termina o processo de imediato, e enquanto o modelo está a carregar o primeiro já basta.

Os URL reais são publicados depois de ambos os sockets de escuta estarem abertos:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

Por exemplo, HTTP 11081 e HTTPS 11445 redirecionam para
`https://ai.example.com:11445/`. O ficheiro é removido ao encerrar e gerado de novo
no arranque; um reinício pode escolher portas diferentes. Os clientes, as firewalls e qualquer
proxy externo têm de usar os valores publicados. A porta faz parte da
[origem do navegador](https://developer.mozilla.org/en-US/docs/Glossary/Origin), pelo que uma
porta HTTPS diferente tem histórico e definições locais separados. As conversas existentes
permanecem na sua origem inicial; a exportação e a importação permitem transferi-las.

A conta do sistema não tem palavra-passe interativa. Os modelos são guardados em
`/opt/moe-gguf-server/models`, e a configuração do programa está na pasta pessoal da
conta do serviço, `/opt/moe-gguf-server/.mgs/config/`: a chave de API (`api.key`), o domínio
web (`web-domain`), o proxy de confiança (`web-proxy-ip`, apenas com `--proxy-ip`), o modelo de
configuração do Apache (`apache2.conf.in`) e as definições da
interface (`ui-settings.json`), todos pertencentes a `moe-gguf-server` e com modo 0600.
A saída da instalação fica registada em `/root/webapp-install.log`, e
`/root/webapp-credentials.txt` guarda todas as credenciais que a implementação cria: a
conta do serviço (`moe-gguf-server`, sem palavra-passe e com o início de sessão desativado) e
a chave de API. Ambos os ficheiros têm modo 600. A reconfiguração mantém as chaves,
os modelos e os certificados existentes.

Para usar o seu próprio certificado TLS, execute:

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

O certificado está em `/opt/moe-gguf-server/certificates/` (`fullchain.pem` e
`privkey.pem`, pertencentes à conta `moe-gguf-server`, e a chave só pode ser lida por ela).
Se não forem indicados ficheiros, o script mantém o certificado que já lá estiver ou, da primeira
vez, cria um autoassinado, que todas as reinstalações seguintes reutilizam, pelo que os navegadores
mantêm a sua exceção de confiança. Confie nele no seu cliente ou indique um certificado em que os seus
clientes já confiem. Os ficheiros indicados são copiados para essa pasta; a renovação do certificado
tem de atualizar essas cópias e reiniciar `moe-gguf-server-web`. O script não obtém nem renova certificados Let's Encrypt.
Use o mesmo comando depois de renovar os certificados de origem.

Quando um proxy inverso (HAProxy, nginx…) está à frente do Apache, indique o seu endereço
com `--proxy-ip` (IPv4 ou IPv6), ao instalador ou ao `configure-web.sh`:

```bash
bash deploy/configure-web.sh --domain ai.example.com --proxy-ip 10.7.10.1
```

O Apache carrega então o `mod_remoteip`. Em HTTPS, exige o cabeçalho do protocolo PROXY
apenas a esse endereço (`RemoteIPProxyProtocol On`, com todos os outros endereços listados em
`RemoteIPProxyProtocolExceptions`), pelo que as outras máquinas continuam a ligar-se
diretamente; em HTTP, confia no cabeçalho `X-Forwarded-For` desse endereço (`RemoteIPHeader`,
`RemoteIPTrustedProxy`). Os registos de acesso guardam o endereço real do cliente. Sem
`--proxy-ip`, não é configurado nenhum proxy.

O serviço usa o modo router. Coloque modelos GGUF legíveis em
`/opt/moe-gguf-server/models`, ou use a gestão de modelos da interface. Para
opções de linha de comandos especializadas, use um override do systemd no Debian/Ubuntu ou a
configuração do serviço OpenRC no Alpine. Mantenha o serviço de inferência no seu socket Unix.
O configurador web gera de novo o ficheiro de serviço que gere sempre que é executado outra vez.

Reinicie o serviço web para voltar a selecionar as portas:

```bash
systemctl restart moe-gguf-server-web
```

No Alpine, use `rc-service moe-gguf-server-web restart`. Os registos de arranque estão em
`journalctl -u moe-gguf-server-web` no Debian ou em `/var/log/moe-gguf-server-web.log`
no Alpine. Os diagnósticos do Apache estão em `/opt/moe-gguf-server/_/temp/web/startup.log`
e `error.log`; os registos de acesso do domínio continuam em `/var/www/DOMAIN-logs/`. O modelo
autónomo de configuração é `/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. Não edite a configuração
gerada em `_/temp/web/`; é reconstruída em cada arranque.

<a id="chat"></a>
## Utilizar a interface

1. Abra o URL do servidor e introduza a respetiva chave de API quando lhe for pedida, ou em Definições → Geral.
2. Selecione um modelo. No modo router, carregue um modelo disponível se nenhum estiver carregado; enquanto não houver
   nenhum, o seletor de modelos tem uma ligação para Definições → Modelos, onde escolhe a pasta dos seus modelos
   ([escolher e carregar modelos](#settings-models)).
3. Escreva uma mensagem e envie-a. Por predefinição, Enter envia; Shift+Enter insere uma quebra
   de linha. Estes atalhos podem ser alterados nas definições.
4. Adicione ficheiros com a ação de anexar. As imagens, o áudio e o vídeo exigem modalidades do modelo
   compatíveis. Os ficheiros PDF podem ser extraídos como texto ou convertidos em imagens;
   os modelos sem visão usam a via de texto.
5. Pare a geração, salte o raciocínio quando for suportado, edite uma mensagem, gere de novo uma
   resposta ou crie uma ramificação de uma conversa com as ações das mensagens.

As três linhas no topo da barra lateral expandem-na e recolhem-na. Esta mostra **Nova conversa**, **Monitor**,
**Modelo** e **Definições**, seguidos das conversas: **Fixadas**, **Conversas recentes** (as cinco com que trabalhou
por último; uma conversa à qual volta a enviar uma mensagem sobe para aí) e **Conversas** (as restantes, das mais
novas para as mais antigas). Cada conversa tem um balão de fala com um número de 01 a 99, atribuído quando é criada e
mantido durante toda a sua existência (um número libertado ao eliminar uma conversa passa para a próxima conversa
nova), para que se possa distingui-la enquanto a barra lateral está recolhida, quando só se veem os ícones do menu e
os balões. Pode afixar, mudar de nome, eliminar e selecionar em massa as conversas. O campo de pesquisa ao lado das
três linhas (Ctrl+K, ou ⌘K num Mac, expande a barra lateral e coloca o foco nele) encontra, sem distinguir maiúsculas de minúsculas nem acentos, as conversas pelo título e
pelo conteúdo das mensagens, as páginas e as respetivas partes, todas as definições, os modelos, as ferramentas e os
servidores MCP. Clique num resultado, ou prima Enter para escolher o primeiro, e abre-se a respetiva conversa ou
página, com o que foi encontrado realçado durante um momento.
As conversas são guardadas pelo servidor em `~/.mgs/chats/` do utilizador que executa o `mgs` (no serviço implementado,
`/opt/moe-gguf-server/.mgs/chats/`), um ficheiro por conversa com o nome da data em que foi criada, em UTC
(`2026-11-03@14-59-59.723.jsonl`), pelo que todos os navegadores que abrem a interface veem as mesmas conversas, e uma
página aberta mostra em poucos segundos as conversas que outro navegador cria, altera ou elimina. As definições são guardadas pelo servidor em
`~/.mgs/config/ui-settings.json` do utilizador que executa o `mgs` (no serviço implementado,
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), pelo que todos os navegadores que abrem a
interface as partilham: os parâmetros de geração, as opções de visualização, o tema, o idioma, os servidores
MCP, os modelos favoritos, as ferramentas desativadas e as sempre permitidas, o esforço de raciocínio
predefinido e a pasta de modelos de Definições → Modelos. Só a chave de API fica em cada navegador, porque autentica
o pedido que carrega o resto, pelo que tem de a introduzir uma vez em cada navegador. A importação e a exportação
estão disponíveis nas definições. As exportações podem conter anexos e, se forem selecionados explicitamente, chaves de API
sensíveis ou cabeçalhos de autorização MCP.

Escolha o idioma em Definições → Geral. As 15 opções estão ordenadas como `de-DE`,
`en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`,
`ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; a predefinição é `en-US`.
O português do Brasil e o português europeu têm traduções separadas. O hebraico usa uma
interface da direita para a esquerda; a caixa de entrada do chat e as mensagens simples do utilizador seguem a direção
do respetivo texto, enquanto o código e a saída do terminal são apresentados da esquerda para a direita.
Mudar de idioma guarda a escolha nas definições do servidor e
recarrega a interface; o servidor publica-a em `/api/config.js`, pelo que todos os navegadores
arrancam nesse idioma. Guarde primeiro as outras definições pendentes. O parâmetro de consulta `lang` do URL
sobrepõe-se a ela nessa página e mantém a escolha quando não é possível guardá-la.
Por exemplo, acrescente `?lang=ja-JP` ao URL da interface para selecionar japonês.

Definições → Geral → **Responder sempre no idioma da interface** está desativado por
predefinição. Ative-o e guarde as definições para dar instruções ao modelo para usar o
idioma e a variedade regional da interface em todas as respostas novas, incluindo em
conversas existentes, respostas geradas de novo, continuações, turnos após chamadas a
ferramentas e títulos gerados pelo modelo. Para `es-AR`, a instrução pede explicitamente
o voseo argentino: “Seguí estos pasos”, “tenés” e “podés”; `es-ES` pede
o uso do espanhol de Espanha. O inglês britânico e o americano e o português do Brasil e o
europeu também recebem instruções regionais distintas.

A instrução acompanha cada novo pedido de inferência e cada preenchimento prévio da cache KV;
é combinada com a mensagem de sistema enviada, sem editar o prompt de sistema
nem a conversa que guardou. Mudar o idioma da interface afeta os novos
pedidos após o recarregamento; uma resposta que já esteja a ser gerada mantém a
instrução que recebeu. Desativar a definição deixa de a acrescentar aos pedidos
seguintes. As respostas existentes não são traduzidas. O código, os comandos, os nomes, as citações
e as traduções pedidas explicitamente mantêm o idioma necessário; as explicações
que os rodeiam seguem o idioma da interface. A coerência depende da capacidade do modelo
para seguir a instrução; não se trata de uma tradução automática nem de uma garantia de cumprimento perfeito.
Os clientes externos da API não são afetados por esta definição do navegador.

Escolha Sistema, Claro ou Escuro para o tema. Tal como o idioma, o servidor publica
o tema guardado em `/api/config.js`, pelo que cada página já abre com ele antes de ser desenhada,
sem guardar nada no navegador; com Sistema, a interface acompanha de imediato o modo claro
ou escuro do sistema operativo, também enquanto está aberta. As definições de geração incluem a utilização
do contexto, a temperatura, os amostradores, as penalizações e os limites de turnos de agente. Algumas capacidades
dependem de opções do servidor ou do modelo selecionado.

Depois de cada resposta, a interface envia novamente a conversa, quando o servidor está
inativo, para que esta já esteja codificada na cache KV enquanto lê e o turno seguinte
comece mais cedo. Está ativado por predefinição; desative Definições → Desenvolvimento → **Preencher cache KV
após cada resposta** para o impedir.

Enquanto o servidor está a resolver outro pedido, uma mensagem que envie espera na
[fila](#request-queue), e a sua resposta mostra um aviso em vez do texto de progresso até
começar: “Em fila de espera: outro pedido enviado diretamente para a API está a ser resolvido
primeiro (pedidos à frente: N).” enquanto está a ser executado um pedido que um programa enviou diretamente
para a API, ou, caso contrário, “Em fila de espera: outro pedido da interface web está a ser resolvido
primeiro (pedidos à frente: N).” (outro separador ou navegador, um título gerado ou o preenchimento
prévio da cache KV). N conta os pedidos em execução e em espera à frente do seu. O aviso desaparece
assim que o seu pedido começa.

<a id="monitor"></a>
## Monitor e barra de estado

O **Monitor** fica por baixo de **Nova conversa** na barra lateral (endereço `#/monitor`). Enquanto está aberto, atualiza-se a cada
dois segundos e mostra:

- **Utilização**: primeiro, um cartão por cada GPU da máquina (NVIDIA, AMD ou Intel, mesmo que nenhum modelo a use), intitulado com a sua
  etiqueta a negrito e o seu nome (por exemplo **gpu-1**: NVIDIA GeForce RTX 4060 Ti; com várias GPU, gpu-1, gpu-2 e
  assim sucessivamente), com a sua utilização, a utilização do controlador de memória (largura de banda), a VRAM em uso, a temperatura e o consumo de energia face ao seu
  limite; depois, um cartão por cada processador, intitulado da mesma forma com o seu modelo (por exemplo **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; com vários processadores, cpu-1, cpu-2 e assim sucessivamente), com a sua utilização, uma pequena barra vertical por cada
  CPU lógica desse processador e a carga média do sistema. Todas as barras de utilização são verdes e enchem-se de vermelho à medida que a utilização
  aumenta; uma barra cujo valor é desconhecido fica cinzenta. As GPU NVIDIA são lidas através da biblioteca NVML do controlador e as GPU AMD
  através de `gpu_busy_percent` no sysfs; qualquer outra GPU (Intel, ou NVIDIA sem NVML) não mostra utilização, e o seu
  nome e a sua memória vêm do backend de inferência assim que um modelo a usa. Num ecrã largo, os cartões ficam dois
  por linha, cada um com metade da largura, como as barras de memória; com um número ímpar de cartões, a última linha
  tem três.
- **Memória**: uma barra por cada GPU (VRAM) e uma para a RAM; uma GPU sem valores de memória, como uma GPU Intel que nenhum
  modelo usa ainda, não tem barra. Cada barra está dividida por finalidade, cada finalidade sempre com a mesma cor, e a sua
  legenda lista todos os segmentos com o respetivo tamanho e a respetiva percentagem do total, um por linha e pela mesma ordem que na
  barra. Passe o cursor sobre um segmento para ver o seu tamanho.

  | Segmento | Conteúdo |
  | --- | --- |
  | Pesos MTP / do modelo de rascunho | Pesos das camadas MTP (nextn) do modelo, de um GGUF de MTP separado passado com `--spec-draft-model`, ou de um modelo de rascunho. |
  | Pesos dos especialistas | Tensores dos especialistas encaminhados do modelo. |
  | Pesos densos | Todos os outros pesos: atenção, especialistas partilhados, embeddings e saída. |
  | Cache KV | Memória de contexto do modelo (cache KV ou estado recorrente). |
  | Cache KV MTP / do modelo de rascunho | Memória de contexto do contexto MTP ou de rascunho. |
  | Buffers de cálculo | Buffers temporários dos grafos de cálculo. |
  | Cache dos especialistas e pré-carregamento | Cache de especialistas MoE (pool, identificadores e área de preparação) e as posições de pré-carregamento de especialistas. |
  | Cache de prompts e pontos de controlo | Prompts guardados com `--cache-ram` e os pontos de controlo de contexto dos slots (RAM). |
  | Outros | Memória usada que nenhuma das categorias anteriores explica: outros processos, o ambiente de trabalho, contextos do controlador, o projetor multimodal. |
  | Livre | O resto do dispositivo ou da RAM. |

  Os pesos mapeados a partir do ficheiro do modelo contam como RAM mesmo quando o kernel indica essa cache de páginas como disponível, pelo que os
  segmentos podem somar mais do que a memória «em uso»; nesse caso, a barra não mostra «Outros» e mostra menos memória livre. A
  cache de especialistas aparece após o primeiro pedido, quando é criada. No modo router, as barras incluem todos os modelos
  carregados, e o Monitor nunca carrega nem acorda um modelo.

A **barra de estado** no fundo de cada página mostra primeiro a utilização de cada GPU e depois a de cada processador
físico, cada uma na sua própria barra, com as mesmas etiquetas que os cartões do Monitor: `gpu-1`, `gpu-2`... e `cpu-1`,
`cpu-2`... As barras têm o mesmo tamanho, atualizam-se a cada dois segundos e, tal como as do Monitor, são verdes e enchem-se
de vermelho à medida que a utilização aumenta; o texto que aparece ao passar o cursor sobre cada uma indica o nome do dispositivo. Sem GPU, mostra uma barra
`GPU` vazia, e quando as barras não cabem num ecrã estreito, a barra de estado desloca-se lateralmente. Clique nela para abrir o
Monitor. Continua a funcionar enquanto o modelo está a carregar. É mostrada por predefinição; desative Definições → Visualização → **Mostrar barra
de estado** para a ocultar.

<a id="models"></a>
## Modelos, ferramentas e MCP

O `mgs` só carrega modelos MoE. Ao arrancar, lê o cabeçalho do GGUF indicado com `--model` (e o do indicado com `--spec-draft-model`); se for um modelo denso, como o GPT-2 ou o Phi-2, o `mgs` regista um erro como este e termina com o código de saída 1 antes de carregar o que quer que seja:

```text
srv    mgs_server: the model '/path/to/model.gguf' is not a Mixture of Experts (MoE) model
srv    mgs_server: mgs only loads MoE models; dense models are not supported
srv    mgs_server: exiting due to a dense model
```

No modo router, só termina o processo desse modelo: `/api/v1/models` marca-o como falhado e os outros modelos continuam a funcionar. As implementações de descodificadores densos não são compiladas, mas as camadas densas dentro de um modelo MoE e uma cabeça só de MTP compatível indicada explicitamente (`--spec-draft-model`) são mantidas.

<a id="settings-models"></a>
### Escolher e carregar modelos em Definições → Modelos

Iniciado sem `--model`, o `mgs` funciona no modo router: não carrega nenhum modelo por si só e lista os modelos que
encontra, para que os possa carregar a partir da interface. Definições → Modelos tem duas partes:

- **Pasta de modelos.** Prima **Escolher pasta**: abre-se um explorador das pastas da máquina onde o `mgs` está em
  execução, na pasta de modelos atual (ou na sua pasta pessoal). Clique numa pasta para a abrir, use a seta para subir e
  a casa para ir para a sua pasta pessoal, ou escreva um caminho no respetivo campo e prima Enter (um `~` no início é a
  sua pasta pessoal); **Mostrar pastas ocultas** acrescenta as pastas cujo nome começa por um ponto, e o explorador
  conta os ficheiros `.gguf` da pasta apresentada. Prima **Escolher** e o caminho absoluto dessa pasta preenche o campo
  e é utilizado de imediato. A pasta pode conter um ficheiro `.gguf` por modelo, ou uma subpasta por modelo com as
  partes de um modelo dividido (`NAME-00001-of-0000N.gguf`) e o respetivo projetor multimodal (`mmproj*.gguf`). O `mgs`
  verifica se consegue ler a pasta, guarda-a com as restantes definições e analisa-a de imediato; cada arranque seguinte
  analisa a mesma pasta. Para a voltar a ler depois de adicionar ou remover ficheiros, escolha novamente a mesma pasta.
  Os ficheiros GGUF densos, incluindo as cabeças MTP separadas, ficam fora da lista, porque o `mgs` só carrega modelos
  MoE. Escolher outra pasta descarrega os modelos em execução da pasta anterior. Os modelos transferidos para
  `~/.cache/mgs/` também são listados. Quando o `mgs` foi iniciado com `--models-dir`, essa pasta prevalece: o botão
  **Escolher pasta** fica desativado e a página indica-o.
- **Modelos disponíveis.** Cada modelo com o seu estado (não carregado, a carregar com a percentagem, carregado, em
  repouso ou falha ao carregar), o seu ficheiro e o seu tamanho em disco, e um botão para o carregar ou descarregar.
  Carregar um modelo aqui também o seleciona no chat.

Quais são os modelos carregados e como foram carregados vê-se na [página Modelo](#model-page), para a qual há uma
ligação no final de Definições → Modelos. O seletor de modelos do chat lista os mesmos modelos e carrega o que
escolher; enquanto a lista estiver vazia, tem uma ligação para Definições → Modelos. Iniciado com `--model`,
Definições → Modelos indica apenas que não é possível mudar o modelo sem reiniciar o `mgs`.

Só fica carregado um modelo de cada vez: carregar outro, aqui, a partir do seletor do chat ou através da API,
descarrega o que estava carregado (`--models-max N` permite N em simultâneo). Uma nova conversa usa o modelo carregado
nesse momento. Cada conversa mantém o seu próprio modelo, o da sua última resposta: quando lhe envia uma mensagem
enquanto está carregado outro modelo, esse outro é descarregado e o modelo da conversa é carregado primeiro. Se o
modelo da conversa já não estiver onde estava (saiu da lista, ou agora é outro ficheiro), uma caixa de diálogo
indica-o e permite cancelar a mensagem, que fica no campo, ou enviá-la com o modelo carregado.

A pasta é guardada em `~/.mgs/config/ui-settings.json` com as restantes definições da interface, como a secção
`modelsFolder`. O `mgs` escreve-a quando escolhe a pasta e lê-a em cada arranque:

```json
{
  "modelsFolder": "/home/user/models"
}
```

O serviço implementado (`deploy/`) inicia o `mgs` com `--models-dir /opt/moe-gguf-server/models`, pelo que é essa a
pasta que lista.

<a id="model-page"></a>
### Página Modelo

A página **Modelo**, na barra lateral a seguir ao **Monitor** (endereço `#/model`), mostra o modelo carregado em duas
caixas: **Ficheiro do modelo**, com o que o seu ficheiro GGUF contém (a descrição, a arquitetura, o ficheiro, o
tamanho em disco, os parâmetros, as camadas, as camadas MTP, o contexto de treino e o projetor multimodal), e
**Parâmetros de carregamento**, com a forma como foi carregado: os parâmetros que o `mgs` escolheu por si só ou que
indicou (tamanho do contexto, slots paralelos e contexto por slot, lote e micro-lote, threads de geração e de lotes, flash attention,
tipos da cache KV, `--fit`, MTP, descodificação especulativa com o número máximo de tokens de rascunho, cache de
especialistas MoE e dispositivos) e, no modo router, a linha de comandos do seu processo. Cada estado ativo (flash
attention, `--fit`, MTP, descodificação especulativa e a cache de especialistas MoE) aparece a verde. O flash
attention indica se foi o `auto` que o ativou; o MTP mostra quantos tokens de rascunho pede em cada passo, ou que
ainda os está a medir (0 a 3), ou que o rascunho não compensa neste computador; e a cache de especialistas MoE indica
se foi criada, com o seu tamanho automático ou pedido: é o primeiro pedido após carregar o modelo que o decide, pelo
que, até lá, é isso que indica, e a cache fica desativada numa GPU integrada ou quando todos os especialistas já
estão na VRAM. Estes estados em tempo real vêm do relatório do [Monitor](#monitor), que também tem a memória que
cada dispositivo usa, por finalidade. Enquanto nenhum modelo estiver carregado, a página tem uma ligação para
Definições → Modelos.

<a id="supported-models"></a>
### Famílias de modelos suportadas

| Família | Arquitetura GGUF | Notas |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP quando o GGUF o inclui. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP quando o GGUF o inclui. A sua tabela de embeddings por camada (cerca de 29 GB) é lida a partir do ficheiro do modelo conforme necessário, pelo que deve manter o ficheiro num disco rápido. Os contextos longos são calculados com atenção densa sob a seleção esparsa: o resultado é correto, mas sem a aceleração dos kernels esparsos. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; imagens com o seu mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Os GGUF publicados não incluem MTP. |
| DeepSeek-V4.1-Flash | `deepseek41` | Carrega as duas estruturas publicadas: os ficheiros DwarfStar do antirez (como `antirez/deepseek-v4.1-flash-gguf`) e os ficheiros do ecossistema llama.cpp. A sua tabela Engram (cerca de 188 GiB dentro do ficheiro) nunca é carregada: cada token lê as suas linhas a partir do ficheiro do modelo, pelo que o ficheiro tem de continuar disponível enquanto o `mgs` está em execução, de preferência num SSD, e é necessário o Linux ou outro sistema POSIX. Os ficheiros sem modelo de chat usam um integrado, e `reasoning_effort` aceita um número de 1 a 100. Sem MTP e sem imagens. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |
| Flint | `flint` | Os GGUF publicados não incluem MTP. |

O Gemma 4 e os modelos DeepSeek anteriores à V4.1 deixaram de ser suportados (consulte [compatibilidade removida](#removed-compatibility)). Também deixaram de ser suportadas as restantes arquiteturas MoE herdadas do llama.cpp, como gpt-oss, Qwen3-MoE, Qwen3-Next, GLM-4.x, Granite 4, MiniMax-M2 ou Kimi-Linear.

Com várias GPU selecionadas, o MTP verifica todas antes de recorrer à RAM. A sua estimativa de memória inclui a cabeça de saída. Os buffers MTP integrados são carregados antes dos outros pesos; um GGUF de MTP externo passado com `--spec-draft-model` é carregado antes do modelo principal, também com `--fit off`, e ativa o MTP por si próprio (consulte [MTP automático](#performance)); um MTP que esteja apenas noutro GGUF nunca é usado, a não ser que seja passado dessa forma. A seleção explícita de dispositivos em vigor continua a determinar que GPU estão disponíveis.

Use `--reasoning on` ou `--reasoning off` para controlar o raciocínio. Quando o raciocínio está desativado (com essa opção, com o interruptor de raciocínio do chat ou, num pedido, com `enable_thinking: false` em `chat_template_kwargs` ou `reasoning_effort: "none"`), o `mgs` também fecha qualquer bloco de raciocínio assim que o modelo o abre; por isso, os modelos cujo modelo de chat ignora essa definição, como o Flint, também respondem sem raciocinar. Passar `enable_thinking` através de `--chat-template-kwargs` é rejeitado. Use os nomes de opções documentados, com hífenes; as grafias com sublinhados são rejeitadas. Passe os valores de listas como um valor separado por vírgulas, em vez de repetir uma opção. Só `--spec-type` continua a poder ser repetida para compor modos especulativos.

O seletor de modelos mostra os modelos disponíveis, os carregados e os favoritos. Os detalhes do modelo
incluem o tamanho do contexto, as modalidades, a quantização e o modelo de chat. O modo router
gere processos filhos separados para os modelos carregados; os pedidos identificam o modelo
através do campo `model` ou do parâmetro de consulta.

Os servidores MCP são configurados em Definições → Servidores MCP, por URL, com cabeçalhos de autorização opcionais. Pode
inspecionar os registos de ligação e os prompts, os recursos e as ferramentas disponíveis. O proxy CORS opcional
do servidor tem de ser ativado explicitamente com a opção correspondente
do motor; só devolve conteúdo inerte e recusa ser aberto como página (consulte
[limites de segurança](#security-limits)). As restrições do navegador relativas a recursos HTTP dentro de páginas HTTPS continuam a aplicar-se.
Quando dois servidores MCP ativados oferecem uma ferramenta com o mesmo nome, nenhuma das cópias é oferecida ao modelo e a
interface mostra um aviso: desative todos esses servidores exceto um para usar a ferramenta.
As ferramentas integradas do servidor e a sandbox de JavaScript do navegador são opcionais.
As aprovações de ferramentas e o limite de turnos de agente continuam disponíveis na interface. O limite de turnos
(Definições → Agente → **Turnos do agente**, o número máximo de ciclos de execução de ferramentas
antes de parar, para evitar ciclos infinitos) é 50 por predefinição.

Os fluxos podem voltar a ligar-se com a identidade da conversa e o cursor guardados enquanto o
servidor mantém o fluxo. Um servidor parado ou um fluxo expirado não pode garantir
uma nova reprodução. A interface indica uma falha na nova ligação em vez de
criar silenciosamente um segundo pedido de geração.

### Servidor MCP da shell local

Ative **Servidor MCP local de shells** em Definições → Servidores MCP, ou inicie o `mgs` com
`--local-shell-mcp-server`, para permitir que os modelos do chat web usem o terminal da máquina onde o `mgs` está em
execução:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- O interruptor atua de imediato, sem reiniciar o `mgs`, e é guardado na secção `localShellMcpServer` de
  `~/.mgs/config/ui-settings.json`, pelo que os arranques seguintes voltam a ativá-lo
  (`POST /api/mcp/shell/enabled` com `{"enabled": true}` faz o mesmo). Com `--local-shell-mcp-server`, mantém-se
  ativo e o interruptor fica desativado. Sem chave de API, só um navegador no computador onde o `mgs` está em execução
  pode usar o interruptor, para que mais ninguém na rede possa ativar aí uma shell; com chave de API, qualquer pessoa
  que tenha a chave pode.

- No arranque, o `mgs` verifica que shells estão instaladas e oferece uma ferramenta para cada uma delas: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) e `psh.run` (PowerShell, `pwsh`). O registo lista as shells que encontrou; uma shell
  que não esteja instalada não recebe nenhuma ferramenta.
- A interface deteta-o, adiciona-o aos servidores MCP como **local shell mcp server** e ativa-o. Pode desativá-lo aí,
  ou numa única conversa, e a sua escolha é mantida. Quando o servidor é desativado, a entrada desaparece.
- Todas as ferramentas recebem `command` e, opcionalmente, `cwd` (diretório de trabalho; por predefinição, a pasta pessoal do utilizador
  que executa o `mgs`) e `timeout` (segundos; 60 por predefinição, 3600 no máximo). A resposta é um objeto JSON com
  `exit_code`, `stdout` e `stderr`, bem como `signal`, `timed_out`, `cancelled`, `timeout`, `seconds`, `shell` e
  `cwd`; as saídas longas mantêm apenas o início e o fim.
- Cada comando é executado num processo novo, sem terminal e com a entrada vazia: `cd` e as variáveis não passam
  para a chamada seguinte, e não é possível usar programas interativos (editores, perguntas). Quando o tempo limite expira ou o
  pedido é cancelado, todos os processos que o comando iniciou são terminados. Os processos deixados em segundo plano têm de
  redirecionar a sua saída.
- A interface pede autorização antes de cada chamada, a não ser que opte por permitir sempre a ferramenta.
- O `sudo` funciona sem terminal (Linux): quando precisa de uma palavra-passe, abre-se uma janela no centro da página,
  seja qual for a página em que esteja, com o pedido do sudo (`[sudo] password for user:`), a ferramenta e o comando que
  a pede, um campo de palavra-passe e, por baixo dele, os botões **Aceitar** e **Cancelar**. O cursor já está no
  campo da palavra-passe, pelo que pode escrevê-la de imediato; **Enter** aceita e **Escape** cancela; um clique
  fora da janela não a fecha. A palavra-passe só chega ao sudo na máquina onde o `mgs`
  está em execução: não é guardada na conversa, não é escrita nos registos e o modelo nunca a vê. Enquanto a
  janela espera, o tempo limite do comando deixa de contar; se carregar em **Cancelar**, ou se ninguém responder no prazo de 5
  minutos, o sudo falha. O sudo volta a pedi-la em cada chamada, porque cada uma é um processo novo, e até três vezes quando
  a palavra-passe está errada; a janela avisa então que a palavra-passe anterior pode estar errada. Os comandos são executados numa sessão
  própria, sem o terminal onde o `mgs` foi iniciado, pelo que o sudo nunca pede a palavra-passe nesse terminal.
  Só a interface web do `mgs` pode responder: com outros clientes MCP, o sudo falha como antes, por falta de terminal.
  O serviço implementado é executado com a conta `moe-gguf-server`, que normalmente não tem palavra-passe nem permissão
  de sudo, pelo que isto se destina ao `mgs` executado com o seu próprio utilizador (por exemplo, a compilação em `~/.local`). Quando o `mgs`
  não tem a variável `DISPLAY`, os comandos recebem `DISPLAY=:mgs-askpass`, porque o sudo só usa o seu auxiliar de
  palavra-passe sem terminal quando `DISPLAY` existe; os programas que tentem abrir uma janela falham de imediato.
- No chat, cada chamada é apresentada com o nome da sua ferramenta, como `bash.run`. Quando é expandida, mostra até três
  caixas cinzentas com letra monoespaçada: **stdin** com o comando simples, **stdout** com a saída e **stderr** com
  o texto do erro. O stdout só aparece quando o comando escreveu alguma coisa e o stderr só quando houve um erro;
  o código de saída fica junto ao respetivo título, por exemplo `stdout (código 2):` (junto ao stdin quando não há saída).
  A caixa stdin tem um contorno branco (cinzento-escuro no tema claro), a caixa stdout um verde e a caixa stderr
  um vermelho, em todas as ferramentas de shell (`bash.run`, `zsh.run`, `sh.run` e `psh.run`).
- Ative **Definições → Visualização → Expandir comandos da shell local automaticamente** para abrir cada chamada por si só: cada
  caixa aparece assim que o seu conteúdo é conhecido, o stdin enquanto o modelo escreve o comando e o stdout e o stderr quando
  o comando termina.
- Cada chamada a uma ferramenta de shell deixa o seu próprio ficheiro em `~/.mgs/local-shell-mcp-server/` (na pasta pessoal do utilizador que
  executa o `mgs`; no serviço implementado, `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), com o nome da data e da hora
  locais da chamada até ao milissegundo, por exemplo `y2026m10d12h23m59s59ms731.jsonl`; acrescenta-se `-2`, `-3`…
  quando várias chamadas começam no mesmo milissegundo. É um ficheiro JSONL, com um objeto JSON por linha: primeiro o comando (`"event": "command"`,
  com a ferramenta, a shell, o diretório de trabalho e o tempo limite), escrito antes de ser executado; depois toda a saída,
  sem cortes, à medida que chega, uma linha por fragmento (`"event": "stdout"` ou `"stderr"`, com o texto em `data`) pela
  ordem em que o comando a escreveu; e, por último, o resultado (`"event": "result"`: código de saída, sinal, se o tempo limite expirou
  ou se foi cancelado, os segundos que demorou e os bytes do stdout e do stderr) ou o motivo pelo qual não conseguiu arrancar
  (`"event": "error"`). Uma chamada rejeitada antes de ser executada (uma ferramenta desconhecida, sem comando, com um diretório de trabalho que não existe, com um
  tempo limite inválido) também deixa o seu ficheiro, com o comando tal como foi pedido e o erro. O chat mantém apenas o
  início e o fim das saídas muito longas, mas o ficheiro tem tudo, e já contém a saída se o `mgs`
  parar enquanto o comando está a ser executado. Para ler todo o stdout de uma chamada, execute `jq -j 'select(.event == "stdout") | .data'
  FILE`. O JSON só contém texto, pelo que os bytes que não são UTF-8 (saída binária) são substituídos por `�`; um comando que
  escreva gigabytes deixa um ficheiro desse tamanho. Só esse utilizador pode ler a pasta e os ficheiros, porque a saída
  de um comando pode conter segredos; nada os apaga, pelo que deve remover os antigos quando já não precisar deles.

No serviço implementado, adicione `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` à secção `[Service]` de um override
do systemd (`systemctl edit moe-gguf-server`) e reinicie `moe-gguf-server`; no Alpine, exporte a mesma variável na
configuração do serviço OpenRC.

**Segurança.** Os comandos são executados com as permissões do utilizador que executa o `mgs` (no serviço implementado, a
conta `moe-gguf-server`, nunca `root`). Qualquer pessoa que consiga abrir a interface ou chamar a API pode executá-los, pelo que deve ativar
a opção apenas em ambientes de confiança e proteger o servidor com uma chave de API. Os pedidos de páginas de outros sites
são rejeitados (a origem CORS predefinida é localhost) e o registo guarda cada comando com o respetivo código de saída. Leia
o comando apresentado na janela da palavra-passe antes de escrever uma palavra-passe do sudo: a palavra-passe autoriza esse comando. O
servidor só a entrega a um auxiliar de palavra-passe cujo processo pai seja o sudo a correr como root em nome do seu utilizador, pelo que um comando
não consegue iniciar o auxiliar por si só para ler a palavra-passe.

<a id="api"></a>
## API HTTP

Os exemplos usam 11443; substitua-a pela porta HTTPS de `ports.json` se for diferente.

A raiz predefinida da API é `/api`. A interface continua em `/`. As rotas `/v1/...` na raiz,
do servidor original, deixaram de estar registadas. Os clientes compatíveis com a OpenAI têm de
usar `https://ai.example.com:11443/api/v1` como URL base: as rotas da OpenAI (`/api/v1/models`,
`/api/v1/chat/completions`, `/api/v1/responses`, `/api/v1/audio/transcriptions` e as respetivas
variantes `input_tokens`) só existem em `/api/v1`, e a verificação do estado de saúde só em `/api/health`.
`/api/rerank` e `/api/v1/rerank` mantêm-se ambas para os clientes TEI/Jina, e as rotas nativas
(`/api/completions`, `/api/embeddings`…) são handlers diferentes dos seus homónimos em `/api/v1`.
Os nomes dos campos dos pedidos e das respostas
mantêm a compatibilidade com o upstream; as interfaces do motor em C/C++ mantêm as suas funções atuais,
mas não as que estavam marcadas como obsoletas ou não tinham utilidade (consulte [compatibilidade removida](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Defina `MGS_API_KEY` com a chave instalada na sua própria shell (só o `curl` a usa; o `mgs` lê a sua chave de
`--api-key`, `MGS_ARG_API_KEY` ou `--api-key-file`). As operações nativas,
os embeddings, a tokenização, a reclassificação (reranking), a gestão de modelos e os fluxos retomáveis estão
documentados no Swagger UI incluído em `/api/doc/`; o respetivo JSON OpenAPI está em
`/api/doc/openapi.json`, gerado pelo servidor em cada pedido, sem um ficheiro JSON estático. O inventário reflete as rotas registadas pelo
processo em execução, incluindo as rotas do router e as rotas de compatibilidade opcionais. As funcionalidades
desativadas podem devolver 403. Uma chave em falta ou inválida devolve 401 e o carregamento do modelo
devolve 503. `/api/health` e `/api/v1/models` mantêm o acesso público para `GET` e `HEAD`; consulte
[limites de segurança](#security-limits) para as rotas que exigem sempre a chave.

`--api-prefix /api/custom` muda as rotas de inferência para baixo de `/api/custom`.
A interface na raiz recebe esse prefixo de `/api/config.js`; a documentação continua em `/api/doc/`.
O espaço de nomes da documentação está reservado e não pode ser usado como prefixo
de inferência. Os processos filhos do router usam `/api` internamente. O servidor valida os prefixos;
os valores fora de `/api/` e os caminhos com barras finais são rejeitados.

O Monitor usa `GET /api/monitor` (relatório completo), `GET /api/monitor/memory` (instantâneos da memória dos modelos carregados)
e `GET /api/monitor/usage` (só a utilização da CPU e da GPU, também disponível enquanto o modelo está a carregar). Estas rotas leem contadores e
instantâneos publicados pelo ciclo de inferência, nunca a fila de tarefas, pelo que consultá-las periodicamente não acorda um modelo adormecido nem
reinicia o seu temporizador de inatividade. Cada modelo do relatório inclui `expert_cache` (com `active` e `pending`) e, com MTP,
`mtp`: `draft_n` (os tokens de rascunho pedidos em cada passo, 0 quando o rascunho não compensou, -1 enquanto estão a ser
medidos), `draft_n_max` e `measuring`. Os respetivos campos estão descritos no Swagger.

A interface lê as suas definições com `GET /api/ui/settings` (`{"settings": DOCUMENT}`, ou
`{"settings": null}` antes da primeira gravação) e guarda-as com `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, que substitui por completo `~/.mgs/config/ui-settings.json`, com modo 0600). Ambas exigem
a chave de API e também respondem enquanto o modelo está a carregar.

As conversas usam `GET /api/chats` (as conversas e a respetiva `version`), `GET /api/chats/version`,
`GET /api/chats/search?q=WORDS&limit=N` (a mensagem mais recente do utilizador ou do assistente de cada conversa que contém
todas as palavras, sem distinguir maiúsculas de minúsculas nem acentos) e `GET`, `POST` e `DELETE /api/chats/{id_chat}`.
Cada ficheiro começa com a conversa e continua com uma mensagem por linha; os ficheiros guardam as datas no formato nativo
`2026-11-03@14-59-59.723` (UTC), e a API em milissegundos desde a época Unix. Uma conversa guardada sem um número válido
recebe o número livre mais baixo. Exigem a chave de API e também respondem enquanto o modelo está a carregar.

No modo router, `GET /api/models/folder` devolve a pasta de modelos: `folder`, `source` (`cli` para `--models-dir`,
`settings` para a secção `modelsFolder` de `~/.mgs/config/ui-settings.json`, ou `none`), `exists`, `model_count`,
`settings_file` e `cache_dir`. `POST /api/models/folder` com `{"folder": "/absolute/path"}` (um `~` no início é a pasta
pessoal; `""` remove-a) verifica se a pasta pode ser lida, guarda-a nessa secção, volta a procurar os modelos e devolve
o mesmo objeto; responde 400 quando a pasta foi definida com `--models-dir` ou não pode ser lida.
`GET /api/models/folder/browse?path=/absolute/path` alimenta o explorador de pastas: devolve `path`, `parent` (vazio em
`/`), `home`, `folders` (as subpastas ordenadas por nome, cada uma com `name`, `path` e `hidden`) e `gguf_files`
(quantos ficheiros `.gguf` estão diretamente na pasta); sem `path`, lista a pasta de modelos ou, na falta desta, a pasta
pessoal, e responde 400 quando o caminho não é uma pasta que possa ser lida. As três exigem a chave de API. No modo
router, cada entrada de `GET /api/v1/models` inclui `path`, `file_size` (bytes, todas as partes de um modelo dividido) e
`mmproj`, e um modelo em execução acrescenta `meta` e `load`, a configuração com que foi carregado (`architecture`,
`description`, `n_layer`, `n_layer_mtp`, `n_ctx`, `n_ctx_slot`, `n_parallel`, `n_batch`, `n_ubatch`, `n_threads`,
`n_threads_batch`, `flash_attn`, `flash_attn_enabled` (se o contexto o usa, também com `auto`), `cache_type_k`,
`cache_type_v`, `fit`, `mtp`, `speculative`, `draft_n_max`, `moe_cache_mib`, `devices` e `mmproj`). Com `--model`, a
única entrada também inclui `load`.

<a id="request-queue"></a>
### Fila de pedidos e pedidos em paralelo

Por predefinição, o `mgs` resolve um pedido de cada vez (`-np 1`) e atribui-lhe o contexto inteiro, pelo que cada pedido pode
usar todos os tokens de contexto, quer venha da interface web, quer venha diretamente da API. Os pedidos que
chegam entretanto esperam numa fila única e são resolvidos pela ordem em que chegaram: os pedidos da interface não
passam à frente dos pedidos da API, nem o contrário, e um pedido que tenha de voltar a esperar mantém o seu lugar.
A API comporta-se da mesma forma, sejam quais forem as definições da interface.

Para resolver vários pedidos ao mesmo tempo, inicie o `mgs` com `-np N` (`--parallel N`, `MGS_ARG_PARALLEL`):
até N pedidos são executados em paralelo e o contexto é repartido entre eles, pelo que cada um recebe uma parte (`n_ctx_slot` na
linha `initializing` do registo); com `--kv-unified`, partilham em vez disso uma única cache KV do contexto
inteiro. Os pedidos para além de N continuam a esperar na fila.

A interface web acrescenta `"mgs_client": "webui"` aos seus pedidos. Enquanto um dos seus pedidos em streaming espera, o
fluxo começa com eventos como

```text
data: {"mgs_queue":{"ahead":2,"running_api":1,"running_webui":0,"waiting_api":0,"waiting_webui":1}}
```

(`ahead`: pedidos em execução ou em espera à sua frente; `running_api` e `running_webui`: quantos deles estão a ser
executados agora e chegaram diretamente à API ou vieram da interface web; `waiting_api` e `waiting_webui`: o mesmo para
os pedidos que esperam à sua frente na fila), e um último com todas as contagens a 0 quando começa. Os clientes que não
enviam `mgs_client` nunca recebem estes eventos, pelo que os clientes compatíveis com a OpenAI recebem o fluxo habitual. O Swagger
descreve o campo juntamente com os outros parâmetros de conclusão.

<a id="security-limits"></a>
### Limites de segurança

Estes limites protegem o servidor e a interface contra pedidos, ficheiros e modelos hostis. São suficientemente generosos
para uma utilização normal: o que os ultrapassar falha com um erro claro, em vez de parar o servidor.

- **Chave de API.** As rotas de estado de saúde e de lista de modelos (`/api/health` e `/api/v1/models`) só são públicas
  para `GET` e `HEAD`. Recarregar a lista de modelos (`GET /api/v1/models?reload=1`) e adicionar ou eliminar modelos no
  modo router (`POST` e `DELETE /api/models`) exige a chave sempre que houver uma configurada.
- **Páginas de outros sites.** Por predefinição (`--cors-origins` não definido, o que equivale a `localhost`), só as páginas de
  `localhost` e deste mesmo servidor podem chamar a API: o servidor responde 403 a qualquer pedido que não seja `GET`,
  `HEAD` ou `OPTIONS` cujo cabeçalho `Origin` indique outro site, e o navegador não deixa que outros sites leiam
  as suas respostas. A interface (também por trás do Apache) e os clientes que não enviam `Origin`, como o curl ou os SDK,
  não são afetados. Para permitir que as páginas de outros sites chamem a API, indique as respetivas origens em `--cors-origins`
  (`*` admite qualquer site: nesse caso, configure uma chave de API). `POST /api/tools` só aceita
  `Content-Type: application/json` (caso contrário, 415).
- **DNS rebinding.** Com HTTP simples (`--http`) numa porta TCP e sem chave de API, o servidor responde 403
  (`Host not allowed`) a todos os pedidos cujo cabeçalho `Host` não seja um endereço IP, `localhost` (ou um nome terminado
  em `.localhost`), o nome da máquina (também com `.local`) ou o nome indicado com `--host`, pelo que uma página de
  outro site que aponte o seu próprio nome para esta máquina não consegue usar o servidor. O HTTPS, uma chave de API, o socket
  Unix da implementação e o Vertex AI (`AIP_MODE=PREDICTION`) não são afetados. Para aceder a um servidor HTTP simples
  através de outro nome, configure uma chave de API.
- **Proxy CORS.** `/api/cors-proxy` recusa os pedidos que um navegador abriria como página, frame ou worker
  (403) e devolve todas as respostas como conteúdo inerte: não pode executar scripts nem definir cookies no endereço do servidor.
- **Listagens do router.** No modo router, `status.args` e `status.preset` de `/api/v1/models` mostram `<redacted>`
  em vez dos valores de `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  e `--mcp-servers-json`; o registo faz o mesmo. Os processos dos modelos continuam a receber os valores reais.
- **Vertex AI.** Com `AIP_MODE=PREDICTION`, o `@requestFormat` de cada instância só chega às rotas de inferência
  (completions, chat completions, responses, messages, infill, embeddings, rerank, tokenize, detokenize,
  apply-template e contagem de tokens); qualquer outra rota, incluindo a própria rota de previsão, responde com um erro
  para essa instância.
- **Transferências de modelos.** O token do Hugging Face (`--hf-token`, `HF_TOKEN`) só é enviado para o endpoint do
  Hugging Face (`MGS_MODEL_ENDPOINT`, `https://huggingface.co/` por predefinição); um modelo, projetor ou modelo de rascunho
  transferido de qualquer outro URL é pedido sem ele.
- **Clientes lentos.** Enquanto o servidor recebe um pedido, cada leitura espera no máximo 30 s (`--timeout` continua a controlar
  as escritas, pelo que as respostas longas não são cortadas), e no máximo 256 ligações aceites esperam por uma thread livre.
- **Corpos dos pedidos.** Os corpos JSON com um aninhamento superior a 1024 arrays ou objetos são rejeitados com 400. Os esquemas JSON
  (`response_format`, `json_schema`) podem ter até 256 níveis de aninhamento; os valores `pattern` vazios e os quantificadores sem
  nada para repetir são rejeitados.
- **Multimédia remota.** Os URL em `image_url`, `input_audio` e `input_video` só podem apontar para endereços públicos da
  Internet. O loopback, as redes privadas (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10), os endereços link-local
  (incluindo o endereço de metadados da cloud 169.254.169.254), os endereços IPv6 locais únicos e outros
  endereços não públicos são rejeitados com 400, também quando um redirecionamento leva até eles. Para os permitir, por exemplo para
  ler imagens de um servidor da sua rede, inicie o `mgs` com `--media-allow-private-urls`
  (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); faça-o apenas quando todos os clientes da API forem de confiança. Cada transferência tem 10 s para
  estabelecer a ligação, 30 s no total, até 5 redirecionamentos e 10 MiB, e o servidor executa no máximo 8 transferências em simultâneo.
  Todos os conteúdos multimédia de um pedido podem ter, em conjunto, no máximo 256 elementos e 512 MiB, e as suas transferências
  remotas têm 120 s no total; os ficheiros lidos com `--media-path` contam para os mesmos limites e são verificados antes de serem lidos.
- **Multimédia descodificada.** Imagens até 32768 píxeis por lado e 100 megapíxeis, áudio até uma hora, fotogramas de vídeo até
  8192 píxeis por lado e 2 GiB de fotogramas descodificados por vídeo, e no máximo 2 GiB de imagens, áudio e fotogramas de vídeo
  descodificados por pedido (incluindo os fotogramas de todos os seus vídeos).
- **Projetores.** Um projetor multimodal cujos metadados não correspondam ao seu grafo (por exemplo, um fator de fusão de patches
  inferior a 2 num projetor que funde patches) falha ao carregar com um erro, em vez de parar o servidor.
- **Ferramentas do servidor.** `read_file` com um intervalo de linhas lê o ficheiro por blocos e mantém apenas essas linhas,
  percorrendo no máximo 256 MiB do ficheiro; sem `end_line`, os ficheiros com mais de 16 KB são recusados.
- **Modelos de chat do modelo.** `range()` produz no máximo um milhão de elementos, as cadeias construídas ficam abaixo de
  256 MiB, a indentação de `indent()` e `tojson()` abaixo de 256 espaços e o aninhamento abaixo de 1024 instruções; um modelo de chat
  que os ultrapasse falha com um erro de modelo de chat.
- **Interface.** As ligações nas mensagens mantêm apenas os esquemas http, https, mailto e tel (ou endereços relativos), e
  os anexos de imagem têm de ser imagens incorporadas. A importação de conversas aceita ficheiros até 512 MB, arquivos ZIP com até
  10 000 entradas, 256 MB por ficheiro JSONL e 1 GB extraído no total. Um ficheiro com registos malformados, ou com uma
  mensagem que pertença a outra conversa, é rejeitado por inteiro; uma conversa cujas mensagens já existam,
  ou cujas mensagens se liguem entre si em ciclo, é ignorada e comunicada. As imagens SVG escritas pelo modelo ficam
  dentro da sua própria caixa e não podem tapar o resto da página. A listagem dos recursos ou dos templates de recursos de um
  servidor MCP termina ao fim de 100 páginas, 10 000 elementos ou 60 s, ou quando se repete um cursor. Um recurso MCP é sempre lido,
  guardado em cache e anexado a partir do servidor em que foi escolhido, mesmo quando outro servidor oferece o mesmo URI.
  Uma aprovação **Permitir sempre** de uma ferramenta MCP aplica-se enquanto o respetivo servidor mantiver o mesmo endereço,
  transporte, uso de proxy e cabeçalhos: se algum deles mudar, a interface volta a perguntar.

Os comandos de instalação continuam a transferir o ramo `main` atual por HTTPS; não verificam uma assinatura nem
uma revisão fixa.

<a id="performance"></a>
## Memória e desempenho

O registo da memória do anfitrião e o pré-carregamento de especialistas estão ativados por predefinição. O pré-carregamento
sobrepõe as transferências de especialistas ao cálculo, sobretudo em prompts grandes.
Os resultados dependem do modelo, da capacidade de memória e da largura de banda da RAM e do PCIe.

Nos modelos com especialistas descarregados para a CPU, `--cpu-moe` ou `--n-cpu-moe N` podem reduzir
a utilização de VRAM. O registo do anfitrião fixa as páginas de pesos mapeadas para acelerar as cópias do anfitrião
para o dispositivo. A memória fixada não pode ser paginada para o disco, pelo que deve deixar RAM suficiente para o
sistema operativo. Esta via de registo é específica de POSIX; o Windows mantém o seu comportamento
de carregamento anterior. `--load-mode none` só é recomendado quando o registo do anfitrião falha
total ou parcialmente, não como requisito normal.

O `mgs` tem em conta o limite de memória do seu cgroup (um serviço do systemd com `MemoryMax`, um
contentor) além da RAM livre. O ajuste automático só mantém os especialistas MoE na RAM quando cabem
nesse limite e, caso contrário, coloca camadas inteiras de especialistas na GPU. Quando os
especialistas ficam na RAM, a cache de prompts (`--cache-ram`, 8 GiB por predefinição) ocupa no
máximo a memória que sobra depois deles e da margem, o que o registo indica com `prompt cache limited to N MiB`;
`--cache-ram` define-a explicitamente. Só as páginas do modelo que ficam na RAM são fixadas, e só tantas quantas deixem uma margem (4 GiB ou 10% da RAM) da memória ainda disponível; para lá disso, o registo mostra `pinning X of Y GiB of mapped weights` e o resto usa memória paginável. Num cgroup de 30 GB com o Qwen3.6-35B-A3B Q8_0 de 37,6
GB, a versão 0.1.17 ficava bloqueada durante o carregamento até o sistema a terminar; agora arranca
em poucos minutos e gera. Até à versão 0.1.22, com camadas inteiras de especialistas na GPU, o `mgs` também fixava os especialistas já copiados para a GPU, pelo que, num computador de secretária com pouca RAM livre e sem swap, todo o sistema podia deixar de responder enquanto o modelo carregava.

Quando os ficheiros do modelo são maiores do que a memória ainda disponível (ou do que o limite do cgroup em que corre), o `mgs` não os lê por inteiro ao carregar, o que apenas despejaria as suas primeiras páginas e a cache de páginas de todos os outros programas: lê o que vai para a GPU e o que fixa, e o resto na primeira utilização. Com Qwen3.8-Flash-Next (113 GB) em 40 GiB de memória e um disco SATA, o carregamento passou de mais de 5 minutos para cerca de um minuto e meio.

O pré-carregamento de especialistas usa por predefinição três posições na VRAM. Cada posição tem de comportar o maior
tensor de especialista transferido. Os valores 2–8 em `GGML_SCHED_PREFETCH_EXPERTS` escolhem
explicitamente o número de posições. Se a reserva de memória falhar, são mantidas pelo menos duas posições reservadas
quando possível; caso contrário, são usadas as transferências normais. O pré-carregamento só começa
quando o lote tem pelo menos o dobro de seleções de especialistas do que especialistas. O seu pool
pertence à primeira GPU que o ativa; as outras GPU mantêm cópias seletivas.
Os lotes maiores podem ajudar no preenchimento prévio, mas exigem mais memória. Numa [GPU integrada](#integrated-gpu) não há pré-carregamento: preencher um slot seria uma cópia de RAM para RAM feita pela CPU, e os slots tirariam RAM à cache de páginas dos especialistas, pelo que os especialistas usados são copiados quando é preciso (com o Qwen3.8-Flash-Next numa Radeon 860M, o prompt passou de 15 para 33 tokens/s e a geração de 4,3 para 5,0 tokens/s).

Cada posição ocupa o tamanho que a GPU reserva para o maior tensor de especialista, incluindo o preenchimento que o CUDA acrescenta às linhas cujo comprimento não é múltiplo de 512 elementos, e deixa `--moe-cache-reserve` (1024 MiB por predefinição) livre para a memória que o cálculo ocupa durante a execução; se uma posição não couber, o pré-carregamento continua com as que cabem, se forem pelo menos duas, ou copia em vez disso os especialistas em uso. Até à versão 0.1.31 as posições deixavam esse preenchimento de fora, pelo que o Qwen3.8-Flash-Next, cujos especialistas têm linhas de 640 elementos, podia parar no primeiro prompt longo com `CUDA error: an illegal memory access was encountered` (ou `an internal operation failed`) sempre que a GPU tinha espaço para duas posições, por exemplo com MTP numa RTX 4060 Ti. Com o Qwen3.8-Flash-Next na RTX 4060 Ti, o MTP de 2 tokens processa agora o prompt a 12 tokens/s e gera a 1,31 tokens/s.

Para diagnóstico, defina `GGML_CUDA_REGISTER_HOST=0`, `GGML_SCHED_PREFETCH_EXPERTS=0` ou
`GGML_VK_STAGING_RING=0` (carregamentos com Vulkan através do anel intermédio) para desativar a otimização correspondente.
Não defina essas variáveis durante o funcionamento normal, a não ser que pretenda prescindir delas intencionalmente.

O MTP automático examina os metadados do GGUF e os nomes dos tensores antes da reserva dos pesos,
incluindo todos os fragmentos. Para o MTP integrado em `--model`, exige uma arquitetura
suportada, um tronco do modelo e todas as cabeças integradas. Um GGUF de MTP separado passado com
`--spec-draft-model` também o ativa quando declara cabeças MTP completas, sem tronco; um modelo de
rascunho sem cabeças MTP mantém o tipo especulativo que escolher. Um MTP que esteja apenas noutro
GGUF nunca é procurado, e uma arquitetura não suportada não o ativa. O limite predefinido de rascunho
é de dois tokens: com MTP, dois geraram tão depressa como três numa RTX 4060 Ti e numa
Radeon 860M integrada, e um rascunho rejeitado desperdiça menos trabalho de verificação. Qualquer seleção explícita com `--spec-type` tem prioridade e desativa a seleção
automática; `--spec-type none` desativa o MTP. `--spec-draft-n-max` altera o limite. O MTP afeta a geração,
não o preenchimento prévio nem, diretamente, o tempo até ao primeiro token. A melhoria depende da taxa de aceitação
e do hardware; aumentar o limite não garante mais velocidade.

Se o MTP acelera a geração depende da máquina: cada passo confere os tokens do rascunho com o modelo inteiro, o que custa pouco quando os especialistas estão na GPU e muito quando a maioria vem da RAM ou do disco. Por isso, quando o MTP é ativado automaticamente, o `mgs` mede-o durante as primeiras respostas: após um breve aquecimento, alterna passo a passo 0 (sem rascunho), 1, 2 e 3 tokens de rascunho até cada um ter gerado 64 tokens, cerca de 500 tokens no total, fica com o mais rápido se o rascunho for mais de 5% mais rápido do que sem rascunho, e indica-o no registo com uma linha como `MTP drafts 2 tokens from now on, the fastest on this machine`. O resultado é guardado em `~/.mgs/tuning/mtp.json` por modelo, máquina e versão do `mgs`, e os arranques seguintes usam-no diretamente; quando o rascunho não compensou, o registo indica `MTP drafting is not worth it on this machine` e, a partir do arranque seguinte, o MTP não é carregado, o que também deixa a sua VRAM aos especialistas, ao contexto e ao micro-lote. Apagar o ficheiro (ou a sua entrada) faz medir de novo. `--spec-draft-n-max N` fixa o número de tokens de rascunho e desliga a medição, `--spec-type draft-mtp` mantém o MTP ativado e um GGUF de MTP separado passado com `--spec-draft-model` é sempre usado. Na Radeon 860M, o Qwen3.8-Flash-Next gerou 5,12 / 7,09 / 7,34 / 6,97 tokens/s com 0 / 1 / 2 / 3 tokens de rascunho e fica com 2; na RTX 4060 Ti gerou 1,81 / 1,79 / 1,30 / 1,59, pelo que a partir do arranque seguinte funciona sem MTP, com 23-24 tokens/s de prompt e 1,69-1,70 de geração em vez de 12 e 1,4, enquanto o Qwen3.6-35B-A3B gerou aí 26,38 / 33,92 / 36,21 / 35,41 e fica com 2.

<a id="mtp-vram"></a>
Os pesos MTP ficam sempre na memória da GPU quando esta os consegue comportar: as camadas MTP injetadas no modelo, um
GGUF de MTP separado passado com `--spec-draft-model` e a cabeça de saída que usam. As opções que movem pesos para a
memória do sistema (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, os seus equivalentes `--spec-draft-*` e o ajuste
automático) não se aplicam a eles, e ficam na GPU mesmo com um `--gpu-layers` baixo. O ajuste automático coloca o
resto do modelo à sua volta, e a [cache de especialistas MoE](#expert-cache) usa a memória que sobra depois disso. Quando uma
camada MTP não cabe na memória livre de nenhuma GPU selecionada, o registo mostra um aviso (`no selected GPU has enough free
memory for MTP layer 0; using system memory`) e só essa camada fica na memória do sistema; um tensor cujo tipo a GPU não
consegue guardar também lá fica. Numa [GPU integrada](#integrated-gpu), a memória livre contada para o MTP é, no máximo, o que a
RAM do sistema ainda consegue dar. Com um GGUF de MTP separado, as camadas MTP injetadas no modelo principal não são carregadas,
para que não ocupem memória duas vezes; com `--spec-type none`, não é carregado nenhum peso MTP. O registo de carregamento
confirma a colocação com linhas como `MTP layer 0 assigned to CUDA0 (856.36 MiB)`, visíveis com `-lv 4`.

<a id="expert-cache"></a>
### Cache de especialistas MoE

Quando os especialistas MoE ficam na memória do sistema e o resto da camada é executado numa GPU, o servidor mantém os especialistas
usados mais recentemente de todas as camadas numa cache que ocupa a VRAM livre;
com várias GPU, cada GPU guarda em cache as camadas que lhe foram atribuídas na sua própria VRAM. Em cada token gerado (e em qualquer
lote com menos de 32 tokens), os especialistas encontrados na cache são executados na GPU, uma parte dos que faltam é carregada
para a cache e o resto é executado na CPU ao mesmo tempo; essa parte acompanha as larguras de banda medidas do PCIe e da
CPU. Os prompts longos mantêm o pré-carregamento de especialistas, que agora copia os especialistas em cache dentro da GPU e só
carrega os outros. A saída coincide com a da execução sem cache, salvo arredondamentos. A GPU e a CPU arredondam de forma diferente e os especialistas
que cada uma calcula dependem do conteúdo da cache, pelo que uma geração gananciosa longa pode derivar, ao fim de alguns tokens, para um texto diferente mas igualmente
válido, de uma execução para a seguinte; a cache do llama.cpp oficial comporta-se da mesma forma.

A cache está ativada por predefinição e é criada pelo primeiro pedido, depois de o modelo, o contexto MTP e o
projetor multimodal estarem carregados. Sem `--gpu-layers`, `--n-cpu-moe` ou `--override-tensor`, o ajuste
automático mantém todos os especialistas na memória do sistema quando cabem com uma margem (4 GiB ou 10% da RAM) e deixa a VRAM livre para a cache (com várias GPU, reparte as camadas entre
elas segundo a memória livre de cada uma, para que todas as GPU tenham cache); caso contrário, coloca camadas de especialistas inteiras na GPU, como antes, e a cache usa o que
sobra. As opções de colocação explícitas são respeitadas: `--cpu-moe` dá à cache o máximo de VRAM e
`--n-cpu-moe N` mantém algumas camadas estáticas na GPU. Os especialistas mantidos na RAM são fixados enquanto a memória ainda disponível mantiver essa margem; o resto continua paginável, pelo que o sistema continua a responder mesmo com pouca RAM.

Cada camada mantém a sua própria parte da cache, igual para todas, pelo que uma rajada de falhas em algumas camadas nunca esvazia a
cache das outras, e dentro de cada parte os especialistas usados de novo ficam protegidos dos que só são usados uma vez: um
especialista enviado para a cache entra numa zona de experiência e passa para uma zona protegida (no máximo metade da parte) quando
volta a ser usado, e os novos especialistas substituem primeiro os da zona de experiência. Quando só uma parte dos especialistas em
falta pode ser enviada, os usados mais recente e frequentemente vão primeiro. Com o Qwen3.6-35B-A3B Q8_0 numa RTX 4060 Ti, a geração
passou de 37,4 para 38,1 tokens/s (de 45,2 para 47,1 com MTP) e os acertos da cache de 72,3 para 74,1%. A divisão também mantém a
cache útil quando um passo traz mais especialistas do que a cache inteira comporta, como com MTP em GPU com menos VRAM, onde uma única
lista partilhada por todas as camadas não guardava quase nada.

<a id="integrated-gpu"></a>
Numa GPU integrada (os gráficos de um processador AMD ou Intel através do Vulkan, ou NVIDIA Jetson e DGX Spark através
do CUDA), a memória que a GPU indica faz parte da RAM do sistema. O ajuste automático conta ambas como um único
orçamento: a GPU pode usar no máximo o que a RAM ainda consegue dar depois dos buffers que ficam na CPU. Se o modelo
inteiro couber aí (reduzindo o contexto se for preciso, como numa GPU dedicada), tudo é executado na GPU; caso
contrário, camadas de especialistas inteiras vão para a GPU até esse limite e a CPU calcula o resto. Não há cache de
especialistas numa GPU integrada, nem mesmo com `--cpu-moe` ou `--n-cpu-moe`, porque copiaria especialistas da RAM
para a mesma RAM, ficando com eles em duplicado. O registo mostra o orçamento numa linha como
`Vulkan0 is an integrated GPU that shares the system memory: N MiB usable` e a decisão em
`no MoE expert cache on an integrated GPU`. Com o modelo mapeado (`--load-mode mmap`, a predefinição), os pesos que ficam na CPU quando todas as camadas estão na GPU (as tabelas de embeddings dos tokens e por camada, lidas linha a linha) não contam nesse orçamento, porque só as linhas lidas ficam em memória: o Qwen3.8-Flash-Next, cujos embeddings por camada ocupam 28 GB, recebe toda a memória que a GPU indica em vez do que sobra depois deles. Numa GPU assim, o `mgs` também volta a executar-se uma vez no arranque com `OMP_WAIT_POLICY=passive`, a não ser que a variável já esteja definida, para que as threads da CPU durmam em vez de girar enquanto a GPU trabalha: partilham com ela a potência do processador e, com as threads a girar, a GPU de uma Radeon 860M funcionava a cerca de metade da sua frequência. O registo mostra-o na linha `the OpenMP threads of the CPU wait passively`; `OMP_WAIT_POLICY=active` mantém as threads a girar.

<a id="context-size"></a>
Quando o modelo não cabe na VRAM e `-c` não é indicado, o ajuste automático também escolhe o tamanho do contexto: a
cache KV ocupa no máximo um quarto da VRAM que sobra depois dos pesos densos, até ao contexto de treino do
modelo, e os especialistas (a cache ou as camadas de especialistas inteiras) ficam com o resto. Anteriormente, estes modelos recebiam sempre o
mínimo de 4096 tokens partilhados por todos os slots, o que cortava as respostas longas (`truncated = 1` no registo). O
registo mostra o resultado numa linha como `context size raised from 4096 to N tokens` e em `n_ctx_slot` da
linha `initializing`. `-c N` define o contexto explicitamente (mais contexto deixa menos VRAM para a cache e reduz a
velocidade de geração: 35,8 tokens/s com 4096 tokens e 33,9 com 122624 na máquina de referência) e
`--fit-ctx N` aumenta o mínimo. O contexto MTP adota o mesmo tamanho que o contexto do modelo, em vez de reservar
VRAM para todo o contexto de treino. Quando camadas inteiras de especialistas preenchem a GPU porque os especialistas não ficam na RAM para a cache (sempre numa [GPU integrada](#integrated-gpu)), a cache KV ocupa no máximo um dezasseis avos, porque cada camada de especialistas que fica fora da GPU é lida a cada token da RAM, ou do disco quando o modelo é maior do que a RAM: o Qwen3.8-Flash-Next numa Radeon 860M com 46 GB de RAM recebe 37376 tokens e gera 5,3 tokens/s, onde com um quarto recebia 137216 tokens e gerava 4,9 tokens/s.

O ajuste automático conta a memória do contexto MTP para o tamanho de contexto que escolhe, e não para o contexto de treino do modelo: até à versão 0.1.32 contava-lhe 1300 MiB de VRAM com o Qwen3.8-Flash-Next numa RTX 4060 Ti, onde ocupa cerca de 160 MiB, e escolhia um contexto mais curto para o modelo. Com o Qwen3.8-Flash-Next e MTP de 2 tokens na RTX 4060 Ti, o ajuste conta agora 130 MiB mais 4,57 KiB por token de contexto para o contexto MTP e dá ao modelo um contexto de 7936 tokens em vez de 6912.

Antes do contexto, o ajuste automático escolhe o microlote (`-ub`), o número de tokens do prompt processados numa
só passagem. Com os especialistas na RAM, cada passagem carrega para a GPU os especialistas que não estão na cache, pelo que passagens
menos numerosas e maiores processam os prompts longos muito mais depressa, sem que a geração se altere. Salvo se `-b` ou `-ub` for indicado,
escolhe o maior de 4096, 2048 e 1024 tokens cujos buffers adicionais caibam num oitavo da VRAM que sobra depois dos
pesos densos, e aumenta o lote (`-b`) em conformidade. Na máquina de referência, um prompt de 2514 tokens passou de 214 para
755 tokens/s com 4096, à mesma velocidade de geração; os buffers maiores encurtam o contexto predefinido (cerca de 90000
tokens sem MTP e 61000 com MTP em 16 GB). O registo mostra a escolha numa linha como `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

Quando camadas de especialistas inteiras preenchem a GPU, deixam abaixo da margem menos memória do que a necessária para o tensor de especialistas seguinte, e nada mais a usa; o ajuste automático aumenta então o micro-lote com essa memória, pelo que nenhum especialista sai da GPU. O registo mostra-o numa linha como `micro-batch size raised from 2048 to 4096 tokens (batch 4096) with the device memory that the whole expert layers leave unused`. Se basta depende da VRAM livre: com o Qwen3.8-Flash-Next e MTP numa RTX 4060 Ti não bastava com a margem predefinida, e bastava com `--fit-target 800`. Nessa execução, o prompt foi processado a 21-22 tokens/s em vez de 12, com a geração a 1,41-1,46 tokens/s (1,39 antes).

Com uma colocação explícita (`--gpu-layers`, `--tensor-split`, `--override-tensor`, `--cpu-moe` ou `--n-cpu-moe`),
o ajuste automático mantém-na tal como foi indicada e apenas ajusta à sua volta o contexto e o microlote que não
definiu, em vez de parar; o registo indica `the device placement was set explicitly`.

| Opção | Variável | Efeito |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_ARG_MOE_CACHE` (`0` ou `false` desativa) | Ativa ou desativa a cache (ativada por predefinição). |
| `--moe-cache-size MiB` | `MGS_ARG_MOE_CACHE_SIZE` | Tamanho total fixo da cache, repartido entre as GPU segundo os especialistas que cada uma guarda em cache; `-1` (predefinição) usa a VRAM livre de todas as GPU. |
| `--moe-cache-reserve MiB` | `MGS_ARG_MOE_CACHE_RESERVE` | VRAM deixada livre em cada GPU para além da cache e das posições de pré-carregamento (predefinição: 1024). Aumente-a quando outros programas usarem a GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnóstico: parte fixa (0–1) dos especialistas em falta que é carregada, em vez do equilíbrio medido. |
| — | `MGS_MOE_CACHE_STATS` | Diagnóstico: regista os contadores da cache a cada N passos de descodificação (visíveis com `-lv 4`). |
| — | `MGS_MOE_CACHE_PROTECTED` | Diagnóstico: fração (0–1) da parte de cada camada reservada aos especialistas usados de novo (predefinição: 0,5); `0` dá uma ordem LRU simples por camada. |

`GET /api/expert-cache` devolve se a cache existe (`active`) ou se o pedido seguinte ainda tem de a criar ou de
aplicar um novo tamanho (`pending`), o tamanho, as camadas e as posições em cache, os acertos e as falhas, os carregamentos, os especialistas na CPU, a
percentagem dos bytes de pré-carregamento servidos a partir da cache e as larguras de banda medidas (com várias GPU,
os tamanhos e os contadores somam-se e a largura de banda de carregamento é a sua média). `POST /api/expert-cache` com
`{"size_mib": N}` altera o tamanho sem reiniciar (`-1` automático, `0` liberta a cache); o pedido seguinte
aplica-o e começa com a cache vazia:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Com `--metrics`, `/api/metrics` acrescenta contadores e indicadores `mgs:moe_cache_*` (passos, acertos, falhas, carregamentos,
especialistas na CPU, bytes de pré-carregamento, tamanho, posições e larguras de banda).

Medição de referência numa RTX 4060 Ti 16 GB com Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 threads da CPU), em comparação com
`--n-cpu-moe 28` sem cache: geração 26,9 → 35,7 tokens/s (+33%), processamento do prompt 767–778 →
718–730 tokens/s (−6%) e, com o MTP ativado, 33,6 → 47,0 tokens/s (+40%). Na mesma máquina, o llama.cpp
oficial com a sua própria cache (`-cmoe --moe-cache-mib 10240`) atingiu 26,7 tokens/s, e 27,8 com MTP. Os ganhos dependem da percentagem de especialistas
que cabe na VRAM, da localidade de encaminhamento do modelo e das larguras de banda do PCIe e da RAM.

<a id="threads"></a>
### Threads da CPU

A CPU calcula os especialistas que não estão na cache. Esse trabalho está limitado pela largura de banda da memória e
satura com poucas threads, enquanto usar todas as CPU lógicas faz com que as threads concorram com a que controla
a GPU. Sem `-t`, o `mgs` usa os núcleos úteis para cálculo, mas nunca mais de metade das CPU lógicas em que pode
ser executado, exceto numa CPU com até 8 CPU lógicas sem SMT, em que usa todos os núcleos: com tão poucos núcleos, o
limite é a CPU (num i7-7700K de 4 núcleos com uma RX 580, em que a CPU calcula 40 das 41 camadas de especialistas, 4 threads
geraram 17% mais depressa do que 2). Numa máquina com SMT (duas threads por núcleo), o limite corresponde ao número de núcleos
físicos, como antes; numa máquina virtual maior que não mostre SMT, é metade das CPU virtuais. Na máquina de referência (16 núcleos
virtuais com uma sessão de ambiente de trabalho, Qwen3.6-35B-A3B Q8_0, sem MTP), 4 a 12 threads geraram à mesma velocidade
(32,9–33,9 tokens/s) e 16 threads a 28,7 tokens/s. O registo de arranque mostra a escolha numa linha como
`using 8 CPU threads for generation and 8 for batches`; `-t` e `-tb` definem-nas explicitamente.

<a id="kv-cache-type"></a>
### Tipo da cache KV

Se `-ctk` (`--cache-type-k`) ou `-ctv` (`--cache-type-v`) não for indicado, o `mgs` dá à cache KV a precisão dos pesos do modelo, segundo o tipo que contém a maior parte dos pesos dos seus especialistas: os modelos não quantizados (F16, BF16, F32) mantêm uma cache `f16`, os de mais de 5 bits por peso (Q5_K_M, Q6_K, Q8_0) recebem `q8_0` e os restantes (Q4_K_M, IQ4_XS, MXFP4, Q3_K, Q2_K…) `q4_0`, os dois tipos quantizados que o flash attention lê em todos os backends. A regra é a mesma em todos os backends (CPU, CUDA, ROCm e Vulkan). Uma cache quantizada precisa de flash attention, pelo que a cache fica em `f16` quando o flash attention está desativado (`-fa off`) ou não consegue funcionar com esse tipo na GPU. O registo de arranque mostra a escolha numa linha como `KV cache type chosen from the weights of the model: q8_0`, e a página Modelo mostra os tipos em uso.

Uma cache `q8_0` ocupa cerca de metade da memória de uma `f16`, e uma `q4_0` cerca de 28%, pelo que o ajuste automático dá muito mais contexto na mesma memória, ou deixa essa memória para a cache de especialistas, à custa de uma atenção um pouco menos precisa e de alguns por cento de velocidade. Ler uma cache quantizada exige algum trabalho adicional em todos os backends, e o CUDA (GPU NVIDIA) e o ROCm (GPU AMD com uma compilação `--rocm`), que partilham os mesmos kernels de flash attention, só a leem diretamente quando se processam muito poucos tokens de cada vez (a geração normal, não o prompt nem a verificação de vários tokens do MTP) com cabeças de atenção de 64, 128 ou 256 valores, e nos restantes casos convertem-na para `f16` em cada passo. Numa Radeon 860M (Vulkan) com o Qwen3.6-35B-A3B Q6_K e sem MTP, o `q8_0` gerou menos 3,8% de tokens por segundo do que o `f16` após um prompt curto e menos 2,7% após um prompt de 8060 tokens, e processou esse prompt 1,3% mais lentamente. Se só uma das duas opções for indicada, a outra fica em `f16`, e `-ctk f16 -ctv f16` repõe a cache `f16` quando a memória é suficiente e esses poucos por cento importam mais do que o contexto. O contexto do MTP e o modelo de rascunho de `--spec-draft-model` seguem a mesma regra; `--spec-draft-type-k` e `--spec-draft-type-v` definem os seus tipos.

<a id="checkpoints"></a>
### Pontos de controlo de contexto

Os modelos com camadas recorrentes ou de janela deslizante (por exemplo, os híbridos Qwen3.5/3.6/3.8 ou o Nemotron-H)
não conseguem fazer recuar o seu estado, pelo que, quando um pedido altera uma parte anterior da conversa, o
servidor retoma a partir do ponto de controlo mais recente anterior à alteração. Os pontos de controlo são colocados no início das mensagens do utilizador,
do assistente e das ferramentas (onde os agentes de programação removem raciocínios antigos ou encurtam resultados antigos de ferramentas) e pelo menos
a cada `--checkpoint-min-step` tokens (predefinição: 1024, anteriormente 8192), sem passagens adicionais pelo prompt. Quando
se atinge `--ctx-checkpoints` (32), é removido o ponto de controlo mais próximo dos seus vizinhos, para que os outros continuem
distribuídos ao longo da conversa. No cenário de agente de referência (14k tokens, oito resultados de ferramentas), encurtar um resultado
de ferramenta tardio passou de 19,3 s para 5,5 s e um inicial de 18,0 s para 13,6 s. Cada ponto de controlo mantém o
estado recorrente ou de janela deslizante na memória do anfitrião (63 MiB para o Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Opções atualizadas e diagnóstico

Mantenha a predefinição `--load-mode mmap` quando o registo do anfitrião for bem-sucedido. Use `--load-mode none` apenas como alternativa quando o registo falhar. `--load-mode dio` passa agora as leituras por, no máximo, 64 MiB de memória adicional; isso não é um limite para a RAM total do modelo.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` foi removida porque não tinha qualquer efeito em tempo de execução. As antigas opções de carregamento (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) e a opção de compilação `GGML_CUDA_FA_ALL_QUANTS` deixaram de ser aceites; consulte [compatibilidade removida](#removed-compatibility) para as respetivas substituições. Numa compilação manual com o CMake, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` acrescenta essa combinação de kernel vetorial K/V; são aceites listas separadas por vírgulas ou separadas por pontos e vírgulas entre aspas. A lista predefinida é `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 está sempre incluído. `all` aumenta o trabalho de compilação. As combinações de kernel vetorial em falta usam uma alternativa de conversão para F16 e podem ser mais lentas.

O MTP separa os grafos com e sem saídas, mantém a ordem original dos tokens entre sequências simultâneas e usa as posições reais depois das imagens.

O ajuste automático conta as camadas nextn (MTP), pelo que a primeira camada deixa de ficar na CPU quando o MTP está desativado. Os pesos MTP ficam agora sempre na memória da GPU; consulte [colocação do MTP](#mtp-vram). O `--checkpoint-min-step` predefinido passa a ser 1024 em vez de 8192; consulte [pontos de controlo de contexto](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` e `--moe-cache-reserve` controlam a [cache de especialistas MoE](#expert-cache). O ajuste automático já não deixa os modelos maiores do que a VRAM com um contexto de 4096 tokens e passa também a escolher o microlote; consulte [tamanho do contexto](#context-size). O número predefinido de threads da CPU é agora, no máximo, metade das CPU lógicas; consulte [threads da CPU](#threads). `-np` passa a ter por predefinição um único slot com o contexto inteiro, em vez do modo automático com quatro slots a partilhar uma cache KV unificada, e `--kv-unified` está desativado por predefinição; consulte [fila de pedidos](#request-queue). Numa GPU integrada, o ajuste automático passa a contar a memória da GPU e a RAM como um único orçamento e deixa de criar a cache de especialistas MoE; consulte [GPU integradas](#integrated-gpu). Desde a 0.1.19, o ajuste automático deixa de parar quando a colocação é indicada explicitamente: mantém essa colocação e ajusta à sua volta o contexto e o microlote; consulte [tamanho do contexto](#context-size).

Com `--metrics`, o `GET /api/metrics` autenticado inclui `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` e `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Os índices de posição começam em zero; essa série aparece depois de um pedido especulativo terminar. Compare as variações dos contadores de tokens aceites e de rascunho ao avaliar a aceitação, e meça a latência separadamente; a aceitação, por si só, não prova uma aceleração. No modo router, selecione o modelo com `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, desativado por predefinição) permite que os URL multimédia dos pedidos cheguem a endereços de loopback, privados e outros não públicos; consulte [limites de segurança](#security-limits).

As exportações de conversas leem a árvore completa de mensagens persistida e os metadados. Se as ferramentas do servidor devolverem 403, as mensagens novas deixam de repetir essa consulta falhada; voltar a abrir o painel de ferramentas tenta-a novamente. As predefinições do servidor na primeira visita mantêm as definições já alteradas pelo utilizador, incluindo a chave de API.

[Implementação e fontes upstream](CODE.pt-PT.md#upstream-review).

<a id="removed-compatibility"></a>
## Compatibilidade removida

Os mecanismos de retrocompatibilidade foram removidos. Os nomes antigos já não são
traduzidos nem geram avisos: o servidor rejeita as opções de linha de comandos removidas como
argumentos inválidos, ignora as variáveis de ambiente removidas e responde 404 às
rotas removidas. Atualize os scripts, os overrides de serviços, as predefinições e os clientes para as
substituições indicadas abaixo.

| Removido | Usar em vez disso |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (ou `mmap+mlock` para combinar ambos) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Nada; não tinha qualquer efeito. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Aliases antigos do modelo de rascunho, como `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` ou `--draft-p-min` | A opção `--spec-draft-*` correspondente, por exemplo `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` ou `--spec-draft-p-min`. `mgs --help` lista todas. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` ou `--spec-ngram-mod-n-max` / `--spec-draft-n-min` ou `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | A opção `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` e `--spec-ngram-*-min-hits` do modo de n-gramas escolhido |
| Variáveis de ambiente, opções do CMake e símbolos da API C com o antigo prefixo do upstream, incluindo os aliases que versões anteriores já tinham deixado de usar | O nome atual com o prefixo `MGS_` (`mgs_` na API C), por exemplo `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` em vez de uma variante `NO_`, `MGS_ARG_LOAD_MODE` em vez das variáveis de mapeamento de memória, de bloqueio e de Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` e `MGS_ARG_SPEC_DRAFT_N_MIN` para os limites de rascunho, ou `MGS_BUILD_UI`; para as opções dos backends, a opção `GGML_*` correspondente, por exemplo `GGML_CUDA` ou `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `HF_HOME` e `MGS_CACHE` | Nada; todas as transferências, incluindo as do Hugging Face, vão para `~/.cache/mgs/` (dentro de `XDG_CACHE_HOME` quando está definida). |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` ou `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valor `deepseek-legacy` de `reasoning_format` (campo do pedido e `--reasoning-format`) | `deepseek` ou `auto`; o raciocínio é devolvido em `reasoning_content` |
| Campo do pedido `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (array de cadeias) |
| `reasoning_in_content` em `generation_settings` | Nada; o campo foi removido. |
| Transferência de uma interface pré-compilada do upstream a partir de um bucket do Hugging Face (as opções do CMake `USE_PREBUILT_UI` e `UI_HF_BUCKET` e a variável de ambiente `HF_UI_VERSION`) | Nada; a interface é sempre compilada a partir de `frontend/` (`MGS_BUILD_UI=ON`) ou deixada de fora. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Número de compilação e commit: as variáveis do CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (também em `mgs-config.cmake`) e a cadeia `b0-unknown` de `build_info`, `system_fingerprint` e do `User-Agent` | `MGS_VERSION`, definida em `backend/cmake/build-info.cmake`; esses campos contêm agora a versão, como `0.1.19` |
| Modelos Llama da Meta (as arquiteturas GGUF `llama`, `llama4` e `llama-embed` e o projetor de visão do Llama 4) e os modelos de chat integrados `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` e `llama4` | Nada; use um dos modelos MoE suportados. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Opção do instalador `--no-sudo` | Nada; os instaladores são sempre executados como root, sem sudo. |
| Funções e campos que as API C tinham marcado como obsoletos: em `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, os aliases `mgs_token_*` (por exemplo `mgs_token_bos`, `mgs_token_get_text` ou `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, o campo `defrag_thold` de `mgs_context_params` e `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; em `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` e `ggml_upscale_ext`; em `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` e o campo `image_marker` | A função atual: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` em vez de `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` e `media_marker`. As execuções de aquecimento são feitas por quem chama a função, e `defrag_thold` não tinha qualquer efeito. |
| Gemma 4: as arquiteturas `gemma4` e `gemma4-assistant`, os projetores `gemma4v`, `gemma4uv`, `gemma4a` e `gemma4ua`, o formato de chat `peg-gemma4`, o tokenizador e o pré-tokenizador `gemma4`, e as predefinições `--vision-gemma-4b-default` e `--vision-gemma-12b-default` | Nada; use uma das [famílias suportadas](#supported-models). |
| Modelos DeepSeek anteriores à V4.1: as arquiteturas `deepseek`, `deepseek2` (também os ficheiros Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 e Youtu convertidos com ela), `deepseek2-ocr` (com os projetores `deepseekocr` e `deepseekocr2`), `deepseek32`, `mistral4` e `deepseek4`, os pré-tokenizadores `deepseek-llm` e `deepseek-coder`, o backbone de rascunho DSpark do DeepSeek-V4 e a deteção do modelo de chat do DeepSeek-V3.2 (DSML com `function_calls`) | DeepSeek-V4.1-Flash (`deepseek41`). |
| As restantes arquiteturas MoE herdadas do llama.cpp: `grok`, `nomic-bert-moe`, `qwen2moe`, `qwen3moe`, `qwen3next`, `qwen3vlmoe`, `phimoe`, `minicpm`, `jamba`, `cohere2moe`, `dbrx`, `olmoe`, `arctic`, `glm4moe`, `exaone-moe`, `granitemoe`, `granitehybrid`, `bailingmoe`, `bailingmoe2`, `dots1`, `afmoe`, `laguna`, `ernie4_5-moe`, `hunyuan-moe`, `hy_v3`, `gpt-oss`, `lfm2moe`, `smallthinker`, `llada-moe`, `grovemoe`, `minimax-m2`, `rnd1`, `mistral3`, `step35`, `kimi-linear` e `mellum`, e os formatos de chat de Ministral 3, gpt-oss, Functionary v3.2, Kimi-K2, Cohere2-MoE, LFM2, GigaChat3 e MiniCPM5 | Nada; use uma das [famílias suportadas](#supported-models). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) e os modelos de chat integrados do motor sem Jinja (`--chat-template` com um nome como `deepseek3`, `chatglm4` ou `mistral-v7-tekken`) | Nada; os modelos de chat passam sempre pelo Jinja. `--chat-template chatml` continua a selecionar o ChatML; para outro modelo de chat, passe o seu código-fonte Jinja com `--chat-template` ou `--chat-template-file`. Os pedidos com `tools` já não precisam de nenhuma opção. |
| `-hfr` | `-hf` |
| Segundos nomes longos de uma opção: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe` (as chaves das predefinições do router seguem os mesmos nomes) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, com o nome das respetivas opções |
| Campo do pedido `thinking_budget_tokens` | `reasoning_budget_tokens` (os clientes Anthropic continuam a enviar `thinking.budget_tokens`) |
| `content` em `POST /api/embeddings` e `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, a struct `mgs_chat_message` e `MGS_FILE_MAGIC_GGLA` em `mgs.h`; `mtmd_image_tokens_get_id` em `mtmd.h` | Nada; a formatação do chat é feita pelos modelos Jinja de `common/chat.h`, e os restantes não tinham qualquer utilidade. |
| `MGS_API_KEY`, `MGS_ARG_BATCH`, `MGS_ARG_UBATCH`, `MGS_ARG_N_PREDICT`, `MGS_ARG_N_PARALLEL`, `MGS_ARG_N_GPU_LAYERS`, `MGS_ARG_EMBEDDINGS`, `MGS_ARG_RERANKING`, `MGS_ARG_LOG_VERBOSITY`, `MGS_ARG_ROPE_SCALING_TYPE`, `MGS_ARG_SPEC_DRAFT_HF_REPO`, `MGS_ARG_STATIC_PATH`, `MGS_ARG_ENDPOINT_METRICS`, `MGS_ARG_ENDPOINT_PROPS`, `MGS_ARG_ENDPOINT_SLOTS` | `MGS_ARG_API_KEY`, `MGS_ARG_BATCH_SIZE`, `MGS_ARG_UBATCH_SIZE`, `MGS_ARG_PREDICT`, `MGS_ARG_PARALLEL`, `MGS_ARG_GPU_LAYERS`, `MGS_ARG_EMBEDDING`, `MGS_ARG_RERANK`, `MGS_ARG_VERBOSITY`, `MGS_ARG_ROPE_SCALING`, `MGS_ARG_SPEC_DRAFT_HF`, `MGS_ARG_PATH`, `MGS_ARG_METRICS`, `MGS_ARG_PROPS`, `MGS_ARG_SLOTS`: cada variável é `MGS_ARG_` seguido do nome longo da respetiva opção em maiúsculas, com `_` em vez de `-` |
| `MGS_MOE_CACHE`, `MGS_MOE_CACHE_MIB`, `MGS_MOE_CACHE_RESERVE_MIB` | `--moe-cache`/`--no-moe-cache`, `--moe-cache-size` e `--moe-cache-reserve`, ou `MGS_ARG_MOE_CACHE`, `MGS_ARG_MOE_CACHE_SIZE` e `MGS_ARG_MOE_CACHE_RESERVE` |
| `MTMD_BACKEND_DEVICE`, `MTMD_DEBUG_EMBEDDINGS` e a opção do CMake `MTMD_VIDEO` | `MGS_MTMD_BACKEND_DEVICE`, `MGS_MTMD_DEBUG_EMBEDDINGS` e `MGS_MTMD_VIDEO` |
| `--spec-mtp-auto`, `--no-spec-mtp-auto` (`MGS_ARG_SPEC_MTP_AUTO`) | Nada: o MTP ativa-se por si próprio; `--spec-type none` desativa-o e qualquer outro `--spec-type` substitui-o. |
| `--spec-default` | `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, os valores que definia |
| As predefinições de modelos `--tts-oute-default`, `--embd-gemma-default`, `--fim-qwen-1.5b-default`, `--fim-qwen-3b-default`, `--fim-qwen-7b-default`, `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec`, `--fim-qwen-30b-default`, `--gpt-oss-20b-default` e `--gpt-oss-120b-default` | Nada; passe o modelo com `--model` ou `-hf`. |
| As opções das ferramentas de linha de comandos que o `mgs` não inclui, e as respetivas variáveis, como `-p`/`--prompt`, `-f`/`--file`, `-sys`/`--system-prompt`, `-i`/`--interactive`, `-cnv`/`--conversation`, `--hellaswag`, `--winogrande`, `--kl-divergence`, `--save-all-logits`, `--ppl-stride`, `-o`/`--output`, `--chunk`, `--image`, `--audio`, `--video`, `-pps`, `-tgs`, `-npp`, `--diffusion-steps`, `-lr`/`--learning-rate`, `-mv`/`--model-vocoder`, `-hfv`/`--hf-repo-v`, `--tts-speaker-file`, `-fitp`/`--fit-print`, `--show-timings`, `--prompt-cache`, `--tensor-filter`, `-gan`/`--grp-attn-n` e as opções de transferência `--mtp`, `--dflash` e `--eagle3` | Nada; o servidor nunca as usou. |
| Os valores `draft-eagle3`, `draft-dflash` e `draft-dspark` de `--spec-type`, com os modelos de rascunho Eagle3, DFlash e DSpark | `draft-mtp`, `draft-simple` ou um dos tipos de n-gramas |
| Os backends do ggml para além de CPU, CUDA, Vulkan e ROCm (HIP): BLAS, CANN, ET, Hexagon, Metal, MUSA, OpenCL, OpenVINO, RPC, SYCL, virtgpu, WebGPU, zDNN e ZenDNN, com as respetivas opções do CMake (como `GGML_BLAS`, `GGML_METAL`, `GGML_RPC` ou `GGML_SYCL`) e `--rpc` | CPU, CUDA (`--cuda`), ROCm (`--rocm`) ou Vulkan (`--vulkan`) |
| Em `mgs.h`, `mgs_get_logits`, `mgs_get_embeddings`, `mgs_encode`, `mgs_set_causal_attn`, `mgs_model_has_encoder`, `mgs_model_has_decoder`, `mgs_model_decoder_start_token`, `mgs_model_is_recurrent`, `mgs_get_ctx_other` e o campo `ctx_other` de `mgs_context_params`; em `mgs-ext.h`, `mgs_set_embeddings_layer_inp`, `mgs_get_embeddings_layer_inp`, `mgs_model_target_layer_ids` e `mgs_model_target_layer_ids_n`; em `mtmd.h`, `mtmd_decode_use_non_causal`, `mtmd_image_tokens_get_n_tokens` e `mtmd_image_tokens_get_n_pos` | `mgs_get_logits_ith`, `mgs_get_embeddings_ith` ou `mgs_get_embeddings_seq`, `mgs_decode` (que também executa modelos sem cache KV), `mtmd_input_chunk_get_n_tokens` e `mtmd_input_chunk_get_n_pos`; as restantes serviam modelos codificador-descodificador, só de embeddings ou densos, ou a cache KV partilhada entre contextos, e não têm substituto. |
| Os projetores multimodais do Gemma 3 (`gemma3`, `gemma3nv`, `gemma3na` e MobileNetV5) | Nada; use o projetor de uma das [famílias suportadas](#supported-models). |
| As rotas da OpenAI sem `/v1` (`GET /api/models`, `POST /api/chat/completions`, `POST /api/responses`, `POST /api/audio/transcriptions`, `POST /api/chat/completions/input_tokens`, `POST /api/responses/input_tokens`) e `GET /api/v1/health` | A mesma rota em `/api/v1` (`GET /api/v1/models`, `POST /api/v1/chat/completions`…) e `GET /api/health`. `POST /api/models` e `DELETE /api/models` do modo router mantêm-se. |
| Os campos `ahead_api` e `ahead_webui` dos eventos `mgs_queue` | `running_api`, `running_webui`, `waiting_api` e `waiting_webui` ([fila de pedidos](#request-queue)) |
| Variáveis do instalador `MGS_INSTALL_SOURCE_DIR`, `MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`, `MGS_BUILD_JOBS` e `MGS_BUILD_BACKEND`, e a opção `--build-ui` dos instaladores e de `build/build-local.sh` | `--build-dir`, `--prefix`, `--jobs` e `--cuda`, `--rocm`, `--vulkan` ou `--cpu`; o código-fonte vai sempre para `/opt/moe-gguf-server-source` e a interface é sempre compilada, a não ser que seja indicado `--no-ui`. |
| Configurar os serviços web como um passo separado depois da instalação | O instalador configura-os na mesma execução (`--domain`, `--certificate`, `--private-key`, `--proxy-ip`); o `configure-web.sh` continua disponível para reconfigurar sem recompilar. |
| O serviço de inferência em `127.0.0.1:8080` (`--port 8080`) atrás do Apache | O socket Unix `/opt/moe-gguf-server/_/temp/web/mgs.sock` ([implementação](#deployment)) |

O CMake indica as variáveis `-D` removidas como não utilizadas em vez de as aplicar, pelo que deve verificar
a saída da configuração das compilações manuais.

A interface web mantém no servidor as suas definições (`~/.mgs/config/ui-settings.json`) e as suas conversas (`~/.mgs/chats/`),
e no navegador apenas a sua chave de API (`MgsUi.apiKey` em `localStorage`). Não lê nem migra as
definições que as versões anteriores guardavam em `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, os modelos favoritos, as listas de ferramentas, o esforço de raciocínio,
as recomendações MCP dispensadas, o idioma e o modo de cor da antiga biblioteca de temas), pelo que as definições começam com
os seus valores predefinidos e a chave de API tem de ser introduzida novamente uma vez. Também não lê nem
migra os dados guardados por versões anteriores com outros nomes, a chave `theme`
autónoma, os marcadores de raciocínio e de chamadas a ferramentas embutidos nas mensagens antigas, os anexos
de texto colado do antigo tipo `context`, o antigo formato de lista das recomendações MCP
dispensadas nem o antigo campo `webui_settings` do servidor. Também não lê as conversas que as versões anteriores guardavam no navegador (a base de dados IndexedDB `MgsUi`):
para as manter, exporte-as a partir da versão anterior antes de atualizar e importe a exportação depois, o que as guarda
no servidor. A importação de conversas só aceita as exportações JSONL e ZIP, dentro dos limites indicados
em [limites de segurança](#security-limits); o antigo formato de exportação JSON é rejeitado. As definições antigas ficam no navegador até serem eliminadas: para
as remover, abra a consola do navegador (F12) na página da interface e execute
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
uma vez para cada endereço e porta com que a interface foi aberta.

Todas as transferências são guardadas em cache em `~/.cache/mgs/` (dentro de `XDG_CACHE_HOME` quando está definida):
os ficheiros transferidos com `-mu`/`--model-url` e, mantendo a sua própria estrutura
(`~/.cache/mgs/hub/models--<organization>--<repo>/`), as transferências do Hugging Face. Nenhuma
variável altera essa pasta. Os ficheiros que as versões anteriores guardaram noutra pasta de cache,
incluindo a cache do Hugging Face (`~/.cache/huggingface/hub`), são transferidos novamente.

### Ficheiros de modelos antigos

O código de compatibilidade para as seguintes conversões antigas também foi removido.
Volte a converter o modelo original, ou transfira um GGUF atual, para os usar.

| Ficheiro antigo | Comportamento atual |
| --- | --- |
| Grok-1 convertido antes de agosto de 2025 (sem `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` ou `grok.attn_logit_softcapping`) | Não carrega: chave em falta. |
| Kimi-Linear com o tensor `blk.N.attn_kv_b` não dividido | Não carrega: faltam `attn_k_b`/`attn_v_b`. Esse caminho já falhava ao construir o grafo. |
| Ficheiros Qwen3-Next apenas com o tensor `ssm_in` fundido | Não carrega: falta o tensor `blk.0.attn_qkv.weight`. |
| Ficheiros sem `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) ou `full_attention_interval` (Qwen3-Next), e ficheiros Cohere2-MoE com um único período SWA em vez do array por camada | Não carrega: chave em falta. |
| Vocabulários BPE sem `tokenizer.ggml.pre` ou com os nomes de pré-tokenizador `llama3`, `llama-v3`, `jina-es`, `jina-de` ou `llada-moe`, e tokenizadores Kimi-K2 sem merges | Não carrega: chave em falta, tipo de pré-tokenizador desconhecido ou merges em falta. |
| Ficheiros GGUF da versão 1 ou 2 | Não carrega: só é aceite a versão 3 do GGUF. |
| Ficheiros mmproj dos antigos scripts pontuais: tipos de projetor `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x a 4.5, MiniCPM-o 2.6) e `adapter` (GLM-Edge-V), e ficheiros sem tipo de projetor | O projetor não carrega: tipo de projetor desconhecido ou em falta. |
| Ficheiros mmproj de conversões antigas com `ffn_up` e `ffn_down` trocados | Já não são corrigidos; o projetor falha. |
| mmproj do Kimi-K2.5 sem `image_min_pixels`/`image_max_pixels` e Step3-VL sem `preproc_image_size` | O projetor não carrega: chave ou tensor em falta. |
| Ficheiros que só contêm as antigas chaves `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` e `tokenizer.ggml.middle_token_id` em vez das `tokenizer.ggml.fim_*_token_id` | Carrega; os tokens de preenchimento (infill) são reconhecidos pelo seu texto (por exemplo `<PRE>` ou `<\|fim_prefix\|>`), e o preenchimento não funciona se não forem encontrados. |

Foram mantidos os mecanismos alternativos de que ainda dependem as conversões atuais ou os GGUF publicados mais transferidos das
famílias suportadas, verificados com base nos respetivos cabeçalhos no
Hugging Face: o `add_bos` predefinido do `llama-bpe`, do Tekken e de pré-tokenizadores
semelhantes; a estrutura das camadas recorrentes do Qwen3.6 derivada de
`full_attention_interval` (os GGUF do Qwen3.6 mais transferidos não têm
`recurrent_layers`); e a estrutura integrada do indexador do GLM-5.2 para os ficheiros GLM-5.2/5.3
sem `attention.indexer.types`.

<a id="maintenance"></a>
## Manutenção e resolução de problemas

- Verifique que versão está instalada com `mgs --version` (mostra, por exemplo, `version: 0.1.57`); a compilação local
  executa-o no final e o CMake mostra `mgs version: 0.1.57` durante a configuração. A interface mostra a mesma versão
  em Definições → Visualização → **Mostrar informações da versão de compilação** (canto inferior direito) e em **Informações de compilação** na
  janela de informações do modelo. Cada alteração ao projeto incrementa o último número: 0.1.0, 0.1.1, 0.1.2, etc.
- Repita `curl | bash` para atualizar a partir de `main`, ou o comando de instalação local adequado para voltar a compilar e substituir o binário.
  Indique novamente as mesmas opções web (`--domain`, `--proxy-ip`…), porque cada execução reescreve a configuração
  web. O instalador reinicia ambos os serviços no final; para reiniciar manualmente o serviço de inferência, execute
  `systemctl restart moe-gguf-server` no Debian, ou `rc-service moe-gguf-server restart` no Alpine.
- Leia `/root/webapp-install.log`, `journalctl -u moe-gguf-server` no Debian,
  ou `/var/log/moe-gguf-server.log` no Alpine. Os registos do Apache estão em
  `/var/www/DOMAIN-logs/`.
- Se a compilação CUDA não estiver disponível, instale o Toolkit externamente ou use
  `--rocm`, `--vulkan` ou `--cpu`. Se o NCCL não for encontrado, verifique a disponibilidade do repositório e a versão do CUDA,
  ou escolha explicitamente `--no-nccl`. Se o ROCm não for encontrado, instale-o com os respetivos ficheiros de
  desenvolvimento ou defina `HIPCXX` ([ROCm](#rocm)).
- Se a compilação da interface falhar, verifique as versões do Node.js e do npm, as transferências das dependências
  e o registo. Não a substitua por uma interface pré-compilada do upstream.
- Nas compilações de implementação, use um diretório novo dentro de `_/temp/` depois de alterações à estrutura do código-fonte.
  Uma cópia do projeto em SSHFS pode impedir a execução de executáveis; use um disco local em produção.
- Na compilação local, volte a executar o comando de uma linha acima para transferir o `main` atual, ou
  `bash build/build-local.sh` para usar o seu código-fonte local, e reinicie a cópia que
  iniciou manualmente. Leia a saída do terminal se falhar ou se indicar bibliotecas
  fora de `lib/`. O `build.log` temporário é eliminado juntamente com o diretório de compilação.
- Os erros 404 da API em clientes mais antigos exigem normalmente alterar o URL base para
  `/api/v1`. A chave de API instalada deve ir no cabeçalho Authorization.

Consulte [CODE.pt-PT.md](CODE.pt-PT.md) para o mapa da implementação e os pontos de extensão.

<a id="license"></a>
## Licença

O `mgs` é distribuído ao abrigo da [PolyForm NonCommercial License 1.0.0](../LICENSE). Pode usá-lo,
modificá-lo e partilhá-lo para qualquer fim não comercial (NonCommercial): utilização pessoal, estudo, investigação, projetos de passatempo,
e utilização por organizações de beneficência, instituições de ensino, organizações públicas de investigação e
instituições governamentais. A utilização comercial, como a de uma empresa que o use para o seu negócio, exige uma
licença separada concedida pelo nipegun.

As partes que provêm do llama.cpp (MIT, The ggml authors), do FreeToken (Apache 2.0) e das
bibliotecas incluídas mantêm as suas próprias licenças; os respetivos avisos estão em `build/licenses/`
e a compilação local instala-os, juntamente com `LICENSE`, em `~/.local/share/doc/mgs/licenses/`.
