# Manual do utilizador

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Conteúdo: [instalação](#installation), [compilação local](#local-build), [implementação](#deployment),
[chat](#chat), [monitor](#monitor), [modelos e ferramentas](#models), [modelos suportados](#supported-models), [API](#api),
[fila de pedidos](#request-queue), [limites de segurança](#security-limits), [Vulkan](#vulkan), [desempenho](#performance), [compatibilidade removida](#removed-compatibility),
[manutenção](#maintenance), [licença](#license).

<a id="installation"></a>
## Instalação e primeiro arranque

Instale como root no servidor de produção. O Debian/Ubuntu aceita o instalador
diretamente pela entrada padrão:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

Não é preciso transferir primeiro nenhum ficheiro adjacente. O script de entrada transfere o
arquivo completo do código-fonte de `main` a partir [deste repositório](https://github.com/nipegun/moe-gguf-server/), verifica-o
e sincroniza uma cópia gerida em `/opt/moe-gguf-server-source`. As transferências
e as caches de compilação ficam em `_/temp/` do projeto. São necessários o curl e os certificados de CA
para obter o script de entrada; as dependências de bootstrap em falta são depois
instaladas automaticamente.

Para escolher outro diretório de código-fonte e passar opções de compilação, use:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

O diretório remoto tem de ser absoluto e estar num disco local. O instalador só
adota um diretório vazio ou reutiliza uma cópia que tenha a sua marca de origem do código-fonte;
os diretórios não relacionados são rejeitados. Um bloqueio impede instalações simultâneas
nessa cópia. As execuções seguintes de `curl | bash` sincronizam os ficheiros do código-fonte, incluindo os ficheiros
removidos de `main`, e preservam `_/` e as caches de compilação. As chaves de API, os modelos e os
serviços implementados ficam fora dessa árvore de código-fonte e não são alterados. Uma transferência
ou uma compilação falhada não instala um novo executável; volte a executar o mesmo comando para
tentar novamente. Não é necessária nenhuma operação Git.

Um script transferido isoladamente usa o mesmo modo remoto. Executar
`bash deploy/install-update-reinstall-debian.sh` a partir de uma cópia local completa
compila essa cópia sem transferir o código-fonte. Os argumentos a seguir a `bash -s --`
são reencaminhados para o instalador de compilação. `--help` não requer root nem a transferência
do código-fonte. `--no-system-deps` desativa também a instalação dos pacotes de bootstrap: o curl,
os certificados de CA, o tar, o gzip, o rsync, o util-linux e o coreutils já têm de existir.
Os subprocessos do instalador não podem consumir o script a partir do pipe; todas as fases partilham
um único fluxo de saída `/root/webapp-install.log`.

A configuração web continua a ser um passo separado, feito com
`/opt/moe-gguf-server-source/deploy/configure-web.sh` ou com o seu equivalente no
diretório de código-fonte escolhido.

O Alpine continua a usar uma cópia local completa e
`bash deploy/install-update-reinstall-alpine.sh`, com o Bash instalado previamente.
O Debian usa por predefinição CUDA/NCCL; o Alpine usa por predefinição a CPU e não suporta o
caminho de instalação CUDA/NCCL. Use `--cpu` para uma compilação de CPU no Debian ou `--vulkan` para GPU AMD, Intel ou NVIDIA
através de Vulkan ([Vulkan](#vulkan)). O CUDA Toolkit e os controladores NVIDIA já têm de estar instalados. `--no-nccl` desativa explicitamente o NCCL;
`--no-system-deps` deixa a gestão de pacotes a cargo do administrador.

A interface local precisa do Node.js 20.19+ na série 20.x ou 22.12+. `--no-ui` omite-a. As outras
opções são `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST`, `--no-openssl` e `--build-ui`.
`MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`,
`MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND` e `CUDACXX` permitem as substituições correspondentes
através de variáveis de ambiente. Os diretórios de compilação e de preparação têm de ser descendentes distintos
e sem sobreposição do diretório `_/temp/` do projeto.

O instalador adiciona os pacotes NCCL correspondentes e o repositório da NVIDIA quando é necessário,
mas nunca instala nem atualiza o CUDA Toolkit nem os controladores da GPU. Liga as bibliotecas
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
bash deploy/install-update-reinstall-debian.sh --cpu
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
| `~/.local/lib/mgs/` | O binário do servidor (`mgs`, com a interface web e a documentação da API incorporadas) e todas as bibliotecas partilhadas de que precisa: o carregador dinâmico e a biblioteca C da glibc, o runtime de C++, o OpenMP, o OpenSSL, os módulos de serviço de nomes da glibc e, nas compilações CUDA, o runtime do CUDA, o cuBLAS e o NCCL. |
| `~/.local/share/doc/mgs/licenses/` | A licença PolyForm NonCommercial 1.0.0 do projeto (`LICENSE`) e os avisos de licença de terceiros copiados de `build/licenses/` da árvore do código-fonte (llama.cpp, FreeToken, Swagger UI, cpp-httplib e nlohmann/json). |

Só o utilizador que executa o script pode usar esta instalação. Tudo é resolvido em relação ao local onde está o
lançador: este executa `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` a partir do seu prefixo, e o servidor e as suas bibliotecas procuram
em `lib/mgs/` antes de qualquer outro sítio, incluindo `LD_LIBRARY_PATH`. A única biblioteca que continua a ser obtida do sistema é o
controlador NVIDIA (`libcuda.so.1` e as bibliotecas `libnvidia-*` que este carrega), porque tem de corresponder ao módulo do kernel
em execução; com `--vulkan`, o controlador Vulkan do sistema ([Vulkan](#vulkan)). No final, o script lista, através do carregador incluído, as bibliotecas que o servidor carregaria, falha se alguma
vier de fora de `lib/mgs/` e executa `--version` através do lançador.

O Debian adiciona `~/.local/bin` ao `PATH` no início de sessão assim que essa pasta existe (através de `~/.profile`), pelo que, depois de voltar
a iniciar sessão, pode executar o `mgs` pelo nome; o script mostra um lembrete enquanto este não estiver no `PATH`. Inicie-o sempre que
quiser e pare-o com Ctrl+C:

```bash
mgs --model /path/model.gguf
```

O caminho completo também funciona: `~/.local/bin/mgs --model /path/model.gguf`.

Abra `https://127.0.0.1:11443/` para o chat e `https://127.0.0.1:11443/api/doc/` para
a documentação da API. Todas as opções deste manual se aplicam; por exemplo,
`--models-dir` ativa o router multimodelo.

Requisitos, que instala por si próprio como root, porque o script apenas os
verifica: `cmake`, um compilador C/C++ e a biblioteca C estática (`build-essential`, que
traz `libc6-dev`), `patchelf`, `libssl-dev` (ou
`--no-openssl`), Node.js 20.19+/22.12+ com npm (ou `--no-ui`), um CUDA
Toolkit já instalado (ou `--cpu`) e o NCCL com os seus ficheiros de desenvolvimento (`libnccl2` e
`libnccl-dev`, ou `--no-nccl`). O `nvcc` é obtido de `CUDACXX`, do `PATH` ou de
`/usr/local/cuda/bin/nvcc`. O Ninja é usado quando está instalado.
Os utilitários básicos incluem `tar`, `coreutils` e `util-linux`. As compilações remotas também
exigem `curl`, `ca-certificates` e `gzip`; não é instalada automaticamente nenhuma dependência.

Opções: `--prefix PATH` (outro prefixo de instalação), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`
e `--help`.

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
- Uma compilação CUDA precisa do controlador NVIDIA para arrancar; use `--vulkan` em computadores com outra GPU e `--cpu` em
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
- O `mgs` verifica onde o kernel mantém realmente essa memória (`/proc/self/fdinfo`, publicado pelo amdgpu, pelo i915 e pelo xe). Se
  outro programa encher a VRAM e o controlador mover parte dela para a GTT, o registo e o Monitor indicam-no, e o
  Monitor mostra a localização real (VRAM e GTT) de cada GPU AMD ou Intel. A cache de especialistas é reservada com uma
  prioridade de memória mais baixa (`VK_EXT_memory_priority`), pelo que o controlador a move antes das camadas MTP, dos pesos ou da
  cache KV. Sem resizable BAR, os buffers do dispositivo também evitam a pequena janela de VRAM que a CPU consegue mapear (256 MiB numa
  RX 580), que o ambiente de trabalho partilha e que o controlador esvazia primeiro para a GTT.
- Só o CUDA consegue fixar as páginas do modelo que ficam na RAM, pelo que, com Vulkan, os especialistas chegam à GPU através de memória
  paginável. `--load-mode none` copia-os, em vez disso, para memória do anfitrião fixada, o que numa RTX 4060 Ti através de Vulkan
  (Qwen3.6-35B-A3B Q4_K) aumentou os carregamentos de especialistas de 2,7 para 13,1 GB/s e a geração de 27,5 para 36,6 tokens/s. Na
  AMD e na Intel, essa memória fixada conta para a GTT: quando os especialistas não cabem nela, o carregamento falha com
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), como aconteceu com os 18,9 GB de especialistas desse
  modelo com 16,8 GB de GTT numa RX 580. Use-o apenas quando os especialistas couberem na GTT e na RAM livre.
- Dimensione o modelo de acordo com a máquina: o GGUF inteiro deve caber na RAM livre mais a VRAM, depois de descontar a cache KV e os
  buffers de cálculo. Com 8 GB de VRAM e 16 GB de RAM, isso significa cerca de 18–20 GB de pesos; os ficheiros maiores são lidos
  do disco uma e outra vez e a geração torna-se muito lenta. O Q4_K de 21,6 GB acima atingiu 7,8 tokens/s numa
  RX 580 com 15 GB de RAM e 15,9 tokens/s quando a máquina passou a ter 31 GB (10,0 com `--spec-type none`).

<a id="deployment"></a>
## Implementação em produção

A partir do diretório do código-fonte (`/opt/moe-gguf-server-source` nas instalações remotas),
depois de instalar o binário, execute:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

O domínio tem de resolver para o servidor de produção. O serviço `moe-gguf-server-web`
executa uma instância dedicada do Apache como `moe-gguf-server`, sem root.
Cada arranque seleciona a primeira porta HTTP disponível a partir de 11080 e a primeira porta HTTPS
disponível a partir de 11443. Cada sequência avança de forma independente até 65535, sem
atribuir a mesma porta aos dois protocolos. Inspeciona os sockets TCP IPv4/IPv6 em escuta
e tenta novamente se o Apache detetar que uma porta ficou ocupada depois de ter sido selecionada. As outras
falhas de configuração interrompem o arranque, em vez de serem confundidas com conflitos de portas.

O HTTP redireciona de forma permanente para a porta HTTPS selecionada e mantém o caminho e a query string.
O redirecionamento não é guardado em cache. O Apache reencaminha os pedidos para a inferência em
`127.0.0.1:8080`, iniciada explicitamente com `--http --port 8080` e também executada como
`moe-gguf-server`. A instalação requer root;
nenhum dos serviços precisa de ser executado como root. Não é configurado nenhum HAProxy e não são importados nem modificados
sites partilhados do Apache.

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
web (`web-domain`), o modelo de configuração do Apache (`apache2.conf.in`) e as definições da
interface (`ui-settings.json`), todos pertencentes a `moe-gguf-server` e com modo 0600.
As credenciais e a saída da instalação ficam registadas em `/root/webapp-credentials.txt`
e `/root/webapp-install.log`, ambos com modo 600. A reconfiguração mantém as chaves,
os modelos e os certificados existentes. A chave privada TLS pertence ao root e o grupo
moe-gguf-server pode lê-la (modo 640), o que permite que o processo web sem privilégios a leia.

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
Use o mesmo comando depois de renovar os certificados de origem. A opção
`--binary` do script admite uma localização do executável diferente da predefinida.

O serviço usa o modo router. Coloque modelos GGUF legíveis em
`/opt/moe-gguf-server/models`, ou use a gestão de modelos da interface. Para
opções de linha de comandos especializadas, use um override do systemd no Debian/Ubuntu ou a
configuração do serviço OpenRC no Alpine. Mantenha o backend associado ao loopback.
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
2. Selecione um modelo. No modo router, carregue um modelo disponível se nenhum estiver carregado.
3. Escreva uma mensagem e envie-a. Por predefinição, Enter envia; Shift+Enter insere uma quebra
   de linha. Estes atalhos podem ser alterados nas definições.
4. Adicione ficheiros com a ação de anexar. As imagens, o áudio e o vídeo exigem modalidades do modelo
   compatíveis. Os ficheiros PDF podem ser extraídos como texto ou convertidos em imagens;
   os modelos sem visão usam a via de texto.
5. Pare a geração, salte o raciocínio quando for suportado, edite uma mensagem, gere de novo uma
   resposta ou crie uma ramificação de uma conversa com as ações das mensagens.

A barra lateral permite pesquisar, afixar, mudar o nome, eliminar e selecionar em massa.
As conversas são guardadas no navegador com o IndexedDB; exporte-as antes de limpar
o armazenamento do navegador ou de mudar de perfil do navegador. As definições são guardadas pelo servidor em
`~/.mgs/config/ui-settings.json` do utilizador que executa o `mgs` (no serviço implementado,
`/opt/moe-gguf-server/.mgs/config/ui-settings.json`), pelo que todos os navegadores que abrem a
interface as partilham: os parâmetros de geração, as opções de visualização, o tema, o idioma, os servidores
MCP, os modelos favoritos, as ferramentas desativadas e as sempre permitidas e o esforço de raciocínio
predefinido. Só a chave de API fica em cada navegador, porque autentica o pedido
que carrega o resto, pelo que tem de a introduzir uma vez em cada navegador. A importação e a exportação estão disponíveis nas
definições. As exportações podem conter anexos e, se forem selecionados explicitamente, chaves de API
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

Escolha Sistema, Claro ou Escuro para o tema. As definições de geração incluem a utilização
do contexto, a temperatura, os amostradores, as penalizações e os limites de turnos de agente. Algumas capacidades
dependem de opções do servidor ou do modelo selecionado.

Depois de cada resposta, a interface envia novamente a conversa, quando o servidor está
inativo, para que esta já esteja codificada na cache KV enquanto lê e o turno seguinte
comece mais cedo. Está ativado por predefinição; desative Definições → Desenvolvimento → **Preencher cache KV
após cada resposta** para o impedir.

Enquanto o servidor está a resolver outro pedido, uma mensagem que envie espera na
[fila](#request-queue), e a sua resposta mostra um aviso em vez do texto de progresso até
começar: “Em fila de espera: outro pedido enviado diretamente para a API está a ser resolvido
primeiro (pedidos à frente: N).” quando um programa usa a API ao mesmo tempo, ou o aviso equivalente
para um pedido da própria interface web (outro separador ou navegador, um título gerado ou o preenchimento
prévio da cache KV). O aviso desaparece assim que o seu pedido começa.

<a id="monitor"></a>
## Monitor e barra de estado

O **Monitor** fica por baixo de **Definições** na barra lateral (endereço `#/monitor`). Enquanto está aberto, atualiza-se a cada
dois segundos e mostra:

- **Utilização**: primeiro, um cartão por cada GPU da máquina (NVIDIA, AMD ou Intel, mesmo que nenhum modelo a use), intitulado com a sua
  etiqueta a negrito e o seu nome (por exemplo **gpu-1**: NVIDIA GeForce RTX 4060 Ti; com várias GPU, gpu-1, gpu-2 e
  assim sucessivamente), com a sua utilização, a utilização do controlador de memória (largura de banda), a VRAM em uso, a temperatura e o consumo de energia face ao seu
  limite; depois, um cartão por cada processador, intitulado da mesma forma com o seu modelo (por exemplo **cpu-1**: AMD Ryzen 9 5950X
  16-Core Processor; com vários processadores, cpu-1, cpu-2 e assim sucessivamente), com a sua utilização, uma pequena barra vertical por cada
  CPU lógica desse processador e a carga média do sistema. Todas as barras de utilização são verdes e enchem-se de vermelho à medida que a utilização
  aumenta; uma barra cujo valor é desconhecido fica cinzenta. As GPU NVIDIA são lidas através da biblioteca NVML do controlador e as GPU AMD
  através de `gpu_busy_percent` no sysfs; qualquer outra GPU (Intel, ou NVIDIA sem NVML) não mostra utilização, e o seu
  nome e a sua memória vêm do backend de inferência assim que um modelo a usa.
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

O `mgs` só aceita modelos principais MoE. Os modelos densos, como o GPT-2 ou o Phi-2, são rejeitados antes de os seus pesos serem carregados; as respetivas implementações de descodificador não são compiladas. As camadas densas dentro de um modelo MoE e uma cabeça só de MTP compatível indicada explicitamente são mantidas.

<a id="supported-models"></a>
### Famílias de modelos suportadas

| Família | Arquitetura GGUF | Notas |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP quando o GGUF o inclui. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP quando o GGUF o inclui. A sua tabela de embeddings por camada (cerca de 29 GB) é lida a partir do ficheiro do modelo conforme necessário, pelo que deve manter o ficheiro num disco rápido. Os contextos longos são calculados com atenção densa sob a seleção esparsa: o resultado é correto, mas sem a aceleração dos kernels esparsos. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; imagens com o seu mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Os GGUF publicados não incluem MTP. Com Vulkan, a sua mistura residual é executada na CPU. |
| DeepSeek-V4.1-Flash | `deepseek41` | Carrega as duas estruturas publicadas: os ficheiros DwarfStar do antirez (como `antirez/deepseek-v4.1-flash-gguf`) e os ficheiros do ecossistema llama.cpp. A sua tabela Engram (cerca de 188 GiB dentro do ficheiro) nunca é carregada: cada token lê as suas linhas a partir do ficheiro do modelo, pelo que o ficheiro tem de continuar disponível enquanto o `mgs` está em execução, de preferência num SSD, e é necessário o Linux ou outro sistema POSIX. Os ficheiros sem modelo de chat usam um integrado, e `reasoning_effort` aceita um número de 1 a 100. Sem MTP e sem imagens. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

O Gemma 4 e os modelos DeepSeek anteriores à V4.1 deixaram de ser suportados (consulte [compatibilidade removida](#removed-compatibility)).

Com várias GPU selecionadas, o MTP verifica todas antes de recorrer à RAM. A sua estimativa de memória inclui a cabeça de saída. Os buffers MTP integrados são carregados antes dos outros pesos; um GGUF de MTP externo passado com `--spec-draft-model` é carregado antes do modelo principal, também com `--fit off`. A seleção explícita de dispositivos em vigor continua a determinar que GPU estão disponíveis.

Use `--reasoning on` ou `--reasoning off` para controlar o raciocínio. Passar `enable_thinking` através de `--chat-template-kwargs` é rejeitado. Use os nomes de opções documentados, com hífenes; as grafias com sublinhados são rejeitadas. Passe os valores de listas como um valor separado por vírgulas, em vez de repetir uma opção. Só `--spec-type` continua a poder ser repetida para compor modos especulativos.

O seletor de modelos mostra os modelos disponíveis, os carregados e os favoritos. Os detalhes do modelo
incluem o tamanho do contexto, as modalidades, a quantização e o modelo de chat. O modo router
gere processos filhos separados para os modelos carregados; os pedidos identificam o modelo
através do campo `model` ou do parâmetro de consulta.

Os servidores MCP são configurados por URL, com cabeçalhos de autorização opcionais. Pode
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

Inicie o `mgs` com `--local-shell-mcp-server` para permitir que os modelos do chat web usem o terminal da máquina onde
o `mgs` está em execução:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- No arranque, o `mgs` verifica que shells estão instaladas e oferece uma ferramenta para cada uma delas: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) e `psh.run` (PowerShell, `pwsh`). O registo lista as shells que encontrou; uma shell
  que não esteja instalada não recebe nenhuma ferramenta.
- A interface deteta-o, adiciona-o aos servidores MCP como **local shell mcp server** e ativa-o. Pode desativá-lo aí,
  ou numa única conversa, e a sua escolha é mantida. Quando o `mgs` arranca sem a opção, a entrada
  desaparece.
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
  (`"event": "error"`). Uma chamada rejeitada antes de ser executada (sem comando, com um diretório de trabalho que não existe, com um
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
são rejeitados, a origem CORS predefinida passa a ser localhost e o registo guarda cada comando com o respetivo código de saída. Leia
o comando apresentado na janela da palavra-passe antes de escrever uma palavra-passe do sudo: a palavra-passe autoriza esse comando. O
servidor só a entrega a um auxiliar de palavra-passe cujo processo pai seja o sudo a correr como root em nome do seu utilizador, pelo que um comando
não consegue iniciar o auxiliar por si só para ler a palavra-passe.

<a id="api"></a>
## API HTTP

Os exemplos usam 11443; substitua-a pela porta HTTPS de `ports.json` se for diferente.

A raiz predefinida da API é `/api`. A interface continua em `/`. As rotas `/v1/...` na raiz,
do servidor original, deixaram de estar registadas. Os clientes compatíveis com a OpenAI têm de
usar `https://ai.example.com:11443/api/v1` como URL base. Os nomes dos campos dos pedidos e das respostas
mantêm a compatibilidade com o upstream; as interfaces do motor em C/C++ mantêm as suas funções atuais,
mas não as que estavam marcadas como obsoletas (consulte [compatibilidade removida](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Defina `MGS_API_KEY` com a chave instalada na sua própria shell. As operações nativas,
os embeddings, a tokenização, a reclassificação (reranking), a gestão de modelos e os fluxos retomáveis estão
documentados no Swagger UI incluído em `/api/doc/`; o respetivo JSON OpenAPI está em
`/api/doc/openapi.json`, gerado pelo servidor em cada pedido, sem um ficheiro JSON estático. O inventário reflete as rotas registadas pelo
processo em execução, incluindo as rotas do router e as rotas de compatibilidade opcionais. As funcionalidades
desativadas podem devolver 403. Uma chave em falta ou inválida devolve 401 e o carregamento do modelo
devolve 503. As rotas de estado de saúde e de lista de modelos mantêm o acesso público para `GET` e `HEAD`; consulte
[limites de segurança](#security-limits) para as rotas que exigem sempre a chave.

`--api-prefix /api/custom` muda as rotas de inferência para baixo de `/api/custom`.
A interface na raiz recebe esse prefixo de `/api/config.js`; a documentação continua em `/api/doc/`.
O espaço de nomes da documentação está reservado e não pode ser usado como prefixo
de inferência. Os processos filhos do router usam `/api` internamente. O servidor valida os prefixos;
os valores fora de `/api/` e os caminhos com barras finais são rejeitados.

O Monitor usa `GET /api/monitor` (relatório completo), `GET /api/monitor/memory` (instantâneos da memória dos modelos carregados)
e `GET /api/monitor/usage` (só a utilização da CPU e da GPU, também disponível enquanto o modelo está a carregar). Estas rotas leem contadores e
instantâneos publicados pelo ciclo de inferência, nunca a fila de tarefas, pelo que consultá-las periodicamente não acorda um modelo adormecido nem
reinicia o seu temporizador de inatividade. Os respetivos campos estão descritos no Swagger.

A interface lê as suas definições com `GET /api/ui/settings` (`{"settings": DOCUMENT}`, ou
`{"settings": null}` antes da primeira gravação) e guarda-as com `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, que substitui por completo `~/.mgs/config/ui-settings.json`, com modo 0600). Ambas exigem
a chave de API e também respondem enquanto o modelo está a carregar.

<a id="request-queue"></a>
### Fila de pedidos e pedidos em paralelo

Por predefinição, o `mgs` resolve um pedido de cada vez (`-np 1`) e atribui-lhe o contexto inteiro, pelo que cada pedido pode
usar todos os tokens de contexto, quer venha da interface web, quer venha diretamente da API. Os pedidos que
chegam entretanto esperam numa fila única e são resolvidos pela ordem em que chegaram: os pedidos da interface não
passam à frente dos pedidos da API, nem o contrário. A API comporta-se da mesma forma, sejam quais forem as definições
da interface.

Para resolver vários pedidos ao mesmo tempo, inicie o `mgs` com `-np N` (`--parallel N`, `MGS_ARG_N_PARALLEL`):
até N pedidos são executados em paralelo e o contexto é repartido entre eles, pelo que cada um recebe uma parte (`n_ctx_slot` na
linha `initializing` do registo); com `--kv-unified`, partilham em vez disso uma única cache KV do contexto
inteiro. Os pedidos para além de N continuam a esperar na fila.

A interface web acrescenta `"mgs_client": "webui"` aos seus pedidos. Enquanto um dos seus pedidos em streaming espera, o
fluxo começa com eventos como

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: pedidos em execução ou em espera à sua frente; `ahead_api` e `ahead_webui`: quantos deles chegaram diretamente
à API ou vieram da interface web), e um último com todas as contagens a 0 quando começa. Os clientes que não
enviam `mgs_client` nunca recebem estes eventos, pelo que os clientes compatíveis com a OpenAI recebem o fluxo habitual. O Swagger
descreve o campo juntamente com os outros parâmetros de conclusão.

<a id="security-limits"></a>
### Limites de segurança

Estes limites protegem o servidor e a interface contra pedidos, ficheiros e modelos hostis. São suficientemente generosos
para uma utilização normal: o que os ultrapassar falha com um erro claro, em vez de parar o servidor.

- **Chave de API.** As rotas de estado de saúde e de lista de modelos só são públicas para `GET` e `HEAD`. Recarregar a lista de modelos
  (`GET /api/models?reload=1`) e adicionar ou eliminar modelos no modo router (`POST` e `DELETE /api/models`)
  exige a chave sempre que houver uma configurada.
- **Páginas de outros sites.** Quando as ferramentas do servidor ou os servidores MCP estão ativados (`--tools`, `--agent`,
  `--mcp-servers-config`, `--mcp-servers-json` ou `--local-shell-mcp-server`) e `--cors-origins` não está definido,
  o servidor responde 403 a qualquer pedido que não seja `GET`, `HEAD` ou `OPTIONS` cujo cabeçalho `Origin` indique
  outro site. A interface (também por trás do Apache) e os clientes que não enviam `Origin`, como o curl ou os SDK,
  não são afetados. `POST /api/tools` só aceita `Content-Type: application/json` (caso contrário, 415).
- **Proxy CORS.** `/api/cors-proxy` recusa os pedidos que um navegador abriria como página, frame ou worker
  (403) e devolve todas as respostas como conteúdo inerte: não pode executar scripts nem definir cookies no endereço do servidor.
- **Listagens do router.** No modo router, `status.args` e `status.preset` de `/api/models` mostram `<redacted>`
  em vez dos valores de `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  e `--mcp-servers-json`; o registo faz o mesmo. Os processos dos modelos continuam a receber os valores reais.
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
- **Multimédia descodificada.** Imagens até 32768 píxeis por lado e 100 megapíxeis, áudio até uma hora, fotogramas de vídeo até
  8192 píxeis por lado e 2 GiB de fotogramas descodificados por vídeo, e no máximo 2 GiB de imagens e áudio descodificados por
  pedido.
- **Modelos de chat do modelo.** `range()` produz no máximo um milhão de elementos, as cadeias construídas ficam abaixo de
  256 MiB, a indentação de `indent()` e `tojson()` abaixo de 256 espaços e o aninhamento abaixo de 1024 instruções; um modelo de chat
  que os ultrapasse falha com um erro de modelo de chat.
- **RPC.** Numa compilação com o backend RPC, o servidor RPC verifica os tamanhos dos frames e os limites dos tensores enviados pelos seus
  pares e fecha apenas a ligação infratora. Exponha-o apenas a máquinas de confiança: continua a servir um cliente
  de cada vez.
- **Interface.** As ligações nas mensagens mantêm apenas os esquemas http, https, mailto e tel (ou endereços relativos), e
  os anexos de imagem têm de ser imagens incorporadas. A importação de conversas aceita ficheiros até 512 MB, arquivos ZIP com até
  10 000 entradas, 256 MB por ficheiro JSONL e 1 GB extraído no total. Um ficheiro com registos malformados, ou com uma
  mensagem que pertença a outra conversa, é rejeitado por inteiro; uma conversa cujas mensagens já existam
  é ignorada e comunicada.

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

O pré-carregamento de especialistas usa por predefinição três posições na VRAM. Cada posição tem de comportar o maior
tensor de especialista transferido. Os valores 2–8 em `GGML_SCHED_PREFETCH_EXPERTS` escolhem
explicitamente o número de posições. Se a reserva de memória falhar, são mantidas pelo menos duas posições reservadas
quando possível; caso contrário, são usadas as transferências normais. O pré-carregamento só começa
quando o lote tem pelo menos o dobro de seleções de especialistas do que especialistas. O seu pool
pertence à primeira GPU que o ativa; as outras GPU mantêm cópias seletivas.
Os lotes maiores podem ajudar no preenchimento prévio, mas exigem mais memória.

Para diagnóstico, defina `GGML_CUDA_REGISTER_HOST=0` ou
`GGML_SCHED_PREFETCH_EXPERTS=0` para desativar a otimização correspondente.
Não defina essas variáveis durante o funcionamento normal, a não ser que pretenda prescindir delas intencionalmente.

O MTP automático examina os metadados do GGUF e os nomes dos tensores antes da reserva dos pesos,
incluindo todos os fragmentos. Exige uma arquitetura suportada, um tronco do modelo e
todas as cabeças integradas. Uma cabeça externa ou uma arquitetura não suportada não
o ativa. O limite predefinido de rascunho é de três tokens. A seleção explícita com `--spec-type`
tem prioridade; `--no-spec-mtp-auto` e `--spec-type none` desativam
a seleção automática. `--spec-draft-n-max` altera o limite. O MTP afeta a geração,
não o preenchimento prévio nem, diretamente, o tempo até ao primeiro token. A melhoria depende da taxa de aceitação
e do hardware; aumentar o limite não garante mais velocidade.

<a id="mtp-vram"></a>
Os pesos MTP ficam sempre na memória da GPU quando esta os consegue comportar: as camadas MTP injetadas no modelo, um
GGUF de MTP separado passado com `--spec-draft-model` e a cabeça de saída que usam. As opções que movem pesos para a
memória do sistema (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, os seus equivalentes `--spec-draft-*` e o ajuste
automático) não se aplicam a eles, e ficam na GPU mesmo com um `--gpu-layers` baixo. O ajuste automático coloca o
resto do modelo à sua volta, e a [cache de especialistas MoE](#expert-cache) usa a memória que sobra depois disso. Se as camadas
MTP não couberem na memória livre da GPU com uma margem de 512 MiB, o registo mostra um aviso e são colocadas como qualquer
outra camada. Com um GGUF de MTP separado, as camadas MTP injetadas no modelo principal não são carregadas, para que não ocupem
memória duas vezes; com `--spec-type none`, não é carregado nenhum peso MTP. O registo de carregamento confirma a colocação com uma linha
como `keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)`, visível com `-lv 4`.

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
`--n-cpu-moe N` mantém algumas camadas estáticas na GPU. Os especialistas mantidos na RAM são fixados, pelo que a RAM tem de comportar
todos eles mais o sistema operativo.

<a id="context-size"></a>
Quando o modelo não cabe na VRAM e `-c` não é indicado, o ajuste automático também escolhe o tamanho do contexto: a
cache KV ocupa no máximo um quarto da VRAM que sobra depois dos pesos densos, até ao contexto de treino do
modelo, e os especialistas (a cache ou as camadas de especialistas inteiras) ficam com o resto. Anteriormente, estes modelos recebiam sempre o
mínimo de 4096 tokens partilhados por todos os slots, o que cortava as respostas longas (`truncated = 1` no registo). O
registo mostra o resultado numa linha como `context size raised from 4096 to N tokens` e em `n_ctx_slot` da
linha `initializing`. `-c N` define o contexto explicitamente (mais contexto deixa menos VRAM para a cache e reduz a
velocidade de geração: 35,8 tokens/s com 4096 tokens e 33,9 com 122624 na máquina de referência) e
`--fit-ctx N` aumenta o mínimo. O contexto MTP adota o mesmo tamanho que o contexto do modelo, em vez de reservar
VRAM para todo o contexto de treino.

Antes do contexto, o ajuste automático escolhe o microlote (`-ub`), o número de tokens do prompt processados numa
só passagem. Com os especialistas na RAM, cada passagem carrega para a GPU os especialistas que não estão na cache, pelo que passagens
menos numerosas e maiores processam os prompts longos muito mais depressa, sem que a geração se altere. Salvo se `-b` ou `-ub` for indicado,
escolhe o maior de 4096, 2048 e 1024 tokens cujos buffers adicionais caibam num oitavo da VRAM que sobra depois dos
pesos densos, e aumenta o lote (`-b`) em conformidade. Na máquina de referência, um prompt de 2514 tokens passou de 214 para
755 tokens/s com 4096, à mesma velocidade de geração; os buffers maiores encurtam o contexto predefinido (cerca de 90000
tokens sem MTP e 61000 com MTP em 16 GB). O registo mostra a escolha numa linha como `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

| Opção | Variável | Efeito |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE` (`0` desativa) | Ativa ou desativa a cache (ativada por predefinição). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | Tamanho total fixo da cache, repartido entre as GPU segundo os especialistas que cada uma guarda em cache; `-1` (predefinição) usa a VRAM livre de todas as GPU. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | VRAM deixada livre em cada GPU para além da cache e das posições de pré-carregamento (predefinição: 1024). Aumente-a quando outros programas usarem a GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnóstico: parte fixa (0–1) dos especialistas em falta que é carregada, em vez do equilíbrio medido. |
| — | `MGS_MOE_CACHE_STATS` | Diagnóstico: regista os contadores da cache a cada N passos de descodificação (visíveis com `-lv 4`). |

`GET /api/expert-cache` devolve o tamanho, as camadas e as posições em cache, os acertos e as falhas, os carregamentos, os especialistas na CPU, a
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
ser executado. Numa máquina com SMT (duas threads por núcleo), isso corresponde ao número de núcleos físicos, como antes; numa
máquina virtual que não mostre SMT, é metade das CPU virtuais. Na máquina de referência (16 núcleos
virtuais com uma sessão de ambiente de trabalho, Qwen3.6-35B-A3B Q8_0, sem MTP), 4 a 12 threads geraram à mesma velocidade
(32,9–33,9 tokens/s) e 16 threads a 28,7 tokens/s. O registo de arranque mostra a escolha numa linha como
`using 8 CPU threads for generation and 8 for batches`; `-t` e `-tb` definem-nas explicitamente.

<a id="checkpoints"></a>
### Pontos de controlo de contexto

Os modelos com camadas recorrentes ou de janela deslizante (por exemplo, os híbridos Qwen3.5/3.6/3.8, o Nemotron-H ou o Granite 4)
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

Os GGUF completos do GLM-4.5-Air (`glm4moe`) com uma única cabeça podem agora ativar o MTP automático. As outras verificações de elegibilidade existentes e a prioridade das opções explícitas continuam a aplicar-se. O MTP separa os grafos com e sem saídas, mantém a ordem original dos tokens entre sequências simultâneas e usa as posições reais depois das imagens.

O ajuste automático conta as camadas nextn (MTP), pelo que a primeira camada deixa de ficar na CPU quando o MTP está desativado. Os pesos MTP ficam agora sempre na memória da GPU; consulte [colocação do MTP](#mtp-vram). O `--checkpoint-min-step` predefinido passa a ser 1024 em vez de 8192; consulte [pontos de controlo de contexto](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` e `--moe-cache-reserve` controlam a [cache de especialistas MoE](#expert-cache). O ajuste automático já não deixa os modelos maiores do que a VRAM com um contexto de 4096 tokens e passa também a escolher o microlote; consulte [tamanho do contexto](#context-size). O número predefinido de threads da CPU é agora, no máximo, metade das CPU lógicas; consulte [threads da CPU](#threads). `-np` passa a ter por predefinição um único slot com o contexto inteiro, em vez do modo automático com quatro slots a partilhar uma cache KV unificada, e `--kv-unified` está desativado por predefinição; consulte [fila de pedidos](#request-queue).

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
| Variáveis de ambiente, opções do CMake e símbolos da API C com o antigo prefixo do upstream, incluindo os aliases que versões anteriores já tinham deixado de usar | O nome atual com o prefixo `MGS_` (`mgs_` na API C), por exemplo `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` em vez de uma variante `NO_`, `MGS_ARG_LOAD_MODE` em vez das variáveis de mapeamento de memória, de bloqueio e de Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` e `MGS_ARG_SPEC_DRAFT_N_MIN` para os limites de rascunho, `MGS_CACHE` ou `MGS_BUILD_UI`; para as opções dos backends, a opção `GGML_*` correspondente, por exemplo `GGML_CUDA` ou `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` ou `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` ou `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valor `deepseek-legacy` de `reasoning_format` (campo do pedido e `--reasoning-format`) | `deepseek` ou `auto`; o raciocínio é devolvido em `reasoning_content` |
| Campo do pedido `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (array de cadeias) |
| `reasoning_in_content` em `generation_settings` | Nada; o campo foi removido. |
| Transferência de uma interface pré-compilada do upstream a partir de um bucket do Hugging Face (as opções do CMake `USE_PREBUILT_UI` e `UI_HF_BUCKET` e a variável de ambiente `HF_UI_VERSION`) | Nada; a interface é sempre compilada a partir de `frontend/` (`MGS_BUILD_UI=ON`) ou deixada de fora. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Número de compilação e commit: as variáveis do CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (também em `mgs-config.cmake`) e a cadeia `b0-unknown` de `build_info`, `system_fingerprint` e do `User-Agent` | `MGS_VERSION`, definida em `backend/cmake/build-info.cmake`; esses campos contêm agora a versão, como `0.1.0` |
| Modelos Llama da Meta (as arquiteturas GGUF `llama`, `llama4` e `llama-embed` e o projetor de visão do Llama 4) e os modelos de chat integrados `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` e `llama4` | Nada; use um dos modelos MoE suportados. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Opção do instalador `--no-sudo` | Nada; os instaladores são sempre executados como root, sem sudo. |
| Funções e campos que as API C tinham marcado como obsoletos: em `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, os aliases `mgs_token_*` (por exemplo `mgs_token_bos`, `mgs_token_get_text` ou `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, o campo `defrag_thold` de `mgs_context_params` e `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; em `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` e `ggml_upscale_ext`; em `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` e o campo `image_marker` | A função atual: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` em vez de `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` e `media_marker`. As execuções de aquecimento são feitas por quem chama a função, e `defrag_thold` não tinha qualquer efeito. |
| Gemma 4: as arquiteturas `gemma4` e `gemma4-assistant`, os projetores `gemma4v`, `gemma4uv`, `gemma4a` e `gemma4ua`, o formato de chat `peg-gemma4`, o tokenizador e o pré-tokenizador `gemma4`, e as predefinições `--vision-gemma-4b-default` e `--vision-gemma-12b-default` | Nada; use uma das [famílias suportadas](#supported-models). |
| Modelos DeepSeek anteriores à V4.1: as arquiteturas `deepseek`, `deepseek2` (também os ficheiros Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 e Youtu convertidos com ela), `deepseek2-ocr` (com os projetores `deepseekocr` e `deepseekocr2`), `deepseek32`, `mistral4` e `deepseek4`, os pré-tokenizadores `deepseek-llm` e `deepseek-coder` e o backbone de rascunho DSpark do DeepSeek-V4 | DeepSeek-V4.1-Flash (`deepseek41`). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) e os modelos de chat integrados do motor sem Jinja (`--chat-template` com um nome como `deepseek3`, `chatglm4` ou `mistral-v7-tekken`) | Nada; os modelos de chat passam sempre pelo Jinja. `--chat-template chatml` continua a selecionar o ChatML; para outro modelo de chat, passe o seu código-fonte Jinja com `--chat-template` ou `--chat-template-file`. Os pedidos com `tools` já não precisam de nenhuma opção. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| Segundos nomes longos de uma opção: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk` (as chaves das predefinições do router seguem os mesmos nomes) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, com o nome das respetivas opções |
| Campo do pedido `thinking_budget_tokens` | `reasoning_budget_tokens` (os clientes Anthropic continuam a enviar `thinking.budget_tokens`) |
| `content` em `POST /api/embeddings` e `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, a struct `mgs_chat_message` e `MGS_FILE_MAGIC_GGLA` em `mgs.h`; `mtmd_image_tokens_get_id` em `mtmd.h` | Nada; a formatação do chat é feita pelos modelos Jinja de `common/chat.h`, e os restantes não tinham qualquer utilidade. |

O CMake indica as variáveis `-D` removidas como não utilizadas em vez de as aplicar, pelo que deve verificar
a saída da configuração das compilações manuais.

A interface web mantém as suas definições no servidor (`~/.mgs/config/ui-settings.json`),
e no navegador apenas a sua chave de API (`MgsUi.apiKey` em `localStorage`) e as suas
conversas (a base de dados IndexedDB `MgsUi`). Não lê nem migra as
definições que as versões anteriores guardavam em `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, os modelos favoritos, as listas de ferramentas, o esforço de raciocínio,
as recomendações MCP dispensadas e o idioma), pelo que as definições começam com
os seus valores predefinidos e a chave de API tem de ser introduzida novamente uma vez. Também não lê nem
migra os dados guardados por versões anteriores com outros nomes, a chave `theme`
autónoma, os marcadores de raciocínio e de chamadas a ferramentas embutidos nas mensagens antigas, os anexos
de texto colado do antigo tipo `context`, o antigo formato de lista das recomendações MCP
dispensadas nem o antigo campo `webui_settings` do servidor. Para manter as
conversas de uma versão anterior, exporte-as a partir dela antes de atualizar e importe
a exportação depois. A importação de conversas só aceita as exportações JSONL e ZIP, dentro dos limites indicados
em [limites de segurança](#security-limits); o antigo formato de exportação JSON é rejeitado. As definições antigas ficam no navegador até serem eliminadas: para
as remover, abra a consola do navegador (F12) na página da interface e execute
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
uma vez para cada endereço e porta com que a interface foi aberta.

Os ficheiros transferidos com `-mu`/`--model-url` são guardados em cache em `~/.cache/mgs/`, pelo que os
que as versões anteriores guardaram na sua antiga pasta de cache são transferidos novamente.
As transferências do Hugging Face continuam a usar a cache do Hugging Face (`~/.cache/huggingface/hub`,
ou a pasta indicada em `MGS_CACHE`, `HF_HUB_CACHE` ou `HF_HOME`).

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

- Verifique que versão está instalada com `mgs --version` (mostra, por exemplo, `version: 0.1.0`); a compilação local
  executa-o no final e o CMake mostra `mgs version: 0.1.0` durante a configuração. A interface mostra a mesma versão
  em Definições → Visualização → **Mostrar informações da versão de compilação** (canto inferior direito) e em **Informações de compilação** na
  janela de informações do modelo. Cada alteração ao projeto incrementa o último número: 0.1.0, 0.1.1, 0.1.2, etc.
- Repita `curl | bash` para atualizar a partir de `main`, ou o comando de instalação local adequado para voltar a compilar e substituir o binário.
  Reinicie depois o serviço: `systemctl restart moe-gguf-server` no Debian,
  ou `rc-service moe-gguf-server restart` no Alpine.
- Leia `/root/webapp-install.log`, `journalctl -u moe-gguf-server` no Debian,
  ou `/var/log/moe-gguf-server.log` no Alpine. Os registos do Apache estão em
  `/var/www/DOMAIN-logs/`.
- Se a compilação CUDA não estiver disponível, instale o Toolkit externamente ou use
  `--cpu`. Se o NCCL não for encontrado, verifique a disponibilidade do repositório e a versão do CUDA,
  ou escolha explicitamente `--no-nccl`.
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

Segundo as regras deste projeto, o assistente de programação não executa testes, implementações, benchmarks de GPU nem verificações
automatizadas em navegadores móveis. As verificações estáticas de sintaxe
não garantem o comportamento em tempo de execução. Consulte [CODE.pt-PT.md](CODE.pt-PT.md) para o
mapa da implementação e os pontos de extensão.

<a id="license"></a>
## Licença

O `mgs` é distribuído ao abrigo da [PolyForm NonCommercial License 1.0.0](../LICENSE). Pode usá-lo,
modificá-lo e partilhá-lo para qualquer fim não comercial: utilização pessoal, estudo, investigação, projetos de passatempo,
e utilização por organizações de beneficência, instituições de ensino, organizações públicas de investigação e
instituições governamentais. A utilização comercial, como a de uma empresa que o use para o seu negócio, exige uma
licença separada concedida pelo nipegun.

A licença abrange apenas as contribuições do nipegun. As partes que provêm do llama.cpp (MIT, The
ggml authors), do FreeToken (Apache 2.0) e das bibliotecas incluídas mantêm as suas próprias licenças; os respetivos
avisos estão em `build/licenses/` e a compilação local instala-os em
`~/.local/share/doc/mgs/licenses/`.
