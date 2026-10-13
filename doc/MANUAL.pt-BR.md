# Manual do usuário

[Deutsch (Deutschland)](MANUAL.de-DE.md) · [English (United Kingdom)](MANUAL.en-GB.md) · [English (United States)](MANUAL.md) · [Español (Argentina)](MANUAL.es-AR.md) · [Español (España)](MANUAL.es-ES.md) · [Français (France)](MANUAL.fr-FR.md) · [עברית (ישראל)](MANUAL.he-IL.md) · [हिन्दी (भारत)](MANUAL.hi-IN.md) · [Italiano (Italia)](MANUAL.it-IT.md) · [日本語 (日本)](MANUAL.ja-JP.md) · [한국어(대한민국)](MANUAL.ko-KR.md) · [Português (Brasil)](MANUAL.pt-BR.md) · [Português (Portugal)](MANUAL.pt-PT.md) · [Русский (Россия)](MANUAL.ru-RU.md) · [中文（中国）](MANUAL.zh-CN.md)

Conteúdo: [instalação](#installation), [compilação local](#local-build), [implantação](#deployment),
[chat](#chat), [monitor](#monitor), [modelos e ferramentas](#models), [modelos compatíveis](#supported-models), [API](#api),
[fila de solicitações](#request-queue), [limites de segurança](#security-limits), [Vulkan](#vulkan), [desempenho](#performance), [compatibilidade removida](#removed-compatibility),
[manutenção](#maintenance), [licença](#license).

<a id="installation"></a>
## Instalação e primeira execução

Instale como root no host de produção. No Debian/Ubuntu, o instalador pode ser
passado diretamente pela entrada padrão:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash
```

Não é preciso baixar nenhum arquivo complementar antes. O script de entrada baixa
o arquivo completo do código-fonte da `main` [deste repositório](https://github.com/nipegun/moe-gguf-server/),
verifica-o e sincroniza uma cópia gerenciada em `/opt/moe-gguf-server-source`. Os
downloads e os caches de compilação ficam em `_/temp/` do projeto. O curl e os
certificados de CA são necessários para obter o script de entrada; as dependências
de inicialização que estiverem faltando são então instaladas automaticamente.

Para escolher outro diretório de código-fonte e passar opções de compilação, use:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | MGS_INSTALL_SOURCE_DIR=/srv/moe-gguf-server-source bash -s -- --cpu --jobs 4
```

O diretório do modo remoto deve ser absoluto e estar em um disco local. O instalador
só adota um diretório vazio ou reutiliza uma cópia que tenha a sua marca de origem
do código-fonte; diretórios não relacionados são rejeitados. Um bloqueio impede
instalações simultâneas nessa cópia. As execuções seguintes de `curl | bash`
sincronizam os arquivos do código-fonte, inclusive removendo os arquivos que saíram
da `main`, mas preservam `_/` e os caches de compilação. As chaves de API, os modelos
e os serviços implantados ficam fora dessa árvore de código-fonte e não são alterados.
Uma falha no download ou na compilação não instala um novo executável; execute o
mesmo comando novamente para tentar de novo. Nenhuma operação do Git é necessária.

Um script baixado de forma avulsa usa o mesmo modo remoto. Executar
`bash deploy/install-update-reinstall-debian.sh` a partir de uma cópia local completa
compila essa cópia sem baixar o código-fonte. Os argumentos depois de `bash -s --`
são repassados ao instalador de compilação. `--help` não exige root nem o download
do código-fonte. `--no-system-deps` também desativa a instalação dos pacotes de
inicialização: curl, certificados de CA, tar, gzip, rsync, util-linux e coreutils já
devem estar presentes. Os subprocessos do instalador não podem consumir o script
pelo pipe; todas as fases compartilham um único fluxo de saída, `/root/webapp-install.log`.

A configuração web continua sendo uma etapa separada, feita com
`/opt/moe-gguf-server-source/deploy/configure-web.sh` ou com o seu equivalente no
diretório de código-fonte escolhido.

O Alpine continua usando uma cópia local completa e
`bash deploy/install-update-reinstall-alpine.sh`, com o Bash instalado antes.
No Debian, o padrão é CUDA/NCCL; no Alpine, o padrão é CPU e o caminho de instalação
com CUDA/NCCL não é suportado. Use `--cpu` para uma compilação para CPU no Debian ou `--vulkan` para GPUs AMD, Intel ou NVIDIA
por meio do Vulkan ([Vulkan](#vulkan)). O CUDA Toolkit e os drivers da NVIDIA já devem estar instalados. `--no-nccl` desativa o NCCL explicitamente;
`--no-system-deps` deixa o gerenciamento de pacotes a cargo do administrador.

A interface local precisa do Node.js 20.19+ na série 20.x ou 22.12+. `--no-ui` a omite. Outras
opções são `--jobs N`, `--prefix PATH`, `--build-dir PATH`,
`--cuda-architectures LIST`, `--no-openssl` e `--build-ui`.
`MGS_BUILD_DIR`, `MGS_SOURCE_STAGE_DIR`, `MGS_INSTALL_PREFIX`,
`MGS_BUILD_JOBS`, `MGS_BUILD_BACKEND` e `CUDACXX` permitem substituir os valores
correspondentes por meio de variáveis de ambiente. Os diretórios de compilação e
de preparação devem ser subdiretórios distintos, sem sobreposição, do diretório
`_/temp/` do projeto.

O instalador adiciona os pacotes do NCCL correspondentes e o repositório da NVIDIA
quando necessário, mas nunca instala nem atualiza o CUDA Toolkit ou os drivers da GPU.
Ele vincula as bibliotecas internas de forma estática e instala `/usr/local/bin/mgs`.
Nenhum dos instaladores usa ou instala o sudo.

No Debian/Ubuntu, `libnccl2` e `libnccl-dev` são instalados juntos, exatamente na
mesma versão selecionada para a versão do CUDA detectada. Esse comando do APT usa
`--allow-downgrades` para que um pacote do NCCL mais recente já instalado não impeça
a instalação da versão selecionada. A opção só é passada ao comando de instalação do NCCL.

Se a distribuição trouxer um Node.js mais antigo, instale antes uma versão compatível.
A interface é sempre compilada localmente; a compilação não pode baixar os arquivos
pré-compilados do upstream, que não têm as traduções nem as rotas de API deste projeto.
Uma falha na compilação da interface interrompe a instalação.

A partir de uma cópia local completa, escolha uma compilação para CPU ou liste as
opções de instalação:

```bash
bash deploy/install-update-reinstall-debian.sh --cpu
bash deploy/install-update-reinstall-debian.sh --help
```

No Alpine, instale o Bash com `apk add --no-cache bash` antes de executar o instalador.

Execute um único modelo com:

```bash
/usr/local/bin/mgs --model /path/to/model.gguf
```

Nenhuma outra opção é necessária: o `mgs` escolhe as configurações que dão a maior
velocidade de inferência na máquina em que está rodando. O ajuste automático coloca
o modelo na GPU e, quando a memória do sistema permite, mantém os especialistas do
MoE na RAM para que a VRAM livre se torne o [cache de especialistas do MoE](#expert-cache);
ele também escolhe o [micro-batch e o tamanho do contexto](#context-size). O número
de threads da CPU acompanha o processador ([threads da CPU](#threads)), e as camadas
MTP encontradas no GGUF são ativadas e mantidas na VRAM ([MTP](#mtp-vram)). Qualquer
opção passada explicitamente (`-t`, `-b`, `-ub`, `-c`, `--gpu-layers`, `--n-cpu-moe`,
`--override-tensor`…) prevalece sobre a escolha automática. O log de inicialização mostra o que foi escolhido.

Abra a URL HTTPS exibida na inicialização, normalmente `https://127.0.0.1:11443/`.
Carregar um modelo grande pode levar algum tempo. A interface web e a documentação
da API continuam disponíveis enquanto as solicitações ao modelo informam que ele
ainda está carregando. `--help` lista as opções herdadas do motor. Os arquivos GGUF,
a memória da GPU, o tamanho do contexto e os projetores específicos de cada modelo
devem ser adequados à carga de trabalho pretendida.

O código do servidor ativa o HTTPS por padrão em todos os métodos de compilação e
instalação. Ele ocupa a primeira porta HTTPS disponível a partir de 11443 e, depois,
uma porta HTTP de redirecionamento a partir de 11080, pulando a porta HTTPS escolhida.
O HTTP responde com um redirecionamento permanente 308 para a porta HTTPS real,
preservando o host, o caminho e a query string solicitados. O log mostra as URLs
selecionadas; os conflitos de porta fazem cada sequência avançar de forma independente.
`--port N` fixa a porta principal e falha se ela estiver ocupada; `--port 0` deixa o
sistema operacional escolhê-la. `--http-port N` muda o início da busca pela porta de
redirecionamento. As variáveis de ambiente correspondentes são `MGS_ARG_PORT` e `MGS_ARG_HTTP_PORT`.

Sem arquivos de certificado, o executável cria um certificado autoassinado e a sua
chave privada na primeira vez que é iniciado, sem precisar de um comando `openssl`
externo, e os salva em `~/.mgs/certificates/` (`fullchain.pem` e
`privkey.pem`, com a chave legível apenas por você). Todas as execuções seguintes, e
todas as novas compilações, os reutilizam, de modo que o navegador mantém a sua
exceção de confiança e não precisa ser recarregado; um novo par só é criado quando
esses arquivos estão faltando, não correspondem entre si ou o certificado expirou
(após 365 dias). O certificado cobre `localhost`, os endereços de loopback, o host
passado com `--host` e o nome da máquina e os endereços de rede que ela tinha quando
ele foi criado; para acabar de vez com o aviso do navegador, importe
`~/.mgs/certificates/fullchain.pem` como confiável, e apague os dois arquivos para
criar um novo. Se não for possível gravar nessa pasta, cada execução gera o seu
próprio certificado na memória. Informe tanto `--ssl-cert-file /path/fullchain.pem` quanto `--ssl-key-file
/path/privkey.pem` para usar um certificado em que os seus clientes confiem. Um par
ausente ou inválido interrompe a inicialização. As compilações feitas com `--no-openssl`
não podem servir HTTPS e exigem `--http` explícito (ou `MGS_ARG_HTTPS=false`); o HTTP
simples começa a sua busca de porta em 11080. Os processos filhos do roteador usam
HTTP explícito em portas de loopback privadas. Os sockets Unix e o
`AIP_MODE=PREDICTION` do Vertex AI mantêm o seu transporte HTTP.

A configuração do CMake fica em `build/CMakeLists.txt`. Tanto os scripts de
instalação quanto os de compilação local incluem `build/` em sua cópia do código-fonte
e o selecionam automaticamente. Os instaladores redefinem a configuração gerada do
CMake se um cache existente apontar para outro diretório de código-fonte. As
compilações locais sempre começam em um diretório novo dentro de `/tmp/` e o removem
ao terminar. Para usar o CMake diretamente, execute
`cmake -S build -B _/temp/cmake` a partir da raiz do projeto; os comandos habituais
do instalador e da compilação local continuam os mesmos. O HTTPS usa o OpenSSL do
sistema por padrão (`MGS_OPENSSL=ON`); em uma compilação direta com o CMake,
`-DMGS_BUILD_BORINGSSL=ON` ou `-DMGS_BUILD_LIBRESSL=ON` baixa e compila o BoringSSL
ou o LibreSSL no lugar dele, e `-DMGS_OPENSSL=OFF` sem nenhum dos dois compila o
servidor sem HTTPS.

<a id="local-build"></a>
## Compilação local para uso manual

`build/build-local.sh` compila o projeto no seu próprio computador e instala uma
cópia autônoma para o seu usuário, que você inicia manualmente. Ele não instala
serviços, não deixa nada rodando em segundo plano, nunca usa root nem sudo e nunca
instala pacotes. Execute-o como seu usuário comum, a partir de qualquer pasta:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash
```

O comando de uma linha baixa, sem Git, o arquivo do código-fonte da `main` para a
pasta de compilação privada em `/tmp`. O arquivo e o código-fonte extraído são
removidos junto com os demais arquivos temporários da compilação ao terminar. Uma
cópia avulsa do script funciona da mesma forma. A partir de uma cópia local completa
do projeto, compile o seu código-fonte local com:

```bash
bash build/build-local.sh
```

Executar esse script pelo seu caminho absoluto também usa o projeto completo que o
contém. A mensagem `Using local sources: ...` confirma qual pasta está sendo
compilada, incluindo as traduções da interface. Se a execução terminar com
`Error: Compilation failed.`, nenhum executável novo foi instalado; uma instalação
existente mantém a interface anterior. Leia o primeiro erro do compilador acima de
`ninja: build stopped`, corrija a causa e execute o mesmo script novamente. Quando a
compilação e a instalação forem concluídas com sucesso, reinicie o `mgs` e recarregue
a interface web.

Erros como `std::vector` não ter um membro chamado `mPushBack` ou
`no declaration matches` em `subproc.cpp` indicam nomes inconsistentes no código-fonte.
Use os arquivos de código-fonte corrigidos e execute a compilação novamente. A
compilação da interface também exige que as importações da sua configuração
correspondam aos nomes exportados pelos módulos do código-fonte; uma exportação
ausente é informada pela etapa do npm/Vite.

Se o SvelteKit informar `Page options are ignored when router.type === 'hash'`
e mencionar `fLoad`, as rotas do chat devem exportar exatamente o nome `load`
do framework. As rotas corrigidas mantêm a validação da chave de API. A compilação
executa `svelte-kit sync` antes de gerar os recursos e compilar a interface, e
para se a sincronização falhar. Uma falha em `Provisioning UI assets` também
impede a instalação, mesmo quando as bibliotecas C++ já foram compiladas.

Para passar opções pelo comando de uma linha, use `bash -s --`; por exemplo, uma compilação para CPU:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --cpu --jobs 4
```

`--help` mostra as opções sem baixar o arquivo do código-fonte nem compilar.

O resultado é instalado para o seu usuário sob o prefixo `~/.local`:

| caminho | conteúdo |
| --- | --- |
| `~/.local/bin/mgs` | O programa que você executa: um pequeno inicializador estático que encontra `../lib/mgs/` em relação à sua própria localização e inicia o servidor por meio do carregador incluído. |
| `~/.local/lib/mgs/` | O binário do servidor (`mgs`, com a interface web e a documentação da API embutidas) e todas as bibliotecas compartilhadas de que ele precisa: o carregador dinâmico e a biblioteca C da glibc, o runtime de C++, o OpenMP, o OpenSSL, os módulos de serviço de nomes da glibc e, nas compilações com CUDA, o runtime do CUDA, o cuBLAS e o NCCL. |
| `~/.local/share/doc/mgs/licenses/` | A licença PolyForm NonCommercial 1.0.0 do projeto (`LICENSE`) e os avisos de licença de terceiros copiados de `build/licenses/` da árvore de código-fonte (llama.cpp, FreeToken, Swagger UI, cpp-httplib e nlohmann/json). |

Somente o usuário que executa o script pode usar essa instalação. Tudo é resolvido em relação ao local onde o inicializador
está: ele executa `lib/mgs/ld-linux-x86-64.so.2 lib/mgs/mgs` a partir do seu prefixo, e o servidor e as suas bibliotecas procuram
em `lib/mgs/` antes de qualquer outro lugar, inclusive `LD_LIBRARY_PATH`. A única biblioteca que ainda vem do sistema é o
driver da NVIDIA (`libcuda.so.1` e as bibliotecas `libnvidia-*` que ele carrega), porque ele precisa corresponder ao módulo
do kernel em execução; com `--vulkan`, também o driver Vulkan do sistema ([Vulkan](#vulkan)). No final, o script lista, por meio do carregador incluído, as bibliotecas que o servidor carregaria, falha se alguma
vier de fora de `lib/mgs/` e executa `--version` por meio do inicializador.

O Debian adiciona `~/.local/bin` ao `PATH` no login assim que essa pasta existe (por meio do `~/.profile`); assim, depois de
entrar de novo na sessão, você pode executar `mgs` pelo nome; o script exibe um lembrete enquanto ela não estiver no `PATH`.
Inicie-o quando quiser e pare-o com Ctrl+C:

```bash
mgs --model /path/model.gguf
```

O caminho completo também funciona: `~/.local/bin/mgs --model /path/model.gguf`.

Abra `https://127.0.0.1:11443/` para o chat e `https://127.0.0.1:11443/api/doc/` para
a documentação da API. Todas as opções deste manual se aplicam; por exemplo,
`--models-dir` ativa o roteador de vários modelos.

Requisitos, que você mesmo instala como root, porque o script apenas os verifica:
`cmake`, um compilador C/C++ e a biblioteca C estática (`build-essential`, que
traz junto `libc6-dev`), `patchelf`, `libssl-dev` (ou
`--no-openssl`), Node.js 20.19+/22.12+ com npm (ou `--no-ui`), um CUDA Toolkit já
instalado (ou `--cpu`) e o NCCL com os seus arquivos de desenvolvimento (`libnccl2` e
`libnccl-dev`, ou `--no-nccl`). O `nvcc` é obtido de `CUDACXX`, do `PATH` ou de
`/usr/local/cuda/bin/nvcc`. O Ninja é usado quando está instalado.
Os utilitários básicos incluem `tar`, `coreutils` e `util-linux`. As compilações
remotas também exigem `curl`, `ca-certificates` e `gzip`; nenhuma dependência é instalada automaticamente.

Opções: `--prefix PATH` (outro prefixo de instalação), `--jobs N`, `--cuda`, `--vulkan`, `--cpu`,
`--cuda-architectures LIST`, `--build-ui`, `--no-ui`, `--no-openssl`, `--no-nccl`
e `--help`.

É bom saber:

- A compilação é otimizada para a CPU deste computador e para as GPUs presentes
  durante a compilação. Compile de novo em cada computador em vez de copiar a instalação.
- A instalação pode ser movida como um prefixo: mova `bin/mgs` e `lib/mgs/` juntos, mantendo os seus níveis `bin/` e
  `lib/`. Inicie sempre `bin/mgs`; `lib/mgs/mgs` é interno e não deve ser executado diretamente, porque isso
  ignoraria o carregador incluído.
- Executar o script novamente atualiza a instalação. Ele substitui apenas `bin/mgs`, `lib/mgs/` e
  `share/doc/mgs/licenses/`. Uma cópia que já esteja rodando continua funcionando; reinicie-a para usar a nova compilação.
- O script recusa um prefixo em que `bin/mgs`, `lib/mgs/` ou essa pasta de licenças já existam, a menos que tenham
  sido criados por ele; ele deixa um marcador oculto `.moe-gguf-server-bundle` em `lib/mgs/`.
- `--prefix PATH` instala em outro prefixo em que o seu usuário possa gravar; ele não pode ser `/` nem uma pasta
  dentro do projeto.
- O código-fonte preparado, os arquivos de compilação, os caches, a instalação montada e o log usam uma
  pasta privada `/tmp/moe-gguf-server-build.XXXXXXXX/`. Depois de copiar o resultado
  para o prefixo, o script remove essa pasta. Ele também faz a limpeza em caso de erros
  e de interrupções tratadas. Cada execução compila do zero, sem gravar na
  árvore de código-fonte nem em `/opt/`. As compilações locais não aceitam `--build-dir`.
- O prefixo fica bloqueado enquanto os artefatos concluídos são copiados e substituídos,
  o que impede atualizações simultâneas da mesma instalação.
- Uma compilação com CUDA precisa do driver da NVIDIA para iniciar; use `--vulkan` em computadores com outra GPU e
  `--cpu` em computadores sem GPU.

<a id="vulkan"></a>
## Vulkan (GPUs AMD, Intel e NVIDIA)

Além de CUDA (o padrão) e CPU, tanto `build/build-local.sh` quanto o instalador do Debian compilam o backend Vulkan
com `--vulkan`, para GPUs sem CUDA, como as placas AMD Radeon. Primeiro, instale as dependências de compilação como root:

```bash
apt-get install build-essential cmake ninja-build patchelf libssl-dev nodejs npm curl ca-certificates \
  libvulkan-dev spirv-headers glslc mesa-vulkan-drivers
```

Depois, como seu usuário comum:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- --vulkan
```

O instalador do Debian aceita a mesma opção e instala esses pacotes do Vulkan por conta própria:

```bash
curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/deploy/install-update-reinstall-debian.sh | bash -s -- --vulkan
```

`mgs --list-devices` mostra a GPU, por exemplo `Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10)`.

- A instalação local inclui o carregador do Vulkan (`libvulkan.so.1`) como qualquer outra biblioteca, mas o driver
  Vulkan vem do sistema: o carregador lê os drivers registrados em `/usr/share/vulkan/icd.d/` (Mesa RADV para AMD,
  ANV para Intel, o driver da NVIDIA) e eles carregam as suas próprias bibliotecas do sistema, porque precisam
  corresponder ao kernel e ao sistema. Compile na mesma versão do Debian que vai executar o binário.
- O usuário que executa o `mgs` precisa do nó de renderização da GPU (`/dev/dri/renderD*`). Um usuário conectado
  ao ambiente de desktop já tem esse acesso; os outros usuários precisam do grupo `render` (e do `video`).
  `configure-web.sh` adiciona a conta do serviço aos dois.
- O MTP na VRAM, o cache de especialistas do MoE, a pré-busca de especialistas, o `--fit` e o Monitor funcionam
  como no CUDA. Na AMD, o Monitor lê o uso, a VRAM e a temperatura pelo sysfs.
- O `mgs` só conta como memória da GPU a VRAM real (os heaps de memória device-local), nunca o GTT. O GTT é a parte
  da RAM do sistema (normalmente metade dela) que uma GPU AMD ou Intel consegue mapear: não é memória adicional, e a
  GPU a lê pelo PCIe, muitas vezes mais devagar do que a sua VRAM. As camadas MTP, os pesos densos, o cache KV e o
  cache de especialistas são alocados na VRAM; os especialistas que não cabem ficam na RAM normal, onde a CPU os calcula.
- O `mgs` verifica onde o kernel realmente mantém essa memória (`/proc/self/fdinfo`, publicado pelo amdgpu, i915 e xe).
  Se outro programa encher a VRAM e o driver mover parte dela para o GTT, o log e o Monitor avisam, e o Monitor
  mostra a localização real (VRAM e GTT) de cada GPU AMD ou Intel. O cache de especialistas é alocado com uma
  prioridade de memória menor (`VK_EXT_memory_priority`), para que o driver o mova antes das camadas MTP, dos pesos
  ou do cache KV. Sem resizable BAR, os buffers do dispositivo também evitam a pequena janela de VRAM que a CPU
  consegue mapear (256 MiB em uma RX 580), que o desktop compartilha e que o driver esvazia para o GTT primeiro.
- Só o CUDA consegue fixar (pin) as páginas do modelo que ficam na RAM; por isso, com Vulkan, os especialistas chegam
  à GPU por meio de memória paginável. `--load-mode none`, em vez disso, os copia para memória do host fixada, o que,
  em uma RTX 4060 Ti com Vulkan (Qwen3.6-35B-A3B Q4_K), aumentou o envio dos especialistas de 2,7 para 13,1 GB/s e a
  geração de 27,5 para 36,6 tokens/s. Na AMD e na Intel, essa memória fixada conta como GTT: quando os especialistas
  não cabem nela, o carregamento falha com
  `radv/amdgpu: Not enough memory for command submission` (`ErrorDeviceLost`), como aconteceu com os 18,9 GB de
  especialistas desse modelo com 16,8 GB de GTT em uma RX 580. Use essa opção apenas quando os especialistas couberem
  no GTT e na RAM livre.
- Dimensione o modelo de acordo com a máquina: o GGUF inteiro deve caber na RAM livre mais a VRAM, descontando o
  cache KV e os buffers de computação. Com 8 GB de VRAM e 16 GB de RAM, isso significa cerca de 18–20 GB de pesos;
  arquivos maiores são lidos do disco repetidamente e a geração fica muito lenta. O Q4_K de 21,6 GB citado acima
  chegou a 7,8 tokens/s em uma RX 580 com 15 GB de RAM e a 15,9 tokens/s quando a máquina passou a ter 31 GB (10,0
  com `--spec-type none`).

<a id="deployment"></a>
## Implantação em produção

A partir do diretório de código-fonte (`/opt/moe-gguf-server-source` nas instalações
remotas), depois de instalar o binário, execute:

```bash
bash deploy/configure-web.sh --domain ai.example.com
```

O domínio deve resolver para o host de produção. O serviço `moe-gguf-server-web`
executa uma instância dedicada do Apache como `moe-gguf-server`, sem root.
A cada inicialização, ele seleciona a primeira porta HTTP disponível a partir de
11080 e a primeira porta HTTPS a partir de 11443. Cada sequência avança de forma
independente até 65535, sem atribuir a mesma porta aos dois protocolos. Ele
inspeciona os sockets TCP em escuta em IPv4/IPv6 e tenta de novo se o Apache detectar
que uma porta foi ocupada depois da seleção. As demais falhas de configuração
interrompem a inicialização em vez de serem confundidas com conflitos de porta.

O HTTP redireciona permanentemente para a porta HTTPS selecionada, preservando o
caminho e a query string. O redirecionamento não é armazenado em cache. O Apache
encaminha as solicitações para a inferência em `127.0.0.1:8080`, iniciada
explicitamente com `--http --port 8080` e que também roda como
`moe-gguf-server`. A instalação exige root;
nenhum dos serviços precisa rodar como root. Nenhum HAProxy é configurado e nenhum
site compartilhado do Apache é importado ou modificado.

As URLs reais são publicadas depois que os dois sockets de escuta são abertos:

```bash
cat /opt/moe-gguf-server/_/temp/web/ports.json
```

Por exemplo, HTTP 11081 e HTTPS 11445 redirecionam para
`https://ai.example.com:11445/`. O arquivo é removido no desligamento e gerado de
novo na inicialização; uma reinicialização pode escolher portas diferentes. Os
clientes, os firewalls e qualquer proxy externo devem usar os valores publicados.
A porta faz parte da [origem do navegador](https://developer.mozilla.org/en-US/docs/Glossary/Origin);
por isso, uma porta HTTPS diferente tem histórico e configurações locais separados.
As conversas existentes permanecem na sua origem original; a exportação/importação
pode transferi-las.

A conta do sistema não tem senha interativa. Os modelos são armazenados em
`/opt/moe-gguf-server/models`, e a configuração do programa fica na pasta pessoal da
conta do serviço, `/opt/moe-gguf-server/.mgs/config/`: a chave de API (`api.key`), o
domínio web (`web-domain`), o modelo de configuração do Apache (`apache2.conf.in`) e as
configurações da interface (`ui-settings.json`), todos de propriedade de `moe-gguf-server`
e com modo 0600. As credenciais e a saída da instalação são registradas em
`/root/webapp-credentials.txt` e `/root/webapp-install.log`, ambos com modo 600. A
reconfiguração preserva as chaves, os modelos e os certificados existentes. A chave
privada TLS pertence ao root e pode ser lida pelo grupo moe-gguf-server (modo 640),
o que permite que o processo web sem privilégios a leia.

Use o seu próprio certificado TLS com:

```bash
bash deploy/configure-web.sh --domain ai.example.com \
  --certificate /path/fullchain.pem --private-key /path/privkey.pem
```

O certificado fica em `/opt/moe-gguf-server/certificates/` (`fullchain.pem` e
`privkey.pem`, de propriedade da conta `moe-gguf-server`, com a chave legível apenas por ela).
Sem arquivos informados, o script mantém o certificado que já estiver lá ou, na
primeira vez, cria um autoassinado, que todas as reinstalações seguintes reutilizam;
assim, os navegadores mantêm a sua exceção de confiança. Confie nele no seu cliente
ou informe um certificado em que os seus clientes já confiem. Os arquivos informados
são copiados para essa pasta; a renovação do certificado deve atualizar essas cópias
e reiniciar o `moe-gguf-server-web`. O script não obtém nem renova certificados do Let's Encrypt.
Use o mesmo comando depois de renovar os certificados de origem. A opção
`--binary` do script permite usar um executável em um local diferente do padrão.

O serviço usa o modo roteador. Coloque modelos GGUF legíveis em
`/opt/moe-gguf-server/models` ou use o gerenciamento de modelos na interface. Para
opções de linha de comando especializadas, use um override do systemd no Debian/Ubuntu
ou a configuração do serviço OpenRC no Alpine. Mantenha o backend vinculado ao loopback.
O configurador web gera de novo o arquivo de serviço que gerencia quando é executado novamente.

Reinicie o serviço web para selecionar as portas outra vez:

```bash
systemctl restart moe-gguf-server-web
```

No Alpine, use `rc-service moe-gguf-server-web restart`. Os logs de inicialização
ficam em `journalctl -u moe-gguf-server-web` no Debian ou em
`/var/log/moe-gguf-server-web.log` no Alpine. Os diagnósticos do Apache ficam em
`/opt/moe-gguf-server/_/temp/web/startup.log` e `error.log`; os logs de acesso do
domínio continuam em `/var/www/DOMAIN-logs/`. O modelo de configuração autônomo é
`/opt/moe-gguf-server/.mgs/config/apache2.conf.in`. Não edite a configuração gerada
em `_/temp/web/`; ela é recriada a cada inicialização.

<a id="chat"></a>
## Usando a interface

1. Abra a URL do servidor e digite a chave de API dele quando for solicitada, ou em Configurações → Geral.
2. Selecione um modelo. No modo roteador, carregue um modelo disponível se nenhum estiver carregado.
3. Digite uma mensagem e envie. Por padrão, Enter envia; Shift+Enter insere uma quebra
   de linha. Esses atalhos podem ser alterados nas configurações.
4. Adicione arquivos com a ação de anexo. Imagens, áudio e vídeo exigem que o modelo
   suporte essas modalidades. Os arquivos PDF podem ser extraídos como texto ou
   convertidos em imagens; os modelos sem visão usam o caminho de texto.
5. Use as ações da mensagem para interromper a geração, pular o raciocínio quando houver
   suporte, editar uma mensagem, gerar uma resposta de novo ou criar uma ramificação da conversa.

A barra lateral permite pesquisar, fixar, renomear, excluir e selecionar várias conversas.
As conversas são armazenadas no navegador usando o IndexedDB; exporte-as antes de limpar
o armazenamento do navegador ou de trocar de perfil do navegador. As configurações são
mantidas pelo servidor em `~/.mgs/config/ui-settings.json` do usuário que executa o `mgs`
(no serviço implantado, `/opt/moe-gguf-server/.mgs/config/ui-settings.json`); assim,
todos os navegadores que abrem a interface as compartilham: parâmetros de geração,
opções de exibição, tema, idioma, servidores MCP, modelos favoritos, ferramentas
desativadas e sempre permitidas e o esforço de raciocínio padrão. Só a chave de API
fica em cada navegador, porque ela autentica a solicitação que carrega o resto; por
isso, digite-a uma vez em cada navegador. A importação/exportação está disponível nas
configurações. As exportações podem conter anexos e, se forem selecionadas explicitamente,
chaves de API confidenciais ou cabeçalhos de autorização de MCP.

Escolha o idioma em Configurações → Geral. As 15 opções estão na ordem `de-DE`,
`en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`,
`ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; o padrão é `en-US`.
O português do Brasil e o português europeu têm traduções separadas. O hebraico usa uma
interface da direita para a esquerda; a caixa de entrada do chat e as mensagens simples
do usuário seguem a direção do seu texto, enquanto o código e a saída do terminal são
exibidos da esquerda para a direita.
Mudar o idioma salva a escolha nas configurações do servidor e
recarrega a interface; o servidor a publica em `/api/config.js`, de modo que todos os
navegadores iniciam nesse idioma. Salve antes as outras configurações pendentes. O
parâmetro de query `lang` da URL o substitui naquela página e mantém a escolha quando
ela não pode ser salva. Por exemplo, acrescente `?lang=ja-JP` à URL da interface para
selecionar o japonês.

Configurações → Geral → **Responder sempre no idioma da interface** vem desativado
por padrão. Ative essa opção e salve as configurações para instruir o modelo a usar o
idioma e a variante regional da interface em todas as novas respostas, inclusive em
conversas existentes, respostas geradas de novo, continuações, turnos depois de chamadas
de ferramentas e títulos gerados pelo modelo. Para `es-AR`, a instrução pede
explicitamente o voseo argentino: “Seguí estos pasos”, “tenés” e “podés”; `es-ES` pede
o uso do espanhol da Espanha. O inglês britânico e o americano e o português do Brasil
e o europeu também recebem instruções regionais distintas.

A instrução acompanha cada nova solicitação de inferência e cada preenchimento prévio
do cache KV; ela é combinada com a mensagem do sistema enviada, sem editar o seu prompt
do sistema salvo nem a conversa. Mudar o idioma da interface afeta as novas
solicitações depois do recarregamento; uma resposta que já esteja sendo gerada mantém a
instrução que recebeu. Desativar a configuração faz com que ela deixe de ser adicionada
às solicitações seguintes. As respostas existentes não são traduzidas. O código, os
comandos, os nomes, as citações e as traduções pedidas explicitamente mantêm o idioma
necessário; as explicações ao redor deles seguem o idioma da interface. A capacidade
do modelo de seguir a instrução determina a consistência; isto não é uma tradução
automática nem uma garantia de cumprimento perfeito. Os clientes externos da API não
são afetados por essa configuração do navegador.

Escolha Sistema, Claro ou Escuro para o tema. As configurações de geração incluem o uso
do contexto, a temperatura, os amostradores, as penalidades e os limites de turnos do
modo agente. Alguns recursos dependem de opções do servidor ou do modelo selecionado.

Depois de cada resposta, a interface envia a conversa de novo, quando o servidor está
ocioso, para que ela já esteja codificada no cache KV enquanto você lê e o próximo
turno comece mais cedo. Isso vem ativado por padrão; desative Configurações →
Desenvolvimento → **Preencher cache KV após cada resposta** para interromper esse comportamento.

Enquanto o servidor está resolvendo outra solicitação, uma mensagem que você envia
espera na [fila](#request-queue), e a resposta dela mostra um aviso no lugar do texto de
progresso até começar: “Na fila: outra solicitação enviada diretamente à API está sendo
resolvida primeiro (solicitações à frente: N).” quando um programa usa a API ao mesmo
tempo, ou o aviso equivalente para uma solicitação da própria interface web (outra aba
ou outro navegador, um título gerado ou o preenchimento prévio do cache KV). O aviso
desaparece assim que a sua solicitação começa.

<a id="monitor"></a>
## Monitor e barra de status

O **Monitor** fica abaixo de **Configurações** na barra lateral (endereço `#/monitor`). Enquanto está aberto, ele
é atualizado a cada dois segundos e mostra:

- **Uso**: primeiro, um cartão por GPU da máquina (NVIDIA, AMD ou Intel, mesmo que nenhum modelo a use), intitulado
  com o seu rótulo em negrito e o seu nome (por exemplo, **gpu-1**: NVIDIA GeForce RTX 4060 Ti; com várias GPUs,
  gpu-1, gpu-2 e assim por diante), com o seu uso, o uso do controlador de memória (largura de banda), a VRAM em uso,
  a temperatura e o consumo de energia em relação ao seu limite; depois, um cartão por processador, intitulado da
  mesma forma com o seu modelo (por exemplo, **cpu-1**: AMD Ryzen 9 5950X 16-Core Processor; com vários
  processadores, cpu-1, cpu-2 e assim por diante), com o seu uso, uma pequena barra vertical por CPU lógica desse
  processador e a carga média do sistema. Todas as barras de uso são verdes e vão se enchendo de vermelho à medida
  que o uso aumenta; uma barra cujo valor é desconhecido fica cinza. As GPUs NVIDIA são lidas pela biblioteca NVML
  do driver e as GPUs AMD por meio de `gpu_busy_percent` no sysfs; qualquer outra GPU (Intel, ou NVIDIA sem NVML)
  não mostra uso, e o seu nome e a sua memória vêm do backend de inferência quando um modelo passa a usá-la.
- **Memória**: uma barra por GPU (VRAM) e uma para a RAM; uma GPU sem dados de memória, como uma GPU Intel que
  nenhum modelo usa ainda, não tem barra. Cada barra é dividida por finalidade, cada finalidade sempre com a mesma
  cor, e a sua legenda lista todos os segmentos com o tamanho e a porcentagem do total, um por linha e na mesma
  ordem da barra. Passe o mouse sobre um segmento para ver o tamanho dele.

  | Segmento | Conteúdo |
  | --- | --- |
  | Pesos MTP / do modelo de rascunho | Pesos das camadas MTP (nextn) do modelo, de um GGUF de MTP separado passado com `--spec-draft-model` ou de um modelo de rascunho. |
  | Pesos dos especialistas | Tensores dos especialistas roteados do modelo. |
  | Pesos densos | Todos os demais pesos: atenção, especialistas compartilhados, embeddings e saída. |
  | Cache KV | Memória de contexto do modelo (cache KV ou estado recorrente). |
  | Cache KV MTP / do modelo de rascunho | Memória de contexto do contexto MTP ou do modelo de rascunho. |
  | Buffers de cálculo | Buffers temporários dos grafos de computação. |
  | Cache dos especialistas e pré-busca | Cache de especialistas do MoE (pool, identificadores e área de preparação) e os slots de pré-busca de especialistas. |
  | Cache de prompts e pontos de controle | Prompts salvos com `--cache-ram` e os checkpoints de contexto dos slots (RAM). |
  | Outros | Memória usada que nenhum dos itens acima explica: outros processos, o desktop, contextos do driver, o projetor multimodal. |
  | Livre | O restante do dispositivo ou da RAM. |

  Os pesos mapeados a partir do arquivo do modelo contam como RAM mesmo quando o kernel informa esse cache de
  páginas como disponível; por isso, os segmentos podem somar mais do que a memória “em uso”, e a barra então não
  mostra “Outros” e mostra menos memória livre. O cache de especialistas aparece depois da primeira solicitação,
  quando é criado. No modo roteador, as barras incluem todos os modelos carregados, e o Monitor nunca carrega nem
  desperta um modelo.

A **barra de status** na parte inferior de todas as páginas mostra primeiro o uso de cada GPU e depois o de cada
processador físico, cada um na sua própria barra, com os mesmos rótulos dos cartões do Monitor: `gpu-1`, `gpu-2`... e
`cpu-1`, `cpu-2`... As barras têm o mesmo tamanho, são atualizadas a cada dois segundos e, como as do Monitor, são
verdes e vão se enchendo de vermelho à medida que o uso aumenta; o texto que aparece ao passar o mouse sobre cada uma
informa o nome do dispositivo. Sem GPU, ela mostra uma barra `GPU` vazia, e, quando as barras não cabem em uma tela
estreita, a barra de status rola para o lado. Clique nela para abrir o Monitor. Ela continua funcionando enquanto o
modelo carrega. Ela é exibida por padrão; desative Configurações → Exibição → **Mostrar barra de status** para ocultá-la.

<a id="models"></a>
## Modelos, ferramentas e MCP

O `mgs` aceita apenas modelos principais MoE. Modelos densos, como GPT-2 ou Phi-2, são rejeitados antes que os seus pesos sejam carregados; as implementações dos decodificadores deles não são compiladas. As camadas densas dentro de um modelo MoE e uma cabeça somente de MTP compatível e informada explicitamente são mantidas.

<a id="supported-models"></a>
### Famílias de modelos compatíveis

| Família | Arquitetura GGUF | Observações |
| --- | --- | --- |
| Qwen3.6 | `qwen35moe` | MTP quando o GGUF o contém. |
| Qwen3.8-Flash-Next | `qwen4exp` | MTP quando o GGUF o contém. A sua tabela de embeddings por camada (cerca de 29 GB) é lida do arquivo do modelo sob demanda; por isso, mantenha o arquivo em um disco rápido. Os contextos longos são calculados com atenção densa sob a seleção esparsa: o resultado é correto, mas sem o ganho de velocidade dos kernels esparsos. |
| GLM-5.3 | `glm-dsa` | MTP. |
| GLM-5.3-Flash | `glm5-next` | MTP; imagens com o seu mmproj `glm5v`. |
| Kimi-K3 | `kimi-k3` | Os GGUFs publicados não contêm MTP. Com Vulkan, a sua mistura residual roda na CPU. |
| DeepSeek-V4.1-Flash | `deepseek41` | Carrega os dois formatos publicados: os arquivos DwarfStar de antirez (como `antirez/deepseek-v4.1-flash-gguf`) e os arquivos do ecossistema do llama.cpp. A sua tabela Engram (cerca de 188 GiB dentro do arquivo) nunca é carregada: cada token lê as linhas dela no arquivo do modelo; por isso, o arquivo precisa continuar disponível enquanto o `mgs` estiver rodando, de preferência em um SSD, e é necessário Linux ou outro sistema POSIX. Os arquivos sem template de chat usam um template integrado, e `reasoning_effort` aceita um número de 1 a 100. Sem MTP e sem imagens. |
| MiniMax-M3, Nemotron-3, MiMo-V2.6 | `minimax-m3`, `nemotron_h_moe`, `mimo2` | |

O Gemma 4 e os modelos DeepSeek anteriores ao V4.1 não são mais compatíveis (veja [compatibilidade removida](#removed-compatibility)).

Com várias GPUs selecionadas, o MTP verifica todas elas antes de recorrer à RAM. A sua estimativa de memória inclui a cabeça de saída. Os buffers do MTP integrado são carregados antes dos demais pesos; um GGUF de MTP externo passado com `--spec-draft-model` é carregado antes do modelo principal, também com `--fit off`. A seleção explícita atual de dispositivos continua determinando quais GPUs estão disponíveis.

Use `--reasoning on` ou `--reasoning off` para controlar o raciocínio. Passar `enable_thinking` por meio de `--chat-template-kwargs` é rejeitado. Use os nomes de opção documentados, com hífen; as grafias com sublinhado são rejeitadas. Passe valores de lista como um valor separado por vírgulas em vez de repetir uma opção. Só `--spec-type` continua podendo ser repetida, para compor modos especulativos.

O seletor de modelos mostra os modelos disponíveis, carregados e favoritos. Os
detalhes do modelo incluem o tamanho do contexto, as modalidades, a quantização e o
template de chat. O modo roteador gerencia processos filhos separados para os modelos
carregados; as solicitações identificam o modelo pelo campo `model` ou pelo
parâmetro de query.

Os servidores MCP são configurados por URL, com cabeçalhos de autorização opcionais.
Você pode inspecionar os logs de conexão e os prompts, recursos e ferramentas
disponíveis. O proxy CORS opcional do servidor precisa ser ativado explicitamente
com a opção correspondente do motor; ele só devolve conteúdo inerte e se recusa a
ser aberto como página (veja [limites de segurança](#security-limits)). As restrições do navegador a recursos HTTP em páginas HTTPS continuam valendo.
Quando dois servidores MCP ativados oferecem uma ferramenta com o mesmo nome, nenhuma das cópias é oferecida ao
modelo e a interface mostra um aviso: desative todos esses servidores menos um para usar a ferramenta.
As ferramentas integradas do servidor e a sandbox de JavaScript do navegador são opcionais.
As aprovações de ferramentas e o limite de turnos do modo agente continuam disponíveis na interface. O limite de turnos
(Configurações → Agente → **Turnos do agente**, o número máximo de ciclos de execução de ferramentas
antes de parar, para evitar loops infinitos) é 50 por padrão.

Os streams podem se reconectar usando a identidade e o cursor salvos da conversa
enquanto o servidor mantém o stream. Um servidor parado ou um stream expirado não
pode garantir a repetição. A interface informa a falha na reconexão em vez de
criar silenciosamente uma segunda solicitação de geração.

### Servidor MCP de shell local

Inicie o `mgs` com `--local-shell-mcp-server` para permitir que os modelos do chat web usem o terminal da máquina em que
o `mgs` está rodando:

```bash
mgs --model /path/to/model.gguf --local-shell-mcp-server
```

- Na inicialização, o `mgs` verifica quais shells estão instalados e oferece uma ferramenta para cada um deles: `bash.run` (Bash),
  `zsh.run` (Zsh), `sh.run` (sh) e `psh.run` (PowerShell, `pwsh`). O log lista os shells encontrados; um shell
  que não está instalado não recebe ferramenta.
- A interface o detecta, o adiciona aos servidores MCP como **local shell mcp server** e o ativa. Você pode desativá-lo ali,
  ou apenas para uma conversa, e a sua escolha é mantida. Quando o `mgs` é iniciado sem a opção, a entrada
  desaparece.
- Toda ferramenta recebe `command` e, opcionalmente, `cwd` (diretório de trabalho; por padrão, a pasta pessoal do usuário
  que executa o `mgs`) e `timeout` (segundos; 60 por padrão, 3600 no máximo). A resposta é um objeto JSON com
  `exit_code`, `stdout` e `stderr`, além de `signal`, `timed_out`, `cancelled`, `timeout`, `seconds`, `shell` e
  `cwd`; as saídas longas mantêm apenas o começo e o fim.
- Cada comando roda em um processo novo, sem terminal e com a entrada vazia: `cd` e as variáveis não passam
  para a chamada seguinte, e não é possível usar programas interativos (editores, perguntas). Quando o tempo limite
  expira ou a solicitação é cancelada, todos os processos que o comando iniciou são encerrados. Os processos deixados
  em segundo plano precisam redirecionar a sua saída.
- A interface pede permissão antes de cada chamada, a menos que você escolha permitir sempre a ferramenta.
- O `sudo` funciona sem terminal (Linux): quando ele precisa de uma senha, uma janela se abre no centro da página,
  em qualquer página em que você estiver, com o prompt do sudo (`[sudo] password for user:`), a ferramenta e o comando
  que a pede, um campo de senha e, abaixo dele, os botões **Aceitar** e **Cancelar**. O cursor já fica no
  campo de senha, então você pode digitar a senha imediatamente; **Enter** aceita e **Escape** cancela; um clique
  fora da janela não a fecha. A senha só vai para o sudo na máquina em que o `mgs`
  está rodando: ela não é salva na conversa, não é gravada nos logs e o modelo nunca a vê. Enquanto a
  janela espera, o tempo limite do comando para de contar; se você pressionar **Cancelar**, ou se ninguém responder
  em 5 minutos, o sudo falha. O sudo pede a senha de novo em cada chamada, porque cada uma é um processo novo, e até
  três vezes quando a senha está errada; a janela então avisa que a senha anterior pode estar errada. Os comandos
  rodam em uma sessão própria, sem o terminal em que o `mgs` foi iniciado; assim, o sudo nunca pede a senha nesse terminal.
  Só a interface web do `mgs` pode responder: com outros clientes MCP, o sudo falha como antes, por falta de terminal.
  O serviço implantado roda com a conta `moe-gguf-server`, que normalmente não tem senha nem permissão
  de sudo; por isso, isso se destina ao `mgs` executado com o seu próprio usuário (por exemplo, a compilação em `~/.local`).
  Quando o `mgs` não tem a variável `DISPLAY`, os comandos recebem `DISPLAY=:mgs-askpass`, porque o sudo só usa o seu
  auxiliar de senha sem terminal quando `DISPLAY` existe; os programas que tentam abrir uma janela falham na hora.
- No chat, cada chamada é mostrada com o nome da sua ferramenta, como `bash.run`. Expandida, ela mostra até três
  caixas cinza com fonte monoespaçada: **stdin** com o comando puro, **stdout** com a saída e **stderr** com
  o texto do erro. O stdout só aparece quando o comando imprimiu algo e o stderr só quando houve um erro;
  o código de saída fica ao lado do título delas, por exemplo `stdout (código 2):` (ao lado do stdin quando não há saída).
  A caixa stdin tem borda branca (cinza-escura no tema claro), a caixa stdout, verde, e a caixa stderr,
  vermelha, em todas as ferramentas de shell (`bash.run`, `zsh.run`, `sh.run` e `psh.run`).
- Ative **Configurações → Exibição → Expandir comandos do shell local automaticamente** para abrir cada chamada sozinha:
  cada caixa aparece assim que o seu conteúdo é conhecido, o stdin enquanto o modelo escreve o comando e o stdout e o
  stderr quando o comando termina.
- Cada chamada a uma ferramenta de shell deixa o seu próprio arquivo em `~/.mgs/local-shell-mcp-server/` (na pasta pessoal do usuário que
  executa o `mgs`; no serviço implantado, `/opt/moe-gguf-server/.mgs/local-shell-mcp-server/`), com o nome formado pela data
  e pela hora locais da chamada até o milissegundo, por exemplo `y2026m10d12h23m59s59ms731.jsonl`; `-2`, `-3`… são
  acrescentados quando várias chamadas começam no mesmo milissegundo. É um arquivo JSONL, com um objeto JSON por linha: primeiro o comando (`"event": "command"`,
  com a ferramenta, o shell, o diretório de trabalho e o tempo limite), gravado antes de ser executado; depois, toda a saída,
  sem cortes, à medida que chega, uma linha por trecho (`"event": "stdout"` ou `"stderr"`, com o texto em `data`) na
  ordem em que o comando a escreveu; e, por último, o resultado (`"event": "result"`: código de saída, sinal, se o tempo
  limite expirou ou se foi cancelado, os segundos que levou e os bytes de stdout e stderr) ou o motivo pelo qual ele não pôde
  começar (`"event": "error"`). Uma chamada rejeitada antes de ser executada (sem comando, com um diretório de trabalho
  que não existe, com um tempo limite inválido) também deixa o seu arquivo, com o comando como foi pedido e o erro. O chat
  mantém apenas o começo e o fim das saídas muito longas, mas o arquivo tem tudo, e ele já contém a saída se o `mgs`
  parar enquanto o comando está rodando. Para ler todo o stdout de uma chamada, execute `jq -j 'select(.event == "stdout") | .data'
  FILE`. O JSON só armazena texto; por isso, os bytes que não são UTF-8 (saída binária) são substituídos por `�`; um
  comando que imprime gigabytes deixa um arquivo desse tamanho. Só esse usuário pode ler a pasta e os arquivos, porque
  a saída de um comando pode conter segredos; nada os apaga, então remova os antigos quando não precisar mais deles.

No serviço implantado, adicione `Environment=MGS_ARG_LOCAL_SHELL_MCP_SERVER=1` à seção `[Service]` de um override do
systemd (`systemctl edit moe-gguf-server`) e reinicie o `moe-gguf-server`; no Alpine, exporte a mesma variável na
configuração do serviço OpenRC.

**Segurança.** Os comandos rodam com as permissões do usuário que executa o `mgs` (no serviço implantado, a conta
`moe-gguf-server`, nunca `root`). Qualquer pessoa que consiga abrir a interface ou chamar a API pode executá-los; por
isso, ative a opção apenas em ambientes confiáveis e proteja o servidor com uma chave de API. As solicitações vindas de
páginas de outros sites são rejeitadas, a origem CORS padrão passa a ser localhost e o log registra cada comando com o seu
código de saída. Leia o comando mostrado na janela de senha antes de digitar uma senha do sudo: a senha autoriza esse
comando. O servidor só a entrega a um auxiliar de senha cujo processo pai seja o sudo rodando como root para o seu
usuário; assim, um comando não pode iniciar o auxiliar por conta própria para ler a senha.

<a id="api"></a>
## API HTTP

Os exemplos usam 11443; substitua pela porta HTTPS de `ports.json` se ela for diferente.

A raiz padrão da API é `/api`. A interface continua em `/`. As rotas `/v1/...` na raiz,
do servidor original, não são mais registradas. Os clientes compatíveis com a OpenAI
devem usar `https://ai.example.com:11443/api/v1` como URL base. Os nomes dos campos de
solicitação/resposta mantêm a compatibilidade com o upstream; as interfaces C/C++ do
motor mantêm as suas funções atuais, mas não as que estavam marcadas como obsoletas
(veja [compatibilidade removida](#removed-compatibility)).

```bash
curl https://ai.example.com:11443/api/v1/chat/completions \
  -H "Authorization: Bearer ${MGS_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-id","messages":[{"role":"user","content":"Hello"}]}'
```

Defina `MGS_API_KEY` com a chave instalada no seu próprio shell. As operações nativas,
os embeddings, a tokenização, o reranking, o gerenciamento de modelos e os streams
retomáveis estão documentados no Swagger UI incluído, em `/api/doc/`; o seu JSON OpenAPI
fica em `/api/doc/openapi.json`, gerado pelo servidor a cada solicitação, sem um arquivo JSON estático. O inventário reflete as rotas registradas
pelo processo em execução, incluindo as rotas do roteador e as rotas de compatibilidade
opcionais. Os recursos desativados podem retornar 403. Uma chave ausente/inválida
retorna 401, e o carregamento do modelo retorna 503. As rotas de saúde e de lista de
modelos mantêm o acesso público para `GET` e `HEAD`; veja
[limites de segurança](#security-limits) para as rotas que sempre precisam da chave.

`--api-prefix /api/custom` muda as rotas de inferência para baixo de `/api/custom`.
A interface na raiz recebe esse prefixo de `/api/config.js`; a documentação continua em
`/api/doc/`. O namespace da documentação é reservado e não pode ser usado como prefixo
de inferência. Os processos filhos do roteador usam `/api` internamente. O servidor
valida os prefixos; os valores fora de `/api/` e os caminhos com barra no final são rejeitados.

O Monitor usa `GET /api/monitor` (relatório completo), `GET /api/monitor/memory` (instantâneos da memória dos modelos
carregados) e `GET /api/monitor/usage` (apenas o uso de CPU e GPU, disponível também enquanto o modelo carrega). Eles
leem contadores e instantâneos publicados pelo loop de inferência, nunca a fila de tarefas; por isso, consultá-los
periodicamente não desperta um modelo adormecido nem reinicia o seu temporizador de inatividade. Os seus campos
estão descritos no Swagger.

A interface lê as suas configurações com `GET /api/ui/settings` (`{"settings": DOCUMENT}`, ou
`{"settings": null}` antes do primeiro salvamento) e as salva com `POST /api/ui/settings`
(`{"settings": DOCUMENT}`, que substitui `~/.mgs/config/ui-settings.json` por inteiro, com modo 0600). As duas exigem
a chave de API e também respondem enquanto o modelo carrega.

<a id="request-queue"></a>
### Fila de solicitações e solicitações paralelas

Por padrão, o `mgs` resolve uma solicitação por vez (`-np 1`) e lhe dá o contexto inteiro; assim, cada solicitação
pode usar todos os tokens de contexto, venha ela da interface web ou diretamente da API. As solicitações que chegam
nesse meio-tempo esperam em uma fila única e são resolvidas na ordem em que chegaram: as solicitações da interface
não passam à frente das solicitações da API, nem o contrário. A API se comporta da mesma forma, sejam quais forem
as configurações da interface.

Para resolver várias solicitações ao mesmo tempo, inicie o `mgs` com `-np N` (`--parallel N`, `MGS_ARG_N_PARALLEL`):
até N solicitações rodam em paralelo e o contexto é dividido entre elas, de modo que cada uma recebe uma parte
(`n_ctx_slot` na linha `initializing` do log); com `--kv-unified`, elas compartilham, em vez disso, um único cache KV
do contexto inteiro. As solicitações além de N continuam esperando na fila.

A interface web adiciona `"mgs_client": "webui"` às suas solicitações. Enquanto uma das suas solicitações com
streaming espera, o stream começa com eventos como

```text
data: {"mgs_queue":{"ahead":2,"ahead_api":1,"ahead_webui":1}}
```

(`ahead`: solicitações em execução ou esperando à frente dela; `ahead_api` e `ahead_webui`: quantas delas chegaram
diretamente à API ou vieram da interface web), e um último evento com todas as contagens em 0 quando ela começa. Os
clientes que não enviam `mgs_client` nunca recebem esses eventos; assim, os clientes compatíveis com a OpenAI recebem o
stream habitual. O Swagger descreve o campo junto com os demais parâmetros de completion.

<a id="security-limits"></a>
### Limites de segurança

Esses limites protegem o servidor e a interface contra solicitações, arquivos e modelos hostis. Eles são generosos
o bastante para o uso normal: o que os ultrapassa falha com um erro claro em vez de derrubar o servidor.

- **Chave de API.** As rotas de saúde e de lista de modelos só são públicas para `GET` e `HEAD`. Recarregar a lista de
  modelos (`GET /api/models?reload=1`) e adicionar ou excluir modelos no modo roteador (`POST` e `DELETE /api/models`)
  exigem a chave sempre que houver uma configurada.
- **Páginas de outros sites.** Quando as ferramentas do servidor ou os servidores MCP estão ativados (`--tools`, `--agent`,
  `--mcp-servers-config`, `--mcp-servers-json` ou `--local-shell-mcp-server`) e `--cors-origins` não está definido,
  o servidor responde 403 a qualquer solicitação que não seja `GET`, `HEAD` ou `OPTIONS` cujo cabeçalho `Origin` indique
  outro site. A interface (também atrás do Apache) e os clientes que não enviam `Origin`, como o curl ou os SDKs,
  não são afetados. `POST /api/tools` só aceita `Content-Type: application/json` (caso contrário, 415).
- **Proxy CORS.** `/api/cors-proxy` recusa as solicitações que um navegador abriria como página, frame ou worker
  (403) e devolve todas as respostas como conteúdo inerte: ele não pode executar scripts nem definir cookies no endereço do servidor.
- **Listagens do roteador.** No modo roteador, `status.args` e `status.preset` de `/api/models` mostram `<redacted>`
  no lugar dos valores de `--hf-token`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--mcp-servers-config`
  e `--mcp-servers-json`; o log faz o mesmo. Os processos dos modelos continuam recebendo os valores reais.
- **Clientes lentos.** Enquanto o servidor recebe uma solicitação, cada leitura espera no máximo 30 s (`--timeout` continua
  regendo as gravações, para que as respostas longas não sejam cortadas), e no máximo 256 conexões aceitas esperam por uma thread livre.
- **Corpos das solicitações.** Os corpos JSON com mais de 1024 níveis de arrays ou objetos aninhados são rejeitados com 400. Os esquemas JSON
  (`response_format`, `json_schema`) podem ter até 256 níveis de aninhamento; os valores `pattern` vazios e os quantificadores
  sem nada para repetir são rejeitados.
- **Mídia remota.** As URLs em `image_url`, `input_audio` e `input_video` só podem apontar para endereços públicos da
  internet. O loopback, as redes privadas (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10), os endereços
  link-local (incluindo o endereço de metadados de nuvem 169.254.169.254), os endereços locais únicos IPv6 e outros
  endereços não públicos são rejeitados com 400, também quando um redirecionamento leva até eles. Para permiti-los, por
  exemplo para ler imagens de um servidor da sua rede, inicie o `mgs` com `--media-allow-private-urls`
  (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`); faça isso apenas quando todos os clientes da API forem confiáveis. Cada download
  tem 10 s para conectar, 30 s no total, até 5 redirecionamentos e 10 MiB, e o servidor executa no máximo 8 downloads ao mesmo tempo.
- **Mídia decodificada.** Imagens de até 32768 pixels por lado e 100 megapixels, áudio de até uma hora, quadros de vídeo
  de até 8192 pixels por lado e 2 GiB de quadros decodificados por vídeo, e no máximo 2 GiB de imagens e áudio
  decodificados por solicitação.
- **Templates de chat do modelo.** `range()` produz no máximo um milhão de elementos, as strings construídas ficam abaixo
  de 256 MiB, a indentação de `indent()` e `tojson()` abaixo de 256 espaços e o aninhamento abaixo de 1024 instruções; um
  template que os ultrapassa falha com um erro de template.
- **RPC.** Em uma compilação com o backend RPC, o servidor RPC verifica os tamanhos dos quadros e os limites dos tensores
  enviados pelos seus pares e fecha apenas a conexão responsável. Exponha-o apenas a máquinas confiáveis: ele continua
  atendendo um cliente por vez.
- **Interface.** Os links nas mensagens mantêm apenas os esquemas http, https, mailto e tel (ou endereços relativos), e
  os anexos de imagem precisam ser imagens embutidas. A importação de conversas aceita arquivos de até 512 MB, arquivos
  ZIP com até 10.000 entradas, 256 MB por arquivo JSONL e 1 GB extraído no total. Um arquivo com registros malformados,
  ou com uma mensagem que pertence a outra conversa, é rejeitado por inteiro; uma conversa cujas mensagens já existem
  é ignorada e informada.

Os comandos de instalação continuam baixando a branch `main` atual por HTTPS; eles não verificam uma assinatura nem
uma revisão fixa.

<a id="performance"></a>
## Memória e desempenho

O registro da memória do host e a pré-busca de especialistas vêm ativados por padrão.
A pré-busca sobrepõe as transferências de especialistas à computação, especialmente
com prompts grandes. Os resultados dependem do modelo, da capacidade de memória e da
largura de banda da RAM/do PCIe.

Para modelos com especialistas descarregados na CPU, `--cpu-moe` ou `--n-cpu-moe N` podem
reduzir o uso de VRAM. O registro do host fixa as páginas de pesos mapeadas para acelerar
as cópias do host para o dispositivo. A memória fixada não pode ser paginada para fora;
por isso, deixe RAM suficiente para o sistema operacional. Esse caminho de registro é
específico de POSIX; o Windows mantém o seu comportamento de carregamento anterior.
`--load-mode none` só é recomendado quando o registro do host falha total ou
parcialmente, não como um requisito normal.

A pré-busca de especialistas usa, por padrão, três slots na VRAM. Cada slot precisa
comportar o maior tensor de especialista transferido. Os valores de 2 a 8 em
`GGML_SCHED_PREFETCH_EXPERTS` escolhem explicitamente a quantidade de slots. Se a
alocação de memória falhar, pelo menos dois slots alocados são mantidos quando possível;
caso contrário, são usadas as transferências normais. A pré-busca só começa quando o
batch tem pelo menos o dobro de seleções de especialistas em relação ao número de
especialistas. O seu pool pertence à primeira GPU que a ativa; as outras GPUs mantêm
cópias seletivas. Batches maiores podem ajudar no preenchimento do prompt (prefill),
mas exigem mais memória.

Para diagnóstico, defina `GGML_CUDA_REGISTER_HOST=0` ou
`GGML_SCHED_PREFETCH_EXPERTS=0` para desativar a otimização correspondente.
Não defina essas variáveis durante a operação normal, a menos que queira desativá-las de propósito.

O MTP automático examina os metadados do GGUF e os nomes dos tensores antes da alocação
dos pesos, incluindo todos os shards. Ele exige uma arquitetura compatível, um tronco de
modelo e todas as cabeças integradas. Uma cabeça externa ou uma arquitetura não compatível
não o ativa. O limite padrão do rascunho é de três tokens. A seleção explícita com
`--spec-type` tem prioridade; `--no-spec-mtp-auto` e `--spec-type none` desativam a
seleção automática. `--spec-draft-n-max` muda o limite. O MTP afeta a geração, não
diretamente o prefill nem o tempo até o primeiro token. A melhoria depende da taxa de
aceitação e do hardware; aumentar o limite não garante mais velocidade.

<a id="mtp-vram"></a>
Os pesos do MTP sempre ficam na memória da GPU quando a GPU consegue comportá-los: as camadas MTP enxertadas no modelo, um
GGUF de MTP separado passado com `--spec-draft-model` e a cabeça de saída que elas usam. As opções que movem pesos para a
memória do sistema (`--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, os seus equivalentes `--spec-draft-*` e o ajuste
automático) não se aplicam a eles, e eles ficam na GPU mesmo com um `--gpu-layers` baixo. O ajuste automático coloca o
restante do modelo em volta deles, e o [cache de especialistas do MoE](#expert-cache) usa a memória que sobra depois disso.
Se as camadas MTP não couberem na memória livre da GPU com uma margem de 512 MiB, o log mostra um aviso e elas são
posicionadas como qualquer outra camada. Com um GGUF de MTP separado, as camadas MTP enxertadas no modelo principal não são
carregadas, para que não ocupem memória duas vezes; com `--spec-type none`, nenhum peso do MTP é carregado. O log de
carregamento confirma o posicionamento com uma linha como
`keeping the MTP weights in CUDA0 (1 layers, 856.36 MiB)`, visível com `-lv 4`.

<a id="expert-cache"></a>
### Cache de especialistas do MoE

Quando os especialistas do MoE ficam na memória do sistema e o restante da camada roda em uma GPU, o servidor mantém os
especialistas usados mais recentemente de todas as camadas em um cache que ocupa a VRAM livre;
com várias GPUs, cada GPU armazena em cache, na sua própria VRAM, as camadas atribuídas a ela. Para cada token gerado (e
qualquer batch com menos de 32 tokens), os especialistas encontrados no cache rodam na GPU, uma parte dos que faltam é
enviada para o cache e o restante roda na CPU ao mesmo tempo; essa parte acompanha as larguras de banda medidas do PCIe
e da CPU. Os prompts longos mantêm a pré-busca de especialistas, que agora copia os especialistas em cache dentro da GPU
e só envia os demais. A saída coincide com a da execução sem cache, salvo pelo arredondamento. A GPU e a CPU arredondam de forma
diferente, e os especialistas que cada uma calcula dependem do conteúdo do cache; por isso, uma geração gulosa (greedy)
longa pode derivar, depois de alguns tokens, para um texto diferente, mas igualmente válido, de uma execução para outra;
o cache do llama.cpp oficial se comporta da mesma forma.

O cache vem ativado por padrão e é criado pela primeira solicitação, depois que o modelo, o contexto do MTP e o
projetor multimodal estão carregados. Sem `--gpu-layers`, `--n-cpu-moe` ou `--override-tensor`, o ajuste
automático mantém todos os especialistas na memória do sistema quando eles cabem com uma margem (4 GiB ou 10% da RAM) e deixa a VRAM livre para o cache (com várias GPUs, ele distribui as camadas entre
elas de acordo com a memória livre de cada uma, para que todas as GPUs tenham cache); caso contrário, ele coloca camadas inteiras de especialistas na GPU como antes e o cache usa o que
sobra. As opções explícitas de posicionamento são respeitadas: `--cpu-moe` dá ao cache o máximo de VRAM e
`--n-cpu-moe N` mantém algumas camadas estáticas na GPU. Os especialistas mantidos na RAM são fixados; por isso, a RAM
precisa comportar todos eles, mais o sistema operacional.

<a id="context-size"></a>
Quando o modelo não cabe na VRAM e `-c` não é informado, o ajuste automático também escolhe o tamanho do contexto: o
cache KV ocupa no máximo um quarto da VRAM que sobra depois dos pesos densos, até o contexto de treinamento do
modelo, e os especialistas (o cache ou camadas inteiras de especialistas) ficam com o restante. Antes, esses modelos
sempre recebiam o mínimo de 4096 tokens compartilhados por todos os slots, o que cortava as respostas longas
(`truncated = 1` no log). O log mostra o resultado em uma linha como `context size raised from 4096 to N tokens` e
em `n_ctx_slot` da linha `initializing`. `-c N` define o contexto explicitamente (mais contexto deixa menos VRAM para o
cache e reduz a velocidade de geração: 35,8 tokens/s com 4096 tokens e 33,9 com 122624 na máquina de referência) e
`--fit-ctx N` aumenta o mínimo. O contexto do MTP recebe o mesmo tamanho do contexto do modelo, em vez de reservar
VRAM para todo o contexto de treinamento.

Antes do contexto, o ajuste automático escolhe o micro-batch (`-ub`), o número de tokens do prompt processados em
uma passada. Com os especialistas na RAM, cada passada envia para a GPU os especialistas que não estão no cache; por
isso, passadas menos numerosas e maiores processam os prompts longos muito mais rápido, enquanto a geração não muda.
A menos que `-b` ou `-ub` seja informado, ele escolhe o maior valor entre 4096, 2048 e 1024 tokens cujos buffers
adicionais caibam em um oitavo da VRAM que sobra depois dos pesos densos, e aumenta o batch (`-b`) junto. Na máquina
de referência, um prompt de 2514 tokens passou de 214 para 755 tokens/s com 4096, com a mesma velocidade de geração;
os buffers maiores reduzem o contexto padrão (cerca de 90 mil tokens sem MTP e 61 mil com MTP em 16 GB). O log
mostra a escolha em uma linha como `micro-batch size
raised from 512 to 4096 tokens (batch 4096)`.

| Opção | Variável | Efeito |
| --- | --- | --- |
| `--moe-cache`, `--no-moe-cache` | `MGS_MOE_CACHE` (`0` desativa) | Ativa ou desativa o cache (ativado por padrão). |
| `--moe-cache-size MiB` | `MGS_MOE_CACHE_MIB` | Tamanho total fixo do cache, dividido entre as GPUs de acordo com os especialistas que cada uma armazena em cache; `-1` (padrão) usa a VRAM livre de todas as GPUs. |
| `--moe-cache-reserve MiB` | `MGS_MOE_CACHE_RESERVE_MIB` | VRAM deixada livre em cada GPU além do cache e dos slots de pré-busca (padrão 1024). Aumente esse valor quando outros programas usarem a GPU. |
| — | `MGS_MOE_CACHE_FILL_RATIO` | Diagnóstico: fração fixa (0–1) dos especialistas ausentes que é enviada, em vez do equilíbrio medido. |
| — | `MGS_MOE_CACHE_STATS` | Diagnóstico: registra no log os contadores do cache a cada N passos de decodificação (visível com `-lv 4`). |

`GET /api/expert-cache` retorna o tamanho, as camadas e os slots em cache, os acertos e as falhas, os envios, os
especialistas na CPU, a fração dos bytes de pré-busca atendidos pelo cache e as larguras de banda medidas (com várias GPUs,
os tamanhos e os contadores se somam e a largura de banda de envio é a média delas). `POST /api/expert-cache` com
`{"size_mib": N}` muda o tamanho sem reiniciar (`-1` automático, `0` libera o cache); a próxima solicitação
aplica a mudança e começa com o cache vazio:

```bash
curl -H "Authorization: Bearer ${MGS_API_KEY}" https://127.0.0.1:11443/api/expert-cache
curl -H "Authorization: Bearer ${MGS_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"size_mib": 2048}' https://127.0.0.1:11443/api/expert-cache
```

Com `--metrics`, `/api/metrics` adiciona os contadores e medidores `mgs:moe_cache_*` (passos, acertos, falhas, envios,
especialistas na CPU, bytes de pré-busca, tamanho, slots e larguras de banda).

Medição de referência em uma RTX 4060 Ti de 16 GB com Qwen3.6-35B-A3B Q8_0 (35 GiB, 8 threads da CPU), em comparação com
`--n-cpu-moe 28` sem cache: geração de 26,9 → 35,7 tokens/s (+33%), processamento do prompt de 767–778 →
718–730 tokens/s (−6%) e, com o MTP ativado, de 33,6 → 47,0 tokens/s (+40%). Na mesma máquina, o llama.cpp
oficial com o seu próprio cache (`-cmoe --moe-cache-mib 10240`) chegou a 26,7 tokens/s, e a 27,8 com MTP. Os ganhos dependem da fração de especialistas
que cabe na VRAM, da localidade do roteamento do modelo e das larguras de banda do PCIe e da RAM.

<a id="threads"></a>
### Threads da CPU

A CPU calcula os especialistas que não estão no cache. Esse trabalho é limitado pela largura de banda da memória e
satura com poucas threads, enquanto usar todas as CPUs lógicas faz as threads competirem com a que controla
a GPU. Sem `-t`, o `mgs` usa os núcleos úteis para cálculo, mas nunca mais da metade das CPUs lógicas em que pode
rodar. Em uma máquina com SMT (duas threads por núcleo), isso equivale ao número de núcleos físicos, como antes; em
uma máquina virtual que não expõe o SMT, é metade das CPUs virtuais. Na máquina de referência (16 núcleos
virtuais com uma sessão de desktop, Qwen3.6-35B-A3B Q8_0, sem MTP), de 4 a 12 threads geraram na mesma velocidade
(32,9–33,9 tokens/s) e 16 threads, a 28,7 tokens/s. O log de inicialização mostra a escolha em uma linha como
`using 8 CPU threads for generation and 8 for batches`; `-t` e `-tb` as definem explicitamente.

<a id="checkpoints"></a>
### Checkpoints de contexto

Os modelos com camadas recorrentes ou de janela deslizante (por exemplo, os híbridos Qwen3.5/3.6/3.8, Nemotron-H ou
Granite 4) não conseguem voltar o seu estado atrás; por isso, quando uma solicitação altera uma parte anterior da
conversa, o servidor retoma a partir do checkpoint mais recente antes da alteração. Os checkpoints são posicionados no
início das mensagens do usuário, do assistente e das ferramentas (onde os agentes de programação removem raciocínios
antigos ou cortam resultados antigos de ferramentas) e pelo menos a cada `--checkpoint-min-step` tokens (padrão 1024,
antes 8192), sem passadas adicionais pelo prompt. Quando `--ctx-checkpoints` (32) é atingido, o checkpoint mais
próximo dos seus vizinhos é removido, para que os demais continuem distribuídos ao longo da conversa. No cenário de
agente de referência (14 mil tokens, oito resultados de ferramentas), cortar um resultado de ferramenta tardio passou de
19,3 s para 5,5 s, e um inicial, de 18,0 s para 13,6 s. Cada checkpoint mantém o estado recorrente ou de janela
deslizante na memória do host (63 MiB para o Qwen3.6-35B-A3B).

<a id="upstream-options"></a>
### Opções atualizadas e diagnóstico

Mantenha o padrão `--load-mode mmap` quando o registro do host for bem-sucedido. Use `--load-mode none` apenas como alternativa quando o registro falhar. `--load-mode dio` agora prepara as leituras usando no máximo 64 MiB de memória adicional; isso não é um limite para a RAM total do modelo.

`GGML_CUDA_PEER_MAX_BATCH_SIZE` foi removida porque não tinha efeito em tempo de execução. As antigas opções de carregamento (`--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, `--no-direct-io`) e a opção de compilação `GGML_CUDA_FA_ALL_QUANTS` não são mais aceitas; veja [compatibilidade removida](#removed-compatibility) para saber o que as substitui. Em uma compilação manual com o CMake, `-DGGML_CUDA_FA_QUANTS=q8_0-q4_0` adiciona essa combinação de kernels vetoriais K/V; são aceitas listas separadas por vírgulas ou listas entre aspas separadas por ponto e vírgula. A lista padrão é `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; F16 está sempre incluído. `all` aumenta o trabalho de compilação. As combinações de kernels vetoriais ausentes usam uma alternativa com conversão para F16 e podem ser mais lentas.

Os GGUFs completos de cabeça única do GLM-4.5-Air (`glm4moe`) agora podem ativar o MTP automático. As demais verificações de elegibilidade existentes e a precedência das opções explícitas continuam valendo. O MTP separa os grafos com e sem saídas, preserva a ordem original dos tokens entre sequências simultâneas e usa as posições reais depois das imagens.

O ajuste automático conta as camadas nextn (MTP); assim, a primeira camada não fica mais na CPU quando o MTP está desativado. Os pesos do MTP agora sempre ficam na memória da GPU; veja [posicionamento do MTP](#mtp-vram). O padrão de `--checkpoint-min-step` agora é 1024 em vez de 8192; veja [checkpoints de contexto](#checkpoints). `--moe-cache`, `--no-moe-cache`, `--moe-cache-size` e `--moe-cache-reserve` controlam o [cache de especialistas do MoE](#expert-cache). O ajuste automático não deixa mais os modelos maiores que a VRAM com um contexto de 4096 tokens e agora também escolhe o micro-batch; veja [tamanho do contexto](#context-size). O número padrão de threads da CPU agora é, no máximo, metade das CPUs lógicas; veja [threads da CPU](#threads). O padrão de `-np` agora é um slot com o contexto inteiro, em vez do modo automático com quatro slots compartilhando um cache KV unificado, e `--kv-unified` vem desativado por padrão; veja [fila de solicitações](#request-queue).

Com `--metrics`, o `GET /api/metrics` autenticado inclui `mgs:spec_decode_num_draft_tokens_total`, `mgs:spec_decode_num_accepted_tokens_total`, `mgs:spec_decode_num_drafts_total` e `mgs:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`. Os índices de posição começam em zero; essa série aparece depois que uma solicitação especulativa é concluída. Compare as variações dos contadores de tokens aceitos/propostos ao avaliar a aceitação e meça a latência separadamente; a aceitação, por si só, não prova um ganho de velocidade. No modo roteador, selecione o modelo com `?model=MODEL_ID`.

`--media-allow-private-urls` (`MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS`, desativado por padrão) permite que as URLs de mídia das solicitações cheguem ao loopback, a endereços privados e a outros endereços não públicos; veja [limites de segurança](#security-limits).

As exportações de conversas leem a árvore completa de mensagens e os metadados persistidos. Se as ferramentas do servidor retornarem 403, as novas mensagens deixam de repetir essa consulta que falhou; reabrir o painel de ferramentas tenta de novo. Os padrões do servidor na primeira visita preservam as configurações já alteradas pelo usuário, incluindo a chave de API.

[Implementação e fontes do upstream](CODE.pt-BR.md#upstream-review).

<a id="removed-compatibility"></a>
## Compatibilidade removida

Os mecanismos de retrocompatibilidade foram removidos. Os nomes antigos não são mais
traduzidos nem geram avisos: o servidor rejeita as opções de linha de comando removidas
como argumentos inválidos, ignora as variáveis de ambiente removidas e responde 404
para as rotas removidas. Atualize os scripts, os overrides de serviço, os presets e os
clientes para os substitutos abaixo.

| Removido | Use em vez disso |
| --- | --- |
| `--mmap` / `--no-mmap` | `--load-mode mmap` / `--load-mode none` |
| `--mlock` | `--load-mode mlock` (ou `mmap+mlock` para combinar os dois) |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | `--load-mode dio` / `--load-mode none` |
| `-dt`, `--defrag-thold` | Nada; não tinha efeito. |
| `--swa-checkpoints` | `-ctxcp`, `--ctx-checkpoints` |
| `--webui`, `--no-webui`, `--webui-config`, `--webui-config-file`, `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | `--ui`, `--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`, `--no-ui-mcp-proxy` |
| Aliases antigos do modelo de rascunho, como `-md`/`--model-draft`, `-ngld`/`--gpu-layers-draft`, `-td`/`--threads-draft`, `-devd`/`--device-draft`, `-ctkd`/`--cache-type-k-draft`, `-hfd`/`--hf-repo-draft` ou `--draft-p-min` | A opção `--spec-draft-*` correspondente, por exemplo `--spec-draft-model`, `--spec-draft-ngl`, `--spec-draft-threads`, `--spec-draft-device`, `--spec-draft-type-k`, `--spec-draft-hf` ou `--spec-draft-p-min`. `mgs --help` lista todas elas. |
| `--draft`, `--draft-n`, `--draft-max` / `--draft-min`, `--draft-n-min` | `--spec-draft-n-max` ou `--spec-ngram-mod-n-max` / `--spec-draft-n-min` ou `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | A opção `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` e `--spec-ngram-*-min-hits` do modo de n-gramas escolhido |
| Variáveis de ambiente, opções do CMake e símbolos da API C com o antigo prefixo do upstream, incluindo os aliases que as versões anteriores já tinham abandonado | O nome atual com o prefixo `MGS_` (`mgs_` na API C), por exemplo `MGS_ARG_PORT`, `MGS_ARG_<NAME>=false` em vez de uma variante `NO_`, `MGS_ARG_LOAD_MODE` em vez das variáveis de mapeamento de memória, de bloqueio e de Direct I/O, `MGS_ARG_SPEC_DRAFT_N_MAX` e `MGS_ARG_SPEC_DRAFT_N_MIN` para os limites do rascunho, `MGS_CACHE` ou `MGS_BUILD_UI`; para as opções do backend, a opção `GGML_*` correspondente, por exemplo `GGML_CUDA` ou `GGML_NATIVE` |
| `HF_ENDPOINT`, `MODEL_ENDPOINT` | `MGS_MODEL_ENDPOINT` |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` ou `MGS_CACHE` |
| `POST /api/completion` | `POST /api/completions` |
| `POST /api/embedding` | `POST /api/embeddings` ou `POST /api/v1/embeddings` |
| `POST /api/reranking`, `POST /api/v1/reranking` | `POST /api/rerank`, `POST /api/v1/rerank` |
| Valor `deepseek-legacy` de `reasoning_format` (campo da solicitação e `--reasoning-format`) | `deepseek` ou `auto`; o raciocínio é retornado em `reasoning_content` |
| Campo de solicitação `reasoning_budget_end_tag` | `reasoning_budget_end_tags` (array de strings) |
| `reasoning_in_content` em `generation_settings` | Nada; o campo foi removido. |
| Download de uma interface pré-compilada do upstream a partir de um bucket do Hugging Face (as opções do CMake `USE_PREBUILT_UI` e `UI_HF_BUCKET` e a variável de ambiente `HF_UI_VERSION`) | Nada; a interface é sempre compilada a partir de `frontend/` (`MGS_BUILD_UI=ON`) ou deixada de fora. |
| CMake `GGML_CUDA_FA_ALL_QUANTS=ON` | `GGML_CUDA_FA_QUANTS=all` |
| Número de compilação e commit: as variáveis do CMake `MGS_BUILD_NUMBER`/`MGS_BUILD_COMMIT` (também em `mgs-config.cmake`) e a string `b0-unknown` de `build_info`, `system_fingerprint` e do `User-Agent` | `MGS_VERSION`, definida em `backend/cmake/build-info.cmake`; esses campos agora trazem a versão, como `0.1.0` |
| Modelos Llama da Meta (as arquiteturas GGUF `llama`, `llama4` e `llama-embed` e o projetor de visão do Llama 4) e os templates de chat integrados `llama2`, `llama2-sys`, `llama2-sys-bos`, `llama2-sys-strip`, `llama3` e `llama4` | Nada; use um dos modelos MoE compatíveis. |
| `bash install/install.sh` | `bash deploy/install-update-reinstall-debian.sh` |
| Opção do instalador `--no-sudo` | Nada; os instaladores sempre rodam como root, sem sudo. |
| Funções e campos que as APIs C tinham marcado como obsoletos: em `mgs.h`, `mgs_load_model_from_file`, `mgs_free_model`, `mgs_new_context_with_model`, `mgs_n_ctx_train`, `mgs_n_embd`, `mgs_n_layer`, `mgs_n_head`, `mgs_n_vocab`, `mgs_get_state_size`, `mgs_copy_state_data`, `mgs_set_state_data`, `mgs_load_session_file`, `mgs_save_session_file`, os aliases `mgs_token_*` (por exemplo `mgs_token_bos`, `mgs_token_get_text` ou `mgs_token_fim_pre`), `mgs_add_bos_token`, `mgs_add_eos_token`, `mgs_vocab_cls`, `mgs_set_warmup`, `mgs_sampler_init_grammar_lazy`, o campo `defrag_thold` de `mgs_context_params` e `MGS_STATE_SEQ_FLAGS_SWA_ONLY`; em `ggml.h`, `ggml_type_sizef`, `ggml_add1`, `ggml_add1_inplace`, `ggml_rope_custom`, `ggml_rope_custom_inplace` e `ggml_upscale_ext`; em `mtmd.h`, `mtmd_encode`, `mtmd_image_tokens_get_nx`, `mtmd_image_tokens_get_ny` e o campo `image_marker` | A função atual: `mgs_model_load_from_file`, `mgs_model_free`, `mgs_init_from_model`, `mgs_model_n_*`, `mgs_vocab_n_tokens`, `mgs_state_*`, `mgs_vocab_*` (`mgs_vocab_bos` em vez de `mgs_vocab_cls`), `mgs_vocab_get_add_bos`, `mgs_vocab_get_add_eos`, `mgs_sampler_init_grammar_lazy_patterns`, `MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY`, `ggml_row_size`, `ggml_add`, `ggml_rope_ext`, `ggml_interpolate`, `mtmd_encode_chunk`, `mtmd_image_tokens_get_decoder_pos` e `media_marker`. As execuções de aquecimento são feitas por quem chama, e `defrag_thold` não tinha efeito. |
| Gemma 4: as arquiteturas `gemma4` e `gemma4-assistant`, os projetores `gemma4v`, `gemma4uv`, `gemma4a` e `gemma4ua`, o formato de chat `peg-gemma4`, o tokenizador e o pré-tokenizador `gemma4` e os presets `--vision-gemma-4b-default` e `--vision-gemma-12b-default` | Nada; use uma das [famílias compatíveis](#supported-models). |
| Modelos DeepSeek anteriores ao V4.1: as arquiteturas `deepseek`, `deepseek2` (também os arquivos Kimi-K2.x, GLM-4.7-Flash, Moonlight, GigaChat3, Kanana-2, Mistral-Large-3 e Youtu convertidos com ela), `deepseek2-ocr` (com os projetores `deepseekocr` e `deepseekocr2`), `deepseek32`, `mistral4` e `deepseek4`, os pré-tokenizadores `deepseek-llm` e `deepseek-coder` e o backbone de rascunho DSpark do DeepSeek-V4 | DeepSeek-V4.1-Flash (`deepseek41`). |
| `--jinja`, `--no-jinja` (`MGS_ARG_JINJA`) e os templates integrados do motor sem Jinja (`--chat-template` com um nome como `deepseek3`, `chatglm4` ou `mistral-v7-tekken`) | Nada; os templates de chat sempre passam pelo Jinja. `--chat-template chatml` continua selecionando o ChatML; para outro template, passe o seu código-fonte Jinja com `--chat-template` ou `--chat-template-file`. As solicitações com `tools` não precisam mais de nenhuma opção. |
| `-hfr`, `-hfrv` | `-hf`, `-hfv` |
| Segundos nomes longos de uma opção: `--usage`, `--n-predict`, `--sampling-seq`, `--temperature`, `--top-n-sigma`, `--typical-p`, `--no-mmproj-auto`, `--n-gpu-layers`, `--embeddings`, `--reranking`, `--log-verbose`, `--log-verbosity`, `--spec-draft-ncmoe`, `--kl-divergence-base`, `--output-file`, `--from-chunk` | `--help`, `--predict`, `--sampler-seq`, `--temp`, `--top-nsigma`, `--typical`, `--no-mmproj`, `--gpu-layers`, `--embedding`, `--rerank`, `--verbose`, `--verbosity`, `--spec-draft-n-cpu-moe`, `--save-all-logits`, `--output`, `--chunk` (as chaves dos presets do roteador seguem os mesmos nomes) |
| `MGS_ARG_CHECKPOINT_MIN_SPACING_NT`, `MGS_ARG_FIT_ESTIMATE`, `MGS_ARG_THINK`, `MGS_ARG_THINK_BUDGET`, `MGS_ARG_THINK_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_K`, `MGS_ARG_SPEC_DRAFT_CACHE_TYPE_V`, `MGS_ARG_N_GPU_LAYERS_DRAFT` | `MGS_ARG_CHECKPOINT_MIN_STEP`, `MGS_ARG_FIT_PRINT`, `MGS_ARG_REASONING_FORMAT`, `MGS_ARG_REASONING_BUDGET`, `MGS_ARG_REASONING_BUDGET_MESSAGE`, `MGS_ARG_SPEC_DRAFT_TYPE_K`, `MGS_ARG_SPEC_DRAFT_TYPE_V`, `MGS_ARG_SPEC_DRAFT_NGL`, com os nomes das suas opções |
| Campo de solicitação `thinking_budget_tokens` | `reasoning_budget_tokens` (os clientes da Anthropic continuam enviando `thinking.budget_tokens`) |
| `content` em `POST /api/embeddings` e `POST /api/v1/embeddings` | `input` |
| `mgs_chat_apply_template`, `mgs_chat_builtin_templates`, a struct `mgs_chat_message` e `MGS_FILE_MAGIC_GGLA` em `mgs.h`; `mtmd_image_tokens_get_id` em `mtmd.h` | Nada; a formatação do chat é feita pelos templates Jinja de `common/chat.h`, e os demais não tinham uso. |

O CMake informa as variáveis `-D` removidas como não utilizadas em vez de aplicá-las;
por isso, verifique a saída da configuração das compilações manuais.

A interface web mantém as suas configurações no servidor (`~/.mgs/config/ui-settings.json`)
e, no navegador, apenas a sua chave de API (`MgsUi.apiKey` no `localStorage`) e as suas
conversas (o banco de dados IndexedDB `MgsUi`). Ela não lê nem migra as configurações
que as versões mais antigas guardavam no `localStorage` (`MgsUi.config`,
`MgsUi.userOverrides`, os modelos favoritos, as listas de ferramentas, o esforço de
raciocínio, as recomendações de MCP dispensadas e o idioma); por isso, as configurações
começam com os seus valores padrão e a chave de API precisa ser digitada de novo uma
vez. Ela também não lê nem migra os dados salvos por versões mais antigas com outros
nomes, a chave avulsa `theme`, os marcadores de raciocínio e de chamadas de ferramentas
embutidos em mensagens antigas, os anexos de texto colado do antigo tipo `context`, o
antigo formato de lista das recomendações de MCP dispensadas nem o antigo campo
`webui_settings` do servidor. Para manter as conversas de uma versão mais antiga,
exporte-as a partir dela antes de atualizar e importe a exportação depois. A importação de conversas aceita apenas as exportações JSONL e ZIP, dentro dos limites listados
em [limites de segurança](#security-limits); o antigo formato de exportação JSON é rejeitado. As configurações antigas ficam no navegador até serem apagadas: para
removê-las, abra o console do navegador (F12) na página da interface e execute
`for (const k of Object.keys(localStorage)) if (k !== 'MgsUi.apiKey') localStorage.removeItem(k);`
uma vez para cada endereço e porta com que a interface foi aberta.

Os arquivos baixados com `-mu`/`--model-url` ficam em cache em `~/.cache/mgs/`; por isso,
os que as versões mais antigas salvaram na sua antiga pasta de cache são baixados de novo.
Os downloads do Hugging Face continuam usando o cache do Hugging Face (`~/.cache/huggingface/hub`
ou a pasta indicada em `MGS_CACHE`, `HF_HUB_CACHE` ou `HF_HOME`).

### Arquivos de modelo antigos

O código de compatibilidade com as conversões antigas a seguir também foi removido.
Para usá-las, converta de novo o modelo original ou baixe um GGUF atual.

| Arquivo antigo | Comportamento atual |
| --- | --- |
| Grok-1 convertido antes de agosto de 2025 (sem `grok.logit_scale`, `grok.embedding_scale`, `grok.attention.output_scale` ou `grok.attn_logit_softcapping`) | Não carrega: chave ausente. |
| Kimi-Linear com o tensor `blk.N.attn_kv_b` não dividido | Não carrega: faltam `attn_k_b`/`attn_v_b`. Esse caminho já falhava ao montar o grafo. |
| Arquivos do Qwen3-Next apenas com o tensor `ssm_in` mesclado | Não carrega: falta o tensor `blk.0.attn_qkv.weight`. |
| Arquivos sem `expert_gating_func` (`glm-dsa`, `glm4-moe`, `hy-v3`, `laguna`), `rope.scaling.finetuned` (Granite 4.0-H) ou `full_attention_interval` (Qwen3-Next), e arquivos do Cohere2-MoE com um único período de SWA em vez do array por camada | Não carrega: chave ausente. |
| Vocabulários BPE sem `tokenizer.ggml.pre` ou com os nomes de pré-tokenizador `llama3`, `llama-v3`, `jina-es`, `jina-de` ou `llada-moe`, e tokenizadores do Kimi-K2 sem merges | Não carrega: chave ausente, tipo de pré-tokenizador desconhecido ou merges ausentes. |
| Arquivos GGUF da versão 1 ou 2 | Não carrega: só a versão 3 do GGUF é aceita. |
| Arquivos mmproj dos antigos scripts avulsos: tipos de projetor `mlp`, `mlp_norm`, `ldp`, `ldpv2`, `resampler` (MiniCPM-V 2.x a 4.5, MiniCPM-o 2.6) e `adapter` (GLM-Edge-V), e arquivos sem tipo de projetor | O projetor não carrega: tipo de projetor desconhecido ou ausente. |
| Arquivos mmproj de conversões antigas com `ffn_up` e `ffn_down` trocados | Não é mais corrigido; o projetor falha. |
| mmproj do Kimi-K2.5 sem `image_min_pixels`/`image_max_pixels` e do Step3-VL sem `preproc_image_size` | O projetor não carrega: chave ou tensor ausente. |
| Arquivos que trazem apenas as antigas chaves `tokenizer.ggml.prefix_token_id`, `tokenizer.ggml.suffix_token_id` e `tokenizer.ggml.middle_token_id` em vez das chaves `tokenizer.ggml.fim_*_token_id` | Carrega; os tokens de preenchimento (infill) são reconhecidos pelo seu texto (por exemplo `<PRE>` ou `<\|fim_prefix\|>`), e o infill não funciona se eles não forem encontrados. |

Foram mantidos os mecanismos de respaldo de que as conversões atuais ou os GGUFs
publicados mais baixados das famílias compatíveis ainda dependem, conferidos nos
seus cabeçalhos no Hugging Face: o `add_bos` padrão dos pré-tokenizadores `llama-bpe`,
Tekken e semelhantes; o layout das camadas recorrentes do Qwen3.6 derivado de
`full_attention_interval` (os GGUFs do Qwen3.6 mais baixados não têm
`recurrent_layers`); e o layout integrado do indexador do GLM-5.2 para os arquivos
GLM-5.2/5.3 sem `attention.indexer.types`.

<a id="maintenance"></a>
## Manutenção e solução de problemas

- Verifique qual versão está instalada com `mgs --version` (ele mostra, por exemplo, `version: 0.1.0`); a compilação
  local o executa no final e o CMake mostra `mgs version: 0.1.0` durante a configuração. A interface mostra a mesma
  versão em Configurações → Exibição → **Mostrar informações da versão de compilação** (canto inferior direito) e em
  **Informações de compilação** na caixa de diálogo de informações do modelo. Cada mudança no projeto aumenta o último
  número: 0.1.0, 0.1.1, 0.1.2 etc.
- Repita o `curl | bash` para atualizar a partir da `main`, ou o comando de instalação local adequado para compilar de novo e substituir o binário.
  Depois, reinicie o serviço: `systemctl restart moe-gguf-server` no Debian
  ou `rc-service moe-gguf-server restart` no Alpine.
- Leia `/root/webapp-install.log`, `journalctl -u moe-gguf-server` no Debian
  ou `/var/log/moe-gguf-server.log` no Alpine. Os logs do Apache ficam em
  `/var/www/DOMAIN-logs/`.
- Se a compilação com CUDA não estiver disponível, instale o Toolkit por fora ou use
  `--cpu`. Se o NCCL não for encontrado, verifique a disponibilidade do repositório e a
  versão do CUDA, ou escolha explicitamente `--no-nccl`.
- Se a compilação da interface falhar, verifique as versões do Node.js e do npm, os
  downloads das dependências e o log. Não substitua a interface por uma interface
  pré-compilada do upstream.
- Para as compilações de implantação, use um diretório novo dentro de `_/temp/` depois de mudanças na estrutura do código-fonte.
  Uma cópia do projeto em SSHFS pode proibir executáveis; use um disco local na produção.
- Para a compilação local, execute de novo o comando de uma linha acima para baixar a `main` atual, ou
  `bash build/build-local.sh` para usar o seu código-fonte local, e reinicie a cópia que você
  iniciou manualmente. Leia a saída do terminal se ela falhar ou informar bibliotecas
  fora de `lib/`. O `build.log` temporário é apagado junto com o diretório de compilação.
- Os erros 404 da API em clientes mais antigos geralmente exigem mudar a URL base para
  `/api/v1`. A chave de API instalada vai no cabeçalho Authorization.

Pelas regras deste projeto, o assistente de programação não executa testes, implantações,
benchmarks de GPU nem verificações automatizadas em navegadores de celular. As
verificações estáticas de sintaxe não garantem o comportamento em tempo de execução.
Veja [CODE.pt-BR.md](CODE.pt-BR.md) para o mapa da implementação e os pontos de extensão.

<a id="license"></a>
## Licença

O `mgs` é distribuído sob a [PolyForm NonCommercial License 1.0.0](../LICENSE). Você pode usá-lo,
modificá-lo e compartilhá-lo para qualquer finalidade não comercial: uso pessoal, estudo, pesquisa,
projetos de hobby e uso por organizações beneficentes, instituições de ensino, organizações públicas
de pesquisa e instituições governamentais. O uso comercial, como o de uma empresa que o utilize no
seu negócio, precisa de uma licença separada de nipegun.

A licença cobre apenas as contribuições de nipegun. As partes que vêm do llama.cpp (MIT, The
ggml authors), do FreeToken (Apache 2.0) e das bibliotecas incluídas mantêm as suas próprias licenças;
os avisos delas estão em `build/licenses/` e a compilação local os instala em
`~/.local/share/doc/mgs/licenses/`.
