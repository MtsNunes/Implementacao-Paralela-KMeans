# Implementacao-Paralela-KMeans

================================================================================
 PROJETO: K-MEANS CLUSTERING - COMPUTACAO PARALELA
 Versoes: Sequencial | OpenMP CPU | OpenMP GPU (Target Offload) | CUDA
================================================================================

--------------------------------------------------------------------------------
1. EXPLICACAO RESUMIDA DA APLICACAO
--------------------------------------------------------------------------------

Este projeto implementa o algoritmo de agrupamento (clustering) K-Means em
quatro versoes diferentes, com o objetivo de comparar o desempenho de
execucao sequencial e paralela em CPU e GPU:

    1. kmeans_seq.c      -> Versao sequencial (baseline)
    2. kmeans_omp_cpu.c  -> Versao paralela com OpenMP (multithread em CPU)
    3. kmeans_omp_gpu.c  -> Versao paralela com OpenMP Target Offloading (GPU)
    4. kmeans_cuda.cu    -> Versao paralela com CUDA (kernels em GPU)

Todas as quatro versoes implementam exatamente o mesmo algoritmo e geram
o mesmo tipo de resultado (clusters de pontos 2D), variando apenas a forma
como o trabalho computacional e distribuido entre os nucleos da CPU ou da
GPU. Isso permite calcular o speedup (ganho de desempenho) de cada versao
paralela em relacao a versao sequencial.

O cenario de teste utilizado em todas as versoes consiste em gerar
20.000.000 (20 milhoes) de pontos aleatorios distribuidos dentro de um
circulo de raio 20.0 centrado na origem (0,0), e agrupa-los em k = 5
clusters. Esse tamanho de entrada foi escolhido para que a versao
sequencial leve aproximadamente 10 segundos para executar em uma maquina
de uso comum (ajuste o valor de "size" dentro da funcao test() de cada
arquivo, se necessario, para atingir esse tempo na sua maquina).

Cada versao gera, ao final da execucao, um arquivo de saida no formato
EPS (PostScript) com a visualizacao dos clusters encontrados:
    - output_seq.eps      (versao sequencial)
    - output_omp_cpu.eps  (versao OpenMP CPU)
    - output_omp_gpu.eps  (versao OpenMP GPU)
    - output_cuda.eps     (versao CUDA)


--------------------------------------------------------------------------------
2. EXPLICACAO DO ALGORITMO K-MEANS
--------------------------------------------------------------------------------

O K-Means e um algoritmo de aprendizado nao supervisionado utilizado para
agrupar um conjunto de pontos (observacoes) em K grupos (clusters), de
forma que pontos no mesmo grupo sejam o mais semelhantes possivel entre
si (no nosso caso, "semelhante" significa estar geometricamente proximo).

O algoritmo segue os seguintes passos, de forma iterativa:

    PASSO 1 - Inicializacao:
        Cada ponto e atribuido aleatoriamente a um dos K clusters.

    PASSO 2 - Calculo dos centroides:
        Para cada cluster, calcula-se o centroide, que e a media das
        coordenadas (x, y) de todos os pontos atualmente pertencentes
        a aquele cluster.

    PASSO 3 - Calculo da distancia:
        Para cada ponto, calcula-se a distancia (euclidiana, ao quadrado,
        para evitar a raiz quadrada) entre o ponto e cada um dos K
        centroides.

    PASSO 4 - Reatribuicao:
        Cada ponto e reatribuido ao cluster cujo centroide esteja mais
        proximo dele.

    PASSO 5 - Convergencia:
        Os passos 2, 3 e 4 sao repetidos ate que a quantidade de pontos
        que mudam de cluster entre uma iteracao e outra seja pequena (ou
        nula), indicando que o algoritmo convergiu, ou ate atingir um
        numero maximo de iteracoes (MAX_ITER, definido como 50 neste
        projeto para tornar os testes de desempenho comparaveis entre as
        diferentes versoes).

As etapas 2, 3 e 4 sao as que possuem maior custo computacional, pois sao
executadas para todos os N pontos a cada iteracao (e, no passo 3, para
cada um dos K clusters). Por isso, sao essas as etapas escolhidas para
paralelizacao nas versoes OpenMP CPU, OpenMP GPU e CUDA.


--------------------------------------------------------------------------------
3. COMO COMPILAR CADA VERSAO
--------------------------------------------------------------------------------

3.1 Versao Sequencial
----------------------
    gcc kmeans_seq.c -o kmeans_seq -lm

(Opcional, recomendado para benchmark: adicionar otimizacao -O2)
    gcc -O2 kmeans_seq.c -o kmeans_seq -lm


3.2 Versao OpenMP CPU
----------------------
    gcc kmeans_omp_cpu.c -o kmeans_omp_cpu -fopenmp -lm

(Opcional, recomendado para benchmark: adicionar otimizacao -O2)
    gcc -O2 kmeans_omp_cpu.c -o kmeans_omp_cpu -fopenmp -lm


3.3 Versao OpenMP GPU (Target Offloading)
------------------------------------------
    gcc kmeans_omp_gpu.c -o kmeans_omp_gpu -fopenmp -foffload=nvptx-none -fno-lto -lm

Requisitos:
    - GCC com suporte a offload para NVIDIA (plugin nvptx instalado).
      Em distribuicoes Ubuntu/Debian isso geralmente requer o pacote
      "gcc-offload-nvptx" (ou compilar um GCC com suporte a offload).
    - Driver NVIDIA e CUDA Toolkit instalados na maquina.

Caso o compilador/maquina nao tenha suporte a offload para GPU
disponivel, o codigo ainda pode ser compilado sem a flag "-foffload",
apenas com "-fopenmp". Nesse caso, as regioes "target" serao executadas
como fallback na propria CPU (host), permitindo testar a corretude da
logica mesmo sem uma GPU disponivel, ainda que sem o ganho de
desempenho de uma GPU real:
    gcc kmeans_omp_gpu.c -o kmeans_omp_gpu -fopenmp -lm


3.4 Versao CUDA
----------------
    nvcc kmeans_cuda.cu -o kmeans_cuda -arch=sm_86 -lm

Requisitos:
    - NVIDIA CUDA Toolkit (nvcc) instalado.
    - GPU NVIDIA com driver compativel.

(Opcional, recomendado para benchmark: adicionar otimizacao -O2)
    nvcc -O2 kmeans_cuda.cu -o kmeans_cuda


--------------------------------------------------------------------------------
4. COMO EXECUTAR CADA VERSAO
--------------------------------------------------------------------------------

4.1 Versao Sequencial
----------------------
    ./kmeans_seq

Saida esperada (exemplo):
    [SEQUENCIAL] Tempo de execucao do K-Means: 10.234567 segundos

Gera o arquivo: output_seq.eps


4.2 Versao OpenMP CPU
----------------------
A quantidade de threads utilizadas e controlada pela variavel de
ambiente OMP_NUM_THREADS. Execute uma vez para cada quantidade de
threads exigida pelo trabalho (1, 2, 4, 8, 16, 32):

    OMP_NUM_THREADS=1  ./kmeans_omp_cpu
    OMP_NUM_THREADS=2  ./kmeans_omp_cpu
    OMP_NUM_THREADS=4  ./kmeans_omp_cpu
    OMP_NUM_THREADS=8  ./kmeans_omp_cpu
    OMP_NUM_THREADS=16 ./kmeans_omp_cpu
    OMP_NUM_THREADS=32 ./kmeans_omp_cpu

Saida esperada (exemplo):
    [OPENMP CPU] Executando com OMP_NUM_THREADS = 8
    [OPENMP CPU] Tempo de execucao do K-Means (8 threads): 1.456789 segundos

Gera o arquivo: output_omp_cpu.eps (sobrescrito a cada execucao)

Obs.: se a maquina utilizada possuir menos nucleos fisicos/logicos do
que a quantidade de threads solicitada (ex.: 16 ou 32 threads em uma
CPU com 8 nucleos), o sistema operacional fara o escalonamento
(time-slicing) das threads, e o ganho de desempenho tende a estagnar
ou ate piorar a partir do numero de nucleos disponiveis - isso e um
resultado esperado e relevante para a analise do trabalho.


4.3 Versao OpenMP GPU
----------------------
    ./kmeans_omp_gpu

Saida esperada (exemplo):
    [OPENMP GPU] Dispositivos de offload disponiveis: 1
    [OPENMP GPU] Tempo de execucao do K-Means: 0.987654 segundos

Gera o arquivo: output_omp_gpu.eps

Obs.: a linha "Dispositivos de offload disponiveis" informa quantas
GPUs foram detectadas pelo runtime do OpenMP (via omp_get_num_devices()).
Se o valor for 0, a execucao ocorrera via fallback na CPU.


4.4 Versao CUDA
-----------------
    ./kmeans_cuda

Saida esperada (exemplo):
    [CUDA] GPUs disponiveis: 1
    [CUDA] Usando dispositivo 0: NVIDIA GeForce RTX 3060
    [CUDA] Tempo de execucao do K-Means: 0.543210 segundos

Gera o arquivo: output_cuda.eps


--------------------------------------------------------------------------------
5. EXPLICACAO DAS MEDICOES DE TEMPO
--------------------------------------------------------------------------------

Em todas as versoes, o tempo medido corresponde exclusivamente a execucao
do algoritmo K-Means (funcao kMeans()), nao incluindo o tempo de geracao
dos pontos de entrada nem o tempo de escrita do arquivo de saida (.eps),
para que a comparacao entre as versoes seja justa.

    - Nas versoes em C puro (sequencial e OpenMP CPU/GPU), o tempo e
      medido usando a funcao clock_gettime() com o relogio
      CLOCK_MONOTONIC, que e o metodo recomendado em sistemas Linux por
      nao ser afetado por ajustes do relogio do sistema (NTP, etc.) e
      ter alta resolucao (nanosegundos).

    - Na versao CUDA (kmeans_cuda.cu), tambem e usado clock_gettime()
      na CPU (host), porem chamando cudaDeviceSynchronize() apos os
      kernels para garantir que o tempo medido reflita o momento em
      que a GPU efetivamente terminou todo o processamento (sem essa
      sincronizacao, o tempo medido na CPU poderia ser menor do que o
      tempo real de execucao na GPU, pois chamadas de kernel sao
      assincronas por padrao).

Apos executar cada versao, anote os tempos impressos no terminal e
preencha o cabecalho de comentario no topo de cada arquivo de codigo
(kmeans_seq.c, kmeans_omp_cpu.c, kmeans_omp_gpu.c, kmeans_cuda.cu), que
segue o seguinte modelo:

    /* Tempos de execucao:
     * Versao Sequencial: Tempo: XXX segundos
     * OpenMP CPU: 1 thread: XXX segundos | 2 threads: XXX segundos | ...
     * OpenMP GPU: Tempo: XXX segundos
     * CUDA GPU: Tempo: XXX segundos
     * Speedup: ...
     */


--------------------------------------------------------------------------------
6. EXPLICACAO DO CALCULO DE SPEEDUP
--------------------------------------------------------------------------------

O speedup mede o quanto uma versao paralela e mais rapida do que a
versao sequencial (baseline), para a mesma entrada de dados. A formula
basica e:

    Speedup = Tempo_Sequencial / Tempo_Paralelo

Exemplos de aplicacao no contexto deste trabalho:

    - Speedup do OpenMP CPU com N threads:
          Speedup(N) = Tempo_Sequencial / Tempo_OpenMP_CPU(N threads)

      Por exemplo, se a versao sequencial levou 10s e a versao OpenMP
      CPU com 8 threads levou 1.5s:
          Speedup(8) = 10 / 1.5 = 6.67x

    - Speedup do OpenMP GPU:
          Speedup = Tempo_Sequencial / Tempo_OpenMP_GPU

    - Speedup do CUDA:
          Speedup = Tempo_Sequencial / Tempo_CUDA

Tambem e interessante calcular a eficiencia da paralelizacao em CPU, que
relaciona o speedup obtido com o numero de threads utilizado:

    Eficiencia(N) = Speedup(N) / N

Uma eficiencia proxima de 1.0 (ou 100%) indica que a paralelizacao esta
aproveitando bem os N nucleos disponiveis. Valores bem menores que 1.0
geralmente indicam que ha overhead de sincronizacao, contencao de
memoria (cache/memory bandwidth) ou que o numero de threads excedeu o
numero de nucleos fisicos/logicos disponiveis na maquina (o que e
esperado ao testar com 16 ou 32 threads em CPUs com menos nucleos).

Ao final dos testes, monte uma tabela (ou grafico) com:
    - Tempo sequencial
    - Tempo OpenMP CPU para 1, 2, 4, 8, 16 e 32 threads
    - Tempo OpenMP GPU
    - Tempo CUDA
    - Speedup de cada versao em relacao a sequencial
    - (Opcional) Eficiencia da versao OpenMP CPU para cada numero de threads

Essa tabela e o principal resultado a ser entregue e discutido no
relatorio do trabalho de Computacao Paralela.


--------------------------------------------------------------------------------
7. ESTRUTURA DO PROJETO
--------------------------------------------------------------------------------

    /projeto
    |
    |-- kmeans_seq.c       (Versao sequencial)
    |-- kmeans_omp_cpu.c   (Versao paralela OpenMP - CPU)
    |-- kmeans_omp_gpu.c   (Versao paralela OpenMP Target Offload - GPU)
    |-- kmeans_cuda.cu     (Versao paralela CUDA - GPU)
    |-- README.txt         (este arquivo)


--------------------------------------------------------------------------------
8. OBSERVACOES FINAIS
--------------------------------------------------------------------------------

    - O tamanho da entrada (variavel "size" dentro da funcao test() em
      cada arquivo) esta definido como 20.000.000 de pontos. Caso a
      versao sequencial nao atinja aproximadamente 10 segundos na sua
      maquina, ajuste esse valor (aumente para mais tempo, diminua para
      menos tempo) de forma IGUAL em todos os quatro arquivos, para que
      a comparacao entre as versoes permaneca justa.

    - O numero de clusters utilizado nos testes e k = 5, e o numero
      maximo de iteracoes do algoritmo foi fixado em MAX_ITER = 50 em
      todas as versoes, para garantir que o tempo de execucao seja
      determinado essencialmente pelo custo computacional por iteracao
      (e nao por variacoes aleatorias na quantidade de iteracoes
      necessarias para convergir).

    - Todas as versoes preservam a logica original do algoritmo K-Means
      (atribuicao inicial aleatoria, calculo de centroides por media,
      atribuicao de cada ponto ao centroide mais proximo por distancia
      euclidiana ao quadrado), alterando apenas a forma de execucao
      (sequencial, multithread em CPU, offload para GPU via OpenMP, ou
      kernels CUDA).

    - Os testes foram todos realizados em um ambiente WSL Ubuntu com as 
      seguintes configurações:
      Versão do WSL: 2.7.8.0
      Versão do kernel: 6.18.33.1-1
      Versão do WSLg: 1.0.73.2
      Versão do MSRDC: 1.2.6676
      Versão do Direct3D: 1.611.1-81528511
      Versão do DXCore: 10.0.26100.1-240331-1435.ge-release
      Versão do Windows: 10.0.26200.8655
