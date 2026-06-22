/* Tempos de execucao:
 * Versao Sequencial: Tempo: XXX segundos
 * OpenMP CPU: 1 thread: XXX segundos | 2 threads: XXX segundos | 4 threads: XXX segundos | 8 threads: XXX segundos | 16 threads: XXX segundos | 32 threads: XXX segundos
 * OpenMP GPU: Tempo: XXX segundos
 * CUDA GPU: Tempo: 0.678636 segundos
 * Speedup: ...
 */

/**
 * @file kmeans_cuda.cu
 * @brief K-Means Clustering - Versao PARALELA com CUDA (GPU)
 * @details
 * Versao do algoritmo K-Means implementada em CUDA C/C++. As partes
 * mais pesadas do algoritmo (calculo de distancias, atribuicao dos
 * pontos aos clusters e calculo dos centroides) sao convertidas em
 * kernels CUDA, executados na GPU.
 *
 * Kernels implementados:
 *   - assignClustersKernel: para cada ponto, calcula a distancia a
 *     todos os centroides e atribui o ponto ao cluster mais proximo
 *     (equivalente a calculateNearst do codigo original).
 *   - accumulateCentroidsKernel: soma as coordenadas dos pontos de
 *     cada cluster usando atomicAdd (equivalente ao PASSO 2 do
 *     algoritmo original).
 *
 * Fluxo geral (mantendo a logica original do K-Means):
 *   1. Atribuicao inicial aleatoria dos pontos aos k clusters (CPU).
 *   2. Copia dos dados para a GPU (cudaMemcpy).
 *   3. Loop de iteracoes:
 *      a. Zera acumuladores de centroides na GPU.
 *      b. Kernel accumulateCentroidsKernel: soma coordenadas por cluster.
 *      c. Calculo da media (centroide) - feito na GPU em kernel simples.
 *      d. Kernel assignClustersKernel: reatribui cada ponto ao cluster
 *         mais proximo e conta quantos pontos mudaram (atomicAdd).
 *      e. Copia o contador de mudancas para a CPU para verificar o
 *         criterio de parada.
 *   4. Copia o resultado final de volta para a CPU (cudaMemcpy).
 *
 * Compilacao:
 *   nvcc kmeans_cuda.cu -o kmeans_cuda
 *
 * @author Original: Lakhan Nad (https://github.com/Lakhan-Nad)
 * @author Adaptacao para Computacao Paralela (CUDA)
 */

#include <cfloat>   /* DBL_MAX */
#include <cmath>    /* sin, cos, M_PI */
#include <cstdio>   /* printf */
#include <cstdlib>  /* rand, malloc */
#include <cstring>  /* memset */
#include <ctime>    /* time */
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* Medicao de tempo PORTATIL (Windows/MinGW+nvcc e Linux/Mac)          */
/* ------------------------------------------------------------------ */
/* No Linux/Mac usamos clock_gettime(CLOCK_MONOTONIC, ...). No
 * Windows, a macro CLOCK_MONOTONIC pode nao estar disponivel
 * dependendo do runtime utilizado pelo nvcc/MSVC, por isso usamos a
 * API nativa QueryPerformanceCounter como alternativa, da mesma forma
 * que nas versoes kmeans_seq.c / kmeans_omp_cpu.c / kmeans_omp_gpu.c. */
#ifdef _WIN32
    #include <windows.h>
    static double get_time_seconds(void)
    {
        static LARGE_INTEGER freq;
        static int freq_initialized = 0;
        LARGE_INTEGER counter;
        if (!freq_initialized)
        {
            QueryPerformanceFrequency(&freq);
            freq_initialized = 1;
        }
        QueryPerformanceCounter(&counter);
        return (double)counter.QuadPart / (double)freq.QuadPart;
    }
#else
    static double get_time_seconds(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
    }
#endif

/* ------------------------------------------------------------------ */
/* Estruturas de dados (mantidas para compatibilidade com o codigo     */
/* original; o trabalho pesado e feito em vetores simples (SoA) que   */
/* sao mais adequados para transferencia e processamento na GPU)      */
/* ------------------------------------------------------------------ */

typedef struct observation
{
    double x;
    double y;
    int group;
} observation;

typedef struct cluster
{
    double x;
    double y;
    size_t count;
} cluster;

#define MAX_ITER 50

/* Configuracao padrao de kernel: 256 threads por bloco. Nao foi feita
   busca por configuracao otima, conforme solicitado - 256 e um valor
   classico e razoavel para a maioria das GPUs NVIDIA. */
#define BLOCK_SIZE 256

/* Macro simples para checagem de erros CUDA */
#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                           \
        {                                                                  \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

/* ==================================================================== */
/* KERNEL 1: Atribuicao de cada ponto ao centroide mais proximo          */
/* Equivalente a funcao calculateNearst() + PASSOS 3 e 4 do codigo       */
/* original, porem executado em paralelo: cada thread CUDA processa     */
/* um unico ponto.                                                      */
/* ==================================================================== */
__global__ void assignClustersKernel(const double* __restrict__ px,
                                      const double* __restrict__ py,
                                      int* __restrict__ pgroup,
                                      const double* __restrict__ ccx,
                                      const double* __restrict__ ccy,
                                      int k, size_t size,
                                      unsigned long long* changedCount)
{
    size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (j >= size) return;

    double bestDist = DBL_MAX;
    int bestIdx = -1;
    double x = px[j];
    double y = py[j];

    for (int i = 0; i < k; i++)
    {
        double dx = ccx[i] - x;
        double dy = ccy[i] - y;
        double dist = dx * dx + dy * dy;
        if (dist < bestDist)
        {
            bestDist = dist;
            bestIdx = i;
        }
    }

    if (bestIdx != pgroup[j])
    {
        pgroup[j] = bestIdx;
        /* atomicAdd evita condicao de corrida no contador global de
           pontos que mudaram de cluster nesta iteracao */
        atomicAdd(changedCount, 1ULL);
    }
}

/* ==================================================================== */
/* KERNEL 2: Acumulacao das coordenadas dos pontos por cluster           */
/* Equivalente ao PASSO 2 do codigo original (soma para depois          */
/* calcular a media/centroide). Cada thread processa um ponto e usa     */
/* atomicAdd para somar de forma segura na posicao do seu cluster.      */
/* ==================================================================== */
__global__ void accumulateCentroidsKernel(const double* __restrict__ px,
                                           const double* __restrict__ py,
                                           const int* __restrict__ pgroup,
                                           double* sumX, double* sumY,
                                           unsigned long long* count,
                                           size_t size)
{
    size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (j >= size) return;

    int g = pgroup[j];
    /* atomicAdd em double requer compute capability >= 6.0 (suportado
       nativamente pelo runtime CUDA moderno) */
    atomicAdd(&sumX[g], px[j]);
    atomicAdd(&sumY[g], py[j]);
    atomicAdd(&count[g], 1ULL);
}

/* ==================================================================== */
/* KERNEL 3: Calculo da media (centroide) a partir das somas acumuladas */
/* Trabalho pequeno (apenas k elementos), mas mantido na GPU para       */
/* evitar transferencias extras entre host e device a cada iteracao.    */
/* ==================================================================== */
__global__ void computeMeansKernel(double* ccx, double* ccy,
                                    const double* sumX, const double* sumY,
                                    const unsigned long long* count, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (count[i] > 0)
    {
        ccx[i] = sumX[i] / (double)count[i];
        ccy[i] = sumY[i] / (double)count[i];
    }
}

/*!
 * Funcao auxiliar (CPU) usada apenas para os casos triviais k<=1 e
 * k>=size, em que a paralelizacao nao traz beneficio relevante.
 */
static void calculateCentroidCPU(observation observations[], size_t size,
                                  cluster* centroid)
{
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (size_t i = 0; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

/*!
 *    --Algoritmo K-Means (versao CUDA)--
 * Mesma logica de alto nivel da versao sequencial/OpenMP, mas os
 * passos 2, 3 e 4 sao executados na GPU atraves dos kernels definidos
 * acima.
 *
 * @param observations  vetor de observacoes (host) a clusterizar
 * @param size  tamanho do vetor de observacoes
 * @param k  numero de clusters
 *
 * @returns ponteiro (host) para o vetor de clusters calculado
 */
cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;

    if (k <= 1)
    {
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroidCPU(observations, size, clusters);
        return clusters;
    }
    else if ((size_t)k >= size)
    {
        clusters = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (size_t j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = (int)j;
        }
        return clusters;
    }

    clusters = (cluster*)malloc(sizeof(cluster) * k);
    memset(clusters, 0, k * sizeof(cluster));

    /* ---- Preparacao dos dados no host (SoA: Struct of Arrays) ---- */
    double* h_px = (double*)malloc(sizeof(double) * size);
    double* h_py = (double*)malloc(sizeof(double) * size);
    int* h_pgroup = (int*)malloc(sizeof(int) * size);

    for (size_t j = 0; j < size; j++)
    {
        h_px[j] = observations[j].x;
        h_py[j] = observations[j].y;
        h_pgroup[j] = rand() % k; /* PASSO 1: atribuicao inicial aleatoria */
    }

    /* ---- Alocacao de memoria na GPU (cudaMalloc) ---- */
    double *d_px, *d_py;
    int* d_pgroup;
    double *d_ccx, *d_ccy;      /* centroides atuais */
    double *d_sumX, *d_sumY;    /* somas para calculo do centroide */
    unsigned long long* d_count;        /* contagem de pontos por cluster */
    unsigned long long* d_changedCount; /* contador de pontos que mudaram */

    CUDA_CHECK(cudaMalloc((void**)&d_px, sizeof(double) * size));
    CUDA_CHECK(cudaMalloc((void**)&d_py, sizeof(double) * size));
    CUDA_CHECK(cudaMalloc((void**)&d_pgroup, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&d_ccx, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_ccy, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_sumX, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_sumY, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_count, sizeof(unsigned long long) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_changedCount,
                           sizeof(unsigned long long)));

    /* ---- Copia dos dados de entrada para a GPU (cudaMemcpy H2D) ---- */
    CUDA_CHECK(cudaMemcpy(d_px, h_px, sizeof(double) * size,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, sizeof(double) * size,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pgroup, h_pgroup, sizeof(int) * size,
                           cudaMemcpyHostToDevice));
    /* Inicializa os centroides com zero (serao calculados na 1a iteracao) */
    CUDA_CHECK(cudaMemset(d_ccx, 0, sizeof(double) * k));
    CUDA_CHECK(cudaMemset(d_ccy, 0, sizeof(double) * k));

    size_t minAcceptedError = size / 10000;
    int iter = 0;
    unsigned long long changed = 0;

    int numBlocksPoints = (int)((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int numBlocksK = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocksK < 1) numBlocksK = 1;

    do
    {
        /* Zera acumuladores de centroides na GPU */
        CUDA_CHECK(cudaMemset(d_sumX, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_sumY, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long) * k));

        /* KERNEL: soma das coordenadas por cluster (PASSO 2) */
        accumulateCentroidsKernel<<<numBlocksPoints, BLOCK_SIZE>>>(
            d_px, d_py, d_pgroup, d_sumX, d_sumY, d_count, size);
        CUDA_CHECK(cudaGetLastError());

        /* KERNEL: calculo da media (centroide) a partir das somas */
        computeMeansKernel<<<numBlocksK, BLOCK_SIZE>>>(d_ccx, d_ccy, d_sumX,
                                                        d_sumY, d_count, k);
        CUDA_CHECK(cudaGetLastError());

        /* Zera o contador de pontos que mudaram de cluster */
        CUDA_CHECK(cudaMemset(d_changedCount, 0,
                               sizeof(unsigned long long)));

        /* KERNEL: distancia de cada ponto aos centroides + reatribuicao
           (PASSOS 3 e 4) */
        assignClustersKernel<<<numBlocksPoints, BLOCK_SIZE>>>(
            d_px, d_py, d_pgroup, d_ccx, d_ccy, k, size, d_changedCount);
        CUDA_CHECK(cudaGetLastError());

        /* Copia o contador de mudancas de volta para o host (D2H) para
           verificar o criterio de parada do loop */
        CUDA_CHECK(cudaMemcpy(&changed, d_changedCount,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());

        iter++;

        /* ---- LOG DE PROGRESSO (acompanhamento da execucao) ---- */
        printf("[CUDA] Iteracao %3d | pontos que mudaram de cluster: %llu\n",
               iter, changed);
    } while (changed > minAcceptedError && iter < MAX_ITER);

    /* ---- Copia dos resultados finais de volta para o host (D2H) ---- */
    CUDA_CHECK(cudaMemcpy(h_pgroup, d_pgroup, sizeof(int) * size,
                           cudaMemcpyDeviceToHost));

    double* h_ccx = (double*)malloc(sizeof(double) * k);
    double* h_ccy = (double*)malloc(sizeof(double) * k);
    unsigned long long* h_count =
        (unsigned long long*)malloc(sizeof(unsigned long long) * k);

    CUDA_CHECK(cudaMemcpy(h_ccx, d_ccx, sizeof(double) * k,
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ccy, d_ccy, sizeof(double) * k,
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_count, d_count,
                           sizeof(unsigned long long) * k,
                           cudaMemcpyDeviceToHost));

    /* Atualiza as estruturas originais (observation[] / cluster[]) para
       manter a interface igual ao restante do programa (printEPS) */
    for (size_t j = 0; j < size; j++)
    {
        observations[j].group = h_pgroup[j];
    }
    for (int i = 0; i < k; i++)
    {
        clusters[i].x = h_ccx[i];
        clusters[i].y = h_ccy[i];
        clusters[i].count = (size_t)h_count[i];
    }

    /* ---- LOG: estado final dos centroides ---- */
    printf("[CUDA] Convergencia apos %d iteracoes. Centroides finais:\n", iter);
    for (int i = 0; i < k; i++)
    {
        printf("[CUDA]   Cluster %d -> x=%.4f, y=%.4f, pontos=%zu\n",
               i, clusters[i].x, clusters[i].y, clusters[i].count);
    }

    /* ---- Liberacao da memoria na GPU (cudaFree) ---- */
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_pgroup));
    CUDA_CHECK(cudaFree(d_ccx));
    CUDA_CHECK(cudaFree(d_ccy));
    CUDA_CHECK(cudaFree(d_sumX));
    CUDA_CHECK(cudaFree(d_sumY));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_changedCount));

    /* ---- Liberacao da memoria no host ---- */
    free(h_px);
    free(h_py);
    free(h_pgroup);
    free(h_ccx);
    free(h_ccy);
    free(h_count);

    return clusters;
}

/*!
 * Gera a saida em formato EPS (mantida do codigo original).
 */
void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = -DBL_MAX, min_y = DBL_MAX, max_y = -DBL_MAX;
    double scale = 0, cx = 0, cy = 0;
    double* colors = (double*)malloc(sizeof(double) * (k * 3));
    int i;
    size_t j;
    double kd = k * 1.0;
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x) max_x = pts[j].x;
        if (min_x > pts[j].x) min_x = pts[j].x;
        if (max_y < pts[j].y) max_y = pts[j].y;
        if (min_y > pts[j].y) min_y = pts[j].y;
    }
    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    }
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    FILE* out = fopen("output_cuda.eps", "w");
    if (!out)
    {
        free(colors);
        return;
    }

    fprintf(out, "%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n",
            W + 10, H + 10);
    fprintf(
        out,
        "/l {rlineto} def /m {rmoveto} def\n"
        "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
        "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
        "	gsave 1 setgray fill grestore gsave 3 setlinewidth"
        " 1 setgray stroke grestore 0 setgray stroke }def\n");
    for (i = 0; i < k; i++)
    {
        fprintf(out, "%g %g %g setrgbcolor\n", *(colors + 3 * i),
                *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i) continue;
            fprintf(out, "%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2,
                    (pts[j].y - cy) * scale + H / 2);
        }
        fprintf(out, "\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2,
                (cent[i].y - cy) * scale + H / 2);
    }
    fprintf(out, "\n%%%%EOF");
    fclose(out);

    /* ---- LOG: confirmacao de arquivo salvo ---- */
    printf("[CUDA] Arquivo 'output_cuda.eps' salvo com sucesso (%zu pontos, %d clusters).\n",
           len, k);

    free(colors);
}

/*!
 * Funcao de teste: gera N pontos em um circulo de raio 20.0
 * centrado em (0,0) e agrupa em k clusters usando a versao CUDA
 * do K-Means.
 */
static void test()
{
    size_t size = 15000000L; /* mesmo tamanho das demais versoes, para
                                 comparacao direta de desempenho */

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("[CUDA] GPUs disponiveis: %d\n", deviceCount);
    if (deviceCount > 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("[CUDA] Usando dispositivo 0: %s\n", prop.name);
    }

    printf("[CUDA] Gerando %zu pontos aleatorios...\n", size);

    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
    if (!observations)
    {
        fprintf(stderr, "Erro: falha ao alocar memoria para observacoes\n");
        exit(1);
    }

    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }

    int k = 5;

    printf("[CUDA] Dados gerados. Iniciando K-Means com k=%d...\n", k);

    /* ---------------- MEDICAO DE TEMPO ---------------- */
    double start = get_time_seconds();

    cluster* clusters = kMeans(observations, size, k);

    double end = get_time_seconds();
    double elapsed = end - start;
    printf("[CUDA] Tempo de execucao do K-Means: %.6f segundos\n", elapsed);
    /* ---------------------------------------------------- */

    printEPS(observations, size, clusters, k);

    free(observations);
    free(clusters);
}

int main()
{
    srand((unsigned int)time(NULL));
    test();
    return 0;
}
