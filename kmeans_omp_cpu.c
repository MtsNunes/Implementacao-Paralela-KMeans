/* Tempos de execucao:
 * Versao Sequencial: Tempo: XXX segundos
 * OpenMP CPU: 1 thread: 5.760341 segundos | 2 threads: 8.286884 segundos | 4 threads: 4.358011 segundos | 8 threads: 3.028210 segundos | 16 threads: 1.906000 segundos | 32 threads: 2.383657 segundos
 * OpenMP GPU: Tempo: XXX segundos
 * CUDA GPU: Tempo: XXX segundos
 * Speedup: ...
 */

/**
 * @file kmeans_omp_cpu.c
 * @brief K-Means Clustering - Versao PARALELA com OpenMP (CPU)
 * @details
 * Versao paralela do algoritmo K-Means utilizando OpenMP para
 * distribuir o trabalho entre multiplas threads de CPU.
 *
 * As partes paralelizadas sao as de maior custo computacional:
 *   - calculo da soma de coordenadas por cluster (centroides);
 *   - calculo da distancia de cada ponto ao centroide mais proximo;
 *   - reatribuicao dos pontos aos clusters.
 *
 * A logica original do algoritmo K-Means foi preservada; apenas foram
 * adicionadas diretivas OpenMP (#pragma omp ...) nos lacos mais
 * custosos, com uso de reduction para evitar condicoes de corrida.
 *
 * Numero de threads pode ser controlado via variavel de ambiente:
 *   OMP_NUM_THREADS=N ./kmeans_omp_cpu
 *
 * @author Original: Lakhan Nad (https://github.com/Lakhan-Nad)
 * @author Adaptacao para Computacao Paralela (OpenMP CPU)
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */
#include <omp.h>          /* OpenMP, inclui omp_get_wtime() */

/* ------------------------------------------------------------------ */
/* Medicao de tempo: usamos omp_get_wtime(), fornecida pela propria
 * biblioteca OpenMP. Ela mede tempo de parede (wall-clock) em
 * segundos (double) e funciona de forma identica e portavel em
 * Windows (MinGW), Linux e Mac, sem depender de CLOCK_MONOTONIC
 * (que nao esta sempre disponivel no MinGW) nem de APIs especificas
 * de sistema operacional. */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Estruturas de dados (iguais ao algoritmo original)                  */
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

#define MAX_ITER 50 /* numero fixo de iteracoes, mesma justificativa da
                        versao sequencial: torna os benchmarks comparaveis */

/*!
 * Retorna o indice do centroide mais proximo da observacao dada.
 * Funcao usada dentro da regiao paralela (chamada por cada thread).
 */
int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

/*!
 * Calcula o centroide de um conjunto de observacoes (usado apenas
 * para o caso k<=1, baixo custo, mantido sequencial).
 */
void calculateCentroid(observation observations[], size_t size,
                        cluster* centroid)
{
    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

/*!
 *    --Algoritmo K-Means (versao paralela OpenMP CPU)--
 * Mesma logica da versao sequencial, mas os lacos mais custosos
 * (somatorio de coordenadas por cluster e busca do centroide mais
 * proximo) sao paralelizados com OpenMP.
 *
 * @param observations  vetor de observacoes a clusterizar
 * @param size  tamanho do vetor de observacoes
 * @param k  numero de clusters
 *
 * @returns ponteiro para o vetor de clusters calculado
 */
cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if ((size_t)k < size)
    {
        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));

        /* PASSO 1: atribuicao inicial aleatoria (sequencial - rand() nao
           e thread-safe de forma simples e o custo aqui e baixo) */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        size_t changed = 0;
        size_t minAcceptedError = size / 10000;
        int iter = 0;

        /* Buffers auxiliares para reducao manual por thread, evitando
           condicoes de corrida no acumulo dos centroides. Cada thread
           acumula em sua propria copia local e, no final, os resultados
           sao combinados (reducao). */
        int max_threads = omp_get_max_threads();
        double* local_x = malloc(sizeof(double) * (size_t)max_threads * k);
        double* local_y = malloc(sizeof(double) * (size_t)max_threads * k);
        size_t* local_count =
            malloc(sizeof(size_t) * (size_t)max_threads * k);

        do
        {
            /* Inicializa os clusters */
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            /* Zera os buffers locais de cada thread */
            memset(local_x, 0, sizeof(double) * (size_t)max_threads * k);
            memset(local_y, 0, sizeof(double) * (size_t)max_threads * k);
            memset(local_count, 0,
                   sizeof(size_t) * (size_t)max_threads * k);

            /* ============================================================
             * PARALELIZACAO 1: PASSO 2 - soma das coordenadas por cluster
             * para calculo do centroide.
             * Cada thread acumula em sua area privada (local_x/local_y/
             * local_count) indexada por thread_id*k + cluster_id, evitando
             * assim qualquer condicao de corrida (race condition) sem
             * precisar de "critical"/"atomic" dentro do laco principal.
             * ============================================================ */
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double* my_x = local_x + (size_t)tid * k;
                double* my_y = local_y + (size_t)tid * k;
                size_t* my_count = local_count + (size_t)tid * k;

                #pragma omp for schedule(static)
                for (size_t j = 0; j < size; j++)
                {
                    int t = observations[j].group;
                    my_x[t] += observations[j].x;
                    my_y[t] += observations[j].y;
                    my_count[t]++;
                }
            }

            /* Combina (reduz) os resultados parciais de cada thread nos
               clusters finais. Esta etapa e barata (max_threads * k) e
               feita sequencialmente. */
            for (int tid = 0; tid < max_threads; tid++)
            {
                double* my_x = local_x + (size_t)tid * k;
                double* my_y = local_y + (size_t)tid * k;
                size_t* my_count = local_count + (size_t)tid * k;
                for (int i = 0; i < k; i++)
                {
                    clusters[i].x += my_x[i];
                    clusters[i].y += my_y[i];
                    clusters[i].count += my_count[i];
                }
            }

            for (int i = 0; i < k; i++)
            {
                if (clusters[i].count > 0)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }

            /* ============================================================
             * PARALELIZACAO 2: PASSOS 3 e 4 - calculo da distancia de
             * cada ponto a todos os centroides (calculateNearst) e
             * reatribuicao do ponto ao cluster mais proximo.
             * Cada iteracao do laco "j" e independente entre si (so
             * depende de observations[j] e do vetor "clusters", que e
             * somente leitura aqui), portanto e seguro paralelizar com
             * #pragma omp parallel for.
             * Usamos "reduction(+:changed)" para contar quantos pontos
             * mudaram de cluster sem condicao de corrida.
             * ============================================================ */
            changed = 0;
            #pragma omp parallel for schedule(static) reduction(+:changed)
            for (size_t j = 0; j < size; j++)
            {
                int t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }

            iter++;

            /* ---- LOG DE PROGRESSO (acompanhamento da execucao) ---- */
            printf("[OPENMP CPU] Iteracao %3d | pontos que mudaram de cluster: %zu\n",
                   iter, changed);
        } while (changed > minAcceptedError && iter < MAX_ITER);

        /* ---- LOG: estado final dos centroides ---- */
        printf("[OPENMP CPU] Convergencia apos %d iteracoes. Centroides finais:\n",
               iter);
        for (int i = 0; i < k; i++)
        {
            printf("[OPENMP CPU]   Cluster %d -> x=%.4f, y=%.4f, pontos=%zu\n",
                   i, clusters[i].x, clusters[i].y, clusters[i].count);
        }

        free(local_x);
        free(local_y);
        free(local_count);
    }
    else
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
    }
    return clusters;
}

/*!
 * Gera a saida em formato EPS (mantida do codigo original, sequencial
 * por natureza - escrita de arquivo nao se beneficia de paralelizacao).
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

    FILE* out = fopen("output_omp_cpu.eps", "w");
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
    printf("[OPENMP CPU] Arquivo 'output_omp_cpu.eps' salvo com sucesso (%zu pontos, %d clusters).\n",
           len, k);

    free(colors);
}

/*!
 * Funcao de teste: gera N pontos em um circulo de raio 20.0
 * centrado em (0,0) e agrupa em k clusters usando a versao
 * paralela (OpenMP) do K-Means.
 */
static void test()
{
    size_t size = 15000000L; /* mesmo tamanho da versao sequencial, para
                                 que as medicoes de tempo sejam comparaveis */

    printf("[OPENMP CPU] Executando com OMP_NUM_THREADS = %d\n",
           omp_get_max_threads());
    printf("[OPENMP CPU] Gerando %zu pontos aleatorios...\n", size);

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

    printf("[OPENMP CPU] Dados gerados. Iniciando K-Means com k=%d...\n", k);

    /* ---------------- MEDICAO DE TEMPO ---------------- */
    double start = omp_get_wtime();

    cluster* clusters = kMeans(observations, size, k);

    double end = omp_get_wtime();
    double elapsed = end - start;
    printf("[OPENMP CPU] Tempo de execucao do K-Means (%d threads): %.6f segundos\n",
           omp_get_max_threads(), elapsed);
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
