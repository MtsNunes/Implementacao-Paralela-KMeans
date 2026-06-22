/* Tempos de execucao:
 * Versao Sequencial: Tempo: XXX segundos
 * OpenMP CPU: 1 thread: XXX segundos | 2 threads: XXX segundos | 4 threads: XXX segundos | 8 threads: XXX segundos | 16 threads: XXX segundos | 32 threads: XXX segundos
 * OpenMP GPU: Tempo: XXX segundos
 * CUDA GPU: Tempo: 2.153399 segundos
 * Speedup: ...
 */

/**
 * @file kmeans_omp_gpu.c
 * @brief K-Means Clustering - Versao PARALELA com OpenMP Target Offloading (GPU)
 * @details
 * Versao do algoritmo K-Means que utiliza as diretivas de offloading do
 * OpenMP (#pragma omp target ...) para executar as partes mais pesadas
 * do algoritmo na GPU:
 *   - calculo da soma de coordenadas por cluster (centroides);
 *   - calculo da distancia de cada ponto ao centroide mais proximo e
 *     reatribuicao do ponto ao cluster correspondente.
 *
 * Os dados (vetores de pontos x,y,group e os centroides) sao copiados
 * para a GPU com "map(to:...)" / "map(tofrom:...)" e os resultados sao
 * trazidos de volta para a CPU ao final de cada iteracao, de forma
 * semelhante ao que e feito explicitamente com cudaMemcpy na versao
 * CUDA.
 *
 * A logica original do algoritmo K-Means foi preservada.
 *
 * Compilacao (necessita de um compilador com suporte a offload nvptx,
 * por exemplo GCC com plugin nvptx instalado):
 *   gcc kmeans_omp_gpu.c -o kmeans_omp_gpu -fopenmp -foffload=nvptx-none -lm
 *
 * @author Original: Lakhan Nad (https://github.com/Lakhan-Nad)
 * @author Adaptacao para Computacao Paralela (OpenMP GPU - Target Offload)
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */
#include <omp.h>          /* OpenMP, inclui omp_get_wtime() */
/* Medicao de tempo via omp_get_wtime() (wall-clock, portavel entre
 * Windows/MinGW, Linux e Mac) - ver justificativa no arquivo
 * kmeans_omp_cpu.c */

/* ------------------------------------------------------------------ */
/* Estruturas de dados                                                  */
/* ------------------------------------------------------------------ */
/* Observacao: para facilitar o offload (map) de vetores para a GPU,
   os dados das observacoes e dos clusters sao tratados tambem como
   vetores "Struct of Arrays" (SoA) dentro do kMeans, em paralelo as
   estruturas originais "Array of Structs" (AoS), que sao mantidas
   para preservar a interface original (printEPS, etc). */

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

/*!
 * Versao da funcao calculateNearst usada na CPU (mantida para
 * compatibilidade / casos com k<=1 ou k>=size).
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
 *    --Algoritmo K-Means (versao OpenMP Target Offload / GPU)--
 *
 * As mesmas 4 etapas da versao sequencial sao mantidas, porem os
 * passos 2 (soma para centroides) e 3/4 (distancia + reatribuicao)
 * sao executados na GPU atraves de "#pragma omp target teams
 * distribute parallel for".
 *
 * Para isso, os dados das observacoes sao extraidos para vetores
 * simples (px, py, pgroup) e os centroides para (cx, cy, ccount),
 * pois o offload de dados costuma funcionar melhor (e de forma mais
 * portavel) com arrays de tipos simples do que com arrays de structs.
 */
cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        clusters = (cluster*)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
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

    clusters = malloc(sizeof(cluster) * k);
    memset(clusters, 0, k * sizeof(cluster));

    /* ---- Vetores "Struct of Arrays" usados no offload para GPU ---- */
    double* px = malloc(sizeof(double) * size);
    double* py = malloc(sizeof(double) * size);
    int* pgroup = malloc(sizeof(int) * size);

    double* ccx = malloc(sizeof(double) * k);
    double* ccy = malloc(sizeof(double) * k);
    long* ccount = malloc(sizeof(long) * k); /* long para reduction OpenMP */

    for (size_t j = 0; j < size; j++)
    {
        px[j] = observations[j].x;
        py[j] = observations[j].y;
        pgroup[j] = rand() % k; /* PASSO 1: atribuicao inicial aleatoria */
    }

    size_t changed = 0;
    size_t minAcceptedError = size / 10000;
    int iter = 0;

    /* Buffers de reducao por cluster: para cada um dos k clusters
       fazemos uma reducao separada (soma de x, soma de y, contagem). */
    do
    {
        for (int i = 0; i < k; i++)
        {
            ccx[i] = 0.0;
            ccy[i] = 0.0;
            ccount[i] = 0;
        }

        /* ================================================================
         * PARALELIZACAO/OFFLOAD 1 (GPU): PASSO 2 - soma das coordenadas
         * por cluster para calculo dos centroides.
         *
         * - "target teams distribute parallel for" distribui o laco
         *   entre equipes (teams) e threads na GPU.
         * - "map(to: px[0:size], py[0:size], pgroup[0:size])" copia os
         *   dados de entrada (somente leitura) para a GPU.
         * - "map(tofrom: ccx[0:k], ccy[0:k], ccount[0:k])" copia os
         *   acumuladores para a GPU e traz o resultado de volta.
         * - "reduction(+:ccx[:k], ccy[:k], ccount[:k])" realiza a soma
         *   de forma segura entre todas as threads/teams, evitando
         *   condicoes de corrida sem necessidade de atomics manuais.
         * ================================================================ */
        #pragma omp target teams distribute parallel for \
            map(to: px[0:size], py[0:size], pgroup[0:size]) \
            map(tofrom: ccx[0:k], ccy[0:k], ccount[0:k]) \
            reduction(+:ccx[:k], ccy[:k], ccount[:k])
        for (size_t j = 0; j < size; j++)
        {
            int g = pgroup[j];
            ccx[g] += px[j];
            ccy[g] += py[j];
            ccount[g] += 1;
        }

        for (int i = 0; i < k; i++)
        {
            if (ccount[i] > 0)
            {
                ccx[i] /= (double)ccount[i];
                ccy[i] /= (double)ccount[i];
            }
        }

        /* ================================================================
         * PARALELIZACAO/OFFLOAD 2 (GPU): PASSOS 3 e 4 - calculo da
         * distancia de cada ponto a cada centroide e reatribuicao do
         * ponto ao cluster mais proximo.
         *
         * - Cada iteracao "j" e completamente independente, ideal para
         *   GPU (laco de grao fino executado massivamente em paralelo).
         * - "map(to: ccx[0:k], ccy[0:k])" envia os centroides (pequenos,
         *   k elementos) para a GPU antes do laco.
         * - "map(tofrom: pgroup[0:size])" traz o vetor de grupos
         *   atualizado de volta para a CPU.
         * - "reduction(+:changed)" conta com seguranca quantos pontos
         *   mudaram de cluster nesta iteracao.
         * ================================================================ */
        long changed_l = 0;
        #pragma omp target teams distribute parallel for \
            map(to: px[0:size], py[0:size], ccx[0:k], ccy[0:k]) \
            map(tofrom: pgroup[0:size]) \
            reduction(+:changed_l)
        for (size_t j = 0; j < size; j++)
        {
            double minD = DBL_MAX;
            int best = -1;
            for (int i = 0; i < k; i++)
            {
                double dx = ccx[i] - px[j];
                double dy = ccy[i] - py[j];
                double dist = dx * dx + dy * dy;
                if (dist < minD)
                {
                    minD = dist;
                    best = i;
                }
            }
            if (best != pgroup[j])
            {
                pgroup[j] = best;
                changed_l += 1;
            }
        }
        changed = (size_t)changed_l;

        iter++;

        /* ---- LOG DE PROGRESSO (acompanhamento da execucao) ---- */
        printf("[OPENMP GPU] Iteracao %3d | pontos que mudaram de cluster: %zu\n",
               iter, changed);
    } while (changed > minAcceptedError && iter < MAX_ITER);

    /* ---- LOG: estado final dos centroides ---- */
    printf("[OPENMP GPU] Convergencia apos %d iteracoes. Centroides finais:\n",
           iter);
    for (int i = 0; i < k; i++)
    {
        printf("[OPENMP GPU]   Cluster %d -> x=%.4f, y=%.4f, pontos=%ld\n",
               i, ccx[i], ccy[i], ccount[i]);
    }

    /* Copia o resultado final de volta para as estruturas originais
       (observation[] e cluster[]), mantendo a interface do restante
       do programa (printEPS, etc) igual a do codigo original. */
    for (size_t j = 0; j < size; j++)
    {
        observations[j].group = pgroup[j];
    }
    for (int i = 0; i < k; i++)
    {
        clusters[i].x = ccx[i];
        clusters[i].y = ccy[i];
        clusters[i].count = (size_t)ccount[i];
    }

    free(px);
    free(py);
    free(pgroup);
    free(ccx);
    free(ccy);
    free(ccount);

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

    FILE* out = fopen("output_omp_gpu.eps", "w");
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
    printf("[OPENMP GPU] Arquivo 'output_omp_gpu.eps' salvo com sucesso (%zu pontos, %d clusters).\n",
           len, k);

    free(colors);
}

/*!
 * Funcao de teste: gera N pontos em um circulo de raio 20.0
 * centrado em (0,0) e agrupa em k clusters usando a versao
 * com offload para GPU do K-Means.
 */
static void test()
{
    size_t size = 15000000L; /* mesmo tamanho das demais versoes, para
                                 comparacao direta de desempenho */

    /* Informa se ha algum dispositivo de offload disponivel (GPU) */
    int num_devices = omp_get_num_devices();
    printf("[OPENMP GPU] Dispositivos de offload disponiveis: %d\n",
           num_devices);
    if (num_devices == 0)
    {
        printf(
            "[OPENMP GPU] Aviso: nenhum dispositivo de offload detectado. "
            "O codigo executara nas regioes target usando o fallback de "
            "host (mais lento que uma GPU real).\n");
    }

    printf("[OPENMP GPU] Gerando %zu pontos aleatorios...\n", size);

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

    printf("[OPENMP GPU] Dados gerados. Iniciando K-Means com k=%d...\n", k);

    /* ---------------- MEDICAO DE TEMPO ---------------- */
    double start = omp_get_wtime();

    cluster* clusters = kMeans(observations, size, k);

    double end = omp_get_wtime();
    double elapsed = end - start;
    printf("[OPENMP GPU] Tempo de execucao do K-Means: %.6f segundos\n",
           elapsed);
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
