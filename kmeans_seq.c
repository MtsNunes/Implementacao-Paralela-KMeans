/* Tempos de execucao:
 * Versao Sequencial: Tempo: 11.039116 segundos
 * OpenMP CPU: 1 thread: XXX segundos | 2 threads: XXX segundos | 4 threads: XXX segundos | 8 threads: XXX segundos | 16 threads: 1.906000 segundos | 32 threads: XXX segundos
 * OpenMP GPU: Tempo: 1.975058 segundos
 * CUDA GPU: Tempo: 1.308400 segundos
 * Speedup: ...
 */

/**
 * @file kmeans_seq.c
 * @brief K-Means Clustering - Versao SEQUENCIAL com medicao de tempo
 * @details
 * Versao sequencial do algoritmo K-Means, derivada do algoritmo original.
 * Foi adicionada apenas a medicao de tempo de execucao (clock_gettime)
 * e o tamanho da base de dados foi ajustado para que a execucao dure
 * aproximadamente 10 segundos, servindo como baseline para comparacao
 * com as versoes paralelas (OpenMP CPU, OpenMP GPU e CUDA).
 *
 * A logica original do algoritmo K-Means (calculo de centroides,
 * atribuicao de pontos aos clusters, calculo de distancias) foi
 * preservada na integra.
 *
 * @author Original: Lakhan Nad (https://github.com/Lakhan-Nad)
 * @author Adaptacao para Computacao Paralela
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */

/* ------------------------------------------------------------------ */
/* Medicao de tempo PORTATIL (Windows/MinGW e Linux/Mac)               */
/* ------------------------------------------------------------------ */
/* No Linux/Mac usamos clock_gettime(CLOCK_MONOTONIC, ...), que e o
 * metodo recomendado por nao ser afetado por ajustes do relogio do
 * sistema (NTP) e ter alta resolucao.
 *
 * No Windows (MinGW), a macro CLOCK_MONOTONIC pode nao estar definida
 * (depende da versao do MinGW/runtime), por isso usamos a API nativa
 * QueryPerformanceCounter, que e o equivalente recomendado pela
 * Microsoft para medir intervalos de tempo de alta resolucao.
 *
 * A funcao get_time_seconds() abaixo abstrai essa diferenca: basta
 * chama-la antes e depois do trecho de codigo e subtrair os valores
 * retornados (em segundos, double) para obter o tempo decorrido. */
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
    #include <time.h>
    static double get_time_seconds(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
    }
#endif

/* ------------------------------------------------------------------ */
/* Estruturas de dados (iguais ao algoritmo original)                  */
/* ------------------------------------------------------------------ */

/*! @struct observation
 *  Armazena um ponto 2D e o grupo (cluster) ao qual pertence
 */
typedef struct observation
{
    double x;  /**< abscissa do ponto 2D */
    double y;  /**< ordenada do ponto 2D */
    int group; /**< indice do cluster ao qual o ponto pertence */
} observation;

/*! @struct cluster
 *  Armazena as coordenadas do centroide e a quantidade de
 *  observacoes pertencentes a esse cluster
 */
typedef struct cluster
{
    double x;     /**< abscissa do centroide */
    double y;     /**< ordenada do centroide */
    size_t count; /**< quantidade de observacoes no cluster */
} cluster;

/* Numero fixo de iteracoes do algoritmo, para tornar o tempo de
 * execucao comparavel entre as versoes sequencial, OpenMP e CUDA.
 * (No codigo original o criterio de parada era baseado em
 * "99.99% dos pontos sem mudar de cluster", o que pode gerar
 * numeros de iteracoes diferentes a cada execucao. Usamos aqui um
 * numero fixo de iteracoes para tornar os benchmarks comparaveis,
 * mantendo a mesma logica interna do algoritmo.) */
#define MAX_ITER 50

/*!
 * Retorna o indice do centroide mais proximo da observacao dada
 *
 * @param o  observacao
 * @param clusters  vetor de clusters (centroides)
 * @param k  tamanho do vetor de clusters
 *
 * @returns indice do centroide mais proximo
 */
int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calcula a distancia ao quadrado (evita sqrt, mais rapido) */
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
 * Calcula o centroide de um conjunto de observacoes
 *
 * @param observations  vetor de observacoes
 * @param size  tamanho do vetor de observacoes
 * @param centroid  ponteiro para o cluster que vai receber o resultado
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
 *    --Algoritmo K-Means (versao sequencial)--
 * 1. Atribui cada observacao a um dos k grupos de forma aleatoria
 * 2. Calcula o centroide de cada cluster
 * 3. Encontra o centroide mais proximo de cada observacao
 * 4. Atribui a observacao ao centroide mais proximo
 * 5. Repete os passos 2,3,4 por MAX_ITER iteracoes (ou ate convergir)
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

        /* PASSO 1: atribuicao inicial aleatoria */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }

        size_t changed = 0;
        size_t minAcceptedError = size / 10000;
        int t = 0;
        int iter = 0;

        do
        {
            /* Inicializa os clusters */
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }

            /* PASSO 2: soma das coordenadas por cluster */
            for (size_t j = 0; j < size; j++)
            {
                t = observations[j].group;
                clusters[t].x += observations[j].x;
                clusters[t].y += observations[j].y;
                clusters[t].count++;
            }
            for (int i = 0; i < k; i++)
            {
                if (clusters[i].count > 0)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }

            /* PASSOS 3 e 4: reatribuicao das observacoes */
            changed = 0;
            for (size_t j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }

            iter++;

            /* ---- LOG DE PROGRESSO (acompanhamento da execucao) ---- */
            printf("[SEQUENCIAL] Iteracao %3d | pontos que mudaram de cluster: %zu\n",
                   iter, changed);
        } while (changed > minAcceptedError &&
                 iter < MAX_ITER); /* limite de iteracoes adicionado para
                                      tornar o tempo de execucao comparavel
                                      entre as versoes */

        /* ---- LOG: estado final dos centroides ---- */
        printf("[SEQUENCIAL] Convergencia apos %d iteracoes. Centroides finais:\n",
               iter);
        for (int i = 0; i < k; i++)
        {
            printf("[SEQUENCIAL]   Cluster %d -> x=%.4f, y=%.4f, pontos=%zu\n",
                   i, clusters[i].x, clusters[i].y, clusters[i].count);
        }
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
 * Gera a saida em formato EPS (mantida do codigo original)
 *
 * @param pts  vetor de observacoes
 * @param len  tamanho do vetor de observacoes
 * @param cent  vetor de centroides
 * @param k  tamanho do vetor de centroides
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

    FILE* out = fopen("output_seq.eps", "w");
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
    printf("[SEQUENCIAL] Arquivo 'output_seq.eps' salvo com sucesso (%zu pontos, %d clusters).\n",
           len, k);

    free(colors);
}

/*!
 * Funcao de teste: gera N pontos em um circulo de raio 20.0
 * centrado em (0,0) e agrupa em k clusters. O tamanho N foi
 * aumentado em relacao ao codigo original para que a versao
 * sequencial demore aproximadamente 10 segundos, servindo de
 * baseline de comparacao com as versoes paralelas.
 */
static void test()
{
    size_t size = 15000000L; /* 15 milhoes de pontos - ajustar se necessario
                                 para atingir ~10s na maquina utilizada */

    printf("[SEQUENCIAL] Gerando %zu pontos aleatorios...\n", size);

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

    int k = 5; /* numero de clusters */

    printf("[SEQUENCIAL] Dados gerados. Iniciando K-Means com k=%d...\n", k);

    /* ---------------- MEDICAO DE TEMPO ---------------- */
    double start = get_time_seconds();

    cluster* clusters = kMeans(observations, size, k);

    double end = get_time_seconds();
    double elapsed = end - start;
    printf("[SEQUENCIAL] Tempo de execucao do K-Means: %.6f segundos\n",
           elapsed);
    /* ---------------------------------------------------- */

    //printEPS(observations, size, clusters, k);

    free(observations);
    free(clusters);
}

/*!
 * Funcao principal
 */
int main()
{
    srand((unsigned int)time(NULL));
    test();
    return 0;
}
