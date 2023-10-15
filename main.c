#include <stdio.h>
#include <stdlib.h>
#include <libpq-fe.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define EMBEDDING_SIZE 512
#define BATCH_SIZE 1000
#define TOP_N 10

typedef struct {
    double values[EMBEDDING_SIZE];
} Embedding;

typedef struct {
    int index;
    double similarity;
} SimilarityIndex;

Embedding* get_embeddings(int* total_size) {
    const char *conninfo = "dbname=<password> user=<username> host=<Your serverUP> password=<insert ur own>.";
    PGconn* conn = PQconnectdb(conninfo);
    if (PQstatus(conn) == CONNECTION_BAD) {
        fprintf(stderr, "Connection to database failed: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        exit(1);
    }


    Embedding* embeddings = NULL;
    *total_size = 0;
    int offset = 0;

    while (1) {
        char query[256];
        snprintf(query, sizeof(query), "SELECT embeddings FROM laws_paragraphs WHERE embeddings IS NOT NULL LIMIT %d OFFSET %d", BATCH_SIZE, offset);

        PGresult* res = PQexec(conn, query);
        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            printf("SELECT command failed: %s", PQerrorMessage(conn));
            PQclear(res);
            exit(1);
        }

        int nrows = PQntuples(res);
        if (nrows == 0) {
            PQclear(res);
            break;  // All rows have been fetched
        }

        embeddings = realloc(embeddings, (*total_size + nrows) * sizeof(Embedding));
        for (int i = 0; i < nrows; i++) {
            char* embedding_str = PQgetvalue(res, i, 0);
            char *token = strtok(embedding_str, ",");
            int j = 0;
            while(token && j < EMBEDDING_SIZE) {
                embeddings[*total_size + i].values[j] = atof(token);
                token = strtok(NULL, ",");
                j++;
            }
        }
        *total_size += nrows;
        offset += nrows;  // Update offset for next batch

        PQclear(res);
    }
    PQfinish(conn);
    return embeddings;
}

Embedding read_embedding_from_file(const char* filename) {
    Embedding embedding;
    FILE *file = fopen(filename, "r");

    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    // Scan past the opening bracket
    fscanf(file, "[");

    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        if (i != EMBEDDING_SIZE - 1) {
            // Read up to comma for all but the last value
            fscanf(file, "%lf,", &embedding.values[i]);
        } else {
            // Last value should be read up to the closing bracket
            fscanf(file, "%lf]", &embedding.values[i]);
        }
    }

    fclose(file);
    return embedding;
}

double cosine_similarity(const Embedding* A, const Embedding* B) {
    double dot_product = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;

    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        dot_product += A->values[i] * B->values[i];
        magnitudeA += A->values[i] * A->values[i];
        magnitudeB += B->values[i] * B->values[i];
    }

    magnitudeA = sqrt(magnitudeA);
    magnitudeB = sqrt(magnitudeB);

    return dot_product / (magnitudeA * magnitudeB);
}

SimilarityIndex* find_top_similar_embeddings(const Embedding* target_embedding, Embedding* all_embeddings, int total_size) {
    SimilarityIndex* top_indices = malloc(TOP_N * sizeof(SimilarityIndex));
    for (int i = 0; i < TOP_N; i++) {
        top_indices[i].similarity = -2;  // Initialize with a value lower than possible cosine similarity
    }

    for (int i = 0; i < total_size; i++) {
        double similarity = cosine_similarity(target_embedding, &all_embeddings[i]);
        if (similarity > top_indices[TOP_N - 1].similarity) {
            top_indices[TOP_N - 1].similarity = similarity;
            top_indices[TOP_N - 1].index = i;

            // Sort the top_indices array
            for (int j = TOP_N - 1; j > 0; j--) {
                if (top_indices[j].similarity > top_indices[j - 1].similarity) {
                    SimilarityIndex temp = top_indices[j];
                    top_indices[j] = top_indices[j - 1];
                    top_indices[j - 1] = temp;
                } else {
                    break;
                }
            }
        }
    }

    return top_indices;
}


int main() {
    int total_size;
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    Embedding* all_embeddings = get_embeddings(&total_size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for get_embeddings: %f seconds\n", cpu_time_used);

    start = clock();
    Embedding target_embedding = read_embedding_from_file("C:/Users/emils/CLionProjects/untitled10/embed.txt");
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for read_embedding_from_file: %f seconds\n", cpu_time_used);

    start = clock();
    SimilarityIndex* top_similar = find_top_similar_embeddings(&target_embedding, all_embeddings, total_size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for find_top_similar_embeddings: %f seconds\n", cpu_time_used);

    for (int i = 0; i < TOP_N; i++) {
        printf("Index: %d, Similarity: %f\n", top_similar[i].index, top_similar[i].similarity);
    }

    free(all_embeddings);
    free(top_similar);
}