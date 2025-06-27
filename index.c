/**
 * @brief Main function for performing parallel inference using ONNX models.
 *
 * This program reads input data from a CSV file, processes each row through one of multiple ONNX models
 * (modalities) using a thread pool, and writes inference results (predicted class, latency, CPU cycles)
 * to an output file. It supports multiple modalities with separate models and calculates aggregate metrics.
 *
 * @param argc Argument count (number of command-line arguments).
 * @param argv Argument vector:
 *             - argv[1]: Input CSV file path (default: "data.csv").
 *             - argv[2]: Output results file path (default: "testing_data.txt").
 *
 * @return int Program exit status:
 *             - 0: Successful execution.
 *             - 1: CSV read error or model loading failure.
 *             - -1: Output file open error.
 *
 * @note
 * - Uses global thread pool configuration for ONNX Runtime (inter-op=1, intra-op=3).
 * - Processes data in batches with NUM_THREADS parallel threads.
 * - Models are pre-loaded for NUM_MODALITIES (defined paths in `modality_model_paths`).
 * - Input CSV must have exactly 25 features (columns) and <= 40,000 rows.
 * - Output includes per-row details and averages of latency/cycles.
 *
 * @example
 *   ./program input.csv results.txt
 *
 * @exception
 * - Terminates with error if CSV file cannot be opened/parsed.
 * - Terminates with error if output file creation fails.
 * - Terminates with error if ONNX model loading fails.
 */

#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "ml_onnx_async.h"
#include <unistd.h>
#include "MLdev.h"

#define MAX_ROWS 40000
#define FEATURES 25
#define CLASSES 2
#define NUM_THREADS 2
#define NUM_MODALITIES 3
pthread_t threads[NUM_THREADS];
ThreadArgs thread_args[NUM_THREADS];
MLDevice devices[NUM_THREADS];
float inputs[MAX_ROWS][FEATURES];
float outputs[MAX_ROWS][CLASSES];
double elapsed_time[MAX_ROWS];
uint64_t cpu_cycles[MAX_ROWS];


OrtEnv* shared_env;
OrtThreadingOptions* thread_options;

const char* modality_model_paths[NUM_MODALITIES] = {
    "rf_10_pkts_model_sklearn.onnx",
    "rf_10_pkts_model_sklearn1.onnx",
    "rf_10_pkts_model_sklearn2.onnx"
};

const char* modality_model_name(ModalityType type){
    switch(type){
        case MODALITY_1: return "Modality 1";
        case MODALITY_2: return "Modality 2";
        case MODALITY_3: return "Modality 3";
    }
}

void *inference_thread_func(void *arg)
{
    ThreadArgs *args = (ThreadArgs *)arg;
    printf("Thread %d (modality %d )Started\n for row %d\n", args->row_id,args->modality,args->row_id+1);
    float *input = args->input;
    float *output = args->output;
    int result = ml_dev_infer(args->dev,
                 input,
                 args->rows,
                 args->cols,
                 output,
                 args->out_size,
                 args->elasped_time,
                 args->cpu_cycles);
    if(result!=0){
        fprintf(stderr, "Inference Failed\n");
    }
    int max_prob = -1;
    int predicted_class = -1;
    for (int i = 0; i < CLASSES; i++)
    {
        if (output[i] > max_prob)
        {
            max_prob = output[i];
            predicted_class = i;
        }
    }
    fprintf(args->output_file, "Row %d : Predicted Class %d | Time: %.3f ms | Cycles : %" PRIu64 "\n", args->row_id + 1, predicted_class, *(args->elasped_time), *(args->cpu_cycles));
    pthread_exit(NULL);
}

int read_csv(const char *filename, float data[MAX_ROWS][FEATURES], size_t *row_count)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("fopen");
        return -1;
    }
    char line[4096];
    *row_count = 0;
    while (fgets(line, sizeof(line), file))
    {
        line[strcspn(line, "\r\n")] = '\0';
        int col = 0;
        char *token = strtok(line, ",");
        while (token && col < FEATURES)
        {
            data[*row_count][col] = strtof(token, NULL);
            token = strtok(NULL, ",");
            col++;
        }
        if (col == FEATURES)
        {
            (*row_count)++;
        }
        if (*row_count >= MAX_ROWS)
        {
            break;
        }
    }
    fclose(file);
    return 0;
}

void* thread_infer(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    // printf("Thread %d (modality %d )Started\n for row %d\n", args->row_id,args->modality,args->row_id+1);
    ml_dev_infer(args->dev, args->input, 1, FEATURES, args->output, CLASSES, args->elasped_time, args->cpu_cycles);
    pthread_exit(NULL);
    return NULL;
}


int main(int argc, char *argv[])
{
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api -> CreateThreadingOptions(&thread_options);
    api->SetGlobalIntraOpNumThreads(thread_options,3);
    api->SetGlobalInterOpNumThreads(thread_options,1);
    api->CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_WARNING,"global_env", thread_options,&shared_env);

  
    // const char *model_path = (argc > 1) ? argv[1] : "rf_10_pkts_model_sklearn.onnx";
    const char *csv_path = (argc > 1) ? argv[1] : "data.csv";
    const char *output_path = (argc > 2) ? argv[2] : "testing_data.txt";
    size_t row_count;
    if (read_csv(csv_path, &inputs, &row_count) != 0)
    {
        fprintf(stderr, "Error reading CSV file\n");
        return 1;
    }
   
    printf("%d", row_count);
    FILE *output_file = fopen(output_path, "w");
    if (!output_file)
    {
        fprintf(stderr, "Error opening output file\n");
        return -1;
    }
    int current_row = 0;

    for(int i=0;i<NUM_THREADS;i++){
        ModalityType modality = (i%NUM_MODALITIES)+1;
        ml_dev_load(&devices[i],modality_model_paths[modality-1],shared_env);
    }
     for(int i=0;i<NUM_THREADS;i++){
        ml_dev_start(&devices[i]);
    }

    while(current_row<row_count){
        int active = 0;
        for(int i=0;i<NUM_THREADS && current_row<row_count;i++,current_row++){
            ModalityType modality = (i%NUM_MODALITIES)+1;
            thread_args[i] = (ThreadArgs){
                .dev = &devices[i],
                .input = inputs[current_row],
                .output = outputs[current_row],
                .elasped_time = &elapsed_time[current_row],
                .cpu_cycles = &cpu_cycles[current_row],
                .row_id = current_row,
                .modality = modality,
                .custom_model_path = modality_model_paths[modality-1]
            };
            pthread_create(&threads[i],NULL,thread_infer,&thread_args[i]);
            active++;
        }
        for(int i=0;i<active;i++){
            pthread_join(threads[i],NULL);
        }
    }

    double total_time = 0;
    uint64_t total_cycles = 0;

    for (size_t i = 0; i < row_count; i++) {
        int pred = outputs[i][0] > outputs[i][1] ? 0 : 1;
        ModalityType m = (i%NUM_MODALITIES)+1;
        total_time += elapsed_time[i];
        total_cycles += cpu_cycles[i];
        fprintf(output_file, "Row %zu: Predicted Class: %d | Time: %.3f ms | Cycles: %" PRIu64" | Modality :  %s | Model : %s\n",
                i + 1, pred, elapsed_time[i], cpu_cycles[i], modality_model_name(m),modality_model_paths[m-1]);
    }

    fprintf(output_file, "\nAverage Time: %.3f ms\nAverage Cycles: %" PRIu64 "\n",
            total_time / row_count, total_cycles / row_count);

    for(int i=0;i<NUM_THREADS;i++){
        printf("Unloading Begins.....\n");
        ml_dev_unload(&devices[i]);
    }
    api->ReleaseThreadingOptions(thread_options);
    api->ReleaseEnv(shared_env);

    // while (current_row < row_count)
    // {
    //     int batch_size = (row_count - current_row < NUM_THREADS) ? (row_count - current_row) : NUM_THREADS;
    //     for (int i = 0; i < batch_size; i++)
    //     {
    //         if (ml_dev_load(&devices[i], model_path) != 0)
    //         {
    //             fprintf(stderr, "Error loading model\n");
    //             return 1;
    //         }
    //         if (ml_dev_start(&devices[i]) != 0)
    //         {
    //             fprintf(stderr, "Error loading model\n");
    //             return 1;
    //         }
    //         thread_args[i].dev = &devices[i];
    //         thread_args[i].input = inputs[i];
    //         thread_args[i].rows = 1;
    //         thread_args[i].cols = FEATURES;
    //         thread_args[i].output = outputs[i];
    //         thread_args[i].out_size = CLASSES;
    //         thread_args[i].elasped_time = &elapsed_time[i];
    //         thread_args[i].cpu_cycles = &cpu_cycles[i];
    //         thread_args[i].row_id = i;
    //         thread_args[i].output_file = output_file;

    //         if (pthread_create(&threads[i], NULL, inference_thread_func, &thread_args[i]) != 0)
    //         {
    //             perror("pthread_create failed\n");
    //             return 1;
    //         }
    //     }
    //     for (int i = 0; i < batch_size; i++)
    //     {
    //         pthread_join(threads[i], NULL);
    //         ml_dev_unload(&devices[i]);
    //     }
    //     current_row += batch_size;
    // }
    fclose(output_file);

    return 0;

 
}
