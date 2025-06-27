#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>
#include <pthread.h>
/**
 * @brief Enum representing supported data modalities for machine learning tasks.
 * 
 * Specifies the type of data modality (e.g., image, text, audio) used in inference.
 * 
 * @var MODALITY_1
 * Denotes the first modality type (e.g., image processing).
 * @var MODALITY_2
 * Denotes the second modality type (e.g., text analysis).
 * @var MODALITY_3
 * Denotes the third modality type (e.g., audio recognition).
 */
#ifndef MLDEV_H
#define MLDEV_H

#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>
#include <pthread.h>

typedef enum{
    MODALITY_1=1,
    MODALITY_2,
    MODALITY_3,
}ModalityType;

typedef struct{
    OrtSession* session;
    OrtEnv* env;
    OrtSessionOptions* session_options;
    OrtMemoryInfo* memory_info;
    OrtAllocator* allocator;
    const OrtApi* api;
    char* input_name;
    char* output_name;
    bool is_loaded;
    bool is_running;
   
}MLDevice;

#define MAX_INPUT_BUFFERS 10
#define MAX_OUTPUT_BUFFERS 10

typedef struct {
    char* model_path;
    char* input_name;
    char* output_name;
    size_t input_count;
    size_t output_count;
    float* input_buffer_pool[MAX_INPUT_BUFFERS];
    float* output_buffer_pool[MAX_OUTPUT_BUFFERS];
    size_t input_buffer_size;
    size_t output_buffer_size;
    size_t input_pool_size;
    size_t output_pool_size;
    pthread_mutex_t pool_mutex;
} MLModel;

typedef struct{
    MLDevice* dev;
    float* input;
    int rows;
    int cols;
    float* output;
    int out_size;
    double* elasped_time;
    uint64_t* cpu_cycles;
    int row_id;
    FILE* output_file;
    ModalityType modality; //image, text, audio, tabular
    const char* custom_model_path;

    MLModel* model;
    size_t input_buffer_index;
    size_t output_buffer_index;
}ThreadArgs;

#endif // MLDEV_H




