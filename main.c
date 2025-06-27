#ifndef ML_ONNX_NATIVE_ASYNC_H
#define ML_ONNX_NATIVE_ASYNC_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include "onnxruntime_c_api.h"
// #include <rte_common.h>
// #include <rte_ring_core.h>
// #include <rte_ring_elem.h>


/* Configuration Constants */
#define MAX_MODELS 32
#define MAX_PP_CORES 32
#define DEFAULT_TIMEOUT_MS 100
#define RING_SIZE 2048
#define MAX_TENSOR_DIMS 8
#define MAX_MODEL_NAME_LEN 256

/* Forward declarations */
typedef struct ml_framework ml_framework_t;
typedef struct ml_model ml_model_t;
typedef struct ml_async_request ml_async_request_t;
typedef struct model_registry model_registry_t;

/* Callback function types */
typedef void (*inference_callback_fn)(void* user_data, float* output, size_t output_size, int status);
typedef void (*batch_callback_fn)(void* user_data, float** outputs, size_t* output_sizes, size_t batch_size, int status);

/* Request status codes */
typedef enum {
    ML_STATUS_SUCCESS = 0,
    ML_STATUS_PENDING,
    ML_STATUS_TIMEOUT,
    ML_STATUS_ERROR,
    ML_STATUS_CANCELLED
} ml_status_t;

/* Model types supported */
typedef enum {
    MODEL_TYPE_RANDOM_FOREST,
    MODEL_TYPE_KNN,
    MODEL_TYPE_NEURAL_NETWORK,
    MODEL_TYPE_SVM,
    MODEL_TYPE_XGBOOST,
    MODEL_TYPE_CUSTOM
} ml_model_type_t;

/* Preprocessing types */
typedef enum {
    PREPROCESS_NONE,
    PREPROCESS_STANDARDIZE,
    PREPROCESS_MINMAX,
    PREPROCESS_CUSTOM
} preprocess_type_t;

/* Tensor information */
typedef struct {
    char name[MAX_MODEL_NAME_LEN];
    ONNXTensorElementDataType dtype;
    int64_t shape[MAX_TENSOR_DIMS];
    size_t num_dimensions;
    size_t total_elements;
    bool is_dynamic;
} tensor_info_t;

/* Preprocessing configuration */
typedef struct {
    preprocess_type_t type;
    float* input_mean;
    float* input_std;
    float* input_min;
    float* input_max;
    void (*custom_preprocess)(void* input, void* output, size_t size);
} preprocessing_config_t;

/* Postprocessing configuration */
typedef struct {
    char** class_labels;
    int num_classes;
    float confidence_threshold;
    float output_scale;
    float output_offset;
    void (*custom_postprocess)(void* output, void* processed, size_t size);
} postprocessing_config_t;

/* Performance hints */
typedef struct {
    bool prefer_gpu;
    int num_threads;
    bool enable_profiling;
    size_t max_memory_mb;
} performance_hints_t;

/* Model metadata based on type */
typedef union {
    struct {
        int num_trees;
        int max_depth;
        int num_features;
    } rf_metadata;
    
    struct {
        int k_value;
        char distance_metric[64];
    } knn_metadata;
    
    struct {
        int num_layers;
        char activation[64];
        bool supports_dynamic_batch;
    } nn_metadata;
} model_metadata_t;

/* Request priority levels */
typedef enum {
    ML_PRIORITY_LOW = 0,
    ML_PRIORITY_NORMAL,
    ML_PRIORITY_HIGH
} ml_priority_t;

/* Memory pool configuration */
typedef struct {
    size_t input_buffer_size;
    size_t output_buffer_size;
    size_t num_input_buffers;
    size_t num_output_buffers;
    bool use_hugepages;
} ml_memory_config_t;

/* Framework configuration */
typedef struct {
    size_t num_pp_cores;          /* Number of packet processing cores */
    ml_memory_config_t memory;    /* Memory configuration */
    uint32_t default_timeout_ms;  /* Default inference timeout */
    bool enable_profiling;        /* Enable performance profiling */
    size_t max_concurrent_requests; /* Max async requests in flight */
} ml_framework_config_t;

typedef struct {
    /* Basic info */
    const char* model_path;
    const char* model_name;
    ml_model_type_t model_type;
    const char* description;
    
    /* Tensor specifications */
    tensor_info_t* input_tensors;
    size_t num_inputs;
    tensor_info_t* output_tensors;
    size_t num_outputs;
    
    /* Processing configs */
    preprocessing_config_t* preprocess_config;
    postprocessing_config_t* postprocess_config;
    
    /* Performance hints */
    performance_hints_t* perf_hints;
    size_t max_batch_size;
    
    /* Model-specific metadata */
    model_metadata_t* metadata;
} ml_model_config_t;

/* Enhanced model structure */
struct ml_model {
    /* Basic information */
    char name[MAX_MODEL_NAME_LEN];
    char path[MAX_MODEL_NAME_LEN];
    char description[MAX_MODEL_NAME_LEN];
    ml_model_type_t type;
    char version[32];
    
    /* ONNX Runtime objects */
    OrtSession* session;
    OrtSessionOptions* session_options;
    
    /* Input/Output specifications */
    tensor_info_t* inputs;
    size_t num_inputs;
    tensor_info_t* outputs;
    size_t num_outputs;
    
    /* Model-specific metadata */
    model_metadata_t metadata;
    
    /* Processing configurations */
    preprocessing_config_t preprocess;
    postprocessing_config_t postprocess;
    
    /* Performance configuration */
    performance_hints_t perf_hints;
    size_t max_batch_size;
    
    /* Runtime statistics */
    struct {
        uint64_t total_inferences;
        uint64_t total_batches;
        double avg_latency_ms;
        uint64_t last_used_timestamp;
    } stats;
    
    /* Model state */
    bool is_loaded;
    bool supports_batching;
    pthread_mutex_t lock;
    
    /* Model-specific functions */
    int (*validate_input)(struct ml_model* model, const void* input, size_t size);
    int (*prepare_batch)(struct ml_model* model, const void** inputs, size_t count, void** batch_output);
};


/* Async request structure */
typedef struct  ml_async_request {
    uint64_t request_id;
    uint32_t pp_core_id;
    ml_model_t* model;
    void* input_buffer;
    void* output_buffer;
    size_t input_size;
    size_t output_size;
    inference_callback_fn callback;
    void* user_data;
    ml_status_t status;
    uint64_t timestamp;
    /* ONNX specific */
    OrtValue* input_tensor;
    OrtValue* output_tensor;
    OrtRunOptions* run_options;
} ml_async_request;

/* Ring buffer for lock-free communication */
typedef struct {
    void** buffer;
    size_t size;
    size_t mask;
    volatile uint64_t head;
    volatile uint64_t tail;
    char pad0[64];  /* Cache line padding */
} ml_ring_buffer_t;

typedef struct{
    ml_async_request* buffer;
    size_t head;
    size_t tail;
    size_t size;
    size_t capacity;
} ring_buffer;

/* Per-core communication channel */
typedef struct {
    uint32_t pp_core_id;
    ml_ring_buffer_t* response_ring;  /* ML -> PP responses */
} ml_core_channel_t;



/* Model registry structure */
struct model_registry {
    ml_model_t** models;
    size_t capacity;
    size_t count;
    // pthread_rwlock_t lock;
    /* Fast lookup structures */
    void* name_to_model_map;  /* TODO: Implement as hash table */
    ml_model_t** type_indices[MODEL_TYPE_CUSTOM + 1];  /* Arrays per type */
    size_t type_counts[MODEL_TYPE_CUSTOM + 1];
};


// typedef struct{
//     OrtSession* session;
//     OrtEnv* env;
//     OrtSessionOptions* session_options;
//     bool is_loaded;
//     bool is_running;
// }MLDevice;
/*
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
}MLDevice;*/


/*
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
}ThreadArgs;
 Main API Functions */



/* Framework lifecycle */
ml_framework_t* ml_framework_init(const ml_framework_config_t* config);
void ml_framework_destroy(ml_framework_t* framework);

/* Model management */
ml_model_t* ml_load_model(ml_framework_t* framework, const ml_model_config_t* config);
void ml_unload_model(ml_framework_t* framework, ml_model_t* model);
ml_model_t* ml_get_model(ml_framework_t* framework, const char* model_name);

/* Model registry functions */
model_registry_t* ml_create_registry(size_t initial_capacity);
void ml_destroy_registry(model_registry_t* registry);
int ml_registry_add_model(model_registry_t* registry, ml_model_t* model);
ml_model_t* ml_registry_get_model(model_registry_t* registry, const char* name);
ml_model_t** ml_registry_get_models_by_type(model_registry_t* registry, ml_model_type_t type, size_t* count);

/* Async inference using native ONNX RunAsync */
uint64_t ml_inference_async(
    ml_framework_t* framework,
    ml_model_t* model,
    const void* input,
    size_t input_size,
    inference_callback_fn callback,
    void* user_data,
    ml_priority_t priority,
    uint32_t timeout_ms
);

/* Batch inference */
uint64_t ml_inference_batch_async(
    ml_framework_t* framework,
    ml_model_t* model,
    const void** inputs,
    size_t* input_sizes,
    size_t batch_size,
    batch_callback_fn callback,
    void* user_data
);



/* Request management */
int ml_cancel_request(ml_framework_t* framework, uint64_t request_id);
ml_status_t ml_get_request_status(ml_framework_t* framework, uint64_t request_id);

/* Memory management */
void* ml_alloc_input_buffer(ml_framework_t* framework, size_t size);
void* ml_alloc_output_buffer(ml_framework_t* framework, size_t size);
void ml_free_input_buffer(ml_framework_t* framework, void* buffer);
void ml_free_output_buffer(ml_framework_t* framework, void* buffer);

/* PP Core registration */
int ml_register_pp_core(ml_framework_t* framework, uint32_t pp_core_id);
ml_async_request_t* ml_poll_response(ml_framework_t* framework, uint32_t pp_core_id);

/* Preprocessing/Postprocessing */
int ml_preprocess_input(ml_model_t* model, const void* raw_input, void* processed_input, size_t size);
int ml_postprocess_output(ml_model_t* model, const void* raw_output, void* processed_output, size_t size);

/* Model introspection */
void ml_print_model_info(ml_model_t* model);
const char* ml_model_type_to_string(ml_model_type_t type);

/* Profiling */
void ml_get_stats(ml_framework_t* framework, char* buffer, size_t size);
void ml_get_model_stats(ml_model_t* model, char* buffer, size_t size);

#endif 
/* ML_ONNX_FRAMEWORK_V2_H */


