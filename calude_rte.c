// ml_onnx_framework_dpdk.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <stdatomic.h>
#include <sys/mman.h>

#include <rte_common.h>
#include <rte_memory.h>
#include <rte_memzone.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>

#include "ml_onnx_async.h"
#include "onnxruntime_c_api.h"

#define RING_SIZE 2048
#define MAX_BATCH_SIZE 32
#define DEFAULT_TIMEOUT_MS 100
#define REQUEST_POOL_SIZE 4096
#define CACHE_SIZE 256

/* Global ONNX API pointer */
static const OrtApi* g_ort = NULL;

/* Request object pool (using DPDK mempool) */
static struct rte_mempool *request_pool = NULL;

/* ML Core Worker structure */
typedef struct {
    unsigned lcore_id;
    bool running;
    struct rte_ring* request_ring;
    struct rte_ring** response_rings;  // Array of response rings per PP core
    ml_framework_t* framework;
    struct {
        uint64_t processed;
        uint64_t timeouts;
        uint64_t errors;
        double total_latency_ms;
    } stats;
} ml_core_worker_t;

/* Framework implementation using DPDK */
struct ml_framework {
    OrtEnv* env;
    OrtThreadingOptions* threading_options;
    
    /* Models */
    model_registry_t* registry;
    
    /* DPDK Memory pools */
    struct rte_mempool* input_pool;
    struct rte_mempool* output_pool;
    
    /* DPDK Rings for communication */
    struct rte_ring* request_ring;      // Shared request ring
    struct rte_ring** response_rings;   // Per PP-core response rings
    
    /* Workers */
    ml_core_worker_t* ml_cores;
    size_t num_ml_cores;
    
    /* Configuration */
    ml_framework_config_t config;
    
    /* State */
    atomic_bool running;
    atomic_uint64_t request_counter;
    
    /* Statistics */
    struct {
        atomic_uint64_t total_requests;
        atomic_uint64_t completed_requests;
        atomic_uint64_t timeout_requests;
        atomic_uint64_t error_requests;
    } stats;
};

/* Utility macros */
#define ORT_CHECK(expr) \
    do { \
        OrtStatus* _status = (expr); \
        if (_status != NULL) { \
            const char* msg = g_ort->GetErrorMessage(_status); \
            fprintf(stderr, "ORT Error: %s\n", msg); \
            g_ort->ReleaseStatus(_status); \
            return -1; \
        } \
    } while(0)

/* Memory allocation wrappers for DPDK */
static void* dpdk_alloc_input_buffer(ml_framework_t* framework) {
    void* buf;
    if (rte_mempool_get(framework->input_pool, &buf) < 0) {
        return NULL;
    }
    return buf;
}

static void* dpdk_alloc_output_buffer(ml_framework_t* framework) {
    void* buf;
    if (rte_mempool_get(framework->output_pool, &buf) < 0) {
        return NULL;
    }
    return buf;
}

static void dpdk_free_input_buffer(ml_framework_t* framework, void* buf) {
    rte_mempool_put(framework->input_pool, buf);
}

static void dpdk_free_output_buffer(ml_framework_t* framework, void* buf) {
    rte_mempool_put(framework->output_pool, buf);
}

/* Async request allocation from mempool */
static ml_async_request_t* alloc_request(void) {
    void* req;
    if (rte_mempool_get(request_pool, &req) < 0) {
        return NULL;
    }
    memset(req, 0, sizeof(ml_async_request_t));
    return (ml_async_request_t*)req;
}

static void free_request(ml_async_request_t* req) {
    if (req) {
        rte_mempool_put(request_pool, req);
    }
}

/* Process inference request */
static void process_inference_request(ml_core_worker_t* worker, ml_async_request_t* request) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    ml_model_t* model = request->model;
    if (!model || !model->is_loaded) {
        request->status = ML_STATUS_ERROR;
        goto done;
    }
    
    // Create input tensor
    int64_t input_shape[] = {1, model->inputs[0].total_elements};
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        worker->framework->env,
        request->input_buffer,
        request->input_size,
        input_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &request->input_tensor
    );
    
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        request->status = ML_STATUS_ERROR;
        goto done;
    }
    
    // Setup run options with timeout
    g_ort->CreateRunOptions(&request->run_options);
    if (request->timeout_ms > 0) {
        g_ort->RunOptionsSetRunTag(request->run_options, "timeout_ms");
    }
    
    // Run inference
    const char* input_names[] = {model->inputs[0].name};
    const char* output_names[] = {model->outputs[0].name};
    
    status = g_ort->Run(
        model->session,
        request->run_options,
        input_names,
        (const OrtValue* const*)&request->input_tensor,
        1,
        output_names,
        1,
        &request->output_tensor
    );
    
    if (status != NULL) {
        request->status = ML_STATUS_ERROR;
        g_ort->ReleaseStatus(status);
    } else {
        // Copy output data
        float* output_data;
        g_ort->GetTensorMutableData(request->output_tensor, (void**)&output_data);
        memcpy(request->output_buffer, output_data, request->output_size);
        request->status = ML_STATUS_SUCCESS;
    }
    
    // Cleanup
    if (request->input_tensor) g_ort->ReleaseValue(request->input_tensor);
    if (request->output_tensor) g_ort->ReleaseValue(request->output_tensor);
    if (request->run_options) g_ort->ReleaseRunOptions(request->run_options);
    
done:
    clock_gettime(CLOCK_MONOTONIC, &end);
    double latency_ms = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    worker->stats.total_latency_ms += latency_ms;
    worker->stats.processed++;
    
    // Send response to appropriate PP core using DPDK ring
    struct rte_ring* response_ring = worker->response_rings[request->pp_core_id];
    if (rte_ring_enqueue(response_ring, request) < 0) {
        RTE_LOG(ERR, USER1, "Response ring full for PP core %u\n", request->pp_core_id);
        // Clean up if we can't enqueue
        dpdk_free_input_buffer(worker->framework, request->input_buffer);
        dpdk_free_output_buffer(worker->framework, request->output_buffer);
        free_request(request);
    }
}

/* Batch processing with DPDK optimizations */
static void process_batch_inference(ml_core_worker_t* worker, ml_async_request_t* requests[], size_t batch_size) {
    if (batch_size == 0) return;
    
    ml_model_t* model = requests[0]->model;
    
    // Prepare batch input using DPDK memory
    size_t input_elements = model->inputs[0].total_elements;
    size_t batch_input_size = batch_size * input_elements * sizeof(float);
    float* batch_input = rte_malloc("batch_input", batch_input_size, RTE_CACHE_LINE_SIZE);
    if (!batch_input) {
        // Handle allocation failure
        for (size_t i = 0; i < batch_size; i++) {
            requests[i]->status = ML_STATUS_ERROR;
            struct rte_ring* response_ring = worker->response_rings[requests[i]->pp_core_id];
            rte_ring_enqueue(response_ring, requests[i]);
        }
        return;
    }
    
    // Copy inputs
    for (size_t i = 0; i < batch_size; i++) {
        memcpy(batch_input + i * input_elements, 
               requests[i]->input_buffer, 
               input_elements * sizeof(float));
    }
    
    // Create batch tensor
    int64_t input_shape[] = {batch_size, input_elements};
    OrtValue* input_tensor;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        worker->framework->env,
        batch_input,
        batch_input_size,
        input_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        rte_free(batch_input);
        // Handle error
        for (size_t i = 0; i < batch_size; i++) {
            requests[i]->status = ML_STATUS_ERROR;
            struct rte_ring* response_ring = worker->response_rings[requests[i]->pp_core_id];
            rte_ring_enqueue(response_ring, requests[i]);
        }
        return;
    }
    
    // Run batch inference
    OrtValue* output_tensor;
    const char* input_names[] = {model->inputs[0].name};
    const char* output_names[] = {model->outputs[0].name};
    
    status = g_ort->Run(
        model->session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );
    
    if (status == NULL) {
        // Extract outputs
        float* output_data;
        g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        
        size_t output_elements = model->outputs[0].total_elements;
        for (size_t i = 0; i < batch_size; i++) {
            memcpy(requests[i]->output_buffer,
                   output_data + i * output_elements,
                   output_elements * sizeof(float));
            requests[i]->status = ML_STATUS_SUCCESS;
            
            // Send response using DPDK ring
            struct rte_ring* response_ring = worker->response_rings[requests[i]->pp_core_id];
            rte_ring_enqueue(response_ring, requests[i]);
        }
        g_ort->ReleaseValue(output_tensor);
    } else {
        // Handle error
        for (size_t i = 0; i < batch_size; i++) {
            requests[i]->status = ML_STATUS_ERROR;
            struct rte_ring* response_ring = worker->response_rings[requests[i]->pp_core_id];
            rte_ring_enqueue(response_ring, requests[i]);
        }
        g_ort->ReleaseStatus(status);
    }
    
    // Cleanup
    g_ort->ReleaseValue(input_tensor);
    rte_free(batch_input);
}

/* ML Core worker thread using DPDK lcore */
static int ml_core_worker_func(void* arg) {
    ml_core_worker_t* worker = (ml_core_worker_t*)arg;
    
    RTE_LOG(INFO, USER1, "ML Core %u starting on lcore %u\n", 
            worker->lcore_id, rte_lcore_id());
    
    ml_async_request_t* batch[MAX_BATCH_SIZE];
    size_t batch_count = 0;
    void* dequeued[MAX_BATCH_SIZE];
    
    while (worker->running) {
        // Try to dequeue multiple requests at once for efficiency
        unsigned nb_deq = rte_ring_dequeue_burst(worker->request_ring, 
                                                 dequeued, 
                                                 MAX_BATCH_SIZE - batch_count, 
                                                 NULL);
        
        if (nb_deq > 0) {
            // Add to batch
            for (unsigned i = 0; i < nb_deq; i++) {
                batch[batch_count++] = (ml_async_request_t*)dequeued[i];
            }
            
            // Check if we should process the batch
            if (batch_count >= MAX_BATCH_SIZE || 
                (batch_count > 0 && batch[0]->timeout_ms < 10)) {
                // Process batch
                if (batch_count == 1) {
                    process_inference_request(worker, batch[0]);
                } else {
                    process_batch_inference(worker, batch, batch_count);
                }
                batch_count = 0;
            }
        } else if (batch_count > 0) {
            // No new requests, process pending batch
            if (batch_count == 1) {
                process_inference_request(worker, batch[0]);
            } else {
                process_batch_inference(worker, batch, batch_count);
            }
            batch_count = 0;
        } else {
            // No work, yield CPU
            rte_pause();
        }
    }
    
    RTE_LOG(INFO, USER1, "ML Core %u stopping\n", worker->lcore_id);
    return 0;
}

/* Framework API Implementation */
ml_framework_t* ml_framework_init(const ml_framework_config_t* config) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return NULL;
    }
    
    ml_framework_t* framework = rte_zmalloc("ml_framework", 
                                            sizeof(ml_framework_t), 
                                            RTE_CACHE_LINE_SIZE);
    if (!framework) return NULL;
    
    framework->config = *config;
    atomic_init(&framework->running, true);
    atomic_init(&framework->request_counter, 0);
    
    // Initialize ONNX environment
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ml_framework", &framework->env));
    
    // Configure threading
    ORT_CHECK(g_ort->CreateThreadingOptions(&framework->threading_options));
    ORT_CHECK(g_ort->SetGlobalIntraOpNumThreads(framework->threading_options, 1));
    ORT_CHECK(g_ort->SetGlobalInterOpNumThreads(framework->threading_options, 1));
    
    // Create DPDK memory pools for input/output buffers
    char pool_name[RTE_MEMPOOL_NAMESIZE];
    
    snprintf(pool_name, sizeof(pool_name), "ml_input_pool");
    framework->input_pool = rte_mempool_create(
        pool_name,
        config->memory.num_input_buffers,
        config->memory.input_buffer_size,
        CACHE_SIZE,
        0,
        NULL, NULL,
        NULL, NULL,
        SOCKET_ID_ANY,
        MEMPOOL_F_SP_PUT | MEMPOOL_F_SC_GET
    );
    
    if (!framework->input_pool) {
        RTE_LOG(ERR, USER1, "Failed to create input memory pool\n");
        ml_framework_destroy(framework);
        return NULL;
    }
    
    snprintf(pool_name, sizeof(pool_name), "ml_output_pool");
    framework->output_pool = rte_mempool_create(
        pool_name,
        config->memory.num_output_buffers,
        config->memory.output_buffer_size,
        CACHE_SIZE,
        0,
        NULL, NULL,
        NULL, NULL,
        SOCKET_ID_ANY,
        MEMPOOL_F_SP_PUT | MEMPOOL_F_SC_GET
    );
    
    if (!framework->output_pool) {
        RTE_LOG(ERR, USER1, "Failed to create output memory pool\n");
        ml_framework_destroy(framework);
        return NULL;
    }
    
    // Create request object pool
    snprintf(pool_name, sizeof(pool_name), "ml_request_pool");
    request_pool = rte_mempool_create(
        pool_name,
        REQUEST_POOL_SIZE,
        sizeof(ml_async_request_t),
        CACHE_SIZE,
        0,
        NULL, NULL,
        NULL, NULL,
        SOCKET_ID_ANY,
        MEMPOOL_F_SP_PUT | MEMPOOL_F_SC_GET
    );
    
    if (!request_pool) {
        RTE_LOG(ERR, USER1, "Failed to create request pool\n");
        ml_framework_destroy(framework);
        return NULL;
    }
    
    // Create model registry
    framework->registry = ml_create_registry(32);
    
    // Create DPDK rings for communication
    char ring_name[RTE_RING_NAMESIZE];
    
    // Shared request ring (multi-producer, multi-consumer)
    snprintf(ring_name, sizeof(ring_name), "ml_request_ring");
    framework->request_ring = rte_ring_create(
        ring_name,
        RING_SIZE,
        SOCKET_ID_ANY,
        RING_F_MP_RTS_ENQ | RING_F_MC_RTS_DEQ
    );
    
    if (!framework->request_ring) {
        RTE_LOG(ERR, USER1, "Failed to create request ring\n");
        ml_framework_destroy(framework);
        return NULL;
    }
    
    // Per-PP-core response rings (single-producer, single-consumer)
    framework->response_rings = rte_zmalloc("response_rings", 
                                           config->num_pp_cores * sizeof(struct rte_ring*),
                                           RTE_CACHE_LINE_SIZE);
    
    for (size_t i = 0; i < config->num_pp_cores; i++) {
        snprintf(ring_name, sizeof(ring_name), "ml_response_ring_%zu", i);
        framework->response_rings[i] = rte_ring_create(
            ring_name,
            RING_SIZE,
            SOCKET_ID_ANY,
            RING_F_SP_ENQ | RING_F_SC_DEQ
        );
        
        if (!framework->response_rings[i]) {
            RTE_LOG(ERR, USER1, "Failed to create response ring %zu\n", i);
            ml_framework_destroy(framework);
            return NULL;
        }
    }
    
    // Initialize ML core workers
    framework->num_ml_cores = 2;
    framework->ml_cores = rte_zmalloc("ml_cores", 
                                     framework->num_ml_cores * sizeof(ml_core_worker_t),
                                     RTE_CACHE_LINE_SIZE);
    
    // Find available lcores for ML processing
    unsigned lcore_id;
    int ml_core_idx = 0;
    RTE_LCORE_FOREACH_SLAVE(lcore_id) {
        if (ml_core_idx >= framework->num_ml_cores) break;
        
        ml_core_worker_t* worker = &framework->ml_cores[ml_core_idx];
        worker->lcore_id = lcore_id;
        worker->running = true;
        worker->request_ring = framework->request_ring;
        worker->response_rings = framework->response_rings;
        worker->framework = framework;
        
        // Launch worker on lcore
        rte_eal_remote_launch(ml_core_worker_func, worker, lcore_id);
        
        RTE_LOG(INFO, USER1, "Launched ML core %d on lcore %u\n", 
                ml_core_idx, lcore_id);
        ml_core_idx++;
    }
    
    if (ml_core_idx < framework->num_ml_cores) {
        RTE_LOG(WARNING, USER1, "Only %d lcores available for ML processing\n", 
                ml_core_idx);
        framework->num_ml_cores = ml_core_idx;
    }
    
    return framework;
}

void ml_framework_destroy(ml_framework_t* framework) {
    if (!framework) return;
    
    atomic_store(&framework->running, false);
    
    // Stop ML core workers
    for (size_t i = 0; i < framework->num_ml_cores; i++) {
        framework->ml_cores[i].running = false;
    }
    
    // Wait for workers to finish
    unsigned lcore_id;
    RTE_LCORE_FOREACH_SLAVE(lcore_id) {
        rte_eal_wait_lcore(lcore_id);
    }
    
    // Cleanup rings
    if (framework->request_ring) rte_ring_free(framework->request_ring);
    if (framework->response_rings) {
        for (size_t i = 0; i < framework->config.num_pp_cores; i++) {
            if (framework->response_rings[i]) {
                rte_ring_free(framework->response_rings[i]);
            }
        }
        rte_free(framework->response_rings);
    }
    
    // Cleanup memory pools
    if (framework->input_pool) rte_mempool_free(framework->input_pool);
    if (framework->output_pool) rte_mempool_free(framework->output_pool);
    if (request_pool) rte_mempool_free(request_pool);
    
    // Cleanup ONNX
    if (framework->threading_options) g_ort->ReleaseThreadingOptions(framework->threading_options);
    if (framework->env) g_ort->ReleaseEnv(framework->env);
    
    // Cleanup registry
    ml_destroy_registry(framework->registry);
    
    rte_free(framework->ml_cores);
    rte_free(framework);
}

/* Model loading remains the same */
ml_model_t* ml_load_model(ml_framework_t* framework, const ml_model_config_t* config) {
    ml_model_t* model = rte_zmalloc("ml_model", sizeof(ml_model_t), RTE_CACHE_LINE_SIZE);
    if (!model) return NULL;
    
    // Copy basic info
    strncpy(model->name, config->model_name, MAX_MODEL_NAME_LEN - 1);
    strncpy(model->path, config->model_path, MAX_MODEL_NAME_LEN - 1);
    model->type = config->model_type;
    
    // Create session options
    ORT_CHECK(g_ort->CreateSessionOptions(&model->session_options));
    ORT_CHECK(g_ort->DisablePerSessionThreads(model->session_options));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(model->session_options, 1));
    ORT_CHECK(g_ort->SetInterOpNumThreads(model->session_options, 1));
    
    // Create session
    ORT_CHECK(g_ort->CreateSession(framework->env, config->model_path, 
                                   model->session_options, &model->session));
    
    // Copy tensor info
    model->inputs = rte_malloc("model_inputs", 
                              config->num_inputs * sizeof(tensor_info_t), 
                              RTE_CACHE_LINE_SIZE);
    model->outputs = rte_malloc("model_outputs", 
                               config->num_outputs * sizeof(tensor_info_t), 
                               RTE_CACHE_LINE_SIZE);
    
    memcpy(model->inputs, config->input_tensors, config->num_inputs * sizeof(tensor_info_t));
    memcpy(model->outputs, config->output_tensors, config->num_outputs * sizeof(tensor_info_t));
    model->num_inputs = config->num_inputs;
    model->num_outputs = config->num_outputs;
    
    // Initialize other fields
    model->max_batch_size = config->max_batch_size;
    model->supports_batching = (config->max_batch_size > 1);
    model->is_loaded = true;
    pthread_mutex_init(&model->lock, NULL);
    
    // Add to registry
    ml_registry_add_model(framework->registry, model);
    
    return model;
}

/* Async inference using DPDK */
uint64_t ml_inference_async(ml_framework_t* framework, ml_model_t* model,
                           const void* input, size_t input_size,
                           inference_callback_fn callback, void* user_data,
                           ml_priority_t priority, uint32_t timeout_ms) {
    uint64_t request_id = atomic_fetch_add(&framework->request_counter, 1);
    
    // Allocate request from pool
    ml_async_request_t* request = alloc_request();
    if (!request) {
        RTE_LOG(ERR, USER1, "Failed to allocate request\n");
        return 0;
    }
    
    // Allocate buffers from DPDK pools
    void* input_buffer = dpdk_alloc_input_buffer(framework);
    void* output_buffer = dpdk_alloc_output_buffer(framework);
    
    if (!input_buffer || !output_buffer) {
        if (input_buffer) dpdk_free_input_buffer(framework, input_buffer);
        if (output_buffer) dpdk_free_output_buffer(framework, output_buffer);
        free_request(request);
        return 0;
    }
    
    // Initialize request
    request->request_id = request_id;
    request->pp_core_id = rte_lcore_id() % framework->config.num_pp_cores;
    request->model = model;
    request->input_buffer = input_buffer;
    request->output_buffer = output_buffer;
    request->input_size = input_size;
    request->output_size = model->outputs[0].total_elements * sizeof(float);
    request->callback = callback;
    request->user_data = user_data;
    request->status = ML_STATUS_PENDING;
    request->timestamp = time(NULL);
    request->timeout_ms = timeout_ms > 0 ? timeout_ms : framework->config.default_timeout_ms;
    
    // Copy input data
    rte_memcpy(request->input_buffer, input, input_size);
    
    // Enqueue request to DPDK ring
    if (rte_ring_enqueue(framework->request_ring, request) < 0) {
        RTE_LOG(ERR, USER1, "Failed to enqueue request\n");
        dpdk_free_input_buffer(framework, input_buffer);
        dpdk_free_output_buffer(framework, output_buffer);
        free_request(request);
        return 0;
    }
    
    atomic_fetch_add(&framework->stats.total_requests, 1);
    return request_id;
}

/* Memory allocation wrappers */
void* ml_alloc_input_buffer(ml_framework_t* framework, size_t size) {
    if (size > framework->config.memory.input_buffer_size) {
        return NULL;
    }
    return dpdk_alloc_input_buffer(framework);
}

void* ml_alloc_output_buffer(ml_framework_t* framework, size_t size) {
    if (size > framework->config.memory.output_buffer_size) {
        return NULL;
    }
    return dpdk_alloc_output_buffer(framework);
}

void ml_free_input_buffer(ml_framework_t* framework, void* buffer) {
    dpdk_free_input_buffer(framework, buffer);
}

void ml_free_output_buffer(ml_framework_t* framework, void* buffer) {
    dpdk_free_output_buffer(framework, buffer);
}

/* PP Core registration */
int ml_register_pp_core(ml_framework_t* framework, uint32_t pp_core_id) {
    if (pp_core_id >= framework->config.num_pp_cores) {
        return -1;
    }
    // PP core is automatically registered when framework is initialized
    return 0;
}

/* Poll for responses using DPDK ring */
ml_async_request_t* ml_poll_response(ml_framework_t* framework, uint32_t pp_core_id) {
    if (pp_core_id >= framework->config.num_pp_cores) {
        return NULL;
    }
    
    void* obj;
    if (rte_ring_dequeue(framework->response_rings[pp_core_id], &obj) < 0) {
        return NULL;
    }
    
    ml_async_request_t* request = (ml_async_request_t*)obj;
    
    // Execute callback if provided
    if (request->callback) {
        request->callback(request->user_data, request->output_buffer, 
                        request->output_size, request->status);
    }
    
    return request;
}

/* Batch polling for responses */
int ml_poll_responses_bulk(ml_framework_t* framework, uint32_t pp_core_id,
                          ml_async_request_t** requests, unsigned n) {
    if (pp_core_id >= framework->config.num_pp_cores) {
        return 0;
    }
    
    void* objs[n];
    unsigned nb_deq = rte_ring_dequeue_burst(framework->response_rings[pp_core_id], 
                                             objs, n, NULL);
    
    for (unsigned i = 0; i < nb_deq; i++) {
        requests[i] = (ml_async_request_t*)objs[i];
        
        // Execute callbacks
        if (requests[i]->callback) {
            requests[i]->callback(requests[i]->user_data, 
                                requests[i]->output_buffer,
                                requests[i]->output_size, 
                                requests[i]->status);
        }
    }
    
    return nb_deq;
}

/* Model registry implementation using DPDK memory */
model_registry_t* ml_create_registry(size_t initial_capacity) {
    model_registry_t* registry = rte_zmalloc("model_registry", 
                                            sizeof(model_registry_t), 
                                            RTE_CACHE_LINE_SIZE);
    if (!registry) return NULL;
    
    registry->capacity = initial_capacity;
    registry->models = rte_zmalloc("registry_models", 
                                  initial_capacity * sizeof(ml_model_t*),
                                  RTE_CACHE_LINE_SIZE);
    registry->count = 0;
    
    return registry;
}

void ml_destroy_registry(model_registry_t* registry) {
    if (!registry) return;
    if (registry->models) rte_free(registry->models);
    rte_free(registry);
}

int ml_registry_add_model(model_registry_t* registry, ml_model_t* model) {
    if (registry->count >= registry->capacity) {
        // Resize using DPDK realloc
        size_t new_capacity = registry->capacity * 2;
        ml_model_t** new_models = rte_realloc(registry->models, 
                                             new_capacity * sizeof(ml_model_t*),
                                             RTE_CACHE_LINE_SIZE);
        if (!new_models) return -1;
        registry->models = new_models;
        registry->capacity = new_capacity;
    }
    
    registry->models[registry->count++] = model;
    return 0;
}

ml_model_t* ml_registry_get_model(model_registry_t* registry, const char* name) {
    for (size_t i = 0; i < registry->count; i++) {
        if (strcmp(registry->models[i]->name, name) == 0) {
            return registry->models[i];
        }
    }
    return NULL;
}

ml_model_t** ml_registry_get_models_by_type(model_registry_t* registry, 
                                           ml_model_type_t type, size_t* count) {
    // Count models of this type
    size_t type_count = 0;
    for (size_t i = 0; i < registry->count; i++) {
        if (registry->models[i]->type == type) {
            type_count++;
        }
    }
    
    if (type_count == 0) {
        *count = 0;
        return NULL;
    }
    
    // Allocate array for results
    ml_model_t** results = rte_malloc("model_type_results", 
                                     type_count * sizeof(ml_model_t*),
                                     RTE_CACHE_LINE_SIZE);
    if (!results) {
        *count = 0;
        return NULL;
    }
    
    // Populate results
    size_t idx = 0;
    for (size_t i = 0; i < registry->count; i++) {
        if (registry->models[i]->type == type) {
            results[idx++] = registry->models[i];
        }
    }
    
    *count = type_count;
    return results;
}

/* Statistics using DPDK logging */
void ml_get_stats(ml_framework_t* framework, char* buffer, size_t size) {
    snprintf(buffer, size,
            "Framework Statistics:\n"
            "Total Requests: %lu\n"
            "Completed: %lu\n"
            "Timeouts: %lu\n"
            "Errors: %lu\n"
            "Request Ring Count: %u\n"
            "Request Ring Free: %u\n",
            atomic_load(&framework->stats.total_requests),
            atomic_load(&framework->stats.completed_requests),
            atomic_load(&framework->stats.timeout_requests),
            atomic_load(&framework->stats.error_requests),
            rte_ring_count(framework->request_ring),
            rte_ring_free_count(framework->request_ring));
    
    // Add per-core statistics
    char core_stats[256];
    for (size_t i = 0; i < framework->num_ml_cores; i++) {
        ml_core_worker_t* worker = &framework->ml_cores[i];
        snprintf(core_stats, sizeof(core_stats),
                "ML Core %zu (lcore %u):\n"
                "  Processed: %lu\n"
                "  Errors: %lu\n"
                "  Avg Latency: %.3f ms\n",
                i, worker->lcore_id,
                worker->stats.processed,
                worker->stats.errors,
                worker->stats.processed > 0 ? 
                    worker->stats.total_latency_ms / worker->stats.processed : 0.0);
        strncat(buffer, core_stats, size - strlen(buffer) - 1);
    }
}

void ml_get_model_stats(ml_model_t* model, char* buffer, size_t size) {
    snprintf(buffer, size,
            "Model Statistics for '%s':\n"
            "Total Inferences: %lu\n"
            "Total Batches: %lu\n"
            "Average Latency: %.3f ms\n"
            "Last Used: %lu\n",
            model->name,
            model->stats.total_inferences,
            model->stats.total_batches,
            model->stats.avg_latency_ms,
            model->stats.last_used_timestamp);
}

/* Batch inference implementation */
uint64_t ml_inference_batch_async(ml_framework_t* framework, ml_model_t* model,
                                 const void** inputs, size_t* input_sizes,
                                 size_t batch_size, batch_callback_fn callback,
                                 void* user_data) {
    if (batch_size == 0 || batch_size > model->max_batch_size) {
        return 0;
    }
    
    // For now, submit individual requests that will be batched by ML cores
    uint64_t batch_id = atomic_fetch_add(&framework->request_counter, 1);
    
    for (size_t i = 0; i < batch_size; i++) {
        ml_inference_async(framework, model, inputs[i], input_sizes[i],
                          NULL, user_data, ML_PRIORITY_NORMAL, 
                          DEFAULT_TIMEOUT_MS);
    }
    
    // TODO: Implement proper batch tracking and callback
    return batch_id;
}

/* Helper functions */
void ml_print_model_info(ml_model_t* model) {
    printf("Model: %s\n", model->name);
    printf("Type: %s\n", ml_model_type_to_string(model->type));
    printf("Path: %s\n", model->path);
    printf("Max Batch Size: %zu\n", model->max_batch_size);
    printf("Inputs: %zu\n", model->num_inputs);
    for (size_t i = 0; i < model->num_inputs; i++) {
        printf("  - %s: %zu elements, shape [", 
               model->inputs[i].name, model->inputs[i].total_elements);
        for (size_t j = 0; j < model->inputs[i].num_dimensions; j++) {
            printf("%ld%s", model->inputs[i].shape[j], 
                   j < model->inputs[i].num_dimensions - 1 ? ", " : "");
        }
        printf("]\n");
    }
    printf("Outputs: %zu\n", model->num_outputs);
    for (size_t i = 0; i < model->num_outputs; i++) {
        printf("  - %s: %zu elements, shape [", 
               model->outputs[i].name, model->outputs[i].total_elements);
        for (size_t j = 0; j < model->outputs[i].num_dimensions; j++) {
            printf("%ld%s", model->outputs[i].shape[j], 
                   j < model->outputs[i].num_dimensions - 1 ? ", " : "");
        }
        printf("]\n");
    }
}

const char* ml_model_type_to_string(ml_model_type_t type) {
    switch (type) {
        case MODEL_TYPE_RANDOM_FOREST: return "Random Forest";
        case MODEL_TYPE_KNN: return "K-Nearest Neighbors";
        case MODEL_TYPE_NEURAL_NETWORK: return "Neural Network";
        case MODEL_TYPE_SVM: return "Support Vector Machine";
        case MODEL_TYPE_XGBOOST: return "XGBoost";
        case MODEL_TYPE_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

/* Model unloading */
void ml_unload_model(ml_framework_t* framework, ml_model_t* model) {
    if (!model) return;
    
    pthread_mutex_lock(&model->lock);
    model->is_loaded = false;
    
    if (model->session) g_ort->ReleaseSession(model->session);
    if (model->session_options) g_ort->ReleaseSessionOptions(model->session_options);
    
    if (model->inputs) rte_free(model->inputs);
    if (model->outputs) rte_free(model->outputs);
    
    pthread_mutex_unlock(&model->lock);
    pthread_mutex_destroy(&model->lock);
    
    rte_free(model);
}

ml_model_t* ml_get_model(ml_framework_t* framework, const char* model_name) {
    return ml_registry_get_model(framework->registry, model_name);
}

/* Request management */
int ml_cancel_request(ml_framework_t* framework, uint64_t request_id) {
    // TODO: Implement request cancellation
    // This would require tracking in-flight requests
    return -1;
}

ml_status_t ml_get_request_status(ml_framework_t* framework, uint64_t request_id) {
    // TODO: Implement request status tracking
    return ML_STATUS_PENDING;
}

/* Preprocessing and postprocessing stubs */
int ml_preprocess_input(ml_model_t* model, const void* raw_input, 
                       void* processed_input, size_t size) {
    // Default: just copy
    rte_memcpy(processed_input, raw_input, size);
    return 0;
}

int ml_postprocess_output(ml_model_t* model, const void* raw_output,
                         void* processed_output, size_t size) {
    // Default: just copy
    rte_memcpy(processed_output, raw_output, size);
    return 0;
}
