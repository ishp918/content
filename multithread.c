/**
 * @brief Loads an ONNX model into the device and initializes necessary resources.
 * 
 * This function sets up the device with the specified model, configures session options,
 * and prepares memory and allocator. It is required to call this before any inference operations.
 * 
 * @param dev Pointer to the MLDevice structure.
 * @param model_path Path to the ONNX model file.
 * @param shared_env Shared ONNX Runtime environment.
 * @return int Returns 0 on success, -1 if device is invalid or model loading fails.
 * @warning Errors during ONNX Runtime operations will abort the program.
 * 
 * Example:
 * @code
 * MLDevice dev;
 * OrtEnv *env;
 * ml_dev_load(&dev, "model.onnx", env);
 * @endcode
 * 
 * Notes:
 * - Configures session with intra-op and inter-op threads set to 1.
 * - Retrieves input/output names from the session.
 * - Uses the provided shared environment for session creation.
 */

/**
 * @brief Starts the device for inference operations.
 * 
 * This function enables the device to execute inference. Must be called after successful load.
 * 
 * @param dev Pointer to the MLDevice structure.
 * @return int Returns 0 on success, -1 if device is not loaded.
 * 
 * Example:
 * @code
 * ml_dev_start(dev);
 * @endcode
 */

/**
 * @brief Unloads the model and releases all associated resources.
 * 
 * This function cleans up memory, session, and options. Must be called after inference completion.
 * 
 * @param dev Pointer to the MLDevice structure.
 * @return int Returns 0 on success, -1 if device is not loaded.
 * 
 * Example:
 * @code
 * ml_dev_unload(dev);
 * @endcode
 * 
 * Notes:
 * - Frees input/output names using the device's allocator.
 * - Does not release the shared environment (managed externally).
 */

/**
 * @brief Executes inference on the loaded model with provided input data.
 * 
 * Runs the model, measures execution time and CPU cycles (if supported), and copies results to output buffer.
 * 
 * @param dev Pointer to the initialized device.
 * @param input_data Pointer to input tensor data.
 * @param input_count Number of input samples.
 * @param feature_cnt Features per sample.
 * @param output_data Buffer for output tensor data.
 * @param output_cnt Expected output elements.
 * @param elapsed_time Pointer to store inference time in milliseconds.
 * @param cpu_cycles Pointer to store CPU cycles used.
 * @return int Returns 0 on success, -1 if device is not running.
 * 
 * Example:
 * @code
 * float input[10], output[5];
 * double elapsed;
 * uint64_t cycles;
 * ml_dev_infer(dev, input, 1, 10, output, 5, &elapsed, &cycles);
 * @endcode
 * 
 * Notes:
 * - Timing uses CLOCK_MONOTONIC and x86-specific rdtsc for CPU cycles.
 * - Output data is copied from ONNX Runtime tensor before release.
 * - Prints timing metrics to stdout.
 */

/**
 * @brief Stops the device and halts inference operations.
 * 
 * This function signals the device to stop processing. Safe to call before unloading.
 * 
 * @param dev Pointer to the MLDevice structure.
 * @return int Returns 0 on success, -1 if device is invalid or not loaded.
 * 
 * Example:
 * @code
 * ml_dev_stop(dev);
 * @endcode
 */

#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include "onnxruntime_c_api.h"
#include <assert.h>
#include "ml_onnx_async.h"
#include <unistd.h>

#define ORT_ABORT_ON_ERROR(expr)                                      \
    do                                                                \
    {                                                                 \
        OrtStatus *onnx_status = (expr);                              \
        if (onnx_status != NULL)                                      \
        {                                                             \
            const char *msg = dev->api->GetErrorMessage(onnx_status); \
            printf("%s\n", msg);                                      \
            dev->api->ReleaseStatus(onnx_status);                     \
            abort();                                                  \
        }                                                             \
    } while (0);


#include "MLdev.h"
#include <stdlib.h>
#include <pthread.h>

int ml_dev_load(MLDevice *dev, MLModel* model, OrtEnv* shared_env)
{
    dev->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    int num_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    printf("number of cores are ::: %d\n", num_cores);
    dev->env = shared_env;
    ORT_ABORT_ON_ERROR(dev->api->CreateSessionOptions(&dev->session_options));
    
    // Check return value for DisablePerSessionThreads
    if (dev->api->DisablePerSessionThreads(dev->session_options) != ORT_OK) {
        // Handle error appropriately, e.g., log it or return an error code
        return -1;
    }

    ORT_ABORT_ON_ERROR(dev->api->SetIntraOpNumThreads(dev->session_options, 1));
    ORT_ABORT_ON_ERROR(dev->api->SetInterOpNumThreads(dev->session_options, 1));

    ORT_ABORT_ON_ERROR(dev->api->CreateSession(dev->env, model->model_path, dev->session_options, &dev->session));
    
    // Check return value for DisablePerSessionThreads again
    if (dev->api->DisablePerSessionThreads(dev->session_options) != ORT_OK) {
        // Handle error appropriately
        return -1;
    }

    ORT_ABORT_ON_ERROR(dev->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &dev->memory_info));
    ORT_ABORT_ON_ERROR(dev->api->GetAllocatorWithDefaultOptions(&dev->allocator));

    // Check return value for AddSessionConfigEntry
    if (dev->api->AddSessionConfigEntry(dev->session_options, "session.intra_op_thread_affinities", "0:1,1:2,2:3,3:4") != ORT_OK) {
        // Handle error appropriately
        return -1;
    }

    // Input/Output names (ONLY if needed)
    ORT_ABORT_ON_ERROR(dev->api->SessionGetInputName(dev->session, 0, dev->allocator, &model->input_name));
    ORT_ABORT_ON_ERROR(dev->api->SessionGetOutputName(dev->session, 0, dev->allocator, &model->output_name));

    OrtTypeInfo *input_type = NULL;
    ORT_ABORT_ON_ERROR(dev->api->SessionGetInputTypeInfo(dev->session, 0, &input_type));
    dev->api->ReleaseTypeInfo(input_type); // release it after use

    pthread_mutex_init(&model->pool_mutex, NULL);

    // Allocate input/output buffer pools
    model->input_buffer_size = 25 * sizeof(float);  // TODO: get actual input size
    model->output_buffer_size = 2 * sizeof(float);  // TODO: get actual output size
    model->input_pool_size = MAX_INPUT_BUFFERS;
    model->output_pool_size = MAX_OUTPUT_BUFFERS;

    for (size_t i = 0; i < model->input_pool_size; i++) {
        model->input_buffer_pool[i] = malloc(model->input_buffer_size);
    }
    for (size_t i = 0; i < model->output_pool_size; i++) {
        model->output_buffer_pool[i] = malloc(model->output_buffer_size);
    }

    dev->is_loaded = 1;
    return 0;
}


#include <stdlib.h>

int ml_dev_unload(MLDevice *dev, MLModel* model)
{
    if (!dev->is_loaded)
        return -1;

    // Free names with same allocator
    if (dev->input_name)
    {
        if (dev->api->AllocatorFree(dev->allocator, dev->input_name) != ORT_OK) {
            // Handle error appropriately, e.g., log it or return an error code
            return -1;
        }
        dev->input_name = NULL;
    }
    if (dev->output_name)
    {
        if (dev->api->AllocatorFree(dev->allocator, dev->output_name) != ORT_OK) {
            // Handle error appropriately
            return -1;
        }
        dev->output_name = NULL;
    }

    // Free input/output buffer pools
    if (model) {
        for (size_t i = 0; i < model->input_pool_size; i++) {
            free(model->input_buffer_pool[i]);
            model->input_buffer_pool[i] = NULL;
        }
        for (size_t i = 0; i < model->output_pool_size; i++) {
            free(model->output_buffer_pool[i]);
            model->output_buffer_pool[i] = NULL;
        }
        pthread_mutex_destroy(&model->pool_mutex);
    }

    dev->api->ReleaseSession(dev->session);
    dev->api->ReleaseSessionOptions(dev->session_options);
    dev->api->ReleaseMemoryInfo(dev->memory_info);
    dev->is_loaded = 0;
    return 0;
}



int ml_dev_infer(MLDevice *dev, MLModel* model, size_t input_buffer_index, size_t output_buffer_index, double *elapsed_time, uint64_t *cpu_cycles)
{
    if (!dev->is_running)
    {
        return -1;
    }

    if (!model)
    {
        return -1;
    }

    if (input_buffer_index >= model->input_pool_size || output_buffer_index >= model->output_pool_size)
    {
        return -1;
    }

    float* input_data = model->input_buffer_pool[input_buffer_index];
    float* output_data = model->output_buffer_pool[output_buffer_index];

    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;
    int64_t input_shape[2] = {1, (int64_t)(model->input_buffer_size / sizeof(float))};

    ORT_ABORT_ON_ERROR(dev->api->CreateTensorWithDataAsOrtValue(dev->memory_info, input_data, model->input_buffer_size, input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    struct timespec start_time, end_time;
    uint64_t start_cycles = 0, end_cycles = 0;
    unsigned int hi = 0, lo = 0;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

#if defined(__i386__)
    __asm__ volatile("rdtsc" : "=A"(start_cycles));
#elif defined(__x86_64__)
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    start_cycles = ((uint64_t)hi << 32) | lo;
#endif

    const char* input_names[] = {model->input_name};
    const char* output_names[] = {model->output_name};

    ORT_ABORT_ON_ERROR(dev->api->Run(dev->session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));

    clock_gettime(CLOCK_MONOTONIC, &end_time);

#if defined(__i386__)
    __asm__ volatile("rdtsc" : "=A"(end_cycles));
#elif defined(__x86_64__)
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    end_cycles = ((uint64_t)hi << 32) | lo;
#endif

    if (elapsed_time)
    {
        *elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    }

    if (end_cycles > start_cycles)
    {
        *cpu_cycles = end_cycles - start_cycles;
    }

    printf("Inference time: %.3f ms\n", *elapsed_time);
    if (*cpu_cycles > 0)
    {
        printf("CPU cycles taken: %" PRIu64 "\n", *cpu_cycles);
    }

    float* result_ptr = NULL;
    ORT_ABORT_ON_ERROR(dev->api->GetTensorMutableData(output_tensor, (void**)&result_ptr));
    for (size_t i = 0; i < model->output_buffer_size / sizeof(float); i++)
    {
        output_data[i] = result_ptr[i];
    }

    dev->api->ReleaseValue(output_tensor);
    dev->api->ReleaseValue(input_tensor);

    return 0;
}
int ml_dev_stop(MLDevice *dev)
{
    if (!dev || !dev->is_loaded)
    {
        return -1;
    }
    dev->is_running = false;
    return 0;
}
