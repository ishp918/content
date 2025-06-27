#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include "onnxruntime_c_api.h"
#include <assert.h>
#include "ml_onnx_async.h"
#include <unistd.h>

#define ORT_ABORT_ON_ERROR(expr)                                 \
    do                                                           \
    {                                                            \
        OrtStatus *onnx_status = (expr);                         \
        if (onnx_status != NULL)                                 \
        {                                                        \
            const char *msg = ort->GetErrorMessage(onnx_status); \
            printf("%s\n", msg);                                 \
            ort->ReleaseStatus(onnx_status);                     \
            abort();                                             \
        }                                                        \
    } while (0);

static const OrtApi *ort = NULL;

int ml_dev_load(MLDevice *dev, const char *model_path)
{
ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort)
    {
        fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
        return -1;
    }
    // OrtStatus *status;
    ORT_ABORT_ON_ERROR(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mldev", &dev->env));
    ORT_ABORT_ON_ERROR(ort->CreateSessionOptions(&dev->session_options));
   
    int num_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    printf("number of cores are :::->>>>>> %d\n",num_cores);
    ORT_ABORT_ON_ERROR(ort->SetIntraOpNumThreads(dev->session_options, 1));
    ORT_ABORT_ON_ERROR(ort->SetInterOpNumThreads(dev->session_options, 1)); 
    // ORT_ABORT_ON_ERROR(ort->SetSessionExecutionMode(dev->session_options, ORT_SEQUENTIAL));
    // ORT_ABORT_ON_ERROR(ort->SetSessionGraphOptimizationLevel(dev->session_options, ORT_ENABLE_ALL));
    ORT_ABORT_ON_ERROR(ort->CreateSession(dev->env, model_path, dev->session_options, &dev->session));
    
    // Enable optimizations
  
    dev->is_loaded = true;
    dev->is_running = false;
    return 0;
}
int ml_dev_start(MLDevice *dev)
{
    if (!dev->is_loaded)
        return -1;
    dev->is_running = true;
    return 0;
}

int ml_dev_unload(MLDevice *dev)
{
    if (dev->session)
    {
        ort->ReleaseSession(dev->session);
    }
    if (dev->session_options)
    {
        ort->ReleaseSessionOptions(dev->session_options);
    }
    if (dev->env)
    {
        ort->ReleaseEnv(dev->env);
    }

    dev->session = NULL;
    dev->session_options = NULL;
    dev->env = NULL;

    return 0;
}



int ml_dev_infer(MLDevice *dev, float *input_data, int input_count, size_t feature_cnt, float *output_data, size_t output_cnt,double* elapsed_time, uint64_t* cpu_cycles)
{

    printf("Running inference...\n");
    if (!dev->is_running)
    {
        return -1;
    }

    //Structure of Timestamp structure
    struct timespec start_time, end_time;
    uint64_t start_cycles = 0, end_cycles = 0;
    unsigned int hi = 0, lo = 0;

    // Get start time
    clock_gettime(CLOCK_MONOTONIC, &start_time);

// Optional: Read CPU cycles if on x86 and supported
#if defined(__i386__)
    __asm__ volatile("rdtsc" : "=A"(start_cycles));
#elif defined(__x86_64__)
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    start_cycles = ((uint64_t)hi << 32) | lo;
#endif
    OrtMemoryInfo *memory_info;
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    int64_t input_shape[2] = {(int64_t)input_count, (int64_t)feature_cnt};
    ORT_ABORT_ON_ERROR(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    ORT_ABORT_ON_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, feature_cnt * input_count * sizeof(float), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    const char *input_names[] = {"X"};
    const char *output_names[] = {"probabilities"};
    ORT_ABORT_ON_ERROR(ort->Run(dev->session, NULL, input_names, (const OrtValue *[]){input_tensor}, 1, output_names, 1, &output_tensor));
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Optional: Read CPU cycles end
#if defined(__i386__)
    __asm__ volatile("rdtsc" : "=A"(end_cycles));
#elif defined(__x86_64__)
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    end_cycles = ((uint64_t)hi << 32) | lo;
#endif

    // Calculating elapsed time in microseconds
    if(elapsed_time) {
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
    float *result_ptr = NULL;
    ORT_ABORT_ON_ERROR(ort->GetTensorMutableData(output_tensor, (void **)&result_ptr));
    for (size_t i = 0; i < output_cnt; i++)
    {
        output_data[i] = result_ptr[i];
    }
    ort->ReleaseValue(output_tensor);
    ort->ReleaseValue(input_tensor);
    ort->ReleaseMemoryInfo(memory_info);
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
