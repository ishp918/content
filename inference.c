#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include "onnxruntime_c_api.h"
#include <assert.h>

const OrtApi *ort_p = NULL;

static OrtEnv* env;
#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = ort_p->GetErrorMessage(onnx_status); \
      printf("%s\n", msg);                                   \
      ort_p->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

static void usage() { printf("usage: <model_path>\n"); }

int run_inference(OrtSession* session)
{
  struct timespec start_time, end_time;
  uint64_t start_cycles = 0, end_cycles = 0;
  unsigned int hi = 0, lo = 0;

  // Get start time
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  // Optional: Read CPU cycles if on x86 and supported
#if defined(__i386__)
  __asm__ volatile ("rdtsc" : "=A" (start_cycles));
#elif defined(__x86_64__)
  __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
  start_cycles = ((uint64_t)hi << 32) | lo;
#endif

  OrtMemoryInfo* memory_info;
  ORT_ABORT_ON_ERROR(ort_p->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtAllocator* allocator;
  ORT_ABORT_ON_ERROR(ort_p->GetAllocatorWithDefaultOptions(&allocator));
  double model_input[] = {3.23223552e+09, 3.23223552e+09, 4.65800000e+04, 8.00000000e+01,
  1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.26237569e+06,
  0.00000000e+00, 0.00000000e+00, 1.34000000e+02, 6.50000000e+02,
  5.00000000e+00, 5.00000000e+00, 9.39944983e-03, 0.00000000e+00,
  1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00,
  0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00,
  0.00000000e+00};
  
  float model_input_float[25];
  for(size_t i=0;i<25;i++){
    model_input_float[i] = (float)model_input[i];
  }
  size_t model_input_len = 25 * sizeof(float);

  OrtValue* input_tensor = NULL;
  const int64_t input_shape[] = {1,25};
  size_t shape_len = 2;
  ORT_ABORT_ON_ERROR(ort_p->CreateTensorWithDataAsOrtValue(memory_info, model_input_float, model_input_len, input_shape,
                                                           shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(ort_p->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);

  ort_p->ReleaseMemoryInfo(memory_info);

  size_t input_cnt = 1;
  const char* input_names[] = {"X"};
  size_t output_cnt = 1;
  const char* output_names[] = {"probabilities"};
  OrtValue* output_tensor = NULL;

  ORT_ABORT_ON_ERROR(ort_p->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, input_cnt, output_names, output_cnt,
                                &output_tensor));
  assert(output_tensor!=NULL);
  ORT_ABORT_ON_ERROR(ort_p->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);
  if (!is_tensor) {
        printf("Output is not a simple tensor. Model output type is seq(map(int64,tensor(float)))\n");
  }else{
    printf("Tensor Output \n");
  }

  // Get end time
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  // Optional: Read CPU cycles end
#if defined(__i386__)
  __asm__ volatile ("rdtsc" : "=A" (end_cycles));
#elif defined(__x86_64__)
  __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
  end_cycles = ((uint64_t)hi << 32) | lo;
#endif

  // Calculate elapsed time in microseconds
  uint64_t elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000ULL + (end_time.tv_nsec - start_time.tv_nsec);
  double elapsed_ms = elapsed_ns / 1000000.0;

  // Calculate cycles taken if cycle count was read
  uint64_t cycles_taken = 0;
  if (end_cycles > start_cycles) {
    cycles_taken = end_cycles - start_cycles;
  }

  printf("Inference time: %.3f ms\n", elapsed_ms);
  if (cycles_taken > 0) {
    printf("CPU cycles taken: %" PRIu64 "\n", cycles_taken);
  }

  // Additional details about inference

  // Print input tensor shape and size
  OrtTensorTypeAndShapeInfo* input_info;
  ORT_ABORT_ON_ERROR(ort_p->GetTensorTypeAndShape(input_tensor, &input_info));
  size_t input_dim_count;
  ORT_ABORT_ON_ERROR(ort_p->GetDimensionsCount(input_info, &input_dim_count));
  int64_t input_dims[8];
  ORT_ABORT_ON_ERROR(ort_p->GetDimensions(input_info, input_dims, input_dim_count));
  printf("Input tensor shape: [");
  for (size_t i = 0; i < input_dim_count; i++) {
    printf("%ld%s", input_dims[i], (i == input_dim_count - 1) ? "" : ", ");
  }
  printf("]\n");
  ort_p->ReleaseTensorTypeAndShapeInfo(input_info);

  // Print output tensor shape and size
  OrtTensorTypeAndShapeInfo* output_info;
  ORT_ABORT_ON_ERROR(ort_p->GetTensorTypeAndShape(output_tensor, &output_info));
  size_t output_dim_count;
  ORT_ABORT_ON_ERROR(ort_p->GetDimensionsCount(output_info, &output_dim_count));
  int64_t output_dims[8];
  ORT_ABORT_ON_ERROR(ort_p->GetDimensions(output_info, output_dims, output_dim_count));
  printf("Output tensor shape: [");
  for (size_t i = 0; i < output_dim_count; i++) {
    printf("%ld%s", output_dims[i], (i == output_dim_count - 1) ? "" : ", ");
  }
  printf("]\n");

  // Get output tensor data pointer
  float* output_tensor_data = NULL;
  ORT_ABORT_ON_ERROR(ort_p->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));

  size_t total_len;
  ORT_ABORT_ON_ERROR(ort_p->GetTensorShapeElementCount(output_info,&total_len));
  float max_prob = output_tensor_data[0];
  size_t max_index=0;
  for(size_t i=0;i<total_len;i++){
    printf("class %zu: %f\n",i,output_tensor_data[i]);
    if(output_tensor_data[i]>max_prob){
      max_prob = output_tensor_data[i];
      max_index=i;
    }
  }
  printf("\n Predicted Class: %zu with probabilty: %f\n",max_index,max_prob);
  ort_p->ReleaseTensorTypeAndShapeInfo(output_info);

  ort_p->ReleaseValue(output_tensor);
  ort_p->ReleaseValue(input_tensor);
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc < 1)
  {
    usage();
    return -1;
  }
  ORTCHAR_T* model_path = argv[1];
  ort_p = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!ort_p) {
    fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
    return -1;
  }

  ORT_ABORT_ON_ERROR(ort_p->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);
  OrtSessionOptions* session_options;
  //OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
  ORT_ABORT_ON_ERROR(ort_p->CreateSessionOptions(&session_options));
  OrtSession* session;
  ORT_ABORT_ON_ERROR(ort_p->CreateSession(env, model_path, session_options, &session));
  printf("model loaded\n");
  run_inference(session);
  printf("Inference done.\n");
  ort_p->ReleaseSessionOptions(session_options);
  ort_p->ReleaseSession(session);
  ort_p->ReleaseEnv(env);
  return 0;

}
