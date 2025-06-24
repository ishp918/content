//The program creates multiple independent instances of the same model 
//by loading it multiple times into separate MLDevice structs.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "ml_onnx_async.h"

#define MAX_MODELS 32
#define FEATURES 25
#define CLASSES 2

// Sample input data for inference
float sample_input[FEATURES] = {
    3.23223552e+09, 3.23223552e+09, 4.65800000e+04, 8.00000000e+01,
    1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.26237569e+06,
    0.00000000e+00, 0.00000000e+00, 1.34000000e+02, 6.50000000e+02,
    5.00000000e+00, 5.00000e+00, 9.39944983e-03, 0.00000000e+00,
    1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00,
    0.00000000e+00
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    MLDevice devices[MAX_MODELS];  //array of MLDevice structs, one for each model instance.
    float outputs[MAX_MODELS][CLASSES]; //stores output probabilities for each model instance.
    double elapsed_time = 0;
    uint64_t cpu_cycles = 0;

    // Load models
    for (int i = 0; i < MAX_MODELS; i++) {  //loads the same ONNX model into each MLDevice instance.
        if (ml_dev_load(&devices[i], model_path) != 0) {
            fprintf(stderr, "Failed to load model instance %d\n", i);
            return -1;
        }
        ml_dev_start(&devices[i]);
    }

    // Run sequential inference on each model instance
    // Calls ml_dev_infer with the sample input to perform inference on that instance.
    for (int i = 0; i < MAX_MODELS; i++) {
        printf("Running inference on model instance %d\n", i);
        if (ml_dev_infer(&devices[i], sample_input, 1, FEATURES, outputs[i], CLASSES, &elapsed_time, &cpu_cycles) != 0) {
            fprintf(stderr, "Inference failed on model instance %d\n", i);
            continue;
        }
        printf("Inference time: %.3f ms, CPU cycles: %" PRIu64 "\n", elapsed_time, cpu_cycles);
        printf("Output probabilities: ");
        for (int j = 0; j < CLASSES; j++) {
            printf("%.6f ", outputs[i][j]);
        }
        printf("\n");
    }

    // Unload models
    for (int i = 0; i < MAX_MODELS; i++) {
        ml_dev_stop(&devices[i]);
        ml_dev_unload(&devices[i]);
    }

    return 0;
}
