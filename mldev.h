#ifndef MLDEV_H
#define MLDEV_H

#include <stddef.h>
#include <stdint.h>

typedef struct MLDevice {
    void *env;
    void *session_options;
    void *session;
    int is_loaded;
    int is_running;
} MLDevice;

int ml_dev_load(MLDevice *dev, const char *model_path);
int ml_dev_start(MLDevice *dev);
int ml_dev_infer(MLDevice *dev, float *input_data, int input_count, size_t feature_cnt, float *output_data, size_t output_cnt, double* elapsed_time, uint64_t* cpu_cycles);
int ml_dev_stop(MLDevice *dev);
int ml_dev_unload(MLDevice *dev);

#endif // MLDEV_H
