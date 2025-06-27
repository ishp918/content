#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ml_onnx_async.h"

#define MAX_ROWS 50
#define FEATURES 25
#define CLASSES 2

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
        // char *token = strtok(line,",");
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
int main(int argc, char *argv[])
{
    const char *model_path = (argc > 1) ? argv[1] : "rf_10_pkts_model_sklearn.onnx";
    const char *csv_path = (argc > 2) ? argv[2] : "data.csv";
    const char* output_path = "output.txt";
    MLDevice dev;
    if (ml_dev_load(&dev, model_path) != 0)
    {
        fprintf(stderr, "Error loading model\n");
        return 1;
    }
    ml_dev_start(&dev);
    float inputs[MAX_ROWS][FEATURES];
    size_t row_count;
    if(read_csv(csv_path,inputs,&row_count)!=0){
        fprintf(stderr,"Error reading CSV file\n");
        return 1;
    }
    double elapsed_time = 0;
    uint64_t cpu_cycle = 0;
    printf("%zu",row_count);
    FILE *output_file = fopen("output.txt","w");
    if(!output_file){
        fprintf(stderr,"Error opening output file\n");
        return -1;
    }

    // double model_input[3][25] = {{167837989.000000,
    //                               3112025659.000000, 40694.000000, 443.000000, 1.000000,
    //                               0.000000, 18.000000, 1192.198238, 0.000000, 0.000000, 169.400000,
    //                               968.400000, 5.000000, 5.000000, 0.022276, 0.000000, 1.000000, 1.000000,
    //                               0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000},
    //                              {167837989.000000, 1609281240.000000, 51413.000000, 16208.000000,
    //                               0.000000, 0.000000, 0.000000, 1403073.106734, 0.800000, 1.000000, 155.000000,
    //                               129.000000, 5.000000, 5.000000, 0.065938, 0.000000, 1.000000, 0.000000, 1.000000,
    //                               1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000},
    //                              {167837989.000000, 3112025659.000000, 40698.000000, 443.000000, 1.000000, 0.000000, 0.000000, 1713.923596, 0.000000, 0.000000, 194.600000, 978.600000, 5.000000, 5.000000, 0.013926, 0.000000,
    //                               1.000000, 1.000000, 0.000000, 1.000000, 0.000000,
    //                               1.000000, 0.000000, 0.000000, 1.000000},
    //                              {167837989.000000, 1423995296.000000, 51413.000000, 16973.000000, 0.000000, 0.000000, 0.000000,
    //                               996123.762980, 0.400000, 1.000000, 159.400000, 178.200000, 5.000000, 5.000000, 0.064660, 0.000000,
    //                               1.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000}};

    // float model_input_float[4][25];
    // for (int k = 0; k < 4; k++)
    // {
    //     for (size_t i = 0; i < 25; i++)
    //     {
    //         model_input_float[k][i] = (float)model_input[k][i];
    //     }
    // }

    // size_t model_input_len = 25 * sizeof(float);
    // printf("%f",inputs[4]);
    printf("Running inference on %zu rows\n", row_count);
    for (size_t i = 0; i < row_count; i++)
    {
        float output[CLASSES] = {0};
        ml_dev_infer(&dev, inputs[i], 1, FEATURES, output, CLASSES,&elapsed_time,&cpu_cycle);
        printf("Row %zu: ", i + 1);
        int max_idx = 0;
        for (int j = 0; j < CLASSES; j++)
        {
            printf("Class %d: %.3f\n ", j, output[j]);
            if (output[j] > output[max_idx])
                max_idx = j;
        }
        printf("Row %zu: Predicted Class: %d\n",i+1,max_idx);
        fprintf(output_file,"Row %zu: Predicted Class: %d | Time: %.3f ms | Cycles: %" PRIu64 "\n",i+1,max_idx,elapsed_time,cpu_cycle);
    }
    ml_dev_stop(&dev);
    ml_dev_unload(&dev);
    return 0;
}



/*#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ml_onnx_async.h"
#include "MLdev.h"
#define MAX_ROWS 1500
#define FEATURES 25
#define CLASSES 2




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
int main(int argc, char *argv[])
{
    const char *model_path = (argc > 1) ? argv[1] : "rf_10_pkts_model_sklearn.onnx";
    const char *csv_path = (argc > 2) ? argv[2] : "data.csv";
    const char* output_path = "output.txt";
    MLDevice dev;
    if (ml_dev_load(&dev, model_path) != 0)
    {
        fprintf(stderr, "Error loading model\n");
        return 1;
    }
    ml_dev_start(&dev);
    float inputs[MAX_ROWS][FEATURES];
    size_t row_count;
    if(read_csv(csv_path,&inputs,&row_count)!=0){
        fprintf(stderr,"Error reading CSV file\n");
        return 1;
    }
    double elapsed_time = 0;
    uint64_t cpu_cycle = 0;
    printf("%d",row_count);
    FILE *output_file = fopen("output.txt","w");
    if(!output_file){
        fprintf(stderr,"Error opening output file\n");
        return -1;
    }

    // double model_input[3][25] = {{167837989.000000,
    //                               3112025659.000000, 40694.000000, 443.000000, 1.000000,
    //                               0.000000, 18.000000, 1192.198238, 0.000000, 0.000000, 169.400000,
    //                               968.400000, 5.000000, 5.000000, 0.022276, 0.000000, 1.000000, 1.000000,
    //                               0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000},
    //                              {167837989.000000, 1609281240.000000, 51413.000000, 16208.000000,
    //                               0.000000, 0.000000, 0.000000, 1403073.106734, 0.800000, 1.000000, 155.000000,
    //                               129.000000, 5.000000, 5.000000, 0.065938, 0.000000, 1.000000, 0.000000, 1.000000,
    //                               1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000},
    //                              {167837989.000000, 3112025659.000000, 40698.000000, 443.000000, 1.000000, 0.000000, 0.000000, 1713.923596, 0.000000, 0.000000, 194.600000, 978.600000, 5.000000, 5.000000, 0.013926, 0.000000,
    //                               1.000000, 1.000000, 0.000000, 1.000000, 0.000000,
    //                               1.000000, 0.000000, 0.000000, 1.000000},
    //                              {167837989.000000, 1423995296.000000, 51413.000000, 16973.000000, 0.000000, 0.000000, 0.000000,
    //                               996123.762980, 0.400000, 1.000000, 159.400000, 178.200000, 5.000000, 5.000000, 0.064660, 0.000000,
    //                               1.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000}};

    // float model_input_float[4][25];
    // for (int k = 0; k < 4; k++)
    // {
    //     for (size_t i = 0; i < 25; i++)
    //     {
    //         model_input_float[k][i] = (float)model_input[k][i];
    //     }
    // }

    // size_t model_input_len = 25 * sizeof(float);
    // printf("%f",inputs[4]);







    double total_time = 0;
    uint64_t total_cycles = 0;
    printf("Running inference on %zu rows\n", row_count);
    for (size_t i = 0; i < row_count+1; i++)
    {
        float output[CLASSES] = {0};
        ml_dev_infer(&dev, inputs[i], 1, FEATURES, output, CLASSES,&elapsed_time,&cpu_cycle);
        total_time += elapsed_time;
        total_cycles+=cpu_cycle;
        printf("Row %zu: ", i + 1);
        int max_idx = 0;
        for (int j = 0; j < CLASSES; j++)
        {
            printf("Class %d: %.3f\n ", j, output[j]);
            if (output[j] > output[max_idx])
                max_idx = j;
        }
        printf("Row %zu: Predicted Class: %d\n",i+1,max_idx);
        fprintf(output_file,"Row %zu: Predicted Class: %d | Time: %.3f ms | Cycles: %" PRIu64 "\n",i+1,max_idx,elapsed_time,cpu_cycle);
    }
    double avg_time = total_time/(row_count+1);
    double avg_cycles = (double)total_cycles/(row_count+1);
    fprintf(output_file,"\n========Summary======\n");
    fprintf(output_file,"Total Rows : %zu\n",row_count);
    fprintf(output_file,"Average Time per Inference: %.3f ms\n",avg_time);
    fprintf(output_file,"Average CPU Cycles per Inference: %.3f \n",avg_cycles);
    ml_dev_stop(&dev);
    ml_dev_unload(&dev);
    return 0;
}
*/
