#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DATA_SIZE_1 100000
#define DATA_SIZE_2 10000
#define NUM_CORES 4

int main() {
    int data1[DATA_SIZE_1];
    int data2[DATA_SIZE_2];

    // Populate data arrays with some values for testing
    for (int i = 0; i < DATA_SIZE_1; i++) data1[i] = i + 1;
    for (int i = 0; i < DATA_SIZE_2; i++) data2[i] = i + 1;

    int partition_size_1 = DATA_SIZE_1 / NUM_CORES;

    // Open file for writing
    FILE *output_file = fopen("partitioning_results.txt", "w");
    if (output_file == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    #pragma omp parallel num_threads(NUM_CORES)
    {
        int core_id = omp_get_thread_num();
        int start = core_id * partition_size_1;
        int end = (core_id == NUM_CORES - 1) ? DATA_SIZE_1 : (core_id + 1) * partition_size_1;

        // Use a critical section to prevent race conditions during file writing
        #pragma omp critical
        {
            fprintf(output_file, "Core %d processing range %d to %d\n", core_id + 1, start, end);
            for (int i = start; i < end; i++) {
                fprintf(output_file, "Core %d processed data[%d] = %d\n", core_id + 1, i, data1[i]);
            }
        }
    }

    fclose(output_file);
    printf("Processing complete. Results written to 'partitioning_results.txt'.\n");
    return 0;
}
