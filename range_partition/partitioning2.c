#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DATA_SIZE_1 100000
#define DATA_SIZE_2 10000
#define NUM_CORES 4
#define TEXT_LENGTH 10
#define UNION_SIZE (DATA_SIZE_1 + DATA_SIZE_2)  // Max possible size after union

// Structure to hold table data (random number and random text)
typedef struct {
    int number;
    char text[TEXT_LENGTH];
} TableRow;

// Function to generate a random string
void generate_random_text(char *str, int length) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < length - 1; i++) {
        str[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    str[length - 1] = '\0';
}

// Comparator function for sorting by number
int compare_rows(const void *a, const void *b) {
    return ((TableRow *)a)->number - ((TableRow *)b)->number;
}

int main() {
    TableRow data1[DATA_SIZE_1];
    TableRow data2[DATA_SIZE_2];
    TableRow union_table[UNION_SIZE];

    // Seed the random number generator
    srand(42);

    // Generate random data for data1 and data2
    for (int i = 0; i < DATA_SIZE_1; i++) {
        data1[i].number = rand() % 200000;  // Random number range
        generate_random_text(data1[i].text, TEXT_LENGTH);
    }
    for (int i = 0; i < DATA_SIZE_2; i++) {
        data2[i].number = rand() % 200000;
        generate_random_text(data2[i].text, TEXT_LENGTH);
    }

    // Merge data1 and data2 into union_table
    memcpy(union_table, data1, DATA_SIZE_1 * sizeof(TableRow));
    memcpy(union_table + DATA_SIZE_1, data2, DATA_SIZE_2 * sizeof(TableRow));

    int total_size = DATA_SIZE_1 + DATA_SIZE_2;

    // Sort the merged table by number for easy union operation
    qsort(union_table, total_size, sizeof(TableRow), compare_rows);

    // Perform union (remove duplicates)
    TableRow final_union[UNION_SIZE]; // Resulting table after union
    int final_size = 0;

    // Parallelizing the union operation
    #pragma omp parallel num_threads(NUM_CORES)
    {
        int core_id = omp_get_thread_num();
        int start = (core_id * total_size) / NUM_CORES;
        int end = ((core_id + 1) * total_size) / NUM_CORES;

        for (int i = start; i < end; i++) {
            if (i == 0 || union_table[i].number != union_table[i - 1].number) {
                #pragma omp critical  // Ensure unique insertion without race conditions
                {
                    final_union[final_size++] = union_table[i];
                }
            }
        }
    }

    // Open file for writing results
    FILE *output_file = fopen("union_results.txt", "w");
    if (output_file == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    // Write results to the file
    #pragma omp parallel num_threads(NUM_CORES)
    {
        int core_id = omp_get_thread_num();
        int start = (core_id * final_size) / NUM_CORES;
        int end = ((core_id + 1) * final_size) / NUM_CORES;

        // Use critical section to prevent race conditions during file writing
        #pragma omp critical
        {
            for (int i = start; i < end; i++) {
                fprintf(output_file, "Core %d processed: %d, %s\n", core_id + 1, final_union[i].number, final_union[i].text);
            }
        }
    }

    fclose(output_file);
    printf("Union operation complete. Results written to 'union_results.txt'.\n");
    return 0;
}
