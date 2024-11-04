#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct
{
    double *array_one;
    double *array_two;

    int array_one_length;
    int array_two_length;

    int mode;
} fields;

void map_fields_to_gpu(fields *my_fields)
{
    #pragma omp target enter data map(alloc: my_fields->array_one[:my_fields->array_one_length],\
                                             my_fields->array_two[:my_fields->array_two_length],\
                                             my_fields->array_one_length,\
                                             my_fields->array_two_length,\
                                             my_fields->mode)
    
    #pragma omp target update to(my_fields->array_one[:my_fields->array_one_length],\
                                 my_fields->array_two[:my_fields->array_two_length],\
                                 my_fields->array_one_length,\
                                 my_fields->array_two_length,\
                                 my_fields->mode)
}

void map_fields_from_gpu(fields *my_fields)
{
    #pragma omp target update from(my_fields->array_one[:my_fields->array_one_length],\
                                   my_fields->array_two[:my_fields->array_two_length],\
                                   my_fields->mode)

    #pragma omp target exit data map(delete: my_fields->array_one[:my_fields->array_one_length],\
                                             my_fields->array_two[:my_fields->array_two_length],\
                                             my_fields->array_one_length,\
                                             my_fields->array_two_length,\
                                             my_fields->mode)
}


int main(int argc, char **argv)
{
    // Initialise struct
    fields my_fields;
    my_fields.array_one_length = 100;
    my_fields.array_two_length = 5 * my_fields.array_one_length;
    my_fields.array_one = malloc(sizeof(double) * my_fields.array_one_length);
    my_fields.array_two = malloc(sizeof(double) * my_fields.array_two_length);
    my_fields.mode = 2;

    // Populate the arrays with initial values
    for (int i=0; i<my_fields.array_one_length; i++) my_fields.array_one[i] = i;
    for (int i=0; i<my_fields.array_two_length; i++) my_fields.array_two[i] = i;

    // Print initial values of array one
    fprintf(stdout, "\n\n[");
    for (int i=0; i<my_fields.array_one_length; i++) fprintf(stdout, "%f,", my_fields.array_one[i]);
    fprintf(stdout, "]\n\n");

    // Call function which maps the struct to the GPU
    map_fields_to_gpu(&my_fields);

    // Do some calculation on the GPU
    #pragma omp target teams distribute parallel for
    for (int i=0; i<my_fields.array_one_length; i++)
    {
        my_fields.array_one[i] = pow(my_fields.array_two[i * 5], my_fields.mode);
    }

    // Call function which maps the struct to the CPU
    map_fields_from_gpu(&my_fields);

    // Print final values of array one
    fprintf(stdout, "\n\n[");
    for (int i=0; i<my_fields.array_one_length; i++) fprintf(stdout, "%f,", my_fields.array_one[i]);
    fprintf(stdout, "]\n\n");

    return 0;
}