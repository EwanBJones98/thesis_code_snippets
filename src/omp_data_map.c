#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[])
{

    // Allocate data for three arrays on the host device
    int *A, *B, *C;
    int arr_length=1000;
    A = malloc(sizeof(int) * arr_length);
    B = malloc(sizeof(int) * arr_length);
    C = malloc(sizeof(int) * arr_length);

    // Allocate data for these three arrays on the target device
    #pragma omp target enter data map(alloc: A[:arr_length], B[:arr_length], C[:arr_length])

    // Give arrays A and B some values on the host
    for (int i=0; i<arr_length; i++)
    {
        A[i] = i + 44;
        B[i] = i * i * i;
    }

    // Copy the values of A and B to the device
    #pragma omp target update to(A[:arr_length], B[:arr_length])

    // Compute the element-wise sum A + B = C on the GPU in a parallel for loop
    // Necessary data has been mapped to the device already
    #pragma omp target teams distribute parallel for
    for (int i=0; i<arr_length; i++)
    {
        C[i] = A[i] + B[i];
    }

    // Copy values from array C back to the host
    #pragma omp target update from(C[:arr_length])

    // Free memory associated with all arrays on the target device
    #pragma omp target exit data map(delete: A[:arr_length], B[:arr_length], C[:arr_length])

    printf("[");
    for (int i=0; i<arr_length; i++)
    {
        printf("%d,", C[i]);
    }
    printf("]\n");

    return 0;
}