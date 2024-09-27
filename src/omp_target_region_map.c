#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    // Allocate data for three arrays on the host device
    int *A, *B, *C;
    int arr_lengths=1000;
    A = malloc(sizeof(int) * arr_lengths);
    B = malloc(sizeof(int) * arr_lengths);
    C = malloc(sizeof(int) * arr_lengths);

    // Give arrays A and B some values
    for (int i=0; i<arr_lengths; i++)
    {
        A[i] = i + 44;
        B[i] = i * i * i;
    }

    // Compute the element-wise sum A + B = C on the GPU in a parallel target region
    // Necessary data will be mapped using the `map` clause.
    #pragma omp target teams distribute parallel for map(tofrom:C[:arr_lengths]) map(to:A[:arr_lengths],B[:arr_lengths])
    for (int i=0; i<arr_lengths; i++)
    {
        C[i] = A[i] + B[i];
    }

    printf("[");
    for (int i=0; i<arr_lengths; i++)
    {
        printf("%d,", C[i]);
    }
    printf("]\n");

    return 0;
}