#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
    #include "omp.h"
#endif

int main(int argc, char *argv[])
{
    int N;
    #pragma omp parallel
    N = omp_get_num_threads();

    int evenN=0, oddN=0;
    #pragma omp parallel reduction(+: evenN, oddN)
    {
        if (omp_get_thread_num() % 2 == 0)
        {
            evenN+=1;
        } else {
            oddN+=1;
        }
    }

    printf("Total number of threads = %d\n", N);
    printf("Number of threads with even IDs = %d\n", evenN);
    printf("Number of threads with odd IDs = %d\n", oddN);

    return 0;
}