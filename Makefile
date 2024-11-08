omp_cpu_1: ./src/omp_cpu_1.c
	gcc -fopenmp -o ./exec/omp_cpu_1 ./src/omp_cpu_1.c

omp_target_region_map: ./src/omp_target_region_map.c
	clang -fopenmp --offload-arch=sm_89 -o ./exec/omp_target_region_map ./src/omp_target_region_map.c

omp_data_map: ./src/omp_data_map.c
	clang -fopenmp --offload-arch=sm_89 -o ./exec/omp_data_map ./src/omp_data_map.c

omp_struct_mapping: ./src/omp_struct_mapping.c
	clang -lm -fopenmp --offload-arch=sm_89 -o ./exec/omp_struct_mapping ./src/omp_struct_mapping.c
