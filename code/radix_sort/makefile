default: compare_results.cu radix_sort.cu reference_calc.cpp utils.h
	g++ -c reference_calc.cpp && nvcc -c compare_results.cu -arch=sm_13 && nvcc -c radix_sort.cu -arch=sm_13 && nvcc -o test reference_calc.o compare_results.o radix_sort.o
