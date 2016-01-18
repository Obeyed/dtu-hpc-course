GPP=g++
NVCC=nvcc
ARCH_SM=sm_13 # change this according to the device's compute capability

radix: compare_results.cu radix_sort.cu reference_calc.cpp utils.h
	$(GPP) -c reference_calc.cpp \
		&& $(NVCC) -c compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c radix_sort.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test reference_calc.o compare_results.o radix_sort.o
