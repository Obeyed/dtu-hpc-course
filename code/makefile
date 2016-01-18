GPP=g++
NVCC=nvcc
ARCH_SM=sm_13 # change this according to the device's compute capability
RADIX_FOLDER=radix_sort

radix: compare_results.cu radix_sort.cu reference_calc.cpp utils.h
	$(GPP) -c $(RADIX_FOLDER)/reference_calc.cpp \
		&& $(NVCC) -c $(RADIX_FOLDER)/compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c $(RADIX_FOLDER)/radix_sort.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test $(RADIX_FOLDER)/reference_calc.o $(RADIX_FOLDER)/compare_results.o $(RADIX_FOLDER)/radix_sort.o
