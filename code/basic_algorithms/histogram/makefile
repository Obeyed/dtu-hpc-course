GPP=g++
NVCC=nvcc
ARCH_SM=sm_13 # change this according to the device's compute capability

reduce: compare_results.cu par_histogram.cu reference_calc.cpp utils.h
	$(GPP) -c reference_calc.cpp \
		&& $(NVCC) -c compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c par_histogram.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test reference_calc.o compare_results.o par_histogram.o \
		&& ./test
