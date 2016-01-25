GPP=g++
NVCC=nvcc
ARCH_SM=sm_35 # change this according to the device's compute capability

coarse_histo: coarse_histogram.cu radix_sort.cu utils.h 
	$(NVCC) -c coarse_histogram.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c radix_sort.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test coarse_histogram.o radix_sort.o \
		&& ./test
