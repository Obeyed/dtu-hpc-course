GPP=g++
NVCC=nvcc
ARCH_SM=sm_35 # change this according to the device's compute capability

fast_histo: fast_coarse_histo.cu radix_sort.cu utils.h
	$(NVCC) -c fast_coarse_histo.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c radix_sort.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test fast_coarse_histo.o radix_sort.o \
		&& ./test
