GPP=g++
NVCC=nvcc
ARCH_SM=sm_13 # change this according to the device's compute capability
RADIX_FOLDER=radix_sort/
ALGORITHMS=basic_algorthms/
MAP_FOLDER=$(ALGORITHMS)map/

radix: $(RADIX_FOLDER)compare_results.cu $(RADIX_FOLDER)radix_sort.cu $(RADIX_FOLDER)reference_calc.cpp $(RADIX_FOLDER)utils.h
	$(GPP) -c $(RADIX_FOLDER)reference_calc.cpp \
		&& $(NVCC) -c $(RADIX_FOLDER)compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c $(RADIX_FOLDER)radix_sort.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test $(RADIX_FOLDER)reference_calc.o $(RADIX_FOLDER)compare_results.o $(RADIX_FOLDER)radix_sort.o

map: $(MAP_FOLDER)compare_results.cu $(MAP_FOLDER)par_mapping.cu $(MAP_FOLDER)reference_calc.h $(MAP_FOLDER)utils.h
	$(GPP) -c $(MAP_FOLDER)reference_calc.h \
		&& $(NVCC) -c $(MAP_FOLDER)compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c $(MAP_FOLDER)par_mapping.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test $(MAP_FOLDER)reference_calc.o $(MAP_FOLDER)compare_results.o $(MAP_FOLDER)par_mapping.o
