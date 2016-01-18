GPP=g++
NVCC=nvcc
ARCH_SM=sm_13 # change this according to the device's compute capability
MKFILE_PATH=$(shell pwd) # must be executed from the makefile's folder
RADIX_FOLDER=radix_sort/
ALGORITHMS=basic_algorthms/
MAP_FOLDER=$(ALGORITHMS)map/

radix: $(RADIX_FOLDER)compare_results.cu $(RADIX_FOLDER)radix_sort.cu $(RADIX_FOLDER)reference_calc.cpp $(RADIX_FOLDER)utils.h
	cd $(MKFILE_PATH)/$(RADIX_FOLDER) \
		&& $(GPP) -c reference_calc.cpp \
		&& $(NVCC) -c compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c radix_sort.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test reference_calc.o compare_results.o radix_sort.o \
		&& ./test \
		&& cd $(MKFILE_PATH)

map: $(MAP_FOLDER)compare_results.cu $(MAP_FOLDER)par_mapping.cu $(MAP_FOLDER)reference_calc.h $(MAP_FOLDER)utils.h
	cd $(MKFILE_PATH)/$(MAP_FOLDER) \
		$(GPP) -c reference_calc.h \
		&& $(NVCC) -c compare_results.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -c par_mapping.cu -arch=$(ARCH_SM) \
		&& $(NVCC) -o test reference_calc.o compare_results.o par_mapping.o
		&& ./test \
		&& cd $(MKFILE_PATH)
