all:
	#nvcc --default-stream per-thread -rdc=true -arch=sm_61 Vec3f.cu main.cu -o gpu.out
	nvcc --default-stream per-thread -arch=sm_61 src/main.cu -o gpu.out
clean:
	rm *.out img.ppm
