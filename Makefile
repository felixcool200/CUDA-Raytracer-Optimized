run:
	nvcc --default-stream per-thread -arch=sm_61 src/main.cu -o gpu.out
gprof: 
	nvcc --profile --default-stream per-thread -arch=sm_61 src/main.cu -o gpu.out 
profiling: 
	nvcc --generate-line-info --default-stream per-thread -arch=sm_61 src/main.cu -o gpu.out 
clean:
	rm -f gpu.out /tmp/img.ppm img.ppm gmon.out
