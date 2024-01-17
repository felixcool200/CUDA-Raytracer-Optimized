ARG CUDA_VERSION=12.2.2
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu22.04

MAINTAINER Felix Söderman <felixsoderman@gmail.com>

WORKDIR /app

# Set the build argument
ARG CUDA_ARCH=sm_61
COPY . .

# Use the build argument in the nvcc command
RUN nvcc --default-stream per-thread -arch=$CUDA_ARCH -o /app/gpu.out /app/src/main.cu

CMD ["/app/gpu.out"]
