sudo docker build --build-arg ARCH=sm_61 --build-arg CUDAV=12.2.2 -t raytracer . && sudo docker run --rm --gpus all raytracer
