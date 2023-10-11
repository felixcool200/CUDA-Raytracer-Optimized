#include <iostream>
#include <iterator>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>

#include "consts.cuh"
#include "cuda_util.cuh"
#include "vecMath.cuh"
#include "light.cuh"
#include "sphere.cuh"
#include "ray.cuh"

//#define tidGLOBAL ((blockIdx.y * blockDim.y) + threadIdx.y*WIDTH) + (blockIdx.x * blockDim.x) + threadIdx.x

//const int MAX_THREADS_PER_BLOCK = 1024;
const int TPB = 32;

//CPU Timer
auto start_CPU = std::chrono::high_resolution_clock::now();
auto elapsed = std::chrono::high_resolution_clock::now() - start_CPU;

void cputimer_start(){
    start_CPU = std::chrono::high_resolution_clock::now();
}
long cputimer_stop(const char* info){
    elapsed = std::chrono::high_resolution_clock::now() - start_CPU;
    long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << "Timing - " << info << ". \t\tElapsed " << microseconds << " microseconds" << std::endl;
    return microseconds;
}

__device__ Color convert_to_color(const float3 &v) {
    return make_float3(v.x * 255.999, (v.y * 255.999), (v.z * 255.999));
}

__device__ int get_closest_intersection(Sphere* spheres, const Ray &r, float* intersections) {

    switch (OBJ_COUNT)
    {
    case 0:
        return -1;
    case 1:
        *intersections = r.has_intersection(spheres[0]);
        return *intersections < 0 ? -1 : 0;
    
    default:
        int hp = -1;
        *intersections = 100.0;
        float tmp_intersection;
        for(int ii = 0; ii < OBJ_COUNT; ++ii) {
            //intersections[ii] = r.has_intersection(spheres[ii]);
            tmp_intersection = r.has_intersection(spheres[ii]);
            if (tmp_intersection < 0.0) {
                continue;
            }
            else if (tmp_intersection < *intersections) {
                *intersections = tmp_intersection;
                hp = ii;
            }
        }
        return hp;
    }
}

__device__ Color get_color_at(const Ray &r, float intersection, Light light, const Sphere &sphere, Sphere* spheres, float3 origin) {
    const float offset_surface = 0.001;
    float shadow = 1.0;

    float3 normal = sphere.get_normal_at(r.at(intersection));

    float3 to_camera(origin - r.at(intersection));
    to_camera = normalizeVec3(to_camera);

    float3 light_ray(light.get_position() - r.at(intersection));
    light_ray = normalizeVec3(light_ray);

    float3 reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
    reflection_ray = normalizeVec3(reflection_ray);

    Ray rr(r.at(intersection) + offset_surface * normal, reflection_ray);
    //float intersections[OBJ_COUNT];
    float closest_intersection;
    int hp = get_closest_intersection(spheres, rr, &closest_intersection);
    bool reflect = false;
    float reflect_shadow = 1;

    if (hp != -1) {
        reflect = true;
        Ray rs(rr.at(closest_intersection) + offset_surface * spheres[hp].get_normal_at(rr.at(closest_intersection)), light.get_position() - rr.at(closest_intersection) + offset_surface * spheres[hp].get_normal_at(rr.at(closest_intersection)));
        for (int i = 0; i < OBJ_COUNT; ++i) {
            if (rs.has_intersection(spheres[i])  > 0.000001f) {
                reflect_shadow = 0.35;
                break;
            }
        }
    }

    float3 ambientDiffuseSpecular = light.get_ambient() * light.get_color(); 
    ambientDiffuseSpecular = ((light.get_diffuse() * fmaxf(dot(light_ray, normal), 0.0f)) * light.get_color()) + ambientDiffuseSpecular;
    ambientDiffuseSpecular = (light.get_specular() * pow(fmaxf(dot(reflection_ray, to_camera), 0.0f), 32) * light.get_color()) + ambientDiffuseSpecular;
    
    Ray shadow_ray(r.at(intersection) + (offset_surface * normal), light.get_position() - (r.at(intersection) + offset_surface * normal));
    for (int i = 0; i < OBJ_COUNT; ++i) {
        if (shadow_ray.has_intersection(spheres[i]) > 0.000001f){
            shadow = 0.35;
            break;
        }
    }
    auto all_light = reflect ? capVec3 ((ambientDiffuseSpecular),1) & (0.55 * (sphere.color - (reflect_shadow * spheres[hp].color))) + capVec3((reflect_shadow * spheres[hp].color),1)
                             : capVec3((ambientDiffuseSpecular),1) & sphere.color;
    return convert_to_color(shadow * all_light);
}

__global__ void cast_ray(float3* fb, Sphere* spheres, Light light, float3 origin) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(i >= WIDTH || j >= HEIGHT) return;
    const int tid = (j*WIDTH) + i;

    const float3 ij = make_float3(2 * (float((i) + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float((j) + 0.5) / (HEIGHT - 1)), -1);
    Ray r(origin, ij - origin);

    //float intersections[OBJ_COUNT];
    float closest_intersection;
    int hp = get_closest_intersection(spheres, r, &closest_intersection);

    if(hp == -1) {
        fb[tid] = make_float3(94, 156, 255);
    } else {
        fb[tid] = get_color_at(r, closest_intersection, light, spheres[hp], spheres, origin);
    }
}

void initDevice(int& device_handle) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printDeviceProps(devProp);
  
    cudaSetDevice(device_handle);
}

//void run_kernel(const int size, float3* fb, Sphere* spheres, Light* light, float3* origin) {
void run_kernel(const int size, float3* fb, Sphere* spheres, Light light, float3 origin) {
    float3* fb_device = nullptr;
    Sphere* spheres_dv = nullptr;

    std::cout << "Size is: " << sizeof(float3) * size << std::endl;

    cputimer_start();
    checkErrorsCuda(cudaMalloc((void**) &fb_device, sizeof(float3) * size));
    checkErrorsCuda(cudaMalloc((void**) &spheres_dv, sizeof(Sphere) * OBJ_COUNT));
    cputimer_stop("CUDA Memory Allocation");
    
    cputimer_start();
    checkErrorsCuda(cudaMemcpy((void*) spheres_dv, spheres, sizeof(Sphere) * OBJ_COUNT, cudaMemcpyHostToDevice));
    cputimer_stop("CUDA HtoD memory transfer");

    cputimer_start();
    dim3 blocks(WIDTH / TPB, HEIGHT / TPB);
    cast_ray<<<blocks, dim3(TPB, TPB)>>>(fb_device, spheres_dv, light, origin);
    cudaDeviceSynchronize();
    cputimer_stop("CUDA Kernal Launch Runtime");

    cputimer_start();

    checkErrorsCuda(cudaMemcpy(fb, fb_device, sizeof(float3) * size, cudaMemcpyDeviceToHost));
    
    cputimer_stop("CUDA DtoH memory transfer");
    cputimer_start();
    checkErrorsCuda(cudaFree(fb_device));
    checkErrorsCuda(cudaFree(spheres_dv));
    cputimer_stop("CUDA Free");
}

int main(int, char**) {
    std::ofstream file("/tmp/img.ppm");

    const int n = WIDTH * HEIGHT;
    int device_handle = 0;

    float3* frame_buffer;
    cudaMallocHost((void**)&frame_buffer, sizeof(float3) * n,cudaHostAllocDefault);

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        std::cerr << "initDevice(): No CUDA Device found." << std::endl;
        return EXIT_FAILURE;
    }
    initDevice(device_handle);

    // Create an array of spheres
    Sphere *spheres = new Sphere[OBJ_COUNT] {
        Sphere(1000, make_float3(0, -1002, 0), make_float3(0.5, 0.5, 0.5)),
        Sphere(0.25, make_float3(-1.5, -0.25, -4), make_float3(1.0, 0.0, 0.0)),
        Sphere(0.25, make_float3(-1.0, -0.25, -4), make_float3(1.0, 0.5, 0.0)),
        Sphere(0.25, make_float3(-0.5, -0.25, -4), make_float3(1.0, 1.0, 0.0)),
        Sphere(0.25, make_float3(0, -0.25, -4), make_float3(0.0, 1.0, 0.0)),
        Sphere(0.25, make_float3(0.5, -0.25, -4), make_float3(0.0, 1.0, 1.0)),
        Sphere(0.25, make_float3(1.0, -0.25, -4), make_float3(0.0, 0.0, 1.0)),

        Sphere(0.25, make_float3(1.5, -0.25, -4), make_float3(0.5, 0.0, 1.0)),
        Sphere(0.25, make_float3(-1.25, 0.25, -3), make_float3(1.0, 0.0, 0.5)),
        Sphere(0.25, make_float3(-0.75, 0.25, -3), make_float3(0.5, 0.0, 0.5)),
        Sphere(0.25, make_float3(-0.25, 0.25, -3), make_float3(0.5, 0.5, 0.5)),
        Sphere(0.25, make_float3(0.25, 0.25, -3), make_float3(1.0, 1.0, 0.5)),
        Sphere(0.25, make_float3(0.75, 0.25, -3), make_float3(0.0, 1.0, 0.5)),

        Sphere(0.25, make_float3(1.25, 0.25, -3), make_float3(0.0, 0.5, 1.0)),
        Sphere(0.25, make_float3(-1.0, 0.75, -2), make_float3(1.0, 0.5, 0.0)),
        Sphere(0.25, make_float3(-0.5, 0.75, -2), make_float3(0.0, 1.0, 1.0)),
        Sphere(0.25, make_float3(0, 0.75, -2), make_float3(0.5, 0.0, 1.0)),
        Sphere(0.25, make_float3(0.5, 0.75, -2), make_float3(0.0, 0.5, 0.0)),
        Sphere(0.25, make_float3(1.0, 0.75, -2), make_float3(1.0, 1.0, 1.0)),

        //Sphere(0.25, make_float3(1.5, 0.75, -2), make_float3(0.0, 0.0, 0.0)),
    };
    
    float3 origin;
    origin = make_float3(0, 0, 1); //Standard
    //origin = make_float3(0, 0, 0); // Nice looking with reflections in the ground

    Light light = Light(make_float3(1, 1, 1), make_float3(1, 1, 1));
    light.set_light(.2, .5, .5);

    std::cout << "===========================================" << std::endl;
    std::cout << ">> Starting kernel for " << WIDTH << "x" << HEIGHT << " image..." << std::endl;
    
    
    run_kernel(n, frame_buffer, spheres, light, origin);
    
    
    std::cout << ">> Finished kernel" << std::endl;

    auto start = std::chrono::steady_clock::now();
    std::cout << ">> Saving Image..." << std::endl;

    file << "P3" << "\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
    for (std::size_t i = 0; i < n; ++i) {
        file << static_cast<int>(frame_buffer[i].x) << " " << static_cast<int>(frame_buffer[i].y) << " " << static_cast<int>(frame_buffer[i].z) << "\n";
    }
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << ">> Finished writing to file in " << end << " ms" << std::endl;
    std::cout << "===========================================" << std::endl;

    //delete[] frame_buffer;
    cudaFreeHost(frame_buffer);
    /*
    delete origin;
    delete light;
    */
    delete[] spheres;

    return EXIT_SUCCESS;
}
