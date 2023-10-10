#include <iostream>
#include <iterator>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>

#include "cuda_util.cuh"
//#include "float3.cuh"
#include "vecMath.cuh"
#include "Light.cuh"

const int WIDTH = 8192;
const int HEIGHT = 8192;
#define OBJ_COUNT 19
//const int OBJ_COUNT = 19;
//#define OBJ_COUNT sizeof(spheres) / sizeof(Sphere)

//const int MAX_THREADS_PER_BLOCK = 1024;
const int TPB = 16;

//CPU Timer
auto start = std::chrono::high_resolution_clock::now();
auto elapsed = std::chrono::high_resolution_clock::now() - start;

void cputimer_start(){
    start = std::chrono::high_resolution_clock::now();
}
long cputimer_stop(const char* info){
    elapsed = std::chrono::high_resolution_clock::now() - start;
    long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << "Timing - " << info << ". \t\tElapsed " << microseconds << " microseconds" << std::endl;
    return microseconds;
}

using Color = float3;
using namespace std::chrono;

struct Sphere {
    float radius;
    float3 center;
    Color color;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(float r, const float3 &c, const float3 &col) : radius(r), center(c), color(col) {}
    __host__ __device__ float3 get_normal_at(const float3 &at) const;
};

struct Ray {
    float3 origin;
    float3 dir;

    __host__ __device__ Ray(const float3 &o, const float3 &d) : origin(o), dir(d) {}

    __host__ __device__ float3 at(float t) const;
    __host__ __device__ float has_intersection(const Sphere &sphere) const;
};

__host__ __device__ float3 Sphere::get_normal_at(const float3& at) const {
    return normalizeVec3((float3)(at - center));
}

__host__ __device__ float3 Ray::at(float t) const {
    return origin + (t * dir); 
}

__host__ __device__ float Ray::has_intersection(const Sphere& sphere) const {
    float a = dot(dir, dir);
    float b = dot((2.0f * (dir)), (origin - sphere.center));
    float c = dot((origin - sphere.center), (origin - sphere.center)) - pow(sphere.radius, 2);

    float d = b*b - 4 * (a * c);
    if(d < 0) return -1.0;

    float t0 = ((-b - std::sqrt(d)) / (2*a));
    float t1 = ((-b + std::sqrt(d)) / (2*a));
    if(d == 0) return t0;
    if(t0 < 0 && t1 < 0) return -1;
    if(t0 > 0 && t1 < 0) return t0;
    if(t0 < 0 && t1 > 0) return t1;
    return fminf(t0,t1);
    //return t0 < t1 ? t0 : t1;
}

__device__ Color convert_to_color(const float3 &v) {
    return make_float3(static_cast<int>((v.x) * 255.999), static_cast<int>((v.y) * 255.999), static_cast<int>((v.z) * 255.999));
}

__device__ int get_closest_intersection(Sphere* spheres, const Ray &r, float* intersections) {
    #if OBJ_COUNT == 0
        return -1;
    #elif OBJ_COUNT == 1
        intersections[0] = r.has_intersection(spheres[ii]);
        return intersections[0] < 0 ? -1 : 0;
    #else
        int hp = -1;
        float min_val = 100.0;
        for(int ii = 0; ii < OBJ_COUNT; ++ii) {
            intersections[ii] = r.has_intersection(spheres[ii]);
            if (intersections[ii] < 0.0) continue;
            else if (intersections[ii] < min_val) {
                min_val = intersections[ii];
                hp = ii;
            }
        }
        return hp;
    #endif
}

__device__ Color get_color_at(const Ray &r, float intersection, Light* light, const Sphere &sphere, Sphere* spheres, float3* origin) {
    const float offset_surface = 0.001f;
    float shadow = 1;

    float3 normal = sphere.get_normal_at(r.at(intersection));

    float3 to_camera(*origin - r.at(intersection));
    to_camera = normalizeVec3(to_camera);

    float3 light_ray(light->get_position() - r.at(intersection));
    light_ray = normalizeVec3(light_ray);

    float3 reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
    reflection_ray = normalizeVec3(reflection_ray);

    Ray rr(r.at(intersection) + offset_surface * normal, reflection_ray);
    float intersections[OBJ_COUNT];
    int hp = get_closest_intersection(spheres, rr, intersections);
    bool reflect = false;
    float reflect_shadow = 1;
    if (hp != -1) {
        reflect = true;
        Ray rs(rr.at(intersections[hp]) + offset_surface * spheres[hp].get_normal_at(rr.at(intersections[hp])), light->get_position() - rr.at(intersections[hp]) + offset_surface * spheres[hp].get_normal_at(rr.at(intersections[hp])));
        for (int i = 0; i < OBJ_COUNT; ++i) {
            if (rs.has_intersection(spheres[i]) > 0.000001f) reflect_shadow = 0.35;
        }
    }

    auto ambient = light->get_ambient() * light->get_color(); 
    auto diffuse = (light->get_diffuse() * fmaxf(dot(light_ray, normal), 0.0f)) * light->get_color();
    auto specular = light->get_specular() * pow(fmaxf(dot(reflection_ray, to_camera), 0.0f), 32) * light->get_color();

    Ray shadow_ray(r.at(intersection) + (offset_surface * normal), light->get_position() - (r.at(intersection) + offset_surface * normal));
    for (int i = 0; i < OBJ_COUNT; ++i) {
        if (shadow_ray.has_intersection(spheres[i]) > 0.000001f) shadow = 0.35;
    }

    auto all_light = reflect ? capVec3 ((ambient + diffuse + specular),1) & (0.55 * (sphere.color - (reflect_shadow * spheres[hp].color))) + capVec3((reflect_shadow * spheres[hp].color),1)
                             : capVec3((ambient + diffuse + specular),1) & sphere.color;
    return convert_to_color(shadow * all_light);
}

__global__ void cast_ray(float3* fb, Sphere* spheres, Light *light, float3 *origin) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(i >= WIDTH || j >= HEIGHT) return;
    int tid = (j*WIDTH) + i;

    float3 ij = make_float3(2 * (float((i) + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float((j) + 0.5) / (HEIGHT - 1)), -1);
    float3 dir = ij - *origin;
    Ray r(*origin, dir);

    float intersections[OBJ_COUNT];
    int hp = get_closest_intersection(spheres, r, intersections);

    if(hp == -1) {
        fb[tid] = make_float3(94, 156, 255);
    } else {
        auto color = get_color_at(r, intersections[hp], light, spheres[hp], spheres, origin);
        fb[tid] = color;
    }
}

void initDevice(int& device_handle) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printDeviceProps(devProp);
  
    cudaSetDevice(device_handle);
}

//void run_kernel(const int size, float3* fb, Sphere* spheres, Light* light, float3* origin) {
void run_kernel(const int size, float3* fb, Sphere* spheres, Light* light, float3* origin) {
    float3* fb_device = nullptr;
    Sphere* spheres_dv = nullptr;
    Light* light_dv = nullptr;
    float3* origin_dv = nullptr;

    std::cout << "Size is: " << sizeof(float3) * size << std::endl;

    cputimer_start();

    checkErrorsCuda(cudaMalloc((void**) &fb_device, sizeof(float3) * size));
    checkErrorsCuda(cudaMalloc((void**) &spheres_dv, sizeof(Sphere) * OBJ_COUNT));
    checkErrorsCuda(cudaMalloc((void**) &light_dv, sizeof(Light) * 1));
    checkErrorsCuda(cudaMalloc((void**) &origin_dv, sizeof(float3) * 1));

    cputimer_stop("CUDA Memory Allocation");
    cputimer_start();

    //checkErrorsCuda(cudaMemset(fb_device,0,sizeof(float3) * size));
    //checkErrorsCuda(cudaMemcpy((void*) fb_device, fb, sizeof(float3) * size, cudaMemcpyHostToDevice));
    checkErrorsCuda(cudaMemcpy((void*) spheres_dv, spheres, sizeof(Sphere) * OBJ_COUNT, cudaMemcpyHostToDevice));
    checkErrorsCuda(cudaMemcpy((void*) light_dv, light, sizeof(Light) * 1, cudaMemcpyHostToDevice));
    checkErrorsCuda(cudaMemcpy((void*) origin_dv, origin, sizeof(float3) * 1, cudaMemcpyHostToDevice));

    cputimer_stop("CUDA HtoD memory transfer");

    //cputimer_start();

    //cputimer_stop("CUDA HtoD memory transfer");

    /*
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    */
    cputimer_start();

    dim3 blocks(WIDTH / TPB, HEIGHT / TPB);
    cast_ray<<<blocks, dim3(TPB, TPB)>>>(fb_device, spheres_dv, light_dv, origin_dv);
    cudaDeviceSynchronize();
    cputimer_stop("CUDA Kernal Launch Runtime");

    /*
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(">> time for kernel: %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */
    cputimer_start();

    checkErrorsCuda(cudaMemcpy(fb, fb_device, sizeof(float3) * size, cudaMemcpyDeviceToHost));
    
    cputimer_stop("CUDA DtoH memory transfer");
    checkErrorsCuda(cudaFree(fb_device));
    checkErrorsCuda(cudaFree(spheres_dv));
    checkErrorsCuda(cudaFree(light_dv));
    checkErrorsCuda(cudaFree(origin_dv));
}

int main(int, char**) {
    std::ofstream file("img.ppm");

    const int n = WIDTH * HEIGHT;
    int device_handle = 0;

    //float3* frame_buffer = new float3[n];
    float3* frame_buffer;
    cudaMallocHost((void**)&frame_buffer, sizeof(float3) * n,cudaHostAllocDefault);
    //std::vector<std::string> mem_buffer;

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
    float3 *origin = new float3();
    *origin = make_float3(0, 0, 1);

    
    Light *light = new Light(make_float3(1, 1, 1), make_float3(1, 1, 1));
    light->set_light(.2, .5, .5);
    
    /*
    Light light = Light(make_float3(1, 1, 1), make_float3(1, 1, 1));
    light.set_light(.2, .5, .5);
    */


    std::cout << "===========================================" << std::endl;
    std::cout << ">> Starting kernel for " << WIDTH << "x" << HEIGHT << " image..." << std::endl;
    
    
    run_kernel(n, frame_buffer, spheres, light, origin);
    
    
    std::cout << ">> Finished kernel" << std::endl;

    auto start = steady_clock::now();
    std::cout << ">> Saving Image..." << std::endl;

    file << "P3" << "\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
    for (std::size_t i = 0; i < n; ++i) {
        file << (int) frame_buffer[i].x << " " << (int) frame_buffer[i].y << " " << (int) frame_buffer[i].z << "\n";
        /*
        mem_buffer.push_back(std::to_string((int) frame_buffer[i].x) + " " + 
                             std::to_string((int) frame_buffer[i].y) + " " + 
                             std::to_string((int) frame_buffer[i].z));
        */
    }
    //std::ostream_iterator<std::string> output_iterator(file, "\n");
    //std::copy(mem_buffer.begin(), mem_buffer.end(), output_iterator);

    auto end = duration_cast<milliseconds>(steady_clock::now() - start).count();
    std::cout << ">> Finished writing to file in " << end << " ms" << std::endl;
    std::cout << "===========================================" << std::endl;

    //delete[] frame_buffer;
    cudaFreeHost(frame_buffer);
    delete origin;
    delete light;
    delete[] spheres;

    return EXIT_SUCCESS;
}
