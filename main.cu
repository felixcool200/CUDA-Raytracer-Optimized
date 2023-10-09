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
const int OBJ_COUNT = 19;
//#define OBJ_COUNT sizeof(spheres) / sizeof(Sphere)

//const int MAX_THREADS_PER_BLOCK = 1024;
const int TPB = 16;

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
    auto a = dot(dir, dir);
    auto b = dot((2.0f * (dir)), (origin - sphere.center));
    auto c = dot((origin - sphere.center), (origin - sphere.center)) - pow(sphere.radius, 2);

    auto d = b*b - 4 * (a * c);
    if(d < 0) return -1.0;

    float t0 = ((-b - std::sqrt(d)) / (2*a));
    float t1 = ((-b + std::sqrt(d)) / (2*a));
    if(d == 0) return t0;
    if(t0 < 0 && t1 < 0) return -1;
    if(t0 > 0 && t1 < 0) return t0;
    if(t0 < 0 && t1 > 0) return t1;
    return t0 < t1 ? t0 : t1;
}

__device__ constexpr float f_max(float a, float b) {
    return a > b ? a : b;
}

__device__ Color convert_to_color(const float3 &v) {
    return make_float3(static_cast<int>(1 * ((v.x) * 255.999)), static_cast<int>(1 * ((v.y) * 255.999)), static_cast<int>(1 * ((v.z) * 255.999)));
}

__device__ int get_closest_intersection(Sphere* spheres, const Ray &r, float* intersections) {
    int hp = -1;
    for(int ii = 0; ii < OBJ_COUNT; ii++) {
        intersections[ii] = r.has_intersection(spheres[ii]);
    }

    int asize = OBJ_COUNT;
    if(asize == 1) {
        hp = intersections[0] < 0 ? -1 : 0;
    } else {
        if(asize != 0) {
            float min_val = 100.0;
            for (int ii = 0; ii < asize; ii++) {
                if (intersections[ii] < 0.0) continue;
                else if (intersections[ii] < min_val) {
                    min_val = intersections[ii];
                    hp = ii;
                }
            }
        }
    }
    return hp;
}

__device__ Color get_color_at(const Ray &r, float intersection, Light* light, const Sphere &sphere, Sphere* spheres, float3* origin) {
    float shadow = 1;

    float3 normal = sphere.get_normal_at(r.at(intersection));

    float3 to_camera(*origin - r.at(intersection));
    to_camera = normalizeVec3(to_camera);

    float3 light_ray(light->get_position() - r.at(intersection));
    light_ray = normalizeVec3(light_ray);

    float3 reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
    reflection_ray = normalizeVec3(reflection_ray);

    Ray rr(r.at(intersection) + 0.001 * normal, reflection_ray);
    float intersections[OBJ_COUNT];
    int hp = get_closest_intersection(spheres, rr, intersections);
    bool reflect = false;
    float reflect_shadow = 1;
    if (hp != -1) {
        reflect = true;
        Ray rs(rr.at(intersections[hp]) + 0.001 * spheres[hp].get_normal_at(rr.at(intersections[hp])), light->get_position() - rr.at(intersections[hp]) + 0.001 * spheres[hp].get_normal_at(rr.at(intersections[hp])));
        for (int i = 0; i < OBJ_COUNT; ++i) {
            if (rs.has_intersection(spheres[i]) > 0.000001f) reflect_shadow = 0.35;
        }
    }

    auto ambient = light->get_ambient() * light->get_color(); 
    auto diffuse = (light->get_diffuse() * f_max(dot(light_ray, normal), 0.0f)) * light->get_color();
    auto specular = light->get_specular() * pow(f_max(dot(reflection_ray, to_camera), 0.0f), 32) * light->get_color();

    Ray shadow_ray(r.at(intersection) + (0.001f * normal), light->get_position() - (r.at(intersection) + 0.001f * normal));
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

    int tid = (j*WIDTH) + i;
    if(i >= WIDTH || j >= HEIGHT) return;

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

    //printf("Sizes: fb:%d, Sphere:%d, Light:%d, origin: %d\n",sizeof(float3) * size, sizeof(Sphere) * OBJ_COUNT,sizeof(Light),sizeof(float3));

    checkErrorsCuda(cudaMalloc((void**) &fb_device, sizeof(float3) * size));
    checkErrorsCuda(cudaMemcpy((void*) fb_device, fb, sizeof(float3) * size, cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**) &spheres_dv, sizeof(Sphere) * OBJ_COUNT));
    checkErrorsCuda(cudaMemcpy((void*) spheres_dv, spheres, sizeof(Sphere) * OBJ_COUNT, cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**) &light_dv, sizeof(Light) * 1));
    checkErrorsCuda(cudaMemcpy((void*) light_dv, light, sizeof(Light) * 1, cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**) &origin_dv, sizeof(float3) * 1));
    checkErrorsCuda(cudaMemcpy((void*) origin_dv, origin, sizeof(float3) * 1, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 blocks(WIDTH / TPB, HEIGHT / TPB);
    cast_ray<<<blocks, dim3(TPB, TPB)>>>(fb_device, spheres_dv, light_dv, origin_dv);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(">> time for kernel: %f ms\n", time);

    checkErrorsCuda(cudaMemcpy(fb, fb_device, sizeof(float3) * size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkErrorsCuda(cudaFree(fb_device));
    checkErrorsCuda(cudaFree(spheres_dv));
    //checkErrorsCuda(cudaFree(light_dv));
    //checkErrorsCuda(cudaFree(origin_dv));
}

int main(int, char**) {
    std::ofstream file("img.ppm");

    const int n = WIDTH * HEIGHT;
    int device_handle = 0;

    float3* frame_buffer = new float3[n];
    std::vector<std::string> mem_buffer;

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
        mem_buffer.push_back(std::to_string((int) frame_buffer[i].x) + " " + 
                             std::to_string((int) frame_buffer[i].y) + " " + 
                             std::to_string((int) frame_buffer[i].z));
    }
    std::ostream_iterator<std::string> output_iterator(file, "\n");
    std::copy(mem_buffer.begin(), mem_buffer.end(), output_iterator);

    auto end = duration_cast<milliseconds>(steady_clock::now() - start).count();
    std::cout << ">> Finished writing to file in " << end << " ms" << std::endl;
    std::cout << "===========================================" << std::endl;

    delete[] frame_buffer;
    delete origin;
    delete light;
    delete[] spheres;

    return EXIT_SUCCESS;
}