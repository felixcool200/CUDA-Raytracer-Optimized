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

// CPU Timer
auto start_CPU = std::chrono::high_resolution_clock::now();

void start_CPU_timer(){
    start_CPU = std::chrono::high_resolution_clock::now();
}

long stop_CPU_timer(const char* info){
    auto elapsed = std::chrono::high_resolution_clock::now() - start_CPU;
    long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << microseconds << " microseconds\t\t" << info << std::endl;
    return microseconds;
}

// Find the closest intersecting sphere of a ray if it exists and set the closest intersection of the sphere if it exists
__device__ int get_closest_intersection(Sphere* spheres, const Ray &r, float* closest_intersection) {
  int hp = -1;
  *closest_intersection = 100.0;
  float current_intersection;

  for (int ii = 0; ii < OBJ_COUNT; ++ii) {
    current_intersection = r.has_intersection(spheres[ii]);

    // Intersection exists and is in front of the ray and the current intersection is closer than the previous one
    if (current_intersection >= 0.0 && current_intersection < *closest_intersection) {
      *closest_intersection = current_intersection;
      hp = ii;
    }
  }

  return hp;
}

// Calculate the color to display at the intersection between ray and sphere
__device__ Color get_color_at(const Ray &r, float intersection, Light* light, const Sphere &sphere, Sphere* spheres, float3* origin) {
  const float surface_offset = 0.001;
  const float shadow_threshold = 0.000001;
  float shadow = 1; // Initialize shadow to full brightness

  float3 normal = sphere.get_normal_at(r.at(intersection));

  // Normalized vector from intersection point to camera
  float3 to_camera = *origin - r.at(intersection);
  to_camera = normalize_float3(to_camera);

  // Normalized vector from intersection point to light source
  float3 light_ray = light->get_position() - r.at(intersection);
  light_ray = normalize_float3(light_ray);

  // Normalized vector of the reflected ray when hitting a sphere
  float3 reflection_ray = (-1 * light_ray) - 2 * dot_float3((-1 * light_ray), normal) * normal;
  reflection_ray = normalize_float3(reflection_ray);

  // Reflection ray from intersection point
  Ray rr(r.at(intersection) + surface_offset * normal, reflection_ray); // Offset ray from surface so that it does not hit the same surface it just reflected away from
  float reflect_closest_intersection;
  int hp = get_closest_intersection(spheres, rr, &reflect_closest_intersection); // What the reflection ray hits
  float reflect_shadow = 1;
  Color reflect_color = make_float3(BGD_R, BGD_G, BGD_B) / 255; // Set color in reflection to be background color by default

  // Reflection ray hit a sphere
  if (hp != -1) {
    // Ray from intersection point on the sphere that is reflected towards the light source
    Ray rs(rr.at(reflect_closest_intersection) + surface_offset * spheres[hp].get_normal_at(rr.at(reflect_closest_intersection)), light->get_position() - rr.at(reflect_closest_intersection) + surface_offset * spheres[hp].get_normal_at(rr.at(reflect_closest_intersection)));

    // Check if ray from intersection point on the sphere that is reflected towards the light source hits any sphere that creates a shadow
    for (int i = 0; i < OBJ_COUNT; ++i) {
      // There is a a sphere creating a shadow on the reflected sphere
      if (rs.has_intersection(spheres[i]) > shadow_threshold) {
        reflect_shadow = 0.35;
        break;
      }
    }

    reflect_color = reflect_shadow * spheres[hp].color;
  }

  // Calculate ambient, diffuse, and specular components of the light
  float3 ambient_diffuse_specular = light->get_ambient() * light->get_color(); // Ambient
  ambient_diffuse_specular = ambient_diffuse_specular + (light->get_diffuse() * fmaxf(dot_float3(light_ray, normal), 0.0f)) * light->get_color(); // Diffuse
  ambient_diffuse_specular = ambient_diffuse_specular + light->get_specular() * pow(fmaxf(dot_float3(reflection_ray, to_camera), 0.0f), 32) * light->get_color(); // Specular

  // Ray from interesection point on original sphere towards the light source
  Ray shadow_ray(r.at(intersection) + (surface_offset * normal), light->get_position() - (r.at(intersection) + surface_offset * normal));

  // Check if ray from intersection point on original sphere towards the light source hits any sphere that creates a shadow
  for (int i = 0; i < OBJ_COUNT; ++i) {
    // There is a sphere creating a shadow on the original sphere
    if (shadow_ray.has_intersection(spheres[i]) > shadow_threshold) {
      shadow = 0.35;
      break;
    }
  }

  // Final color before adding the shadow on the original sphere
  float3 all_light = cap_float3(ambient_diffuse_specular, 1) & cap_float3(0.55 * (sphere.color - reflect_color) + reflect_color, 1);

  return 255.999 * (shadow * all_light); // Converts to color
}

// Cast one ray per pixel
__global__ void cast_ray(float3* fb, Sphere* spheres, Light* light, float3* origin) {
  __shared__ Sphere spheres_shared[OBJ_COUNT];
  
  const int local_tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Transfer spheres to shared memory
  if (local_tid < OBJ_COUNT) {
    spheres_shared[local_tid] = spheres[local_tid];
  }

  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  const int tid = (j*WIDTH) + i;

  // Outside rendered pixels
  if (i >= WIDTH || j >= HEIGHT) {
    return;
  }

  // Calculate the ray from the position of the camera toward the 3D scene through the current pixel on the 2D image plane
  const float3 ij = make_float3(2 * (float((i) + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float((j) + 0.5) / (HEIGHT - 1)), -1); // Direction vector for the ray
  const float3 dir = ij - *origin;
  Ray r(*origin, dir);

  __syncthreads(); // Sync threads in block before spheres_shared is accessed

  float closest_intersection; // The closest intersection of the closest intersecting sphere
  const int hp = get_closest_intersection(spheres_shared, r, &closest_intersection); // The closest intersecting sphere

  // Did not hit any spheres (background color)
  if (hp == -1) {
    fb[tid] = make_float3(BGD_R, BGD_G, BGD_B); // Color
  }
  // Did hit a sphere
  else {
    fb[tid] = get_color_at(r, closest_intersection, light, spheres_shared[hp], spheres_shared, origin);
  }
}

void initDevice(int& device_handle) {
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printDeviceProps(devProp);

  cudaSetDevice(device_handle);
}

void run_kernel(const int pixels, float3* fb, Sphere* spheres, Light* light, float3* origin) {
  // Device
  float3* fb_device = nullptr;
  Sphere* spheres_dv = nullptr;
  Light* light_dv = nullptr;
  float3* origin_dv = nullptr;

  start_CPU_timer();

  // Device memory allocation
  checkErrorsCuda(cudaMalloc((void**) &fb_device, sizeof(float3) * pixels));
  checkErrorsCuda(cudaMalloc((void**) &spheres_dv, sizeof(Sphere) * OBJ_COUNT));
  checkErrorsCuda(cudaMalloc((void**) &light_dv, sizeof(Light) * 1));
  checkErrorsCuda(cudaMalloc((void**) &origin_dv, sizeof(float3) * 1));

  stop_CPU_timer("Device memory allocation");
  start_CPU_timer();

  // Host to device memory transfer
  checkErrorsCuda(cudaMemcpy((void*) spheres_dv, spheres, sizeof(Sphere) * OBJ_COUNT, cudaMemcpyHostToDevice));
  checkErrorsCuda(cudaMemcpy((void*) light_dv, light, sizeof(Light) * 1, cudaMemcpyHostToDevice));
  checkErrorsCuda(cudaMemcpy((void*) origin_dv, origin, sizeof(float3) * 1, cudaMemcpyHostToDevice));

  stop_CPU_timer("HtoD memory transfer");
  start_CPU_timer();

  // Launch kernel
  dim3 blocks(WIDTH / TPB, HEIGHT / TPB);
  cast_ray<<<blocks, dim3(TPB, TPB)>>>(fb_device, spheres_dv, light_dv, origin_dv);

  cudaDeviceSynchronize();

  stop_CPU_timer("CUDA kernel");
  start_CPU_timer();

  // Device to host memory transfer
  checkErrorsCuda(cudaMemcpy(fb, fb_device, sizeof(float3) * pixels, cudaMemcpyDeviceToHost));

  stop_CPU_timer("DtoH memory transfer");
  start_CPU_timer();

  // Free device memory
  checkErrorsCuda(cudaFree(fb_device));
  checkErrorsCuda(cudaFree(spheres_dv));
  checkErrorsCuda(cudaFree(light_dv));
  checkErrorsCuda(cudaFree(origin_dv));

  stop_CPU_timer("Freeing device memory");
}

int main(int argc, char *argv[]) {
  int write_to_file = true;
  
  if (argc == 2) {
    write_to_file = atoi(argv[1]);
  }

  std::ofstream file("img.ppm");

  const int pixels = WIDTH * HEIGHT;
  int device_handle = 0;

  int deviceCount = 0;
  checkErrorsCuda(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "initDevice(): No CUDA Device found." << std::endl;
    return EXIT_FAILURE;
  }

  initDevice(device_handle);

  std::cout << "===========================================" << std::endl;

  // Host memory allocation
  start_CPU_timer();

  float3* frame_buffer; // Frame buffer for all pixels
  Sphere* spheres;
  float3* origin;
  Light* light;

  checkErrorsCuda(cudaMallocHost((void**) &frame_buffer, sizeof(float3) * pixels, cudaHostAllocDefault));
  checkErrorsCuda(cudaMallocHost((void**) &spheres, sizeof(Sphere) * OBJ_COUNT, cudaHostAllocDefault));
  checkErrorsCuda(cudaMallocHost((void**) &origin, sizeof(float3), cudaHostAllocDefault));
  checkErrorsCuda(cudaMallocHost((void**) &light, sizeof(Light), cudaHostAllocDefault));

  // Create an array of spheres
  spheres[0] = Sphere(1000, make_float3(0, -1002, 0), make_float3(0.5, 0.5, 0.5));

  spheres[1] = Sphere(0.25, make_float3(-1.5, -0.25, -4), make_float3(1.0, 0.0, 0.0));
  spheres[2] = Sphere(0.25, make_float3(-1.0, -0.25, -4), make_float3(1.0, 0.5, 0.0));
  spheres[3] = Sphere(0.25, make_float3(-0.5, -0.25, -4), make_float3(1.0, 1.0, 0.0));
  spheres[4] = Sphere(0.25, make_float3(0, -0.25, -4), make_float3(0.0, 1.0, 0.0));
  spheres[5] = Sphere(0.25, make_float3(0.5, -0.25, -4), make_float3(0.0, 1.0, 1.0));
  spheres[6] = Sphere(0.25, make_float3(1.0, -0.25, -4), make_float3(0.0, 0.0, 1.0));

  spheres[7] = Sphere(0.25, make_float3(1.5, -0.25, -4), make_float3(0.5, 0.0, 1.0));
  spheres[8] = Sphere(0.25, make_float3(-1.25, 0.25, -3), make_float3(1.0, 0.0, 0.5));
  spheres[9] = Sphere(0.25, make_float3(-0.75, 0.25, -3), make_float3(0.5, 0.0, 0.5));
  spheres[10] = Sphere(0.25, make_float3(-0.25, 0.25, -3), make_float3(0.5, 0.5, 0.5));
  spheres[11] = Sphere(0.25, make_float3(0.25, 0.25, -3), make_float3(1.0, 1.0, 0.5));
  spheres[12] = Sphere(0.25, make_float3(0.75, 0.25, -3), make_float3(0.0, 1.0, 0.5));

  spheres[13] = Sphere(0.25, make_float3(1.25, 0.25, -3), make_float3(0.0, 0.5, 1.0));
  spheres[14] = Sphere(0.25, make_float3(-1.0, 0.75, -2), make_float3(1.0, 0.5, 0.0));
  spheres[15] = Sphere(0.25, make_float3(-0.5, 0.75, -2), make_float3(0.0, 1.0, 1.0));
  spheres[16] = Sphere(0.25, make_float3(0, 0.75, -2), make_float3(0.5, 0.0, 1.0));
  spheres[17] = Sphere(0.25, make_float3(0.5, 0.75, -2), make_float3(0.0, 0.5, 0.0));
  spheres[18] = Sphere(0.25, make_float3(1.0, 0.75, -2), make_float3(1.0, 1.0, 1.0));

  // Origin of the camera
  *origin = make_float3(0, 0, 1);

  // Light source in the scene
  *light = Light(make_float3(1, 1, 1), make_float3(1, 1, 1));
  // light->set_light(.2, .5, .5);
  light->set_light(.1, .7, .7);

  stop_CPU_timer("Host memory allocation");

  std::cout << ">> Starting kernel for " << WIDTH << "x" << HEIGHT << " image..." << std::endl;
  run_kernel(pixels, frame_buffer, spheres, light, origin);
  std::cout << ">> Finished kernel" << std::endl;

  std::cout << ">> Saving Image..." << std::endl;

  start_CPU_timer();

  if (write_to_file == 1) {
    // Write from the frame buffer to image file
    file << "P3" << "\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";

    for (std::size_t i = 0; i < pixels; ++i) {
      file << static_cast<int>(frame_buffer[i].x) << " " << static_cast<int>(frame_buffer[i].y) << " " << static_cast<int>(frame_buffer[i].z) << "\n";
    }
  }

  stop_CPU_timer("Writing to file");

  start_CPU_timer();

  // Free host memory
  cudaFreeHost(frame_buffer);
  cudaFreeHost(origin);
  cudaFreeHost(light);
  cudaFreeHost(spheres);

  stop_CPU_timer("Freeing host memory");

  std::cout << "===========================================" << std::endl;

  return EXIT_SUCCESS;
}
