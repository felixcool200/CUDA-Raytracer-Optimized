#pragma once
#include "sphere.cuh"

struct Sphere {
    float radius;
    float3 center;
    Color color;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(float r, const float3 &c, const float3 &col) : radius(r), center(c), color(col) {}
    __host__ __device__ float3 get_normal_at(const float3 &at) const;
};

__host__ __device__ float3 Sphere::get_normal_at(const float3& at) const {
    return normalizeVec3((float3)(at - center));
}