#pragma once

#include "sphere.cuh"

struct Ray {
    float3 origin;
    float3 dir;

    __host__ __device__ Ray(const float3 &o, const float3 &d) : origin(o), dir(d) {}

    __host__ __device__ float3 at(float t) const;
    __host__ __device__ float has_intersection(const Sphere &sphere) const;
};


__host__ __device__ float3 Ray::at(float t) const {
    return origin + (t * dir); 
}

__host__ __device__ float Ray::has_intersection(const Sphere& sphere) const {
    const float a = dot(dir, dir);
    const float b = dot((2.0f * dir), (origin - sphere.center));
    const float c = dot((origin - sphere.center), (origin - sphere.center)) - sphere.radius*sphere.radius;

    const float d = b*b - 4 * (a * c);
    
    //const bool d_is_neg = d < 0;

    if(d < 0){
        return -1.0;
    }

    const float t0 = ((-b - std::sqrt(d)) / (2*a));
    const float t1 = ((-b + std::sqrt(d)) / (2*a));

    const bool t0_is_neg = t0 < 0;
    const bool t1_is_neg = t1 < 0;

    //Optimzed to stop warp control divergance on SIMD
    //return ((d_is_neg*-1) + !d_is_neg*((t0_is_neg*t1_is_neg * -1) + (t0_is_neg*!t1_is_neg * t1) + (!t0_is_neg*t1_is_neg *t0) + (!t0_is_neg*!t1_is_neg *fminf(t0,t1))));
    
    return ((t0_is_neg*t1_is_neg * -1) + (t0_is_neg*!t1_is_neg * t1) + (!t0_is_neg*t1_is_neg *t0) + (!t0_is_neg*!t1_is_neg *fminf(t0,t1)));

    /*
    if(d == 0 || t0 > 0 && t1 < 0) {
        return t0;
    }
    
    if(t0 < 0 && t1 < 0) {
        return -1;
    }

    if(t0 < 0 && t1 > 0) {
        return t1;
    }
    return fminf(t0,t1);
    //return t0 < t1 ? t0 : t1;
    */
    
}