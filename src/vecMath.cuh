#pragma once

#include <cmath>
#include <ostream>

__host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(const float &a, const float3 &b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ float3 operator*(const float3 &b, const float &a) {
  return a * b;
}

__host__ __device__ float3 operator/(const float3 &a, const float &b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

// Element wise multiplication
__host__ __device__ float3 operator&(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float length_float3(const float3 &a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ float3 normalize_float3(const float3 &a) {
  return (a / length_float3(a));
}

// Sets the components of the vector to be within a maximum value
__host__ __device__ float3 cap_float3(const float3 &a, const float &max) {
  float nx = a.x;
  float ny = a.y;
  float nz = a.z;

  if (a.x > max) {
    nx = max;
  }

  if (a.y > max) {
    ny = max;
  }

  if (a.z > max) {
    nz = max;
  }

  return make_float3(nx, ny, nz);
}

__host__ __device__ float dot_float3(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross_float3(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ std::ostream& operator<<(std::ostream& os, const float3 &a) {
  os << "(" << a.x << " " << a.y << " " << a.z << ")";
  return os;
}