#include <cmath>

__host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
  //return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__host__ __device__ float3 operator/(const float3 &a,const float &b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__host__ __device__ float3 operator&(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float sizeOfVec3(const float3 &a){
    return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

/*
__host__ __device__ Vec3f Vec3f::scale(float factor) const {
    return Vec3f(factor * mx, factor * my, factor * mz);
}
*/

__host__ __device__ float3 normalizeVec3(const float3 &a) {
    return (a / sizeOfVec3(a));;
}

__host__ __device__ float3 capVec3(const float3 &a,const float max) {
    float nx = a.x;
    float ny = a.y;
    float nz = a.z;
    if(a.x > max) nx = max;
    if(a.y > max) ny = max;
    if(a.z > max) nz = max;
    return make_float3(nx, ny, nz);
}

// free functions

__host__ __device__ float3 operator*(const float a, const float3 &b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ float3 operator*(const float3 &b, float a) {
    return a * b;
}
/*
__host__ __device__ float3 crossVec3(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, 
                    a.z * b.x - a.x * b.z, 
                    a.x * b.y - a.y * b.x);
}
*/
__host__ std::ostream& operator<<(std::ostream& os, const float3 &a) {
    os << "(" << a.x << " " << a.y << " " << a.z << ")";
    return os;
}

__host__ __device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
