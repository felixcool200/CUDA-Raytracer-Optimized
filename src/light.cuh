#pragma once

// Light source within rendered scene
class Light {
  private:
    float ambient; // Ambient intensity of the light
    float diffuse; // Diffuse intensity of the light
    float specular; // Specular intensity of the light

    float3 position;
    float3 color;

  public:
    __host__ __device__ Light(const float3 &position, const float3 &color) : position(position), color(color) {}
    __host__ __device__ float3 get_position() const {
      return position;
    }
    __host__ __device__ float3 get_color() const {
      return color;
    }

    __host__ __device__ void set_light(float amb, float diff, float spec) {
      ambient = amb;
      diffuse = diff;
      specular = spec;
    }

    __host__ __device__ float get_ambient() const {
      return ambient;
    }
    __host__ __device__ float get_diffuse() const {
      return diffuse;
    }
    __host__ __device__ float get_specular() const {
      return specular;
    }
};
