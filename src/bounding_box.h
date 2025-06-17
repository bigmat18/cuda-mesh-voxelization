#include "mesh_io.h"
#include <functional>
#include <optional>
#include <span>
#include <utility>

#ifndef BOUNDING_BOX_H


template<bool device = false> __host__ __device__ 
float CalculateBoundingBox(std::span<Vertex> vertices,
                           std::optional<std::reference_wrapper<std::pair<float, float>>> minmaxX = std::nullopt,
                           std::optional<std::reference_wrapper<std::pair<float, float>>> minmaxY = std::nullopt,
                           std::optional<std::reference_wrapper<std::pair<float, float>>> minmaxZ = std::nullopt)
{
    if constexpr (device) {
        // Run GPU version 
    } else {
        
        float minX = vertices[0].X, maxX = vertices[0].X;
        float minY = vertices[0].Y, maxY = vertices[0].Y;
        float minZ = vertices[0].Z, maxZ = vertices[0].Z;   

        for (unsigned int i = 1; i < vertices.size(); ++i) {
            if(vertices[i].X < minX) minX = vertices[i].X;
            else if(vertices[i].X > maxX) maxX = vertices[i].X;

            if(vertices[i].Y < minY) minY = vertices[i].Y;
            else if(vertices[i].Y > maxY) maxY = vertices[i].Y;

            if(vertices[i].Z < minZ) minZ = vertices[i].Z;
            else if(vertices[i].Z > maxZ) maxZ= vertices[i].Z;
        }
        
        #ifdef DEBUG 
        LOG_INFO("minX: %f, maxX: %f", minX, maxX);
        LOG_INFO("minY: %f, maxY: %f", minY, maxY); 
        LOG_INFO("minZ: %f, maxZ: %f", minZ, maxZ);
        #endif // DEBUG

        if (minmaxX) minmaxX->get() = {minX, maxX};
        if (minmaxY) minmaxY->get() = {minY, maxY};
        if (minmaxZ) minmaxZ->get() = {minZ, maxZ};

        return std::max({maxX - minX, maxY - minY, maxZ - minZ});

    }
}

#endif // !BOUNDING_BOX_H
