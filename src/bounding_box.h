#include <functional>
#include <optional>
#include <span>
#include <utility>

#include <debug_utils.h>
#include <mesh/mesh.h>

#ifndef BOUNDING_BOX_H


template<bool device = false> __host__ __device__ 
float CalculateBoundingBox(std::span<Position> coordinates,
                           std::optional<std::reference_wrapper<std::pair<float, float>>> minmaxX = std::nullopt,
                           std::optional<std::reference_wrapper<std::pair<float, float>>> minmaxY = std::nullopt,
                           std::optional<std::reference_wrapper<std::pair<float, float>>> minmaxZ = std::nullopt)
{
    if constexpr (device) {
        // Run GPU version 
    } else {
        
        float minX = coordinates[0].X, maxX = coordinates[0].X;
        float minY = coordinates[0].Y, maxY = coordinates[0].Y;
        float minZ = coordinates[0].Z, maxZ = coordinates[0].Z;   

        for (unsigned int i = 1; i < coordinates.size(); ++i) {
            if(coordinates[i].X < minX) minX = coordinates[i].X;
            else if(coordinates[i].X > maxX) maxX = coordinates[i].X;

            if(coordinates[i].Y < minY) minY = coordinates[i].Y;
            else if(coordinates[i].Y > maxY) maxY = coordinates[i].Y;

            if(coordinates[i].Z < minZ) minZ = coordinates[i].Z;
            else if(coordinates[i].Z > maxZ) maxZ= coordinates[i].Z;
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
