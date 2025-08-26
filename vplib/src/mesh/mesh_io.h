#ifndef MESH_IO
#define MESH_IO

#include <string>
#include <mesh/mesh.h>
#include <debug_utils.h>

/**
 * @brief Imports a mesh from a file.
 * 
 * @param filename Path to the mesh file to import.
 * @param mesh Reference to a Mesh object where the imported data will be stored.
 * @return true if the mesh was successfully imported, false otherwise.
 */
bool ImportMesh(const std::string filename, Mesh& mesh);

/**
 * @brief Exports a mesh to a file.
 * 
 * @param filename Path to the file where the mesh will be saved.
 * @param mesh Reference to the Mesh object containing the data to export.
 * @return true if the mesh was successfully exported, false otherwise.
 */
bool ExportMesh(const std::string filename, const Mesh& mesh);

#endif // MESH_IO
