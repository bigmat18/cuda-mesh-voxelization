#ifndef MESH_IO
#define MESH_IO

#include <string>
#include <mesh/mesh.h>
#include <debug_utils.h>

bool ImportMesh(const std::string filename, Mesh& mesh);

bool ExportMesh(const std::string filename, const Mesh& mesh);

#endif // MESH_IO
