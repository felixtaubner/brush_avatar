from typing import List

import einops
import numpy as np
from plyfile import PlyData, PlyElement


def normalize_property(prop, axis=0) -> None:
    prop_max = np.max(prop, axis=axis, keepdims=True)
    prop_min = np.min(prop, axis=axis, keepdims=True)

    prop_normalized = (prop - prop_min) / (prop_max - prop_min)

    return prop_normalized, prop_min[axis], prop_max[axis]


def quantize_property(prop: np.ndarray, axis=0) -> None:
    prop_normalized, prop_min, prop_max = normalize_property(prop, axis)
    prop_quantized = (prop_normalized * 255).astype(np.uint8)

    return prop_quantized, prop_min, prop_max


def save_ply(
    ply_path: str, 
    xyz_local: np.ndarray,
    log_scale_local: np.ndarray,
    rotation_local: np.ndarray,
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    raw_opacities: np.ndarray,
    binding: np.ndarray,
    faces: np.ndarray,
    vertices_list: List[np.ndarray],
    quantize_vertex_offsets: bool = True,
) -> None:
    """
    Saves the data into a PLY file format.

    Parameters:
    ply_path (str): 
        The path where the PLY file will be saved.
    
    xyz_local (np.ndarray) [N_splats, 3]: 
        A NumPy float array representing the local 3D coordinates of the splats. 
        Each row is a vertex with x, y, z coordinates.
    
    log_scale_local (np.ndarray) [N_splats, 1]: 
        A NumPy float array representing the log-scale values for each splat. 
    
    rotation_local (np.ndarray) [N_splats, 4]: 
        A NumPy float array containing the rotation data (wxyz Quaternions) for each vertex. 
    
    f_dc (np.ndarray) [N_splats, 3]: TODO: What is the shape?
        A NumPy float array representing zero order SH-coefficients for each splat.
    
    f_rest (np.ndarray) [N_splats, N_sh * 3]: 
        A NumPy float array of shape (N_faces, N_rest_dims) representing rest of the SH coefficients (flattened!) for each vertex.
    
    raw_opacities (np.ndarray) [N_splats, 1]: 
        A NumPy float array of shape (N_vertices,) representing the opacity values for each vertex.
    
    binding (np.ndarray) [N_splats]: 
        A NumPy int array of shape (N_vertices,) representing the binding (face index) information for each splat.
    
    faces (np.ndarray): 
        A NumPy int array of shape (N_faces, 3) representing the vertex indices of each faces in the mesh. 
    
    vertices_list (List[np.ndarray]) N_frames x [N_vertices, 3]: 
        A list of NumPy float arrays, where each array represents the vertex positions for a given frame.
    
    quantize_vertex_offsets (bool, optional): 
        A boolean flag indicating whether to quantize the per-frame vertex offsets (default is True).
        Significantly decreases the file size at the cost of accuracy.
    """
        
    n_frames = len(vertices_list)
    assert n_frames > 0
    init_vertices = vertices_list[0]

    ply_elements = []

    # save mesh faces
    l = ['index_0', 'index_1', 'index_2']
    dtype_full = [(attribute, 'u4') for attribute in l]
    face_elements = np.empty(faces.shape[0], dtype=dtype_full)
    face_elements[:] = list(map(tuple, faces))
    face_el = PlyElement.describe(face_elements, 'faces')
    ply_elements.append(face_el)
    
    # save initial vertices 
    l = ['x', 'y', 'z']
    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(init_vertices.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, init_vertices))
    el = PlyElement.describe(elements, 'base_vertex')
    ply_elements.append(el)

    # flatten SH coefficients
    f_dc = einops.rearrange(f_dc, 'n sh rgb -> n (rgb sh)')
    f_rest = einops.rearrange(f_rest, 'n sh rgb -> n (rgb sh)')

    # save local splats and bindings
    l = ['x', 'y', 'z']
    for j in range(f_dc.shape[1]):
        l.append(f'f_dc_{j}')
    for j in range(f_rest.shape[1]):
        l.append(f'f_rest_{j}')
    l.append('opacity')
    for j in range(log_scale_local.shape[1]):
        l.append(f'scale_{j}')
    for j in range(rotation_local.shape[1]):
        l.append(f'rot_{j}')
    l.append('binding')

    attributes = np.concatenate([
        xyz_local, 
        f_dc,
        f_rest,
        raw_opacities,
        log_scale_local,
        rotation_local,
        binding[..., None],
    ], axis=1)

    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(xyz_local.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    ply_elements.append(el)
    
    # save per frame offsets (optionally quantized)
    for i in range(n_frames):
        offset = vertices_list[i] - init_vertices
        if quantize_vertex_offsets:
            offset_quant, offset_min, offset_max = quantize_property(offset)
        else:
            offset_quant, offset_min, offset_max = normalize_property(offset)

        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]
        elements = np.empty(1, dtype=dtype_full)
        elements[:] = list(map(tuple, offset_min[None]))
        el = PlyElement.describe(elements, f'meta_delta_min_{i:05d}')
        ply_elements.append(el)
        
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]
        elements = np.empty(1, dtype=dtype_full)
        elements[:] = list(map(tuple, offset_max[None]))
        el = PlyElement.describe(elements, f'meta_delta_max_{i:05d}')
        ply_elements.append(el)

        dtype_full = [(attribute, 'u1') for attribute in ['x', 'y', 'z']]
        elements = np.empty(offset_quant.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, offset_quant))
        el = PlyElement.describe(elements, f'delta_vertex_{i:05d}')
        ply_elements.append(el)

    # write data to ply file
    PlyData(ply_elements).write(ply_path)
