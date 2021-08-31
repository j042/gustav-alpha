import logging
import os
import ctypes

import numpy as np
import h5py

from .utils import *

def hdf5(fname, mesh):
    """
    Experimental mesh output in HDF5 format.
    It follows exact same 'mesh-data-structure' as mixd.
    Naming convension is same as CAMPIGA output files.

    Parameters
    -----------
    fname: str
    mesh: `Mesh`

    Returns
    --------
    None
    """
    # Check mesh input
    check_mesh_input(mesh)

    # Although the information is retrievable through array sizes,
    # we provide nice summerized infos.
    dim = mesh.vertices.shape[1]
    base_element_name, quad = check_element_info(mesh)

    # Let's jump into hdf5
    # First, create file
    hf = h5py.File(fname, "w")
    mesh_group = hf.create_group("mesh")
    # Second, vertices, 
    mesh_group.create_dataset(
        "coords",
        data=mesh.vertices.astype(ctypes.c_double)
    )
    # Third, connectivity
    # TODO: this is not valid for "embedded problems"
    connectivity = mesh.faces if dim == 2 else mesh.elements
    mesh_group.create_dataset(
        "connectivity",
        data=connectivity.astype(ctypes.c_int32)
    )
    # Fourth, boundary
    #   Formulate
    boundaries = np.ones_like(connectivity) * - 1
    for i, bgi in enumerate(mesh.bc_global_indices_):
        (global_element_ind,
         local_subelement_ind) = bc_global_and_local(bgi, dim, quad=quad)
        boundaries[global_element_ind, local_subelement_ind] = i + 1
    #   Write
    mesh_group.create_dataset(
        "boundary",
        data=boundaries.astype(ctypes.c_int32)
    )
    # Fifth, info
    mesh_info_group = hf.create_group("mesh/info")
    mesh_info_group.create_dataset(
        "nn",
        data=int(mesh.vertices.shape[0])
    )
    mesh_info_group.create_dataset(
        "ne",
        data=int(connectivity.shape[0])
    )
    mesh_info_group.create_dataset(
        "nsd",
        data=int(dim)
    )
    mesh_info_group.create_dataset(
        "nen",
        data=int(connectivity.shape[1])
    )
    # Sixth, boundary info
    # We skip for now #

    # Seventh, misc
    mesh_group.create_dataset(
        "edges",
        data=mesh.edges.astype(ctypes.c_int32)
    )
    # to avoid duplicating faces
    if len(mesh.faces) != len(connectivity):
        mesh_group.create_dataset(
            "faces",
            data=mesh.faces.astype(ctypes.c_int32)
        )

    # Water tight 3D meshes and tets don't have outlines
    if len(mesh.outlines) != 0:
        mesh_group.create_dataset(
            "outlines",
            data=mesh.outlines.reshape(-1, 1).astype(ctypes.c_int32)
        )

        if mesh.outline_2d_normals is not None:
            mesh_group.create_dataset(
                "outline_2d_normals",
                data=mesh.outline_2d_normals.astype(ctypes.c_double)
            )

    if len(mesh.faces) != len(mesh.surfaces):
        mesh_group.create_dataset(
            "surfaces",
            data=mesh.surfaces.reshape(-1, 1).astype(ctypes.c_int32)
        )
