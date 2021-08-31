import logging
import struct
import os

import numpy as np

from .utils import *

def mixd(fname, mesh, space_time):
    """
    Write mixd file with given information.
    Currently supports triangle, quadrilateral, tetrahedron, and hexahedron
    semi-discrete mesh outputs.

    Parameters
    -----------
    fname: str
    mesh: `Mesh`
    
    Returns
    --------
    None
    """
    ####################
    ### Start Output ###
    ####################


    # Check input
    # First, mesh 
    if not hasattr(mesh, ("vertices" and "faces")):
        raise TypeError("Your mesh object does not have `vertices` and `faces` "+
            "attributes!")

    # Second, BC_names and BC_global_indices
    if len(mesh.bc_names_) != len(mesh.bc_global_indices_):
        raise ValueError(
            'length of bc_names_ and bc_global_indices_ do not match!'
        )

    # Gather some data
    # -----------------
    dim = mesh.vertices.shape[1]
    
    # big endian
    big_endian_int = ">i"
    big_endian_double = ">d"

    # Split ext
    base, ext = os.path.splitext(fname)

    # open files
    if ext == ".campiga":
        vertices_file  = open(base + ".coords", "wb")
        connectivity_file  = open(base + ".connectivity", "wb")
        boundary_file = open(base + ".boundary", "wb")
        info_file  = open(base + ".info", "w")

    elif ext == ".xns":
        # Special case if fname was "_.xns", output mxyz, mien, mrng, minf.
        if os.path.basename(base) == "_":
            logging.debug("Export - Congratulation!") 
            logging.debug("Export - You've found a special export name.") 
            logging.debug("Export - `_.xns` will be transformed into:") 
            logging.debug("Export -    mxyz, mien, mrng, minf.")

            prepend = "/" if os.path.isabs(base) else ""
            base = prepend + os.path.join(*base.split("/")[:-1]) + "/"

        else:
            base += "."

        vertices_file  = open(base + "mxyz", "wb")
        connectivity_file  = open(base + "mien", "wb")
        boundary_file = open(base + "mrng", "wb")
        info_file  = open(base + "minf", "w")
       
    # Write vertices
    for v in mesh.vertices.flatten():
        vertices_file.write(
            struct.pack(big_endian_double, v)
        )

    # For xns, spacetime meshes just have vertices twice.
    if space_time and ext == ".xns":
        for v in mesh.vertices.flatten():
            vertices_file.write(
                struct.pack(big_endian_double, v)
            )

    vertices_file.close()

    # Write connectivity
    #   2D: faces
    #   3D: elements
    quad = False
    if dim == 2:
        connectivity = mesh.faces
        boundary_width = 3
        mesh_type = "triangle"

        if connectivity.shape[1] == 4:
            quad = True
            boundary_width = 4
            mesh_type = "quadrilateral"

    elif dim == 3:
        connectivity = mesh.elements
        boundary_width = 4
        mesh_type = "tetrahedron"

        if connectivity.shape[1] == 8:
            quad = True
            boundary_width = 6
            mesh_type = "hexahedron"

    # Connectivity index begins with 1.
    for c in (connectivity.flatten() + 1):
        connectivity_file.write(
            struct.pack(big_endian_int, c)
        )
    connectivity_file.close()

    # Write boundary. Boundary index begins with 1.
    # Non-Boundary entries are all -1. This could be (-1 * neighbor_elem_ind),
    # but it isn't.
    boundaries = np.ones((connectivity.shape[0], boundary_width)) * - 1
    for i, bgi in enumerate(mesh.bc_global_indices_):
        (global_element_ind,
         local_subelement_ind) = bc_global_and_local(bgi, dim, quad=quad)
        boundaries[global_element_ind, local_subelement_ind] = i + 1

    for b in boundaries.flatten():
        boundary_file.write(
            struct.pack(big_endian_int, int(b))
        )
    boundary_file.close()

    # Conclude Info file
    # Start with general info
    info_file.write("# dim: "+ str(dim) + "\n")
    info_file.write("# mesh type: " + mesh_type + "\n\n")

    # Crucial info
    # Supports semi-descrete and xns space-time.
    st_factor = 2 if ext == ".xns" and space_time else 1

    info_file.write("nn "+ str(int(mesh.vertices.shape[0] * st_factor)) + "\n")
    info_file.write("ne "+ str(connectivity.shape[0]) + "\n")
    info_file.write("nsd "+ str(dim) + "\n")
    info_file.write("nen "+ str(int(connectivity.shape[1] * st_factor)) + "\n")
    if space_time and ext == ".xns":
        info_file.write("space-time on" + "\n\n\n")
        
    else:
        info_file.write("semi-discrete on" + "\n\n\n")

    # BC guide
    info_file.write("# Info: BCs should be referenced by the numbers stated "+\
        "in `< >`." + "\n")

    # BC info
    for i, bc in enumerate(mesh.bc_names_):
        info_file.write(
            "# Name of boundary <" + str(i + 1) + "> : " + bc + "\n"
        )

    # Signature
    info_file.write("\n\n\n" + "# MIXD Generated using `gustav`." + "\n")
    info_file.close()
