import logging
from decimal import Decimal
import os
import numpy as np

def rjust(word, total,fill=" "):
    return word.rjust(total, fill)

def ljust(word, total, fill=" "):
    return word.ljust(total, fill)

def ljust15(word, fill=" "):
    return word.ljust(15, fill)

def ljust20(word, fill=" "):
    return word.ljust(20, fill)

def rjust20(word, fill=" "):
    return word.rjust(20, fill)

def check_mesh_input(mesh,):
    """
    Checks if 
      1. mesh input has `vertices` and `faces` as input
      2. len of BC names and BC global indices are same.
    Raises TypeError is it doesnt.

    Parameters
    -----------
    mesh: `Mesh`

    Returns
    --------
    None
    """
    # First attribute check
    if not hasattr(mesh, ("vertices" and "faces")):
        raise TypeError("Your mesh object does not have `vertices` and `faces` "
            + "attributes!")

    # Second, BC_names and BC_global_indices
    if len(mesh.bc_names_) != len(mesh.bc_global_indices_):
        raise ValueError(
            'length of bc_names_ and bc_global_indices_ do not match!'
        )

def check_element_info(mesh,):
    """
    Gives name of base element in string and whether it is quad based element
    or not.

    Parameters
    -----------
    mesh: `Mesh`

    Returns
    --------
    base_element_name: str
    quad: bool 
    """
    quad = False
    base_element_name = ""

    if mesh.elements is None:
        if mesh.faces.shape[1] == 4:
            quad = True
            base_element_name = "quadrilateral"

        else:
            base_element_name = "triangle"
    else:
        if mesh.elements.shape[1] == 8:
            quad = True
            base_element_name = "hexahedron"

        else:
            base_element_name = "tetrahedron"

    return base_element_name, quad

def get_nbsurf(BC_global_indices):
    """
    Parameters
    -----------
    BC_global_indices: list
      list of lists

    Returns
    --------
    nbsurf: int
    """
    nbsurf = 0
    for l in BC_global_indices:
        nbsurf += int(len(l))
    return nbsurf

def formulate_volumes(mesh, volume_type):
    """
    First block of tetrex.
    Fit all the lines in a list.
    Number starts at 1.

    Parameters
    -----------
    mesh: `Mesh`
    volume_type: int
      Triangle volume_type is 6.
      Tetrahedron volume_type is 1.

    Returns
    --------
    volumes: list
    """
    volumes = []
    if int(volume_type) == 6: 
        max_element_ind = int(mesh.faces.max())
        mei_length = int(len(str(max_element_ind)))
        num_just = int(mei_length + 1)
        elements = mesh.faces
    elif int(volume_type) == 1: 
        max_element_ind = int(mesh.elements.max())
        mei_length = int(len(str(max_element_ind)))
        num_just = int(mei_length + 1)
        elements = mesh.elements

    for i, e in enumerate(elements):
        line = rjust(str(i+1), num_just)
        line += rjust(str(volume_type), 20)
        line += "         "
        for ei in e:
            line += str(ei+1)
            line += " "
        volumes.append(line + "\n")
    return volumes

def formulate_vertices(mesh, dim=2,):
    """
    Note: If big E is an issue, fix that.

    Parameters
    -----------
    mesh: trimesh.Trimesh
    dim: int

    Returns
    --------
    vertices: list
    """
    if dim == 2:
        vertices = mesh.vertices[:,[0,1]].copy()
    elif dim == 3:
        vertices = mesh.vertices.copy()
    else:
        raise ValueError("Mesh dimension (`dim`) should me either 2 or 3.")

    verts = []
    for v in vertices:
        line = ""
        for ve in v:
            line += str("%.9E" % Decimal(str(ve)))
            line += " "
        verts.append(line + '\n')
    return verts


def bc_global_and_local(BC_global_indices, dim=2, quad=False):
    """
    Only support meshes with following properties:
    Edges are splited from faces. Thus (num_edges == num_faces * 3) is True.
    Faces are splited from elements. Thus (num_faces == num_elements * 3) is True.

    Parameters
    -----------
    BC_global_indices: list 
      list of int

    Returns
    --------
    global_element_ind: list
      list of int.
      In 2D, this is face ind.
      In 3D, this is element ind.
    local_subelement_ind: list
      list of int.
      In 2D, this is local edge ind.
      In 3D, this is local faces ind.
    """
    gei_np = np.asarray(BC_global_indices)
    if dim == 2:
        denom = 4 if quad else 3
        global_element_ind = (gei_np // denom).tolist()
        local_subelement_ind = (gei_np % denom).tolist()
    elif dim == 3:
        denom = 6 if quad else 4
        global_element_ind = (gei_np // denom).tolist()
        local_subelement_ind = (gei_np % denom).tolist()

    return global_element_ind, local_subelement_ind

def formulate_BC(BC_names, BC_global_indices, dim=2):
    """
    Maximum length of index is 20. If you need more than that, fix!

    Parameters
    -----------
    mesh: trimesh.Trimesh
    BC_names: list
    BC_global_indices: list

    Returns
    --------
    BC: list
    """
    BC = []
    assert len(BC_names) == len(BC_global_indices), "Length of BC_names "+\
        "and BC_global_indices do not match!"

    for n, e in zip(BC_names, BC_global_indices):
        (global_element_ind,
         local_subelement_ind) = bc_global_and_local(e, dim)
        for gei, lsi in zip(global_element_ind, local_subelement_ind):
            line = n + " "
            line += rjust20(str(gei+1))
            line += rjust(str(lsi+1), 20)
            BC.append(line + '\n')
    return BC

def check_and_makedirs(fname):
    """
    Given fname, it checks if the directories of the given path exists.
    If not, it makesdirs.

    Parameters
    -----------
    fname: str
      Path of exporting file.

    Returns
    --------
    None
    """
    dirs = os.path.dirname(fname)

    if dirs == "":
        return

    if not os.path.isdir(dirs):
        logging.debug("Export - Given fname's path does not exist.")
        logging.debug("Export -   Directories will be made.")

        os.makedirs(dirs)
