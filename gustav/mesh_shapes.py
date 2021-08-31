import numpy as np
from . import utils
from .mesh import Mesh

def window(bounds, resolutions, quad=False, backslash=False):
    """
    Wrapper for utils.window. Direct mesh output.
    Attribute `boundary` is added to this Mesh, which is a list containing
    boundary line or each window edge. Index starts from the bottom and
    continues counter clock-wise. Line direction is also CCW.

    Parameters
    -----------
    bounds: (2, d) list-like
    resolutions: (2,) list-like
    quad: bool
    backslash: bool

    Returns
    --------
    window_mesh: Mesh
    """
    # Vertices and faces
    v, f = utils.window(bounds, resolutions, quad, backslash)

    # Mesh
    m = Mesh(vertices=v, faces=f)

    # Boundary
    b0 = v[:resolutions[0], :]
    b1 = v[int(resolutions[0] - 1)::resolutions[0], :]
    b2 = v[int(-1 * resolutions[0]):, :][::-1]
    b3 = v[::resolutions[0], :][::-1]

    m.boundary = [b0, b1, b2, b3]

    return m

def box3d(bounds, resolutions,):
    """
    Returns hexa elements based on bounds and resolutions.
    For element node index, take a look at `utils.make_hexa_elements`.
    For element face indedx, take a look at `mesh.faces`
    For raw vertex index, take a look here.

    Parameters
    -----------
    bounds: (2, 3) list-like
    resolutions: (r,) list-like

    Returns
    --------
    box3d: Mesh
    """
    bounds = np.asarray(bounds)

    bottom_line = np.linspace(
        bounds[0],
        [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
        resolutions[0],
    )

    ys = np.linspace(bounds[0, 1], bounds[1, 1], resolutions[1])

    bottom_vertices = [bottom_line]
    for y in ys[1:]:
        tmp_vertices = bottom_line.copy()
        tmp_vertices[:, 1] = y
        bottom_vertices.append(tmp_vertices)

    bottom_vertices = np.vstack(bottom_vertices)

    zs = np.linspace(bounds[0, 2], bounds[1, 2], resolutions[2])

    box_vertices = [bottom_vertices]
    for z in zs[1:]:
        tmp_vertices = bottom_vertices.copy()
        tmp_vertices[:, 2] = z
        box_vertices.append(tmp_vertices)

    box_vertices = np.vstack(box_vertices)

    box_elements = utils.make_hexa_elements(resolutions)

    return Mesh(vertices=box_vertices, elements=box_elements)
