import os
import meshio
import splinelibpy
import numpy as np
from .mesh import Mesh
from .bspline import BSpline
from .nurbs import NURBS

def load_mesh(fname):
    """
    Loads surface mesh using meshio.
    Not meant for mixed shape meshes.

    Parameters
    -----------
    fname: str

    Returns
    --------
    mesh: Mesh
    """
    fname = abs_fname_(fname)

    m = meshio.read(fname)
    mesh = Mesh()
    mesh.vertices = m.points

    for i, c in enumerate(m.cells):
        if i == 0:
            faces = c.data
        else:
            faces = np.vstack((faces, c.data))

    mesh.faces = faces

    return mesh

def load_volume_mesh(fname):
    """
    Loads volume mesh using meshio.
    Not meant for mixed shape meshes.

    Parameters
    -----------
    fname: str

    Returns
    --------
    mesh: Mesh
    """
    fname = abs_fname_(fname)

    m = meshio.read(fname)
    mesh = Mesh()
    mesh.vertices = m.points

    for i, c in enumerate(m.cells):
        if i == 0:
            elements = c.data
        else:
            elements = np.vstack((elements, c.data))

    mesh.elements = elements

    return mesh

def mixd_load_(fname=None, mxyz=None, mien=None):
    """
    Raw loading function that can be used from `load_mixd` and 
    `load_volume_mixd`. Meant for internal use.

    Parameters
    -----------
    fname: str
    mxyz: str
    mien: str

    Returns
    --------
    vertices: (n,) np.ndarray
    connectivity: (m,) np.ndarray
    """
    fname = abs_fname_(fname)

    if fname is None and (mxyz is None and mien is None):
        raise ValueError(
            "Either `fname` or (`mxyz` and `mien`) needs to be defined."
        )

    if fname is None:
        if (
            (mxyz is None and mien is not None)
            or (mxyz is not None and mien is None)
        ):
            raise ValueError(
                "Both `mxyz` and `mien` needs to be defined."
            )

    if fname is not None:
        base, ext = os.path.splitext(fname)

        if ext == ".campiga":
            mxyz = base + ".coords"
            mien = base + ".connectivity"

        elif ext == ".xns":
            mxyz = base + ".mxyz"
            mien = base + ".mien"

    vertices = np.fromfile(mxyz, dtype=">d").astype(np.double)
    #> Starts at 1, but need 0. Thus, -1.
    connectivity = (np.fromfile(mien, dtype=">i") - int(1)).astype(np.int32)

    return vertices, connectivity


def load_mixd(dim, fname=None, mxyz=None, mien=None, quad=False):
    """
    Loads mixd meshes.

    TODO: Import boundaries, when boundary viewing is ready.

    Parameters
    -----------
    fname: str
      This can be used if MIXD files has same base name and different exts.
      Ext should either be ".xns" or ".campiga"

      If both fname and (mxyz and mien) is defined, this function takes fname.

      Ex1) `fname=mesh.xns` for mesh.mxyz, mesh.mien, mesh.mrng, mesh.minf.

      Ex2) `fname=mesh.campiga` for mesh.coords, mesh.connectivity, ...
    mxyz: str
    mien: str
    dim: int
    quad: bool

    Returns
    --------
    mesh: Mesh
    """
    vertices, faces = mixd_load_(fname, mxyz, mien)

    mesh = Mesh()
    mesh.vertices = vertices.reshape(-1, dim)

    if quad:
        mesh.faces = faces.reshape(-1, 4)
    else:
        mesh.faces = faces.reshape(-1, 3)

    return mesh

def load_volume_mixd(dim, fname=None, mxyz=None, mien=None, hexa=False):
    """
    Loads mixd volume meshes.

    Parameters
    -----------
    fname: str
      This can be used if MIXD files has same base name and different exts.
      Ext should either be ".xns" or ".campiga"

      If both fname and (mxyz and mien) is defined, this function takes fname.

      Ex1) `fname=mesh.xns` for mesh.mxyz, mesh.mien, mesh.mrng, mesh.minf.

      Ex2) `fname=mesh.campiga` for mesh.coords, mesh.connectivity, ...
    mxyz: str
    mien: str
    dim: int
    hexa: bool

    Returns
    --------
    mesh: Mesh
    """
    vertices, elements = mixd_load_(fname, mxyz, mien)

    mesh = Mesh()
    mesh.vertices = vertices.reshape(-1, dim)

    if hexa:
        mesh.elements = elements.reshape(-1, 8)
    else:
        mesh.elements = elements.reshape(-1, 4)

    return mesh

def load_splines(fname):
    """
    Loads spline files of extension 
      - `.iges`
      - `.xml`
      - `.itd`

    Parameters
    -----------
    fname: str

    Returns
    --------
    splines: list of BSpline or/and NURBS
    """
    fname = str(fname)
    fname = abs_fname_(fname)

    sr = splinelibpy.Reader()

    ext = os.path.splitext(fname)[1]
    
    if ext == ".iges":
        loaded_splines = sr.read_iges(fname)
    elif ext == ".xml":
        loaded_splines = sr.read_xml(fname)
    elif ext == ".itd":
        loaded_splines = sr.read_irit(fname)
    else:
        raise ImportError(
            "We can only import < .iges | .xml | .itd > spline files"
        )

    splines = []
    # Format s => [weights, degrees, knot_vectors, control_points]
    for s in loaded_splines:
        if s[0] is None:
            # Bbspline.
            tmp_spline = BSpline()
            tmp_spline.degrees = s[1]
            tmp_spline.knot_vectors = s[2]
            tmp_spline.control_points = s[3]

            splines.append(tmp_spline)
 
        else:
            # Make nurbs
            tmp_spline = NURBS()
            tmp_spline.weights = s[0]
            tmp_spline.degrees = s[1]
            tmp_spline.knot_vectors = s[2]
            tmp_spline.control_points = s[3]

            splines.append(tmp_spline)

    return splines

def abs_fname_(fname):
    """
    Checks if fname is absolute. If not, turns it into an abspath. Tilde safe.

    Parameters
    -----------
    fname: str

    Returns
    --------
    abs_fname: str
      Maybe same to fname, maybe not.
    """
    if os.path.isabs(fname):
        pass
    elif '~' in fname:
        fname = os.path.expanduser(fname)
    else:
        fname = os.path.abspath(fname)

    return fname
