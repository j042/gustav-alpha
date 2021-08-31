import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

def configure_logging(debug=False, logfile=None):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("GUSTAV - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def get_rotation_matrix(rotation, degree=True):
    '''
    Compute rotation matrix.
    Works for both 2D and 3D point sets.
    In 2D, it can rotate along the (virtual) z-axis.
    In 3D, it can rotate along [x, y, z]-axis.

    Parameters
    -----------
    rotation: list or float
      Amount of rotation along [x,y,z] axis. Default is in degrees.
      In 2D, it can be float.
    degree: bool
      (Optional) rotation given in degrees.
      Default is `True`. If `False`, in radian.

    Returns
    --------
    rotation_matrix: np.ndarray (3,3)
    '''
    rotation = np.asarray(rotation).flatten()

    if degree == True:
        rotation = np.radians(rotation)

    # 2D
    if len(rotation) == 1:
        return R.from_rotvec([0, 0, rotation]).as_matrix()[:2, :2]

    # 3D
    elif len(rotation) == 3:
        return R.from_rotvec(rotation).as_matrix()


def rotate(points, rotation, degree=True):
    """
    Rotates given points. For more information, see `get_rotation_matrix()`.

    Parameters
    -----------
    points: (n, d) list-like
    rotation: list or float

    Returns
    --------
    rotated_points: (n, d) np.ndarray
    """
    return np.matmul(points, get_rotation_matrix(rotation, degree))


def bounding_box(points):
    '''
    Gets a smallest bounding box for given points.
    Its edges are parallel to the x,y,z axis.

    Parameters
    -----------
    points: np.ndarray (n,d)
      An array containing points.

    Returns
    --------
    box_corners: np.ndarray(8,d)
      8 Corner points to describe a bounding box.
    '''
    d = points.shape[1]
    if not (d == 2 or d == 3):
        raise ValueError("Input points should be 2D or 3D.")

    x_min = np.min(points[:,0])
    x_max = np.max(points[:,0])
    x_range = np.array([x_min, x_max])
    x = x_range[[0,1,1,0]]

    y_min = np.min(points[:,1])
    y_max = np.max(points[:,1])
    y_range = np.array([y_min, y_max])
    y = y_range[[0,0,1,1]]

    if points.shape[1] == 3:
        z_min = np.min(points[:,2])
        z_max = np.max(points[:,2])
        z_range = np.array([z_min, z_max])
        z = np.repeat(z_range, 4)

        x = np.tile(x, 2)
        y = np.tile(y, 2)

        box_corners = np.vstack((x,y,z)).transpose()

    elif points.shape[1] == 2:
        box_corners = np.vstack((x,y)).transpose()

    return box_corners


def bounds(points):
    '''
    Parameters
    -----------
    points: np.ndarray (n,d)
      An array containing points.

    Returns
    --------
    bounds: np.ndarray(2,d)
    '''
    bounds = np.vstack(
        (points.min(axis=0).reshape(1,-1),
         points.max(axis=0).reshape(1,-1),)
    )
    return bounds


def bounding_box_centroid(nodes):
    """
    Paramters
    ----------
    nodes: np.ndarray (n,d)

    Returns
    --------
    bounding_box_centroid:(d,) np.ndarray
      len = d
    """
    x = (np.min(nodes[:,0]) + np.max(nodes[:,0])) / 2
    y = (np.min(nodes[:,1]) + np.max(nodes[:,1])) / 2

    if nodes.shape[1] == 2:
        return np.array([x, y])
    elif nodes.shape[1] == 3:
        z = (np.min(nodes[:,2]) + np.max(nodes[:,2])) / 2
        return np.array([x, y, z])


def translate():
    pass


def scale():
    pass


def raster_points(dim, box_range=[-1,1], force_3d=False, lexsort=None):
    '''
    Gives you raster points in a [-1,1]^dim box.
    Returning points are alwyas 3D, as embree always expects 3D.

    Parameters
    -----------
    dim: list or tuple
      Resolution of points in each dimension.
    force_3d: bool

    Returns
    --------
    raster_points: (n, 3) np.ndarray
    '''
    if len(dim) == 2 and force_3d:
        raster_points = np.mgrid[
            box_range[0]:box_range[1]:(dim[0] * 1j),
            box_range[0]:box_range[1]:(dim[1] * 1j),
             0:0:1j
        ]
        p_dim = 3

    elif len(dim) == 2 and not force_3d:
        raster_points = np.mgrid[
            box_range[0]:box_range[1]:(dim[0] * 1j),
            box_range[0]:box_range[1]:(dim[1] * 1j),
        ]
        p_dim = 2

    elif len(dim) == 3:
        raster_points = np.mgrid[
            box_range[0]:box_range[1]:(dim[0] * 1j),
            box_range[0]:box_range[1]:(dim[1] * 1j),
            box_range[0]:box_range[1]:(dim[2] * 1j),
        ]
        p_dim = 3

    raster_points = raster_points.reshape(p_dim, -1).transpose()

    if lexsort is not None:
        raster_points = raster_points[
            np.lexsort([raster_points[:,i] for i in lexsort])
        ]


    return raster_points

def window(bounds, resolutions, quad=False, backslash=False):
    """
    Creates a structured or diagonalized grid of given bounds and resolutions.

    Parameters
    -----------
    bounds: (2, d) list-like
      [lower_left_corner, upper_right_corner]
    resolutions: (d,) list-like
    quad: bool
    backslash: bool

    Returns
    --------
    window_vertices: (n, 2 or 3) np.ndarray
    window_faces: (m, 3 or 4) np.ndarray
    """
    bounds = np.asarray(bounds)
    # Bounds to 4 points
    # Counter clock wise
    b0 = bounds[0].copy()

    #b1 = bounds[0].copy()
    #b1[-1] = bounds[1,-1]

    b2 = bounds[1].copy()

    b3 = bounds[1].copy()
    b3[-1] = bounds[0,-1]   
 
    bottom_line = np.linspace(b0, b3, resolutions[0],)

    heights = np.linspace(b3[-1], b2[-1], resolutions[1])

    window_vertices = [bottom_line]
    for h in heights[1:]:
        tmp_vertices = bottom_line.copy()
        tmp_vertices[:,-1] = h
        window_vertices.append(tmp_vertices)

    window_vertices = np.vstack(window_vertices)

    window_faces = make_quad_faces(resolutions)
    if not quad:
        window_faces = diagonalize_quad(window_faces, backslash=backslash)

    return window_vertices, window_faces

def ray_direction():
    pass


def bounds_to_center(bounds):
    """
    Computes center of given bounds.

    Parameters
    -----------
    bounds: (2, d) np.ndarray or similar

    Returns
    --------
    None
    """
    return np.asarray(bounds).mean(axis=0)

def bounds_to_ranges(bounds):
    """
    Turns bounds into range.
    Ex)
    > bounds_to_range([[x1, y1, z1], [x2, y2, z2]])
    > [[x1, x2], [y1, y2], [z1, z2]]

    Shhh, it is actually just a transpose.

    Parameters
    -----------
    bounds: (2, d) np.ndarray

    Returns
    --------
    ranges: (d, 2) np.ndarray
    """
    bounds = np.asarray(bounds)
    return bounds.transpose()

def box_scale(vertices,):
    """
    Computes normalization scale factor that will fit all 
    vertices into a [-1,1]^d box.

    Parameters
    -----------
    vertices: (n, d) np.ndarray 

    Returns
    --------
    scale_factor: double
    """
    # First, center the vertices to a bounding box centroid
    centered = vertices.copy()
    centered -= bounding_box_centroid(centered)

    return np.max(abs(centered))


def ball_scale(vertices,):
    """
    Computes normalization scale factor that will fit all
    vertices into a ball with 1^d radius.

    Parameters
    -----------
    vertices: (n, d) np.ndarray 

    Returns
    --------
    scale_factor: double
    """
    return np.max(np.linalg.norm(vertices, axis=1))


def closed_loop_index_train(ind_range):
    """
    Generates index train.
    Originally meant to connect the vertices.

    Parameters
    -----------
    ind_range: int or list

    Returns
    --------
    closed_loop_index_train: (n, 2) np.ndarray
    """
    if isinstance(ind_range, int):
        indices = np.arange(ind_range)
    elif isinstance(ind_range, (list or tuple)):
        indices = np.arange(*ind_range)

    indices = np.repeat(indices, 2)
    indices = np.append(indices, indices[0])[1:]
    indices = indices.reshape(-1,2)

    return indices


def open_loop_index_train(ind_range):
    """
    Generates index train. Unlike `closed_loop_index_train`,
    This does not connect last index to the first one.

    Parameters
    -----------
    ind_range: int or list

    Returns
    --------
    open_loop_index_train: (n, 2) np.ndarray
    """
    if isinstance(ind_range, int):
        indices = np.arange(ind_range)
    elif isinstance(ind_range, (list, tuple)):
        indices = np.arange(ind_range[0], ind_range[1]+1)
    indices = np.repeat(indices, 2)[1:-1]
    indices = indices.reshape(-1,2)

    return indices


def sequential_ind_to_edges(vertices, closed=True):
    """
    Given a sequnce of indices, this function turns them into edges.

    Parameters
    -----------
    vertices: (n,) np.ndarray or list

    Returns
    --------
    edges: (n, 2) np.ndarray
    """
    edges = np.asarray(vertices).copy()
    edges = np.repeat(edges, 2)[1:-1]
    edges = edges.reshape(-1,2)
    if closed:
        edges = np.vstack(
            (edges,
             [edges[-1,-1],edges[0,0]])
        )

    return edges


def is_same(vertices1, vertices2, tol=1e-10):
    """
    Checks if two given vertices are same unter the given tolerance.
    
    Parameters
    -----------
    vertices1: (n, d) np.ndarray
    vertices2: (n, d) np.ndarray
    tol: double

    Returns
    --------
    is_same: list
     list of bool
    """
    diff = abs(np.linalg.norm(vertices1 - vertices2, axis=1))

    return diff < tol
    

def make_quad_faces(number_of_nodes_per_dimension):
    """
    Given number of nodes per each dimension, returns connectivity information 
    of a structured mesh.
    Counter clock wise connectivity.

    (3)*------*(2)
       |      |
       |      |
    (0)*------*(1)

    Parameters
    -----------
    number_of_nodes_per_dimension: list

    Returns
    --------
    faces: (n, 4) np.ndarray
    """
    nnpd = np.asarray(number_of_nodes_per_dimension)
    total_nodes = np.product(nnpd)
    total_faces = (nnpd[0] - 1) * (nnpd[1] - 1)
    node_indices = np.arange(total_nodes).reshape(nnpd[1], nnpd[0])
    faces = np.ones((total_faces, 4)) * -1

    faces[:,0] = node_indices[:(nnpd[1] - 1), :(nnpd[0] - 1)].flatten()
    faces[:,1] = node_indices[:(nnpd[1] - 1), 1:nnpd[0]].flatten()
    faces[:,2] = node_indices[1:nnpd[1], 1:nnpd[0]].flatten()
    faces[:,3] = node_indices[1:nnpd[1], :(nnpd[0]-1)].flatten()

    if faces.all() == -1:
        raise ValueError("Something went wrong during `make_quad_faces`.")

    return faces.astype(np.int32)


def diagonalize_quad(quad_faces, backslash=False):
    """
    Given quad faces, diagonalize them to turn them into triangles.
    If quad is CCW, triangle will also be CCW and vise versa.
    Default diagonalization looks like this:

    (3) *---* (2)
        |  /|
        | / |
        |/  |
    (0) *---* (1)
    resembling `slash`.
    If you want to diagonalize the other way, set `backslash=True`.
    

    Parameters
    -----------
    quad_faces: (n, 4) np.ndarray
    backslash: bool

    Returns
    --------
    tri_faces: (n*2, 3) np.ndarray
    """
    tri_faces = np.ones((quad_faces.shape[0] * 2, 3), dtype=np.int32) * -1
    tf_half = int(quad_faces.shape[0])
    if not backslash:
        tri_faces[:tf_half] = quad_faces[:, :3]
        tri_faces[tf_half:] = quad_faces[:, [2, 3, 0]]
    else:
        tri_faces[:tf_half] = quad_faces[:, [0, 1, 3]]
        tri_faces[tf_half:] = quad_faces[:, [3, 1, 2]]

    return tri_faces

def make_hexa_elements(number_of_nodes_per_dimension):
    """
    Given number of nodes per each dimension, returns connectivity information 
    of structured hexahedron elements.
    Counter clock wise connectivity.

       (7)*-------*(6)
         /|      /|
        / | (5) / |
    (4)*-------*  |
       |  *----|--*(2)
       | /(3)  | /
       |/      |/
    (0)*-------*(1)

    Parameters
    -----------
    number_of_nodes_per_dimension: list

    Returns
    --------
    elements: (n, 8) np.ndarray
    """
    nnpd = np.asarray(number_of_nodes_per_dimension)
    total_nodes = np.product(nnpd)
    total_elements = np.product(nnpd - 1)
    node_indices = np.arange(total_nodes, dtype=np.int32).reshape(nnpd[::-1])
    elements = np.ones((total_elements, 8), dtype=np.int32) * int(-1)

    elements[:, 0] = node_indices[
        :(nnpd[2] - 1),
        :(nnpd[1] - 1),
        :(nnpd[0] - 1)
    ].flatten()
    elements[:, 1] = node_indices[
        :(nnpd[2] - 1),
        :(nnpd[1] - 1),
        1:nnpd[0]
    ].flatten()
    elements[:, 2] = node_indices[
        :(nnpd[2] - 1),
        1:nnpd[1],
        1:nnpd[0]
    ].flatten()
    elements[:, 3] = node_indices[
        :(nnpd[2] - 1),
        1:nnpd[1],
        :(nnpd[0]-1)
    ].flatten()
    elements[:, 4] = node_indices[
        1:nnpd[2],
        :(nnpd[1] - 1),
        :(nnpd[0] - 1)
    ].flatten()
    elements[:, 5] = node_indices[
        1:nnpd[2],
        :(nnpd[1] - 1),
        1:nnpd[0]
    ].flatten()
    elements[:, 6] = node_indices[
        1:nnpd[2],
        1:nnpd[1],
        1:nnpd[0]
    ].flatten()
    elements[:, 7] = node_indices[
        1:nnpd[2],
        1:nnpd[1],
        :(nnpd[0]-1)
    ].flatten()

    if (elements == -1).any():
        raise ValueError("Something went wrong during `make_hexa_elements`.")

    return elements.astype(np.int32)

def outline_to_line(vertices, outline_edges):
    """
    Turn outline edges to a point sequence.
    Meant to turn outline into a polygon.

    Parameters
    -----------
    vertices: (n, d) np.ndarray
    outline_edges: (m, 2) np.ndarray

    Returns
    --------
    line: (l, d) np.ndarray
    """
    unique_vertex_ind = np.unique(outline_edges)
    line = np.empty((len(unique_vertex_ind), vertices.shape[1]))
    first_edge = outline_edges[0].copy()
    line[[0,1], :] = vertices[first_edge]

    next_edges = outline_edges[1:]
    previous_edge = first_edge
    vertex_mask = [False, True]

    for i in range(2, len(line)):
        ind, _ = np.where(next_edges == previous_edge[vertex_mask])
        edge = next_edges[int(ind)]
        vertex_mask = edge - previous_edge[vertex_mask]
        vertex_mask = abs(vertex_mask) > 1e-8

        line[i, :] = vertices[edge[vertex_mask]]
        ne_mask = np.ones(next_edges.shape[0], dtype=bool) # next_edge_mask
        ne_mask[ind] = False
        next_edges = next_edges[ne_mask]
        previous_edge = edge.copy()

    return line
