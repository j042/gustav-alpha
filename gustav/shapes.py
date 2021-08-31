import numpy as np

"""
TODO: rename this to point_shapes
"""

def sample_from_circle(
    ecks,
    r=.5,
    z_offset=None,
    random=False
):
    '''
    Samples points from a circle of given radius. 
    CCW.

    Parameters
    -----------
    ecks: int
    r: float
    z: float
      Optional. If specified, returns 3D co-planar circle.
    random: bool
      Don't use it unless desperate.

    Returns
    --------
    sampled_points: (int(ecks), d) np.ndarray
    '''

    angles = np.arange(0,360, 360/ecks) * ( np.pi / 180 )

    if random:
        angles += np.random.uniform(0,(360/ecks), size=len(angles))

    x = r * np.cos(angles)
    y = r * np.sin(angles)

    if z_offset is not None:
        z = np.zeros(int(ecks)) + z_offset
        return np.vstack((x,y,z)).transpose()

    else:
        return np.vstack((x,y)).transpose()
    

def sample_from_ellipse(
    ecks,
    x=1,
    y=1,
    r=.5,
    z_offset=None,
    random=False
):
    '''
    Samples points from a parametric ellipse. 
    CCW.

    Parameters
    -----------
    ecks: int
    x: float
    y: float
    r: float
    z_offset: float
      Optional. If specified, returns 3D co-planar ellipse.
    random: bool
      Don't use it unless desperate.

    Returns
    --------
    sampled_points: (int(ecks), d) np.ndarray
    '''
    angles = np.arange(0,360, 360/ecks) * ( np.pi / 180 )

    if random:
        angles += np.random.uniform(0,(360/ecks), size=len(angles))

    x_coord = r*x*np.cos(angles)
    y_coord = r*y*np.sin(angles)

    if z_offset is not None:
        z_coord = np.zeros(int(ecks)) + z_offset
        return np.vstack((x_coord,y_coord,z_coord)).transpose()

    else:
        return np.vstack((x_coord,y_coord)).transpose()


def box_from_bounds(bounds, connectivity=False, CCW=False):
    """
    Given bounds, returns vertices of corresponding box.
    Strictly assumes that bounds[0] is [min, min] and
    bounds[1] is [max, max] coordinate of the bounding box.
    Winds Clockwise unless specified otherwise
    For 2D only, for now.

    Parameters
    -----------
    bounds: (2, 2) np.ndarray
    connectivity: bool
    CCW: bool

    Returns
    --------
    box: (4, 2) np.ndarray
    """
    box = np.array(
        [[bounds[0,0], bounds[0,1]],
         [bounds[0,0], bounds[1,1]],
         [bounds[1,0], bounds[1,1]],
         [bounds[1,0], bounds[0,1]],]
    )

    if connectivity:
        connectivity = closed_loop_index_train(box.shape[0])
        if CCW:
            connectivity = connectivity[[0,3,2,1]]
            connectivity = np.hstack(
                (connectivity[:,1].reshape(-1,1),
                 connectivity[:,0].reshape(-1,1))
            )

        return box, connectivity
    else:
        return box

def box3d(bounds, resolutions,):
    """
    Returns vertices of regular gird, defined by bounds and resolutions.

    Parameters
    -----------
    bounds: (2, 3) list-like
    resolutions: (r,) list-like

    Returns
    --------
    box_vertices: (resolutions^3, dim) np.ndarray
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

    return np.vstack(box_vertices)
