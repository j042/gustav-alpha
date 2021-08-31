import numpy as np

from . import utils
from . import shapes
from .bspline import BSpline
from .nurbs import NURBS

def box2d(
    degrees=None,
    elements=None,
    parametric_bounds=[[0,1], [0,1]],
    physical_bounds=[[0,1], [0,1]],
):
    """
    """
    pass

def box3d(
    degrees=None,
    elements=None,
    parametric_bounds=[[0,0,0], [1,1,1]],
    physical_bounds=[[0,0,0], [1,1,1]],
):
    """
    """
    pass

def box(
    parametric_bounds,
    physical_bounds,
    degrees=None,
    elements=None,
    NURBS=False,
):
    """
    Simple box spline generator. Wraps around `naive_spline()`.
    Applicable for parametric/physical dimension 1,2,3.
    Box in 1D, is a line.
    Could be useful for FFD, where one can generate splines with desired
    specification.

    Parameters
    -----------
    parametric_bounds: (2, para_dim) list-like
    physical_bounds: (2, phys_dim) list-like
    degrees: (phys_dim, )list-like
    elements: (phys_dim,) list-like
    NURBS: bool

    Returns
    --------
    box_spline: `BSpline` or `NURBS`
    """
    if degrees is None and elements is None:
        raise ValueError("Please define either `degrees` or `elements`.")

    # Implicitly check, if bounds are given in a correct from.
    para_b = np.asarray(parametric_bounds)
    assert para_b.min() >= 0, "Paremetric bounds should be positive definite."
    phys_b = np.asarray(physical_bounds) 

    # Extract para/phys dims
    para_dim = para_b.shape[1]
    phys_dim = phys_b.shape[1]

    # Apply default values if degrees or elements is not defined
    if degrees is None:
        degrees = [1 for _ in range(para_dim)]

    if elements is None:
        elements = [1 for _ in range(para_dim)]

    # Some assert to check degrees and elements input
    #if degrees is not None:
    assert len(degrees) == para_dim,\
        "len of `degrees` does not match parametric dimension."

    #if elements is not None:
    assert len(elements) == para_dim,\
        "len of `elements` does not match parametric dimension."

    # Get naive_spline
    box_spline = naive_spline(para_dim, phys_dim, NURBS)

    # Manipulate naive_spline into "street_smart_spline"
    #
    # Knots
    # Knot vectors in parameteric dimension should be [0,0,1,1].
    # So first re-define those according to bounds
    new_knots = [] # Define new knots to use setter (avoids manual update_c_)
    for i, kv in enumerate(box_spline.knot_vectors):
        assert len(kv) == 4,\
            "Something went wrong during generation of naive_spline"
        new_knots.append(
            [
                para_b[0, i],
                para_b[0, i],
                para_b[1, i],
                para_b[1, i],
            ]
        )
    # Apply new knots using setter -> Should update c spline.
    box_spline.knot_vectors = new_knots

    # One more loop to adjsut number of elements
    for i, e in enumerate(elements):
        if int(e) == int(1):
            continue
        else:
            knots_to_add = np.linspace(
                para_b[0,i],
                para_b[1,i],
                e + 1,
            )[1:-1]
            box_spline.insert_knots(i, knots_to_add)

    # Degrees
    for i, d in enumerate(degrees):
        # Could hard code, but this is "safe".
        for _ in range(d - box_spline.degrees[i]):
            box_spline.elevate_degree(i)

    return box_spline

def naive_spline(
    parametric_dim,
    physical_dim,
    NURBS=False,
):
    """
    Returns naive spline. In other words, spline with both parametric and
    physical coordinates with range [0, 1]

    Parameters
    -----------
    parametric_dim: int
    physical_dim: int
    NURBS: bool
      (Optional) Default is False.

    Returns
    --------
    spline: `BSpline` or `NURBS`
      NURBS if `NURBS=True` else BSpline.
    """
    # Prepare knot vectors, physical bounds, resolutions, and degrees
    knot_vectors = [[0, 0, 1, 1] for _ in range(parametric_dim)]
    physical_bounds = [
        [0 for _ in range(physical_dim)],
        [1 for _ in range(physical_dim)],
    ]
    resolutions = [2 for _ in range(parametric_dim)]
    degrees = [1 for _ in range(parametric_dim)]
    

    # Prepare control points
    if parametric_dim == 1:
        # This will be a "diagonal" line
        control_points = physical_bounds

    elif parametric_dim == 2:
        # TODO: put `utils.window` to `shapes.window`
        control_points, _ = utils.window(
            bounds=physical_bounds,
            resolutions=resolutions,
            quad=True,
        )

    elif parametric_dim == 3:
        control_points = shapes.box3d(
            bounds=physical_bounds,
            resolutions=resolutions,
        )

    else:
        raise ValueError(
            "Sorry, `gustav` cannot yet imagine parametric dimensions "+\
            "other than 1,2,3."
        )

    # Get Spline
    # - could use `Bspline.nurbs`, but that'd be unnecessary step.
    if NURBS:
        spline = NURBS()
        spline.weights = np.ones(control_points.shape[0])
    else:
        spline = BSpline()

    spline.knot_vectors = knot_vectors
    spline.control_points = control_points
    spline.degrees = degrees

    return spline
