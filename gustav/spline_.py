import abc
import splinelibpy as sp
import numpy as np
import logging
import copy
import os
import itertools
from .mesh import Mesh
from . import utils
from .export.utils import check_and_makedirs 

class Spline(abc.ABC):

    def __init__(self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
        weights=None,
    ):
        """
        Spline.

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim, n) list
        control_points: (m, dim) list-like
        weights: (m,) list-like

        Attributes
        -----------
        para_dim_: int
        dim_: int
        degrees_: (para_dim_,) np.ndarray
        knot_vectors_: (para_dim_,) list
        control_points_: (n, dim_) np.ndarray
        weights_: (n,) np.ndarray
        spline_: `splinelibpy.BSpline` or `splinelibpy.NURBS`
        need_to_update_p_: bool
        whatami_: str
        sassy_counter: int
        fitting_queries_: (m, dim_) np.ndarray
          Only used by `BSpline`.
        unique_knots_: (para_dim_,) list 

        Returns
        --------
        None
        """
        self.para_dim_ = 0
        self.dim_ = 0
        self.degrees_ = None
        self.knot_vectors_ = None
        self.control_points_ = None
        self.weights_ = None
        self.spline_ = None
        self.fitting_queries_ = None
        self.unique_knots_ = None

        self.degrees = degrees
        self.knot_vectors = knot_vectors
        self.control_points = control_points

        # Only for NURBS
        if weights is not None:
            self.weights = weights

        self.need_to_update_p_ = False
        self.whatami_ = "Nothing"
        self.sassy_counter = 0

    @property
    def whatami(self):
        """
        Answers a deep philosophical question of "what am i?"

        Parameters
        -----------
        None

        Returns
        --------
        self.whatami_: what_you_are
        """
        return self.whatami_

    @whatami.setter
    def whatami(self, how_dare_you):
        """
        Sassily refuse to tell me what i am.
        Totally unnecessary.

        Parameters
        -----------
        how_dare_you: seriously

        Returns
        --------
        None
        """
        logging.warning("Spline - Am I {h}?".format(h=how_dare_you))

        if self.sassy_counter > 1:
            raise Exception(
                "Oh no you didn't! I am {i}.".format(i=self.whatami_)
            )
        
        logging.warning("Spline - Excuse me, you cannot tell me what I am.")
        logging.warning("Spline - I am {i}.".format(i=self.whatami_))

        self.sassy_counter += 1

    @property
    def parametric_dim(self):
        """
        Returns parametric dimension.

        Parameters
        -----------
        None

        Returns
        --------
        self.para_dim_: int
        """
        return self.para_dim_

    @property
    def physical_dim(self):
        """
        Returns physical dimension.

        Parameters
        -----------
        None

        Returns
        --------
        self.dim_: int
        """
        return self.dim_

    @property
    def degrees(self):
        """
        Returns Degrees.

        Parameters
        -----------
        None

        Returns
        --------
        self.degrees_: list-like
        """
        return self.degrees_

    @degrees.setter
    def degrees(self, degrees):
        """
        Sets Degrees.

        Parameters
        -----------
        degrees: list-like

        Returns
        --------
        None
        """
        if degrees is None:
            self.degrees_ = None
            return

        if not isinstance(degrees, np.ndarray):
            degrees = np.ascontiguousarray(degrees)
        degrees = degrees.flatten()

        if self.para_dim_ == 0:
            self.para_dim_ = len(degrees)
        else:
            assert self.para_dim_ == len(degrees), "Current Spline parametric"+\
                "dimension does not match input dimension."

        self.degrees_ = degrees.astype(np.int32)
        logging.debug("Spline - Degrees set: {d}".format(d=self.degrees_))

        self.update_c_()

    @property
    def knot_vectors(self):
        """
        Returns knot vectors.

        Parameters
        -----------
        None

        Returns
        --------
        self.knot_vectors: list-like
        """
        return self.knot_vectors_

    @knot_vectors.setter
    def knot_vectors(self, knot_vectors):
        """
        Set knot vectors.

        Parameters
        -----------
        knot_vectors: list

        Returns
        --------
        None
        """
        if knot_vectors is None:
            self.knot_vectors_ = None
            return

        if self.para_dim_ == 0:
            self.para_dim_ = len(knot_vectors)
        else:
            assert int(self.para_dim_) == int(len(knot_vectors)), "Current "+\
                "Spline parametric dimension does not match input dimension."

        self.knot_vectors_ = knot_vectors
        logging.debug("Spline - Knot vectors set:")
        for i in range(len(self.knot_vectors_)):
            logging.debug("Spline -   " + str(i) + ". knot vector length:"+\
                " {kv}".format(kv=len(self.knot_vectors_[i])))

        self.update_c_()

    @property
    def unique_knots(self,):
        """
        Returns unique knots.
        TODO: Use cpp-function!

        Parameters
        -----------
        None

        Returns
        self.unique_knots_: (para_dim,) list
        """
        logging.warning("Spline - Computing unique knots using `np.unique`.")
        self.unique_knots_ = []
        for k in self.knot_vectors:
            self.unique_knots_.append(np.unique(k).tolist())

        return self.unique_knots_

    @property
    def control_points(self,):
        """
        Returns control points.

        Parameters
        -----------
        None

        Returns
        --------
        self.control_points_: (n, dim) list-like
        """
        return self.control_points_

    @control_points.setter
    def control_points(self, control_points):
        """
        Set control points.

        Parameters
        -----------
        control_points: (n, dim) list-like

        Returns
        --------
        None
        """
        if control_points is None:
            self.knot_vectors_ = None
            return

        control_points = np.ascontiguousarray(control_points, dtype=np.double)

        if self.dim_ == 0:
            self.dim_ = control_points.shape[1]
        else:
             assert self.dim_ == control_points.shape[1], "Current Spline "+\
                "physical dimension does not match input dimension."

        self.control_points_ = control_points
        logging.debug("Spline - {n_cps} Control points set.".format(
            n_cps=self.control_points_.shape[0]))

        self.update_c_()

    def lexsort_control_points(self, order):
        """
        Lexsorts control points.
        SplineLib (probably) takes control points index orders of:
            for w in range(ws):
                for v in range(vs):
                    for u in range(us):
                        # control point sequence.
        Not always the solution.

        Parameters
        -----------
        order: list

        Returns
        --------
        None
        """
        ind = np.lexsort([self.control_points_[:,i] for i in order])
        logging.debug("Spline - Reorganizing control points "+\
            "({l})".format(l=order))
        self.control_points = self.control_points_[ind]

    @abc.abstractmethod
    def update_c_(self,):
        """
        Updates/Init cpp spline, if it is ready to be updated.
        Checks if all the entries are filled before updating.

        This needs to be separately implemented, as BSpline and NURBS have 
        different update process. 

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        pass

    @abc.abstractmethod
    def update_p_(self,):
        """
        Reads cpp spline and writes it here.
        Probably get an error if cpp isn't ready for this.

        This needs to be separately implemented, as BSpline and NURBS have 
        different update process. 

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        pass

    def evaluate(self, queries):
        """
        Evaluates spline.

        Parameters
        -----------
        queries: (n, para_dim) list-like

        Returns
        --------
        results: (n, dim) np.ndarray
        """
        queries = np.ascontiguousarray(queries, dtype=np.double)

        assert int(queries.shape[1]) == int(self.para_dim_), "Queries does "+\
            "not match current pametric dimension."
        logging.debug("Spline - Evaluating spline...")

        return self.spline_.evaluate(queries=queries)

    def derivative(self, queries, orders):
        """
        Evaluates derivatives of spline.

        Parameters
        -----------
        queries: (n, para_dim) list-like
        orders: (para_dim,) list-like

        Returns
        --------
        results: (n, dim) np.ndarray
        """
        queries = np.ascontiguousarray(queries, dtype=np.double)
        orders = np.ascontiguousarray(orders, dtype=np.double)
        assert int(queries.shape[1]) == int(self.para_dim_), "Queries does "+\
            "not match current parametric dimension."
        assert len(orders) == int(self.para_dim_), "Query "+\
            "derivative orders does not match current parametric dimension."

        logging.debug("Spline - Evaluating derivatives of the spline...")

        return self.spline_.derivative(queries=queries, orders=orders)

    def insert_knots(self, parametric_dimension, knots):
        """
        Inserts knots. 

        Parameters
        -----------
        parametric_dimension: int
        knots: list or float

        Returns
        --------
        None
        """
        assert parametric_dimension < self.para_dim_,\
            "Invalid parametric dimension to insert knots."


        if isinstance(knots, float):
            knots = [knots]
        elif isinstance(knots, np.ndarray):
            knots = knots.tolist()

        if not isinstance(knots, list):
            raise TypeError(
                "We couldn't convert input to `list`. Please pass `list`."
            )

        assert max(knots) < max(self.knot_vectors[parametric_dimension]),\
            "One of the query knots not in valid knot range. (Too big)"

        assert min(knots) > min(self.knot_vectors[parametric_dimension]),\
            "One of the query knots not in valid knot range. (Too small)"

        self.spline_.insert_knots(int(parametric_dimension), knots)

        logging.debug("Spline - Inserted {nk} knot(s).".format(nk=len(knots)))

        self.need_to_update_p_ = True
        self.update_p_()

    def remove_knots(self, parametric_dimension, knots, tolerance=1e-8):
        """
        Tries to removes knots. If you've compiled `splinelibpy` in `Debug`
        and your removal request is not "accepted", you will get an error.
        See the comments for `Nurbs::remove_knots` @ 
        `splinelibpy/src/nurbs.hpp` for more info.

        Parameters
        -----------
        parametric_dimension: int
        knots: list or float
        tolderance: float

        Returns
        --------
        None
        """
        assert parametric_dimension < self.para_dim_,\
            "Invalid parametric dimension to insert knots."

        if isinstance(knots, float):
            knots = [knots]
        elif isinstance(knots, np.ndarray):
            knots = knots.tolist()

        if not isinstance(knots, list):
            raise TypeError(
                "We couldn't convert input to `list`. Please pass `list`."
            )

        assert max(knots) < max(self.knot_vectors[parametric_dimension]),\
            "One of the query knots not in valid knot range. (Too big)"

        assert min(knots) > min(self.knot_vectors[parametric_dimension]),\
            "One of the query knots not in valid knot range. (Too small)"

        total_knots_before = len(self.knot_vectors_[int(parametric_dimension)])

        self.spline_.remove_knots(int(parametric_dimension), knots, tolerance)

        self.need_to_update_p_ = True
        self.update_p_()

        logging.debug("Spline - Tried to remove {nk} knot(s).".format(
            nk=len(knots)))
        logging.debug(
            "Spline - Actually removed {nk} knot(s).".format(
                nk=(
                    total_knots_before
                    - len(self.knot_vectors_[int(parametric_dimension)])
                )
            )
        )

    def elevate_degree(self, parametric_dimension):
        """
        Elevate degree.

        Parameters
        -----------
        parametric_dimension: int

        Returns
        --------
        None
        """
        assert parametric_dimension < self.para_dim_,\
            "Invalid parametric dimension to elevate degrees."

        self.spline_.elevate_degree(parametric_dimension)
        logging.debug("Spline - Elevated {p}.-dim. degree of the spline.".format(
            p=parametric_dimension))

        self.need_to_update_p_ = True
        self.update_p_()

    def reduce_degree(self, parametric_dimension, tolerance=1e-8):
        """
        Tries to reduce degree.

        Parameters
        -----------
        parametric_dimension: int
        tolerance: float

        Returns
        --------
        reduced: bool
        """
        assert parametric_dimension < self.para_dim_,\
            "Invalid parametric dimension to elevate degrees."

        reduced = self.spline_.reduce_degree(parametric_dimension, tolerance)
        logging.debug("Spline - Tried to reduce "+\
            "{p}.-dim. degree of the spline.".format(p=parametric_dimension))

        if reduced:
            logging.debug(
                "Spline - Successfully reduced {p}.-dim. degree".format(
                    p=parametric_dimension
                )
            )

        else:
            logging.debug(
                "Spline - Could not reduce {p}.-dim. degree".format(
                    p=parametric_dimension
                )
            )

        self.need_to_update_p_ = True
        self.update_p_()

        return reduced

    def sample(self, query_resolutions, built_in=True):
        """
        Uniformly sample along each parametric dimensions from spline.

        Parameters
        -----------
        query_resolutions: (n, m) list-like
        built_in: bool
          (Optional) Default is True. If false, use `evaluate` functions to
          sample points. 

        Returns
        --------
        results: (n*m, dim) list-like
          Fix first order, list second order, ...
        """
        query_resolutions = np.ascontiguousarray(
            query_resolutions,
            dtype=np.int32
        ).flatten()

        assert int(query_resolutions.shape[0]) == int(self.para_dim_),\
            "Query resolutions does not match current parametric dimension."

        is_one_or_less= np.array([int(i) <= 1 for i in query_resolutions])
        if is_one_or_less.any():
            logging.debug("Spline - You cannot sample less than 2 points "+\
                "per each parametric dimension.")
            logging.debug("Spline - Applying minimum sampling resolution 2.")

            query_resolutions[is_one_or_less] = 2

        logging.debug("Spline - Sampling {t} points from our spline.".format(
            t=np.product(query_resolutions)))

        if built_in or len(query_resolutions) == 1:
            return self.spline_.sample(query_resolutions.astype(np.int32))

        else:
            return self.evaluate(
                utils.raster_points(query_resolutions, [0,1], lexsort=[0,1])
            )

    def show(
        self,
        control_points=True,
        knots=True,
        resolutions=100,
        quads=True,
        show_queries=True,
        offscreen=False, # <- Implies that it returns plot and objects
        fname=None,
        dashed_line=False,
        surface_only=True,
        colorful_elements=False,
    ):
        """
        Sample, form a mesh, then show.

        Parameters
        -----------
        control_points: bool
          (Optional) Default is True. Show control points.
        knots: bool or float
          Show knots. Only for para_dim == 1, this can be float.
          If it is float, it is then used as size for `vedo.Box`.
        resolutions: int or list
          (Optional) Default is 100.
        quads: bool
          (Optional) Default is True. If false, triangle.
        offscreen: bool
          (Optional) Default is False. If true, returns `vedo.Plotter` and a
          list of things that should be on the plot.
        fname: str
          (Optional) Default is None. If specified, saves plot.
        surface_only: bool
        colorful_elements: bool
          Give each element a different color. Currently only implemented for
          lines.

        Returns
        --------
        plot: `vedo.Plotter`
          (Optional) Only if `offscreen=True`.
        things_to_show: list
          (Optional) Only if `offscreen=True`. List of vedo objects.
        """
        from vedo import show, Points, colors, Line, Box

        vedo_colors = [*colors.colors.keys()]
        vedo_colors = [c for c in vedo_colors if not "white" in c]
        things_to_show = []

        if self.para_dim_ == 1:
            if colorful_elements:
                # TODO: could be cool to sample from a given range
                #   in parametric space -> add `param_range` in `sample()`
                for i in range(int(len(self.unique_knots[0]) - 1)):
                    things_to_show.append(
                        Line(
                            self.evaluate(
                                np.linspace(
                                    self.unique_knots[0][i],
                                    self.unique_knots[0][i+1],
                                    resolutions,
                                ).reshape(-1,1)
                            )
                        ).color(np.random.choice(vedo_colors))
                        .lw(6)
                    )

            else:
                things_to_show.append(
                    self.line_(resolutions, dashed_line=False)
                )

            if knots:
                box_size = knots if not isinstance(knots, bool) else 0.03

                for uk in self.unique_knots[0]:
                    pos = self.evaluate([[uk]])[0].tolist()
                    if self.dim_ == 2:
                        pos.append(0)

                    things_to_show.append(
                        Box(
                            pos=pos,
                            length=box_size,
                            width=box_size,
                            height=box_size,
                            c="black"
                        )
                    )

        elif self.para_dim_ == 2:
            if isinstance(resolutions, int):
                resolutions = [resolutions for _ in range(self.para_dim_)]

            things_to_show.append(
                self.mesh_(
                    resolutions=resolutions,
                    quads=quads,
                    mode="vedo"
                ).color("green").lighting("glossy")
            )
            if knots:
                for u in self.knot_vectors[0]:
                    things_to_show.append(
                        self.line_(
                            resolution=resolutions[0],
                            raw=False,
                            extract=[0, u],
                            dashed_line=False,
                        )
                    )

                for v in self.knot_vectors[1]:
                    things_to_show.append(
                        self.line_(
                            resolution=resolutions[1],
                            raw=False,
                            extract=[1, v],
                            dashed_line=False,
                        )
                    )

            else:
                # Show just edges
                things_to_show.extend(
                    [
                        self.line_(
                            resolution=resolutions[0],
                            raw=False,
                            extract=[0, self.knot_vectors[0][0]],
                            dashed_line=False,
                        ),
                        self.line_(
                            resolution=resolutions[0],
                            raw=False,
                            extract=[0, self.knot_vectors[0][-1]],
                            dashed_line=False,
                        ),
                        self.line_(
                            resolution=resolutions[1],
                            raw=False,
                            extract=[1, self.knot_vectors[1][0]],
                            dashed_line=False,
                        ),
                        self.line_(
                            resolution=resolutions[1],
                            raw=False,
                            extract=[1, self.knot_vectors[1][-1]],
                            dashed_line=False,
                        ),
                    ]
                )

        elif self.para_dim_ == 3:
            if isinstance(resolutions, int):
                resolutions = [resolutions for _ in range(self.para_dim_)]
            resolutions = np.asarray(resolutions)

            things_to_show.append(
                self.mesh_(
                    resolutions=resolutions,
                    surface_only=surface_only,
                    mode="vedo",
                ).color("green").lighting("glossy")
            )

            things_to_show.extend(
                self.lines_(
                    resolution=resolutions,
                    outlines=not knots
                )
            )



        if control_points:
            c_points, c_lines = self.control_mesh_(
                points_and_lines=True,
                dashed_line=dashed_line,
            )
            things_to_show.extend([c_points, *c_lines])

        if show_queries and self.fitting_queries_ is not None:
            things_to_show.append(Points(self.fitting_queries_, c="blue", r=15))

        # TODO: one plot obj.
        if not offscreen:
            show(things_to_show,).close()
            if fname is None:
                return


        if fname is not None:
            plot = show(things_to_show, interactive=False, offscreen=True)
            plot.screenshot(fname)
            plot.close()

        return things_to_show

        

    def mesh_(self, resolutions=100, quads=True, mode=None, surface_only=True):
        """
        Returns spline mesh.

        Warning: Faces of quad meshes are not guaranteed to be coplanar.
          It is okay for visualization using `vedo`. If this is an issue,
          set quads=False, and get triangles.

        Parameters
        -----------
        resolutions: int or list
        quad: bool
        mode: str
          (Optional) options are <"vedo" | "trimesh">.
          If unspecified, regular internal mesh.
        surface_only: bool
          Only for volumes since sampling takes a long long time.

        Returns
        --------
        spline_mesh: `Mesh`
        """
        if isinstance(resolutions, int):
            resolutions = [resolutions for _ in range(self.para_dim_)]

        if self.para_dim_ == 2:
            # Spline Mesh
            physical_points = self.sample(resolutions)
            spline_faces = utils.make_quad_faces(resolutions)

            if not quads:
                spline_faces = utils.diagonalize_quad(spline_faces)

            spline_mesh = Mesh(
                vertices=physical_points,
                faces=spline_faces
            )

        elif self.para_dim_ == 3:
            if surface_only:
                # Spline to surfaces
                vertices = []
                faces = []
                offset = 0
                for i in range(self.para_dim_):
                    extract = i

                    # Get extracting dimension
                    extract_along = [0, 1, 2] 
                    extract_along.pop(extract)

                    # Extract range
                    extract_range = [
                        [min(self.knot_vectors[extract_along[0]]),
                         max(self.knot_vectors[extract_along[0]]),],
                        [min(self.knot_vectors[extract_along[1]]),
                         max(self.knot_vectors[extract_along[1]]),],
                    ]

                    extract_list = [
                        min(self.knot_vectors[extract]),
                        max(self.knot_vectors[extract]),
                    ]
                    extract_list_0 = np.linspace(
                        extract_range[0][0],
                        extract_range[0][1],
                        resolutions[extract_along[0]]
                    )
                    extract_list_1 = np.linspace(
                        extract_range[1][0],
                        extract_range[1][1],
                        resolutions[extract_along[1]]
                    )
                    surface_point_queries = list(
                        itertools.product(
                            extract_list, extract_list_0, extract_list_1
                        )
                    )
                    surface_point_queries = np.ascontiguousarray(
                        surface_point_queries,
                        dtype=np.double,
                    )
                    surface_point_queries = surface_point_queries[
                        :,
                        np.argsort(
                            [extract, extract_along[0], extract_along[1]]
                        )
                    ]
                    vertices.append(
                        self.evaluate(
                            surface_point_queries[
                                :int(surface_point_queries.shape[0] / 2)
                            ]
                        )
                    )

                    if len(vertices) > 1:
                        offset += vertices[-1].shape[0]

                    tmp_faces = utils.make_quad_faces(
                        [
                            resolutions[extract_along[0]],
                            resolutions[extract_along[1]],
                        ]
                    )
 
                    faces.append(tmp_faces + int(offset))

                    vertices.append(
                        self.evaluate(
                            surface_point_queries[
                                int(surface_point_queries.shape[0] / 2):
                            ]
                        )
                    )

                    offset += vertices[-1].shape[0]

                    faces.append(tmp_faces + int(offset))

                spline_mesh = Mesh(
                    vertices=np.vstack(vertices),
                    faces=np.vstack(faces)
                )

            else:
                # Spline Hexa
                physical_points = self.sample(resolutions)
                spline_elements = utils.make_hexa_elements(resolutions)

                spline_mesh = Mesh(
                    vertices=physical_points,
                    elements=spline_elements
                )

        else:
            logging.debug("Spline - Mesh is only supported for 2D parametric "+\
                "spaces. Skippping.")

        if mode == "vedo":
            spline_mesh = spline_mesh.vedo_mesh
        elif mode == "trimesh":
            spline_mesh = spline_mesh.trimesh_mesh
        else:
            logging.debug("Spline - `mode` is either None or invalid. "+\
                "Returning `gustav.Mesh`.")

        return spline_mesh

    def line_(self, resolution, raw=False, extract=None, dashed_line=False):
        """
        Returns line.

        Parameters
        -----------
        resolution: int
        raw: bool
          (Optional) Default is False. Returns vertices and edges.
        extract: list or tuple
          (Optional) Default is None.
          ex) [0, .4] -> [parametric_dim, knot]
          Extracts line from a surface.

        Returns
        --------
        lines: list
          list of vedo.Line
        physical_points: (n, dim) np.ndarray
          (Optional) Only if `raw=True`.
        edges: (m, 2) np.ndarray
          (Optional) Only if `raw=True`.
        """
        if self.para_dim_ == 1:
            if not raw:
                from vedo import Points, Line, DashedLine

                physical_points = Points(self.sample(resolution))
                if not dashed_line:
                    lines = Line(physical_points, closed=False, c="black", lw=6)

                else:
                    lines = DashedLine(
                        physical_points,
                        closed=False,
                        c="black",
                        lw=6
                    )

                return lines

            else:
                physical_points = self.sample(resolution)
                edges = utils.closed_loop_index_train(physical_points.shape[1])

                return physical_points, edges

        elif self.para_dim_ == 2:
            if extract is not None:
                # Get non-extracting dimension
                extract_along = [0,1] 
                extract_along.pop(extract[0])

                # Extract range
                extract_range = [
                    min(self.knot_vectors[extract_along[0]]),
                    max(self.knot_vectors[extract_along[0]]),
                ]
                queries = np.zeros((resolution, 2), dtype=np.double)
                queries[:, extract[0]] = extract[1]
                queries[:, extract_along[0]] = np.linspace(
                    extract_range[0],
                    extract_range[1],
                    resolution
                )

                # Extract
                physical_points = self.evaluate(queries)

            if not raw:
                from vedo import Points, Line, DashedLine

                physical_points = Points(physical_points)
                if not dashed_line:
                    lines = Line(
                        physical_points,
                        closed=False,
                        c="black",
                        lw=2
                    )

                else:
                    lines = DashedLine(
                        physical_points,
                        closed=False,
                        c="black",
                        lw=2
                    )


                return lines

            else:
                edges = utils.open_loop_index_train(
                    physical_points.shape[0]
                )

                return physical_points, edges

    def lines_(self, resolution, outlines=False):
        """
        Returns lines. This is meant to be only for volume visualization.
 
        Parameters
        -----------
        resolution: int or list
        outlines: bool

        Returns
        --------
        lines: list
          list of vedo.Line
        """
        if self.para_dim_ != 3:
            raise ValueError(
                "Sorry, this function (lines_) is only for Solids."
            )

        if isinstance(resolution, list):
            raise ValueError("For para-dim=3, line extraction needs a "+\
                "list of resolutions")

        from vedo import Points, Line

        # Fill lines
        lines_list = []
        for i in range(self.para_dim_):
            extract = [i]

            # Get extracting dimension
            extract_along = [0, 1, 2] 
            extract_along.pop(extract[0])

            # Extract range
            extract_range = [
                [min(self.knot_vectors[extract_along[0]]),
                 max(self.knot_vectors[extract_along[0]]),],
                [min(self.knot_vectors[extract_along[1]]),
                 max(self.knot_vectors[extract_along[1]]),],
            ]

            # Outlines?
            if not outlines:
                last_line_set_queries = list(
                    itertools.product(
                        self.knot_vectors[extract_along[0]],
                        self.knot_vectors[extract_along[1]],
                     )
                )

            else:
                last_line_set_queries = list(
                    itertools.product(
                        extract_range[0],
                        extract_range[1],
                    )
                )

            # Sample lines
            for i, ks in enumerate(last_line_set_queries):
                queries = np.zeros(
                    (resolution[extract[0]], 3),
                    dtype=np.double
                )
                queries[:, extract[0]] = np.linspace(
                    min(self.knot_vectors[extract[0]]),
                    max(self.knot_vectors[extract[0]]),
                    resolution[extract[0]]
                )
                queries[:, extract_along[0]] = ks[0]
                queries[:, extract_along[1]] = ks[1]
                lines_list.append(
                    Line(
                        Points(self.evaluate(queries)),
                        closed=False,
                        c="black",
                        lw=2
                    )
                )

        return lines_list


    def control_mesh_(
        self,
        mode=None,
        points_and_lines=False,
        raw=False,
        dashed_line=True,
    ):
        """
        Returns control mesh.

        Parameters
        -----------
        mode: str
          (Optional) options are <"vedo" | "trimesh">.
          If unspecified, regular internal mesh.
          trimesh, will be triangular.
        points_and_lines: bool
        raw: bool

        Returns
        --------
        control_mesh: `Mesh`

        """
        from vedo import Points, Line, DashedLine

        # Formulate points
        if not raw:
            c_points = Points(self.control_points, c="red", r=10)

        if self.para_dim_ == 1:
            if not raw:
                c_lines = [DashedLine(c_points, closed=False, c="red", lw=3)]
            else:
                return (
                    self.control_points,
                    utils.closed_loop_index_train(self.control_points.shape[1]),
                )
            
        elif self.para_dim_ == 2:
            control_mesh_dims = []
            for i in range(self.para_dim_):
                control_mesh_dims.append(
                    len(self.knot_vectors[i]) - self.degrees_[i] - 1
                )

            cp_faces = utils.make_quad_faces(control_mesh_dims)
            control_mesh = Mesh(
                vertices=self.control_points_,
                faces=cp_faces,
            )

            if mode == "vedo" and not points_and_lines:
                control_mesh = control_mesh.vedo_mesh

            elif mode == "trimesh" and not points_and_lines:
                control_mesh = control_mesh.trimesh_mesh

            elif points_and_lines:
                pass

            else:
                logging.debug("Spline - `mode` is either None or invalid. "+\
                    "Returning `gustav.Mesh`.")

            if not points_and_lines:
                return control_mesh

            if not raw:
                c_lines = []
                for ue in control_mesh.unique_edges:
                    if dashed_line:
                        c_lines.append(
                            DashedLine( # <- too many objects and could be slow
                                p0=control_mesh.vertices[ue[0]],
                                p1=control_mesh.vertices[ue[1]],
                                c="red",
                                alpha=.8,
                                lw=3,
                            )
                        )

                    else:
                        c_lines.append(
                            Line(
                                p0=control_mesh.vertices[ue[0]],
                                p1=control_mesh.vertices[ue[1]],
                                c="red",
                                alpha=.8,
                                lw=3,
                            )
                        )


            else:
                c_lines = control_mesh.unique_edges
                c_points = control_mesh.vertices

        elif self.para_dim_ == 3:
            control_mesh_dims = []
            for i in range(self.para_dim_):
                control_mesh_dims.append(
                    len(self.knot_vectors[i]) - self.degrees_[i] - 1
                )

            cp_elements = utils.make_hexa_elements(control_mesh_dims)
            control_mesh = Mesh(
                vertices=self.control_points_,
                elements=cp_elements,
            )

            if mode == "vedo" and not points_and_lines:
                control_mesh = control_mesh.vedo_mesh

            elif mode == "trimesh" and not points_and_lines:
                control_mesh = control_mesh.trimesh_mesh

            elif points_and_lines:
                pass

            else:
                logging.debug("Spline - `mode` is either None or invalid. "+\
                    "Returning `gustav.Mesh`.")

            if not points_and_lines:
                return control_mesh

            if not raw:
                c_lines = []
                for ue in control_mesh.unique_edges:
                    if dashed_line:
                        c_lines.append(
                            DashedLine( # <- too many objects and could be slow
                                p0=control_mesh.vertices[ue[0]],
                                p1=control_mesh.vertices[ue[1]],
                                c="red",
                                alpha=.8,
                                lw=3,
                            )
                        )

                    else:
                        c_lines.append(
                            Line(
                                p0=control_mesh.vertices[ue[0]],
                                p1=control_mesh.vertices[ue[1]],
                                c="red",
                                alpha=.8,
                                lw=3,
                            )
                        )

            else:
                c_lines = control_mesh.unique_edges
                c_points = control_mesh.vertices

        return c_points, c_lines

    def export(self, fname):
        """
        Export spline. Please be aware of the limits of `.iges`

        Parameters
        -----------
        fname: str

        Returns
        --------
        None
        """
        self.update_c_()

        fname = str(fname)

        check_and_makedirs(fname)

        ext = os.path.splitext(fname)[1]
    
        if ext == ".iges":
            self.spline_.write_iges(fname)

        elif ext == ".xml":
            self.spline_.write_xml(fname)

        elif ext == ".itd":
            self.spline_.write_irit(fname)

        else:
            raise Exception(
                "We can only export < .iges | .xml | .itd > spline files"
            )

        logging.info("Spline - Exported current spline as {f}.".format(f=fname))

    @abc.abstractmethod
    def copy(self,):
        """
        Returns freshly initialized Spline of self.

        Needs to be implemented separately

        Parameters
        -----------
        None

        Returns
        --------
        new_spline: `Spline`
        """
        pass
