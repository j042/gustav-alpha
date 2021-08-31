import splinelibpy as sp
import numpy as np
import logging
import copy
from .spline_ import Spline
from .nurbs import NURBS


class BSpline(Spline):

    def __init__(
        self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
    ):
        """
        BSpline.

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim, n) list
        control_points: (m, dim) list-like

        Returns
        --------
        None
        """
        super().__init__(
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
        )

    @property
    def bspline_(self,):
        """
        Property wrapper for `self.spline_`. Meant for internal use.

        Parameters
        -----------
        None

        Returns
        --------
        self.splines_: splinelibpy's one of BSpline
        """
        return self.spline_

    @bspline_.setter
    def bspline_(self, bspline):
        """
        Setter wrapper for `self.spline_`. Meant for internal use.

        Parameters
        -----------
        bspline: splinelibpy's one of BSpline

        Returns
        --------
        None
        """
        self.spline_ = bspline

    def update_c_(self,):
        """
        Updates/Init cpp spline, if it is ready to be updated.
        Checks if all the entries are filled before updating.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if not (
            self.para_dim_ != 0 
            and self.dim_ != 0 
            and self.degrees_ is not None
            and self.knot_vectors_ is not None
        ):
            logging.debug("Spline - Not enough information to update cpp "+\
                "spline. Skipping.")
            return

        if self.para_dim_ == 1:
            if self.dim_ == 2:
                self.bspline_ = sp.BSplineCurve2D()
                self.bspline_.knot_vectors = self.knot_vectors_
                self.bspline_.degrees = self.degrees_
                self.bspline_.control_points = self.control_points_
                self.whatami_ = "BSplineCurve2D"

            elif self.dim_ == 3:
                self.bspline_ = sp.BSplineCurve3D()
                self.bspline_.knot_vectors = self.knot_vectors_
                self.bspline_.degrees = self.degrees_
                self.bspline_.control_points = self.control_points_
                self.whatami_ = "BSplineCurve3D"

        elif self.para_dim_ == 2:
            if self.dim_ == 2:
                self.bspline_ = sp.BSplineSurface2D()
                self.bspline_.knot_vectors = self.knot_vectors_
                self.bspline_.degrees = self.degrees_
                self.bspline_.control_points = self.control_points_
                self.whatami_ = "BSplineSurface2D"

            elif self.dim_ == 3:
                self.bspline_ = sp.BSplineSurface3D()
                self.bspline_.knot_vectors = self.knot_vectors_
                self.bspline_.degrees = self.degrees_
                self.bspline_.control_points = self.control_points_
                self.whatami_ = "BSplineSurface3D"

        elif self.para_dim_ == 3:
            assert self.dim_ == 3, "We do not support `para_dim=3`, `dim!=3` "+\
                "splines."
            self.bspline_ = sp.BSplineSolid()
            self.bspline_.knot_vectors = self.knot_vectors_
            self.bspline_.degrees = self.degrees_
            self.bspline_.control_points = self.control_points_
            self.whatami_ = "BSplineSolid"

        self.bspline_.update_c_()

        logging.debug("Spline - Your spline is {w}.".format(w=self.whatami_))

        self.need_to_update_p_ = True
        self.update_p_()

    def update_p_(self,):
        """
        Reads cpp spline and writes it here.
        Probably get an error if cpp isn't ready for this.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if not self.need_to_update_p_:
            logging.debug("Spline - Nothing to update for python. Skipping.")
            return

        self.degrees_ = self.bspline_.degrees
        self.knot_vectors_ = self.bspline_.knot_vectors
        self.control_points_ = self.bspline_.control_points
        logging.debug("Spline - Updated python spline. CPP spline and python "+\
            "spline are now identical.")

        self.need_to_update_p_ = False

    def interpolate_curve(
        self,
        query_points,
        degree,
        centripetal=True,
        save_query=True
    ):
        """
        Interpolates BSpline Curve through query points.

        Parameters
        -----------
        query_points: (n, 2 or 3) list-like
        degree: int
        centripetal: bool
          (Optional) Default is True.
        save_query: bool
          (Optional) Default is True. Saves query points for plotting, or 
          whatever.

        Returns
        --------
        None
        """
        query_points = np.ascontiguousarray(query_points, dtype=np.double)

        self.para_dim_ = 1
        self.dim_ = query_points.shape[1]

        if self.dim_ == 2:
            self.bspline_ = sp.BSplineCurve2D()
            self.bspline_.interpolate_curve(
                points=query_points,
                degree=degree,
                centripetal=centripetal
            )
            self.whatami_ = "BSplineCurve2D"

        elif self.dim_ == 3:
            self.bspline_ = sp.BSplineCurve3D()
            self.bspline_.interpolate_curve(
                points=query_points,
                degree=degree,
                centripetal=centripetal
            )
            self.whatami_ = "BSplineCurve3D"
        else:
            raise ValueError(
                "Invalid query points dimension for curve interpolation"
            )

        logging.debug("Spline - BSpline curve interpolation complete. "+\
            "Your spline is {w}.".format(w=self.whatami_))

        if save_query:
            self.fitting_queries_ = query_points

        self.need_to_update_p_ = True
        self.update_p_()

    def interpolate_surface(
        self,
        query_points,
        size_u,
        size_v,
        degree_u,
        degree_v,
        centripetal=True,
        reorganize=True,
        save_query=True,
    ):
        """
        Interpolates BSpline Surface through query points.

        Parameters
        -----------
        query_points: (n, 2 or 3) list-like
        size_u: int
        size_v: int
        degree_u: int
        degree_v: int
        centripetal: bool
          (Optional) Default is True.
        reorganize: bool
          (Optional) Default is False. Reorganize control points, assuming they
          are listed v-direction first, along u-direction.
        save_query: bool
          (Optional) Default is True. Saves query points for plotting, or 
          whatever.
         
        Returns
        --------
        None
        """
        query_points = np.ascontiguousarray(query_points, dtype=np.double)

        self.para_dim_ = 2
        self.dim_ = query_points.shape[1]

        if self.dim_ == 2:
            self.bspline_ = sp.BSplineSurface2D()
            self.bspline_.interpolate_surface(
                points=query_points,
                size_u=size_u,
                size_v=size_v,
                degree_u=degree_u,
                degree_v=degree_v,
                centripetal=centripetal,
            )
            self.whatami_ = "BSplineSurface2D"

        elif self.dim_ == 3:
            self.bspline_ = sp.BSplineSurface3D()
            self.bspline_.interpolate_surface(
                points=query_points,
                size_u=size_u,
                size_v=size_v,
                degree_u=degree_u,
                degree_v=degree_v,
                centripetal=centripetal,
            )
            self.whatami_ = "BSplineSurface3D"

        logging.debug("Spline - BSpline surface interpolation complete. "+\
            "Your spline is {w}.".format(w=self.whatami_))

        self.need_to_update_p_ = True
        self.update_p_()

        # Reorganize control points.
        if reorganize:
            ind = [v + size_v * u for v in range(size_v) for u in range(size_u)]
            self.control_points = self.control_points_[ind] 

        if save_query:
            self.fitting_queries_ = query_points

    @property
    def nurbs(self,):
        """
        Returns NURBS version of current BSpline by defining all the weights as 
        1.

        Parameters
        -----------
        None

        Returns
        --------
        same_nurbs: NURBS
        """
        same_nurbs = NURBS()
        same_nurbs.degrees = copy.deepcopy(self.degrees)
        same_nurbs.knot_vectors = copy.deepcopy(self.knot_vectors)
        same_nurbs.control_points = copy.deepcopy(self.control_points)
        same_nurbs.weights = np.ones(self.control_points.shape[0])

        return same_nurbs

    def copy(self,):
        """
        Returns freshly initialized BSpline of self.

        Parameters
        -----------
        None

        Returns
        --------
        new_bspline: `BSpline`
        """
        new_bspline = BSpline()
        new_bspline.degrees = copy.deepcopy(self.degrees)
        new_bspline.knot_vectors = copy.deepcopy(self.knot_vectors)
        new_bspline.control_points = copy.deepcopy(self.control_points)

        return new_bspline
