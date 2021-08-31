import splinelibpy as sp
import numpy as np
import logging
import copy
from .spline_ import Spline

class NURBS(Spline):

    def __init__(self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
        weights=None,
    ):
        """
        NURBS.

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim, n) list
        control_points: (m, dim) list-like
        weights: (m,) list-like

        Returns
        --------
        None
        """
        super().__init__(
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
            weights=weights,
        )

    @property
    def nurbs_(self,):
        """
        Property wrapper for `self.spline_`. Meant for interal use.

        Parameters
        -----------
        None

        Returns
        --------
        self.splines_: splinelibpy's one of NURBS
        """
        return self.spline_

    @nurbs_.setter
    def nurbs_(self, nurbs):
        """
        Setter wrapper for `self.spline_`. Meant for internal use.

        Parameters
        -----------
        nurbs: splinelibpy's one of NURBS

        Returns
        --------
        None
        """
        self.spline_ = nurbs

    @property
    def weights(self,):
        """
        Returns weights.

        Parameters
        -----------
        None

        Returns
        --------
        self.weights_: (n, 1) list-like
        """
        return self.weights_

    @weights.setter
    def weights(self, weights):
        """
        Set weights.

        Parameters
        -----------
        weights: (n,) list-like

        Returns
        --------
        None
        """
        if weights is None:
            self.weights_ = None
            return

        weights = np.ascontiguousarray(weights, dtype=np.double).reshape(-1,1)
        
        self.weights_ = weights
        logging.debug("Spline - {n_cps} Weights set.".format(
            n_cps=self.weights_.shape[0]))

        self.update_c_()

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
            and self.weights_ is not None
        ):
            logging.debug("Spline - Not enough information to update cpp "+\
                "spline. Skipping.")
            return

        if self.para_dim_ == 1:
            if self.dim_ == 2:
                self.nurbs_ = sp.NurbsCurve2D()
                self.nurbs_.knot_vectors = self.knot_vectors_
                self.nurbs_.degrees = self.degrees_
                self.nurbs_.control_points = self.control_points_
                self.nurbs_.weights = self.weights_
                self.whatami_ = "NurbsCurve2D"

            elif self.dim_ == 3:
                self.nurbs_ = sp.NurbsCurve3D()
                self.nurbs_.knot_vectors = self.knot_vectors_
                self.nurbs_.degrees = self.degrees_
                self.nurbs_.control_points = self.control_points_
                self.nurbs_.weights = self.weights_
                self.whatami_ = "NurbsCurve3D"

        elif self.para_dim_ == 2:
            if self.dim_ == 2:
                self.nurbs_ = sp.NurbsSurface2D()
                self.nurbs_.knot_vectors = self.knot_vectors_
                self.nurbs_.degrees = self.degrees_
                self.nurbs_.control_points = self.control_points_
                self.nurbs_.weights = self.weights_
                self.whatami_ = "NurbsSurface2D"

            elif self.dim_ == 3:
                self.nurbs_ = sp.NurbsSurface3D()
                self.nurbs_.knot_vectors = self.knot_vectors_
                self.nurbs_.degrees = self.degrees_
                self.nurbs_.control_points = self.control_points_
                self.nurbs_.weights = self.weights_
                self.whatami_ = "NurbsSurface3D"

        elif self.para_dim_ == 3:
            assert self.dim_ == 3, "We do not support `para_dim=3`, `dim!=3` "+\
                "splines."
            self.nurbs_ = sp.NurbsSolid()
            self.nurbs_.knot_vectors = self.knot_vectors_
            self.nurbs_.degrees = self.degrees_
            self.nurbs_.control_points = self.control_points_
            self.nurbs_.weights = self.weights_
            self.whatami_ = "NurbsSolid"

        self.nurbs_.update_c_()

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

        self.degrees_ = self.nurbs_.degrees
        self.knot_vectors_ = self.nurbs_.knot_vectors
        self.control_points_ = self.nurbs_.control_points
        self.weights_ = self.nurbs_.weights
        logging.debug("Spline - Updated python spline. CPP spline and python "+\
            "spline are now identical.")

        self.need_to_update_p_ = False

    def copy(self,):
        """
        Returns freshly initialized Nurbs of self.

        Parameters
        -----------
        None

        Returns
        --------
        new_nurbs: `Nurbs`
        """
        new_nurbs = Nurbs()
        new_nurbs.degrees = copy.deepcopy(self.degrees)
        new_nurbs.knot_vectors = copy.deepcopy(self.knot_vectors)
        new_nurbs.control_points = copy.deepcopy(self.control_points)
        new_nurbs.weights = copy.deepcopy(self.weights)

        return new_nurbs
