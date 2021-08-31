from .load import load_mixd, load_splines
import logging

class FreeFormDeformation:

    def __init__(self):
        """
        FreeFormDeformation.

        Parameters
        ----------
        None

        Attributes
        ----------
        deformed_spline : Spline
        input_mesh : Mesh
        deformed_mesh : Mesh

        undeformed_spline_ : Spline
        unit_mesh_ : Mesh
        deformed_unit_mesh_ : Mesh

        offset_ : (dim,) numpy.ndarray
        scale_ : (dim,) numpy.ndarray

        Returns
        -------
        None
        """
        self.deformed_spline = None
        self.input_mesh = None
        self.deformed_mesh = None

        self.undeformed_spline_ = None
        self.unit_mesh_ = None
        self.deformed_unit_mesh_ = None

        self.offset_ = None
        self.scale_ = None

    def set_mesh(self, mesh):
        """
        Sets the member `input_mesh` to the specified mesh instance.

        Parameters
        ----------
        mesh : Mesh

        Returns
        -------
        None
        """
        self.input_mesh = mesh

    def set_deformed_spline(self, deformed_spline):
        """
        Sets the member `deformed_spline` to the specified spline instance.

        Parameters
        ----------
        deformed_spline : Spline

        Returns
        -------
        None
        """
        self.deformed_spline = deformed_spline

    def set_undeformed_spline(self, undeformed_spline):
        """
        Sets the member `undeformed_spline_` to the specified spline instance.
        This quantity is not required for the computation and just provides 
        additional flexibility for visualization purposes.

        Parameters
        ----------
        undeformed_spline : Spline

        Returns
        -------
        None
        """
        self.undeformed_spline_ = undeformed_spline

    def deform_mesh(self):
        """
        Deforms the provided `input_mesh` with the provided `deformed_spline`.
        The algorithm can be summarized as follows:
        1) Linearly transform `input_mesh` onto the multidimensional unit-cube.
            Store the result in `unit_mesh_`.
        2) Evaluate `deformed_spline` at each vertex of `unit_mesh_` 
            (deformation). Store the result in `deformed_unit_mesh_`.
        3) Invert the transformation from 1). Store the result in 
            `deformed_mesh`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logging.debug(
            "FreeFormDeformation - Bounds of input_mesh:\n{}.".format(
                self.input_mesh.bounds
            )
        )

        # offset for linear transformation
        self.offset_ = self.input_mesh.vertices.min(axis=0)
        logging.debug(
            "FreeFormDeformation - Transformation offset {}.".format(
                self.offset_
            )
        )

        self.unit_mesh_ = self.input_mesh.copy()
        self.unit_mesh_.vertices -= self.offset_
        
        # scaling for linear transformation
        self.scale_ = self.unit_mesh_.vertices.max(axis=0)
        logging.debug(
            "FreeFormDeformation - Transformation scale {}.".format(
                self.scale_
            )
        )

        self.unit_mesh_.vertices /= self.scale_
        logging.debug(
            "FreeFormDeformation - Bounds of unit_mesh_:\n{}.".format(
                self.unit_mesh_.bounds
            )
        )

        # transform the mesh with the spline
        self.deformed_unit_mesh_ = self.input_mesh.copy()
        self.deformed_unit_mesh_.vertices = self.deformed_spline.evaluate(
            self.unit_mesh_.vertices
        )
        logging.debug(
            "FreeFormDeformation - Bounds of deformed_unit_mesh_:\n{}.".format(
                self.deformed_unit_mesh_.bounds
            )
        )

        # rescale the mesh
        self.deformed_mesh = self.deformed_unit_mesh_.copy()
        self.deformed_mesh.vertices *= self.scale_
        self.deformed_mesh.vertices += self.offset_
        logging.debug(
            "FreeFormDeformation - Bounds of deformed_mesh:\n{}.".format(
                self.deformed_mesh.bounds
            )
        )

    def show(self):
        """
        Visualize the individual steps of the free form deformation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        from vedo import Plotter
        # retrieve the vedo representation of the meshes and modify their
        # graphical representation (wireframe for meshes per default)
        vedo_input_mesh = self.input_mesh.vedo_mesh
        vedo_input_mesh.wireframe(True)
        vedo_unit_mesh = self.unit_mesh_.vedo_mesh
        vedo_unit_mesh.wireframe(True)
        vedo_deformed_unit_mesh = self.deformed_unit_mesh_.vedo_mesh
        vedo_deformed_unit_mesh.wireframe(True)
        vedo_deformed_mesh = self.deformed_mesh.vedo_mesh
        vedo_deformed_mesh.wireframe(True)

        # retrieve the vedo representation of the splines and modify their
        # graphical representation (opacity for the spline surface)
        vedo_deformed_spline = self.deformed_spline.show(offscreen=True)
        vedo_deformed_spline[0].opacity(0.5)
        if self.undeformed_spline_ is not None:
            vedo_undeformed_spline = self.undeformed_spline_.show(
                offscreen=True
            )
            vedo_undeformed_spline[0].opacity(0.5)

        plt = Plotter(N=4, axes=0, sharecam=False)
        # upper left: input mesh
        plt.show(vedo_input_mesh, at=0) 
        # upper right: deformed mesh
        plt.show(vedo_deformed_mesh, at=1) 
        # lower left: unit mesh and undeformed spline
        if self.undeformed_spline_ is not None:
            plt.show(vedo_unit_mesh, vedo_undeformed_spline, at=2)
        else:
            plt.show(vedo_unit_mesh, at=2)
        # lower right: deformed unit mesh and deformed spline
        plt.show(
            vedo_deformed_unit_mesh, vedo_deformed_spline, 
            at=3, interactive=True
        ).close()