import logging
import os
from . import utils
import numpy as np
import matplotlib.pyplot as plt
from .interactive_selector import InteractiveSelector
from . import export

class Mesh:

    def __init__(self, vertices=None, faces=None, elements=None):
        """
        Mesh constructor.

        Parameters
        -----------
        vertices: (n,m) np.ndarray
          float. `m` is either 2 or 3.
        faces: (k,l) np.ndarray
          int. `l` is either 3 or 4.
          This is also often referred as cells.
        elements: (p,q) np.ndarray
          int. `q` is either 4 or 8.

        Returns
        --------
        None

        Attributes
        -----------
        vertices_: np.ndarray
        faces_: np.ndarray
        edges_: np.ndarray
        edges_computed_: bool
        unique_edges_: np.ndarray
        unique_edges_computed_: bool
        outlines_: list
        outlines_computed_: bool
        bc_names_: list
        bc_global_indices_: list
        elements_: np.ndarray
        surfaces_: list
        surfaces_computed_: bool
        faces_center_: np.ndarray
        faces_center_computed_: bool
        unique_faces_: np.ndarray
        unique_faces_computed_: bool
        """
        logging.debug('Mesh - Init')

        if vertices is not None:
            self.vertices = vertices
        else:
            self.vertices_ = None

        if faces is not None:
            self.faces = faces
        else:
            self.faces_ = None

        if elements is not None:
            self.elements = elements
        else:
            self.elements_ = None

        # This is naive way to avoid computing same things twice.
        self.edges_computed_ = False
        self.outlines_computed_ = False
        self.faces_computed_ = False
        self.surfaces_computed_ = False
        self.faces_center_computed_ = False
        self.unique_edges_computed_ = False
        self.unique_faces_computed_ = False

        self.bc_names_ = []
        self.bc_global_indices_ = []


    @property
    def vertices(self,):
        """
        Returns vertices.

        Parameters
        ----------
        None

        Returns
        --------
        self.vertices_: (n, d) np.ndarray
        """
        if not isinstance(self.vertices_, np.ndarray):
            self.vertices_ = np.asarray(self.vertices_, dtype=np.double)

        return self.vertices_


    @vertices.setter
    def vertices(self, vertices):
        """
        Vertices setter

        Parameters
        -----------
        vertices: (n,m) np.ndarray

        Returns
        --------
        None
        """
        self.vertices_ = vertices
        logging.debug(
            "Mesh - Vertices {v_shape} set.".format(v_shape=vertices.shape)
        )


    @property
    def faces(self,):
        """
        Returns faces.
        Computes faces according to the following index scheme, if 
        `elements` is defined.

        ``Tetrahedron``

        .. code-block::

            Ref: (node_ind), face_ind

                     (0)
                     /\ 
                    / 1\ 
                (1)/____\(3)
                  /\    /\ 
                 / 0\ 2/ 3\ 
                /____\/____\ 
              (0)   (2)    (0)

            face_ind | node_ind
            ---------|----------
            0        | 0 2 1
            1        | 1 3 0
            2        | 2 3 1
            3        | 3 2 0


        ``Hexahedron``

        .. code-block::


                    (6)    (7)
                    *------*
                    |      |
             (6) (2)| 3    |(3)   (7)    (6)
             *------*------*------*------*
             |      |      |      |      |
             | 2    | 0    | 4    | 5    |
             *------*------*------*------*
             (5) (1)|      |(0)   (4)    (5)
                    | 1    |
                    *------*
                    (5)    (4)

            face_ind | node_ind
            ---------|----------
            0        | 1 0 3 2
            1        | 0 1 5 4
            2        | 1 2 6 5
            3        | 2 3 7 6
            4        | 3 0 4 7
            5        | 4 5 6 7

        Parameters
        -----------
        None

        Returns
        --------
        self.faces_: (n, m) np.ndarray
        """
        if not isinstance(self.faces_, np.ndarray):
            self.faces_ = np.asarray(self.vertices_, dtype=np.int32)

        if self.elements_ is None:
            return self.faces_

        if self.faces_computed_:
            return self.faces_

        # Frau Tetra, Schwester von Petra, Enklin von Cleopatra.
        if self.elements_.shape[1] == 4:
            faces_per_element = 4

            self.faces_ = np.ones(
                (int(self.elements_.shape[0] * faces_per_element), 3),
                dtype=np.int32
            ) * -1 # -1 for safety.
            self.faces_[:,0] = self.elements_.flatten()

            # First Face
            self.faces_[::faces_per_element, 1] = self.elements_[:,2]
            self.faces_[::faces_per_element, 2] = self.elements_[:,1]
            # Second Face
            self.faces_[1::faces_per_element, 1] = self.elements_[:,3]
            self.faces_[1::faces_per_element, 2] = self.elements_[:,0]
            # Third Face
            self.faces_[2::faces_per_element, 1] = self.elements_[:,3]
            self.faces_[2::faces_per_element, 2] = self.elements_[:,1]
            # Fourth Face
            self.faces_[3::faces_per_element, 1] = self.elements_[:,2]
            self.faces_[3::faces_per_element, 2] = self.elements_[:,0]

        # Herr Hexa, ohne Familie.
        elif self.elements_.shape[1] == 8:
            faces_per_element = 6

            self.faces_ = np.ones(
                (int(self.elements_.shape[0] * faces_per_element), 4),
                dtype=np.int32
            ) * -1 # -1 for safety.

            # Zeroth Face
            self.faces_[::faces_per_element, 0] = self.elements_[:,1]
            self.faces_[::faces_per_element, 1] = self.elements_[:,0]
            self.faces_[::faces_per_element, 2] = self.elements_[:,3]
            self.faces_[::faces_per_element, 3] = self.elements_[:,2]

            # First Face
            self.faces_[1::faces_per_element, 0] = self.elements_[:,0]
            self.faces_[1::faces_per_element, 1] = self.elements_[:,1]
            self.faces_[1::faces_per_element, 2] = self.elements_[:,5]
            self.faces_[1::faces_per_element, 3] = self.elements_[:,4]

            # Second Face
            self.faces_[2::faces_per_element, 0] = self.elements_[:,1]
            self.faces_[2::faces_per_element, 1] = self.elements_[:,2]
            self.faces_[2::faces_per_element, 2] = self.elements_[:,6]
            self.faces_[2::faces_per_element, 3] = self.elements_[:,5]

            # Third Face
            self.faces_[3::faces_per_element, 0] = self.elements_[:,2]
            self.faces_[3::faces_per_element, 1] = self.elements_[:,3]
            self.faces_[3::faces_per_element, 2] = self.elements_[:,7]
            self.faces_[3::faces_per_element, 3] = self.elements_[:,6]

            # Fourth Face
            self.faces_[4::faces_per_element, 0] = self.elements_[:,3]
            self.faces_[4::faces_per_element, 1] = self.elements_[:,0]
            self.faces_[4::faces_per_element, 2] = self.elements_[:,4]
            self.faces_[4::faces_per_element, 3] = self.elements_[:,7]

            # Fifth Face
            self.faces_[5::faces_per_element, 0] = self.elements_[:,4]
            self.faces_[5::faces_per_element, 1] = self.elements_[:,5]
            self.faces_[5::faces_per_element, 2] = self.elements_[:,6]
            self.faces_[5::faces_per_element, 3] = self.elements_[:,7]

        self.faces_ = self.faces_.astype(np.int32)

        # Quick sanity check - No entries left untouched.
        assert self.faces_.all() != -1,\
             "There was an error while assigning faces."

        self.faces_computed_ = True
        logging.debug(
            "Mesh - Faces {f_shape} computed!".format(f_shape=self.faces_.shape)
        )

        return self.faces_
           

    @faces.setter
    def faces(self, faces):
        """
        Sets faces. In CAMPIGA, this is referred as `cells`

        Parameters
        -----------
        faces: (n,m) np.ndarray

        Returns
        --------
        None
        """
        self.faces_ = faces
        logging.debug("Mesh - Faces {c_shape} set.".format(c_shape=faces.shape))

    @property
    def unique_faces(self,):
        """
        Returns unique faces. `faces` give internal faces twice for volumes.
        Good for plotting.

        Parameters
        -----------
        None

        Returns
        --------
        self.unique_face_: (n, m) np.ndarray
        """
        if self.unique_faces_computed_:
            logging.debug("Mesh - Unique faces are already computed."+\
                " Returning stored value.")
            return self.unique_faces_

        # Copy array and sort
        tmp_faces = self.faces.copy().astype(np.int32)
        tmp_faces.sort(axis=1)

        # Thanks Numpy.
        self.unique_faces_ = np.unique(
            tmp_faces,
            axis=0
        )

        self.unique_faces_computed_ = True

        logging.debug("Mesh - Unique faces {u_shape} computed!".format(
            u_shape=self.unique_faces_.shape))

        return self.unique_faces_


    @property
    def edges(self,):
        """
        Returns edges.

        .. code-block::

            Ref: (node_ind), edge_ind

                 (0)
                 /\ 
              0 /  \ 2
               /____\ 
            (1)  1   (2)

                  2
            (3)*-----*(2)
               |     |
             3 |     | 1
            (0)*-----*(1)
                  0

        TODO: if `edges` index matter for tets, reorder it!

        Parameters
        -----------
        None

        Returns
        --------
        self.edges_ :  (n, 2) np.ndarray 
        """
        if self.edges_computed_:
            logging.debug("Mesh - Edges are already computed."+\
                " Returning stored value.")
            return self.edges_

        # Frequently used values
        num_faces = self.faces.shape[0]
        vertices_per_face = self.faces.shape[1]
           
        num_edges = int(num_faces * vertices_per_face)
        self.edges_ = np.ones((num_edges, 2),dtype=np.int32) * -1 # -1: safety
        self.edges_[:,0] = self.faces_.flatten()
        for i in range(vertices_per_face):
            # v_ind : corresponding vertex index for i value
            if i == int(vertices_per_face - 1):
                v_ind = 0
            else:
                v_ind = i + 1
            self.edges_[i::vertices_per_face, 1] = self.faces_[:,v_ind]

        # Quick sanity check - No entries are left untouched.
        assert self.edges_.all() != -1,\
             "There was an error while assigning edges."

        self.edges_computed_ = True
        logging.debug(
            "Mesh - Edges {e_shape} computed!".format(e_shape=self.edges_.shape)
        )

        return self.edges_

    @property
    def unique_edges(self,):
        """
        Returns unique edges. `edges` give internal edges twice.
        Good for plotting.

        Parameters
        -----------
        None

        Returns
        --------
        self.unique_edges_: (n, 2) np.ndarray
        """
        if self.unique_edges_computed_:
            logging.debug("Mesh - Unique edges are already computed."+\
                " Returning stored value.")
            return self.unique_edges_

        # Copy array and sort
        tmp_edges = self.edges.copy().astype(np.int32)
        tmp_edges.sort(axis=1)

        # Thanks Numpy.
        self.unique_edges_ = np.unique(
            tmp_edges,
            axis=0
        )

        self.unique_edges_computed_ = True

        logging.debug("Mesh - Unique edges {u_shape} computed!".format(
            u_shape=self.unique_edges_.shape))

        return self.unique_edges_

    @property
    def elements(self,):
        """
        Returns elements.

        Parameters
        -----------
        None

        Returns
        --------
        self.elements_: (n, m) np.ndarray
        """
        return self.elements_


    @elements.setter
    def elements(self, elements):
        """
        Sets elements. Support tetrahedron only.

        Parameters
        -----------
        elements: (n, m) np.ndarray

        Returns
        --------
        None
        """
        self.elements_ = elements
        logging.debug("Mesh - elements {el_shape} set.".format(
            el_shape=self.elements_.shape))


    @property
    def outlines(self,):
        """
        Computes outlines. Outlines are the edges that are only used once.
        Just returns corresponding edge indices.

        Parameters
        -----------
        None

        Returns
        --------
        self.outlines_ : (n,) np.ndarray
        """
        if self.outlines_computed_:
            logging.debug("Mesh - Outlines are already computed."+\
                "Returning stored value.")
            return self.outlines_

        # Copy array and sort
        tmp_edges = self.edges.copy()
        tmp_edges.sort(axis=1)

        # Numpy Magic.
        # This should be in `utils`, maybe?
        (_,
         unique_ind,
         unique_counts) = np.unique(
            tmp_edges,
            return_index=True,
            return_counts=True,
            axis=0
        )
        self.outlines_= unique_ind[unique_counts == 1]

        self.outlines_computed_ = True
        logging.debug("Mesh - Outlines {o_shape} computed!".format(
            o_shape=self.outlines_.shape))

        return self.outlines_

    @property
    def outline_2d_normals(self,):
        """
        Returns unit normals of outline edges. Experimental, thus, no dedicated
        attribute to save this value.
        Normals point "outwards" as long as mesh has counter clockwise winding.
        Only returns if self is 2d mesh.

        Parameters
        -----------
        None

        Returns
        --------
        outline_2d_normals: (n, 2) np.ndarray
        """
        if self.vertices.shape[1] == 3:
            #raise ValueError(
            #    "`outline_2d_normals` is only available for 2D meshes."
            #)
            return None

        outline_coords = self.vertices[self.edges[self.outlines]]
        # dx = x2 - x1, dy = y2 - y1
        dx_and_dy = np.diff(outline_coords, axis=1).reshape(-1, 2)

        # Make it a unit vector
        norm = np.linalg.norm(dx_and_dy, axis=1)
        dx_and_dy /= np.vstack((norm, norm)).transpose()

        logging.debug("Mesh - 2D outline normals computed. Returning "
            + "normalized (dy, -dx).")
            
        # (dy, -dx)
        return dx_and_dy[:,[1,0]] * [1, -1]


    @property
    def surfaces(self,):
        """
        Returns a unique faces, that forms surface.
        This is useful for Tetrahedron meshes, as it will only return outer
        meshes.
        For 2D triangles, this will return all the faces.

        Parameters
        -----------
        None

        Returns
        --------
        self.surfaces_: (n,) np.ndarray
        """
        if self.surfaces_computed_:
            logging.debug("Mesh - Surfaces are already computed. "+\
                "Returning stored value.")
            return self.surfaces_

        # Copy array and sort.
        tmp_faces = self.faces.copy()
        tmp_faces.sort(axis=1)

        # Numpy Magic.
        (_,
         unique_ind,
         unique_counts) = np.unique(
            tmp_faces,
            return_index=True,
            return_counts=True,
            axis=0
        )
        self.surfaces_= unique_ind[unique_counts == 1]

        self.surfaces_computed_ = True
        logging.debug("Mesh - Surfaces {s_shape} computed!".format(
            s_shape=self.surfaces_.shape))

        return self.surfaces_

    @property
    def bounds(self,):
        """
        Returns bounds of the mesh.

        Parameters
        -----------
        None

        Returns
        --------
        bounds: (2, d) np.ndarray
        """
        return utils.bounds(self.vertices)

    @property
    def bounds_norm(self,):
        """
        Returns norm of the bounds.

        Parameters
        -----------
        None

        Returns
        --------
        bounds_norm: float
        """
        bounds = self.bounds

        return np.linalg.norm((bounds[0] - bounds[1]))

    @property
    def bouding_box(self,):
        """
        Returns bounding box hexa mesh.

        Parameters
        -----------
        None

        Returns
        --------
        bouding_box_mesh: `Mesh`
        """
        return Mesh(
            vertices=utils.bounding_box(self.vertices),
            elements=np.array([0,1,2,3,4,5,6,7]),
        )

    @property
    def bounding_box_centroid(self,):
        """
        Returns bounding box centroid.

        Parameters
        -----------
        None

        Returns
        --------
        bounding_box_centroid: (d,) np.ndarray
        """
        return utils.bounding_box_centroid(self.vertices)

    def copy(self,):
        """
        Returns newly made Mesh.

        Parameters
        -----------
        None

        Returns
        --------
        new_mesh: `Mesh`
        """
        if self.elements is None:
            return Mesh(
                vertices=self.vertices.copy(),
                faces=self.faces.copy()
            )

        else:
            return Mesh(
                vertices=self.vertices.copy(),
                elements=self.elements.copy()
            )

    def select_vertices(
        self,
        method='interactive',
        fig_size=(10,10),
        PSLG=None
    ):
        """
        Select vertices. Currently only supports interactive mode and 2D meshes.

        Parameters
        -----------
        method: str
        PSLG: idk yet. probably `Segment`

        Returns
        --------
        v_ind: list 
        """
        if method == "interactive":
            v_ind = self.interactive_selector_(
                self.vertices,
                fig_size=fig_size,
                PSLG=PSLG,
            )
        else:
            v_ind = None

        return v_ind


    def interactive_selector_(
        self,
        points,
        fig_size=(10,10),
        PSLG=None
    ):
        """
        Select vertices. Currently only supports interactive mode and 2D meshes.

        Parameters
        -----------
        method: str
        PSLG: list
          [nodes, connectivity, (optional) pslg_only]
          [`np.ndarray`, `np.ndarray`, `bool`]
          For now, if there's third entry, regardless of the entry it displays
          PSLG only.

        Returns
        --------
        v_ind: list 
        """
        fig, ax = plt.subplots(figsize=fig_size)
        pts = ax.scatter(
            points[:,0],
            points[:,1],
            c='fuchsia',
            zorder=10,
        )
        v_selector = InteractiveSelector(ax, pts)

        logging.info(
            "Select points in the figure by enclosing them within a polygon."
        )
        logging.info("Press the 'esc' key to start a new polygon.")
        logging.info("Try holding the 'shift' key to move all of the vertices.")
        logging.info("Try holding the 'ctrl' key to move a single vertex.")

        if PSLG is not None:
            draw_edges = True

            if len(PSLG) == 3:
                draw_edges = False

        for i in range(PSLG[1].shape[0]):
            ax.plot(
                PSLG[0][PSLG[1][i]][:,0],
                PSLG[0][PSLG[1][i]][:,1],
                c='forestgreen',
                lw=1,
                zorder=100,
            )

        if draw_edges:
            for i in range(self.unique_edges.shape[0]):
                ax.plot(
                    self.vertices[self.unique_edges_[i]][:,0],
                    self.vertices[self.unique_edges_[i]][:,1],
                    c='cornflowerblue',
                    lw=.5,
                    zorder=100,
                )

        plt.title("Info: pink dots represent selectable points. Blue/Green"+\
            " lines are edges for reference.", fontsize=16)
        plt.show()
        v_selector.disconnect()

        return  v_selector.ind


    def remove_vertices(self, ind):
        """
        Removes vertices reference by indices. 
        Returns a new mesh, thus, leaving original mesh intact.

        TODO: Another removal method could be neighbor based.
            set protected edges and delete everything else sequentially.

        Parameters
        -----------
        ind: list

        Returns
        --------
        new_mesh: `Mesh`
        """
        # All you need to do is turn ind into a mask and update.
        mask = np.ones(self.vertices.shape[0], dtype=bool)
        mask[ind] = False

        # A bit work around but we want to keep the mesh intact.
        mesh = self.update_vertices(mask=mask)

        return mesh

    def remove_unreferenced_vertices(self):
        """
        Remove all the vertices that aren't referenced by a cell.
        Adapted from `Trimesh`.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        referenced = np.zeros(len(self.vertices), dtype=bool)

        if self.elements is None:
            referenced[self.faces] = True

        else:
            referenced[self.elements] = True

        inverse = np.zeros(len(self.vertices), dtype=np.int64)
        inverse[referenced] = np.arange(referenced.sum())

        return self.update_vertices(mask=referenced, inverse=inverse)

    def merge_vertices(self, tolerance=1e-10):
        """
        Merges vertices and returns a new mesh with merged vertices.

        Parameters
        -----------
        tolerance: float
          Default is 1e-10.

        Returns
        --------
        new_mesh: `Mesh`
        """
        return merge_vertices(self, tolerance=tolerance)


    def update_vertices(self, mask, inverse=None,):
        """
        Keep only masked vertices. Returns a new Mesh.
        Removes "ghost" faces afterwards also.

        Parameters
        -----------
        mask: (self.vertices.shape[0]) of bool
        inverse: (self.vertices.shape[0]) of int

        Returns
        --------
        mesh: `Mesh`
        """
        mask = np.asanyarray(mask)

        if (mask.dtype.name == 'bool' and mask.all()) or len(mask) == 0:
            # mask doesn't remove any vertices so exit early
            return Mesh(vertices=self.vertices, faces=self.faces)

        # Here, we need to treat surface meshes and volume meshes differently.
        if self.elements is None:
            connectivity = self.faces.copy()
            is_volume = False
        else:
            connectivity = self.elements.copy()
            is_volume = True

        ###### Don't think this is necessary ######
        # First remove all the faces/elements that include ind
        # We need inverted ind
        # ?
        #inv_mask = ~mask
        #ind = np.arange(len(inv_mask))[inv_mask]
        #for i, i_ind in enumerate(ind):
        #    if i == 0:
        #        c_mask =  == i_ind
        #    else:
        #        c_mask = np.logical_or(
        #            c_mask,
        #            self.faces == i_ind
        #        )
        #c_mask = (c_mask.sum(axis=1).astype(np.int) == 0).astype(np.bool) 
        #faces = self.faces[c_mask]

        # create the inverse mask if not passed
        if inverse is None:
            inverse = np.zeros(len(self.vertices), dtype=np.int64)
            if mask.dtype.kind == 'b':
                inverse[mask] = np.arange(mask.sum())
            elif mask.dtype.kind == 'i':
                inverse[mask] = np.arange(len(mask))
            else:
                inverse = None

        # re-index faces/elements from inverse
        if inverse is not None:
            connectivity = inverse[connectivity.reshape(-1)].reshape(
                (-1, connectivity.shape[1])
            )

        # Now vertices. Update.
        vertices = self.vertices[mask]
        logging.debug(
            "Mesh - Retuning a new Mesh with updated {v_shape} vertices.".format(
                v_shape=self.vertices.shape
            )
        )

        if is_volume:
            return Mesh(vertices=vertices, elements=connectivity)

        else:
            return Mesh(vertices=vertices, faces=connectivity)

    def remove_faces(self, ind, optimize=None):
        """
        Remove faces and remove unreferenced vertices?

        Parameters
        -----------
        ind: list of int

        Returns
        --------
        new_mesh: `mesh`
        """
        c_mask = np.ones(self.faces.shape[0], dtype=bool)
        c_mask[ind] = False
        faces = self.faces[c_mask]
        mesh = Mesh(
            vertices=self.vertices,
            faces=faces,
        ).remove_unreferenced_vertices()

        if optimize is not None:
            mesh = optimize_mesh(mesh, optimize)

        return mesh
        

    @property
    def faces_center(self):
        """
        Should work for both tri and quad?

        Parameters
        -----------
        None

        Returns
        --------
        faces_center: (n,m) np.ndarray
        """
        if self.faces_center_computed_:
            return self.faces_center_

        self.faces_center_ = self.vertices[self.faces].mean(axis=1)
        self.faces_center_computed_ = True

        return self.faces_center_


    def select_faces(
        self,
        method='interactive',
        criteria=None,
        only_surface=False,
        fig_size=(10,10),
        PSLG=None
    ):
        """
        Face selector.
        For `xyz_range`, please specify [[x_min, x_max], ...].

        Parameters
        -----------
        method: str
          Options <"interactive"|"xyz_range">
          Default is "interactive".
        criteria: list
          Needed for method == xyz_range
        fig_size: (n, m) tuple
        PSLG: !?

        Returns
        --------
        c_ind: list of int
        """

        if method == "interactive":
            c_ind = self.interactive_selector_(
                self.faces_center,
                fig_size=fig_size,
                PSLG=PSLG,
            )

        elif method == "xyz_range":
            masks = []
            for i, c in enumerate(criteria):
                if c is None:
                    continue
                else:
                    if only_surface:
                        lower = self.faces_center[self.surfaces, i] > c[0]
                        upper = self.faces_center[self.surfaces, i] < c[1]

                    else:
                        lower = self.faces_center[:,i] > c[0]
                        upper = self.faces_center[:,i] < c[1]

                    masks.append(
                        np.logical_and(lower,upper)
                    )

            if len(masks) > 1:
                if only_surface:
                    mask = np.zeros(self.surfaces.shape[0], dtype=bool)

                else:
                    mask = np.zeros(self.faces.shape[0], dtype=bool)

                for i, m in enumerate(masks):
                    if i == 0:
                        mask = np.logical_or(mask, m)

                    else:
                        mask = np.logical_and(mask, m)

            else:
                mask = masks[0]

            if only_surface:                    
                c_ind = np.arange(self.surfaces.shape[0])[mask]
                c_ind = self.surfaces[c_ind]

            else:
                c_ind = np.arange(self.faces.shape[0])[mask]

        else:
            c_ind = None

        return c_ind


    def set_BC(
        self,
        name,
        method="interactive",
        criteria=None,
        fig_size=(10,10),
    ):
        """
        BC setter. Highlight of this `Mesh` class.

        Parameters
        -----------
        name: str
          BC Name.
        method: str
          Options are <'interactive' | 'index' | 'xyz_range'>
          Reference mesh to show its
        criteria: list-like
          If `method="index"`, it is a list of indices.
          If `method="xyz_range"`, it is a list of ranges.
        fig_size: tuple or list
          For `method="interactive"`.

        Returns
        --------
        None 
        """
        # Set name.
        self.bc_names_.append(name)

        # Interactive
        if method == "interactive":
            edge_mid_points = self.vertices[
                self.edges[self.outlines]
            ].mean(axis=1)
        
            fig, ax = plt.subplots(figsize=fig_size)

            pts = ax.scatter(
                edge_mid_points[:,0],
                edge_mid_points[:,1],
                c='fuchsia',
                zorder=10,
            )

            # This would be a proper place for future `if` method == !?
            bc_selector = InteractiveSelector(ax, pts)
            logging.info(
                "Select points in the figure by enclosing them within " +\
                "a polygon."
            )
            logging.info("Press the 'esc' key to start a new polygon.")
            logging.info(
                "Try holding the 'shift' key to move all of the vertices."
            )
            logging.info("Try holding the 'ctrl' key to move a single vertex.")

            # Hard Coded for 2D
            ax.scatter(
                self.vertices[self.edges_[self.outlines_]].reshape(-1,2)[:,0],
                self.vertices[self.edges_[self.outlines_]].reshape(-1,2)[:,1],
                s=10,
                alpha=.5,
                c='b',
            )

            # Shows only outline edges
            for i in range(self.outlines_.shape[0]):
                ax.plot(
                    self.vertices[self.edges_[self.outlines_[i]]][:,0],
                    self.vertices[self.edges_[self.outlines_[i]]][:,1],
                    c='cornflowerblue',
                    lw=.5,
                    zorder=100,
                )

            plt.title(name + ". Info: pink dots represent edges. Blue dots" +\
                " are vertices for reference", fontsize=16)
            plt.show()
            bc_selector.disconnect()

            self.bc_global_indices_.append(
                self.outlines_[bc_selector.ind].tolist()
            )

        # Index
        elif method == "index":
            self.bc_global_indices_.append(criteria)

        # XYZ Range
        elif method == "xyz_range":
            masks = []
            for i, c in enumerate(criteria):
                if c is None:
                    continue
                else:
                    lower = self.faces_center[self.surfaces, i] > c[0]
                    upper = self.faces_center[self.surfaces, i] < c[1]
                    masks.append(
                        np.logical_and(lower,upper)
                    )

            if len(masks) > 1:
                mask = np.zeros(self.surfaces.shape[0], dtype=bool)
                for i, m in enumerate(masks):
                    if i == 0:
                        mask = np.logical_or(mask, m)

                    else:
                        mask = np.logical_and(mask, m)

            else:
                mask = masks[0]

            self.bc_global_indices_.append(self.surfaces[mask])

        logging.debug("Mesh - Boundary Condition set!")
        logging.debug("Mesh - Name of BC: " + str(self.bc_names_[-1]))
        logging.debug("Mesh - Number of BC edges/faces: " +\
            str(len(self.bc_global_indices_[-1])))

    def append_BC(self, existing_bc_name, **kwargs):
        """
        Appends to existing bc.
        This is done by:

         1. set BC normally - to use existing `set_BC` function
         2. append newly added tmp BC to correct BC
         3. remove newly added tmp BC

        Parameters
        -----------
        existing_bc_name: str
        method: str
        criteria: list-like
        fig_size: tuple or list

        Returns
        --------
        None
        """
        # Check if `existing_bc_name` really exists
        if not existing_bc_name in self.bc_names_:
            raise ValueError(
                "Given BC name does not exist. Sorry, we can't append it then."
            )

        assert len(self.bc_names_) == len(self.bc_global_indices_),\
            "Well, number of bc names and globcal indices does not match."

        num_prev_bc = len(self.bc_names_)

        # Add as usual
        tmp_bc_name = "ich_bin_tmp_bc_ich_will_perm_bc_sein"
        self.set_BC(name=tmp_bc_name, **kwargs)

        # Append properly
        existing_bc_ind = self.bc_names_.index(existing_bc_name)
        self.bc_global_indices_[existing_bc_ind] = np.append(
            self.bc_global_indices_[existing_bc_ind],
            self.bc_global_indices_[-1]
        )
        logging.debug(
            "Mesh - Appended to bc ('{ebc}')".format(ebc=existing_bc_name)
        )

        # Remove newly added tmp BC
        self.bc_names_.pop(-1)
        self.bc_global_indices_.pop(-1)

        # Quick sanity check
        assert len(self.bc_names_) == num_prev_bc,\
            "Something went wrong during `appnend_BC`. number of names and " +\
            "global indices does not match."

        logging.debug(
            "Mesh - Removed temporary bc ('{tbc}')".format(tbc=tmp_bc_name)
        )


    def load(self, fname):
        """
        Uses meshIO to import mesh.
        Assumes there's only one mesh in file.

        Parameters
        -----------
        fname: str

        Returns
        --------
        None
        """
        import meshio

        m = meshio.read(fname)
        self.vertices = m.points
        self.faces = m.cells[0].data

        logging.debug(
            "Mesh - Imported {fn}, with ({v}) vertices and ({f}) faces".format(
                fn=fname,
                v=self.vertices_.shape,
                f=self.faces_.shape,
            )
        )

    def export(self, fname, space_time=False):
        """
        Exports `tetrex` or `mixd` files based on extension of fname.
        Space time output is only possible for xns.

        Parameters
        -----------
        fname: str
          Valid extensions are <`.grd` | `.xns` | `.campiga` | `.h5`>
        space_time: bool
          Only relevant for `.xns` exports.

        Returns
        --------
        None
        """
        ext = os.path.splitext(fname)[1]

        export.utils.check_and_makedirs(fname)

        if ext == ".grd":
            if self.elements_ is None:
                export.tetrex.tetrex2d(fname, self,)
                logging.debug(
                    "Mesh - Exported 2D Mesh as {fname}".format(fname=fname)
                )

            else:
                export.tetrex.tetrex3d(fname, self,)
                logging.debug(
                    "Mesh - Exported 3D Mesh as {fname}".format(fname=fname)
                )

        elif ext == ".xns" or ext == ".campiga":
            export.mixd.mixd(fname, self, space_time=space_time)
            logging.debug(
                "Mesh - Exported Mesh as {fname}".format(fname=fname)
            )

        elif ext == ".h5":
            export.hdf5.hdf5(fname, self,)
            logging.debug(
                "Mesh - Exporing Mesh using hdf5 as {fname}".format(fname=fname)
            )
            logging.debug(
                "Mesh -    This is EXPERIMENTAL"
            )

        else:
            logging.warning(
                "Mesh - We don't support {ext} format. Skipping export.".format(
                    ext=ext)
            )

    @property
    def trimesh_mesh(self,):
        """
        Returns trimesh.

        Parameters
        -----------
        None

        Returns
        --------
        trimesh_mesh: `trimesh.Trimesh`
        """
        import trimesh

        if self.vertices_.shape[1] == 2:
            tm = trimesh.Trimesh(
                vertices=np.hstack(
                    (self.vertices_,
                     np.zeros(self.vertices_.shape[0]).reshape(-1,1))
                ),
                faces=self.faces_,
                process=False
            )

        else:
            tm = trimesh.Trimesh(
                vertices=self.vertices_,
                faces=self.faces_,
                process=False
            )

        return tm

    def optimize(self, config=["laplace", 1e-5, 10]):
        """
        Wapper around `optimize_mesh`.

        Parameters
        -----------
        config: str or list
          If `str` -> method.

          If `list` -> [method(str), tolerance(float), iteration(int)].

          methods: lloyd, cvt-diagonal, cvt-block-diagonal, cvt-full,
          cpt-linear-solve, cpt-fixed-point, cpt-quasi-newton, laplace,
          odt-fixed-point

        Returns
        -------
        optimized_mesh: Mesh
        """
        logging.debug(
            "Mesh - Optimizing mesh with following config: {c}".format(c=config)
        )

        return optimize_mesh(self.copy(), config)

    @property
    def vedo_mesh(self,):
        """
        Returns vedo mesh. That consists of vertices and faces.

        Parameters
        -----------
        None

        Returns
        --------
        vedo_mesh: `vedo.Mesh`
        """
        import vedo

        logging.debug("Mesh - Generating `vedo` mesh.")

        return vedo.Mesh([self.vertices, self.faces], c="black")

    @property
    def vedo_surface_mesh(self,):
        """
        Returns vedo surface mesh. Useful for visualizing volume meshes.

        Parameters
        -----------
        None

        Returns
        --------
        vedo_mesh: `vedo.Mesh`
        """
        import vedo

        logging.debug("Mesh - Generating `vedo` surface mesh.")

        tmp_mesh = Mesh(
            vertices=self.vertices,
            faces=self.faces[self.surfaces],
        ).remove_unreferenced_vertices()
        tmp_mesh = tmp_mesh.merge_vertices()

        return vedo.Mesh([tmp_mesh.vertices, tmp_mesh.faces], c="black")

    def show(
        self,
        shrink=True,
        surface_only=True,
        BC=False,
        backend="vedo"
    ):
        """
        Show mesh using different backends.

        Parameters
        ----------
        surface_only: bool
          Only make difference for volume mesh. Only plots surface.
        backend: str

        Returns
        --------
        None
        """
        if backend=="vedo":
            # Here, you see the "power" rank of each options
            if BC:
                import vedo

                colors = [*vedo.colors.colors.keys()]

                bc_meshes = []
                for bc in self.bc_global_indices_:
                    bc_meshes.append(
                        Mesh(
                            vertices=self.vertices,
                            faces=self.faces[bc],
                        ).remove_unreferenced_vertices()
                        .vedo_mesh
                        .color(np.random.choice(colors))
                    )
                vedo.show(bc_meshes).close()

            elif shrink and self.elements is not None:
                import vedo
                from vtk import VTK_TETRA as frau_tetra
                from vtk import VTK_HEXAHEDRON as herr_hexa

                if self.elements.shape[1] == 4:
                    u = vedo.UGrid(
                        [
                            self.vertices,
                            self.elements,
                            np.repeat([frau_tetra], self.elements.shape[0])
                        ]
                    )

                elif self.elements.shape[1] == 8:
                    u = vedo.UGrid(
                        [
                            self.vertices,
                            self.elements,
                            np.repeat([herr_hexa], self.elements.shape[0])
                        ]
                    )

                um = u.tomesh(shrink=0.8).color("yellow")
                um.show().close()               

            elif surface_only and self.elements is not None:
                self.vedo_surface_mesh.show().close()

            else:
                self.vedo_mesh.show().close()

        elif backend=="trimesh":
            self.trimesh_mesh.show()


def optimize_mesh(mesh, optimize):
    """
    Mesh optimization using optimesh.

    Parameters
    -----------
    mesh: `Mesh`
    optimize: str or list
      str -> method
      list -> [method(str), tolerance(float), iteration(int)]
      methods:
        lloyd
        cvt-diagonal
        cvt-block-diagonal
        cvt-full
        cpt-linear-solve
        cpt-fixed-point
        cpt-quasi-newton
        laplace
        odt-fixed-point

    Returns
    --------
    optimized_mesh: `Mesh`
    """

    import optimesh

    if isinstance(optimize, str):
        logging.debug(
            "Mesh - Optimizing mesh with `{m}` - method".format(m=optimize)
        )

        v, f = optimesh.optimize_points_cells(
            mesh.vertices,
            mesh.faces,
            [optimize, 1e-3, 10], #
        )

    elif isinstance(optimize, list):
        logging.debug(
            "Mesh - Optimizing mesh with `{m}` - method".format(m=optimize[0])
        )

        v, f = optimesh.optimize_points_cells(
            mesh.vertices,
            mesh.faces,
            optimize[0], # <- Method str
            optimize[1], # <- tol float
            optimize[2], # <- iteration int
        )

    else:
        raise ValueError(
            "Invalid mesh optimization input. It takes either `str` or `list`."
        )

    return Mesh(vertices=v, faces=f)

def merge_vertices(mesh, tolerance=1e-10):
    """
    Removes duplicating vertices, grouped by position. Adapted from trimesh.

    Parameters
    -----------
    mesh: `Mesh`
    tolerance: float
      Merge tolerance. 

    Returns
    --------
    new_mesh: `Mesh`
    """
    logging.debug("Mesh - Vertex merge requested.")
    logging.debug("Mesh -   Original number of vertices: {nv}".format(
        nv=len(mesh.vertices)))

    # Get referenced vertices mask
    referenced = np.zeros(len(mesh.vertices), dtype=bool)
    if mesh.elements is not None:
        referenced[mesh.elements] = True

    elif mesh.faces is not None:
        referenced[mesh.faces] = True

    # In case it is a pure
    else:
        referenced[:] = True

    # Alternative is `round()` or `ceil()`.
    # round example from trimesh 
    #   -> np.round(int_vertices - 1e-6).astype(np.int64)
    int_vertices = np.floor(mesh.vertices * int(1 / tolerance)).astype(np.int64)

    (_,
     unique,
     inv) = np.unique(
        int_vertices[referenced],
        return_index=True,
        return_inverse=True,
        axis=0,
    )

    # Inverse mask
    inverse = np.zeros(len(mesh.vertices), dtype=np.int64)
    inverse[referenced] = inv
    # Vertex mask
    mask = np.nonzero(referenced)[0][unique]

    logging.debug("Mesh -   Number of vertices after merge: {nv}".format(
        nv=mask.sum()))

    return mesh.update_vertices(mask=mask, inverse=inverse)
