import logging
from . import utils
from .interactive_drawer import InteractiveDrawer 
import numpy as np
from scipy.spatial import cKDTree as KDTree

class Segment:

    def __init__(
        self, 
        nodes=None,
        connectivity=None,
        global_subsection=None,
        global_spacing=None
    ):
        """
        Segment Constructor.

        Parameters
        -----------
        nodes: (n,m) np.ndarray (Optional)
        connectivity: (k,l) np.ndarray (Optional)
        global_subsection_: int
        global_spacing_: float

        Returns
        --------
        None

        Attributes
        -----------
        nodes_: np.ndarray
        connectivity_: np.ndarray
        global_subsection_: int
        global_spacing_: float
        last_squence_ind_: int
        add_node_counter_: int
        is_new_sequence_: bool
        reference_nodes: list of int
        polygons_: list
        boundary_edge_ids_: list
        boundary_facet_ids_: list
        facet_with_holes_: list
        """
        logging.debug("Segment - Init")

        # Some initial values
        self.nodes_ = None
        self.connectivity_ = None
        self.global_subsections_ = 1
        logging.debug("Segment - Applying default `global_subsections_` "+\
            "({g_sub}).".format(g_sub=self.global_subsections_))
        self.global_spacing_ = None
        self.last_sequence_ind_ = None
        self.is_new_sequence_ = False
        self.reference_nodes = []
        self.polygons_ = []
        self.boundary_edge_ids_ = []
        self.boundary_facet_ids_ = []
        self.facet_with_holes_ = []

        if nodes is not None:
            self.nodes = nodes
        if connectivity is not None:
            self.connectivity = connectivity
        if global_subsection is not None:
            self.global_subsection = global_subsection
        if global_spacing is not None:
            self.global_spacing = global_spacing

    @property
    def nodes(self,):
        """
        Returns Nodes.

        Parameters
        -----------
        None

        Returns
        --------
        self.nodes_: (n,m) np.ndarray
        """
        return self.nodes_

    @nodes.setter
    def nodes(self, nodes):
        """
        Nodes setter. Also sets `last_sequence_ind` to the last node.

        Parameters
        -----------
        nodes: (n,m) np.ndarray

        Returns
        --------
        None
        """
        self.nodes_ = nodes
        self.last_sequence_ind = int(self.nodes_.shape[0] - 1)
        logging.debug(
            "Segment - Nodes {n_shape} set.".format(n_shape=self.nodes_.shape)
        )

    @property
    def connectivity(self,):
        """
        Returns connectivity

        Parameters
        -----------
        None

        Returns
        --------
        self.connectivity_: (n,m) np.ndarray
        """
        return self.connectivity_

    @connectivity.setter
    def connectivity(self, connectivity):
        """
        Connectivity setter. 

        Parameters
        -----------
        connectivity: (n, 2) np.ndarray

        Returns
        --------
        None
        """
        # Is connectivity referring to valid indices?
        assert self.nodes_.shape[0] == int(connectivity.max() + 1),\
            "Your connectivity array is referring to an index that does "+\
            "not exist."
        self.connectivity_ = connectivity
        logging.debug("Segment - Connectivity {c_shape} set.".format(
            c_shape=self.connectivity_.shape))

    @property
    def polygons(self,):
        """
        Returns polygons.

        Parameters
        -----------
        None

        Returns
        --------
        self.polygons_: list
          list of list of int. Containing list could be in different length
        """
        return self.polygons_

    @polygons.setter
    def polygons(self, polygons):
        """
        Polygons setter.

        Parameters
        -----------
        polygons: list
          list of list. No more, no less. It is not tested, though.
          
        Returns
        --------
        None
        """
        assert isinstance(polygons, list), "Polygons must be a `list` object."
        self.polygons_ = polygons
        logging.debug("Segment - {p_len} Polygons set.".format(
            p_len=len(self.polygons_)))

    @property
    def boundary_edge_ids(self,):
        """
        Returns boundary edge ids. Meant for 2D cases.

        Parameters
        -----------
        None

        Returns
        --------
        self.boundary_edge_ids_: list
          list of int. Length should be the same as n-connectivity
        """
        return self.boundary_edge_ids_

    @boundary_edge_ids.setter
    def boundary_edge_ids(self, boundary_ids):
        """
        Boundary edge IDs setter. Meant for 2D cases.

        Parameters
        -----------
        boundary_ids: list

        Returns
        --------
        None
        """
        self.boundary_edge_ids_ = boundary_ids
        logging.debug("Segment - {bid_len} Boundary edge IDs set.".format(
            bid_len=len(self.boundary_edge_ids_)))

    @property
    def boundary_facet_ids(self,):
        """
        Returns boundary facet ids. Meant for 3D cases.

        Parameters
        -----------
        None

        Returns
        --------
        self.boundary_facet_ids_: list
          list of int. Length should be same as n-facets/polygons
        """
        return self.boundary_facet_ids_

    @boundary_facet_ids.setter
    def boundary_facet_ids(self, boundary_ids):
        """
        Boundary facet IDs setter. Meant for 3D cases.

        Parameters
        -----------
        boundary_ids: int

        Returns
        --------
        None
        """
        self.boundary_facet_ids_ = boundary_ids
        logging.debug("Segment - {bid_len} Boundary facet IDs set.".format(
            bid_len=len(self.boundary_facet_ids)))

    @property
    def facet_with_holes(self,):
        """
        Returns facet with holes. Strictly for `tetgen` input.

        Format:
          [[hfacepolygonpoints, hfaceholes, boundary_id], ...]

        Parameters
        -----------
        None

        Returns
        --------
        self.facet_with_holes_: list
        """
        return self.facet_with_holes_

    @facet_with_holes.setter
    def facet_with_holes(self, facet_with_holes):
        """
        Facet with holes setter. Strictly for `tetgen` input.

        Format:
          [[hfacepolygonpoints, hfaceholes, boundary_id], ...]

        Parameters
        -----------
        facet_with_holes: list
          See format.

        Returns
        --------
        None
        """
        self.facet_with_holes_ = facet_with_holes
        # Quick check: see if the last element has length of 3.
        #   - Every element should have length 3. But it is a quick check.
        assert len(facet_with_holes[-1]) == 3,\
            "Invalid format for facet with holes!"

        logging.debug("Segment - {nfwh} Facets with holes set.".format(
            nfwh=len(facet_with_holes)))

    def add_node(
        self,
        node,
        subsections=None,
        spacing=None,
        close_sequence=False,
        new_sequence=False,
        boundary_id=-1,
        closing_boundary_id=None,
    ):
        """
        Add Node.

        Parameters
        -----------
        node: (n,) np.ndarray
        subsections: int
        spacing: float
        close_loop: bool
          Should this node close the segment?
          Use with care. If used with
        new_sequence: bool
          Start a new sequence of lines.
        boundary_id: int
          Boundary condition number. Default is -1 for "no-assignment".
          Use Positive values > 0 to assign values.
        closing_boundary_id: int
          If defined, assigns different boundary_id to the closing connectivity.
          Otherwise, uses boundary_id.

        Returns
        --------
        None
        """
        node = np.asarray(node).reshape(1,-1)

        if self.nodes_ is None:
            logging.debug("Segment - Adding the first node.")
            self.nodes_= node
            self.last_sequence_ind = 0
            self.reference_nodes.append(int(self.nodes_.shape[0] - 1))
            return

        if self.is_new_sequence_:
            new_sequence = True
            self.is_new_sequence_ = False

        if new_sequence:
            self.nodes_= np.vstack((self.nodes_, node))
            logging.debug("Segment - Staring new sequence. ")
            self.last_sequence_ind = int(self.nodes_.shape[0] - 1)
            self.reference_nodes.append(int(self.nodes_.shape[0] - 1))
            return

        subsections = self.compute_subsections_(
            subsections,
            spacing,
            compute_distance(node, self.nodes_[-1])
        )

        # This is redundent. But makes sure that everything worked so far.
        if subsections is not None:
            previous_node_ind = int(self.reference_nodes[-1])
            self.nodes_ = np.vstack((self.nodes_, node))

            self.reference_nodes.append(int(self.nodes_.shape[0] - 1))

            # Add connectivity
            self.connect_nodes(
                [-2,-1],
                reference_nodes=True,
                subsections=subsections,
                spacing=spacing,
                boundary_id=boundary_id,
            )

            if close_sequence:
                logging.warning("Segment - Closing last sequence...")

                # Determine boundary_id for closing connectivity
                if closing_boundary_id is None:
                    cbid = boundary_id
                    logging.warning("Segment -   Applying same boundary_id, "+\
                        "since it wasn't specified.")
                else:
                    cbid = closing_boundary_id

                self.connect_nodes(
                    indices=(self.reference_nodes[-1], self.last_sequence_ind),
                    reference_nodes=False,
                    spacing=spacing,
                    boundary_id=cbid,
                )

    def add_line_from_snap(
        self,
        two_nodes,
        subsections=None,
        spacing=None,
        boundary_id=-1,
    ):
        """
        Using naive snap duplicates points, and it tends to yield segmentation
        fault. So this one snaps to the first one and add the second one. 
        Probably useful if you want to continue drawing from some vertex.
        If you mean to creat a polygon, any polygon starting with this line
        will have to be closed using `connect_nodes` with snap. 
        This is always a new sequence.
 
        Parameters
        -----------
        two_nodes: (2, d) np.ndarray
        subsection: int
        spacing: float
        boundary_id: int
          Boundary condition number. Default is -1 for "no-assignment".
          Use Positive values > 0 to assign values.

        Returns
        --------
        None
        """
        two_nodes = np.asarray(two_nodes)
        if two_nodes.shape[0] != 2:
            raise ValueError("`two_nodes` takes two nodes only.")

        logging.debug(
            "Segment - Snap node requested: {sn}.".format(sn=two_nodes[0])
        )
        # Abuse KDTree
        # It is fast enough that we can abuse it.
        kdt = KDTree(self.nodes_)
        _, nn_ind = kdt.query(two_nodes[0])
        logging.debug(
            "Segment - Snapping to [{ni}] node: {n}.".format(
                ni=nn_ind,
                n=self.nodes[nn_ind]
            )
        )

        # Manipulate some values
        self.reference_nodes.append(nn_ind)

        self.add_node(two_nodes[-1], new_sequence=True)

        self.last_sequence_ind = nn_ind

        self.connect_nodes(
            [nn_ind, self.nodes.shape[0] - 1],
            reference_nodes=False,
            subsections=subsections,
            spacing=spacing,
            boundary_id=boundary_id,
        )

        logging.debug("Segment - Added a line from snap")

    def add_nodes(
        self,
        nodes,
        subsections=None,
        spacing=None,
        is_first_new=True,
        close=True,
        is_polygon=False,
        boundary_id=-1,
        boundary_ids=None,
    ):
        """
        Wrapper around `add_node`. Useful for adding a single polygon.
        Resulting number of connectivity is:
          num_nodes + (is_first_new) - (~close)
          Booleans in () are evaluated 1 if True, 0 if False.

        Parameters
        -----------
        nodes: (n, d) np.ndarray
        subsections: int
        spacing: float
        is_first_new: bool
        close: bool
        is_polygon: bool
        boundary_id: int
          Boundary condition number. Default is -1 for "no-assignment".
          Use Positive values > 0 to assign values.
        boundary_ids: list
          List of boundary_id. Should be the same number as resulting number
          of connectivity. Overwrites boundary_id, if defined.

        Returns
        --------
        None
        """
        nodes = np.asarray(nodes)

        # Get index of the first new connectivity info.
        # Only used if `is_polygon` is true.
        if self.connectivity_ is None:
            new_connectivity_begin = 0
        else:
            new_connectivity_begin = self.connectivity_.shape[0]

        # In case of polygon, don't add any points on the edge
        if is_polygon:
            subsections = 1
            spacing = None
            if self.nodes is None:
                ind_offset = 0
            else:
                ind_offset = self.nodes.shape[0]

        # Prepare boundary_ids
        first_new = 0 if is_first_new else 1
        closing = 0 if close else -1
        num_new_connectivity = int(len(nodes) + first_new + closing)

        # Set boundary_ids
        if boundary_ids is not None:
            assert num_new_connectivity == len(boundary_ids),\
                "Length of boundary_ids is inappropriate."

            boundary_ids = np.asarray(boundary_ids).astype(np.int32).flatten()
            logging.debug("Segment - Applying specified boundary id to each " +\
                "new connectivity.")

        else:
            # Form boundary_ids using boundary_id
            boundary_ids = np.repeat(
                boundary_id, num_new_connectivity
            ).astype(np.int32).flatten()
            logging.debug("Segment - Applying same boundary id {bid} to all " +\
                "new connectivities.".format(bid=boundary_id))

        # Carefully assgin the first boundary_id
        # -1 here is just a dummy, since it won't be applied in that case
        if is_first_new:
            fbid = -1

        else:
            fbid = boundary_ids[0]
            boundary_ids = boundary_ids[1:]

        # Add first node
        self.add_node(
            node=nodes[0],
            subsections=subsections,
            spacing=spacing,
            new_sequence=is_first_new,
            boundary_id=fbid,
        )

        # Add mid-nodes
        # If `close=True`, len(boundary_ids) == len(nodes[1:-1]) + 2 (or +1)
        #   but `zip` will stop at whatever is shorter.
        if len(nodes[1:-1]) != 0:
            for n, b in zip(nodes[1:-1], boundary_ids):
                self.add_node(
                        node=n,
                        subsections=subsections,
                        spacing=spacing,
                        boundary_id=b,
                )

        # Carefully assgin the last two `boundary_id`s
        if close:
            lbid = boundary_ids[-2] 
            cbid = boundary_ids[-1]

        else:
            lbid = boundary_ids[-1]
            cbid = None

        # Add last node
        self.add_node(
            node=nodes[-1],
            subsections=subsections,
            spacing=spacing,
            close_sequence=close,
            boundary_id=lbid,
            closing_boundary_id=cbid,
        )

        logging.debug("Segment - Added {nn} nodes.".format(nn=nodes.shape[0]))
        logging.debug("Segment -   First was new: {n}.".format(n=is_first_new))
        logging.debug("Segment -   Closed the sequence: {c}.".format(c=close))

        # In case of polygon, append freshly added nodes.
        if is_polygon:
            self.polygons_.append(
            #    self.connectivity_[new_connectivity_begin:,0].tolist()
                (np.arange(nodes.shape[0]) + ind_offset).tolist()
            )
            logging.debug(
                "Segment - Polygon with {nv} vertices are added.".format(
                    nv=len(self.polygons_[-1])
                )
            )

            # Set facet BC
            self.add_boundary_id(
                boundary_id=boundary_id,
                num_boundary_id=1,  # One polygon at a time
                facet=True,
            )
                

    def add_polygon(
        self,
        nodes,
        boundary_id=-1,
    ):
        """
        A wrapper around `add_nodes` with `is_polygon=True`. Can't set neither
        spacing nor subsections since added node sequence won't be valid.
        Also, polygon needs to be coplanar.

        Parameters
        -----------
        nodes: (n, d) np.ndarray
        boundary_id: int
          Boundary condition number. Default is -1 for "no-assignment".
          Use Positive values > 0 to assign values.

        Returns
        --------
        None
        """
        self.add_nodes(
            nodes=nodes,
            subsections=1, # Need to be 1, otherwise it won't be a polygon
            is_polygon=True,
            boundary_id=boundary_id,
        )

    def add_mesh(
        self,
        mesh,
        boundary_id=-1,
        destroy_order=True,
    ):
        """
        Turn mesh into nodes, connectivity, and polygon.
        Meshes are clusters of polygons, after all.

        Parameters
        -----------
        mesh: `Mesh`
        boundary_id: int
        destroy_order: bool
          Default is True. If true, just adds nodes and faces as polygons.
          Otherwise it loops through each faces and use `add_polygon` to add
          mesh, which also considers connectivity info. As this function is
          meant to be used in 3D, default is True to boost speed.

        Returns
        --------
        None
        """
        assert hasattr(mesh, ("vertices" and "faces")), "Invalid Mesh type!"

        if not destroy_order:

            # (Probably) slow, but proper.
            # Takes care of connectivity, meaning, also usable in 2D. 
            for f in mesh.faces:
                self.add_polygon(
                    nodes=mesh.vertices[f],
                    subsections=1, # Don't alter anything
                    boundary_id=boundary_id,
                )
            logging.debug("Segment - Succesfully added mesh as `nodes`, "+\
                "`connectivity`, `polygon`.")

        else:
            # Act tough.
            logging.warning("Segment - Destroying ORDNUNG! Adding mesh with "+\
                "`destroy_order=True`")
            logging.warning("Segment - Destroying ORDNUNG! `connectivity` is "+\
                "no more valid.")
            logging.warning("Segment - Destroying ORDNUNG! `reference_node` "+\
                "is no more valid.")
            logging.warning("Segment - Destroying ORDNUNG! "+\
                "`last_sequence_ind` is no more valid.")
            logging.warning("Segment - Destroying ORDNUNG! More stuffs are "+\
                "no more valid.")
            logging.warning("Segment - Destroying ORDNUNG! I hope you only "+\
                "add mesh from now.")

            # Add nodes and polygons
            if self.nodes is None:
                self.nodes = mesh.vertices
                ind_offset = self.nodes.shape[0]

            else:
                ind_offset = self.nodes.shape[0]
                self.nodes_ = np.vstack(
                    (self.nodes_,
                     mesh.vertices)
                )

            self.polygons_.extend(
                (mesh.faces + ind_offset).tolist()
            )

            # And boundary conditions
            self.add_boundary_id(
                boundary_id,
                len(mesh.faces),
                facet=True,
            ) 

    def add_facet_with_holes(
        self,
        polygon_nodes,
        holes,
        boundary_id=-1,
    ):
        """
        This is a special case of facet: it has holes. Forms one of the inputs
        for `tetgen`. Stored separately.

        Format:
          [[hfacepolygonpoints, hfaceholes, boundary_id]]

          hfacepolygonpoints: (float) [[[p1x, p1y, p1z,], ...], ...]
          hfaceholes: (float) [[h1x, h1y, h1z,], ...]
          boundary_id: (int) bid

        Parameters
        -----------
        polygon_nodes: list
          List of list-like objects. len(polygon_nodes) should be number of
          polygons, len(<each-polygon_nodes-elements>) should be number of 
          each polygon vertices.
        holes: (n, 3) list-like
        boundary_id: int

        Returns
        --------
        None
        """

        logging.debug("Segment - Adding facet with holes:")

        fwh = []

        # Append each polygon nodes as np.ndarray
        hfacepolygonpoints = []
        for pn in polygon_nodes:
            hfacepolygonpoints.append(np.asarray(pn))
        fwh.append(hfacepolygonpoints)

        logging.debug("Segment -   Added {nhfp} polygons.".format(
            nhfp=len(hfacepolygonpoints)))


        # Append holes as np.ndarray
        hfaceholes = np.asarray(holes)
        fwh.append(hfaceholes)

        logging.debug("Segment -   Added {nhfh} holes.".format(
            nhfh=len(hfaceholes)))

        # Append boundary id
        fwh.append(boundary_id)
        logging.debug("Segment -   This facet has boundary id: {bid}.".format(
            bid=boundary_id))

        self.facet_with_holes_.append(fwh)


    def add_boundary_id(self, boundary_id, num_boundary_id, facet=False):
        """
        Adds boundary id based on given `boundary_id` and repeats 
        `num_boundary_id` times.

        Parameters
        -----------
        boundary_id: int
        num_boundary_id: int
        facet: bool
          Set True if the segments are for `tetgen`. Default is False.

        Returns
        --------
        None
        """
        boundary_ids = (
            np.ones(num_boundary_id, dtype=np.int32) 
            * int(boundary_id)
        ).tolist()

        if not facet:
            self.boundary_edge_ids_.extend(boundary_ids)
            logging.debug(
                "Segment - Boundary ID ({bid}) set for {nc} connectivities.".format(
                    bid=boundary_id,
                    nc=num_boundary_id,
                )
            )

        else:
            self.boundary_facet_ids_.extend(boundary_ids)
            logging.debug(
                "Segment - Boundary ID ({bid}) set for {nc} connectivities.".format(
                    bid=boundary_id,
                    nc=num_boundary_id,
                )
            )

    def connect_nodes(
        self,
        indices,
        reference_nodes=True,
        subsections=None,
        spacing=None,
        boundary_id=-1,
    ):
        """
        Connect given 2 indices and appropriately adjusts nodes and connectivity.
        As you can't set BCs without connectivity, `boundary_id` is processed
        here.

        Parameters
        -----------
        indices: list or tuple of int
          Length 2. 
        reference_nodes: bool
          Default is True. If False, indices refer to absolute node index.
        subsections: int
        spacing: float

        Returns
        --------
        None
        """
        assert len(indices) == 2, "I can only connect two nodes at a time."

        # Get first/last indices and vertices
        if reference_nodes:
            ind_first = self.reference_nodes[indices[0]]
            ind_last = self.reference_nodes[indices[1]]
        else:
            ind_first = indices[0]
            ind_last = indices[1]
        vertex_first = self.nodes[ind_first]
        vertex_last = self.nodes[ind_last]

        # Get subsections
        subsections = self.compute_subsections_(
            subsections,
            spacing,
            compute_distance(vertex_first, vertex_last)
        )

        # Get nodes and connectivity
        new_nodes = np.linspace(
            vertex_first,
            vertex_last,
            (subsections+1)
        )[1:-1]

        num_additional_nodes = len(new_nodes)
        logging.debug(
            'Segment - Connecting node with {nan} additional node(s)'.format(
                nan=num_additional_nodes
            )
        )

        if num_additional_nodes == 0:
            new_connectivity = np.array([[ind_first, ind_last]])
        elif num_additional_nodes == 1:
            new_connectivity = np.array(
                [[ind_first, self.nodes.shape[0]],
                 [self.nodes.shape[0], ind_last]]
            )
        else:
            last_added_node_ind = self.nodes.shape[0] + num_additional_nodes - 1

            first_con = np.array([[ind_first, self.nodes.shape[0]]])
            mid_con = utils.open_loop_index_train(
                (self.nodes.shape[0], # This points to the first new node.
                 last_added_node_ind)
            )
            last_con = np.array(
                [[last_added_node_ind, ind_last]]            
            )
            new_connectivity = np.vstack(
                (first_con,
                 mid_con,
                 last_con)
            )

        # Stack them!
        if num_additional_nodes >= 1:
            self.nodes_ = np.vstack(
                (self.nodes,
                 new_nodes)
            )

        if self.connectivity is None:
            self.connectivity = new_connectivity
        else:
            self.connectivity = np.vstack(
                (self.connectivity,
                 new_connectivity)
            )

        # Take care of boundary ids
        num_boundary_ids = len(new_connectivity)
        self.add_boundary_id(boundary_id, num_boundary_ids, facet=False)

    def snap_connect_nodes(
        self,
        two_nodes,
        subsections=None,
        spacing=None,
        boundary_id=-1,
    ):
        """
        Snap connect two nodes. This is quite safe. Just make sure that the
        segment doesn't already exist. That we don't check.

        Parameters
        -----------
        two_nodes: (2, d) np.ndarray
        subsections: int
        spacing: float

        Returns
        --------
        None
        """
        two_nodes = np.asarray(two_nodes)
        if two_nodes.shape[0] != 2:
            raise ValueError("`two_nodes` takes two nodes only.")

        # Abuse KDT since it is fast enough for this.
        kdt = KDTree(self.nodes_)
        _, nn_ind = kdt.query(two_nodes) # there are two indices
        logging.debug(
            "Segment - Snapping and connecting [{ni}] nodes: {n}.".format(
                ni=nn_ind,
                n=self.nodes[nn_ind]
            )
        )

        self.connect_nodes(
            nn_ind,
            reference_nodes=False,
            subsections=subsections,
            spacing=spacing,
            boundary_id=boundary_id,
        )

    def compute_subsections_(self, subsections, spacing, distance=None):
        """
        Internal function for computing subsections based on the value of
        `subsections` or `spacing`.

        Paramters
        ----------
        subsections: int
        spacing: float

        Returns
        --------
        subsections: int
          Overwrites `subsections` value.
        """
        if spacing is not None and distance is None:
            raise ValueError("For spacing, you need to specify distance.")

        if subsections is None and spacing is None:
            logging.debug("Segment - Both `subsections` and `spacing` are "+\
                "None. Applying global `subsections` or `spacing` if defined.")
            subsections = self.global_subsections_
            spacing = self.global_spacing_

        if subsections is not None and spacing is not None:
            if subsections != 1:
                logging.warning("Segment - Both `subsections` and `spacing`"+\
                    "are specified. Taking `spacing`.")

        # Spacing is actually "smart" subsection.
        if spacing is not None:
            subsections = int(distance // spacing)
            # We can't take 0 subsections.
            if subsections == 0:
                subsections = 1
            logging.debug(
                "Segment - Subsections ({sub}) Computed from spacing.".format(
                    sub=subsections
                )
            )

        else:
            logging.debug(
                "Segment - Subsections ({sub})".format(sub=subsections)
            )

        return subsections


    @property
    def global_spacing(self,):
        """
        Returns global spacing.

        Parameters
        -----------
        None

        Returns
        --------
        self.global_spacing_: float
        """
        return self.global_spacing_

    @global_spacing.setter
    def global_spacing(self, global_spacing):
        """
        Set global spacing.

        Parameters
        -----------
        global_spacing: float

        Returns
        --------
        None
        """
        self.global_spacing_ = global_spacing
        logging.debug(
            "Segment - Gloabal spacing: {gs}".format(gs=self.global_spacing_)
        )

    @property
    def global_subsections(self,):
        """
        Returns global subsections.

        Parameters
        -----------
        None

        Returns
        --------
        self.global_subsections_: float
        """
        return self.global_subsections_

    @global_subsections.setter
    def global_subsections(self, global_subsections):
        """
        Set global subsections.

        Parameters
        -----------
        global_spacing: float

        Returns
        --------
        None
        """
        self.global_subsections_ = global_subsections
        logging.debug(
            "Segment - Gloabal subsection: {gs}".format(
                gs=self.global_subsections_
            )
        )

    @property
    def last_sequence_ind(self,):
        """
        Returns last sequnce ind.
        Intended to be used internally, but not legally binding prerequisite.

        Parameters
        -----------
        None

        Returns
        --------
        self.last_sequence_ind_: int
        """
        return self.last_sequence_ind_

    @last_sequence_ind.setter
    def last_sequence_ind(self, last_sequence_ind):
        """
        Sets last sequence ind.

        Parameters
        -----------
        last_sequence_ind: int

        Returns
        --------
        self.last_sequence_ind_
        """
        self.last_sequence_ind_ = last_sequence_ind
        logging.debug(
            "Segment - Last sequence ind: {lsi}".format(
                lsi=self.last_sequence_ind
            )
        )

    def add_interactive(
        self,
        grid=None,
        snap=False,
        references=True,
        polygon=True,
    ):
        """
        Adds polygon or lines interactively.
        Supports 2D.
        Snap does not snap to the reference nodes. Try not to duplicate nodes.
        This also does not set BCs. As interactive drawing only supports 2D,
        it should also be possible to add BCs interactively at mesh level.

        Parameters
        -----------
        grid: list
          [[x_min, x_max], [y_min, y_max], [x_resolution, y_resolution]]
        snap: bool
        reference: bool
        polygon: bool

        Returns
        --------
        None
        """
        drawer = InteractiveDrawer()

        if references:
            ref = []

            if self.nodes_ is not None:
                ref.append(self.nodes_)

                if self.connectivity_ is not None:
                    ref.append(self.connectivity_)

            else:
                ref = None

        drawer.polygon(
            grid=grid,
            snap=snap,
            show_edges=True,
            references=ref,
            close_=polygon,
        )

        if self.connectivity_ is not None:
            self.connectivity_ = np.vstack(
                (self.connectivity_,
                 drawer.connectivity + self.nodes_.shape[0])
            )

        else:
            self.connectivity_ = drawer.connectivity.copy()

        if self.nodes_ is not None:
            self.nodes_ = np.vstack(
                (self.nodes_,
                 drawer.nodes)
            )

        else:
            self.nodes_ = drawer.nodes.copy()

        logging.debug("Segment - Added {nn} nodes interactively.".format(
            nn=drawer.nodes.shape[0]))
        logging.debug("Segment - Added {nc} connectivity interactively.".format(
            nc=drawer.connectivity.shape[0]))

    def show(self):
        """
        Shows segment. Supports 2D.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if self.nodes_ is None:
            logging.debug("Segment - Nothing to show. Skipping.")
            return

        if len(self.polygons_) != 0:
            logging.debug("Segment - Showing 3D Segments using `vedo`.")
            logging.warning("Segment - Showing 3D Segments can be slow!.")

            import vedo

            points = vedo.Points(self.nodes)
            lines = []
            for p in self.polygons:
                p = np.asarray(p).astype(np.int32)
                lines.append(vedo.Line(self.nodes[p]))

            vedo.show([points, *lines]).show().close()

        else:
            logging.debug("Segment - Showing 2D Segments using `matplotlib`.")

            import matplotlib.pyplot as plt

            plt.scatter(
                self.nodes_[:, 0],
                self.nodes_[:, 1],
                c="pink",
                zorder=1000,
            )

            for c in self.connectivity_:
                plt.plot(
                    self.nodes_[c][:,0],
                    self.nodes_[c][:,1],
                    c="grey",
                    lw=2,
                    zorder=10,
                )

            plt.show()


def compute_distance(node1, node2):
    """
    Computes distance between two nodes.

    Parameters
    -----------
    node1: (n, ) np.ndarray
    node2: (n, ) np.ndarray

    Returns
    --------
    distance: float
    """
    return np.linalg.norm(node1 - node2)
