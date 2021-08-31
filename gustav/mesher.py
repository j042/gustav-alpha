import logging
#import triangle
from . import segment as s 
from . import mesh as m
from .interactive_selector import InteractiveSelector
import numpy as np

class MeshMaker:
    def __init__(self,
        segment=None,
        segments=None,
    ):
        """
        MeshMaker.
        Wraps some crucial functions from segments.

        Parameters
        -----------
        segment: Segment

        Attributes
        -----------
        segment_: `Segment`
        segments_: list of `Segment`
        final_nodes_: np.ndarray
        final_connectivity: np.ndarray
        meshes_: list of `Mesh`
        pyt_: list of `pytriangle`
        pyt_input_: list of dict
        interactive_selector_: `InteractiveSelector`
        """
        self.segment_ = s.Segment()
        self.segments_ = []
        self.in_mesh_ = []
        self.meshes_ = []
        self.pyt_ = []
        self.pyt_input_ = []

    @property
    def nodes(self,):
        """
        `Segment.nodes` wrapper. Returns Nodes.

        Parameters
        -----------
        None

        Returns
        --------
        self.segment_.nodes: (n,m) np.ndarray
        """
        return self.segment_.nodes


    @nodes.setter
    def nodes(self, nodes):
        """
        `Segment.nodes` setter wrapper. Sets Nodes.

        Parameters
        -----------
        nodes: (n,m) np.ndarray

        Returns
        --------
        None
        """
        self.segment_.nodes = nodes

    def add_node(self, node, **kwargs):
        """
        `Segment.add_node` wrapper. Adds a node.

        Parameters
        -----------
        node: (n,) np.ndarray
        subsections: int
        spacing: double
        close_loop: bool
          Should this node close the segment?
          Use with care. If used with
        new_sequence: bool
          Start a new sequence of lines.

        Returns
        --------
        None
        """
        self.segment_.add_node(node, **kwargs)

    @property
    def connectivity(self,):
        """
        `Segment.connectivity` wrapper. Returns connectivity.

        Parameters
        -----------
        None

        Returns
        --------
        self.segment_.connectivity_: (n,m) np.ndarray
        """
        return self.segment_.connectivity_

    @connectivity.setter
    def connectivity(self, connectivity):
        """
        `Segment.connectivity` setter wrapper. Connectivity setter.

        Parameters
        -----------
        Connectivity

        Returns
        --------
        None
        """
        # Is connectivity referring to valid indices?
        assert self.segment_.nodes.shape[0] == int(connectivity.max() + 1),\
            "Your connectivity array is referring to an index that does not exist."
        self.segment_.connectivity_ = connectivity


    @property
    def segment(self,):
        """
        Returns segment.

        Parameters
        -----------
        None

        Returns
        --------
        self.segment_: `Segment`
        """
        return self.segment_

    @segment.setter
    def segment(self, segment):
        """
        Segment setter. 

        Parameters
        -----------
        segment: `Segment`

        Returns
        --------
        None
        """
        self.segment_ = segment
        logging.debug('MeshMaker - Segment Set.')

    @property
    def segments(self,):
        """
        Returns Segments.

        Parameters
        -----------
        None

        Returns
        --------
        self.segments_: list
        """
        return self.segments_

    @segments.setter
    def segments(self, segments):
        """
        Segments setter.

        Paramters
        ----------
        segments: list
          list of `Segment`s
        
        Returns
        --------
        None
        """
        self.segments_ = segments
        logging.debug('MeshMaker - {n_seg} segments set.'.format(n_seg=len(segments)))

    def add_segment(self, segment):
        """
        Append a segment to the segments list.

        Parameters
        -----------
        segment: `Segment`

        Returns
        --------
        None
        """
        self.segments_.append(segment)
        logging.debug(
            'MeshMaker - Segment with {n_nodes} nodes and{n_connectivity} added.'.format(
                n_nodes=segment.nodes.shape[0],
                n_connectivity=segment.connectivity.shape[0]
            )
        )

    def add_mesh(self, mesh, destroy_order=False):
        """
        Append mesh to `in_mesh` and process it as polygon.
        Then append it to `self.segment_.polygons_` and `self.segment_.nodes`

        Parameters
        -----------
        mesh: `Mesh`

        Returns
        --------
        None
        """
        assert hasattr(mesh, ("vertices" and "faces")), "Invalid Mesh type!"

        # Save mesh. (For now, purely because "why not?".)
        self.in_mesh_.append(mesh)

        self.segment_.add_mesh(mesh=mesh, destroy_order=destroy_order)

    @property
    def global_spacing(self,):
        """
        `Segment.global_spacing` wrapper. Returns global spacing.

        Parameters
        -----------
        None

        Returns
        --------
        self.segment_.global_spacing_: double
        """
        return self.segment_.global_spacing_

    @global_spacing.setter
    def global_spacing(self, global_spacing):
        """
        `Segment.global_spacing` setter wrapper. Set global spacing.

        Parameters
        -----------
        global_spacing: double

        Returns
        --------
        None
        """
        self.segment_.global_spacing_ = global_spacing

    @property
    def global_subsections(self,):
        """
        `Segment.global_subsections` wrapper. Returns global subsections.

        Parameters
        -----------
        None

        Returns
        --------
        self.segment_.global_subsections_: double
        """
        return self.segment_.global_subsections_

    @global_subsections.setter
    def global_subsections(self, global_subsections):
        """
        `Segment.global_subsections` setter wrapper. Set global subsections.

        Parameters
        -----------
        global_spacing: double

        Returns
        --------
        None
        """
        self.segment_.global_subsections_ = global_subsections

    @property
    def mesh_type(self,):
        pass

    @mesh_type.setter
    def mesh_type(self, **kwargs):
        pass


    def run(self, **kwargs):
        pass

    def show(self, mesh_ind):
        """
        """
        # maybe this should be up above.
        import matplotlib.pyplot as plt
        import triangle

        triangle.compare(
            plt,
            self.pyt_input_[mesh_ind],
            self.pyt_[mesh_ind]
        )
        plt.show()

    def remove_vertices(self, mesh_ind=0, method="interactive", fig_size=(10,10)):
        """
        Remove from a given mesh. This only supports 2D.

        Parameters
        -----------
        mesh_ind: int
        method: str

        Returns
        --------
        new_mesh: `Mesh`
        """
        v_remove_ind = self.meshes_[mesh_ind].select_vertices(
            method=method,
            fig_size=fig_size,
        )
        logging.debug("MeshMaker - {num_v} vertices will be "+\
            "removed.".format(num_v=len(v_remove_ind)))

        return self.meshes_[mesh_ind].remove_vertices(v_remove_ind)

    def remove_faces(
        self,
        mesh_ind=0,
        method="interactive",
        fig_size=(10,10),
        return_both=False,
        optimize=None,
        PSLG="reference",
    ):
        """
        Remove faces from a given mesh. Only supports 2D.

        Parameters
        -----------
        mesh_ind: int
        method: str
        fig_size: (n, m) tuple
        return_both: bool
        PSLG: str
          <"only" | "reference">

        Returns
        --------
        new_mesh: `Mesh`
        OR
        rest_mesh: `Mesh`
        removed_mesh: `Mesh`
        """

        # Form pslg_in from PSLG
        if PSLG is None:
            pslg_in = None
        else:
            pslg_in = [self.segment_.nodes, self.segment_.connectivity]

        if PSLG == "only":
            pslg_in.append("drehdinetum") # <- dummy third element in list

        f_remove_ind = self.meshes_[mesh_ind].select_faces(
            method=method,
            fig_size=fig_size,
            PSLG=pslg_in
        )
        logging.debug("MeshMaker - {num_f} faces will be removed.".format(
            num_f=len(f_remove_ind))
        )

        if not return_both:
            
            return self.meshes_[mesh_ind].remove_faces(
                f_remove_ind,
                optimize=optimize
            )

        else:
            rest_mesh = self.meshes_[mesh_ind].remove_faces(
                f_remove_ind,
                optimize=optimize
            )
            f_total = self.meshes_[mesh_ind].faces_.shape[0]
            inverted_mask = np.ones(f_total, dtype=bool)
            inverted_mask[f_remove_ind] = False
            removed_mesh = self.meshes_[mesh_ind].remove_faces(
                np.arange(f_total)[inverted_mask],
                optimize=optimize
            )

            return rest_mesh, removed_mesh

    def triangulate_(
        self,
        holes=None,
        regions=None,
        PSLG=True,
        min_angle=30,
        max_area=None,
        convex_hull=False,
        conforming_delaunay=False,
        exact_arithmetic=True,
        steiner_points=None,
        incremental_algorithm=False,
        sweepline_algorithm=False,
        vertical_cuts_only=False,
        segment_splitting=False,
        check_consistency=False,
        return_neighbors=False,
        return_edges=False,
        optimize=None,
    ):
        """
        Triangulation wrapper of `pytriangle`, that wraps `triangle`, 
        from Jonathan Richard Shewchuk.
        If the function does not terminate, use smaller `min_angle`.
        Default algorithm is `divide-and-conquer`.

        `max_area` can take up to 12 digits after point.

        Parameters
        -----------
        optimize: str or list
          str -> method
          list -> [method(str), tolerance(float), iteration(int)]
        holes: (n, 3) list-like
        regions: (n, 4) list-like,
        PSLG: bool
          Default is True.
        min_angle: float
          Default is 30. If it does not terminate, bring it own to 20.
        max_area: float
        convex_hull: bool
          Default is False.
        conforming_delaunay: bool
          Default is False.
        exact_arithmetic: bool
          Default is True.
        steiner_points: int
        incremental_algorithm: bool
          Default is False.
        sweepline_algorithm: bool
          Default is False.
        vertical_cuts_only: bool
          Default is False.
        segment_splitting: bool
          Default is False.
        check_consistency: bool
          Default is False.
        return_neighbors: bool
          Default is False.
        return_edges: bool
          Default is False.
        optimize: list
          Format is - 
          [str(optimization_type), float(tolerance), int(num_iteration)]

        Returns
        --------
        tri_mesh: `Mesh`
        """
        import triangle

        nodes, connectivity = self.finalize_()        

        options = ""

        tri = dict(vertices=nodes)
        tri.update(segments=connectivity)
        if regions is not None:
            tri.update(regions=regions)
            options += "a"
            
        if len(self.segment_.boundary_edge_ids) != 0:
            # Prepare boundary markers
            segment_markers = np.ascontiguousarray(
                self.segment_.boundary_edge_ids,
                dtype=np.int32,
            ).flatten()

            # Add it to `tri`
            tri.update(segment_markers=segment_markers)

        if PSLG:
            options += "p"
        if min_angle is not None:
            options += "q" + str(min_angle)
        if max_area is not None:
            options += "a" + "{:.12f}".format(max_area)
        if convex_hull:
            options += "c"
        if conforming_delaunay:
            options += "D"
        if not exact_arithmetic:
            options += "X"
        if steiner_points is not None:
            options += "S" + str(steiner_points)
        if incremental_algorithm:
            options += "i"
        if sweepline_algorithm:
            options += "F"
        if vertical_cuts_only:
            options += "l"
        if segment_splitting:
            options += "s"
        if check_consistency:
            options += "C"
        if return_neighbors:
            options += "n"
        if return_edges:
            options += "e"

        # for loop for holes before execution

        if holes is not None:
            tri.update(holes=holes)

        self.pyt_input_.append(tri)
        pytri = triangle.triangulate(tri, options)
        self.pyt_.append(pytri)
        mesh = m.Mesh(vertices=pytri['vertices'], faces=pytri['triangles'])

        # Thank you KDT
        # Orders of BCs is different. So we need a mapping
        from scipy.spatial import cKDTree as KDT

        bcs = pytri["segment_markers"]
        bcs = bcs.astype(np.int32)
        unique_bcs = np.unique(bcs).astype(np.int32)
        unique_bcs.sort() # This is probably redundant
        # Let's skip -1
        logging.debug("MeshMaker - Skipping boundary id `-1`.")
        unique_bcs = unique_bcs[1:] if int(unique_bcs[0]) == int(-1) else unique_bcs

        # Get outline edges from generated mesh
        tri_outlines = mesh.edges[mesh.outlines].copy().astype(np.int32)
        tri_outlines.sort(axis=1)
        # Get edges that was formed from segments
        #   - this still includes internal edges
        bc_edges = pytri["segments"]
        bc_edges = bc_edges.astype(np.int32)
        bc_edges.sort(axis=1)

        # This does "other-way-around" compared to tetgen.
        kdt = KDT(bc_edges)
        dist, ind = kdt.query(tri_outlines)

        assert dist.sum() < 1E-10,\
            "Something went wrong while finding edge matches " +\
            "for boundary conditions."

        # Now, apply bcs
        #
        # First, we need to reorganize/reselect bcs.
        #   - hint: len(bc_edges) == len(bcs)
        bcs = bcs[ind].flatten() # now, len(tri_outlines) == len(bcs)

        self.bcs = bcs
        self.o = mesh.outlines

        # Really apply
        for bc in unique_bcs:
            if not (bcs == bc).any():
                continue

            mesh.set_BC(
                name="BC" + str(bc),
                method="index",
                criteria=mesh.outlines[bcs==bc],
            )

        # Optimize, if you wish
        if optimize is not None:
            opt_mesh = m.optimize_mesh(mesh, optimize)
            mesh.vertices = opt_mesh.vertices.copy()
        
        self.meshes_.append(mesh)

        return mesh

    def tetrahedralize_(
        self,
        preserve_input=False,
        quality=True,
        max_volume=None,
        optimize=None,
        max_steiner_points=None,
        coplanar_tolerance=None,
        convex_hull=False,
        check_consistency=False,
        quadratic_elements=False,
        elements_per_memory_block=None,
        detect_self_intersection=False,
        verbose=False,
        quiet=False,
        exact_arithmetic=True,
        merge_duplicates=True,
        switches=None,
        fmarkers=None,
        holes=None,
        regions=None,
    ):
        """
        Tetrahedralization wrapper based on `pyvista/tetgen` that wraps `tetgen`
        from Hang Si.
        All parameters are optional.

        Parameters
        -----------
        preserve_input: bool
          Default is False.
        quality: bool
        max_volume: float
        optimize: int
          Between 0 and 10
        max_steiner_points: int
          Maximum number of additional steiner points
        coplanar_tolerance: float
          Default is None. Internal default is 1e-8.
        convex_hull: bool
          Default is False.
        check_consistency: bool
          Default is False. Prints delauany-ness.
        quadratic_elements: bool
          Default is False. If True,  Tets with 10 nodes. Since our `Mesh` does
          not support this, this will return (nodes, elements, bcs) separately.
        elements_per_memory_block: int
          Default is None. Internal default is 8188. Increase it in if you are
          making a huuuuge mesh.
        detect_self_intersection: bool
          Default is False. If True, it does not tetrahedralize. Instead prints
          info about self intersection.
        verbose: bool
        quiet: bool
        exact_arithmetic: bool
          Default is True.
        merge_duplicates: bool
          Default is True. Merges duplicating PLC and vertices.
        switches: str
          If you want to bypass all the params and know what you want, use this.

        Returns
        --------
        tet_mesh: `Mesh`
          Only if `quadriatic_elements=False`.
        nodes: (n, 3) np.ndarray
          Only if `quadratic_elements=True`.
        elements: (m, 10) np.ndarray
          Only if `quadratic_elements=True`.
        bc: (l,) np.ndarray
          Only if `quadratic_elements=True`.
        faces: (k, 3) np.ndarray
          Only if `quadratic_elements=True`.
        """
        import tetgen

        if switches is None:
            switches = ""
            if regions is not None:
                switches +="aA"
            if preserve_input:
                switches += "Y"
            if quality:
                switches += "q"
            if max_volume:
                switches += "a"
                switches += str(max_volume) # It takes scientific notation with `e`
            if optimize:
                switches += "O"
                switches += str(optimize)
            if max_steiner_points:
                switches += "S"
                switches += str(max_steiner_points)
            if coplanar_tolerance:
                switches += "T"
                switches += str(coplanar_tolerance)
            if convex_hull:
                switches += "c"
            if check_consistency:
                switches += "CC" # Could also do one C.
            if quadratic_elements:
                switches += "o2"
            if elements_per_memory_block:
                switches += "x"
                switches += str(elements_per_memory_block)
            if verbose:
                switches += "V"
            if quiet:
                switches += "Q"
            if detect_self_intersection:
                switches += "d"

            if not exact_arithmetic:
                switches += "X"
            if not merge_duplicates:
                switches += "M"

            # Switches that currently won't have any effects
            #
            #if assign_attributes:
            #    switches += "A"
            #if reconstruct:
            #    switches += "r"
            #if coarsen:
            #    switches += "R"
            #if mesh_sizing_function:
            #    switches += "-m"
            #if additional_points:
            #    switches += "-i"
            #if weighted_delaunay:
            #    switches += "w"
            #if no_iteration_numbers:
            #    switches += "I"

        bytes_switches = bytes(switches, "utf-8")

        # Currently only supports a single segment use
        if len(self.segment_.polygons_) != 0:
            n_vertices_per_face = np.asarray(
                [len(p) for p in self.segment_.polygons_]
            )
            # flattened polygon (facet) list
            f = np.asarray(
                [p for polygon in self.segment_.polygons_ for p in polygon]
            )

            # Form fmarkers - function param overrides whatever
            if fmarkers is None and len(self.segment_.boundary_facet_ids) != 0:
                fmarkers = np.asarray(
                    self.segment_.boundary_facet_ids
                ).flatten()
            else:
                fmarkers = np.asarray(fmarkers).flatten()

            # Form hfaces (facets with holes)
            hfv = None # hfacepoints
            nhf = None # nhfaces 
            nhfp = None # nhfacepolygons
            pperhf = None # nhfacepolygonpoints
            hfp = None # hfacepolygons
            nhfh = None # nhfaceholes -> h per hfaces
            hfh = None # hfaceholes
            hfmarkers = None
            if len(self.segment_.facet_with_holes) != 0:

                # For polygon offset
                global_start_ind = len(self.segment_.nodes)
                local_start_ind = 0

                nhf = len(self.segment_.facet_with_holes)
                nhfp = []
                pperhf = []
                hfp = []
                nhfh = []
                hfmarkers = []

                for fwh in self.segment_.facet_with_holes:

                    # Take care of facet polygons
                    nhfp.append(len(fwh[0]))

                    for hfacepolygonpoints in fwh[0]:
                        pperhf.append(len(hfacepolygonpoints))

                        if hfv is None:
                            hfv = hfacepolygonpoints

                        else:
                            hfv = np.vstack((hfv, hfacepolygonpoints))

                        hfp.extend(
                            (np.arange(int(pperhf[-1]))
                             + global_start_ind
                             + local_start_ind).tolist()
                        )
                        local_start_ind += pperhf[-1]

                    # Take care of facet holes
                    nhfh.append(len(fwh[1]))
                    if hfh is None:
                        hfh = fwh[1]
                    else:
                        if len(fwh[1]) == 0:
                            pass

                        else:
                            hfh = np.vstack((hfh, fwh[1]))

                    # Take care of boundary id
                    hfmarkers.append(fwh[2])

            else:
                pass

            vertices, elements, bcs, tfs = tetgen.Tetrahedralize(
                v=self.segment_.nodes,
                f=f,
                nf=len(self.segment_.polygons_),
                vperf=n_vertices_per_face,
                switches=bytes_switches,
                fmarkers=fmarkers,
                holes=holes,
                regions=regions,
                hfv=hfv,
                nhf=nhf,
                nhfp=nhfp,
                pperhf=pperhf,
                hfp=hfp,
                nhfh=nhfh,
                hfh=hfh,
                hfmarkers=hfmarkers,
            )

            # Return raw data if `quadratic_elements=True`
            if quadratic_elements:
                return vertices, elements, bcs, tfs

            # Convert this to internal mesh
            tet_mesh = m.Mesh(vertices=vertices, elements=elements)

            # Return if there're no BCs 
            if (bcs < 0).all():
                return tet_mesh

            # Thank you KDT
            # Orders of BCs is different. So we need a mapping
            from scipy.spatial import cKDTree as KDT

            bcs = bcs.astype(np.int32).flatten()
            unique_bcs = np.unique(bcs)
            unique_bcs.sort() # This is probably redundant
            # Let's skip -1
            logging.debug("MeshMaker - Skipping boundary id `-1`.")
            unique_bcs = unique_bcs[1:] if unique_bcs[0] == -1 else unique_bcs


            # Surface mesh only would be faster, but needs more mapping.
            # And this is still fast, so far.
            tet_faces = tet_mesh.faces.copy().astype(np.int32)
            tet_faces.sort(axis=1)
            bc_faces = tfs.astype(np.int32)
            bc_faces.sort(axis=1)

            kdt = KDT(tet_faces)
            dist, ind = kdt.query(bc_faces)

            assert dist.sum() < 1E-10,\
                "Something went wrong while finding face matches " +\
                "for boundary conditions."

            # Now, apply bcs
            for bc in unique_bcs:
                tet_mesh.set_BC(
                    name="BC" + str(bc),
                    method="index",
                    criteria=ind[bcs==bc],
                )

            return tet_mesh

        else:
            vertices, elements = tetgen.Tetrahedralize(
                v=self.nodes,
                f=self.connectivity,
                switches=bytes_switches,
            )
                
            #tet_mesh = m.Mesh(vertices=vertices, elements=elements)
            return m.Mesh(vertices=vertices, elements=elements)

        #return tet_mesh, bcs, tfs

    @property
    def mesh(self,):
        """
        Returns the latest mesh.

        Parameters
        -----------
        None

        Returns
        --------
        self.meshes_[-1]: `Mesh`
        """
        return self.meshes_[-1]

    def rectangulate_(self,):
        pass

    def finalize_(self,):
        if len(self.segments_) == 0:
            return self.segment_.nodes, self.segment_.connectivity

#    def segments_from_interactive(self, **kwargs):
#        pass
#
#    def segments_from_PSLG(self, **kwargs):
#        pass
#
#    @segments_subsections.setter
#    def segments_subsections(self, **kwargs):
#        pass
#
#    @property
#    def segments_subsections(self, **kwargs):
#        pass
