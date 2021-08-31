import gustav as gus
import numpy as np
import vedo

if __name__ == "__main__":
    gus.utils.configure_logging(debug=True)

    ############ Box Params #############
    shape_scale = .0315/10
    box_x = .0315
    box_y = .0185 * 1.3
    box_z = .0075
    in_window_x_len = .0315 / 3
    in_window_x_res = 30
    in_window_z_len = .0075 / 1.5
    in_window_z_res = 20
    out_window_fine_res = [50, 20]
    out_window_coarse_res = [10, out_window_fine_res[1]]
    shape_offset = [0, -.00555/2, 0]
    fine_box_x_ratio = 1.5 * 1.5 # finebox_x : micschelement_bbox
    fine_box_area = 0.00000000005,
    coarse_box_area = 0.000000001,
    #####################################

    mid_check = []
    regions = []

    ########################
    # Mischelement Process #
    ########################

    # Import mischelement
    mix = gus.load_mesh("periodic.ply")
    mix = mix.remove_unreferenced_vertices()
    # Scale and apply offset
    mix.vertices *= shape_scale
    # First center mesh at origin before applying offset
    mix.vertices -= gus.utils.bounding_box_centroid(mix.vertices)
    mix.vertices += shape_offset
    # Bring it to z=0
    mix.vertices[:, 2] -= mix.vertices[:, 2].min()

    # Extract outline polygon - This case, adapter plate's outline.
    #mix_o = mix.outlines
    mix_e = mix.edges[mix.outlines]
    # Enforce outline edges to be coplanar
    uniq = np.unique(mix_e)
    for u in uniq:
        mix.vertices[u, 2] = mix.vertices[:, 2].min()

    mix_o_polygon = gus.utils.outline_to_line(mix.vertices, mix_e)

    mid_check.append(mix.vedo_mesh)

    ############
    # Fine box #
    ############

    # Mid box
    fine_box = gus.mesh_shapes.box3d(
        bounds=gus.utils.bounds(mix.vertices),
        resolutions=[2, 2, 2],
    )

    # Adjust x_vertices
    fine_box.vertices[:, 0] *= fine_box_x_ratio
    # Adjust y_vertices
    # First "front" then "back"
    fine_box.vertices[[0,1,4,5], 1] = -1 * box_y / 2
    fine_box.vertices[[2,3,6,7], 1] = box_y / 2
    # Adjust z_vetices
    # Top only, bottom should be at 0
    fine_box.vertices[[4,5,6,7], 2] = box_z

    mid_check.append(fine_box.vedo_mesh)

    regions.append(
        [
            *(fine_box.vertices[0] + [.0001, .0001, .0001]),
            1,
            *fine_box_area, # don't know why, but it was being added as tuple.
        ]
    )

    #############
    # In Window #
    #############
    # Find center
    in_window_center = gus.utils.bounding_box_centroid(
        fine_box.vertices[fine_box.faces[1]]
    )

    corner_offset = np.array([in_window_x_len / 2, 0, in_window_z_len / 2])

    in_window_bounds = [
        in_window_center - corner_offset,
        in_window_center + corner_offset,
    ]

    in_window = gus.mesh_shapes.window(
        bounds=in_window_bounds,
        resolutions=[in_window_x_res, in_window_z_res],
        quad=False,
    )

    mid_check.append(in_window.vedo_mesh)

    ###################
    # Out Window Fine #
    ###################
    
    out_window_fine = gus.mesh_shapes.window(
        bounds=fine_box.vertices[[2, 7]],
        resolutions=out_window_fine_res,
        quad=False,
    )

    mid_check.append(out_window_fine.vedo_mesh)

    ########################################
    # Coarse box "left" and its out window #
    ########################################
    left_coarse_box = gus.mesh_shapes.box3d(
        bounds=[
            [-1 * box_x / 2, *fine_box.vertices[0, 1:]],
            fine_box.vertices[6]
        ],
        resolutions=[2, 2, 2],
    )

    mid_check.append(left_coarse_box.vedo_mesh)

    left_out_window = gus.mesh_shapes.window(
        bounds=left_coarse_box.vertices[[2,7]],
        resolutions=out_window_coarse_res,
    )

    mid_check.append(left_out_window.vedo_mesh)

    regions.append(
        [
            *(left_coarse_box.vertices[0] + [.0001, .0001, .0001]),
            1,
            *coarse_box_area
        ]
    )


    ########################################
    # Coarse box "right" and its out window #
    ########################################
    right_coarse_box = gus.mesh_shapes.box3d(
        bounds=[
            fine_box.vertices[1],
            [box_x / 2, *fine_box.vertices[7, 1:]]
        ],
        resolutions=[2,2,2],
    )

    mid_check.append(right_coarse_box.vedo_mesh)

    right_out_window = gus.mesh_shapes.window(
        bounds=right_coarse_box.vertices[[2,7]],
        resolutions=out_window_coarse_res,
    )

    mid_check.append(right_out_window.vedo_mesh)

    regions.append(
        [
            *(right_coarse_box.vertices[0] + [.0001, .0001, .0001]),
            3,
            *coarse_box_area
        ]
    )

    vedo.show(mid_check).close()

    #################
    # Build Segment #
    #################
    # RNG 1 : Inflow
    # RNG 2 : Right wall
    # RNG 3 : Outflow
    # RNG 4 : Left wall
    # RNG 5 : Top
    # RNG 6 : Bottom - non Spline area
    # RNG 7 : Bottom - Spline area
    # RNG 8 : Small portion of inflow

    s = gus.Segment()

    ###########
    # Mid Box #
    ###########
    # Top mid
    s.add_polygon(
        nodes=[
            fine_box.vertices[5],
            *out_window_fine.boundary[2],
            fine_box.vertices[4],
        ],
        boundary_id=5,
    )

    # Out mid
    s.add_mesh(out_window_fine, boundary_id=3)

    # Mid left
    s.add_polygon(
        nodes=[
            fine_box.vertices[4],
            *out_window_fine.boundary[3],
            fine_box.vertices[0]
        ]
    )

    # Mid right
    s.add_polygon(
        nodes=[
            fine_box.vertices[1],
            *out_window_fine.boundary[1],
            fine_box.vertices[5],
        ]
    )

    # Mid bottom, including mischelement
    s.add_facet_with_holes(
        [
            mix_o_polygon,
            np.vstack(
                [
                    out_window_fine.boundary[0],
                    fine_box.vertices[[1,0]]
                ]
            )
        ],
        #[[0, 0, mix_o_polygon[:, 2].min()]],
        [gus.utils.bounding_box_centroid(mix_o_polygon)],
        boundary_id=6,
    )
    s.add_mesh(
        mix,
        boundary_id=7,
    )

    # Mid in
    ##################################################################
    # Special in
    #s.add_mesh(
    #    in_window,
    #    boundary_id=8
    #)
    
    # TODO: there's something wrong with this. Find it.
    #s.add_facet_with_holes(
    #    polygon_nodes=[
    #        fine_box.vertices[fine_box.faces[1]],
    #        np.vstack(
    #            [
    #                *in_window.boundary[0],
    #                *in_window.boundary[1][1:],
    #                *in_window.boundary[2][1:],
    #                *in_window.boundary[3][1:-1],
    #            ]
    #        )
    #    ],
    #    holes=[gus.utils.bounding_box_centroid(op)],
    #    boundary_id=1,
    #)
    # ^ Alternative is to use:
    #gus.utils.outline_to_line(
    #    in_window.vertices,
    #    in_window.edges[in_window.outlines]
    #)
    ##################################################################

    
    # A way to add window without explicitly giving in window.
    # Need to set boundary condition separately
    in_win_verts = [[v] for v in in_window.vertices.tolist()]
    s.add_facet_with_holes(
        [
            fine_box.vertices[fine_box.faces[1]],
            *in_win_verts,
        ],
        [],
    )

    ############
    # LEFT BOX #
    ############
    # Top left
    s.add_polygon(
        nodes=[
            left_coarse_box.vertices[5],
            *left_out_window.boundary[2],
            left_coarse_box.vertices[4],
        ],
        boundary_id=5,
    )
    # Out left
    s.add_mesh(left_out_window, boundary_id=3)
    # Left left
    s.add_polygon(
        nodes=[
            left_coarse_box.vertices[4],
            *left_out_window.boundary[3],
            left_coarse_box.vertices[0]
        ],
        boundary_id=4,
    )
    # Left right - nothing

    # Left bottom
    lb = np.vstack(
        (left_out_window.boundary[0],
         left_coarse_box.vertices[[1,0]])
    )
    s.add_polygon(
        nodes=lb,
        boundary_id=6,
    )

    # Left in
    s.add_polygon(
        nodes=left_coarse_box.vertices[left_coarse_box.faces[1]],
        boundary_id=1,
    )

    ############
    # RIGHT BOX #
    ############
    # Top right
    s.add_polygon(
        nodes=[
            right_coarse_box.vertices[5],
            *right_out_window.boundary[2],
            right_coarse_box.vertices[4],
        ],
        boundary_id=5,
    )

    # Out right
    s.add_mesh(right_out_window, boundary_id=3)

    # Right left - nothing

    # Left right 
    s.add_polygon(
        nodes=[
            right_coarse_box.vertices[1],
            *right_out_window.boundary[1],
            right_coarse_box.vertices[5],
        ],
        boundary_id=2,
    )

    # Bottom right
    br = np.vstack(
        (right_out_window.boundary[0],
         right_coarse_box.vertices[[1,0]])
    )
    s.add_polygon(
        nodes=br,
        boundary_id=6,
    )

    # In right
    s.add_polygon(
        nodes=right_coarse_box.vertices[right_coarse_box.faces[1]],
        boundary_id=1,
    )

    ########
    # Tet! #
    ########
    m = gus.MeshMaker()
    m.segment = s
    tet = m.tetrahedralize_(
        verbose=True,
        regions=regions,
        check_consistency=True,
        elements_per_memory_block=200000,
    )

    # Taking care of missing boundarys of mid-in and special-in
    mid_in_criteria = gus.utils.bounds_to_ranges(
        gus.utils.bounds(
            fine_box.vertices[fine_box.faces[1]]
        )
    )
    # expand y
    mid_in_criteria[1, 0] -= 0.0001
    mid_in_criteria[1, 1] += 0.0001

    in_window_criteria = gus.utils.bounds_to_ranges(in_window.bounds)
    # expand y
    in_window_criteria[1, 0] -= 0.0001
    in_window_criteria[1, 1] += 0.0001

    face_mask_template = np.zeros(tet.faces.shape[0], dtype=bool)
    face_ind = np.arange(tet.faces.shape[0], dtype=np.int32)

    mid_in_ind = tet.select_faces(
        method="xyz_range",
        criteria=mid_in_criteria,
        only_surface=True,
    )
    in_window_ind = tet.select_faces(
        method="xyz_range",
        criteria=in_window_criteria,
        only_surface=True,
    )

    # Mask for area outside in_window
    mid_mask = face_mask_template.copy()
    mid_mask[mid_in_ind] = True
    in_mask = face_mask_template.copy()
    in_mask[in_window_ind] = True
    face_mask = np.logical_and(
        mid_mask,
        ~in_mask
    )

    tet.append_BC(
        existing_bc_name="BC1",
        method="index",
        criteria=face_ind[face_mask],
    )

    # Mask for in_window
    tet.set_BC(
        name="BC8",
        method="index",
        criteria=in_window_ind,
    )

    tet.show()
