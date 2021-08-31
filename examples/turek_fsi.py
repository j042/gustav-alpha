import gustav as gus
import numpy as np

if __name__ == "__main__":
    gus.utils.configure_logging(debug=True)

    m = gus.MeshMaker()
    s = gus.Segment()


    ########## CONFIG ########
    num_circle_verts = 71
    area_left_half = 0.0001
    area_flag = 0.00005
    area_right_half = 0.5

    s.global_spacing = .3
    ##########################

    # Box
    s.add_node([0,0])  #0
    s.add_node([0,.41])  #1
    s.add_node([2.5,.41])  #2
    s.add_node([2.5,0], close_sequence=True)  #3

    # Pole
    pole_nodes = gus.shapes.sample_from_circle(num_circle_verts ,r=0.05) + [0.2, 0.2]

    s.add_nodes(pole_nodes)

    # Flag
    flag_nodes = gus.shapes.box_from_bounds(
        np.array(
            [[.6 - .355, .19], [.6, .19 + .02]]
        )
    )

    # Draw extended
    s.add_node(flag_nodes[1], new_sequence=True)
    s.add_node(flag_nodes[2],)
    s.add_node(flag_nodes[3],)
    s.add_node(flag_nodes[0],)

    # Snap variant
    #s.add_line_from_snap(flag_nodes[[1,2]], spacing=0.005)
    #s.add_node(flag_nodes[3], spacing=0.005)
    #s.snap_connect_nodes(flag_nodes[[3,0]], spacing=0.005)

    # A wall to separate different triangle size area

    # Draw extended
    #s.add_node([.9,.5], new_sequence=True)#, snap=True)
    #s.add_node([.9,-.01],)# snap=True)

    #Snap
    s.snap_connect_nodes([[.92, .4], [.9, 0.0]])

    # Holes
    holes = []
    holes.append([.2,.2])

    # Segment Check
    s.show()
        
    m.segment = s

    # Assign different areas
    regions = [
        [.1,.1,0, area_left_half],
        [.5,.191,1, area_flag],
        [2,.191,2, area_right_half],
    ]

    # Triangulate
    m.triangulate_(
        regions=regions,
        max_area= 0.002,
        min_angle=30,
        return_edges=True,
        return_neighbors=True,
        conforming_delaunay=True,
        holes=holes,
    )

    mm=m.meshes_[0]
    mm.show()
    fluid_m, structure_m = m.remove_faces(
        return_both=True,
        optimize=["laplace", 1e-5, 100],
        PSLG="only"
    ) # Should select structure

    fluid_m.show()
    structure_m.show()

    # BCs
    #fluid_m.set_BC("BC_1_LEFT")
    #fluid_m.set_BC("BC_2_TOP")
    #fluid_m.set_BC("BC_3_RIGHT")
    #fluid_m.set_BC("BC_4_BOTTOM")
    #fluid_m.set_BC("BC_5_BALL")
    #fluid_m.set_BC("BC_6_FLAG")
    #fluid_m.export("flag_fluid_1.grd")

    #structure_m.show()
    #structure_m.set_BC("BC_1_LEFT")
    #structure_m.set_BC("BC_2_TOP")
    #structure_m.set_BC("BC_3_RIGHT")
    #structure_m.set_BC("BC_4_BOTTOM")
    #structure_m.set_BC("BC_5_234")
    #structure_m.export("flag_stucture_1.grd")
