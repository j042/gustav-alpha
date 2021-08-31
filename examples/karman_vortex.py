import gustav as gus
import numpy as np

if __name__ == "__main__":
    gus.utils.configure_logging(debug=True)

    m = gus.MeshMaker()
    s = gus.Segment()

    # Domain
    domain_nodes = gus.shapes.box_from_bounds(
        np.array(
            [[0, 0], [40, 20]]
        )
    )

    # Draw domain
    # Set multiple BCs at once
    s.add_nodes(domain_nodes, boundary_ids=[1,2,3,4])

    # Same as above, but a segment at a time.
    #s.add_node(domain_nodes[0], new_sequence=True)
    #s.add_node(domain_nodes[1], boundary_id=1)
    #s.add_node(domain_nodes[2], boundary_id=2)
    #s.add_node(domain_nodes[3], boundary_id=3)
    #s.snap_connect_nodes(domain_nodes[[3,0]], boundary_id=4)

    # cylinder
    cylinder_nodes = gus.shapes.sample_from_circle(31 ,r=1) + [15, 10 + .25]

    # Draw cylinder
    s.add_nodes(cylinder_nodes, boundary_id=5)

    # Holes
    holes = []
    holes.append([15, 10])

    # Segment Check
    s.show()
        
    m.segment = s

    # Triangulate
    m.triangulate_(
        max_area= 0.2,
        min_angle=30,
        return_edges=True,
        return_neighbors=True,
        conforming_delaunay=True,
        holes=holes,
    )

    mm=m.meshes_[0]
    mm.show()

    # Export
    #mm.export("karman.xns")
