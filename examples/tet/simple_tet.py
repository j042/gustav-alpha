import gustav as gus
import numpy as np

if __name__ == "__main__":

    # Prepare vertices and elements
    vertices = np.array(
        [
            [0,0,0],
            [1,0,0],
            [1,1,0],
            [0,1,0],
            [0,0,1],
            [1,0,1],
            [1,1,1],
            [0,1,1],
        ]
    )
    vertices = np.vstack((vertices, vertices + [1,0,0]))
    elements = np.array(
       [[0,1,2,3,4,5,6,7]], dtype=np.int32
    )
    elements = np.vstack((elements, elements + 8))

    # Make mesh - here, duplicating faces are removed since that's not 
    # appreciated from `tetgen`
    mesh = gus.Mesh(vertices=vertices, elements=elements)
    mask = np.ones(12, dtype=bool)
    mask[10] = False
    mesh = gus.Mesh(vertices=vertices, faces=mesh.faces[mask])

    # Add mesh to segment
    s = gus.Segment()
    s.add_mesh(mesh, boundary_id=5)

    # Add segment to meshmaker
    m = gus.MeshMaker()
    m.segment = s

    # Redefine fmarkers (BC marker) - Totally optional
    fmarkers = np.array([1,2,3,4,5,6,7,8,9,10,11])

    # Define different areas for different regions
    regions = np.array([[.5,.5,.5,2,0.00001],[1.5,.5,.5, 3, 0.001]])

    # tet! here, if you pass fmarkers, it overwrites segment's internal
    # BCs. Meaning boundary_id=5 won't be considered.
    mm = m.tetrahedralize_(
        fmarkers=fmarkers,
        regions=regions,
        verbose=True,
        elements_per_memory_block=100000
    )

    # Show only surface mesh
    gus.Mesh(vertices=mm.vertices, faces=mm.faces[mm.surfaces]).show()

    # Export in campiga-ready format
    #mm.export("test_tet.campiga")
