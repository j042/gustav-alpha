import gustav as gus

if __name__ == "__main__":

    gus.utils.configure_logging(debug=True)

    # free form deformation instance
    FFD = gus.FreeFormDeformation()

    # set the mesh
    input_mesh = gus.mesh_shapes.window([[-5,-1],[5,1]], [10,10])
    FFD.set_mesh(input_mesh)

    # set the deformed spline
    kv = [
        [0,0,0,1,1,1],
        [0,0,0,1,1,1]
    ]
    cp_deformed = [
        [0.0,0.0],
        [0.5,0.3],
        [1.0,0.0],
        [0.0,0.5],
        [0.5,0.8],
        [1.0,0.5],
        [0.0,1.0],
        [0.5,1.3],
        [1.0,1.0]
    ]
    deformed_spline = gus.BSpline(
        [2,2], 
        knot_vectors=kv, 
        control_points=cp_deformed
    )
    FFD.set_deformed_spline(deformed_spline)

    # set the undeformed spline (only for visualization purposes)
    cp_undeformed = [
        [0.0,0.0],
        [0.5,0.0],
        [1.0,0.0],
        [0.0,0.5],
        [0.5,0.5],
        [1.0,0.5],
        [0.0,1.0],
        [0.5,1.0],
        [1.0,1.0]
    ]
    undeformed_spline = gus.BSpline(
        [2,2], 
        knot_vectors=kv, 
        control_points=cp_undeformed
    )
    FFD.set_undeformed_spline(undeformed_spline)

    # compute actual deviation
    FFD.deform_mesh()

    # visualize the results
    FFD.show()