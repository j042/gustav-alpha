import gustav as gus
import vedo
import numpy as np

if __name__=="__main__":
    gus.utils.configure_logging(debug=True)

    kv = [[0,0,0,0.5,1,1,1],
          [0,0,0,1,1,1],
    ]
    cp1 = [
           [0,0,1],
           [0,1,0],
           [1,1.5,0],
           [3,1.5,0],
           [-1,0,0],
           [-1,2,0],
           [1,4,0],
           [3,4,0],
           [-2,0,0],
           [-2,2,0],
           [1,5,0],
           [3,5,-1],
    ]

    b = gus.BSpline(
        degrees=[2, 2],
        knot_vectors=kv,
        control_points=cp1,
    )

    b.show()

    # Get mesh - it is always 0th element
    bm = b.show(offscreen=True)[0]

    # Compute color for each vertex
    #   - here, it is x-coordinate
    col = bm.vertices()[:,0]

    # Assign Colors to each vertex
    bm.pointColors(col, cmap="jet")

    # Check if it is what you want
    #   - It is nice to add `.close()` after `show()` to make sure that
    #   - the window closes if you press q or esc.
    vedo.show(bm).close()
