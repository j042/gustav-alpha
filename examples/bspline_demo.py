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

    # Insert knots
    b.insert_knots(0, [.2,.4,.6,.7])
    b.show()
    b.insert_knots(1, [.2,.4,.5,.6,.7])
    b.show()

    # Elevation
    b.elevate_degree(0)
    b.show()
    b.elevate_degree(1)
    b.show()


    a = gus.BSpline()
    a.interpolate_curve([[0,0], [0,.5], [0, 0.75], [0,1], [1,0], [1,.5], [1, 0.75], [1,1]], 3, True)
    a.show()


    things = b.show(offscreen=True)
    p  = vedo.show(things, interactive=False, offscreen=True, size=[2000,2000])
    p.screenshot("test.png")
