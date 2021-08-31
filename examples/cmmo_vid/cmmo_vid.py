import gustav as gus
import vedo
import numpy as np
import copy
import cv2

def cut_with_range(
    vertices,
    x_range,
    y_range,
):
    """
    Assumes given vertices is (n x n) grid
    """
    orig_res = int(np.sqrt(vertices.shape[0]))

    vx_bounds = [vertices[:,0].min(), vertices[:,0].max()]
    vy_bounds = [vertices[:,1].min(), vertices[:,1].max()]

    vertices = vertices[vertices[:,0] > x_range[0]]
    vertices = vertices[vertices[:,0] < x_range[1]]

    x_res = int(vertices.shape[0] / orig_res)

    vertices = vertices[vertices[:,1] > y_range[0]]
    vertices = vertices[vertices[:,1] < y_range[1]]

    y_res = int(vertices.shape[0] / x_res)

    print('x-res: ' + str(x_res))
    print('y-res: ' + str(y_res))

    return vertices, x_res, y_res

def extend_cps(cps, offset, lexsort=[1,0], raw=True):
    cps_flat = cps.copy()
    cps_flat[:,-1] += offset
    if not raw:
        if offset > 0:
            cps_flat[:,-1] = cps_flat[:,-1].max()
        else:
            cps_flat[:,-1] = cps_flat[:,-1].min()

    new_cps = np.vstack((cps, cps_flat))

    return new_cps


#    ind = np.lexsort([new_cps[:,i] for i in lexsort])

#    return new_cps[ind]

if __name__ == "__main__":
    # Load surface mesh
    m = gus.Mesh()
    m.load("surface.ply")
    pc = m.vertices.copy()
    pc = pc[np.lexsort([pc[:,i] for i in [0,1]])]

    # Determine fitting range 
    diff =   0.01147708 - 0.0108595

    v_range1 = [0.1, 0.1+diff]
    u_range1 = [.001, 0.015]

    v_range2 = [.15, .15+diff]
    u_range2 = [.001, 0.015]

    # Offsets
    pc2_offset = .0004 * 1.5

    # Extract from range
    pc1, u_count1, v_count1 = cut_with_range(pc, u_range1, v_range1)
    pc2, u_count2, v_count2 = cut_with_range(pc, u_range2, v_range2)
    pc2[:, 2] += pc2_offset

    # Curve fitting
    curv1 = gus.BSpline()
    curv1.interpolate_curve(
        pc1[:,[0,2]] * [1, 5],
        3,
        centripetal=False,
    )

    curv2 = gus.BSpline()
    curv2.interpolate_curve(
        pc2[:,[0,2]] *[1, 5],
        3,
        centripetal=True,
    )

    # Turn curves into surface
    surf1 = gus.BSpline()
    kv = copy.deepcopy(curv1.knot_vectors)
    kv.append([0.0, 0.0, 1.0, 1.0])
    surf1.knot_vectors = kv
    ds = curv1.degrees.tolist()
    ds.append(1)
    surf1.degrees = ds
    cp = extend_cps(
        curv1.control_points,
        -0.001,
    ).astype(np.double)
    surf1.control_points = cp

    surf2 = gus.BSpline()
    kv = copy.deepcopy(curv2.knot_vectors)
    kv.append([0.0, 0.0, 1.0, 1.0])
    surf2.knot_vectors = kv
    ds = curv2.degrees.tolist()
    ds.append(1)
    surf2.degrees = ds
    cp = extend_cps(
        curv2.control_points,
        0.001,
    ).astype(np.double)
    surf2.control_points = cp

    # Mesh fluid field
    mesher = gus.MeshMaker()
    seg = gus.Segment()
    seg.add_nodes(
        curv1.sample(100),
        close=False,
    )
    seg.add_nodes(
        curv2.sample(100)[::-1],
        is_first_new=False,
        close=False,
    )
    seg.connect_nodes([seg.nodes.shape[0] - 1, 0], reference_nodes=False)
    mesher.segment = seg
    mesher.triangulate_(max_area=0.00000003, min_angle=30, optimize=["laplace", 1e-5, 100])
    m = mesher.meshes_[0]

    things1 = curv1.show(resolutions=180, offscreen=True)
    things2 = curv2.show(resolutions=180, offscreen=True)


    # PNG bucket and fps
    pngs = []
    scene = []
    fps = 60 
    short_cut = 1

    # First Scene - query points.
    # Slowly appears - 2 secs
    qps = vedo.Points(pc1[:,[0,2]] * [1, 5], c="green", r=10)
    #qps = things1[3].color("green").radius(10)
    frames = int(2 * fps / short_cut)
    for i in range(frames):
        qps.alpha(i/frames)
        p = vedo.show(qps, interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()
    scene.append(qps)

    # Second Scene - fit lines.
    # 2 secs
    frames = int(2 * fps / short_cut)
    sc = curv1.sample(frames + 1)
    for i in range(2,frames + 2):
        sc1 = vedo.Line(sc[:i], c="black", lw=6)
        p = vedo.show([sc1, *scene], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()
    scene.append(sc1)

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int( 1 * fps))])

        
    # Third Scene - control points.
    # 1 sec
    frames = int(1 * fps / short_cut)
    cp = things1[1]
    cn = things1[2]
    for i in range(frames):
        cp.alpha(i/frames)
        cn.alpha(i/frames)
        p = vedo.show([cp, cn, *scene], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # We don't need cps and curves anymore
#    scene.append(cp, cn)
    scene.pop(-1)

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int( 1 * fps))])

    # Fourth Scene - extrude.
    # 2 sec
    frames = int(2 * fps / short_cut)
    surf1.control_points_[int(surf1.control_points.shape[0]/2):,-1] =\
        surf1.control_points_[:int(surf1.control_points.shape[0]/2), -1] 
    for i in range(frames):
        surf1.control_points_[int(surf1.control_points.shape[0]/2):,-1] -=\
            0.001 * (1 / frames)
        surf1_things = surf1.show(dashed_line=True, offscreen=True)
        surf1_things[0].color("gray")
        p = vedo.show([*surf1_things, *scene], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Fourth point one - flat bottom.
    # 1 sec
    frames = int(1 * fps / short_cut)
    end_goal = (
        np.ones(int(surf1.control_points.shape[0]/2))
        * surf1.control_points_[int(surf1.control_points.shape[0]/2):,-1].min()
    )
    positions = np.linspace(
        surf1.control_points_[int(surf1.control_points.shape[0]/2):,-1],
        end_goal,
        frames
    )
    for i in range(frames):
        surf1.control_points_[int(surf1.control_points.shape[0]/2):,-1] =\
            positions[i]
        surf1_things = surf1.show(dashed_line=True, offscreen=True)
        surf1_things[0].color("gray")
        p = vedo.show([*surf1_things, *scene], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(fps)])

    # Fifth Scene - knot insertion
    # 3 knot insertsion -> 0.5 sec to remove, .5 sec to put back new
    frames = int(.5 * fps / short_cut)
    for i in range(frames):
        for st in surf1_things:
            st.alpha(1 - (i + 1)/frames)
        p = vedo.show([*scene, *surf1_things], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int(fps / 2))])


    frames = int(.5 * fps / short_cut)
    knots = np.linspace(0,1,5)[1:-1]
    surf1.insert_knots(1, knots.tolist())
    surf1_things = surf1.show(dashed_line=True, offscreen=True)
    surf1_things[0].color("gray")
    for i in range(frames):
        for st in surf1_things:
            st.alpha((i+2)/frames)
        #p = vedo.show([*surf1_things, *scene], interactive=False)#, offscreen=True)
        p = vedo.show([*scene, *surf1_things], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int( 2 * fps))])

    # Fifth point one elevate degrees
    frames = int(.5 * fps / short_cut)
    for i in range(frames):
        for st in surf1_things:
            st.alpha(1 - (i + 1)/frames)
        p = vedo.show([*scene, *surf1_things], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int(fps / 2))])


    frames = int(.5 * fps / short_cut)
    surf1.elevate_degree(0)
    surf1.elevate_degree(1)
    surf1_things = surf1.show(dashed_line=True, offscreen=True)
    surf1_things[0].color("gray")
    for i in range(frames):
        for st in surf1_things:
            st.alpha((i+2)/frames)
        p = vedo.show([*scene, *surf1_things], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int( 2 * fps))])


    # Sixth scene - emerging upper points
    ## First, disappear.  
    frames = int(.5 * fps)
    for i in range(frames):
        for st in surf1_things:
            st.alpha(1 - (i + 2) / frames)
        for s in scene:
            s.alpha(1 - (i + 2) / frames)
        p = vedo.show([*scene, *surf1_things], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    # Pause
    pngs.extend([pngs[-1] for _ in range(int(fps))])

    # Emerging upper points
    # Slowly appears - 2 secs
#    qps2 = things2[3]
    qps2 = vedo.Points(pc2[:,[0,2]] * [1, 5], c="green", r=10)
    frames = int(2 * fps / short_cut)
    # Overwrite surf1_things without control nets
    surf1_things = surf1.show(control_points=False, offscreen=True)
    surf1_things[0].color("gray")

    for i in range(frames):
        qps2.alpha(i/frames)
        for st in surf1_things:
            st.alpha((i + 2) / frames)
        for s in scene:
            s.alpha((i + 2) / frames)

        p = vedo.show([qps2, *surf1_things, qps], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Sixth scene point one - line through points
    # 2 secs
    frames = int(2 * fps / short_cut)
    sc = curv2.sample(frames + 1)
    for i in range(2,frames + 2):
        sc2 = vedo.Line(sc[:i], c="black", lw=6)
        p = vedo.show([sc2, *surf1_things, qps2, qps], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(fps)])

    # Sixth scene point two - extrude
    # 2 sec
    frames = int(2 * fps / short_cut)
    surf2.control_points_[int(surf2.control_points.shape[0]/2):,-1] =\
        surf2.control_points_[:int(surf2.control_points.shape[0]/2), -1] 
    for i in range(frames):
        surf2.control_points_[int(surf2.control_points.shape[0]/2):,-1] +=\
            0.001 * (1 / frames)
        surf2_things = surf2.show(knots=False, control_points=False, offscreen=True)
        surf2_things[0].color("dimgray")
        p = vedo.show([*surf2_things, *surf1_things, qps2, qps], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Sixth scene point one point one - flat bottom.
    # 1 sec
    frames = int(1 * fps / short_cut)
    end_goal = (
        np.ones(int(surf2.control_points.shape[0]/2))
        * surf2.control_points_[int(surf2.control_points.shape[0]/2):,-1].max()
    )
    positions = np.linspace(
        surf2.control_points_[int(surf2.control_points.shape[0]/2):,-1],
        end_goal,
        frames
    )
    for i in range(frames):
        surf2.control_points_[int(surf2.control_points.shape[0]/2):,-1] =\
            positions[i]
        surf2_things = surf2.show(knots=False, control_points=False, offscreen=True)
        surf2_things[0].color("dimgray")
        p = vedo.show([*surf2_things, *surf1_things, qps2, qps], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(fps)])

    # Seventh scene - Fluid mesh
    # First, remove query points
    # 1 sec
    frames = int(1 * fps / short_cut)
    for i in range(frames):
        qps.alpha(1 - ((i + 2) / frames))
        qps2.alpha(1 - ((i + 2) / frames))
        p = vedo.show([*surf2_things, *surf1_things, qps2, qps], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int( .5 *fps))])

    # Mesh Fluid.
    # 4 sec
    frames = int(4 * fps / short_cut)
    f_center = m.faces_center
    x_range = np.linspace(m.vertices[:,0].min(), m.vertices[:,0].max(), frames)
    for x in x_range:
        tmp_m = m.remove_faces(np.where(f_center[:,0] > x)[0])
        p = vedo.show([*surf2_things, *surf1_things, tmp_m.vedo_mesh.wireframe()], interactive=False, offscreen=True)
        pngs.append(
            p.screenshot("-", returnNumpy=True)
        )

    p.close()

    # Pause a bit.
    pngs.extend([pngs[-1] for _ in range(int( 2 *fps))])

    # Make video

    import imageio
    writer = imageio.get_writer("fitting_demo.mp4", fps=60)
    for p in pngs:
        p = p[160:-160]
        writer.append_data(p)

    writer.close()
