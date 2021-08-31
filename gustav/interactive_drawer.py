"""Adapted from https://matplotlib.org/3.3.3/gallery/event_handling/ginput_manual_clabel_sgskip.html#sphx-glr-gallery-event-handling-ginput-manual-clabel-sgskip-py"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from . import utils

def tellme(info):
    """
    Logs comments during interactive drawing mode.
    Also prints that as a title of the figure.

    Parameter
    ----------
    info: str

    Returns
    --------
    None
    """
    logging.info("InteravtiveDrawer - " + str(info))
    plt.title(info, fontsize=16)
    plt.draw()

def np_nx2(vertices):
    return np.asarray(vertices).reshape(-1,2)

class InteractiveDrawer:

    def __init__(self,):
        self.is_there_drawing = False

    def polygon(
        self,
        grid=None,
        snap=False,
        show_edges=True,
        references=None,
        close_=True,
    ):
        """
        Draws a polygon. Vertices of the polygon is specified with clicks.
        While drawing, you will have an exclusive experience.
        `grid` 

        Parameters
        -----------
        grid: tuple or list
          (Optional) background grid dot range and resolution.
          Default is None.
          [x_range, y_range, resolution]
            => [[-1,1], [-2,2], [100,120]]
        snap: bool
          (Optional) Default is False.
          If True, clicks snap to the closest background gird point.
        show_edges: bool
          (Optional) Default is True.
        references: list or tuple
          (Optional) Show reference points.
          [vertices, connectivity]
        close_: bool
          (Optional) Default is True.
          If false, it is no more polygon, instead just a segment.


        Returns
        --------
        None
        """
        plt.clf()
        plt.setp(plt.gca(), autoscale_on=True)

        if grid is not None:
            logging.debug('InteravtiveDrawer - ')
            grid_vertices = np.mgrid[
                grid[0][0]:grid[0][1]:1j * grid[2][0],
                grid[1][0]:grid[1][1]:1j * grid[2][1],
            ].reshape(2,-1).transpose()

            plt.scatter(
                grid_vertices[:,0],
                grid_vertices[:,1],
                c="pink",
                marker="P",
                alpha=.5,
                zorder=1000,
            )
 

        if references is not None:
            reference_vertices = references[0]

            if len(references) == 2:
                reference_connectivity = references[1]

                for c in reference_connectivity:
                    plt.plot(
                        reference_vertices[c][:,0],
                        reference_vertices[c][:,1],
                        'green',
                        alpha=.5,
                        lw=2
                    )

            plt.scatter(
                reference_vertices[:,0],
                reference_vertices[:,1],
                c="red",
                marker="o",
                alpha=.5,
                zorder=1000,
            )
 

           
        if snap:
            assert len(grid_vertices) > 0,\
                "We can't snap if there's no grid!"

            from scipy.spatial import cKDTree as KDT
            # Snaps only to grid vertices to avoid duplicating vertices:
            #  It causes error.
            # Whether added point is duplicating point is up to the user.
            # You been warned!
            self.kdt = KDT(grid_vertices, compact_nodes=True)
            grid_and_snap = True

        else:
            grid_and_snap = False

        tellme("May it please Your Majesty, here I provide the finest paper. "+\
            "(Click to continue)")
        plt.waitforbuttonpress()
        tellme("Your Majesty, clicks shall mark vertices on this paper. "+\
            "(Click to begin)")
        plt.waitforbuttonpress()

        pts = []
        while True:
            tellme('I believe Your Majesty will bestow clicks upon the very '+\
                'deserving location')

            # grid and snap setup
            if grid_and_snap:
                tmp_pt = np_nx2(plt.ginput(1, timeout=-1))
                _, ind = self.kdt.query(tmp_pt)
                pts.append(grid_vertices[int(ind)])

            else:
                pts.append(plt.ginput(1, timeout=-1))

            # Scatter plot the latest point
            np_pts = np.asarray(pts).reshape(-1,2)
            plt.scatter(np_pts[-1, 0], np_pts[-1,1], zorder=10)

            # Plot the connecting lines 
            if np_pts.shape[0] == 1:
                pass

            else:
                if show_edges:
                    plt.plot(np_pts[-2:, 0], np_pts[-2:,1], 'b', lw=2)

                else:
                    pass
                    #plt.plot(
                    #    np_pts[-2:, 0],
                    #    np_pts[-2:,1],
                    #    'green',
                    #    alpha=.5,
                    #    lw=2
                    #)

            tellme('More clicks to bestow, Your Majesty? (click for yes, press n for no)')
            if plt.waitforbuttonpress():
                break

        # Show the masterpiece
        if close_:
            plt.fill(np_pts[:,0], np_pts[:,1], 'r')

        tellme("Your Majesty, what a masterpiece! (press q to exit and continue)")
        if plt.waitforbuttonpress():
            pass

        self.nodes = np_pts

        logging.info("Selected vertices are: " + str(self.nodes))

        if close_:
            self.connectivity = utils.closed_loop_index_train(
                self.nodes.shape[0]
            )

        else:
            self.connectivity = utils.open_loop_index_train(self.nodes.shape[0])
            
        self.is_there_drawing = True

        return self.nodes, self.connectivity

