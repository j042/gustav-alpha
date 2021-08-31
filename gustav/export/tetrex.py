import logging
import numpy as np
from .utils import *

def tetrex2d(fname, mesh):#, BC_names, BC_global_indices):
    """
    Write 2D tetrix file with given information.
    Some values are hard coded, thus not ready to be flexible yet.
    Hard coded part with be extra highlighted with many #'s.

    Parameters
    -----------
    fname: str
    mesh: any "Mesh" type.
      `Mesh` should have `vertices` and `faces` attributes.

    Returns
    --------
    None
    """
    # First, mesh
    if not hasattr(mesh, ("vertices" and "faces")):
        raise TypeError("Your mesh object does not have `vertices` and `faces` "+
            "attributes!")

    # Second, BC_names and BC_global_indices
    if len(mesh.bc_names_) != len(mesh.bc_global_indices_):
        raise ValueError('length of bc_names_ and bc_global_indices_ does not match!')

    # Let's start
    tetrex = open(fname, 'w', encoding='utf-8')
    #############
    # Meta info #
    #############
    ########## Hard coded 
    tetrex.write('1\n')
    tetrex.write('Tetrex Grid File exported with `gustav`. With issues, "+\
        "please contact jlee@ilsb.tuwien.ac.at\n')
    ########## Hard coded
    tetrex.write('1\n')
    tetrex.write(ljust15('ndimn'))
    tetrex.write(ljust15('nzone'))
    tetrex.write(ljust15('npoin'))
    tetrex.write(ljust15('nvp') + '\n') #nvp is edge count
    ########## Hard coded
    tetrex.write(ljust15('2'))
    ########## Hard coded
    tetrex.write(ljust15('1'))
    tetrex.write(ljust15(str(mesh.vertices.shape[0])))
    tetrex.write(ljust15(str(mesh.edges.shape[0])) + '\n')
    tetrex.write(ljust15('zone name'))
    tetrex.write(ljust15('ncell'))
    tetrex.write(ljust15('nbsurf') + '\n')
    ########## Hard coded
    tetrex.write(ljust15('1'))
    tetrex.write(ljust15(str(mesh.faces.shape[0])))
    tetrex.write(ljust15(str(get_nbsurf(mesh.bc_global_indices_))) + '\n')

    #########################
    # Triangle info (faces) #
    #########################
    tetrex.write(ljust20("volume number"))
    tetrex.write(ljust20("volume type"))
    tetrex.write("point list\n")
    ######### Hard coded
    f = formulate_volumes(mesh, 6)
    tetrex.writelines(f)

    
    #################
    # Vertices info #
    #################
    tetrex.write("coordinates\n")
    ########## Hard coded
    v = formulate_vertices(mesh, dim=2)
    tetrex.writelines(v)


    #######################
    # Boundary Conditions #
    #######################
    tetrex.write(ljust15('bc name'))
    tetrex.write(ljust20('volume number'))
    tetrex.write('local surface number\n')
    bc = formulate_BC(
        mesh.bc_names_,
        mesh.bc_global_indices_
    )
    tetrex.writelines(bc)
    tetrex.close()
    print("Export Finished!")
    print("File name: " + str(fname)) 

def tetrex3d(fname, mesh):#, BC_names, BC_global_indices):
    """
    Write 3D tetrix file with given information.
    3D here, means Tetrahedron.
    Some values are hard coded, thus not ready to be flexible yet.
    Hard coded part with be extra highlighted with many #'s.

    Parameters
    -----------
    fname: str
    mesh: `Mesh`

    Returns
    --------
    None
    """
    # First, mesh
    #if not hasattr(mesh, ("vertices" and "elements"):
    if not hasattr(mesh, ("vertices" and "elements")):
        raise TypeError("Your mesh object does not have `vertices` and `elements` "+
            "attributes!")

    # Second, BC_names and BC_global_indices
    if len(mesh.bc_names_) != len(mesh.bc_global_indices_):
        raise ValueError('length of bc_names_ and bc_global_indices_ does not match!')

    # Let's start
    tetrex = open(fname, 'w', encoding='utf-8')
    #############
    # Meta info #
    #############
    ########## Hard coded 
    tetrex.write('1\n')
    tetrex.write('Tetrex Grid File exported by `gustav`. With issues, '+\
        'please contact jlee@ilsb.tuwien.ac.at\n')
    ########## Hard coded
    tetrex.write('1\n')
    tetrex.write(ljust15('ndimn'))
    tetrex.write(ljust15('nzone'))
    tetrex.write(ljust15('npoin'))
    tetrex.write(ljust15('nvp') + '\n') #nvp is face count 
    ########## Hard coded
    tetrex.write(ljust15('3')) #ndimn
    ########## Hard coded
    tetrex.write(ljust15('1')) #nzone
    tetrex.write(ljust15(str(mesh.vertices.shape[0])))
    tetrex.write(ljust15(str(mesh.faces.shape[0])) + '\n') # This should be element * 4
    tetrex.write(ljust15('zone name'))
    tetrex.write(ljust15('ncell'))
    tetrex.write(ljust15('nbsurf') + '\n')
    ########## Hard coded
    tetrex.write(ljust15('1')) # zone name
    tetrex.write(ljust15(str(mesh.elements.shape[0])))
    tetrex.write(ljust15(str(get_nbsurf(mesh.bc_global_indices_))) + '\n')

    #########################
    # Tetrahedron info (elements) #
    #########################
    tetrex.write(ljust20("volume number"))
    tetrex.write(ljust20("volume type"))
    tetrex.write("point list\n")
    ######### Hard coded
    f = formulate_volumes(mesh, 1)
    tetrex.writelines(f)

    
    #################
    # Vertices info #
    #################
    tetrex.write("coordinates\n")
    ########## Hard coded
    v = formulate_vertices(mesh, dim=3)
    tetrex.writelines(v)


    #######################
    # Boundary Conditions #
    #######################
    tetrex.write(ljust15('bc name'))
    tetrex.write(ljust20('volume number'))
    tetrex.write('local surface number\n')
    bc = formulate_BC(
        mesh.bc_names_,
        mesh.bc_global_indices_,
        3
    )
    tetrex.writelines(bc)
    tetrex.close()
    print("Export Finished!")
    print("File name: " + str(fname)) 

