from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt 
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import  CifWriter
from .geometry import neighbor_dis


def layer_modifier(atoms,gap_width_pm,ori_lattice_par,cell_scale=[1,1,1],plot=True,legend=True)->Atoms:
    ori_lattice_par=np.array(ori_lattice_par)

    "Modify the layer-distance or lattice constant of a lattice"
    o = ori_lattice_par.copy()
    o[0]=o[0]*cell_scale[0]
    o[1]=o[1]*cell_scale[1]
    o[2]=o[2]*cell_scale[2]
    pos=atoms.arrays["positions"]
    #we also need atomic number information when we sort the data
    numbers = atoms.arrays["numbers"].reshape((-1,1))
    coord = np.hstack((numbers,atoms.arrays["positions"]))
    #find the longest axis
    max_xyz=np.max(pos,axis=0)
    #+1 because the first column is atomic number
    index_xyz=np.where(max_xyz==np.max(max_xyz))[0][0]+1
    coord_sort=coord[coord[:,index_xyz].argsort()]
    
    #calculate distance between atoms along the longest axis
    diff_z = []
    for i in range(len(coord_sort)-1):
        diff_z.append(coord_sort[:,index_xyz][i+1]-coord_sort[:,index_xyz][i])
        
    diff_zz=diff_z.copy()
    ## find the atom index before the gap
    gap_index=np.where(np.round(diff_z,index_xyz)==np.round(np.max(diff_z),index_xyz))[0]
    #modified the gap width
    for x in gap_index:
        diff_zz[x]=diff_zz[x]+gap_width_pm
    #modify the coordinates
    for j in range(0,len(coord_sort)-1):
        #First move up atoms that before the gap by gap_width_pm/2 to keep pbc 
        if j<=gap_index[0]:
            coord_sort[:,index_xyz][j]=coord_sort[:,index_xyz][j]                                 #If we have boundary gap: +gap_width_pm/2
        #Then, the atoms after the gap move along the modified atom to atom distance.
        if j>=gap_index[0]:
            coord_sort[:,index_xyz][j+1]=diff_zz[j]+coord_sort[:,index_xyz][j] 
    co_reb=np.hstack((coord_sort[:,1].reshape(-1,1),
                      coord_sort[:,2].reshape(-1,1),
                      coord_sort[:,3].reshape(-1,1)))
    # for j in range(0,len(coord_sort)-1):
    #     coord_sort[:,index_xyz][j+1]=diff_zz[j]+coord_sort[:,index_xyz][j] 
    # co_reb=np.hstack((coord_sort[:,1].reshape(-1,1),
    #                   coord_sort[:,2].reshape(-1,1),
    #                   coord_sort[:,3].reshape(-1,1)))
    num_reb=coord_sort[:,0]
    
    crystal_re = atoms.copy()
    crystal_re.arrays["numbers"]=num_reb
    crystal_re.arrays["positions"]=co_reb
    #modify the cell size
    o[index_xyz-1][2]=o[index_xyz-1][2]+(len(gap_index))*(gap_width_pm)                          #If we have oundary gap:len(gap_index)+1
    crystal_re.cell=np.round(np.array(o),8)
    print(crystal_re)
    stru=Structure(np.round(np.array(o),8),crystal_re.arrays['numbers'],crystal_re.arrays['positions'],coords_are_cartesian=True)
    
    if plot==True:
        plt.plot(np.arange(len(coord_sort)-1),diff_z,label="original atom-atom distance along Z direction(sorted coord)")
        plt.plot(np.arange(len(coord_sort)-1),diff_zz,label="modified atom-atom distance along Z direction(sorted coord)")
        plt.xlabel("Index of atom at the specific atom-atom distance")
        plt.ylabel("Atom-Atom distance($\AA$)")
    if legend==True:
            plt.legend(bbox_to_anchor=(0.5,0,0.5,1.2))
    return stru

def f_r(var):
    """
    distortion function

    Args:
        var (float): constant value

    Returns:
        var: y=var
    """
    return var
    
    
def spherical_distortion(atoms,f_r,pos_center,cutoff,list_of_numbers,filename,save=False):
    """
    Distortion of center atom local enviroment with spherical distortion function f_r and cutoff radius r. 

    Args:
        atoms (Atoms): ase Atoms object.
        f_r (function handle): distortion function f_r.
        pos_center (np.array): numpy array of center atom position.
        cutoff (float):  cutoff radius for distortion. The atoms within cutoff radius will be distorted.
        save (bool): if True, save distorted structures to cif files.

    Returns:
        Atoms: distorted atoms object.
    """
    index_NN,dist_NN=neighbor_dis(atoms,pos_center,cutoff)
    #distortion fr=k,r'=rf(r)
    for k in list_of_numbers:
        
        modi_model=atoms.copy()
        for i in range(len(index_NN)):
            # print(index_NN[i])
            x=k*atoms.arrays['positions'][index_NN[i]][0]+pos_center[0]*(1-f_r(k))
            y=k*atoms.arrays['positions'][index_NN[i]][1]+pos_center[1]*(1-f_r(k))
            z=k*atoms.arrays['positions'][index_NN[i]][2]+pos_center[2]*(1-f_r(k))
            modi_model.arrays['positions'][index_NN[i]]=np.array([x,y,z])
        if save==True:
            fi=filename+str(k)+".cif"
            atom2pymat_cif(modi_model,fi)
            # except:
            #     print("unable to save the file!")
    return modi_model    

def rotation_matrix(alpha, beta,gamma):
    r_z=np.array([[np.cos(alpha),-np.sin(alpha),0],
                  [np.sin(alpha),np.cos(alpha),0],
                  [0,0,1]])
    r_y=np.array([[np.cos(beta),0,np.sin(beta)],
                  [0,1,0],
                  [-np.sin(beta),0,np.cos(beta)]])
    r_x=np.array([[1,0,0],
                  [0,np.cos(gamma),-np.sin(gamma)],   
                  [0,np.sin(gamma),np.cos(gamma)]])
    return r_z@r_y@r_x

def angle_r(vector1,vector2):
    return np.arccos(np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))
def chang_basis(new_basis:np.array,positions:np.array)->np.array:
    """
    new_basis: need to be column vectors
    """
    scale=np.array([[1/np.linalg.norm(new_basis[:,0]),0,0],
                [0,1/np.linalg.norm(new_basis[:,1]),0],
                [0,0,1/np.linalg.norm(new_basis[:,2])]])
    basis_scale_new=np.dot(new_basis,scale)
    A=np.round(np.linalg.inv(basis_scale_new),3)
    pos_new=np.round(np.dot(A,positions.T).T,3)
    return pos_new

def centerize_pos(atoms:Atoms) -> Atoms:
    """
    centerize the positions
    """
    positions=atoms.get_positions()
    atoms.arrays['positions']=positions-positions.mean(axis=0)
    return atoms

def change_basis_atom(atoms:Atoms,basis)->Atoms:
    """
    atoms: Atoms object
    basis: 3*3 matrix
    """
    positions=atoms.get_positions()
    new_pos=chang_basis(basis,positions)
    atoms_new=atoms.copy()
    atoms_new.arrays['positions']=np.round(new_pos,3)
    return atoms_new # this cannot be the same atoms object

    
def cut_z(atoms:Atoms,layers)->Atoms:
    if layers==0:
        return atoms
    positions=np.round(atoms.get_positions(),3)
    z_list=positions[:,2]
    z_list=np.round(z_list,decimals=3)
    z_unique=np.unique(z_list)
    z_list_order=np.round(np.sort(z_unique),3)
    positions_cut=np.round(positions[positions[:,2]<z_list_order[-layers]],3)

    atoms_cut=atoms.copy()
    atoms_cut.arrays['positions']=positions_cut
    atoms_cut.arrays['numbers']=atoms_cut.arrays['numbers'][:len(positions_cut)]
    return atoms_cut

def atom2pymat_cif(atoms,filename):
    """
    Convert ase atoms object to pymatgen structure object and write to cif file

    Args:
        atoms (ase.Atoms): ase atoms object
        filename (str): filename for cif file:
        eg: filename="task1_distorted_structures//KMgCl3_"+str(k)+".cif"
    """
    structure=AseAtomsAdaptor.get_structure(atoms)
    CifWriter(structure).write_file(filename)   