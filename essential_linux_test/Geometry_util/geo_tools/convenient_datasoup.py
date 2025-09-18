from .geometry import *
from.gr_util import *
from dataclasses import dataclass
from .lattice_modifier import change_basis_atom,cut_z
from ..io.io_general import write_xyz
from ase.cluster import Octahedron
from scipy.spatial.distance import cdist
from ase import Atoms
import numpy as np
from ase.visualize import view
from collections import defaultdict
import pandas as pd
import platform
from tqdm import tqdm
system=platform.system()


@dataclass
class particle:
    """
    Author: Kaifeng Zheng
    This class creates a particle object with its associated properties and methods.

    """
    name: str
    atom_obj: Atoms
    lattice_par: np.float64=None

    def __post_init__(self):
        """
        This method is called after the particle object is initialized.
        It recenters the atomic positions of the particle.
        """
        pos=self.atom_obj.get_positions()
        center=np.mean(pos,axis=0)
        pos_center=pos-center
        self.atom_obj.set_positions(np.round(pos_center,3))
    def shape_size(self):
        """
        This method returns the shape descriptor of the particle storing in a pandas DataFrame.
        """
        return descriptor_table(self.atom_obj)
    def num_atom(self):
        """
        This method returns the number of atoms in the particle.
        """
        return len(self.atom_obj.get_positions())
    def positions(self):
        """
        This method returns the atomic positions of the particle.
        """
        return self.atom_obj.get_positions()
    def cut_by_surface(self,plane,layer):
        """
        This method cuts the particle by a specific surface plane and layer.
        """
        if plane==111:
            basis=np.array([[1.,1.,-2.],[1.,-1.,0.],[1.,1.,1.]]).T
        if plane==100:
            basis=np.array([[0.,1.,1.],[0.,-1.,1.],[1.,0.,0.]]).T
        if plane==110:
            basis=np.array([[-1.,1.,1.],[-1.,1.,-2.],[1.,1.,0.]]).T
        try:
            atoms_align_z=change_basis_atom(self.atom_obj,basis)
            return cut_z(atoms_align_z,layer)
        except Exception as e:
            print(e)
    def distance_score(self):
        """
        This method calculates the distance score of the particle.
        """
        return np.sum(cdist(self.atom_obj.positions, self.atom_obj.positions, metric="euclidean"))
    def view_particle(self,viewer=None):
        """
        This method visualizes the particle using ASE's view function.
        """
        view(self.atom_obj,viewer)


class particle_database:
    """
    Author: Kaifeng Zheng
    This class creates a database of particle objects for a specific element.
    """
    def __init__(self,element):
        self.element=element
        self.shape_all=list()
    def predefine_data(self,lattice_par):
        """
           Generate predefined particles for a specific element and lattice parameter (octahedron, cuboctahedron, and other regular particles
           with their truncated shapes).
           lattice_par: float, lattice parameter of the crystal, it is better to set lattice_par round
                        to two decimal numbers.
        """
        reg_oct_2=Octahedron(self.element,2,latticeconstant=lattice_par)
        reg_oct_3=Octahedron(self.element,3,latticeconstant=lattice_par)
        reg_oct_4=Octahedron(self.element,4,latticeconstant=lattice_par)
        reg_oct_5=Octahedron(self.element,5,latticeconstant=lattice_par)
        reg_oct_6=Octahedron(self.element,6,latticeconstant=lattice_par)
        reg_oct_7=Octahedron(self.element,7,latticeconstant=lattice_par)
        reg_oct_8=Octahedron(self.element,8,latticeconstant=lattice_par)

        COC_O1=Octahedron(self.element,3,latticeconstant=lattice_par, cutoff=1)
        COC_O2=Octahedron(self.element,5,latticeconstant=lattice_par, cutoff=2)
        COC_O3=Octahedron(self.element,7,latticeconstant=lattice_par,cutoff=3)
        COC_O4=Octahedron(self.element,9,latticeconstant=lattice_par,cutoff=4)
        COC_O5=Octahedron(self.element,11,latticeconstant=lattice_par,cutoff=5)

        HEX_O1_SQ_O1=Octahedron(self.element,4,latticeconstant=lattice_par, cutoff=1)

        HEX_O2_SQ_O2=Octahedron('Pt',7,latticeconstant=lattice_par,cutoff=2)
        HEX_O3_SQ_O2=Octahedron('Pt',8,latticeconstant=lattice_par,cutoff=2)


        HEX_O1_2_SQ_O1=Octahedron(self.element,5,latticeconstant=lattice_par, cutoff=1)
        HEX_O1_2_SQ_O2=Octahedron(self.element,6,latticeconstant=lattice_par,cutoff=2)
        HEX_O1_3_SQ_O1=Octahedron(self.element,6,latticeconstant=lattice_par,cutoff=1)
        HEX_O1_4_SQ_O1=Octahedron('Pt',7,latticeconstant=lattice_par,cutoff=1)

        # reg_oct_7=Octahedron('Pt',7,latticeconstant=3.92)
        # reg_oct_8=Octahedron('Pt',8,latticeconstant=3.92)
        # reg_oct_9=Octahedron('Pt',9,latticeconstant=3.92)
        
        # reg_oct_10=Octahedron('Pt',10,latticeconstant=3.92)
        # read into shape list
        shape_list=[particle(f'OCT_2',reg_oct_2,lattice_par),particle(f'OCT_3',reg_oct_3,lattice_par),particle(f'OCT_4',reg_oct_4,lattice_par),particle(f'OCT_5',reg_oct_5,lattice_par),particle(f'OCT_6',reg_oct_6,lattice_par), particle(f'OCT_7',reg_oct_7,lattice_par),particle(f'OCT_8',reg_oct_8,lattice_par),
            particle(f'COC_O1',COC_O1,lattice_par),particle(f'COC_O2',COC_O2,lattice_par),particle(f'COC_O3',COC_O3,lattice_par),particle(f'COC_O4',COC_O4,lattice_par), particle(f'COC_O5',COC_O5,lattice_par),
            particle(f'HEX_O1_SQ_O1',HEX_O1_SQ_O1,lattice_par),particle(f'HEX_O2_SQ_O2',HEX_O2_SQ_O2,lattice_par),particle(f'HEX_O3_SQ_O2',HEX_O3_SQ_O2,lattice_par),
            particle(f'HEX_O1_2_SQ_O1',HEX_O1_2_SQ_O1,lattice_par),particle(f'HEX_O1_3_SQ_O1',HEX_O1_3_SQ_O1,lattice_par),particle(f'HEX_O1_4_SQ_O1',HEX_O1_4_SQ_O1,lattice_par),
            particle(f'HEX_O1_2_SQ_O2',HEX_O1_2_SQ_O2,lattice_par)]
        # print(shape_list)
        # Append those particles to the main shape list
        for p in shape_list:
            self.shape_all.append(p)
        # print(f"shape_list1={len(shape_list)}") #DEBUG
        for i in range(len(shape_list)):
            # print(shape_list[i])
            self.cut_particle_multi(shape_list[i],111)
        # print(f"shape_list2={len(shape_list)}") #DEBUG
        for i in range(len(shape_list)):
            self.cut_particle_multi(shape_list[i],100)
        # print(f"shape_list3={len(shape_list)}") #DEBUG
    # load particles from xyz files
    def load_from_xyz(self,dir):
        if system=="Windows":
            split_symbol='\\'
        else:
            split_symbol='/'
        filename_list=glob(dir+split_symbol+"*.xyz")
        for filename in filename_list:
            try:
                shape=read(filename)
                self.shape_all.append(particle(filename.split(split_symbol)[-1],shape))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    def write(self,dir,num=None):
        """
        in case we want to generate particles based on different lattice parameters
        """
        for shape in self.shape_all:
            if num==None:
                file1=open(f'{dir}\{shape.name}.xyz','w')
                write_xyz(file1,shape.atom_obj,comment=f"{shape.name} lattice constant={shape.lattice_par}")
                file1.close()
            else:
                file1=open(f'{dir}\{shape.name}_{num}.xyz','w')
                write_xyz(file1,shape.atom_obj,comment=f"{shape.name} lattice constant={shape.lattice_par}")
                file1.close()
    def shape_info(self):
        # try:
        shape_dict={}
        for shape in tqdm(self.shape_all):
            # print(shape.name)
            shape_dict[shape.name]=shape.shape_size()
        return pd.DataFrame(shape_dict).astype(np.float64) 
        # except Exception as e:
        #     print(e)
    def gr_analysis(self,rmin,rmax,bins=500):
        """
        Perform pair distribution function (g(r)) analysis on all particles.
        Note: This analysis is suitable for small nanoparticles and does not include density normalization.
        """
        gr_results = {}
        rmesh = np.linspace(rmin, rmax, bins)
        for shape in tqdm(self.shape_all):
            positions = shape.atom_obj.get_positions()
            dist_matrix=cdist(positions,positions)
            gr_accum = np.zeros(bins-1)
            for i in range(dist_matrix.shape[0]):
                r, gr = np.array(gr_atom_from_dis_norho(dist_matrix[i], rmesh))
                gr_accum += gr
            gr_avg = gr_accum / dist_matrix.shape[0]
            gr_results[shape.name] = gr_avg
        gr_results=pd.DataFrame(gr_results).astype(np.float64)
        gr_results.insert(0,column='r',value=r)
        return gr_results

    def remove_by_name(self,names,inplace=False):
        """
            Remove particles by their names.
            name: list
        """
        shapes=self.shape_all.copy()
        if inplace==True:
            for name in names:
                for shape in shapes:
                    if name==shape.name:
                        self.shape_all.remove(shape)
        else:
            print(f"REMOVE:\n{names}\n Please set inplace=True to delete")
    def append(self,particle_obj):
        """
            Append a new particle object to the shape list.
        """
        self.shape_all.extend(particle_obj)
    def similar_check(self,remove=False):
        dis=[]
        shapes=self.shape_all.copy()
        for i in range(len(shapes)):
            dis.append(shapes[i].distance_score())
        similar=[dup for dup in sorted(list_duplicates(dis))]
        index=[i[1][j] for i in similar for j in range(1,len(i[1]))]
        if remove==True:
            names=[]
            for i in index:
                names.append(shapes[i].name)
            for name in names:
                self.remove_by_name(name)
        return similar
    def search_by_name(self,name):
        """
        Search for particles by their name.
        """
        names=[]
        for shape in self.shape_all:
            names.append(shape.name)
        retrieve=[i for i in names if name in i]
        objects=[]
        for shape in self.shape_all:
            if shape.name in retrieve:
                objects.append(shape)
        return objects
    def cut_particle_multi(self,shape_obj,plane):
        """
        Cut a particle the particle along a specified plane, the generated particles will be added to the shape list.
        """
        name_particle=shape_obj.name
        lattice_par=shape_obj.lattice_par
        for layer in range(1,100):
            cut_atoms=shape_obj.cut_by_surface(plane,layer)
            if len(cut_atoms.positions)<6:
                break
            else:
                self.shape_all.append(particle(f'{name_particle}_cut_ori_{plane}_l_{layer}',cut_atoms,lattice_par))
            
        
def particle_cut(particle_obj,plane,layer):
    """
    Cut a particle along a specified plane and layer.
    """
    name_particle=particle_obj.name
    lattice_par=particle_obj.lattice_par
    cut_atoms=particle_obj.cut_by_surface(plane,layer)
    return particle(f'{name_particle}_cut_ori_{plane}_l_{layer}',cut_atoms,lattice_par)

def list_duplicates(seq):
    """
    List duplicate items in a sequence.
    """
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)
