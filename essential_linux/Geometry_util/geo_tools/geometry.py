import numpy as np
import pandas as pd
from collections import Counter
from ase import Atoms
# from pymatgen.core.structure import Structure
from pymatgen.core import Structure, Element,Molecule
import pymatgen.symmetry.analyzer
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase.io import read
from tqdm import tqdm
from glob import glob
import os
from os import sys
import platform
system=platform.system()
"""
   Author: Kaifeng Zheng
   This codes provides various geometric descriptors for atomic structures.
   The descriptors include:
        1. CN1
        2. CN2
        3. CN3
        4. CN4
        5. surface_CN
        6. GCN1
        7. GCN2
        8. GCN3
        9. GCN4
        10. surface_GCN1
        11. bond_length
        12. diameter_2radius
        13. diameter_pca
        14. diameter_xy
        15. MIAD
        16. SVR_CN
        17. SVR_GCN
        18. Departure from sphere(moment)
        19. oblateness_moment
        20. oblateness_pca
        21. atom_number
    User can also add other descriptors in the database
    
"""
def centerize_pos(atoms:Atoms) -> Atoms:
    """
    centerize the positions
    """
    positions=atoms.get_positions()
    atoms.arrays['positions']=positions-positions.mean(axis=0)
    return atoms

def cut_by_surface(particle,plane,layer):
    if plane==111:
        basis=np.array([[1.,1.,-2.],[1.,-1.,0.],[1.,1.,1.]]).T
    if plane==100:
        basis=np.array([[0.,1.,1.],[0.,-1.,1.],[1.,0.,0.]]).T
    if plane==110:
        basis=np.array([[-1.,1.,1.],[-1.,1.,-2.],[1.,1.,0.]]).T
    try:
        atoms_align_z=change_basis_pos(particle,basis)
        return cut_z(atoms_align_z,layer)
    except Exception as e:
        print(e)
def change_basis_pos(particle,basis)->np.ndarray:
    """
    atoms: Atoms object
    basis: 3*3 matrix
    """
    new_pos=chang_basis(basis,particle)
    new_pos=np.round(new_pos,3)
    return new_pos # this cannot be the same atoms object
def chang_basis(new_basis:np.array,positions:np.array)->np.array:
    """
    new_basis: need to be column vectors
    """
    scale=np.array([[1/np.linalg.norm(new_basis[:,0]),0,0],
                [0,1/np.linalg.norm(new_basis[:,1]),0],
                [0,0,1/np.linalg.norm(new_basis[:,2])]])
    basis_scale_new=np.round(np.dot(new_basis,scale),5)
    A=np.round(np.linalg.inv(basis_scale_new),5)
    pos_new=np.round(np.dot(A,positions.T).T,5)
    return pos_new
    

def cut_z(particle,layers)->np.ndarray:
    """
    Cut the particle along its height.
    """
    if layers==0:
        return particle
    positions=np.round(particle,3)
    z_list=positions[:,2]
    z_list=np.round(z_list,decimals=3)
    z_unique=np.unique(z_list)
    z_list_order=np.round(np.sort(z_unique),3)
    positions_cut=np.round(positions[positions[:,2]<z_list_order[-layers]],3)
    return positions_cut
def diameter_max(positions):
    """
    This algorithm calculates the radius of
    the circumscribed sphere of the polyhedron
    particle
    :param positions: coordinates
    :return: radius of particle
    """
    center = positions.mean(axis=0)
    vector = positions - center
    dis = np.linalg.norm(vector,axis=1)
    D = 2*np.max(dis)
    return D
def diameter_pca(positions):
    """
    This algorithm calculates the diameter of the particle using PCA.
    """
    pca=PCA(n_components=3)
    pos = np.round(positions,5)
    pos_next=np.zeros_like(pos)
    max_value=np.max(pos,axis=0)
    min_value=np.min(pos,axis=0)
    c,b,a=np.sort(max_value-min_value)
    c_next,b_next,a_next=0,0,0
    i=0
    while a-a_next>1e-4 or c-c_next>1e-4 or b-b_next>1e-4:
        """
              SCF loop
        """
        pos_next=pos
        max_value=np.max(pos_next,axis=0)
        min_value=np.min(pos_next,axis=0)
        c_next,b_next,a_next=np.round(np.sort(max_value-min_value),5)
        
        result=pca.fit(pos_next)
        pos_fit=result.fit_transform(pos_next)
        pos=np.round(pos_fit,5)
        max_value=np.max(pos,axis=0)
        min_value=np.min(pos,axis=0)
        c,b,a=np.round(np.sort(max_value-min_value),5)
        i+=1
    diameter=np.round(np.sqrt(a**2+b**2+c**2),2) #another method to calculate diameter
    return diameter

def diameter_xy(positions):
    """
    This algorithm calculate radius based on xy plane
    """
    positions_xy=positions[:,:2]

    center = positions_xy.mean(axis=0)
    vector = positions_xy - center
    dis = np.linalg.norm(vector,axis=1)
    D = 2*np.max(dis)
    return D
def surface_per(atoms:Atoms,method='CN'):
    """
    calculate the percentage of surface atoms
    method=[CN,GCN]
    """
    if method=='CN':
        CNs,diss,CN_ave=getCN_dis_N(atoms,1,option='CN')
    if method=='GCN':
        CNs,diss,CN_ave=getCN_dis_N(atoms,1,option='GCN')
    CN_surface=[cn for cn in CNs if cn<12]
    return len(CN_surface)/len(atoms.get_positions())

def surface_CN(atoms:Atoms,method='CN'):
    """
    calculate the percentage of surface atoms
    method=[CN,GCN]
    """
    if method=='CN':
        CNs,diss,CN_ave=getCN_dis_N(atoms,1,option='CN')
    if method=='GCN':
        CNs,diss,CN_ave=getCN_dis_N(atoms,1,option='GCN')
    CN_surface=[cn for cn in CNs if cn<12]
    return np.mean(CN_surface)


def equ_sites(path:str,absorber,cutoff,randomness=4):
    """
    :param positions: coordinates
    :param cutoff:    cutoff distance defined by mutiple scattering radius
    :return:          non-equ position indexes
    """
    # cutoff method
    def duplicates(lst, item):
        """
        :param lst: the whole list
        :param item: item which you want to find the duplication in the list
        :return: the indexes of duplicate items
        """
        return [i for i, x in enumerate(lst) if x == item]
    
    if path.split('.')[1]=='xyz':
        structure=Molecule.from_file(path)
    else:
        structure = Structure.from_file(path) 

    absorber_species = Element(absorber)
    print(absorber_species)
    absorber_list = np.where(np.array(structure.species) == absorber_species)[0]
    positions=structure.cart_coords
    dis_all =  np.around(cdist(np.array(positions)[absorber_list],np.array(positions), metric="euclidean"),decimals=randomness)
    dis_all.sort(axis=1)
    dis_cut = [list(dis_all[i][dis_all[i] < cutoff]) for i in range(len(dis_all))]
    dup = []
    for i in range(len(dis_cut)):
        dup.append(duplicates(dis_cut, dis_cut[i])[0])
    #unique_index = list(set(dup))  # set can delete all duplicated items in a list
    unique_index = dict()
    for i in range(len(dup)):
        if dup[i] in unique_index:
            unique_index[dup[i]].append(i)
        else:
            unique_index.update({dup[i]:[i]})
    num_sites=[]
    uni_sites=list(unique_index.keys())
    for i in range(len(uni_sites)):
        num_sites.append(len(unique_index[uni_sites[i]]))
    # sort it using sorted method. Do not use list.sort() method, because it returns a nonetype.
    #unique_index = np.array(sorted(unique_index))
    #print("number of atoms: {}".format(len(positions)))
    #print("number of unique atoms: {}".format(len(atom_index))) #
    
    return np.array(uni_sites),np.array(num_sites)  #keys are those unique sites, values are the cooresponding equ-sites for those unique_sites
def equ_sites_pointgroup(pos_dir):
    mol=Molecule.from_file(pos_dir)
    pointgroup=pymatgen.symmetry.analyzer.PointGroupAnalyzer(mol).get_equivalent_atoms()['eq_sets']
    keys=list(pointgroup.keys())
    num_sites=[len(pointgroup[keys[i]]) for i in range(len(keys))]
    return keys, num_sites


def getCN_dis_Oneshell(positions,center_position,N,thickness=1e-1):
    """
    calculate coordination number and distance for N nearest neighbors with fixed error bar

    Args:
        positions (np.array): coordinates of one atom
        center_position (np.array): coordinates of focused atom
        N (int): nth nearest neighbor

    Returns:
        atom_shell(int): coordination number of Nth nearest neighbors of the center atom
        dis(np.array): distance of Nth nearest neighbors of the center atom
    

    
    """
    # center_position = positions.mean(axis=0)
    # The precision is 0.1A
    dis_all = np.around(cdist([center_position], positions,metric='euclidean'), decimals=1)[0] #must add [] here
    sort_dis=np.unique(np.sort(dis_all[dis_all>0]))
    if N>len(sort_dis):
        return 0,sort_dis[-1]
        #second GCN calculates the second nearest neighbor coordination numbers, and weighted by the 
        #first coordination numbers of the second shell atoms.
    # print(sort_dis)
    atom_shell=len(np.where(np.isclose(dis_all,sort_dis[N-1],atol=thickness))[0]) 
    return atom_shell,sort_dis[N-1]
def getGCN_dis_Oneshell(positions,center_position,N,thickness=1e-1,bulk_CN=12):
    """
    calculate Generalized coordination number and distance for N nearest neighbors with fixed error bar
    Parameters
    ----------
    positions : np.ndarray
        Atomic positions.
    center_position : np.ndarray
        Center atom position.
    N : int
        Nth nearest neighbor.
    thickness : float
        Error bar thickness.
    bulk_CN : float
        Bulk coordination number (maximum coordination in the lattice configuration). Default is 12 for FCC.

    Returns
    -------
    atom_shell : int
        Coordination number of Nth nearest neighbors of the center atom.
    dis : float
        Distance of Nth nearest neighbors of the center atom.

    """
    dis_all = np.around(cdist([center_position], positions,metric='euclidean'), decimals=1)[0] #must add [] here
    sort_dis=np.unique(np.sort(dis_all[dis_all>0]))
    if N>len(sort_dis):
        return 0,sort_dis[-1]
        #second GCN calculates the second nearest neighbor coordination numbers, and weighted by the 
        #first coordination numbers of the second shell atoms.
    atom_shell=np.where(np.isclose(dis_all,sort_dis[N-1],atol=thickness))[0]
    CNs=[]
    for i in atom_shell:
        dis_all = np.around(cdist([positions[i]], positions,metric='euclidean'), decimals=1)[0] #must add [] here
        min_dis=np.min(dis_all[dis_all>0])
        atom_shell=np.where(np.isclose(dis_all,min_dis,atol=thickness))[0]
        CNs.append(len(atom_shell))
    # print(CNs)
    # print(np.sum(CNs)/np.max(CNs))
    # print(CNs)
    return np.sum(CNs)/bulk_CN,sort_dis[N-1]

def getCN_dis_N(atom:Atoms,N:int,option='CN'):
    """
    get atom-atom distance and coordination number for N nearest neighbors

    Args:
        atom (Atoms): cluster or molecule
        N (int): nth nearest neighbor
        option: select using "CN" or "GCN" method 

    Returns:
        CN (list): coordination numbers of Nth nearest neighbors
        dis (list): distance of Nth nearest neighbors
        CN_ave (float): average coordination number of Nth nearest neighbors for the particle
        dis_ave (float): average distance of Nth nearest neighbors for the particle
        
    """
    pos=atom.arrays['positions']
    CNs=[]
    diss=[]
    for i in range(len(pos)):
        if option=='CN':
            CN,dis=getCN_dis_Oneshell(pos,pos[i],N)
        if option=='GCN':
            CN,dis=getGCN_dis_Oneshell(pos,pos[i],N)
            # print("pass_1")
        # print(CN)
        CNs.append(CN)
    CN_ave=np.mean(CNs)
    
    return CNs,dis,CN_ave
def dim_pos(pos):
    def num_zero_element(eigens):
        eigens=np.round(eigens,5)
        num=0
        for i in range(len(eigens)):
            if eigens[i]>1e-5:
                num+=1
        return num
    sig_value=np.linalg.svd(np.array(pos))[1]
    return num_zero_element(sig_value)


    
def ellipsoid(atom:Atoms,tor=1e-3):
    def khachiyan_algorithm(atom:Atoms,tor=1e-3):
        positions=atom.get_positions()
        (n,d) = positions.shape
        Q=np.vstack([np.copy(positions.T),np.ones(n)])
        u=np.ones(n)/n
        error = float('inf')
        while error > tor:
            X=np.dot(np.dot(Q,np.diag(u)),Q.T)
            M=np.diag(np.dot(np.dot(Q.T,np.linalg.inv(X)),Q))
            j=np.argmax(M)
            step_size=(M[j]-d-1)/((d+1)*(M[j]-1))
            new_u=(1-step_size)*u
            new_u[j]+=step_size
            error=np.linalg.norm(new_u-u)
            u=new_u
        center=np.dot(u,positions)
        A_inv=np.linalg.inv(np.dot(np.dot(positions.T,np.diag(u)),positions)-np.outer(center,center))/d
        return center,A_inv
    center,A_inv=khachiyan_algorithm(atom,tor=tor)
    eignvalue,eignvector=np.linalg.eig(A_inv)
    a=np.sqrt(1/eignvalue[0])
    b=np.sqrt(1/eignvalue[1])
    c=np.sqrt(1/eignvalue[2])
    return a,b,c,center,eignvector
    
        
def moment_descriptor(atom:Atoms):
    """
    used for cluster with regular shape
    """
    pos=atom.get_positions()
    dimension=dim_pos(pos)
    # print(dimension)
    if dimension<3:
        return 0,1
    moment_atom=atom.get_moments_of_inertia(vectors=False)
    # print(moment_atom)
    I=np.sort(moment_atom) #I2 is the largest moment of inertia
    zeta=((I[2]-I[1])**2+(I[1]-I[0])**2+(I[0]-I[2])**2)/(I[0]**2+I[1]**2+I[2]**2)
    eta=(2*I[1]-I[0]-I[2])/I[2]
    return zeta,eta
def descriptor_table(atom:Atoms,all=True,descriptors=[],display_keys=False):
    if display_keys==True:
        print("The available properties are: ")
        print("1. CN1")
        print("2. CN2")
        print("3. CN3")
        print("4. CN4")
        print("5. surface_CN")
        print("6. GCN1")
        print("7. GCN2")
        print("8. GCN3")
        print("9. GCN4")
        print("10. surface_GCN1")
        print("11. bond_length")
        print("12. diameter_2radius")
        print("13. diameter_pca")
        print("14. diameter_xy")
        print("15. MIAD")
        print("16. SAF_CN")
        print("17. SAF_GCN")
        print("18. Departure from sphere(moment)")
        print("19. oblateness_moment")
        print("20. oblateness_pca")
        print("21. atom_number")
    if display_keys==False:
        if all==True:
            zeta,eta=moment_descriptor(atom)
            flatten=pca_oblate(atom)

            # dis=distance_matrix(atom.arrays['positions'],atom.arrays['positions'])
            # dis_sort=np.round(np.sort(dis,axis=1),5) #set a tolerance of distance
            # cn_n=[]
            # for i in range(len(dis_sort)):
            #     cn=np.unique(dis_sort[i],return_counts=True)[1][1]
            #     cn_n.append(cn)
            # cn_n = np.array(cn_n)
            CNS1,diss1,CN_ave1=getCN_dis_N(atom,1)
            CNS2,diss2,CN_ave2=getCN_dis_N(atom,2)
            CNS3,diss3,CN_ave3=getCN_dis_N(atom,3)
            CNS4,diss4,CN_ave4=getCN_dis_N(atom,4)
            GCNs1,diss1,GCN_ave1=getCN_dis_N(atom,1,option='GCN')
            GCNs2,diss2,GCN_ave2=getCN_dis_N(atom,2,option='GCN')
            GCNs3,diss3,GCN_ave3=getCN_dis_N(atom,3,option='GCN')
            GCNs4,diss4,GCN_ave4=getCN_dis_N(atom,4,option='GCN')
            # print(GCN_ave1,GCN_ave2,GCN_ave3,GCN_ave4)
            # mean_c=CN_ave1
            # RMS_c=np.sqrt(np.sum((CNS-mean_c)**2/len(CNS)))
            diameter_2radius=diameter_max(atom.get_positions())
            diameter_pcaM=diameter_pca(atom.get_positions())
            diameter_xyM=diameter_xy(atom.get_positions())
            atom_num=len(atom.get_positions())#"Departure from sphere":np.round(zeta,6),
            sur_per_CN=surface_per(atom,method='CN')
            sur_per_GCN=surface_per(atom,method='GCN')
            sur_per_CN_value=surface_CN(atom,method='CN')
            sur_per_GCN_value=surface_CN(atom,method='GCN')
            try:
                ellipsoid_oblate=ellipsoid(atom)
            except:
                ellipsoid_oblate=[0,0,0]
                    # "flatten":np.round(eta,2)+1,
            dis_dict={
                    "CN1":np.round(CN_ave1,2),
                    "CN2":np.round(CN_ave2,2),
                    "CN3":np.round(CN_ave3,2),
                    "CN4":np.round(CN_ave4,2),
                    "surface_CN":np.round(sur_per_CN_value,2),
                    "GCN1":np.round(GCN_ave1,2),
                    "GCN2":np.round(GCN_ave2,2),
                    "GCN3":np.round(GCN_ave3,2),
                    "GCN4":np.round(GCN_ave4,2),
                    "surface_GCN":np.round(sur_per_GCN_value,2),
                    "bond_length":np.round(np.mean(diss1),4),
                    "diameter_2radius":np.round(diameter_2radius,2),
                    "diameter_pca":np.round(diameter_pcaM,2),
                    "diameter_xy":np.round(diameter_xyM,2),
                    "MIAD":np.round(MIAD(atom),2),
                    "SAF_CN":np.round(sur_per_CN,2),
                    "SAF_GCN":np.round(sur_per_GCN,2),
                    "Departure from sphere(moment)":np.round(zeta,6),
                    "oblateness_moment":np.round(eta,2)+1,
                    "oblateness_pca":np.round(flatten,2),
                    "atom_number":atom_num,
                    "ellipsoid_a":ellipsoid_oblate[0],
                    "ellipsoid_b":ellipsoid_oblate[1],
                    "ellipsoid_c":ellipsoid_oblate[2]}
            return dis_dict
        if all==False:
            dis_dict={}
            if "CN1" in descriptors:
                CNS1,diss1,CN_ave1=getCN_dis_N(atom,1)
                dis_dict["CN1"]=np.round(CN_ave1,2)
            if "CN2" in descriptors:
                CNS2,diss2,CN_ave2=getCN_dis_N(atom,2)
                dis_dict["CN2"]=np.round(CN_ave2,2)
            if "CN3" in descriptors:
                CNS3,diss3,CN_ave3=getCN_dis_N(atom,3)
                dis_dict["CN3"]=np.round(CN_ave3,2)
            if "CN4" in descriptors:
                CNS4,diss4,CN_ave4=getCN_dis_N(atom,4)
                dis_dict["CN4"]=np.round(CN_ave4,2)
            if "surface_CN" in descriptors:
                surface_CN_values=surface_CN(atom,method='CN')
                dis_dict["surface_CN"]=np.round(surface_CN_values,2)
            if "GCN1" in descriptors:
                CNS1,diss1,GCN_ave1=getCN_dis_N(atom,1,option='GCN')
                dis_dict["GCN1"]=np.round(GCN_ave1,2)
            if "GCN2" in descriptors:
                CNS2,diss2,GCN_ave2=getCN_dis_N(atom,2,option='GCN')
                dis_dict["GCN2"]=np.round(GCN_ave2,2)
            if "GCN3" in descriptors:
                CNS3,diss3,GCN_ave3=getCN_dis_N(atom,3,option='GCN')
                dis_dict["GCN3"]=np.round(GCN_ave3,2)
            if "GCN4" in descriptors:
                CNS4,diss4,GCN_ave4=getCN_dis_N(atom,4,option='GCN')
                dis_dict["GCN4"]=np.round(GCN_ave4,2)
            if "surface_GCN" in descriptors:
                surface_CN_values=surface_CN(atom,method='GCN')
                dis_dict["surface_GCN"]=np.round(surface_CN_values,2)
            if "bond_length" in descriptors:
                CNS1,diss1,CN_ave1=getCN_dis_N(atom,1)
                dis_dict["bond_length"]=np.round(np.mean(diss1),2)
            if "diameter_2radius" in descriptors:
                diameter_2radius=diameter_max(atom.get_positions())
                dis_dict["diameter_2radius"]=np.round(diameter_2radius,2)
            if "diameter_pca" in descriptors:
                diameter_pcaM=diameter_pca(atom.get_positions())
                dis_dict["diameter_pca"]=np.round(diameter_pcaM,2)
            if "diameter_xy" in descriptors:
                diameter_xyM=diameter_xy(atom.get_positions())
                dis_dict["diameter_xy"]=np.round(diameter_xyM,2)
            if "MIAD" in descriptors:
                MIAD_value=MIAD(atom)
                dis_dict["MIAD"]=np.round(MIAD_value,2)
            if "SAF_CN" in descriptors:
                sur_per=surface_per(atom)
                dis_dict["SAF_CN"]=np.round(sur_per,2)
            if "SAF_GCN" in descriptors:
                sur_per=surface_per(atom)
                dis_dict["SAF_GCN"]=np.round(sur_per,2)
            if "Departure from sphere(moment)" in descriptors:
                zeta,eta=moment_descriptor(atom)
                dis_dict["Departure from sphere(moment)"]=np.round(zeta,6)
            if "oblateness_moment" in descriptors:
                zeta,eta=moment_descriptor(atom)
                # print(zeta,eta)
                dis_dict["oblateness_moment"]=np.round(eta,2)
            if "oblateness_pca" in descriptors:
                flatten=pca_oblate(atom)
                dis_dict["oblateness_pca"]=np.round(flatten,2)
            if "atom_number" in descriptors:
                atom_num=len(atom.get_positions())
                dis_dict["atom_number"]=atom_num   
            if "ellipsoid" in descriptors:
                try:
                    ellipsoid_oblate=ellipsoid(atom)
                    dis_dict["ellipsoid_a"]=ellipsoid_oblate[0]
                    dis_dict["ellipsoid_b"]=ellipsoid_oblate[1]
                    dis_dict["ellipsoid_c"]=ellipsoid_oblate[2]
                except:
                    dis_dict["ellipsoid_a"]=0
                    dis_dict["ellipsoid_b"]=0
                    dis_dict["ellipsoid_c"]=0
                    print("ellipsoid cannot be calculated")
            reordered_dict = {k: dis_dict[k] for k in descriptors}    
            if dis_dict=={}:
                print("No descriptors are found in the program")
            return reordered_dict
    


def pca_oblate(atom:Atoms):
    pca=PCA(n_components=3)
    pos=atom.get_positions()
    center=pos.mean(axis=0)
    pos=pos-center
    pos = np.round(pos,5)
    dimension=dim_pos(pos)
    if dimension<3:
        return 1

    # c_next,b_next,a_next=np.round(np.sort(max_value-min_value),5)
    
    result=pca.fit(pos)
    # pos_fit=result.fit_transform(pos)
    variences=result.explained_variance_ratio_
    sigma1,sigma2,sigma3=np.round(np.sqrt(sorted(variences,reverse=True)),5)
    sigma_eq=(sigma1+sigma2)/2
    # flatten=np.round(1-c/b,5)
    # elongate=np.round(sigma_eq/sigma_eq,5)
    # diameter=np.round(np.sqrt(a**2+b**2+c**2)) #another method to calculate diameter
    
    sigma_pole=sigma3
    oblateness=(sigma_eq-sigma_pole)/sigma_eq
    flatten=np.round(oblateness,5)
    return flatten



def radius(atom:Atoms):
    """
    This algorithm calculates the radius of
    the circumscribed sphere of the polyhedron
    particle
    :param positions: coordinates
    :return: radius of particle
    """
    positions = atom.get_positions()
    center = positions.mean(axis=0)
    vector = positions - center
    dis = np.linalg.norm(vector,axis=1)
    R = np.max(dis)
    return R


def my_ceil(a, precision=2):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def center_atom_finder(ele_num, supercell_info):
    positions = supercell_info.arrays['positions']
    num = supercell_info.arrays['numbers']
    center_pos = np.mean(positions, axis=0)
    atom_dis = cdist([center_pos], positions,metric='euclidean')
    stacked_data = np.hstack((num.reshape(-1, 1), atom_dis.reshape(-1, 1), positions[:, :]))
    data = pd.DataFrame(stacked_data, columns=["atom", "dis2center", "x", "y", "z"])
    data_ele = data[data['atom'] == float(ele_num)]
    center_ele = data_ele[data_ele['dis2center'] == data_ele['dis2center'].min()]
    index = center_ele.index[0]

    center_atom_pos = np.array([center_ele['x'], center_ele['y'], center_ele['z']]).reshape(1, 3)
    update_atom_dis = cdist(center_atom_pos, positions,metric='euclidean').reshape(-1, 1)
    data['dis2center'] = data['dis2center'].replace(np.array(data['dis2center']), update_atom_dis)

    # shift the center position to the assigned center atom
    data['x'] = data['x'] - center_atom_pos[0][0]
    data['y'] = data['y'] - center_atom_pos[0][1]
    data['z'] = data['z'] - center_atom_pos[0][2]
    data = data.sort_values(by="dis2center", ascending=True)
    return index, data


def nn_dis_rough(data, layer=1):
    return my_ceil(data['dis2center'], 2).unique()[layer]


def atom_by_layers(ele_num, supercell, layer=1):
    _, data = center_atom_finder(ele_num, supercell)
    distance = nn_dis_rough(data, layer)
    supercell_2 = supercell.copy()
    data_cut_sphere = data[data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2 <= distance ** 2]
    rebu_coord = np.hstack((np.array(data_cut_sphere['x']).reshape(-1, 1),
                            np.array(data_cut_sphere['y']).reshape(-1, 1),
                            np.array(data_cut_sphere['z']).reshape(-1, 1)))
    rebu_num = np.array(data_cut_sphere['atom'])
    supercell_2.arrays['positions'] = rebu_coord
    supercell_2.arrays['numbers'] = np.int0(rebu_num)
    print(
        f"#{layer} layer is {distance} Angstrom from center element (The distance is the 2 digits ceil of exact value of nn distance).")
    # view(supercell_2,viewer='x3d')=
    return supercell_2

def neighbor_dis(atoms,pos_center,cutoff):
    """
    calculate the distance between the center atom and its neighbors

    Args:
        atoms (ase.Atoms): the atoms object of undistorted structure
        pos_center (np.array): the position of the center atom 
        cutoff (float): cutoff distance, the distortion will be calculated within this distance

    Returns:
        np.array,np.array: the list of the index of the neighbors and the list of the distance between the center atom and its neighbors
    """
    pos=atoms.arrays['positions']
    dis=cdist(pos_center, pos, metric="euclidean")
    dist_unique=np.unique(dis)
    index_NN=[]
    dist_NN=[]
    for i in range(len(dist_unique)):
        if dist_unique[i]<=cutoff and dist_unique[i]!=0:
            index=np.where(dis==dist_unique[i])[0]
            for k in index:
                index_NN.append(int(k))
                dist_NN.append(dist_unique[i])
    return np.array(index_NN),np.array(dist_NN)

def MIAD(atoms):
    """
    Mean interatomic distance:
       $l_{miad}=\frac{1}{N(N-1)}\sum^N_{i,j=1}|R_i-R_j|$
    """
    dis=atoms.get_all_distances(mic=False)
    lmiad=1/(len(dis)*(len(dis)-1))*np.sum(dis)
    return lmiad

def read_xyz_to_properties(foldername:str)->pd.DataFrame:
    """
    read the xyz files in the folder and calculate the properties of each structure
    """


    # Initialize an empty list to store the data
    data = []
    index=[]
    if system=="Windows":
        split_symbol='\\'
    else:
        split_symbol='/'
    filename_list=glob(foldername+"/*.xyz")
    # Loop through all files in the folder
    for filename in tqdm(filename_list):
        try:
            atoms = read(filename)
            properties_temp = descriptor_table(atoms)
            filename=filename.split(split_symbol)[-1]  # Add  to properties
            data.append(properties_temp)
            index.append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data, index=index)

    return df

