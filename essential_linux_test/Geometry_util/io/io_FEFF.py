
import os
import shutil
import re
import glob
from .io_general import dir_list, dir_create, read_xyz
from pymatgen.core import Structure, Element,Molecule
import pymatgen.io.feff
import numpy as np
from tabulate import tabulate
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
from ase.io.cif import read_cif
def read_xmu(path):
    """ 
    :param path: sites folder
    :return:
    """
    xmu= pd.read_csv(path+"/xmu.dat", delimiter=r' +', skiprows=17,
                         names=["omega", "e", "k", "mu", "mu0", "chi"], engine='python')
    omega = xmu["omega"]
    mu    = xmu["mu"]
    return omega,mu

def write_spectrum(omega,mu,name,to_dir,n_atoms):
    with open(to_dir+"/"+name+".txt","w") as file1:
        file1.write("NAME:\t"+name+"\n")
        file1.write("number of atoms:\t" +str(n_atoms)+"\n")

        file1.write("omega\tmu\n")
        for i in range(len(omega)):
            file1.write(str(omega[i])+'\t'+str(mu[i])+"\n")

def copy_sites(from_dir, to_dir):
    """
    :param parent_dir: The dir contains all sites
    :return:
    """
    sites = dir_list(from_dir)
    for i in range(len(sites)):
        shutil.copy(sites[i]+"/xmu.dat",to_dir)
        str1 = re.search(r"[a-z]+[0-9]+([A-Z][0-9]+\.[0-9]+ [A-Z][0-9]+\.[0-9]+)?",from_dir).group()
        str2 = re.search(r"sites\.[0-9]+",sites[i]).group()
        os.rename(to_dir+"/xmu.dat",to_dir+'/'+str1+"_"+str2+"_xmu.dat")



def makedir(name):
    """helper function: check if the directory exists. if no, then create it."""
    strr="./"+name
    if not os.path.exists(strr):
        os.mkdir(strr)


def FEFF_writing_one(element,atom_file,FEFF_headfilename,outputdir,equ_sites,n,code_num,options="none"):
    """
    :param element: chemical element
    :param atom_file: xyz file
    :param FEFF_headfilename: The FEFF input headfile
    :param outputdir: the directory you want put your files in
    :param n: the index of absorb atom
    :param code_num: the index of file
    :param options: 1: running on windows
                   -1: only write files (default) (new feature)
    :return: no return file
    """
    with open(FEFF_headfilename,'r') as file1:
        head = file1.readlines()
    positions=read_xyz(atom_file)
    num_atom = len(positions)
    dir_create(outputdir+'/sites.'+str(code_num))

    path = outputdir+"/sites." + str(code_num) + "/feff.inp"

    if options=="windows":
        shutil.copy("files_need/run_feff.bat",outputdir+"/sites." + str(code_num)+"/")
    if options=="unix":
        shutil.copy("files_need/feff.sh", outputdir + "/sites." + str(code_num) + "/")

    with open(outputdir+"/sites." + str(code_num)+'/'+"equ_sites.txt","w") as file1:
        file1.write("absorber:\t"+str(n)+"\n")
        file1.write("# of equ-sites(include it self):\t"+str(len(equ_sites))+'\n')
        strr=""
        for i in range(len(equ_sites)):
            strr+=str(equ_sites[i])+' '
        file1.write("equ-sites for the absorber:\t" + strr+'\n')

    with open(path,'w') as file1:
        for str_line in head:
            file1.write(str_line)
        file1.write("\n\nATOMS\t* this\tlist\tcontains\t"+str(num_atom)+'\tatoms')
        file1.write("\n*\tx\ty\tz\tipot\ttag\tdistance\n")
        for i in range(len(positions)):
            if i == n:
                file1.write(str(positions[i][0]) + '\t' + str(positions[i][1]) + '\t' + str(positions[i][2]) + '\t' + str(
                    0) + '\t' + element+'\n')
            else:
                file1.write(str(positions[i][0]) + '\t' + str(positions[i][1]) + '\t' + str(positions[i][2]) + '\t' + str(
                    1) + '\t' + element+'\n')
        file1.write("END")

def FEFF_writing_multiple(element,inputfilename,inputpath,headfile,cutoff,options="none"):
    """
    :param element: chemical element
    :param inputfilename:
    :param inputpath: xyz file
    :param headfile: FEFF file
    :param cutoff: cutoff
    :param options: windows: running on windows
                   unix: only write files (default) (new feature)
    """
    positions = read_xyz(inputpath)

    absorber=particles.equ_sites((positions),cutoff)
    absorber_indexes=list(absorber.keys())

    outputdir =inputfilename
    #dir_create("sites")
    path = "sites/" + outputdir
    if options=="windows":
        shutil.copy("files_need/run.bat", path+"/")
        #shutil.copy("files_need/run.bat", "sites/")
    #if options=="unix":
    #    shutil.copy("files_need/run_feff.sh", "sites/")
    dir_create(path)
    file_index=1
    for i in absorber_indexes:
        FEFF_writing_one(element,inputpath, headfile, path, absorber[i],i,file_index,options)
        file_index= file_index+1


def write_labelfile(path,labels):
    """

    :param path:
    :param label_array:
    """
    name = path+"/label.txt"
    with open(name,'w') as file1:
        file1.write("shape\tA\tB\tR\n")
        for i in range(len(labels)):
            strr = str()  # write labels
            for j in range(len(labels[0])):
                    strr += str(labels[i][j]) + '\t'
            strr = strr + '\n'
            file1.write(strr)

def get_averaging(path):
    """
    :param path: only need to set the position to the shape folder
    :return:
    """
    def get_factor(sites_path):
        with open(sites_path+"\equ_sites.txt") as file1:
            lines=file1.readlines()
        n_equ_site=int(lines[1].split('\t')[1])
        return n_equ_site
    sites = dir_list(path)
    n_equ_site=[]
    sum_omega=0
    sum_mu=0
    for i in range(len(sites)):
        factor = get_factor(sites[i])
        omega,mu = read_xmu(sites[i])
        sum_omega += omega*factor
        sum_mu += mu*factor
        n_equ_site.append(factor)
    ave_omega = sum_omega/sum(n_equ_site)
    ave_mu = sum_mu/sum(n_equ_site)
    return ave_omega, ave_mu, sum(n_equ_site)




def calc_pot_atoms_list(path=None, sructure=None,absorber = None, radius = 8, absorber_list = [],supercell=[]):
    """
    Calculate the POTENTIAL and ATOMS card of feff input of given structure.
    """
    if path!=None and path.split('.')[1]=='xyz':
        structure=Molecule.from_file(path)
    elif path!=None and path.split('.')[1]=='cif':
        structure=Structure.from_file(path)
        if supercell!=[]:
            structure=structure.make_supercell(supercell)
    elif path==None and sructure is not None:
        structure=sructure
        if supercell!=[]:
            structure=structure.make_supercell(supercell)
    else:
        raise ValueError("Please specify the path of structure file or the structure object.") 
    
    
    pot_atoms_list = []
    
    if len(absorber_list) == 0:
        if absorber is None:
            raise ValueError("Please specify the absorber element.")
        
        absorber_species = Element(absorber)
        absorber_list = np.where(np.array(structure.species) == absorber_species)[0]
    
    for i in absorber_list:
        pot = pymatgen.io.feff.inputs.Potential(structure, int(i))
        central_element = Element(pot.absorbing_atom)
        ipotrow = [[0, central_element.Z, central_element.symbol, -1, -1, 0.001, 0]]
        for el, amt in pot.struct.composition.items():
            ipot = pot.pot_dict[el.symbol]
            ipotrow.append([ipot, el.Z, el.symbol, -1, -1, amt, 0])
              
        cluster = np.array(pymatgen.io.feff.inputs.Atoms(structure, int(i), radius).get_lines())
        # sort by distance
        cluster = cluster[np.argsort(cluster[:, 5].astype(float))]
        
        # obtain unique potential
        unique_potential = np.unique(cluster[:, 3])
        map_potential = {unique_potential[i]: str(i) for i in range(len(unique_potential))}
        
        #pot index
        pot_index=list(np.array(ipotrow)[:,0]) 
        if len(map_potential)!=len(ipotrow):
            miss_pot=set(pot_index).difference(list(map_potential.keys()))
            raise ValueError(f"The radius is too short to include all potentials, please choose a larger radius(missing potential {miss_pot}).")
        # replace the potential label
        #cluster[:, 3] = [map_potential[str(i)] for i in cluster[:, 3]]
        
        # pot = []
        # for i in map_potential.keys():
        #     pot.append([map_potential[i], *ipotrow[int(i)][1:]])
        
        pot_atoms_list.append({"potential": tabulate(ipotrow, tablefmt="plain"), "atoms": tabulate(cluster, tablefmt="plain")})
        
    return pot_atoms_list

def cif_to_feff(cif_path:str,absober:str,edge:str,radius=12,absorber_list=[],supercell=[3,3,3],save_path=None):
    if save_path is None:
        save_path="feff.inp"
    atoms=read_cif(cif_path)
    struc=AseAtomsAdaptor.get_structure(atoms)
    pot=calc_pot_atoms_list(path=None, sructure=struc,absorber = "Cr", radius = 12, absorber_list = [],supercell=[3,3,3])
    with open(save_path, "w") as f:
        f.write("TITLE\n")
        f.write("xx\n")
        f.write("\n\n")
        f.write("EDGE\n")
        f.write(f"{edge}\n")
        f.write("\n")
        f.write("S02  1.0\n")
        f.write("COREHOLE RPA\n")
        f.write("CONTROL 1 1 1 1 1 1\n")
        f.write("\n\n")
        f.write("FMS 7.5 0\n")
        f.write("EXCHANGE 0  0.9 -1\n")
        f.write("SCF 5.6 0 100 0.1 1\n")
        f.write("XANES 6 0.05 0.1\n")
        f.write("\n\n")
        f.write("POTENTIAL\n")
        f.write(pot[0]["potential"])
        f.write("\n\n")
        f.write("ATOMS\n")
        f.write(pot[0]["atoms"])
        f.write("\n\n")
        f.write("END")

