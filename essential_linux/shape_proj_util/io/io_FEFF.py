
import os
import shutil
import re
import glob
from .io_general import dir_list, dir_create, read_xyz

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

