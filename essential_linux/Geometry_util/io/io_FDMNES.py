from ase import Atoms


def write_inp_from_atom(atom,filepath):
    with open(filepath,'w') as file1:
        file1.write("Filout\n")
        file1.write("fdmnes\n\n")
        file1.write("Edge\n")
        file1.write("K\n\n")
        file1.write("Range                 !Energy range of calculation(eV)\n")
        file1.write("-10     0.25 10.0    0.5 20.0     1.0 35.0   ! first energy, step, intermediary energy, step ..., last energy\n\n")
        file1.write("Radius                                        ! Radius of the cluster where final state calculation is performed\n")
        file1.write("6                                  ! For a good calculation, this radius must be increased up to 6 or 7\n")
        file1.write("Z_absorber\n")
        file1.write("78\n\n\n")
        file1.write("Molecule\n")
        file1.write("1  1   1   90  90  90\n")
        for i in range(len(atom.arrays['positions'])):
            file1.write(str(atom.arrays['numbers'][i])+'   '+str(atom.arrays['positions'][i][0])+'   '+str(atom.arrays['positions'][i][1])+'   '+str(atom.arrays['positions'][i][2])+'\n')
        file1.write("\n\nConvolution\n\n")
        file1.write("End")

def write_inp_from_xyz(xyzfile,filepath,atom_num):
    with open(xyzfile,'r') as file0:
        lines=file0.readlines()
    

    with open(filepath,'w') as file1:
        file1.write("Filout\n")
        file1.write("fdmnes\n\n")
        file1.write("Edge\n")
        file1.write("K\n\n")
        file1.write("Range                 !Energy range of calculation(eV)\n")
        file1.write("-5     0.25 10.0    0.5 20.0     1.0 30.0   ! first energy, step, intermediary energy, step ..., last energy\n\n")
        file1.write("Radius                                        ! Radius of the cluster where final state calculation is performed\n")
        file1.write("6                                  ! For a good calculation, this radius must be increased up to 6 or 7\n")
        file1.write("Absorber\n")
        file1.write("78\n\n\n")
        file1.write("Molecule\n")
        file1.write("1  1   1   90  90  90\n")
        for i in range(2,len(lines)):
            line = lines[i].split()
            file1.write(str(atom_num)+'   '+line[1]+'   '+line[2]+'   '+line[3]+'\n')
        file1.write("\n\nConvolution\n\n")
        file1.write("End")
