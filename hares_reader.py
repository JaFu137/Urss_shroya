import glob, os
import numpy as np

"""
Convert Hares .out file into .csv file readable by the script
"""
def reader(filename):
    array = np.empty([0,6])
    with open("hares/" + filename + ".out", 'r') as file:
        line = file.readline()
        while line:
            split = line.split()
            if split[0] != '#':
                array = np.append(array, [[None, filename, split[1], split[0], split[2], split[3]]], axis=0)
            line = file.readline()
    array = array[array[:, 2].argsort()]
    np.savetxt("input_data/" + filename + ".csv", array, delimiter=",", fmt='%s')

def main():
    os.chdir("hares")
    files = glob.glob("*.out")
    os.chdir("../")
    files = [os.path.splitext(file)[0] for file in files]

    for file in files:
        print(file)
        reader(file)

if __name__ == "__main__":
    main()
