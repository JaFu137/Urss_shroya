import glob, os
import methodA
import methodC 
import json
import concurrent.futures

"""
Run scripts on all .csv file in input_data folder three files at a time
"""

os.chdir("input_data")
files = glob.glob("*.csv")
os.chdir("../")
os.chdir("output/method_A/")
finished_files = glob.glob("*.png")
os.chdir("../")
os.chdir("../")
files = [os.path.splitext(file)[0] for file in files]
finished_files = [os.path.splitext(file)[0] for file in finished_files]

for file in finished_files:
    try:
        files.remove(file)
    except ValueError:
        pass

def run(file):
    try:
        methodA.main(file, 1000, 500, 1000, spread=True)
        # methodC.main(file, 1000, 500, 1000, 58, spread=True)
        print(file + ":Ok")
    except TypeError:
        print(file + ":skipped")
        pass

def main():
    print(files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for file in zip(files, executor.map(run, files)):
            pass

if __name__ == '__main__':
    main()
