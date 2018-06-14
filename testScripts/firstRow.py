import csv
import glob

files = glob.glob("testRuns/run2/*")

for filename in sorted(files):
  with open(filename, newline='') as f:
    reader = csv.reader(f)
    row1 = next(reader)  # gets the first line
    row2 = next(reader)
  
    print(filename)
    print("  " + str(float(row2[3])/1000))

