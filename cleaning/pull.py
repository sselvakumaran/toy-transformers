import sys
import csv
# pulls all non-pulled rows from sources.csv

def pull_data(src_path):
  src_file = open(src_path, 'r')
  reader = csv.reader(src_file)
  next(reader)
  sources = list(reader)
  for line in sources:
    print(line)

if __name__ == "__main__":
  source_csv_path = sys.argv[1]
  pull_data(source_csv_path)