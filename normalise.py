import csv
import sys
import random

if(len(sys.argv) == 2):
	_set = str(sys.argv[1])
else:
	sys.exit("enter a file to normalise")

counts = [0, 0, 0, 0, 0]

with open('data/%s.csv' % (_set)) as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		counts[int(row["level"])] += 1

print(counts)
_min = min(counts)

print(_min)
new_csv = []
fresh_counts = [0, 0, 0, 0, 0]
with open('data/%s.csv' % (_set)) as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		if(fresh_counts[int(row["level"])] < _min):
			new_csv.append(row["image"] + "," + row["level"])
			fresh_counts[int(row["level"])] += 1

print(new_csv)
print(fresh_counts)
random.shuffle(new_csv)

file = open("new.csv", "w")

print(new_csv)
for line in range(0, len(new_csv)):
	file.write(new_csv[line])
	file.write("\n")

