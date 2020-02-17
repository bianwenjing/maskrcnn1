import json
import csv

K = {}
with open("/home/wenjing/scannetv2-labels.combined.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    lines = []
    for line in tsvreader:
        lines.append(line)
    for i in range(1, len(lines)):
        raw_category = lines[i][1]
        category = lines[i][2]
        id = lines[i][0]
        K[id] = 0
with open('/home/wenjing/storage/anno/train_git_many_100.txt') as json_file:
    data = json.load(json_file)

# for i in data['annotations']:
#     category_id = str(i['category_id'])
#     K[category_id] += 1

for i in data['annotations']:
    category_id = str(i['category_id'])
    K[category_id] += 1

valid_category = []
for key, item in K.items():
    if item >= 0:
        valid_category.append(key)

print(len(valid_category))

with open('/home/wenjing/storage/category_full.txt', 'w') as f:
    for item in valid_category:
        f.write("%s\n" % item)