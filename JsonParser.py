import json

f = open('result.json')

openedJson = json.load(f)
stat = []

for j in openedJson:
    objects = j['objects']
    temp = [0] * 80
    for o in objects:
        temp[int(o['class_id'])] += 1
    stat.append(temp)

for s in stat:
    print(s)