import os
import glob

os.chdir('./TimeAnalyze/data/npz/result')  
index = [str(i) for i in range(1,51)]
writeback = [f"name,kind,{','.join(index)}\n"]

for file in glob.glob("*.csv"):
    data = [[], []]
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            d = line.split(',')
            for i in [0,1]:
                data[i].append(d[i])
    
    writeback.append(','.join([file] + data[0]) + '\n')
    writeback.append(','.join([file] + data[1]) + '\n')
   
os.chdir('./../../..')               
with open('./concate.csv', 'w', encoding='utf8') as f:
    f.writelines(writeback) 