import json, os
import time
datapath = os.path.dirname(os.path.dirname(__file__)) + '\\result.json'
Analyze_Threshold = 0

class TimeCollector(object):
    timestamp = []
    data_collected = []
    
    def __init__(self):
        super().__init__()
        with open(datapath, 'r', encoding='utf8') as f:
            lines = f.read().replace('\n', '')
            datum = json.loads(lines)
            
            for data in datum:
                try:
                    # "../drive/MyDrive/AIIntro/[processed]dream_picture/20200610-20200620-camera-3/Camera-3_2020-06-18_00:45:06-check.jpg"
                    timestamp = data['filename'].split('/')[-1].replace('-check.jpg','') # Camera-3_2020-06-18_00:45:06-check.jpg
                    timestamp = timestamp.split('_')
                    timestamp = time.strptime(f'{timestamp[1]} {timestamp[2]}', "%Y-%m-%d %H:%M:%S")
                    self.AddNewData(time= int(time.mktime(timestamp)), objects = data['objects'])
                except Exception as e:
                    print( f"{e} while {data['filename']}" )
        
        self.analyzed_data = self.Compile()
    
    def AddNewData(self, time = int, objects = object):
        """傳入時間戳記與{物體種類:數量}的字典集"""
        obj = dict()
        
        # filter 1
        for o in objects:
            if o['confidence'] > Analyze_Threshold:
                obj[o['name']] = o['confidence']
                
        self.data_collected.append((time, obj))
    
    def Compile(self):
        """
        重新計算人物出現的時間\n
        回傳: (時間(int分鐘), 物體(dict), 下次有人的時間(int分鐘))
        """
                
        # sort
        def sortkey(elem):
            (time, objects) = elem
            return time
        self.data_collected.sort(key = sortkey)
        
        # count time
        for (time, objects) in self.data_collected:
            if "person" in objects.keys():
                self.timestamp.append(time)
                
        # seek again
        seeker = 0 # 搜索指針
        analyzed_data = [] # 回傳資料
        for (time, objects) in self.data_collected:
            # 計算距離下個時間戳記的差值
            nt = int((self.timestamp[seeker] - time)/60)
            analyzed_data.append((int(time / 60), objects, nt ))
            
            if time == self.timestamp[seeker]:
                seeker += 1            
        
        return analyzed_data
    
    def Reset(self):
        """刪除所有資料"""
        self.timestamp = []
        self.data_collected = []
        self.analyzed_data = None

class TimeMarker(object):
    int_minuteGate = 40
    filepath = os.path.dirname(os.path.dirname(__file__))
    categories = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','trafficlight','firehydrant','stopsign','parkingmeter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sportsball','kite','baseballbat','baseballglove','skateboard','surfboard','tennisracket','bottle','wineglass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hotdog','pizza','donut','cake','chair','couch','pottedplant','bed','diningtable','toilet','tv','laptop','mouse','remote','keyboard','cellphone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddybear','hairdrier','toothbrush','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','pottedplant','sheep','sofa','train','tvmonitor']
    print(len(categories))
    categories = list(set(categories))
    
    def __init__(self, analyzed_data):
        import numpy as np
        x = []
        y = []
        
        def Tolist(az):
            """把dict 轉成 one-hot list"""
            ret = [float(0)]*len(self.categories)
            for index, id in enumerate(self.categories):
                if id in az.keys():
                    ret[index] = float(1)
            return ret
        
        switch = False
        
        cntr = [0, 0]
        for (t, az, nt) in analyzed_data: # time, analyze , next time
            if 'person' in az:
                switch = True
            x.append(Tolist(az))
            if nt > self.int_minuteGate or switch is False:
                if not 'person' in az:
                    y.append([float(0)])
                    cntr[0] += 1
                switch = False
            else:
                y.append([float(1)])
                cntr[1] += 1
        print(x)
        print(cntr[0] / (cntr[0] + cntr[1]))
        np.savez('./data.npz', x = x, y = y)

if __name__ == "__main__":
    ti = TimeCollector()
    tm = TimeMarker(ti.analyzed_data)