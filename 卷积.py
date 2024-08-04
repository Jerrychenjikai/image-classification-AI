import cv2
import math
import os
import json
import random
import copy
import torch
import datetime
from changeFunction import change_function
import quicksort

yes_folder='D:\\Jerry\\photo AI training\\CatVsDog\\Cat\\'
no_folder='D:\\Jerry\\photo AI training\\CatVsDog\\Dog\\'

extension='.jpg'

model_file='AI data.txt'

image_chang=94*2+4
image_kuan=94*2+4

if os.path.exists(model_file):
    num_models=7
else:
    num_models=45

def find_files_with_extension(folder_path, extension):
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            if file.endswith(extension):
                files.append(folder_path+file)
    return files

yes_files=find_files_with_extension(yes_folder,extension)
no_files=find_files_with_extension(no_folder,extension)

class juanjihe:
    def __init__(self,size=3):
        self.size=size
        self.data=torch.rand(self.size,self.size)*4-2
        self.data=self.data.to('cuda').unsqueeze(0).unsqueeze(0)

    def apply(self,image):
        image=image.clone()

        return torch.nn.functional.conv2d(image,self.data)

    def write(self,model_file=model_file):
        with open(model_file,'a') as fo:
            for i in range(self.size):
                for j in range(self.size):
                    fo.write(str(float(self.data[0][0][i][j]))+'\n')
    def read(self,file):
        for i in range(self.size):
            for j in range(self.size):
                self.data[0][0][i][j]=float(file[i*self.size+j])
    def change(self,direc):
        if direc[0][0]==0:
            delta=torch.rand(self.size,self.size)*0.4-0.2
            delta=delta.to('cuda').unsqueeze(0).unsqueeze(0)
            self.data+=delta
        else:
            delta=torch.rand(self.size,self.size)*0+0.1
            delta=delta.to('cuda')
            direc=torch.tensor(direc).to(torch.device('cuda'))

            delta*=direc
            delta.unsqueeze(0).unsqueeze(0)
            self.data+=delta
            
    def clone(self,sample):
        self.data=sample.data.clone()

class model:
    def __init__(self):
        self.filter0=[]
        for i in range(2):
            self.filter0.append(juanjihe(5))

        
        self.filter1=[]
        for i in range(4):
            self.filter1.append(juanjihe())

        self.filter2=[]
        for i in range(4):
            self.filter2.append(juanjihe())

        self.filter3=[]
        for i in range(4):
            self.filter3.append(juanjihe())

        self.model_1=[]  #长度为100
        self.model_2=[]  #长度为100
        self.model_3=[]  #长度为100
        self.model_4=[]  #长度为100
        self.model_11=[[],[],[],[]]    #长度为4

        for i in range(100):
            self.model_1.append([])
            self.model_2.append([])
            self.model_3.append([])
            self.model_4.append([])

        for j in range(100):
            for k in range(3):
                self.model_1[j].append(random.uniform(-2,2))
                self.model_2[j].append(random.uniform(-2,2))
                self.model_3[j].append(random.uniform(-2,2))
                self.model_4[j].append(random.uniform(-2,2))

        self.model_1=torch.tensor(self.model_1).to(torch.device('cuda'))
        self.model_2=torch.tensor(self.model_2).to(torch.device('cuda'))
        self.model_3=torch.tensor(self.model_3).to(torch.device('cuda'))
        self.model_4=torch.tensor(self.model_4).to(torch.device('cuda'))

        for i in range(4):
            for j in range(3):
                self.model_11[i].append(random.uniform(-2,2))

        self.model_11=torch.tensor(self.model_11).to(torch.device('cuda'))

        self.summ=0#误差之和

        self.delta_filter0=[[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                            [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]]

        self.delta_filter1=[[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]]

        self.delta_filter2=[[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]]

        self.delta_filter3=[[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]]

        self.delta_model_1=[]
        self.delta_model_2=[]
        self.delta_model_3=[]
        self.delta_model_4=[]
        self.delta_model_11=[[],[],[],[]]

        for i in range(100):
            self.delta_model_1.append([])
            self.delta_model_2.append([])
            self.delta_model_3.append([])
            self.delta_model_4.append([])

        for j in range(100):
            for k in range(3):
                self.delta_model_1[j].append(0)
                self.delta_model_2[j].append(0)
                self.delta_model_3[j].append(0)
                self.delta_model_4[j].append(0)

        for i in range(4):
            for j in range(3):
                self.delta_model_11[i].append(0)

    def clone(self,sample,delta=True):
        self.model_1=sample.model_1.clone()
        self.model_2=sample.model_2.clone()
        self.model_3=sample.model_3.clone()
        self.model_4=sample.model_4.clone()

        self.model_11=sample.model_11.clone()

        for i in range(len(self.filter0)):
            self.filter0[i].clone(sample.filter0[i])

        for i in range(len(self.filter1)):
            self.filter1[i].clone(sample.filter1[i])

        for i in range(len(self.filter2)):
            self.filter2[i].clone(sample.filter2[i])

        for i in range(len(self.filter3)):
            self.filter3[i].clone(sample.filter3[i])

        if delta:
            self.delta_filter0=copy.deepcopy(sample.delta_filter0)
            self.delta_filter1=copy.deepcopy(sample.delta_filter1)
            self.delta_filter2=copy.deepcopy(sample.delta_filter2)
            self.delta_filter3=copy.deepcopy(sample.delta_filter3)

            self.delta_model_1=copy.deepcopy(sample.delta_model_1)
            self.delta_model_2=copy.deepcopy(sample.delta_model_2)
            self.delta_model_3=copy.deepcopy(sample.delta_model_3)
            self.delta_model_4=copy.deepcopy(sample.delta_model_4)
            self.delta_model_11=copy.deepcopy(sample.delta_model_11)

    def change(self):
        for i in range(len(self.filter0)):
            self.filter0[i].change(self.delta_filter0[i])
            
        for i in range(len(self.filter1)):
            self.filter1[i].change(self.delta_filter1[i])

        for i in range(len(self.filter2)):
            self.filter2[i].change(self.delta_filter2[i])

        for i in range(len(self.filter3)):
            self.filter3[i].change(self.delta_filter3[i])

        if self.delta_model_1[0][0]==0:
            for j in range(100):
                for k in range(3):
                    self.model_1[j][k]+=random.uniform(-0.2,0.2)
                    self.model_2[j][k]+=random.uniform(-0.2,0.2)
                    self.model_3[j][k]+=random.uniform(-0.2,0.2)
                    self.model_4[j][k]+=random.uniform(-0.2,0.2)

            for i in range(4):
                for j in range(3):
                    self.model_11[i][j]+=random.uniform(-0.2,0.2)
        else:
            for j in range(100):
                for k in range(3):
                    self.model_1[j][k]+=0.05*self.delta_model_1[j][k]
                    self.model_2[j][k]+=0.05*self.delta_model_2[j][k]
                    self.model_3[j][k]+=0.05*self.delta_model_3[j][k]
                    self.model_4[j][k]+=0.05*self.delta_model_4[j][k]

            for i in range(4):
                for j in range(3):
                    self.model_11[i][j]+=0.05*self.delta_model_11[i][j]

    def apply(self,image):
        #卷积+池化
        image1=self.filter0[0].apply(image)
        image2=self.filter0[1].apply(image)

        image1=torch.nn.functional.max_pool2d(image1,2,2,0)
        image2=torch.nn.functional.max_pool2d(image2,2,2,0)

        image=self.filter1[0].apply(image1)
        image+=self.filter1[1].apply(image1)
        image1=image

        image=self.filter1[2].apply(image2)
        image+=self.filter1[3].apply(image2)
        image2=image

        image1=torch.nn.functional.max_pool2d(image1,2,2,0)
        image2=torch.nn.functional.max_pool2d(image2,2,2,0)

        
        image=self.filter2[0].apply(image1)
        image+=self.filter2[1].apply(image1)
        image1=image

        image=self.filter2[2].apply(image2)
        image+=self.filter2[3].apply(image2)
        image2=image

        image1=torch.nn.functional.max_pool2d(image1,2,2,0)
        image2=torch.nn.functional.max_pool2d(image2,2,2,0)


        image=self.filter3[0].apply(image1)
        image+=self.filter3[1].apply(image1)
        image1=image

        image=self.filter3[2].apply(image2)
        image+=self.filter3[3].apply(image2)
        image2=image

        image1=torch.nn.functional.max_pool2d(image1,2,2,0)
        image2=torch.nn.functional.max_pool2d(image2,2,2,0)

        image=image1+image2

        #神经网络
        image=torch.flatten(image)

        summs=[0,0,0,0]

        summs[0]=float(change_function(image,self.model_1[:,0],self.model_1[:,1],self.model_1[:,2]).sum())

        summs[1]=float(change_function(image,self.model_2[:,0],self.model_2[:,1],self.model_2[:,2]).sum())
        summs[2]=float(change_function(image,self.model_3[:,0],self.model_3[:,1],self.model_3[:,2]).sum())
        summs[3]=float(change_function(image,self.model_4[:,0],self.model_4[:,1],self.model_4[:,2]).sum())

        final_sum=0
        for i in range(4):
            final_sum+=change_function(summs[i],self.model_11[i][0],self.model_11[i][1],self.model_11[i][2])

        return torch.sigmoid(torch.tensor(final_sum)).item()

    def write(self,model_file=model_file):
        with open(model_file,'w') as fo:
            for i in self.model_1:
                for j in i:
                    fo.write(str(float(j))+'\n')
            for i in self.model_2:
                for j in i:
                    fo.write(str(float(j))+'\n')
            for i in self.model_3:
                for j in i:
                    fo.write(str(float(j))+'\n')
            for i in self.model_4:
                for j in i:
                    fo.write(str(float(j))+'\n')
            for i in self.model_11:
                for j in i:
                    fo.write(str(float(j))+'\n')

        for i in self.filter0:
            i.write()
        
        for i in self.filter1:
            i.write()

        for i in self.filter2:
            i.write()

        for i in self.filter3:
            i.write()

        with open(model_file,'a') as fo:
            fo.write(str(self.summ))

        

    def read(self,model_file=model_file):
        with open(model_file,'r') as fo:
            content=fo.readlines()

        for i in range(len(content)):
            content[i]=float(content[i])
        
        for i in range(100):
            for j in range(3):
                self.model_1[i][j]=content[i*3+j]

        for i in range(100):
            for j in range(3):
                self.model_2[i][j]=content[i*3+j+300]

        for i in range(100):
            for j in range(3):
                self.model_3[i][j]=content[i*3+j+300*2]

        for i in range(100):
            for j in range(3):
                self.model_4[i][j]=content[i*3+j+300*3]

        for i in range(4):
            for j in range(3):
                self.model_11[i][j]=content[i*3+j+300*4]

        current=1212
        for i in self.filter0:
            i.read(content[current:current+25])
            current+=25

        for i in self.filter1:
            i.read(content[current:current+9])
            current+=9

        for i in self.filter2:
            i.read(content[current:current+9])
            current+=9

        for i in self.filter3:
            i.read(content[current:current+9])
            current+=9

def cmp(a,b):
    return a[1]>b[1]

models=[]

if os.path.exists(model_file):    
    a=model()
    models.append(a)
    a.read()

    photos=[]
    cnt=0

    print("direction determination")
    print(datetime.datetime.now())
    
    for i in yes_files:
        photos.append([])
        photos[cnt].append(i)
        
        image=cv2.imread(i)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image=cv2.resize(image,(image_chang,image_kuan))
        image=image.astype(float)

        image=image.tolist()
        image=[[image]]

        image=torch.tensor(image).to(torch.device('cuda'))
        
        cache=a.apply(image)
        photos[cnt].append(1-cache)
        photos[cnt].append(1)
        a.summ+=1-cache

        cnt+=1

    for i in no_files:
        photos.append([])
        photos[cnt].append(i)
        
        image=cv2.imread(i)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        image=cv2.resize(image,(image_chang,image_kuan))
        image=image.astype(float)

        image=image.tolist()
        image=[[image]]

        image=torch.tensor(image).to(torch.device('cuda'))

        cache=a.apply(image)
        photos[cnt].append(cache)
        photos[cnt].append(0)
        a.summ+=cache

        cnt+=1

    print(a.summ)
    quicksort.quicksort(photos,0,len(photos)-1,cmp)

    for i1 in range(0):
        print(i1)
        image=cv2.imread(photos[i1][0])

        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
        image=cv2.resize(image,(image_chang,image_kuan))
        image=image.astype(float)

        image=image.tolist()
        image=[[image]]

        image=torch.tensor(image).to(torch.device('cuda'))
            
        cache=a.apply(image)

        print(cache)

        if photos[i1][2]==1:
            cache=1-cache

        print(cache)

        print("trying each variable to determine direction")
        print(datetime.datetime.now())

        for i in range(len(a.filter0)):
            for j in range(len(a.filter0[i].data[0][0])):
                for k in range(len(a.filter0[i].data[0][0][j])):
                    cache2=a.filter0[i].data[0][0][j][k].item()
                    a.filter0[i].data[0][0][j][k]+=0.01
                    cache1=a.apply(image)

                    if photos[i1][2]==1:
                        cache1=1-cache1

                    a.delta_filter0[i][j][k]+=cache-cache1
                    a.filter0[i].data[0][0][j][k]=cache2

        for i in range(len(a.filter1)):
            for j in range(len(a.filter1[i].data[0][0])):
                for k in range(len(a.filter1[i].data[0][0][j])):
                    cache2=a.filter1[i].data[0][0][j][k].item()
                    a.filter1[i].data[0][0][j][k]+=0.01
                    cache1=a.apply(image)

                    if photos[i1][2]==1:
                        cache1=1-cache1

                    a.delta_filter1[i][j][k]+=cache-cache1
                    a.filter1[i].data[0][0][j][k]=cache2

        for i in range(len(a.filter2)):
            for j in range(len(a.filter2[i].data[0][0])):
                for k in range(len(a.filter2[i].data[0][0][j])):
                    cache2=a.filter2[i].data[0][0][j][k].item()
                    a.filter2[i].data[0][0][j][k]+=0.01
                    cache1=a.apply(image)

                    if photos[i1][2]==1:
                        cache1=1-cache1

                    a.delta_filter2[i][j][k]+=cache-cache1
                    a.filter2[i].data[0][0][j][k]=cache2


        for i in range(len(a.filter3)):
            for j in range(len(a.filter3[i].data[0][0])):
                for k in range(len(a.filter3[i].data[0][0][j])):
                    cache2=a.filter3[i].data[0][0][j][k].item()
                    a.filter3[i].data[0][0][j][k]+=0.01
                    cache1=a.apply(image)

                    if photos[i1][2]==1:
                        cache1=1-cache1

                    a.delta_filter3[i][j][k]+=cache-cache1
                    a.filter3[i].data[0][0][j][k]=cache2


        for i in range(len(a.model_1)):
            for j in range(len(a.model_1[i])):
                cache2=a.model_1[i][j].item()
                a.model_1[i][j]+=0.01
                cache1=a.apply(image)

                if photos[i1][2]==1:
                    cache1=1-cache1

                a.delta_model_1[i][j]+=cache-cache1
                a.model_1[i][j]=cache2

                cache2=a.model_2[i][j].item()
                a.model_2[i][j]+=0.01
                cache1=a.apply(image)

                if photos[i1][2]==1:
                    cache1=1-cache1

                a.delta_model_2[i][j]+=cache-cache1
                a.model_2[i][j]=cache2

                cache2=a.model_3[i][j].item()
                a.model_3[i][j]+=0.01
                cache1=a.apply(image)

                if photos[i1][2]==1:
                    cache1=1-cache1

                a.delta_model_3[i][j]+=cache-cache1
                a.model_3[i][j]=cache2

                cache2=a.model_4[i][j].item()
                a.model_4[i][j]+=0.01
                cache1=a.apply(image)
                
                if photos[i1][2]==1:
                    cache1=1-cache1

                a.delta_model_4[i][j]+=cache-cache1
                a.model_4[i][j]=cache2

        for i in range(len(a.model_11)):
            for j in range(len(a.model_11[i])):
                cache2=a.model_11[i][j].item()
                a.model_11[i][j]+=0.01
                cache1=a.apply(image)

                if photos[i1][2]==1:
                    cache1=1-cache1

                a.delta_model_11[i][j]+=cache-cache1
                a.model_11[i][j]=cache2
        print()

    for i in range(num_models-1):
        a=model()
        if i>4 and random.randint(0,1)==0:
            a.clone(models[-1],False)
        else:
            a.clone(models[-1])
        a.change()
        models.append(a)

    '''for i in models:
        for j in i.model_1:
            print(j)
        print()'''
else:
    for i in range(num_models):
        a=model()
        models.append(a)

print("training started")
print(datetime.datetime.now())

models[0].summ=0

for i in yes_files:
    image=cv2.imread(i)

    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    image=cv2.resize(image,(image_chang,image_kuan))
    image=image.astype(float)

    image=image.tolist()
    image=[[image]]

    image=torch.tensor(image).to(torch.device('cuda'))
    
    for a in models:
        cache=a.apply(image)
        #if a==models[0]:
            #print(cache)
        #if cache>0.5:
            #a.summ-=1
        a.summ+=1-cache

print()
print()
print(datetime.datetime.now())

for i in no_files:
    image=cv2.imread(i)

    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    image=cv2.resize(image,(image_chang,image_kuan))
    image=image.astype(float)

    image=image.tolist()
    image=[[image]]

    image=torch.tensor(image).to(torch.device('cuda'))

    for a in models:
        cache=a.apply(image)
        #if a==models[0]:
            #print(cache)
        #if cache<0.5:
            #a.summ-=1
        a.summ+=cache

minn=10000000000

for i in models:
    print(i.summ)
    if i.summ<minn:
        minn=i.summ
        minmark=i

minmark.write()
print(minmark.summ)
os.startfile("卷积.py")
