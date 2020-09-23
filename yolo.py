import tensorflow as tf
import tensorflow.keras.backend as K
import os,random,json,re,time
from PIL import Image
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def Iou(a,b):#width,height
  if len(a.shape)==1:
    c=tf.where(a>b,b,a)
    d=c[:,0]*c[:,1]
    similarities=d/(b[:,0]*b[:,1]+a[0]*a[1]-d)
  elif len(a.shape)==2:
    similarities=[]
    for i in a:
      c=tf.where(i>b,b,i)
      d=c[:,0]*c[:,1]
      similarities.append(d/(b[:,0]*b[:,1]+i[0]*i[1]-d))
    # print(similarities,'\n'*3)
    similarities=tf.stack(similarities,axis=1)
  else:
    print('IOU error')
  return similarities

def Iou2(a,b):#x1,y1,x2,y2
  if len(a.shape)==1:
    max=tf.where(a[:2]>b[:,:2],a[:2],b[:,:2])
    min=tf.where(a[2:]<b[:,2:],a[2:],b[:,2:])
    m=min-max
    s=m[:,0]*m[:,1]
    s1=(a[2]-a[0])*(a[3]-a[1])
    s2=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    iou=s/(s1+s2-s)
    # print(s1)
    # print(s[:5])
    # print(s2[:5])
    # print(iou[:5])
    iou=tf.where((max[:,0]<min[:,0]) & (max[:,1]<min[:,1]),iou,0)
    # print(iou[:10])
    return iou
  elif len(a.shape)==2:
    print('Iou2 error')
  else:
    print('Iou2 error')
def NMS(B,S,iouthreshold):
  l=[[],[]]
  
  while S.shape[0]>0:
    max_indices=tf.math.argmax(S[:,0])
    print(S.shape[0])
    print(S[:10,0])
    l[0].append(B[max_indices])
    l[1].append(S[max_indices])
    iou=Iou2(B[max_indices],B)
    print(iou.shape,iou[:10],int(max_indices),iou[max_indices])
    print(tf.where(iou<iouthreshold).shape)
    B=tf.gather_nd(B,tf.where(iou<iouthreshold))
    S=tf.gather_nd(S,tf.where(iou<iouthreshold))
  return l

class yolov2(keras.callbacks.Callback,tf.keras.utils.Sequence):
    def __init__(self):
        super().__init__()
        self.train_filepath=r'./drive/Colab Notebooks/fgo_picture/fgo-support'
        self.seg='/'
        self.validation_filepath=r'./drive/Colab Notebooks/fgo_picture/fgo-support-validation'
        self.model_filepath=r'./drive/Colab Notebooks/model'
        self.base_modelname='fgo-support'
        #self.custom_metrics=[]
        self.prev_modelname=None
        self.model_count=0
        self.train_filelist=[]
        self.validation_filelist=[]
        self.time0=time.time()
        self.time1=time.time()
        self.label_list=[]
        for i in os.listdir(self.train_filepath):
            if '.jpg' in i:
                j=self.train_filepath+self.seg+i
                k=j.replace('jpg','json')
                if os.path.exists(k):
                    self.train_filelist.append([j,k])
        self.train_data_num=len(self.train_filelist)
        for i in os.listdir(self.validation_filepath):
            if '.jpg' in i:
                j=self.validation_filepath+self.seg+i
                k=j.replace('jpg','json')
                if os.path.exists(k):
                    self.validation_filelist.append([j,k])
        self.validation_data_num=len(self.validation_filelist)
        for i in os.listdir(self.model_filepath):
            a=re.search(self.base_modelname+'(\d+)\.h5',i)
            if a!=None:
                print(i)
                b=a.group(1)
                if self.model_count<int(b):
                  self.model_count=int(b)
                  self.prev_modelname=self.model_filepath+self.seg+self.base_modelname+b
        self.model_count+=1
        self.current_model=self.model_filepath+self.seg+self.base_modelname+str(self.model_count)
        self.config_detail=[]
        # self.train_filenum=len(self.filelist)
        self.batch_size=32
        self.stride=16
        self.o_width,self.o_height=Image.open(self.train_filelist[0][0]).size
        self.hs=int(self.o_height/self.stride)
        self.ws=int(self.o_width/self.stride)
        self.width=self.ws*self.stride
        self.height=self.hs*self.stride
        self.shape=[self.width,self.height]
        self.average_gt_per_pic=4500/1000
        self.abn=5
        self.data_generate_count=0
        self.default_anchor_size=tf.convert_to_tensor([[75.26903043,55.88105597],
                                           [76.47330906,23.21770062],
                                           [39.75233832,63.9556357 ],
                                           [19.54460527,35.62049355],
                                           [82.59166936,64.44740316]])#xy
        self.default_anchor_size=self.default_anchor_size/np.array([self.o_width,self.o_height])
        self.true_num=0
    # def custom_metrics(target_y,predicted_y):
    #     pass
    # def custom_callback(self):


    def make_model(self,succeed=True,yanyong=False):
        if (not succeed) or (self.prev_modelname==None) or (not os.path.exists(self.prev_modelname+'.json')):
            # self.model= keras.Sequential([
            #     Conv2D(20,3,padding='same',activation='relu'),
            #     BatchNormalization(training=False),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     Conv2D(40,3,padding='same',activation='relu'),
            #     BatchNormalization(training=False),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     Conv2D(80,3,padding='same',activation='relu'),
            #     BatchNormalization(),
            #     Conv2D(40,1,padding='same',activation='relu'),
            #     BatchNormalization(),
            #     Conv2D(80,3,padding='same',activation='relu'),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     Conv2D(160,3,padding='same',activation='relu'),
            #     BatchNormalization(),
            #     Conv2D(80,1,padding='same',activation='relu'),
            #     BatchNormalization(),
            #     Conv2D(160,3,padding='same',activation='relu'),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     Conv2D((1+4+self.label_num)*self.abn,1,padding='same')
            # ])
            self.model= keras.Sequential([
                Conv2D(20,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),input_shape=(self.height,self.width,3)),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D(40,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D(80,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                Conv2D(40,1,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                Conv2D(80,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D(160,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                Conv2D(80,1,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                Conv2D(160,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D((1+4+self.label_num)*self.abn,1,padding='same')
            ])
            self.config_detail=[['begin_epoch',0],['training_epoch',0]]
        else:
          pass
          # self.config_detail=[['begin_epoch',0],['training_epoch',0]]
          # if self.prev_modelname!=None:
          #   if os.path.exists(self.prev_modelname+'.json'):
          #     with open(self.prev_modelname+'.json','r') as f:
          #       a==json.load(f)[0]
          #       self.config_detail=[['begin_epoch',a[0][1]+a[1][1]],['training_epoch',0]]
          # self.model=keras.models.load_model(self.prev_modelname+'.h5',custom_objects={'custom_loss': self.custom_loss})
          # if yanyong:
          #   self.current_model=self.prev_modelname
          # self.config_detail=[['begin_epoch',0],['training_epoch',0]]
          self.model=keras.models.load_model(self.prev_modelname+'.h5',custom_objects={'custom_loss': self.custom_loss})
          with open(self.prev_modelname+'.json','r') as f:
            a=json.load(f)
          if yanyong:
            self.current_model=self.prev_modelname
            self.config_detail=a
          else:
            self.config_detail=[['begin_epoch',a[0][1]+a[1][1]],['training_epoch',0]]

    def custom_loss(self,target_y,predicted_y):
        target_y=tf.reshape(target_y,[-1,1+4+self.label_num])
        predicted_y=tf.reshape(predicted_y,(-1,1+4+self.label_num))
        # print(1)
        true_idx=tf.cast(tf.where(target_y[...,0:1]>0.5,1,0),dtype=tf.float32)
        false_idx=tf.cast(tf.where(target_y[...,0:1]<0.5,1,0),dtype=tf.float32)
        a=tf.reduce_sum(true_idx)+0.01
        b=tf.reduce_sum(false_idx)+0.01
        # c=tf.reduce_sum(target_y[...,0:1])
        # d=tf.reduce_sum(predicted_y[...,0:1])
        # print(2)
        # print(target_y.shape)
        # tf.print(tf.shape(false_idx))
        loss1=tf.reduce_sum(tf.math.log(1+tf.exp(-predicted_y[:,0:1]))*true_idx)/a
        loss2=tf.reduce_sum(tf.math.log(1+tf.exp(predicted_y[:,0:1]))*false_idx)/b
        loss3=tf.reduce_sum(tf.square(target_y[:,1:3]-tf.sigmoid(predicted_y[:,1:3]))*true_idx)/a
        loss4=tf.reduce_sum(tf.square(target_y[:,3:5]-tf.exp(predicted_y[:,3:5]))*true_idx)/a
        # loss5=tf.reduce_sum(tf.square(target_y[...,5:]-predicted_y[...,5:])*true_idx)/a
        loss5=tf.reduce_sum(tf.reduce_mean((target_y[...,5:]*tf.math.log(1+tf.exp(-predicted_y[...,5:]))+
                                            (1-target_y[...,5:])*tf.math.log(1+tf.exp(predicted_y[...,5:])))*true_idx,axis=1))/a
        # print(3)
        tf.print(loss1,' ',loss2,' ',loss3,' ',loss4,' ',loss5)
        #self.custom_metrics[-1].append([loss1,loss2,loss3,loss4,loss5])
        # print(4)
        return loss1+loss2+loss3+loss4+loss5
    
    def decode_box(self,txtytwth):
        xywh=tf.zeros(tf.shape(txtytwth))
        x1y1x2y2=tf.zeros(tf.shape(txtytwth))
        gridx=tf.tile(tf.reshape(tf.range(self.ws),[1,self.ws,1,1,1]),[1,1,self.hs,1,1])
        gridy=tf.tile(tf.reshape(tf.range(self.hs),[1,1,self.hs,1,1]),[1,self.ws,1,1,1])
        grid_xy=tf.concat((gridx,gridy),axis=-1)
        xywh[...,0:2]=tf.sigmoid(txtytwth[...,0:2])+grid_xy
        xywh[...,2:4]=tf.exp(txtytwth[...,2:4])*tf.reshape(self.default_anchor_size,[1,1,1,self.abn,2])
        x1y1x2y2[...,0]=xywh[...,0]-xywh[...,2]/2
        x1y1x2y2[...,2]=xywh[...,0]+xywh[...,2]/2
        x1y1x2y2[...,1]=xywh[...,1]-xywh[...,3]/2
        x1y1x2y2[...,3]=xywh[...,1]+xywh[...,3]/2
        return x1y1x2y2
    def __len__(self):
        return 6
    
    def generate_dataset(self):
      self.train_dataset_y_list=[]
      self.validation_dataset_x=tf.zeros((self.validation_data_num,self.height,self.width,3))
      self.validation_dataset_y=tf.zeros((self.validation_data_num,self.hs,self.ws,self.abn,1+4+1))
      self.train_dataset_x=tf.zeros((self.train_data_num,self.height,self.width,3))
      #self.train_dataset_y=tf.zeros((self.train_data_num,self.hs,self.ws,self.abn,1+4+self.label_num))
      self.train_dataset_y=[]
      self.validation_dataset_y_list=[]
      
      cc=[]
      ll=[]
      for j,i in enumerate(self.validation_filelist):
        pic1=Image.open(i[0])
        width,height=pic1.size
        self.validation_dataset_x=tf.tensor_scatter_nd_update(self.validation_dataset_x,[[j]],tf.expand_dims(tf.convert_to_tensor(np.array(pic1.resize((self.width,self.height),Image.BILINEAR)),dtype=tf.float32),axis=0)/255.0)
        with open(i[1],'r') as f:
          data=json.load(f)
        cc.append(len(data))
        for k in data:
          if k[0] not in self.label_list:
            self.label_list.append(k[0])
          x=(k[1][2]+k[1][0])/2/width*self.ws
          # print(k[1][3],k[1][1],height,self.hs)
          y=(k[1][3]+k[1][1])/2/height*self.hs
          w=(k[1][2]-k[1][0])/width
          h=(k[1][3]-k[1][1])/height
          similarities=Iou(np.array([w,h]),self.default_anchor_size)
          max_indice=tf.math.argmax(similarities)
          ab_wh=self.default_anchor_size[max_indice]
          # print(tf.convert_to_tensor([[j,int(y),int(x),max_indice]]))
          # print(tf.convert_to_tensor([[1,x-int(x),y-int(y),w/ab_wh[0],h/ab_wh[1],self.label_list.index(k[0])+1]]))
          self.validation_dataset_y=tf.tensor_scatter_nd_update(self.validation_dataset_y,tf.convert_to_tensor([[j,int(y),int(x),max_indice]]),tf.convert_to_tensor([[1,x-int(x),y-int(y),w/ab_wh[0],h/ab_wh[1],self.label_list.index(k[0])+1]]))
          # print(tf.gather_nd(self.validation_dataset_y,[[j,int(y),int(x),max_indice]]))
          #l.append([k[1][0]/width*self.ws,k[1][1]/height*self.hs,k[1][2]/width*self.ws,k[1][3]/height*self.hs,self])
          ll.append([k[1][0]/width,k[1][1]/height,k[1][2]/width,k[1][3]/height,self.label_list.index(k[0])+1])
          #
      c=[]
      l=[]
      for j,i in enumerate(self.train_filelist):
            pic1=Image.open(i[0])
            width,height=pic1.size
            self.train_dataset_x=tf.tensor_scatter_nd_update(self.train_dataset_x,[[j]],tf.expand_dims(tf.convert_to_tensor(np.array(pic1.resize((self.width,self.height),Image.BILINEAR)),dtype=tf.float32),axis=0)/255.0)
            with open(i[1],'r') as f:
              data=json.load(f)
            c.append(len(data))
            for k in data:
              if k[0] not in self.label_list:
                self.label_list.append(k[0])
              l.append([k[1][0]/width,k[1][1]/height,k[1][2]/width,k[1][3]/height,self.label_list.index(k[0])+1])
              #x1,y1,x2,y2 x1<x2,y1<y2
      self.label_num=len(self.label_list)
      self.validation_dataset_y=tf.concat([self.validation_dataset_y[:,:,:,:,:-1],
                                           tf.one_hot(indices=tf.cast(self.validation_dataset_y[:,:,:,:,-1]-1,dtype=tf.int32),depth=self.label_num)],
                                          axis=-1)
      l=np.array(l)
      l=tf.concat([l[:,:-1],
                    tf.one_hot(indices=l[:,-1]-1,depth=self.label_num)],
                  axis=-1)
      count=0
      countcount=0
      ll=np.array(ll)
      ll=tf.concat([ll[:,:-1],
                    tf.one_hot(indices=ll[:,-1]-1,depth=self.label_num)],
                  axis=-1)
      for i in c:
        self.train_dataset_y_list.append(l[count:count+i])
        count+=i
      for i in cc:
        self.validation_dataset_y_list.append(ll[countcount:countcount+i])
        countcount+=i
      # print(self.train_dataset_y_list)
      # for i in self.validation_dataset_y_list:
      #   print(i.shape[0])
      # print(c,cc)
      num=self.train_dataset_x.shape[0]
      self.train_dataset_y=tf.zeros((num,self.hs,self.ws,self.abn,1+4+self.label_num))
      for i in range(num):
        ttt=self.train_dataset_y_list[i]
        # print(ttt.shape[0])
        central=tf.stack([(ttt[:,1]+ttt[:,3])/2*self.hs,(ttt[:,0]+ttt[:,2])/2*self.ws],axis=1)#yx
        t_shape=tf.stack((ttt[:,2]-ttt[:,0],ttt[:,3]-ttt[:,1]),axis=1)#裁剪后的shape,xy
        floor=tf.floor(central)#yx
        t_num=ttt.get_shape()[0]
        similarities=Iou(t_shape,self.default_anchor_size)
        max_indice=tf.math.argmax(similarities,axis=0)
        ab_wh=tf.gather(self.default_anchor_size,max_indice)#width,height
        a=tf.fill([t_num,1],i)
        b=tf.reshape(max_indice,[-1,1])
        indices=tf.concat((a,tf.cast(floor,tf.int32),tf.cast(b,tf.int32)),axis=1)
        c=tf.ones([t_num,1],dtype=tf.float32)
        d=tf.cast(central-floor,tf.float32)
        e=tf.cast(t_shape,tf.float32)/tf.cast(ab_wh,tf.float32)#xy
        f=tf.cast(ttt[:,4:],tf.float32)
        updates=tf.concat((c,d,e[:,1:2],e[:,0:1],f),axis=1)
        # print(tf.gather_nd(self.batch_y,indices))
        self.train_dataset_y=tf.tensor_scatter_nd_update(self.train_dataset_y,indices,updates)


    def __getitem__(self,idx=None):
        self.time0=time.time()
        print('\ntraining time:',self.time0-self.time1)
        if self.data_generate_count==0:
          self.data_generate_count=0
          self.true_num=0
          box_indices = tf.random.uniform(shape=(self.batch_size,), maxval=self.train_data_num, dtype=tf.int32)
          # boxes = tf.random.uniform(shape=(self.batch_size, 4),maxval=0.1)
          # boxes[:,2:4]=tf.random.uniform(shape=(self.batch_size,2),minval=0.5,maxval=1)
          # boxes=tf.concat([tf.random.uniform(shape=(self.batch_size, 2),maxval=0.1),tf.random.uniform(shape=(self.batch_size, 2),maxval=0.1)],axis=1)
          #boxes:y1,x1,y2,x2  y1<y2,x1<x2
          #train_dataset_y_list:x1,y1,x2,y2 x1<x2,y1<y2
          boxes=tf.concat((tf.random.uniform(shape=(self.batch_size,2),maxval=0.2),tf.random.uniform(shape=(self.batch_size,2),maxval=1,minval=0.5)),axis=1)
          #tf.image.crop_and_resize(image, boxes, box_indices, crop_size, method=‘bilinear’, extrapolation_value=0,name=None )
          self.batch_x=tf.image.crop_and_resize(self.train_dataset_x,boxes=boxes,box_indices=box_indices,crop_size=(self.height,self.width))
          self.batch_y=tf.zeros((self.batch_size,self.hs,self.ws,self.abn,1+4+self.label_num))
          for i,j in enumerate(box_indices):
            t=self.train_dataset_y_list[j]
            a=tf.where(((t[:,0]+t[:,2])/2 > boxes[i,1]) & ((t[:,0]+t[:,2])/2 < boxes[i,3]) & ((t[:,1]+t[:,3])/2> boxes[i,0]) & ((t[:,1]+t[:,3])/2 < boxes[i,2]))
            # print('a:',a)
            tt=tf.gather(t,tf.squeeze(a,axis=1))
            # t[tf.where(((t[0]+t[2])/2 > boxes[1]) & ((t[0]+t[2])/2 < boxes[3]) & ((t[1]+t[3])/2> boxes[0]) & ((t[1]+t[3])/2 < boxes[2]))]
            # print(t.shape,tt.shape)
            if tt.shape[0]==0:
              continue
            width=boxes[i,3]-boxes[i,1]
            height=boxes[i,2]-boxes[i,0]
            ttt=(tt[:,:4]-tf.convert_to_tensor([[boxes[i,1],boxes[i,0],boxes[i,1],boxes[i,0]]]))/tf.convert_to_tensor([[width,height,width,height]])
            ttt=tf.where(ttt>0,ttt,0)
            central=tf.stack([(ttt[:,1]+ttt[:,3])/2*self.hs,(ttt[:,0]+ttt[:,2])/2*self.ws],axis=1)#yx
            t_shape=tf.stack((ttt[:,2]-ttt[:,0],ttt[:,3]-ttt[:,1]),axis=1)#裁剪后的shape,xy
            floor=tf.floor(central)#yx
            t_num=tt.get_shape()[0]
            similarities=Iou(t_shape,self.default_anchor_size)
            max_indice=np.argmax(similarities,axis=0)
            ab_wh=tf.gather(self.default_anchor_size,max_indice)
            a=tf.fill([t_num,1],i)
            b=tf.reshape(max_indice,[-1,1])
            self.true_num+=t_num
            indices=tf.concat((a,tf.cast(floor,tf.int32),tf.cast(b,tf.int32)),axis=1)
            c=tf.ones([t_num,1],dtype=tf.float32)
            d=tf.cast(central-floor,tf.float32)
            e=tf.cast(t_shape,tf.float32)/tf.cast(ab_wh,tf.float32)#xy
            f=tf.cast(tt[:,4:],tf.float32)

            updates=tf.concat((c,d,e[:,1:2],e[:,0:1],f),axis=1)
            # print(tf.gather_nd(self.batch_y,indices))
            self.batch_y=tf.tensor_scatter_nd_update(self.batch_y,indices,updates)
            # print(tf.gather_nd(self.batch_y,indices))
        else:
          self.data_generate_count-=1
        self.time1=time.time()
        print('data generation time:',self.time1-self.time0)
        
        return self.batch_x,tf.reshape(self.batch_y,(self.batch_size,self.hs,self.ws,self.abn*(1+4+self.label_num)))

    def on_train_begin(self,logs=None):
      self.epoch_count=0
      self.loss=[]
      print('on_train_begin')
    def on_epoch_begin(self,epoch,logs=None):
      self.epoch_count+=1
      self.loss.append([])

    def on_epoch_end(self,epoch=None,logs=None):
      # print(self.loss)
      # print(logs)
      if logs!=None:
        self.loss[-1]=[logs['loss'],logs['val_loss']]
      if self.epoch_count%20==0:
        self.config_detail[1][1]+=20
        a=self.config_detail[1][1]
        print(self.config_detail[1][1])
        self.model.save(self.current_model+'.h5')
        self.config_detail.append(['train_epoch '+str(a),self.loss[-1]])
        with open(self.current_model+'.json','w') as f:
            json.dump(self.config_detail,f,ensure_ascii=False)

    def begin_train(self,augmentation=True,succeed=False,yanyong=False):
        self.make_model(succeed=succeed,yanyong=yanyong)
        self.model.compile(optimizer='adam',
                           loss=self.custom_loss)
              #loss=self.custom_loss)
        #validation_data=tuple(self.__getitem__())
        # self.history=self.model.fit(self,epochs=1000,validation_data=(self.validation_dataset_x,self.validation_dataset_y),callbacks=[self])
        if augmentation:
          self.history=self.model.fit(self,initial_epoch=self.config_detail[0][1],epochs=10000,verbose=0,
                                      validation_data=(self.validation_dataset_x,self.validation_dataset_y),callbacks=[self])
        else:
          self.history=self.model.fit(self.train_dataset_x,self.train_dataset_y,initial_epoch=self.config_detail[0][1],epochs=10000,verbose=0,
                                      validation_data=(self.validation_dataset_x,self.validation_dataset_y),callbacks=[self])
        # print(self.hsitory)
    
    def load_model(self,model_name=None):
      if model_name==None:
        if self.prev_modelname==None:
          print('error')
        elif not os.path.exists(self.prev_modelname+'.h5'):
          print('error')
        else:
          self.model=keras.models.load_model(self.prev_modelname+'.h5',custom_objects={'custom_loss': self.custom_loss})
      elif not os.path.exists(model_name):
        print('error')
      else:
        self.model=keras.models.load_model(model_name,custom_objects={'custom_loss': self.custom_loss})

    def return_dataset(self,dataset_type='train',augmentation=False,batch_size=1,which=0):
      if dataset_type=='train':
        dataset_x=self.train_dataset_x
        dataset_y=self.train_dataset_y_list
      elif dataset_type=='validation':
        datasety=self.validation_dataset_y
        dataset_x=self.validation_dataset_x
      else:
        print('dataset_type error')
      # if which==None:
      #   which=tf.random.uniform(shape=(batch_size,), maxval=dataset_x.shape[0], dtype=tf.int32)
      return dataset_x[which],dataset_y[which]

    def predict_model(self,input):
      return self.model(input)
    
    def  post_process(self,output,scorethreshold,iouthreshold):#one picture
      # if len(output.shape)==4:
      #   batch_num=1
      # elif len(output.shape)==5:
      #   batch_num==output.shape[0]
      output=tf.reshape(output,(1,self.hs,self.ws,self.abn,1+4+self.label_num))
      print(output.shape)
      indices=tf.where(output[...,0]>scorethreshold)
      print(indices.shape)
      # a=tf.tile(tf.reshape(tf.range(self.abn),(1,1,self.abn,1)),tf.constant([self.hs,self.ws,1,1]))
      # b=tf.tile(tf.reshape(tf.range(self.hs),(self.hs,1,1,1)),tf.constant([1,self.ws,self.abn,1]))
      # c=tf.tile(tf.reshape(tf.range(self.ws),(1,self.ws,1,1)),tf.constant([self.hs,1,self.abn,1]))
      # output=tf.concat([output,a,b,c],axis=3)

      output=tf.gather_nd(output,indices)
      S=tf.concat([output[:,0:1],output[:,5:]],axis=1)
      anchor=tf.gather_nd(self.default_anchor_size,indices[:,3:4])
      print(indices[:5])
      print(anchor[:5])
      # anchor=tf.cast(anchor,dtype=tf.float32)
      # print(anchor[:5])
      width=tf.exp(output[:,4])*anchor[:,0]
      height=tf.exp(output[:,3])*anchor[:,1]
      central_x=(tf.sigmoid(output[:,2])+tf.cast(indices[:,2],dtype=tf.float32))/self.ws
      central_y=(tf.sigmoid(output[:,1])+tf.cast(indices[:,1],dtype=tf.float32))/self.hs
      B=tf.stack([central_x-width/2,central_y-height/2,central_x+width/2,central_y+height/2],axis=1)
      print(output[:5,4],output[:5,3],width[:5],height[:5])
      print(anchor[:5])
      print(self.default_anchor_size)
      l=NMS(B,S,iouthreshold)
      return l

    def evaluate_x(self):
      num=self.train_dataset_x.shape[0]
      train_dataset_y=tf.zeros((num,self.hs,self.ws,self.abn,1+4+self.label_num))
      for i in range(num):
        ttt=self.train_dataset_y_list[i]
        # print(ttt.shape[0])
        central=tf.stack([(ttt[:,1]+ttt[:,3])/2*self.hs,(ttt[:,0]+ttt[:,2])/2*self.ws],axis=1)#yx
        t_shape=tf.stack((ttt[:,2]-ttt[:,0],ttt[:,3]-ttt[:,1]),axis=1)#裁剪后的shape,xy
        floor=tf.floor(central)#yx
        t_num=ttt.get_shape()[0]
        similarities=Iou(t_shape,self.default_anchor_size)
        max_indice=tf.math.argmax(similarities,axis=0)
        ab_wh=tf.gather(self.default_anchor_size,max_indice)
        a=tf.fill([t_num,1],i)
        b=tf.reshape(max_indice,[-1,1])
        indices=tf.concat((a,tf.cast(floor,tf.int32),tf.cast(b,tf.int32)),axis=1)
        c=tf.ones([t_num,1],dtype=tf.float32)
        d=tf.cast(central-floor,tf.float32)
        e=tf.cast(t_shape,tf.float32)/tf.cast(ab_wh,tf.float32)#xy
        f=tf.cast(ttt[:,4:],tf.float32)
        updates=tf.concat((c,d,e[:,1:2],e[:,0:1],f),axis=1)
        # print(tf.gather_nd(self.batch_y,indices))
        train_dataset_y=tf.tensor_scatter_nd_update(train_dataset_y,indices,updates)
      numnum=self.validation_dataset_x.shape[0]
      validation_dataset_y=tf.zeros((numnum,self.hs,self.ws,self.abn,1+4+self.label_num))
      for i in range(numnum):
        ttt=self.validation_dataset_y_list[i]
        central=tf.stack([(ttt[:,1]+ttt[:,3])/2*self.hs,(ttt[:,0]+ttt[:,2])/2*self.ws],axis=1)#yx
        t_shape=tf.stack((ttt[:,2]-ttt[:,0],ttt[:,3]-ttt[:,1]),axis=1)#裁剪后的shape,xy
        floor=tf.floor(central)#yx
        t_num=ttt.get_shape()[0]
        # print(t_num,i)
        similarities=Iou(t_shape,self.default_anchor_size)
        max_indice=np.argmax(similarities,axis=0)
        ab_wh=tf.gather(self.default_anchor_size,max_indice)
        a=tf.fill([t_num,1],i)
        b=tf.reshape(max_indice,[-1,1])
        indices=tf.concat((a,tf.cast(floor,tf.int32),tf.cast(b,tf.int32)),axis=1)
        c=tf.ones([t_num,1],dtype=tf.float32)
        d=tf.cast(central-floor,tf.float32)
        e=tf.cast(t_shape,tf.float32)/tf.cast(ab_wh,tf.float32)#xy
        f=tf.cast(ttt[:,4:],tf.float32)
        updates=tf.concat((c,d,e[:,1:2],e[:,0:1],f),axis=1)
        # print(tf.gather_nd(self.batch_y,indices))
        # print(tf.gather_nd(validation_dataset_y,indices))
        validation_dataset_y=tf.tensor_scatter_nd_update(validation_dataset_y,indices,updates)
        # print(tf.gather_nd(validation_dataset_y,indices))
      print(self.model.evaluate(self.train_dataset_x,train_dataset_y))
      print(self.model.evaluate(self.validation_dataset_x,self.validation_dataset_y))
      print(self.model.evaluate(self.validation_dataset_x,validation_dataset_y))
      return self.validation_dataset_y,validation_dataset_y,train_dataset_y

# a=yolov2()
# # print(2)
# a.generate_dataset()
# a.begin_train(augmentation=False,succeed=False,yanyong=True)
# a.generate_dataset()
# a.load_model(r'./drive/Colab Notebooks/model/fgo-support21.h5')
# l=a.post_process(a.predict_model(a.train_dataset_x[0:1]),1,0.5)
# # # x,y=a.return_dataset('train')
# n,m,k=a.evaluate_x()
# print(tf.reduce_sum(m[...,0]),tf.reduce_sum(m[...,-2]),tf.reduce_sum(m[...,-1]),tf.reduce_sum(n[...,0]),tf.reduce_sum(n[...,-2]),tf.reduce_sum(n[...,-1]),tf.reduce_sum(k[...,0]),tf.reduce_sum(k[...,-2]),tf.reduce_sum(k[...,-1]))
# print(tf.stack(l[1],axis=0))
# print(tf.concat(a.train_dataset_y_list[0],axis=0))
# print(tf.stack(l[0],axis=0))
# print(a.train_dataset_y_list[0])
