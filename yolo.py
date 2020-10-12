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
%matplotlib inline
print(1)
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
def kmeans(data,k,method='iou'):
  print('datashape',data.shape)
  central=tf.gather_nd(data,tf.reshape(tf.range(k),(k,1)))
  if method=='iou':
    def distance(central,data):
      t_shape2=tf.tile(tf.reshape(data,(-1,1,2)),tf.constant((1,k,1)))
      t_shape3=tf.reshape(central,(1,-1,2))
      ss=tf.where(t_shape2>t_shape3,t_shape3,t_shape2)
      ddd=ss[:,:,0]*ss[:,:,1]
      iou=ddd/(tf.reshape(central[:,0],(1,-1))*tf.reshape(central[:,1],(1,-1))+t_shape2[:,:,0]*t_shape2[:,:,1]-ddd)
      return 1-iou
  elif method=='Euclidean_distance':
    def distance(central,data):
      t_shape2=tf.tile(tf.reshape(data,(-1,1,2)),tf.constant((1,k,1)))
      distance=tf.square(t_shape2-tf.reshape(central,(1,-1,2)))
      return tf.reduce_sum(distance,axis=2)
  argmin2=tf.zeros(data.shape[0],dtype=tf.int64)
  # argmax1=tf.ones([k],dtype=tf.int32)
  count=0
  while True:
    count+=1
    d=distance(central,data)
    argmin1=tf.math.argmin(d,axis=1)
    print('central',central)
    for i in range(k):
      print(i)
      central=tf.tensor_scatter_nd_update(central,[[i]],tf.reduce_mean(tf.gather_nd(data,tf.where(argmin1==i)),axis=0,keepdims=True))
      tf.debugging.check_numerics(central,'central'+str(i))    
    if tf.math.reduce_sum(tf.math.square(argmin1-argmin2))<1:
      break
    argmin2=argmin1
  print(count)
  return central

def Iou2(a,b):#x1,y1,x2,y2
  if len(a.shape)==1:
    max=tf.where(a[:2]>b[:,:2],a[:2],b[:,:2])
    min=tf.where(a[2:]<b[:,2:],a[2:],b[:,2:])
    m=min-max
    s=m[:,0]*m[:,1]
    s1=(a[2]-a[0])*(a[3]-a[1])
    s2=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    iou=s/(s1+s2-s)
    iou=tf.where((max[:,0]<min[:,0]) & (max[:,1]<min[:,1]),iou,0)
    return iou
  elif len(a.shape)==2:
    print('Iou2 error')
  else:
    print('Iou2 error')
def NMS(B,S,iouthreshold):
  l=[[],[]]
  
  while S.shape[0]>0:
    max_indices=tf.math.argmax(S[:,0])
    # print(S.shape[0])
    # print(S[:10,0])
    l[0].append(B[max_indices])
    l[1].append(S[max_indices])
    iou=Iou2(B[max_indices],B)
    # print(iou.shape,iou[:10],int(max_indices),iou[max_indices])
    # print(tf.where(iou<iouthreshold).shape)
    B=tf.gather_nd(B,tf.where(iou<iouthreshold))
    S=tf.gather_nd(S,tf.where(iou<iouthreshold))
  return l

class yolov2(keras.callbacks.Callback,tf.keras.utils.Sequence):
    def __init__(self):
        super().__init__()
        self.train_filepath=[r'./drive/Colab Notebooks/fgo_picture/fgo-support',
                             r'./drive/Colab Notebooks/fgo_picture/fgo-support-4',
                             r'./drive/Colab Notebooks/fgo_picture/fgo-support-validation',
                             r'./drive/Colab Notebooks/fgo_picture/test-2',
                             r'./drive/Colab Notebooks/fgo_picture/test-3']
        self.seg='/'
        self.validation_filepath=[r'./drive/Colab Notebooks/fgo_picture/test-3',
                                  r'./drive/Colab Notebooks/fgo_picture/test-2']
        self.model_filepath=r'./drive/Colab Notebooks/model'
        self.base_modelname='fgo-support'
        #self.custom_metrics=[]
        self.prev_modelname=None
        self.model_count=0
        # self.train_filelist=[]
        # self.validation_filelist=[]
        self.time0=time.time()
        self.time1=time.time()
        self.label_list=[]
        self.class_list=[]
        self.train_filelist=self.get_folder_filename(self.train_filepath)
        self.train_data_num=len(self.train_filelist)
        self.validation_filelist=self.get_folder_filename(self.validation_filepath)
        self.validation_data_num=len(self.validation_filelist)
        for i in os.listdir(self.model_filepath):
            a=re.search(self.base_modelname+'(\d+)\.h5',i)
            if a!=None:
                # print(i)
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
        self.data_generate_count=0
        # self.default_anchor_size=tf.constant([[0.12669209,0.14717807],
        #                                       [0.20282197,0.05999681],
        #                                       [0.20131494,0.15261464],
        #                                       [0.12768319,0.05639816]])#xy
        self.default_anchor_size=tf.constant([[0.05416219,0.09319306],
                                              [0.1060535 ,0.06356987],
                                              [0.14459075,0.3840488 ],
                                              [0.11091662,0.15959485],
                                              [0.05781969,0.03113795]])
        self.abn=self.default_anchor_size.shape[0]
        # self.default_anchor_size=self.default_anchor_size/np.array([self.o_width,self.o_height])
        self.true_num=0
        self.max_objects_per_image=40
        self.max_attribute_per_box=4
        self.generate_dataset()
    def get_folder_filename(self,folder1):
      filelist=[]
      for folder in folder1:
        for i in os.listdir(folder):
          if '.jpg' in i:
            j=folder+self.seg+i
            k=j.replace('jpg','json')
            if os.path.exists(k):
              filelist.append([j,k])
      return filelist

    def make_model(self,succeed=True,yanyong=False):
        if (not succeed) or (self.prev_modelname==None) or (not os.path.exists(self.prev_modelname+'.json')):
            input=tf.keras.Input((self.height,self.width,3))
            model= keras.Sequential([
                # BatchNormalization(),
                # tf.keras.layers.InputLayer(input_shape=(self.height,self.width,3),batch_size=32),
                Conv2D(20,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D(40,3,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D(80,3,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                Conv2D(40,1,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                Conv2D(80,3,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2),strides=2),
                Conv2D(160,3,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                Conv2D(80,1,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                Conv2D(160,3,padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
                BatchNormalization(scale=False,momentum=0.8),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2),strides=2)
            ])
            model=model(input)
            output1=Conv2D((1+4+self.label_num)*self.abn,1,padding='same',name='detection')(model)
            output2=Conv2D(10,3,padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(model)
            output2=BatchNormalization(scale=False,momentum=0.8)(output2)
            output2=Activation('relu')(output2)
            output=MaxPooling2D(pool_size=(2,2),strides=2)(model)
            output2=Flatten()(output2)
            output2=Dense(units=self.class_num,activation='sigmoid',name='classification')(output2)
            self.model=keras.Model(inputs=[input],outputs=[output1,output2])
            # self.model= keras.Sequential([
            #     BatchNormalization(),
            #     Conv2D(20,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),input_shape=(self.height,self.width,3)),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     BatchNormalization(),
            #     Conv2D(40,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     BatchNormalization(),
            #     Conv2D(80,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     BatchNormalization(),
            #     Conv2D(40,1,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     BatchNormalization(),
            #     Conv2D(80,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     BatchNormalization(),
            #     Conv2D(160,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     BatchNormalization(),
            #     Conv2D(80,1,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     BatchNormalization(),
            #     Conv2D(160,3,padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            #     MaxPooling2D(pool_size=(2, 2),strides=2),
            #     Conv2D((1+4+self.label_num)*self.abn,1,padding='same')
            # ])
            self.config_detail=[['begin_epoch',0],['training_epoch',0]]
        else:
          self.model=keras.models.load_model(self.prev_modelname+'.h5',custom_objects={'custom_loss': self.custom_loss,'custom_loss2':self.custom_loss2})
          with open(self.prev_modelname+'.json','r') as f:
            a=json.load(f)
          if yanyong:
            self.current_model=self.prev_modelname
            self.config_detail=a
          else:
            self.config_detail=[['begin_epoch',a[0][1]+a[1][1]],['training_epoch',0]]
    def custom_loss2(self,target_y,predicted_y):
      # loss=tf.reduce_mean(target_y*(1-predicted_y)+(1-target_y)*predicted_y)
      # target_y=tf.cast(target_y,dtype=tf.float32)

      loss=tf.reduce_mean(-target_y*tf.math.log(predicted_y+0.0001)-(1-target_y)*tf.math.log(1-predicted_y+0.001))
      tf.print('customloss2',loss)
      return loss
    def custom_loss(self,target_y,predicted_y):
        target_y=tf.reshape(target_y,(-1,1+4+self.label_num),name='reshape1')
        predicted_y=tf.reshape(predicted_y,(-1,1+4+self.label_num),name='reshape2')
        # print(1)
        true_idx=tf.cast(tf.where(target_y[...,0:1]>0.5,1,0),dtype=tf.float32)
        false_idx=tf.cast(tf.where(target_y[...,0:1]<0.5,1,0),dtype=tf.float32)
        a=tf.reduce_sum(true_idx)+0.01
        b=tf.reduce_sum(false_idx)+0.01
        loss1=tf.reduce_sum(tf.math.log(1+tf.exp(-predicted_y[:,0:1])+0.0001)*true_idx)/a
        # tf.print(a)
        # tf.print()
        loss2=tf.reduce_sum(tf.math.log(1+tf.exp(predicted_y[:,0:1]))*false_idx)/b
        loss3=tf.reduce_sum(tf.square(target_y[:,1:3]-tf.sigmoid(predicted_y[:,1:3]))*true_idx)/a
        loss4=tf.reduce_sum(tf.square(target_y[:,3:5]-tf.exp(predicted_y[:,3:5]))*true_idx)/a
        # loss5=tf.reduce_sum(tf.square(target_y[...,5:]-predicted_y[...,5:])*true_idx)/a
        loss5=tf.reduce_sum(tf.reduce_mean((target_y[...,5:]*tf.math.log(1+tf.exp(-predicted_y[...,5:]))+
                                            (1-target_y[...,5:])*tf.math.log(1+tf.exp(predicted_y[...,5:])))*true_idx,axis=1))/a
        tf.print('custom_loss:',loss1,' ',loss2,' ',loss3,' ',loss4,' ',loss5)
        loss=loss1+loss2+loss3+loss4+loss5
        return loss
    
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
      self.train_dataset_x,self.train_dataset_y1_aug,self.train_dataset_y2=self.withoutname1(self.train_filelist)
      self.validation_dataset_x,v_y,self.validation_dataset_y2=self.withoutname1(self.validation_filelist)
      self.label_num=len(self.label_list)
      self.class_num=len(self.class_list)

      self.train_dataset_y1_aug,v_y=self.withoutname3([self.train_dataset_y1_aug,v_y])
      self.train_dataset_y2,self.validation_dataset_y2=self.withoutname4([self.train_dataset_y2,self.validation_dataset_y2])
      box_indices=tf.range(self.validation_data_num)
      boxes=tf.tile(tf.constant([[0,0,1,1]],tf.float32),tf.constant([self.validation_data_num,1],tf.int32))
      self.validation_dataset_y1=self.withoutname2(v_y,box_indices,boxes)
      box_indices=tf.range(self.train_data_num)
      boxes=tf.tile(tf.constant([[0,0,1,1]],tf.float32),tf.constant([self.train_data_num,1],tf.int32))
      self.train_dataset_y1=self.withoutname2(self.train_dataset_y1_aug,box_indices,boxes)

    def withoutname1(self,filelist):
      num=len(filelist)
      x=tf.zeros((num,self.height,self.width,3))
      y1=tf.zeros((num,self.max_objects_per_image,1+4+self.max_attribute_per_box))
      y2=tf.zeros((num),dtype=tf.int32)
      for j,i in enumerate(filelist):
          pic1=Image.open(i[0])
          width,height=pic1.size
          x=tf.tensor_scatter_nd_update(x,[[j]],tf.expand_dims(tf.convert_to_tensor(np.array(pic1.resize((self.width,self.height),Image.BILINEAR)),dtype=tf.float32),axis=0)/255.0)
          with open(i[1],'r',encoding='gbk') as f:
            data=json.load(f)
          # c.append(len(data))
          for n,k in enumerate(data):
            if k[0]=='class_name':
              if k[1] not in self.class_list:
                self.class_list.append(k[1])
              y2=tf.tensor_scatter_nd_update(y2,tf.constant([[j]]),tf.constant([self.class_list.index(k[1])+1]))
            else:
              y1=tf.tensor_scatter_nd_update(y1,tf.constant([[j,n]]),
                            tf.constant([[1,k[1][0]/width,k[1][1]/height,k[1][2]/width,
                                          k[1][3]/height]+[0]*self.max_attribute_per_box]))
              for m,l in enumerate(k[0].split('-')):
                if l not in self.label_list:
                  self.label_list.append(l)
                y1=tf.tensor_scatter_nd_update(y1,tf.constant([[j,n,1+4+m]]),tf.constant([self.label_list.index(l)+1],dtype=tf.float32))
          #x1,y1,x2,y2 x1<x2,y1<y2
      return x,y1,y2
    def withoutname2(self,y1,box_indices,boxes):
      y2=tf.zeros((box_indices.shape[0],self.hs,self.ws,self.abn,1+4+self.label_num))
      a=tf.gather_nd(y1,tf.reshape(box_indices,(-1,1),name='reshape3'))
      index=tf.where((a[:,:,0]>0.5))
      b=tf.gather_nd(a[:,:,1:],index)
      c=tf.gather_nd(boxes,index[:,0:1])
      index2=tf.where(((b[:,0]+b[:,2])/2 > c[:,1]) & ((b[:,0]+b[:,2])/2 < c[:,3]) & 
                        ((b[:,1]+b[:,3])/2> c[:,0]) & ((b[:,1]+b[:,3])/2 < c[:,2]))
      index3=tf.gather_nd(index,index2)[:,0:1]
      d=tf.gather_nd(b,index2)
      e=tf.gather_nd(c,index2)
      width=e[:,3]-e[:,1]
      height=e[:,2]-e[:,0]
      ttt=(d[:,:4]-tf.stack([e[:,1],e[:,0],e[:,1],e[:,0]],axis=1))/tf.stack([width,height,width,height],axis=1)
      ttt=tf.where(ttt>0,ttt,0)
      ttt=tf.where(ttt<1,ttt,1)
      central=tf.stack([(ttt[:,1]+ttt[:,3])/2*self.hs,(ttt[:,0]+ttt[:,2])/2*self.ws],axis=1)
      t_shape=tf.stack((ttt[:,2]-ttt[:,0],ttt[:,3]-ttt[:,1]),axis=1)
      floor=tf.floor(central)#yx
      t_num=ttt.get_shape()[0]
      t_shape2=tf.tile(tf.reshape(t_shape,(-1,1,2),name='reshape4'),tf.constant((1,self.abn,1)))
      ss=tf.where(t_shape2>tf.reshape(self.default_anchor_size,(1,-1,2)),tf.reshape(self.default_anchor_size,(1,-1,2)),t_shape2)
      ddd=ss[:,:,0]*ss[:,:,1]
      similarities=ddd/(tf.reshape(self.default_anchor_size[:,0],(1,-1))*tf.reshape(self.default_anchor_size[:,1],(1,-1))+t_shape2[:,:,0]*t_shape2[:,:,1]-ddd)
      max_indice=tf.argmax(similarities,axis=1)
      ab_wh=tf.gather(self.default_anchor_size,max_indice)
      indices=tf.concat((tf.cast(index3,dtype=tf.int32),tf.cast(floor,tf.int32),tf.cast(tf.reshape(max_indice,[-1,1]),tf.int32)),axis=1)
      f=tf.ones_like(max_indice,dtype=tf.float32)
      g=tf.cast(central-floor,tf.float32)
      h=tf.cast(t_shape,tf.float32)/tf.cast(ab_wh,tf.float32)#xy
      i=tf.cast(d[:,4:],tf.float32)
      updates=tf.concat((tf.reshape(f,(-1,1)),g,h[:,1:2],h[:,0:1],i),axis=1)
      y2=tf.tensor_scatter_nd_update(y2,indices,updates)
      return y2
    def withoutname3(self,l1):
      l2=[]
      for i in l1:
        b=tf.reduce_sum(tf.one_hot(indices=tf.cast(i[...,5:]-1,dtype=tf.int32),depth=self.label_num),axis=-2)
        l2.append(tf.concat([i[...,:5],b],
                                          axis=-1))
        print(b.shape)
      return l2
    def withoutname4(self,l1):
      l2=[]
      for i in l1:
        l2.append(tf.one_hot(indices=tf.cast(i-1,dtype=tf.int32),depth=self.class_num))
      return l2
    @tf.function
    def __getitem__(self,idx=None):
        self.time0=time.time()
        tf.print('aasda')
        tf.print('\ntraining time:',self.time0-self.time1)
        box_indices = tf.random.uniform(shape=(self.batch_size,), maxval=self.train_data_num, dtype=tf.int32)
        #boxes:y1,x1,y2,x2  y1<y2,x1<x2
        #train_dataset_y_list:x1,y1,x2,y2 x1<x2,y1<y2
        boxes=tf.concat((tf.random.uniform(shape=(self.batch_size,1),minval=-0.1,maxval=0.2),
                          tf.random.uniform(shape=(self.batch_size,1),minval=-0.3,maxval=0.2),
                          tf.random.uniform(shape=(self.batch_size,1),maxval=1.3,minval=0.9),
                          tf.random.uniform(shape=(self.batch_size,1),maxval=1.0,minval=0.4)),axis=1)
        self.batch_x=tf.image.crop_and_resize(self.train_dataset_x,boxes=boxes,box_indices=box_indices,crop_size=(self.height,self.width))
        self.batch_y1=tf.reshape(self.withoutname2(self.train_dataset_y1_aug,box_indices,boxes),(-1,1+4+self.label_num))
        self.batch_y2=tf.cast(tf.gather_nd(self.train_dataset_y2,tf.reshape(box_indices,(-1,1))),dtype=tf.float32)
        self.time1=time.time()
        tf.print('data generation time:',self.time1-self.time0)
        # tf.print('batch_y1shape',self.batch_y1.shape)
        # return self.batch_x,{'detection':tf.reshape(self.batch_y1,(self.batch_size,self.hs,self.ws,self.abn*(1+4+self.label_num)),name='reshape6'),
        #                      'classification':self.batch_y2}
        return self.batch_x,{'detection':self.batch_y1,
                              'classification':self.batch_y2}
    def on_train_begin(self,logs=None):
      self.epoch_count=0
      self.loss=[]
      self.custom_loss=[]
      print('on_train_begin')
    def on_epoch_begin(self,epoch,logs=None):
      print('epoch_begin')
      self.epoch_count+=1
      # self.loss.append(None)
    def on_epoch_end(self,epoch=None,logs=None):
      # print(self.loss)
      # print(logs)
      print('epoch_end')
      if logs!=None:
        self.loss.append(logs)
        # print('custom_loss:',self.custom_loss[-1])
        # print('loss:',self.loss[-1])
        print(logs)
      if self.epoch_count%20==0:
        self.config_detail[1][1]+=20
        a=self.config_detail[1][1]
        print(self.config_detail[1][1])
        self.model.save(self.current_model+'.h5')
        self.config_detail.append(['train_epoch:',str(a),self.loss[-1]])
        with open(self.current_model+'.json','w') as f:
            json.dump(self.config_detail,f,ensure_ascii=False)


        # print(self.hsitory)
    
    def load_model(self,model_name=None):
      if model_name==None:
        if self.prev_modelname==None:
          print('error')
        elif not os.path.exists(self.prev_modelname+'.h5'):
          print('error')
        else:
          self.model=keras.models.load_model(self.prev_modelname+'.h5',custom_objects={'custom_loss': self.custom_loss,'custom_loss2':self.custom_loss2})
      elif not os.path.exists(model_name):
        print('error')
      else:
        self.model=keras.models.load_model(model_name,custom_objects={'custom_loss': self.custom_loss,'custom_loss2':self.custom_loss})

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
      indices=tf.where(output[...,0]>scorethreshold)
      # a=tf.tile(tf.reshape(tf.range(self.abn),(1,1,self.abn,1)),tf.constant([self.hs,self.ws,1,1]))
      # b=tf.tile(tf.reshape(tf.range(self.hs),(self.hs,1,1,1)),tf.constant([1,self.ws,self.abn,1]))
      # c=tf.tile(tf.reshape(tf.range(self.ws),(1,self.ws,1,1)),tf.constant([self.hs,1,self.abn,1]))
      # output=tf.concat([output,a,b,c],axis=3)

      output=tf.gather_nd(output,indices)
      S=tf.concat([output[:,0:1],output[:,5:]],axis=1)
      anchor=tf.gather_nd(self.default_anchor_size,indices[:,3:4])
      # print(indices[:5])
      # print(anchor[:5])
      # anchor=tf.cast(anchor,dtype=tf.float32)
      # print(anchor[:5])
      width=tf.exp(output[:,4])*anchor[:,0]
      height=tf.exp(output[:,3])*anchor[:,1]
      central_x=(tf.sigmoid(output[:,2])+tf.cast(indices[:,2],dtype=tf.float32))/self.ws
      central_y=(tf.sigmoid(output[:,1])+tf.cast(indices[:,1],dtype=tf.float32))/self.hs
      B=tf.stack([central_x-width/2,central_y-height/2,central_x+width/2,central_y+height/2],axis=1)
      # print(output[:5,4],output[:5,3],width[:5],height[:5])
      # print(anchor[:5])
      # print(self.default_anchor_size)
      l=NMS(B,S,iouthreshold)#x1,y1,x2,y2
      return l

    def seeing_result(self,num,scorethreshold,iouthreshold,dataset_type='validation',  augmentation=False):
      if dataset_type=='validation':
        dataset_x=self.validation_dataset_x[num:num+1]
        # dataset_y=self.validation_dataset_y[num:num+1]
      elif dataset_type=='train':
        if augmentation==False:
          dataset_x=self.train_dataset_x[num:num+1]
          # dataset_y=self.train_dataset_y[num:num+1]
        else:
          data=self.__getitem__()

          dataset_x=data[0][0:1]
          # dataset_y=data[1][0:1]
      y1,y2=self.model(dataset_x)
      l=self.post_process(y1,scorethreshold,iouthreshold)
      l0=tf.stack(l[0],axis=0)
      print(y2)
      print(l0)
      print(self.class_list)
      print(self.label_list)
      for i in l[1]:
        print(i)
      if l0.shape[0]>0:
        colors = tf.cast(tf.tile(tf.constant([[1,0,0]]),(l0.shape[0],1)),dtype=tf.float32)
        c=tf.stack((l0[:,1],l0[:,0],l0[:,3],l0[:,2]),axis=1)
        dataset_x=tf.image.draw_bounding_boxes(dataset_x,tf.reshape(c,(1,-1,4)),colors)
      plt.figure(figsize=(12,13))
      plt.imshow(dataset_x[0])

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

    def kmeans(self,central_num=4):
      shape_list=tf.zeros((0,2),tf.float32)
      tf.debugging.check_numerics(self.train_dataset_y1_aug,'train_dataset_y1_aug')
      for _ in range(100):
        box_indices = tf.random.uniform(shape=(self.batch_size,), maxval=self.train_data_num, dtype=tf.int32)
        #boxes:y1,x1,y2,x2  y1<y2,x1<x2
        #train_dataset_y_list:x1,y1,x2,y2 x1<x2,y1<y2
        boxes=tf.concat((tf.random.uniform(shape=(self.batch_size,1),minval=-0.2,maxval=0.2),
                          tf.random.uniform(shape=(self.batch_size,1),minval=-0.2,maxval=0.2),
                          tf.random.uniform(shape=(self.batch_size,1),maxval=1.2,minval=0.8),
                          tf.random.uniform(shape=(self.batch_size,1),maxval=1.2,minval=0.8)),axis=1)
        # boxes=tf.tile(tf.constant([[0,0,1,1]],dtype=tf.float32),(self.batch_size,1))
        a=tf.gather_nd(self.train_dataset_y1_aug,tf.reshape(box_indices,(-1,1)))
        index=tf.where((a[:,:,0]>0.5))
        b=tf.gather_nd(a[:,:,1:],index)
        c=tf.gather_nd(boxes,index[:,0:1])
        index2=tf.where(((b[:,0]+b[:,2])/2 > c[:,1]) & ((b[:,0]+b[:,2])/2 < c[:,3]) & 
                          ((b[:,1]+b[:,3])/2> c[:,0]) & ((b[:,1]+b[:,3])/2 < c[:,2]))
        d=tf.gather_nd(b,index2)#xy
        e=tf.gather_nd(c,index2)#yx
        width=e[:,3]-e[:,1]
        height=e[:,2]-e[:,0]
        ttt=(d[:,:4]-tf.stack([e[:,1],e[:,0],e[:,1],e[:,0]],axis=1))/tf.stack([width,height,width,height],axis=1)#xy
        ttt=tf.where(ttt>0,ttt,0)
        ttt=tf.where(ttt<1,ttt,1)
        t_shape=tf.stack((ttt[:,2]-ttt[:,0],ttt[:,3]-ttt[:,1]),axis=1)#xy
        shape_list=tf.concat((shape_list,t_shape),axis=0)
      # tf.debugging.check_numerics()
      tf.debugging.check_numerics(shape_list,'shape_list')
      print('min',tf.math.reduce_min(shape_list),'max',tf.math.reduce_max(shape_list))
      matplotlib.pyplot.scatter(shape_list[:,0],shape_list[:,1])
      central=kmeans(shape_list,central_num,method='iou')#xy
      print(central)
      tf.debugging.check_numerics(central,'central')
      matplotlib.pyplot.plot(central[:,0],central[:,1],'or')
      matplotlib.pyplot.show()
    def write_config(self):
      filename=r'./drive/Colab Notebooks/fgo_picture/fgo-support-config.json'
      new_list={'width':self.width,'height':self.height,'ws':self.ws,'hs':self.hs,'class_list':tuple(self.class_list),
                'label_list':tuple(self.label_list)}
      with open(filename,'w') as f:
        json.dump(new_list,f,ensure_ascii=False)
    def begin_train(self,augmentation=True,succeed=False,yanyong=False):
      self.make_model(succeed=succeed,yanyong=yanyong)
      adam = tf.keras.optimizers.Adam(clipvalue=1)
      self.model.compile(optimizer=adam,
                          loss={'detection':self.custom_loss,'classification':self.custom_loss2}
                         ,loss_weights={'detection':0.8,'classification':0.2})
      if augmentation:
        self.history=self.model.fit(self,initial_epoch=self.config_detail[0][1],epochs=1000,verbose=0,
                                    validation_data=(self.validation_dataset_x,
                                                     {'detection':self.validation_dataset_y1,'classification':self.validation_dataset_y2}),callbacks=[self])
      else:
        self.history=self.model.fit(self.train_dataset_x,self.train_dataset_y,initial_epoch=self.config_detail[0][1],epochs=1000,verbose=0,
                                    validation_data=(self.validation_dataset_x,
                                                     {'detection':self.validation_dataset_y1,'classification':self.validation_dataset_y2}),callbacks=[self])
      # print(self.hsitory)

a=yolov2()
# print(2)
# a.generate_dataset()
# a.kmeans(5)
# a.begin_train(augmentation=True,succeed=True,yanyong=False)
# a.generate_dataset()
# a.load_model(r'./drive/Colab Notebooks/model/fgo-support42.h5')
# a.seeing_result(num=0,scorethreshold=0.5,iouthreshold=0.5,dataset_type='validation')
# matplotlib.pyplot()
# # # x,y=a.return_dataset('train')
# n,m,k=a.evaluate_x()
# print(tf.reduce_sum(m[...,0]),tf.reduce_sum(m[...,-2]),tf.reduce_sum(m[...,-1]),tf.reduce_sum(n[...,0]),tf.reduce_sum(n[...,-2]),tf.reduce_sum(n[...,-1]),tf.reduce_sum(k[...,0]),tf.reduce_sum(k[...,-2]),tf.reduce_sum(k[...,-1]))
# print(tf.stack(l[1],axis=0))
# print(tf.concat(a.train_dataset_y_list[0],axis=0))
# print(tf.stack(l[0],axis=0))
# print(a.train_dataset_y_list[0])
