import tensorflow as tf

class modified_transformer(tf.keras.Model):
    def __init__(self,feature_size,heads,latent_dim,useConv,filters,kernel_size):
        super().__init__()
        self.feature_size=feature_size
        self.useConv=useConv
        
        self.concat=tf.keras.layers.Concatenate(axis=1)
        self.LSTM=tf.keras.layers.LSTM(self.feature_size,return_state=False)
        self.Add=tf.keras.layers.Add()
        self.layernorm1=tf.keras.layers.LayerNormalization()
        self.layernorm2=tf.keras.layers.LayerNormalization()
        self.layernorm3=tf.keras.layers.LayerNormalization()
        self.MultiHeadAttention=tf.keras.layers.MultiHeadAttention(heads,latent_dim)

        if useConv:
            self.layer1=tf.keras.layers.Conv2D(filters=filters,kernel_size=[kernel_size,kernel_size],activation='relu',padding='same')
            self.layer2=tf.keras.layers.Conv2D(filters=filters,kernel_size=[kernel_size,kernel_size],activation='relu',padding='same')
            self.layer3=tf.keras.layers.Conv2D(filters=1,kernel_size=[kernel_size,kernel_size],activation='relu',padding='same')
        else:
            self.layer1=tf.keras.layers.Dense(self.feature_size,activation='relu')
            self.layer2=tf.keras.layers.Dense(self.feature_size,activation='relu')
        
    def call(self,inputs):
        input1=tf.zeros(shape=[tf.shape(inputs)[0],2,self.feature_size])
        input2=self.concat([input1,inputs])
        
        x1=tf.signal.frame(input2,frame_length=3,frame_step=1,axis=1)
        x1=tf.reshape(x1,shape=[-1,3,self.feature_size])
        x1=self.LSTM(x1)
        x1=tf.reshape(x1,shape=[-1,tf.shape(inputs)[1],self.feature_size])
        x1=self.Add([x1,inputs])
        x1=self.layernorm1(x1)
        
        x2=self.MultiHeadAttention(x1,x1)
        x2=self.Add([x2,x1])
        x2=self.layernorm2(x2)
        
        if self.useConv:
            x3=tf.expand_dims(x2,axis=-1)
            x3=self.layer1(x3)
            x3=self.layer2(x3)
            x3=self.layer3(x3)
            x4=tf.squeeze(x3,axis=-1)
        else:
            x3=self.layer1(x2)
            x4=self.layer2(x3)
        
        x5=self.Add([x2,x4])
        x5=self.layernorm3(x5)
        return x5

class SEtransformer(tf.keras.Model):
    def __init__(self,units,feature_size,new_feature_size,heads,latent_dim,useConv,filters,kernel_size):
        super().__init__()
        self.units=units
        self.Dense1=tf.keras.layers.Dense(new_feature_size,activation='softmax')
        self.Dense2=tf.keras.layers.Dense(feature_size,activation='relu')
        self.modified_transformer=[]
        self.norm=[]
        for i in range(self.units):
            self.modified_transformer.append(modified_transformer(new_feature_size,heads,latent_dim,useConv,filters,kernel_size))
            self.norm.append(tf.keras.layers.LayerNormalization())
        
    def call(self, inputs):
        dim=tf.shape(inputs)
        x=tf.reshape(inputs,shape=[-1,dim[-2],dim[-1]])
        x=self.Dense1(x)
        for i in range(self.units):
            x1=x
            x=self.modified_transformer[i](x)
            x=tf.keras.layers.Add()([x,x1])
            x=self.norm[i](x)

        x=self.Dense2(x)
        x=tf.reshape(x,shape=[dim[0],dim[1],dim[2],dim[3]])
        x=tf.multiply(inputs,x)
        return x

class trainer():
    def __init__(self,
                 units:int,
                 new_feature_size:int,
                 heads:int,
                 latent_dim:int,
                 useConv:bool,
                 filters:int,
                 kernel_size:int,
                 input_shape:list=[1,124,257]):
        self.units=units
        self.new_feature_size=new_feature_size
        self.heads=heads
        self.latent_dim=latent_dim
        self.useConv=useConv
        self.filters=filters
        self.kernel_size=kernel_size
        self.input_shape=input_shape
    
    def pack_model(self,input_shape,optimizer,loss,metrics):
        self.input_shape=input_shape
        SEmodel=SEtransformer(self.units,self.input_shape[-1],self.new_feature_size,self.heads,self.latent_dim,self.useConv,self.filters,self.kernel_size)
        SEmodel.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return SEmodel

if __name__=="__main__":
    pass
    # SET=SEtransformer(512)
    # SET.build(input_shape=[10,2,154,257])
    # SET.summary()