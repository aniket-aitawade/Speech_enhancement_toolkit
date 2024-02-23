import tensorflow as tf

class modified_transformer(tf.keras.Model):
    def __init__(self,feature_size,heads,latent_dim,useConv,filters,kernel_size):
        super().__init__()
        self.feature_size=feature_size
        
        self.Add=tf.keras.layers.Add()
        self.layernorm1=tf.keras.layers.LayerNormalization()
        self.layernorm2=tf.keras.layers.LayerNormalization()
        self.MultiHeadAttention=tf.keras.layers.MultiHeadAttention(heads,latent_dim)

        self.GRU=tf.keras.layers.GRU(units=feature_size,activation='relu',return_sequences=True)
        self.Dense=tf.keras.layers.Dense(units=feature_size)
        
    def call(self,inputs):
        x=inputs
        x=self.MultiHeadAttention(x,x)
        x=self.Add([x,inputs])
        x=self.layernorm1(x)

        x1=x
        x=self.GRU(x)
        x=self.Dense(x)
        x=self.Add([x,x1])
        x=self.layernorm2(x)
        
        return x

class SEtransformer(tf.keras.Model):
    def __init__(self,units,input_shape,new_feature_size,heads,latent_dim,useConv,filters,kernel_size):
        super().__init__()
        self.units=units
        self.Dense1=tf.keras.layers.Dense(new_feature_size,activation='softmax')
        self.Dense2=tf.keras.layers.Dense(input_shape[-2],activation='relu')
        self.conv1=tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding="same",activation="relu")
        self.conv2=tf.keras.layers.Conv1D(filters=2,kernel_size=kernel_size,padding="same",activation="relu")
        self.Add=tf.keras.layers.Add()

        self.layernorm1=[]
        self.layernorm2=[]
        self.feature_transformer=[]
        self.frames_transformer=[]
        for i in range(self.units):
            self.feature_transformer.append(modified_transformer(new_feature_size,heads,latent_dim,useConv,filters,kernel_size))
            self.frames_transformer.append(modified_transformer(input_shape[-3],heads,latent_dim,useConv,filters,kernel_size))
            self.layernorm1.append(tf.keras.layers.LayerNormalization())
            self.layernorm2.append(tf.keras.layers.LayerNormalization())
        
    def call(self, inputs):
        x=self.conv1(inputs)
        x=tf.squeeze(x,axis=-1)
        dim=tf.shape(x)
        x=tf.reshape(x,shape=[-1,dim[-2],dim[-1]])
        x=self.Dense1(x)
        for i in range(self.units):
            x1=x
            x=self.feature_transformer[i](x)
            x=self.Add([x,x1])
            x=self.layernorm1[i](x)
            x=tf.transpose(x,perm=[0,2,1])
            x1=x
            x=self.frames_transformer[i](x)
            x=self.Add([x,x1])
            x=self.layernorm2[i](x)
            x=tf.transpose(x,perm=[0,2,1])
        x=self.Dense2(x)
        x=tf.reshape(x,shape=[dim[0],dim[1],dim[2],dim[3]])
        x=tf.expand_dims(x,axis=-1)
        x=self.conv2(x)
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
                 input_shape:list=[1,251,257,2]):
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
        SEmodel=SEtransformer(self.units,self.input_shape,self.new_feature_size,self.heads,self.latent_dim,self.useConv,self.filters,self.kernel_size)
        SEmodel.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return SEmodel

if __name__=="__main__":
    pass
    # SET=SEtransformer(512)
    # SET.build(input_shape=[10,2,154,257])
    # SET.summary()