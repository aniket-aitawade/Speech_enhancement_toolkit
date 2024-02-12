import tensorflow as tf

class modified_transformer(tf.keras.layers.Layer):
    def __init__(self,n_fft):
        super().__init__()
        self.n_fft=n_fft
        
        self.concat=tf.keras.layers.Concatenate(axis=1)
        self.LSTM=tf.keras.layers.LSTM(self.n_fft,return_state=False)
        self.Add=tf.keras.layers.Add()
        self.layernorm1=tf.keras.layers.LayerNormalization()
        self.layernorm2=tf.keras.layers.LayerNormalization()
        self.layernorm3=tf.keras.layers.LayerNormalization()
        self.MultiHeadAttention=tf.keras.layers.MultiHeadAttention(4,64)
        self.Dense1=tf.keras.layers.Dense(self.n_fft,activation='relu')
        self.Dense2=tf.keras.layers.Dense(self.n_fft,activation='relu')
        
    def call(self,inputs):
        input1=tf.zeros(shape=[tf.shape(inputs)[0],2,self.n_fft])
        input2=self.concat([input1,inputs])
        
        x1=tf.signal.frame(input2,frame_length=3,frame_step=1,axis=1)
        x1=tf.reshape(x1,shape=[-1,3,self.n_fft])
        x1=self.LSTM(x1)
        x1=tf.reshape(x1,shape=[-1,tf.shape(inputs)[1],self.n_fft])
        x1=self.Add([x1,inputs])
        x1=self.layernorm1(x1)
        
        x=self.Dense2(x1)
        x1=self.Add([x,x1])
        x2=self.MultiHeadAttention(x1,x1)
        x2=self.Add([x2,x1])
        x2=self.layernorm2(x2)
        
        x3=self.Dense1(x2)
      
        x5=self.Add([x2,x3])
        x5=self.layernorm3(x5)
        return x5

class SEtransformer(tf.keras.Model):
    def __init__(self,units,input_shape):
        super().__init__()
        self.units=units
        self.Dense1=tf.keras.layers.Dense(2*(input_shape[-1]-1),activation='softmax')
        self.Dense2=tf.keras.layers.Dense(input_shape[-1],activation='relu')
        
        self.features_trans=[]
        self.frames_trans=[]

        for i in range(units):
            self.features_trans.append(modified_transformer(2*(input_shape[-1]-1)))
            self.frames_trans.append(modified_transformer(input_shape[-2]))
        
    def call(self, inputs):
        dim=tf.shape(inputs)
        inputs1=tf.reshape(inputs,shape=[-1,dim[-2],dim[-1]])
        x=self.Dense1(inputs1)
        for i in range(self.units):
            x=self.features_trans[i](x)
            x=tf.transpose(x,perm=[0,2,1])
            x=self.frames_trans[i](x)
            x=tf.transpose(x,perm=[0,2,1])
        x=self.Dense2(x)
        x=tf.reshape(x,shape=[dim[0],dim[1],dim[2],dim[3]])
        x=tf.multiply(inputs,x)
        return x

class trainer():
    def __init__(self,
                 units:int,
                 input_shape:list=[10,2,124,257]):
        self.units=units
        self.input_shape=input_shape
    
    def pack_model(self,input_shape,optimizer,loss,metrics):
        self.input_shape=input_shape
        SEmodel=SEtransformer(units=self.units,input_shape=input_shape)
        SEmodel.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return SEmodel

if __name__=="__main__":
    pass
    # SET=SEtransformer(512)
    # SET.build(input_shape=[10,2,154,257])
    # SET.summary()