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
        
        x2=self.MultiHeadAttention(x1,x1)
        x2=self.Add([x2,x1])
        x2=self.layernorm2(x2)
        
        x3=self.Dense1(x2)
        x4=self.Dense2(x3)
        
        x5=self.Add([x2,x4])
        x5=self.layernorm3(x5)
        return x5

class SEtransformer(tf.keras.Model):
    def __init__(self,n_fft):
        super().__init__()
        self.n_fft=n_fft
        self.Dense1=tf.keras.layers.Dense(self.n_fft,activation='softmax')
        self.Dense2=tf.keras.layers.Dense(int(self.n_fft/2)+1,activation='relu')
        
        self.modified_transformer1=modified_transformer(self.n_fft)
        self.modified_transformer2=modified_transformer(self.n_fft)
        
    def call(self, inputs):
        dim=tf.shape(inputs)
        inputs=tf.reshape(inputs,shape=[-1,dim[-2],dim[-1]])
        x1=self.Dense1(inputs)
        x2=self.modified_transformer1(x1)
        x3=self.modified_transformer2(x2)
        x4=self.Dense2(x3)
        x4=tf.reshape(x4,shape=[dim[0],dim[1],dim[2],dim[3]])
        return x4

class trainer():
    def __init__(self,
                 n_fft:int,
                 input_shape:list=[1,124,257,2]):
        self.n_fft=n_fft
        self.input_shape=input_shape
    
    def pack_model(self,input_shape,optimizer,loss,metrics):
        self.input_shape=input_shape
        SEmodel=SEtransformer(n_fft=self.n_fft)
        SEmodel.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return SEmodel

if __name__=="__main__":
    pass
    # SET=SEtransformer(512)
    # SET.build(input_shape=[10,2,154,257])
    # SET.summary()