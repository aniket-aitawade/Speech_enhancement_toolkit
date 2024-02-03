import tensorflow as tf
from omegaconf import OmegaConf

config=OmegaConf.load('./config.yaml')

class Dense_layer(tf.keras.layers.Layer):
    def __init__(self,kernel,filter,layers):
        super().__init__()
        
        self.filters=filter
        self.layers=layers
        self.conv_layers=[]
        self.Norm_layer=[]
        self.PRelu=[]
        for i  in range(layers):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=self.filters,kernel_size=[kernel[0],kernel[1]],strides=1,padding='same',dilation_rate=2))
            self.Norm_layer.append(tf.keras.layers.LayerNormalization())
            self.PRelu.append(tf.keras.layers.ReLU())
            
    def call(self,inputs):
        output=inputs
        for i in range(self.layers):
            input=output
            output=self.conv_layers[i](input)
            output=self.Norm_layer[i](output)
            output=self.PRelu[i](output)
            if(i<self.layers-1):
                output=tf.keras.layers.concatenate([output,input],axis=3)
        return output

class modified_transformer(tf.keras.layers.Layer):
    def __init__(self,n_fft):
        super().__init__()
        self.n_fft=n_fft
        
        self.concat=tf.keras.layers.Concatenate(axis=1)
        self.LSTM=tf.keras.layers.LSTM(self.n_fft,return_state=True)
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
        _,x1,_=self.LSTM(x1)
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

        self.Add=tf.keras.layers.Add()
        self.layernorm=tf.keras.layers.LayerNormalization()
        self.Dense1=tf.keras.layers.Dense(self.n_fft,activation='softmax')
        self.Dense2=tf.keras.layers.Dense(int(self.n_fft/2)+1,activation='relu')
        
        self.modified_transformer1=modified_transformer(self.n_fft)
        self.modified_transformer2=modified_transformer(self.n_fft)

        if not config.preprocessing.abs:
            self.Dense_layer=Dense_layer(kernel=[5,5],filter=2,layers=4)
            self.conv1=tf.keras.layers.Conv2D(filters=1,kernel_size=[1,1],strides=1,padding='same')
            self.conv2=tf.keras.layers.Conv2D(filters=2,kernel_size=[1,1],strides=1,padding='same')
        
    def call(self, inputs):
        dim=tf.shape(inputs)
        inputs=tf.reshape(inputs,shape=[-1,dim[-3],dim[-2],dim[-1]])

        x1=self.Dense_layer(inputs)
        x1=self.Add([x1,inputs])
        x1=self.layernorm(x1)
        x1=self.conv1(x1)
        print(x1.shape)
        x1=tf.squeeze(x1,axis=-1)

        x1=self.Dense1(x1)
        x2=self.modified_transformer1(x1)
        x3=self.modified_transformer2(x2)
        x4=self.Dense2(x3)

        x4=tf.expand_dims(x4,axis=-1)
        x4=self.conv2(x4)
        x4=tf.stack([x4[:,:,:,0],x4[:,:,:,1]],axis=-1)

        # x4=tf.math.multiply(inputs,x4)

        x4=tf.reshape(x4,shape=[dim[0],dim[1],dim[2],dim[3],dim[4]])

        return x4
      
def trainer(n_fft):
    SEmodel=SEtransformer(n_fft)
    SEmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),loss=tf.keras.losses.mse)
    return SEmodel

if __name__=="__main__":
    pass