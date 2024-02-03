import tensorflow as tf
import numpy as np

class Dense_layer(tf.keras.Model):
    def __init__(self,kernel,filter,layers):
        super().__init__()
        self.shape=None
        self.filters=filter
        self.layer=layers
        self.conv_layers=[]
        self.Norm_layer=[]
        self.PRelu=[]
        for i  in range(layers):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=self.filters,kernel_size=[kernel[0],kernel[1]],strides=1,padding='same',dilation_rate=2))
            self.Norm_layer.append(tf.keras.layers.LayerNormalization())
            self.PRelu.append(tf.keras.layers.ReLU())
            
    def call(self,inputs):
        output=inputs
        for i in range(self.layer):
            input=output
            output=self.conv_layers[i](input)
            output=self.Norm_layer[i](output)
            output=self.PRelu[i](output)
            if (i<self.layer-1):
                output=tf.keras.layers.concatenate([output,input],axis=-1)
        return output
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class encoder(tf.keras.Model):
    def __init__(self,filter,layers):
        super().__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=filter,kernel_size=[1,1],strides=1,padding="same")
        self.Dense_block=Dense_layer(kernel=[2,3],filter=filter,layers=layers)
        self.shape=None
        
    def call(self,inputs):
        x=self.conv1(inputs)
        x=self.Dense_block(x)
        return x
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class decoder(tf.keras.Model):
    def __init__(self,filter,layers):
        super().__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=2,kernel_size=[1,1],strides=1,padding="same")
        self.Dense_block=Dense_layer(kernel=[2,3],filter=filter,layers=layers)
        self.shape=None
        
    def call(self,inputs):
        x=self.Dense_block(inputs)
        x=self.conv1(x)
        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class LSTM_embedder(tf.keras.Model):
    def __init__(self,features,frames=3):
        super().__init__()     
        self.shape=None
        self.features=features   
        self.frames=frames
        self.concat=tf.keras.layers.Concatenate(axis=1)
        self.LSTM=tf.keras.layers.LSTM(self.features,return_state=False)
        self.Add=tf.keras.layers.Add()
        self.layernorm1=tf.keras.layers.LayerNormalization()
        
    def call(self,inputs):
        padding=tf.zeros(shape=[tf.shape(inputs)[0],self.frames-1,self.features])
        input=self.concat([padding,inputs])
        
        x=tf.signal.frame(input,frame_length=self.frames,frame_step=1,axis=1)
        x=tf.reshape(x,shape=[-1,self.frames,self.features])
        x=self.LSTM(x)
        x=tf.reshape(x,shape=[-1,tf.shape(inputs)[1],self.features])

        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class transformer(tf.keras.Model):
    def __init__(self,input_shape,features,frames,heads,latent_dim,dense_units):
        super().__init__()
        self.shape=input_shape
        self.LSTM_embedder=LSTM_embedder(features=features,frames=frames)
        self.attention_layer=tf.keras.layers.MultiHeadAttention(heads,latent_dim)
        self.layer_norm1=tf.keras.layers.LayerNormalization()
        self.dense_layer=tf.keras.layers.Dense(dense_units)
        self.layer_norm2=tf.keras.layers.LayerNormalization()
        self.add1=tf.keras.layers.Add()
        self.add2=tf.keras.layers.Add()

    def call(self,inputs):
        x=tf.reshape(inputs,shape=[-1,self.shape[-2],self.shape[-1]])
        x=self.LSTM_embedder(x)
        x=self.attention_layer(x,x)
        x=tf.reshape(x,shape=[-1,self.shape[-3],self.shape[-2],self.shape[-1]])
        x=self.add1([x,inputs])
        x1=self.layer_norm1(x)
        x=self.dense_layer(x1)
        x=self.add2([x,x1])
        x=self.layer_norm2(x)

        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class DPT(tf.keras.Model):
    def __init__(self,input_shape,units,features,frames,heads,latent_dim,dense_units):
        super().__init__()
        self.shape=input_shape
        self.units=units
        self.intra_transformers=[]
        self.inter_transformers=[]
        self.intra_batch_norm=[]
        self.inter_batch_norm=[]
        self.intra_add=[]
        self.inter_add=[]

        for i in range(units):
            self.intra_transformers.append(transformer(input_shape,features,frames,heads,latent_dim,dense_units))
            self.inter_transformers.append(transformer([None,input_shape[-2],input_shape[-3],input_shape[-1]],features,frames,heads,latent_dim,dense_units))
            self.intra_batch_norm.append(tf.keras.layers.BatchNormalization())
            self.inter_batch_norm.append(tf.keras.layers.BatchNormalization())
            self.intra_add.append(tf.keras.layers.Add())
            self.inter_add.append(tf.keras.layers.Add())

    def call(self,inputs):
        x=inputs
        for i in range(self.units):
            take=x
            x=self.intra_transformers[i](x)
            x=self.intra_batch_norm[i](x)
            x=self.intra_add[i]([x,take])
            
            x=tf.transpose(x,perm=[0,2,1,3])
            take=x
            x=self.inter_transformers[i](x)
            x=self.inter_batch_norm[i](x)
            x=self.inter_add[i]([x,take])
            x=tf.transpose(x,perm=[0,2,1,3])
        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()

class DPTPM(tf.keras.Model):
    def __init__(self,input_shape,units,features,frames,heads,latent_dim,dense_units):
        super().__init__()
        self.shape=None
        self.conv_layer1=tf.keras.layers.Conv2D(filters=32,kernel_size=[1,1],strides=1,padding="same")
        self.DPT=DPT(input_shape,units,features,frames,heads,latent_dim,dense_units)
        self.conv_layer2=tf.keras.layers.Conv2D(filters=64,kernel_size=[1,1],strides=1,padding="same")

    def call(self,inputs):
        x=self.conv_layer1(inputs)
        x=self.DPT(x)
        x=self.conv_layer2(x)
        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class DPTFSNET(tf.keras.Model):
    def __init__(self,filter,layers,input_shape,units,features,frames,heads,latent_dim,dense_units):
        super().__init__()
        self.shape=None
        self.encoder=encoder(filter,layers)
        self.DPTPM=DPTPM(input_shape,units,features,frames,heads,latent_dim,dense_units)
        self.decoder=decoder(filter,layers)

    def call(self,inputs):
        x=self.encoder(inputs)
        x=self.DPTPM(x)
        x=self.decoder(x)
        x=tf.math.multiply(inputs,x)
        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
class model(tf.keras.Model):
    def __init__(self,filter,input_shape,units,features,frames,heads,latent_dim,dense_units):
        super().__init__()
        self.shape=None
        self.model=DPTFSNET(filter,input_shape,units,features,frames,heads,latent_dim,dense_units)
        
    def call(self,inputs):
        shape=tf.shape(inputs)
        x=tf.reshape(inputs,shape=[shape[0]*shape[1],shape[2],shape[3],shape[4]])
        x=self.model(x)
        x=tf.reshape(x,shape=[shape[0],shape[1],shape[2],shape[3],shape[4]])
        return x
    
    def build(self,input_shape):
        self.shape=input_shape
        return
    
    def summary(self):
        x=tf.keras.Input(shape=self.shape)
        model=tf.keras.Model(inputs=x,outputs=self.call(x))
        return model.summary()
    
# model=model(filter=64,input_shape=[124,257,32],units=2,features=32,frames=3,heads=4,latent_dim=64,dense_units=32)
# model.build(input_shape=[124,257,2])
# print(model.summary())
if __name__=="__main__":
    pass