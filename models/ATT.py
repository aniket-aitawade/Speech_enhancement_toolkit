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

class discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.concat=tf.keras.layers.Concatenate(axis=3)
        
        self.conv1=tf.keras.layers.Conv1D(filters=16,kernel_size=31,strides=2,padding="same")
        self.batch1=tf.keras.layers.BatchNormalization()
        self.LReLU1=tf.keras.layers.LeakyReLU(alpha=0.3)
        
        self.conv2=tf.keras.layers.Conv1D(filters=32,kernel_size=31,strides=2,padding="same")
        self.batch2=tf.keras.layers.BatchNormalization()
        self.LReLU2=tf.keras.layers.LeakyReLU(alpha=0.3)
        
        self.conv3=tf.keras.layers.Conv1D(filters=64,kernel_size=31,strides=2,padding="same")
        self.batch3=tf.keras.layers.BatchNormalization()
        self.LReLU3=tf.keras.layers.LeakyReLU(alpha=0.3)
        
        self.conv4=tf.keras.layers.Conv1D(filters=128,kernel_size=31,strides=2,padding="same")
        self.batch4=tf.keras.layers.BatchNormalization()
        self.LReLU4=tf.keras.layers.LeakyReLU(alpha=0.3)
        
        # self.conv5=tf.keras.layers.Conv1D(filters=1,kernel_size=1,strides=1,padding="same")
        # self.flatten=tf.keras.layers.Flatten()
        self.FC1=tf.keras.layers.Dense(100,activation='relu')
        self.FC2=tf.keras.layers.Dense(1,activation='sigmoid')

        
    def call(self,input1,input2,training=False):
        input_1=tf.expand_dims(input1,axis=3)
        input_2=tf.expand_dims(input2,axis=3)
        input=self.concat([input_1,input_2])
        # print(input.shape)
        
        x1=self.conv1(input)
        x1=self.batch1(x1)
        x1=self.LReLU1(x1)
        # print(x1.shape)
        
        x2=self.conv2(x1)
        x2=self.batch2(x2)
        x2=self.LReLU2(x2)
        # print(x2.shape)
        
        x3=self.conv3(x2)
        x3=self.batch3(x3)
        x3=self.LReLU3(x3)
        # print(x3.shape)
        
        x4=self.conv4(x3)
        x4=self.batch4(x4)
        x4=self.LReLU4(x4)
        # print(x4.shape)
        
        x5=x4
        # x5=self.conv5(x4)
        # print(x5.shape)
        x5=tf.reshape(x5,shape=[tf.shape(x5)[0],tf.shape(x5)[1],tf.shape(x5)[2]*tf.shape(x5)[3]])
        # x5=self.flatten(x5)
        # print(x5.shape)
        x5=self.FC1(x5)
        # print(x5.shape)
        x5=self.FC2(x5)
        # print(x5.shape)

        return x5

class ATT(tf.keras.Model):
    def __init__(self,n_fft, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.generator=SEtransformer(n_fft)
        self.discriminator=discriminator()
        
    def compile(self,g_opt,d_opt,*args,**kwargs):
        super().compile(*args,**kwargs)
        
        self.g_opt=g_opt
        self.d_opt=d_opt
        
    def train_step(self, batch):
        noisy,clean=batch
        real=clean
                        
        with tf.GradientTape() as d_tape,tf.GradientTape() as g_tape:
            fake=self.generator(noisy,training=True) 
            dim=tf.shape(fake)
            noisy=tf.reshape(noisy,shape=[-1,dim[-2],dim[-1]])
            fake=tf.reshape(fake,shape=[-1,dim[-2],dim[-1]])
            real=tf.reshape(real,shape=[-1,dim[-2],dim[-1]])
            y_hat_real=self.discriminator(noisy,real,training=True)
            y_hat_fake=self.discriminator(noisy,fake,training=True)
            total_d_loss=tf.reduce_mean(tf.math.squared_difference(y_hat_real,1.))+tf.reduce_mean(tf.math.squared_difference(y_hat_fake,0.))
            total_g_loss=tf.reduce_mean(tf.math.squared_difference(y_hat_fake,1.))+100*tf.reduce_mean(tf.abs(tf.subtract(fake,real)))
            
        d_grad=d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad,self.discriminator.trainable_variables))
        
        g_grad=g_tape.gradient(total_g_loss,self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad,self.generator.trainable_variables))
        return {"d_loss":total_d_loss,"g_loss":total_g_loss}
    
    def test_step(self, batch):
        noisy,clean=batch
        real=clean
                        
        fake=self.generator(noisy,training=False) 
        dim=tf.shape(fake)
        noisy=tf.reshape(noisy,shape=[-1,dim[-2],dim[-1]])
        fake=tf.reshape(fake,shape=[-1,dim[-2],dim[-1]])
        real=tf.reshape(real,shape=[-1,dim[-2],dim[-1]])
        y_hat_real=self.discriminator(noisy,real,training=False)
        y_hat_fake=self.discriminator(noisy,fake,training=False)
        total_d_loss=tf.reduce_mean(tf.math.squared_difference(y_hat_real,1.))+tf.reduce_mean(tf.math.squared_difference(y_hat_fake,0.))
        total_g_loss=100*tf.reduce_mean(tf.abs(tf.subtract(fake,real)))
        return {"d_loss":total_d_loss,"g_loss":total_g_loss}    
    
    def predict_step(self, noisy):
        dim=tf.shape(noisy)
        noisy=tf.reshape(noisy,shape=[-1,dim[-2],dim[-1]])   
        fake=self.generator(noisy,training=False) 
        fake=tf.reshape(fake,shape=dim)
        return fake
    
class trainer():
    def __init__(self,
                 n_fft:int,
                 input_shape:list=[1,124,257,2]):
        self.n_fft=n_fft
        self.input_shape=input_shape
    
    def pack_model(self,input_shape,optimizer,loss,metrics):
        self.input_shape=input_shape
        SEmodel=ATT(n_fft=self.n_fft)
        SEmodel.compile(g_opt=optimizer,d_opt=tf.keras.optimizers.RMSprop(learning_rate=0.0002))
        return SEmodel

if __name__=="__main__":
    pass
    # SET=SEtransformer(512)
    # SET.build(input_shape=[10,2,154,257])
    # SET.summary()