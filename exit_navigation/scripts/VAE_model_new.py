import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
import focal_loss


def downsample(filters, size, apply_batchnorm=True):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same'))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same'))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_batchnorm=True,apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same'))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.LeakyReLU())
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=1,
                                    padding='same'))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.LeakyReLU())
    return result

def get_stack_layerOutput(input_tmp,stack):
    x_tmp1 = input_tmp
    skips = []
    for i in range(len(stack)):
        x_tmp2 = stack[i](x_tmp1)    
        skips.append(x_tmp2)
        x_tmp1 = x_tmp2
    return skips


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        #dim = tf.shape(z_mean)[1] (batch,dim)
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        # print(z_mean.shape, z_log_var.shape, epsilon.shape)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  

    
class Conv2D_Encoder_Decoder(keras.Model):
    def __init__(self, input_size=128,latent_dim=2,layerList=[32,64,128,128],sampling_layer_Flag=False,multi_task_Flag=True,camera_Flag=True,skip_type='add'):
        super(Conv2D_Encoder_Decoder, self).__init__()
        self.sampling_layer_Flag = sampling_layer_Flag
        self.skip_type = skip_type
        self.camera_Flag = camera_Flag
        self.multi_task_Flag = multi_task_Flag
        self.x = layers.InputLayer(input_shape=(input_size, input_size, 4))
        self.down_stack = [
            layers.Conv2D(layerList[0], 4, activation='relu',strides=1, padding="same"),
        ]
        for i in range(1,len(layerList)):
            num_channel = layerList[i]
            self.down_stack.append(downsample(num_channel, 3))  # (batch_size, 8, 8, 128)
        self.z_mean_conv = downsample(latent_dim, 3)
        if self.sampling_layer_Flag:
            self.z_var_conv = downsample(latent_dim, 3)

        self.up_stack_exit = []
        for i in range(len(layerList)):
            num_channel = layerList[::-1][i]
            self.up_stack_exit.append(upsample(num_channel, 3))
        self.last_exit = layers.Conv2DTranspose(1, 3, activation="sigmoid",strides=1, padding="same",name='exit_last')

        if self.multi_task_Flag:
            self.up_stack_semantics = []
            for i in range(len(layerList)):
                num_channel = layerList[::-1][i]
                self.up_stack_semantics.append(upsample(num_channel, 3,apply_batchnorm=True))
            self.last_semantics = layers.Conv2DTranspose(3, 3, activation="softmax",strides=1, padding="same",name='semantics_last')
            
            self.up_stack_centroid = []
            for i in range(len(layerList)):
                num_channel = layerList[::-1][i]
                self.up_stack_centroid.append(upsample(num_channel, 3))
            self.last_centroid = layers.Conv2DTranspose(1, 3, activation="sigmoid",strides=1, padding="same",name='centroid_last')
                 

    def call(self,encoder_input, method=0, z=0.0):
        x = self.x(encoder_input)
        skips = get_stack_layerOutput(x,self.down_stack)
        z_mean = self.z_mean_conv(skips[-1])
        z_log_var = self.z_var_conv(skips[-1]) if self.sampling_layer_Flag else z_mean
        # z_mean = layers.Dense(latent_dim, name="z_mean")(x5)
        # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x5)
        


        if method == 0:
            if self.sampling_layer_Flag:
                z = Sampling()([z_mean, z_log_var])
            else:
                z = z_mean
        elif method == 1:
            if self.sampling_layer_Flag:
                z = Sampling()([z_mean, z_log_var])
            else:
                z = z_mean
        elif method == 2:
            z = z_mean
        

        x = z
        # up_exit = get_stack_layerOutput(z,self.up_stack_exit)
        # output_exit = self.last_exit(up_exit[-1])
        camera_skips = reversed(skips[0::])
        for up, skip in zip(self.up_stack_exit, camera_skips):
            x = up(x) 
            x = x + skip if self.skip_type=='add' else tf.concat((x,skip),axis=-1)
        output_exit = self.last_exit(x)

        if self.multi_task_Flag:
            x = z
            laser_skips = reversed(skips[0::])
    #         up_semantics = get_stack_layerOutput(z,self.up_stack_semantics[0:-1])
            # Upsampling and establishing the skip connections
            for up, skip in zip(self.up_stack_semantics, laser_skips):
                x = up(x) 
                x = x + skip if self.skip_type=='add' else tf.concat((x,skip),axis=-1)
            # output_semantics = self.up_stack_semantics[-1](x)
            output_semantics = self.last_semantics(x)
            
            up_centroid = get_stack_layerOutput(z,self.up_stack_centroid)
            output_centroid = self.last_centroid(up_centroid[-1])
            return z_mean,z_log_var, z, output_semantics,output_centroid,output_exit
        else:
            return z_mean,z_log_var, z, output_exit
        


class Conditional_VAE(keras.Model):
    def __init__(self,vae_model=None,sampling_layer_Flag=False,multi_task_Flag=True,exit_loss_weight=1.0,semi_learning=False):# **kwargs):
        super(Conditional_VAE, self).__init__()#(**kwargs)
        self.semi_learning = semi_learning
        self.multi_task_Flag = multi_task_Flag
        self.encoder_decoder = vae_model
#         self.rot_weight = 0.0
        self.recon_weight = 1.0
        self.exit_loss_weight = exit_loss_weight
        self.sampling_layer_Flag = sampling_layer_Flag

        self.train_total_loss_tracker = keras.metrics.Mean(
            name="train_total_loss"
        )
        self.val_total_loss_tracker = keras.metrics.Mean(
            name="val_total_loss"
        )
        if self.sampling_layer_Flag:
            self.train_kld_tracker = keras.metrics.Mean(
            name="train_kld_loss"
            )
            self.val_kld_tracker = keras.metrics.Mean(
            name="val_kld_loss"
            )

        self.train_exit_loss_tracker = keras.metrics.Mean(
                name="train_exit_loss"
            )
        self.train_exit_accuracy_tracker = keras.metrics.MeanIoU(num_classes=2,dtype=tf.double,
            name='train_exit_accuracy')
        self.val_exit_loss_tracker = keras.metrics.Mean(
            name="val_exit_loss"
        )
        self.val_exit_accuracy_tracker = keras.metrics.MeanIoU(num_classes=2,dtype=tf.double,
            name='val_exit_accuracy')
            
        if self.multi_task_Flag:
            self.train_semantics_loss_tracker = keras.metrics.Mean(
                name="train_semantics_loss"
            )
            self.train_semantics_accuracy_tracker = keras.metrics.MeanIoU(num_classes=3,dtype=tf.double,
                name='train_semantics_accuracy')
            self.train_centroid_loss_tracker = keras.metrics.Mean(
                name="train_centroid_loss"
            )
            self.train_centroid_accuracy_tracker = keras.metrics.MeanIoU(num_classes=2,dtype=tf.double,
                name='train_centroid_accuracy')
    #         self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        
            self.val_semantics_loss_tracker = keras.metrics.Mean(
                name="val_semantics_loss"
            )
            self.val_semantics_accuracy_tracker = keras.metrics.MeanIoU(num_classes=3,dtype=tf.double,
                name='val_semantics_accuracy')
            self.val_centroid_loss_tracker = keras.metrics.Mean(
                name="val_centroid_loss"
            )
            self.val_centroid_accuracy_tracker = keras.metrics.MeanIoU(num_classes=2,dtype=tf.double,
                name='val_centroid_accuracy')
            

    def call(self, x,method=0):
        return self.encoder_decoder.call(x,method=method)

    def reconst_loss0(self, y0, reconstruction):
        # return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y1, reconstruction)
        return focal_loss.SparseCategoricalFocalLoss(gamma=2.0,class_weight=(0.3,0.3,0.35))(y0, reconstruction)
    
    def reconst_loss1(self, y1, reconstruction):
        # return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y1, reconstruction)
        return focal_loss.BinaryFocalLoss(gamma=2.0,pos_weight=1.5)(y1, reconstruction)
    
    def reconst_loss2(self, y2, reconstruction):
        # return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y1, reconstruction)
        return focal_loss.BinaryFocalLoss(gamma=2.0,pos_weight=5)(y2, reconstruction) #pos_weight=1.5
        # pos_weight=5
    def mask_exit_output(self,x_temp,exit_predict,method=0):
        x_temp  = tf.concat((x_temp[:,:,:,0:3],tf.dtypes.cast(exit_predict[:,:,:,0]>=0.1, tf.float32)[...,tf.newaxis]),axis=-1)
        _,_,_,reconstruction_0,reconstruction_1,reconstruction_2=self.call(x_temp,method=method)
        return x_temp,reconstruction_2

    def extra_loss(self,x_temp,y_temp,exit_predict, method=0):
        # losses = tf.convert_to_tensor(0.0)
        x_temp,y_predict = self.mask_exit_output(x_temp,exit_predict,method=method)
        losses = self.reconst_loss2(exit_predict>=0.5,y_predict)
        return losses

    def train_step(self, data):
        x1, y1 = data
        # print(x1.shape)
        with tf.GradientTape() as tape:
            #z_mean, z_log_var, z = self.encoder(x1)
            #reconstruction = self.decoder(z) #* tf.expand_dims(x1[:, :, :, -1], axis=-1) + (x1[:, :, :, 0:1]*1.0 + x1[:, :, :, 1:2]*0.0) * (1-tf.expand_dims(x1[:, :, :, -1], axis=-1))
            # print(tf.shape(x1))
            if self.multi_task_Flag:
                z_mean, z_log_var, z, reconstruction_0, reconstruction_1, reconstruction_2 = self.call(x1,method=0)
                reconstruction_0_loss = self.reconst_loss0(y1[0], reconstruction_0)
                reconstruction_1_loss = self.reconst_loss1(y1[1], reconstruction_1) 
                reconstruction_2_loss = self.reconst_loss2(y1[2], reconstruction_2)
                reconstruction_total_loss = self.exit_loss_weight*reconstruction_2_loss + \
                    reconstruction_0_loss + reconstruction_1_loss 
                # if self.semi_learning:
                #     reconstruction_total_loss=reconstruction_total_loss+self.extra_loss(x1, y1[2],reconstruction_2,method=0)
            else:
                z_mean, z_log_var, z, reconstruction_2 = self.call(x1,method=0)
                reconstruction_2_loss = self.reconst_loss2(y1[2], reconstruction_2)
                reconstruction_total_loss = reconstruction_2_loss
#             print('check',tf.shape(x1),tf.shape(z),tf.shape(reconstruction))
            if self.sampling_layer_Flag:
                kld_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                # print(kl_loss)
                kld_loss =  tf.reduce_sum(kld_loss, axis=(1,2,3))#+ kl_loss
                self.train_kld_tracker.update_state(kld_loss)
                reconstruction_total_loss = reconstruction_total_loss + kld_loss
            
        # tf.print(y1,tf.math.argmax(reconstruction,axis=-1),tf.shape(y1),tf.shape(tf.math.argmax(reconstruction,axis=-1)))
        grads = tape.gradient(reconstruction_total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.train_total_loss_tracker.update_state(total_loss)
        self.train_total_loss_tracker.update_state(reconstruction_total_loss)
        self.train_exit_loss_tracker.update_state(reconstruction_2_loss)
        self.train_exit_accuracy_tracker.update_state(y1[2],reconstruction_2>0.5)
        returnDict = {
    #             "train_loss": self.train_total_loss_tracker.result(),
                "total_loss": self.train_total_loss_tracker.result(),
                "exit_loss":self.train_exit_loss_tracker.result(),
                "exit_accuracy":self.train_exit_accuracy_tracker.result()}
        if self.sampling_layer_Flag:
            returnDict.update({"kld_loss":self.train_kld_tracker.result()})
        if self.multi_task_Flag:
            self.train_semantics_loss_tracker.update_state(reconstruction_0_loss)
            self.train_semantics_accuracy_tracker.update_state(y1[0],tf.math.argmax(reconstruction_0,axis=-1)[...,tf.newaxis])
            self.train_centroid_loss_tracker.update_state(reconstruction_1_loss)
            self.train_centroid_accuracy_tracker.update_state(y1[1],reconstruction_1>0.5)
            multi_task_measures={
                "semantics_loss":self.train_semantics_loss_tracker.result(),
                "semantics_accuracy":self.train_semantics_accuracy_tracker.result(),
                "centroid_loss":self.train_centroid_loss_tracker.result(),
                "centroid_accuracy":self.train_centroid_accuracy_tracker.result(),
    #             "check":tf.reduce_sum(y1[0])
            }
            returnDict.update(multi_task_measures)
        return returnDict


    def test_step(self, data):
        x1, y1 = data
        # z_mean, z_log_var, z = self.encoder(x1, training=False)
        # reconstruction = self.decoder(z) # * tf.expand_dims(x1[:, :, :, -1], axis=-1) + (x1[:, :, :, 0:1]*1.0 + x1[:, :, :, 1:2]*0.0) * (1-tf.expand_dims(x1[:, :, :, -1], axis=-1))
        if self.multi_task_Flag:
            z_mean, z_log_var, z, reconstruction_0, reconstruction_1, reconstruction_2 = self.call(x1,method=1)
            reconstruction_0_loss = self.reconst_loss0(y1[0], reconstruction_0)
            reconstruction_1_loss = self.reconst_loss1(y1[1], reconstruction_1)
            reconstruction_2_loss = self.reconst_loss2(y1[2], reconstruction_2)
            reconstruction_total_loss = self.exit_loss_weight*reconstruction_2_loss + reconstruction_0_loss + reconstruction_1_loss
            # if self.semi_learning:
            #     reconstruction_total_loss=reconstruction_total_loss+self.extra_loss(x1, y1[2],reconstruction_2,method=1)
        else:
            z_mean, z_log_var, z, reconstruction_2 = self.call(x1,method=1)
            reconstruction_2_loss = self.reconst_loss2(y1[2], reconstruction_2)
            reconstruction_total_loss = reconstruction_2_loss
        
        if self.sampling_layer_Flag:
            kld_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # print(kl_loss)
            kld_loss =  tf.reduce_sum(kld_loss, axis=(1,2,3))#+ kl_loss
            self.train_kld_tracker.update_state(kld_loss)
            reconstruction_total_loss = reconstruction_total_loss + kld_loss
        self.val_total_loss_tracker.update_state(reconstruction_total_loss)
        self.val_exit_loss_tracker.update_state(reconstruction_2_loss)
        self.val_exit_accuracy_tracker.update_state(y1[2],reconstruction_2>0.5)
        returnDict = {
                "total_loss": self.val_total_loss_tracker.result(),
                "exit_loss":self.val_exit_loss_tracker.result(),
                "exit_accuracy":self.val_exit_accuracy_tracker.result()}
        
        if self.sampling_layer_Flag:
            returnDict.update({"kld_loss":self.val_kld_tracker.result()})
        if self.multi_task_Flag:
            self.val_semantics_loss_tracker.update_state(reconstruction_0_loss)
            self.val_semantics_accuracy_tracker.update_state(y1[0],tf.math.argmax(reconstruction_0,axis=-1)[...,tf.newaxis])
            self.val_centroid_loss_tracker.update_state(reconstruction_1_loss)
            self.val_centroid_accuracy_tracker.update_state(y1[1],reconstruction_1>0.5)
            multi_task_measures={
                "semantics_loss":self.val_semantics_loss_tracker.result(),
                "semantics_accuracy":self.val_semantics_accuracy_tracker.result(),
                "centroid_loss":self.val_centroid_loss_tracker.result(),
                "centroid_accuracy":self.val_centroid_accuracy_tracker.result(),
    #             "check":tf.reduce_sum(y1[0])
            }
            returnDict.update(multi_task_measures)
        return returnDict
