import numpy as np

import focal_loss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

def downsample(filters, size, apply_batchnorm=True):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same'))

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

    result.add(tf.keras.layers.ReLU())
    return result

def get_stack_layerOutput(input_tmp,stack):
    x_tmp1 = input_tmp
    skips = []
    for i in range(len(stack)):
        x_tmp2 = stack[i](x_tmp1)    
        skips.append(x_tmp2)
        x_tmp1 = x_tmp2
    return skips
 

    
class Conditional_Encoder_Decoder(keras.Model):
    def __init__(self,input_size=128,latent_dim=2,laser_encoder_model=None,sampling_layer_Flag=False,exit_Flag=True,camera_Flag=True,skip_type='add'):
        super(Conditional_Encoder_Decoder, self).__init__()
        self.exit_Flag = exit_Flag
        self.skip_type = skip_type
        self.camera_Flag = camera_Flag
        self.sampling_layer_Flag=sampling_layer_Flag
        self.x0_0 = layers.InputLayer(input_shape=(128, 128, 3))
        self.conv0_0 = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same",name='init')#laser_encoder_model.layers[1]
        self.conv1_0 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")#laser_encoder_model.layers[2]
        self.conv2_0 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")#laser_encoder_model.layers[3]
        self.conv3_0 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")#laser_encoder_model.layers[4]
        self.conv4_0 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")#laser_encoder_model.layers[5]
        self.conv5_0 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")#laser_encoder_model.layers[6]
        self.z_mean_conv_0 = layers.Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same", name="z_mean")#laser_encoder_model.layers[7]
        # z_log_var = z_mean#self.z_var_conv(x5)
        if self.sampling_layer_Flag:
            self.z_var_conv_laser =layers.Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same", name="z_var")#laser_encoder_model.layers[7]
        if self.camera_Flag:
            self.x0_1 = layers.InputLayer(input_shape=(128, 128, 3))
            self.conv0_1 = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same",name='init')
            self.conv1_1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
            self.conv2_1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
            self.conv3_1= layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
            self.conv4_1 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")
            self.conv5_1 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")
            self.z_mean_conv_1 = layers.Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same", name="z_mean")
        
#         self.z_var_conv = layers.Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same", name="z_var")
        self.deconv5_0 = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")
        self.deconv4_0 = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")
        self.deconv3_0 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.deconv2_0 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.deconv1_0 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        
        self.last_01 = layers.Conv2DTranspose(32, 3, activation="relu",strides=2, padding="same")
#         self.last_02 = layers.Conv2DTranspose(16, 3, activation="relu",strides=1, padding="same")
        self.last_02= layers.Conv2DTranspose(3, 3, activation="softmax",strides=1, padding="same",name='semantics_last')
        
        self.deconv5_1 = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")
        self.deconv4_1 = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")
        self.deconv3_1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.deconv2_1 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.deconv1_1 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        
        self.last_11 = layers.Conv2DTranspose(32, 3, activation="relu",strides=2, padding="same")
        self.last_12 = layers.Conv2DTranspose(1, 3, activation="sigmoid",strides=1, padding="same",name='centroid_last')
        
        if self.exit_Flag:
            self.deconv5_2 = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")
            self.deconv4_2 = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")
            self.deconv3_2 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
            self.deconv2_2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
            self.deconv1_2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
            self.last_21 = layers.Conv2DTranspose(32, 3, activation="relu",strides=2, padding="same")
            self.last_22 = layers.Conv2DTranspose(1, 3, activation="sigmoid",strides=1, padding="same",name='exit_last')

    def call(self,encoder_input, method=0, z=0.0):
        x0_0 = self.x0_0(encoder_input[:,:,:,:3])
        x0_0 = self.conv0_0(x0_0)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)
        z_mean_0 = self.z_mean_conv_0(x5_0)
        z_log_var_0 = z_mean_0#self.z_var_conv(x5)
        
        # z_mean = layers.Dense(latent_dim, name="z_mean")(x5)
        # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x5)
        if method == 0:
            if self.sampling_layer_Flag:
                z_0 = Sampling()([z_mean_0, z_log_var_0])
            else:
                z_0 = z_mean_0
        elif method == 1:
            z_0 = z_mean_0
        elif method == 2:
            z_0 = z_0
        if self.camera_Flag:
            x0_1 = self.x0_1(encoder_input[:,:,:,3::])
            x0_1 = self.conv0_1(x0_1)
            x1_1 = self.conv1_1(x0_1)
            x2_1 = self.conv2_1(x1_1)
            x3_1 = self.conv3_1(x2_1)
            x4_1 = self.conv4_1(x3_1)
            x5_1 = self.conv5_1(x4_1)
            z_1 = self.z_mean_conv_1(x5_1)
            
            z = tf.concat((z_0,z_1),axis=-1)
        else:
            z = z_0
        # encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # y5 = layers.Dense(4 * 4 * 128, activation="relu")(z)
        # y4 = layers.Reshape((4, 4, 128))(y5) + x4
#         print(tf.shape(x0_0),tf.shape(x1_0),tf.shape(x2_0),
#               tf.shape(x3_0),tf.shape(x4_0),tf.shape(x5_0),
#               tf.shape(z_mean_0),tf.shape(self.deconv5_0(z)))
        y5_0 = self.deconv5_0(z) + x5_0 if self.skip_type=='add' else tf.concat((self.deconv5_0(z),x5_0),axis=-1)
        y4_0 = self.deconv4_0(y5_0) + x4_0 if self.skip_type=='add' else tf.concat((self.deconv4_0(y5_0),x4_0),axis=-1)
        y3_0 = self.deconv3_0(y4_0) + x3_0 if self.skip_type=='add' else tf.concat((self.deconv3_0(y4_0),x3_0),axis=-1)
        y2_0 = self.deconv2_0(y3_0) + x2_0 if self.skip_type=='add' else tf.concat((self.deconv2_0(y3_0),x2_0),axis=-1)
        y1_0 = self.deconv1_0(y2_0) + x1_0 if self.skip_type=='add' else tf.concat((self.deconv1_0(y2_0),x1_0),axis=-1)
        y0_0 = self.last_01(y1_0) + x0_0 if self.skip_type=='add' else tf.concat((self.last_01(y1_0),x0_0),axis=-1)
        decoder_outputs_0 = self.last_02(y0_0)
        
        y5_1 = self.deconv5_1(z)
        y4_1 = self.deconv4_1(y5_1)
        y3_1 = self.deconv3_1(y4_1)
        y2_1 = self.deconv2_1(y3_1)
        y1_1 = self.deconv1_1(y2_1)
        y0_1 = self.last_11(y1_1) 
        decoder_outputs_1 = self.last_12(y0_1)
        if self.exit_Flag:
            y5_2 = self.deconv5_2(z) 
            y4_2 = self.deconv4_2(y5_2) 
            y3_2 = self.deconv3_2(y4_2) 
            y2_2 = self.deconv2_2(y3_2) 
            y1_2 = self.deconv1_2(y2_2)
            y0_2 = self.last_21(y1_2) 
            decoder_outputs_2 = self.last_22(y0_2)
            return z,0, z, decoder_outputs_0,decoder_outputs_1,decoder_outputs_2
        else:
            return z,0, z, decoder_outputs_0,decoder_outputs_1


# /rplan_new_cameraTrue_multiTask_exit_u-net_latent64_0315
class Conditional_New_Encoder_Decoder(keras.Model):
    def __init__(self, input_size=128,latent_dim=2,laser_encoder_model=None,sampling_layer_Flag=False,exit_Flag=True,camera_Flag=True,skip_type='add'):
        super(Conditional_New_Encoder_Decoder, self).__init__()
        self.sampling_layer_Flag = sampling_layer_Flag
        self.exit_Flag = exit_Flag
        self.skip_type = skip_type
        self.camera_Flag = camera_Flag
        self.x_laser = layers.InputLayer(input_shape=(128, 128, 3))
        self.down_stack_laser = [
            layers.Conv2D(32, 3, activation='relu',strides=1, padding="same"),
            # downsample(32, 3),  # (batch_size, 64, 64, 32)
            downsample(64, 3),  # (batch_size, 32, 32, 64)
            downsample(128, 3),  # (batch_size, 16, 16, 128)
            downsample(128, 3),  # (batch_size, 8, 8, 128)
        ]
        self.z_mean_conv_laser = downsample(latent_dim, 3)
        
        if self.sampling_layer_Flag:
            self.z_var_conv_laser = downsample(latent_dim, 3)
        if camera_Flag:
            self.x_camera = layers.InputLayer(input_shape=(128, 128, 1))
            self.down_stack_camera = [
                layers.Conv2D(32, 3, activation='relu',strides=1, padding="same"),
                # downsample(32, 3),  # (batch_size, 64, 64, 32)
                downsample(64, 3),  # (batch_size, 32, 32, 64)
                downsample(128, 3),  # (batch_size, 16, 16, 128)
                downsample(128, 3)  # (batch_size, 8, 8, 128)
            ]
            self.z_mean_conv_camera = downsample(latent_dim, 3) # (batch_size, 2, 2, latent_dim)

        self.up_stack_semantics = [
            upsample(128, 3,apply_batchnorm=False),  # (batch_size, 8, 8, 256)
            upsample(128, 3,apply_batchnorm=False),  # (batch_size, 16, 16, 256)
            upsample(64, 3,apply_batchnorm=False),  # (batch_size, 32, 32, 128)
            # upsample(32, 3),  # (batch_size, 64, 64, 64)
            upsample(32, 3,apply_batchnorm=False)
        ]

    #         self.last_02 = layers.Conv2DTranspose(16, 3, activation="relu",strides=1, padding="same")
        
        self.last_semantics = layers.Conv2DTranspose(3, 3, activation="softmax",strides=1, padding="same",name='semantics_last')
        
        self.up_stack_centroid = [
            upsample(128, 3),  # (batch_size, 8, 8, 256)
            upsample(128, 3),  # (batch_size, 16, 16, 256)
            upsample(64, 3),  # (batch_size, 32, 32, 128)
            # upsample(32, 3),  # (batch_size, 64, 64, 64)
            upsample(32, 3)
        ]
        self.last_centroid = layers.Conv2DTranspose(1, 3, activation="sigmoid",strides=1, padding="same",name='centroid_last')
        
        if self.exit_Flag:
            self.up_stack_exit = [
                upsample(128, 3),  # (batch_size, 8, 8, 256)
                upsample(128, 3),  # (batch_size, 16, 16, 256)
                upsample(64, 3),  # (batch_size, 32, 32, 128)
                # upsample(32, 3),  # (batch_size, 64, 64, 64)
                upsample(32, 3)
            ]
            self.last_exit = layers.Conv2DTranspose(1, 3, activation="sigmoid",strides=1, padding="same",name='exit_last')

    def call(self,encoder_input, method=0, z=0.0):
        x_laser = self.x_laser(encoder_input[:,:,:,:3])
        laser_skips = get_stack_layerOutput(x_laser,self.down_stack_laser)
        z_mean_0 = self.z_mean_conv_laser(laser_skips[-1])
        z_log_var_0 = self.z_var_conv_laser(laser_skips[-1]) if self.sampling_layer_Flag else z_mean_0
        # z_mean = layers.Dense(latent_dim, name="z_mean")(x5)
        # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x5)
        if method == 0:
            if self.sampling_layer_Flag:
                z_0 = Sampling()([z_mean_0, z_log_var_0])
            else:
                z_0 = z_mean_0
        elif method == 1:
            z_0 = z_mean_0
        elif method == 2:
            z_0 = z_0

        if self.camera_Flag:
            x_camera = self.x_laser(encoder_input[:,:,:,3::])
            camera_skips = get_stack_layerOutput(x_camera,self.down_stack_camera)
            z_1 = self.z_mean_conv_camera(camera_skips[-1])
            
            z = tf.concat((z_0,z_1),axis=-1)
        else:
            z = z_0
        
        x = z
        laser_skips = reversed(laser_skips[0::])
#         up_semantics = get_stack_layerOutput(z,self.up_stack_semantics[0:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack_semantics, laser_skips):
            x = up(x) 
            x = x + skip if self.skip_type=='add' else tf.concat((x,skip),axis=-1)
        # output_semantics = self.up_stack_semantics[-1](x)
        output_semantics = self.last_semantics(x)
        
        up_centroid = get_stack_layerOutput(z,self.up_stack_centroid)
        output_centroid = self.last_centroid(up_centroid[-1])
        if self.exit_Flag:
            x = z
            # up_exit = get_stack_layerOutput(z,self.up_stack_exit)
            # output_exit = self.last_exit(up_exit[-1])
            camera_skips = reversed(camera_skips[0::])
            for up, skip in zip(self.up_stack_exit, camera_skips):
                x = up(x) 
                x = x + skip if self.skip_type=='add' else tf.concat((x,skip),axis=-1)
            output_exit = self.last_exit(x)
            return z_mean_0,z_log_var_0, z, output_semantics,output_centroid,output_exit
        else:
            return z_mean_0,z_log_var_0, z, output_semantics,output_centroid



class Conditional_New_VAE(keras.Model):
    def __init__(self,input_size=128,latent_dim=2,vae_model=None,sampling_layer_Flag=False,exit_Flag=True,exit_loss_weight=1.0):# **kwargs):
        super(Conditional_New_VAE, self).__init__()#(**kwargs)
        self.exit_Flag = exit_Flag
        self.encoder_decoder = Conditional_New_Encoder_Decoder(latent_dim=latent_dim,skip_type='concat')# Conditional_New_Encoder_Decoder(input_size=input_size,latent_dim=latent_dim,\
        #     laser_encoder_model=None,sampling_layer_Flag=sampling_layer_Flag,\
        #         exit_Flag=exit_Flag,camera_Flag=True,skip_type='concat')
#         self.rot_weight = 0.0
        self.recon_weight = 1.0
        self.exit_loss_weight = exit_loss_weight
        self.train_total_loss_tracker = keras.metrics.Mean(
            name="train_total_loss"
        )
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
        
        self.val_total_loss_tracker = keras.metrics.Mean(
            name="val_total_loss"
        )
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
        if self.exit_Flag:
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
        return focal_loss.BinaryFocalLoss(gamma=2.0,pos_weight=1.5)(y2, reconstruction)

    def train_step(self, data):
        x1, y1 = data
        
        with tf.GradientTape() as tape:
            #z_mean, z_log_var, z = self.encoder(x1)
            #reconstruction = self.decoder(z) #* tf.expand_dims(x1[:, :, :, -1], axis=-1) + (x1[:, :, :, 0:1]*1.0 + x1[:, :, :, 1:2]*0.0) * (1-tf.expand_dims(x1[:, :, :, -1], axis=-1))
#             print(tf.shape(x1))
            if self.exit_Flag:
                z_mean, z_log_var, z, reconstruction_0, reconstruction_1, reconstruction_2 = self.call(x1,method=0)
                reconstruction_0_loss = self.reconst_loss0(y1[0], reconstruction_0)
#                 kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                # print(kl_loss)
#                 kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1,2,3))) + kl_loss
                reconstruction_1_loss = self.reconst_loss1(y1[1], reconstruction_1) 
                reconstruction_2_loss = self.reconst_loss2(y1[2], reconstruction_2)
                reconstruction_total_loss = reconstruction_0_loss + reconstruction_1_loss + self.exit_loss_weight*reconstruction_2_loss
            else:
                z_mean, z_log_var, z, reconstruction_0, reconstruction_1 = self.call(x1,method=0)
                reconstruction_0_loss = self.reconst_loss0(y1[0], reconstruction_0)
                reconstruction_1_loss = self.reconst_loss1(y1[1], reconstruction_1)
                reconstruction_total_loss = reconstruction_0_loss + reconstruction_1_loss
#             print('check',tf.shape(x1),tf.shape(z),tf.shape(reconstruction))
            
        # tf.print(y1,tf.math.argmax(reconstruction,axis=-1),tf.shape(y1),tf.shape(tf.math.argmax(reconstruction,axis=-1)))
        grads = tape.gradient(reconstruction_total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.train_total_loss_tracker.update_state(total_loss)
        self.train_total_loss_tracker.update_state(reconstruction_total_loss)
        self.train_semantics_loss_tracker.update_state(reconstruction_0_loss)
        self.train_semantics_accuracy_tracker.update_state(y1[0],tf.math.argmax(reconstruction_0,axis=-1)[...,tf.newaxis])
        self.train_centroid_loss_tracker.update_state(reconstruction_1_loss)
        self.train_centroid_accuracy_tracker.update_state(y1[1],reconstruction_1>0.5)
        if self.exit_Flag:
            self.train_exit_loss_tracker.update_state(reconstruction_2_loss)
            self.train_exit_accuracy_tracker.update_state(y1[2],reconstruction_2>0.5)
            return {
    #             "train_loss": self.train_total_loss_tracker.result(),
                "total_loss": self.train_total_loss_tracker.result(),
                "semantics_loss":self.train_semantics_loss_tracker.result(),
                "semantics_accuracy":self.train_semantics_accuracy_tracker.result(),
                "centroid_loss":self.train_centroid_loss_tracker.result(),
                "centroid_accuracy":self.train_centroid_accuracy_tracker.result(),
                "exit_loss":self.train_exit_loss_tracker.result(),
                "exit_accuracy":self.train_exit_accuracy_tracker.result(),
    #             "check":tf.reduce_sum(y1[0])
            }
        else:
            return {
    #             "train_loss": self.train_total_loss_tracker.result(),
                "total_loss": self.train_total_loss_tracker.result(),
                "semantics_loss":self.train_semantics_loss_tracker.result(),
                "semantics_accuracy":self.train_semantics_accuracy_tracker.result(),
                "centroid_loss":self.train_centroid_loss_tracker.result(),
                "centroid_accuracy":self.train_centroid_accuracy_tracker.result(),
    #             "check":tf.reduce_sum(y1[0])
            }

    def test_step(self, data):
        x1, y1 = data
        # z_mean, z_log_var, z = self.encoder(x1, training=False)
        # reconstruction = self.decoder(z) # * tf.expand_dims(x1[:, :, :, -1], axis=-1) + (x1[:, :, :, 0:1]*1.0 + x1[:, :, :, 1:2]*0.0) * (1-tf.expand_dims(x1[:, :, :, -1], axis=-1))
        if self.exit_Flag:
            z_mean, z_log_var, z, reconstruction_0, reconstruction_1, reconstruction_2 = self.call(x1,method=1)
            reconstruction_0_loss = self.reconst_loss0(y1[0], reconstruction_0)
            reconstruction_1_loss = self.reconst_loss1(y1[1], reconstruction_1)
            reconstruction_2_loss = self.reconst_loss2(y1[2], reconstruction_2)
            reconstruction_total_loss = reconstruction_0_loss + reconstruction_1_loss + self.exit_loss_weight*reconstruction_2_loss
        else:
            z_mean, z_log_var, z, reconstruction_0,reconstruction_1 = self.call(x1,method=1)
            reconstruction_0_loss = self.reconst_loss0(y1[0], reconstruction_0)
            reconstruction_1_loss = self.reconst_loss1(y1[1], reconstruction_1)
            reconstruction_total_loss = reconstruction_0_loss + reconstruction_1_loss
        self.val_total_loss_tracker.update_state(reconstruction_total_loss)
        self.val_semantics_loss_tracker.update_state(reconstruction_0_loss)
        self.val_semantics_accuracy_tracker.update_state(y1[0],tf.math.argmax(reconstruction_0,axis=-1)[...,tf.newaxis])
        self.val_centroid_loss_tracker.update_state(reconstruction_1_loss)
        self.val_centroid_accuracy_tracker.update_state(y1[1],reconstruction_1>0.5)
        if self.exit_Flag:
            self.val_exit_loss_tracker.update_state(reconstruction_2_loss)
            self.val_exit_accuracy_tracker.update_state(y1[2],reconstruction_2>0.5)
            return {
    #             "train_loss": self.train_total_loss_tracker.result(),
                "total_loss": self.val_total_loss_tracker.result(),
                "semantics_loss":self.val_semantics_loss_tracker.result(),
                "semantics_accuracy":self.val_semantics_accuracy_tracker.result(),
                "centroid_loss":self.val_centroid_loss_tracker.result(),
                "centroid_accuracy":self.val_centroid_accuracy_tracker.result(),
                "exit_loss":self.val_exit_loss_tracker.result(),
                "exit_accuracy":self.val_exit_accuracy_tracker.result(),
    #             "check":tf.reduce_sum(y1[0])
            }
        else:
            return {
    #             "train_loss": self.train_total_loss_tracker.result(),
                "total_loss": self.val_total_loss_tracker.result(),
                "semantics_loss":self.val_semantics_loss_tracker.result(),
                "semantics_accuracy":self.val_semantics_accuracy_tracker.result(),
                "centroid_loss":self.val_centroid_loss_tracker.result(),
                "centroid_accuracy":self.val_centroid_accuracy_tracker.result(),
    #             "check":tf.reduce_sum(y1[0])

            }