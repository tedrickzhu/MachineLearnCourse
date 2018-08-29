'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import h5py
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

from AEDataPro.dataset import animals

# import matplotlib.pyplot as plt
# from keras import losses

batch_size = 1000
original_dim = 4096
latent_dim = 85
intermediate_dim = 2048
intermediate_dim1 = 1024
intermediate_dim2 = 256
epochs = 40000
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
h1 = Dense(intermediate_dim1, activation='relu')(h)
h2 = Dense(intermediate_dim2, activation='relu')(h1)
# z_mean = Dense(latent_dim)(h2)
l = Dense(latent_dim, activation='relu')(h2)

# z_log_var = Dense(latent_dim)(h)


# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., std=epsilon_std)  #std change
#     return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])  ##z_mean and z_log_var are inputting, merging the results for tuning.

# we instantiate these layers separately so as to reuse them later
decoder_l = Dense(latent_dim, activation='relu')
decoder_h2 = Dense(intermediate_dim2, activation='relu')
decoder_h1 = Dense(intermediate_dim1, activation='relu')
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='relu')

l_decoded = decoder_l(l)
h2_decoded = decoder_h2(l_decoded)
h1_decoded = decoder_h1(h2_decoded)
h_decoded = decoder_h(h1_decoded)
x_decoded_mean = decoder_mean(h_decoded)

# x_pro_decoded_mean = merge([x_decoded_mean,l], mode='concat')

# def vae_loss(y_true, y_pred):
#     xent_loss = metrics.mean_squared_error(y_true, y_pred)
#     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     return K.mean(xent_loss + kl_loss)

vae = Model(x, [x_decoded_mean, l])
# encoder, from inputs to latent space
# encoder = Model(x, z_mean)

# # generator, from latent space to reconstructed inputs
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)


# vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.compile(optimizer='sgd', loss={'dense_9': 'mean_squared_error', 'dense_4': 'mean_absolute_error'},
            loss_weights=[1, 1.])

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = animals.load_org_data_continuous()
test_classes = animals.getTestClass()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


## extend training dataset for batch operation 1000 size
x_train = np.row_stack((x_train, x_train[0:505]))
y_train = np.row_stack((y_train, y_train[0:505]))
x_train = np.row_stack((x_train, x_train[0:200]))
y_train = np.row_stack((y_train, y_train[0:200]))

## extend testing dataset for batch operation 1000 size
x_test = np.row_stack((x_test, x_test[0:720]))

y_label = np.column_stack((x_train, y_train))

vae.fit(x_train, [x_train, y_train],
        shuffle=True,
        nb_epoch=epochs,
        batch_size=batch_size)

vae.save('models/animal/allData%dTrain%d.h5' % (epochs, original_dim))

x_train_encoded = vae.predict(x_train, batch_size=batch_size)
## save encoded file
encodedfile = h5py.File("dataset/animal/encodedTrainFile%d_85_add5_epoch%d.h5" % (original_dim, epochs), 'w')
encodedfile.create_dataset('trainx', data=x_train_encoded[0])
encodedfile.create_dataset('trainy', data=x_train_encoded[1])
encodedfile.close()

x_test_encoded = vae.predict(x_test, batch_size=batch_size)
## save encoded file
encodedfile = h5py.File("dataset/animal/encodedTestFile%d_85_add20_epoch%d.h5" % (original_dim, epochs), 'w')
encodedfile.create_dataset('trainx', data=x_test_encoded[0])
encodedfile.create_dataset('trainy', data=x_test_encoded[1])
encodedfile.close()

### adding$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ generate data


## call produce_property_code_testinstance.py to form constructed properties description
#####

# from aec import produce_property_code_testInstance4096


# ### creating model for instances producation
decoder_input = Input(shape=(latent_dim,))
_f_decoder = decoder_l(decoder_input)
_h_decoded = decoder_h(_f_decoder)
_x_decoded_mean = decoder_mean(_h_decoded)

generator = Model(decoder_input, _x_decoded_mean)

# ### loading produced encoded instances to form 255 dim data
# ### file proEncodeTestInstancesFromSimilarClass is created in python file produce_property_code_testInstance
# encodedfile = h5py.File("dataset/animal/proEncodeTestInstancesFromSimilarClass%d_epoch%d.h5" %(original_dim, epochs), 'r')
# x_test_encoded85 = []
# label = 1
# labels = []
# for i in test_classes:
#     tempInst = encodedfile[i][:]
#     print np.shape(tempInst)
#     x_test_encoded85.extend(encodedfile[i][:])
#     labels.extend([label for i in range(len(tempInst))])
#     label = label + 1

# encodedfile.close()

# # x_test_encoded85 = np.row_stack((x_test_encoded85,x_test_encoded85[0:91])) ## extend size to fit for batch
# # labels.extend([1 for i in range(91)]) ##expend the labels for train data

# x_test_encoded85 = np.row_stack((x_test_encoded85,x_test_encoded85[0:50])) ## extend size to fit for batch
# labels.extend([1 for i in range(50)]) ##expend the labels for train data



# #### producing encoding training data with labels

generator.save('models/animal/generater%dTrain%d.h5' % (epochs, original_dim))
x_decoded_generated4096 = generator.predict(y_test)
encodedfile = h5py.File("dataset/animal/generateTrainDataByDL_For_%d_With_labels_epoch%d.h5" % (original_dim, epochs),
                        'w')
encodedfile.create_dataset('trainx', data=x_decoded_generated4096)
encodedfile.create_dataset('trainy', data=y_test)
encodedfile.close()


# from aec import svm_predict4096



## testing for debug
# x_train_encoded_debug = vae.predict(x_train[0:2],batch_size =batch_size)
# # judge equaling
# print x_test_encoded_debug[0,255:]
# print y_test[0]
# a = np.array([0 for i in range(85)])
# for  i,j in zip(x_test_encoded_debug[0:2,255:],y_test[0:2]):
#     a[np.where(i<0)[0]] = -1
#     a[np.where(i>0)[0]] = 1
#     print a
#     print j - a




# # build a model to project inputs on the latent space
# encoder = Model(x, l)

# # # display a 2D plot of the digit classes in the latent space
# # x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# # plt.figure(figsize=(6, 6))
# # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# # plt.colorbar()
# # plt.show()

# ## see the different encoded values of every instances
# x_train_encoded = encoder.predict(x_train[0:2], batch_size=batch_size)

# print x_train_encoded
# print y_train[0:2]
# a = np.array([0 for i in range(85)])
# for  i,j in zip(x_train_encoded[0:2],y_train[0:2]):
#     a[np.where(i<0)[0]] = -1
#     a[np.where(i>0)[0]] = 1
#     print a
#     print j - a


# ###

# # Testing one instance encodered
# x_test_encoded = encoder.predict(x_train[0:2], batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()



# # build a digit generator that can sample from the learned distribution
# decoder_input = Input(shape=(latent_dim,))
# _f_decoder =decoder_l(decoder_input)
# _h_decoded = decoder_h(_f_decoder)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)

# x_decoded_generated = generator.predict(y_test[0:2])


# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()
