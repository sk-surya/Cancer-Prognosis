#!/usr/bin/env python
# coding: utf-8

# # Micrograph Classification

# ## Necessary Imports

# In[1]:


import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn import svm
import time
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier


# In[2]:


raw_input_df = pd.read_csv('Datasets\Assignment 2\micrograph.csv')
#display(raw_input_df.head())

# selecting only spheroidite, network and pearlite data
labels = set(['spheroidite', 'network', 'pearlite'])
input_df = raw_input_df[[x in labels for x in raw_input_df['primary_microconstituent']] ]
#display(input_df.head())


# ## Train/Test split

# In[3]:


train_spher_df = input_df[input_df['primary_microconstituent']=='spheroidite'].iloc[:100, :]
train_net_df = input_df[input_df['primary_microconstituent']=='network'].iloc[:100, :]
train_pear_df = input_df[input_df['primary_microconstituent']=='pearlite'].iloc[:100, :]

assert train_spher_df.shape[0] == 100
assert train_net_df.shape[0] == 100
assert train_pear_df.shape[0] == 100

assert pd.Series(train_spher_df['primary_microconstituent']=='spheroidite').all()
assert pd.Series(train_net_df['primary_microconstituent']=='network').all()
assert pd.Series(train_pear_df['primary_microconstituent']=='pearlite').all()

#display(train_spher_df.head())
#display(train_net)
#display(train_pear)

test_spher_df = input_df[input_df['primary_microconstituent']=='spheroidite'].iloc[100:, :]
test_net_df = input_df[input_df['primary_microconstituent']=='network'].iloc[100:, :]
test_pear_df = input_df[input_df['primary_microconstituent']=='pearlite'].iloc[100:, :]

#print(test_spher.shape)
#print(test_net.shape)
#print(test_pear.shape)

train_df = pd.concat([train_spher_df, train_net_df, train_pear_df])
test_df = pd.concat([test_spher_df, test_net_df, test_pear_df])

#display(train_df.head())
print(train_df.shape)
print(test_df.shape)



# ### Pre-processing Images


def preprocess_images(img_dir, filename_list):
    img_list = [image.load_img(img_dir + img_file) for img_file in filename_list]

    x_list = [image.img_to_array(img) for img in img_list]

    from matplotlib import pyplot as plt

    #Cropping out subtitles
    x_list = np.array([x[0:484, :, :] for x in x_list])
    #adding dummy dimension to each image in x_list (to access after expansion, use x_list[0 to n][0][j][k])
    x_list = np.array([np.expand_dims(x, axis=0) for x in x_list])

    #plt.imshow(np.asarray(img_list[300-1])[0:484,:,:])
    #plt.show()

    x_list = np.array([preprocess_input(x) for x in x_list])

    #print(x_list.shape)

    #plt.imshow(x_list[300-1][0])
    #plt.show()
    
    return x_list


# ## Defining VGG16 CNN model

# In[7]:


vgg16 = VGG16(weights='imagenet', include_top=False)
#vgg16.summary()


# ### Accessing convolutional layers and creating extractor models

# In[8]:


#s = time.time()
conv_block_1_out = vgg16.get_layer('block1_pool').output
conv_block_2_out = vgg16.get_layer('block2_pool').output
conv_block_3_out = vgg16.get_layer('block3_pool').output
conv_block_4_out = vgg16.get_layer('block4_pool').output
conv_block_5_out = vgg16.get_layer('block5_pool').output

feature_extractor_64 = Model(inputs=vgg16.input, outputs=conv_block_1_out)
feature_extractor_128 = Model(inputs=vgg16.input, outputs=conv_block_2_out)
feature_extractor_256 = Model(inputs=vgg16.input, outputs=conv_block_3_out)
feature_extractor_512a = Model(inputs=vgg16.input, outputs=conv_block_4_out)
feature_extractor_512b = Model(inputs=vgg16.input, outputs=conv_block_5_out)
e = time.time()

#print((e-s)*1000000, 'microseconds')


# Now all we have to do is, preprocess the image and pass it to model.predict() to get the channels.

# Lets pass the entire training image set to our 'preprocess_images' utility function that we created.

# In[9]:


s = time.time()
img_dir = 'Datasets\Assignment 2\micrograph\\'
img_files_train = train_df['path'].values
img_files_test = test_df['path'].values
#img_files = raw_input_df['path'].values

#x_train = preprocess_images(img_dir, img_files_train)
#np.save('x_train.npy', x_train)
#x_test = preprocess_images(img_dir, img_files_test)
#np.save('x_test.npy', x_test)

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')

print(x_train.shape)
y_train = train_df['primary_microconstituent'].values
y_test = test_df['primary_microconstituent'].values
print(type(x_train))
print(type(y_train))
e = time.time()
print('pre-preprocessing', (e-s)*1000, 'milliseconds')


# In[10]:


def get_2D_mean_64(x):
    return np.mean(feature_extractor_64.predict(x), axis=(1,2))
def get_2D_mean_128(x):
    return np.mean(feature_extractor_128.predict(x), axis=(1,2))
def get_2D_mean_256(x):
    return np.mean(feature_extractor_256.predict(x), axis=(1,2))
def get_2D_mean_512a(x):
    return np.mean(feature_extractor_512a.predict(x), axis=(1,2))
def get_2D_mean_512b(x):
    return np.mean(feature_extractor_512b.predict(x), axis=(1,2))

def get_2D_mean(x):
    return np.mean(x, axis=(1,2))


# Now, we have 300 preprocessed image arrays. Lets send them for feature extraction

# In[11]:


start = time.time()
def get_channel_means2(x_train):
    #x_train = x_train[:1]
    #for x in x_train:
    #pool = mp.Pool(processes=11)
    #f_64 = feature_extractor_64.predict(x)
    f_64_mean = list(map(get_2D_mean_64, x_train))
    #print(f_64_mean[0])
    #f_64_mean = [np.mean(x, axis=(1,2)) for x in f_64]
    #pool.close()
    #pool.join()
    #f_128 = feature_extractor_128.predict(x)
    #f_128_mean = pool.map(get_2D_mean_128, x_train)
    #f_256_mean = pool.map(get_2D_mean_256, x_train)
    #f_512a_mean = pool.map(get_2D_mean_512a, x_train)
    #f_512b_mean = pool.map(get_2D_mean_512b, x_train)
    #f_128_mean = [np.mean(feature_extractor_128.predict(x), axis=(1,2)) for x in x_train]

    #f_256_mean = [np.mean(feature_extractor_256.predict(x), axis=(1,2)) for x in x_train]

    #f_512a_mean = [np.mean(feature_extractor_512a.predict(x), axis=(1,2)) for x in x_train]

    #f_512b_mean = [np.mean(feature_extractor_512b.predict(x), axis=(1,2)) for x in x_train]

    np.save('f_64_mean', f_64_mean)
    print('w')
    return 0
    #np.save('f_128_mean', f_128_mean)
    #np.save('f_256_mean', f_256_mean)
    #np.save('f_512a_mean', f_512a_mean)
    #np.save('f_512b_mean', f_512b_mean)
		
def get_channel_means(x_train):
    #x_train = x_train[:1]

    f_64_mean = [np.mean(feature_extractor_64.predict(x), axis=(1,2)) for x in x_train]
    f_128_mean = [np.mean(feature_extractor_128.predict(x), axis=(1,2)) for x in x_train]
    f_256_mean = [np.mean(feature_extractor_256.predict(x), axis=(1,2)) for x in x_train]
    f_512a_mean = [np.mean(feature_extractor_512a.predict(x), axis=(1,2)) for x in x_train]
    f_512b_mean = [np.mean(feature_extractor_512b.predict(x), axis=(1,2)) for x in x_train]
    
    np.save('f_64_mean', f_64_mean)
    np.save('f_128_mean', f_128_mean)
    np.save('f_256_mean', f_256_mean)
    np.save('f_512a_mean', f_512a_mean)
    np.save('f_512b_mean', f_512b_mean)
    
    return 0
    
#f_64_mean = [f_64[image][0, :, :, x].mean() for x in range(0,64)] for image in range(0,300)
#f_64_mean = [np.mean(z, axis=(1,2)) for z in f_64]
#f_128 = [feature_extractor_128.predict(x) for x in x_train]
#f_256 = [feature_extractor_256.predict(x) for x in x_train]
#f_512a = [feature_extractor_512a.predict(x) for x in x_train]
#f_512b = [feature_extractor_512b.predict(x) for x in x_train]
end = time.time()


# In[ ]:





# In[ ]:


start = time.time()
#x = get_channel_means(x_train,)
#print(x)
#x = get_channel_means(x_test, 'test')
end = time.time()

print('list comp: ', end - start, 'seconds')


f_64_mean = np.load('f_64_mean.npy')
f_128_mean = np.load('f_128_mean.npy')
f_256_mean = np.load('f_256_mean.npy')
f_512a_mean = np.load('f_512a_mean.npy')
f_512b_mean = np.load('f_512b_mean.npy')

f_64_mean = np.squeeze(f_64_mean, axis=(1,))
f_128_mean = np.squeeze(f_128_mean, axis=(1,))
f_256_mean = np.squeeze(f_256_mean, axis=(1,))
f_512a_mean = np.squeeze(f_512a_mean, axis=(1,))
f_512b_mean = np.squeeze(f_512b_mean, axis=(1,))


df1 = pd.DataFrame(data= f_64_mean.tolist())
df2 = pd.DataFrame(data= f_128_mean.tolist())
df3 = pd.DataFrame(data= f_256_mean.tolist())
df4 = pd.DataFrame(data= f_512a_mean.tolist())
df5 = pd.DataFrame(data= f_512b_mean.tolist())

#df1.to_csv('f_64_mean.csv')
#df2.to_csv('f_128_mean.csv')
#df3.to_csv('f_256_mean.csv')
#df4.to_csv('f_512a_mean.csv')
#df5.to_csv('f_512b_mean.csv')

# #### Non-linear SVM with Radial Basis Function kernel

# In[4]:


clf = svm.SVC(kernel='rbf', gamma='auto')

print(f_64_mean.shape)
#print(np.squeeze(f_64_mean, axis=(1,)).shape)
'''
indices_sp = np.where( y_train == 'spheroidite' )[0]
indices_pl = np.where( y_train == 'pearlite' )[0]
indices_nw = np.where( y_train == 'network' )[0]

x_train_sp = f_64_mean[indices_sp]
x_train_pl = f_64_mean[indices_pl]
x_train_nw = f_64_mean[indices_nw]

y_train_sp = y_train[indices_sp]
y_train_pl = y_train[indices_pl]
y_train_nw = y_train[indices_nw]


#p1_df = pd.DataFrame(data=x_train)
#print(p1_df)


x_train_p1 = np.vstack((x_train_sp, x_train_nw))
x_train_p2 = np.vstack((x_train_sp, x_train_pl))
x_train_p3 = np.vstack((x_train_pl, x_train_nw))

y_train_p1 = np.hstack((y_train_sp, y_train_nw))
y_train_p2 = np.hstack((y_train_sp, y_train_pl))
y_train_p3 = np.hstack((y_train_pl, y_train_nw))
'''


def get_XY(parent_X, parent_Y, child_1, child_2):
    indices_1 = np.where( parent_Y == child_1 )[0]
    if child_2 != '':
        indices_2 = np.where( parent_Y == child_2 )[0]
    
    x_train_1 = parent_X[indices_1]
    if child_2 != '':
        x_train_2 = parent_X[indices_2]
    else:
        x_train_2 = []

    y_train_1 = parent_Y[indices_1]
    if child_2 != '':
        y_train_2 = parent_Y[indices_2]
    else:
        y_train_2 = []

    if child_2 != '':
        x_train_pair = np.vstack((x_train_1, x_train_2))
        y_train_pair = np.hstack((y_train_1, y_train_2))
    else:
        x_train_pair = x_train_1
        y_train_pair = y_train_1
    
    return x_train_pair, y_train_pair


'''
assert np.array(x_p1 == x_train_p1).all()
assert np.array(x_p2 == x_train_p2).all()
assert np.array(x_p3 == x_train_p3).all()

assert np.array(y_p1 == y_train_p1).all()
assert np.array(y_p2 == y_train_p2).all()
assert np.array(y_p3 == y_train_p3).all()
'''

#print(x_train_p1)
#print(x_train_p2)
#print(x_train_p3)

#print(x_train_nw)
#print(x_train_p1)
#print(y_train_p2)
#print(y_train_p3)


#clf.fit(x_train_p1, y_train_p1)
#print(x_train_p1.shape)
#print(y_train_p1.shape)

#print(1-cross_val_score(clf, x_p1, y_p1, cv=10).mean())

x_p1_1, y_p1_1 = get_XY(f_64_mean, y_train, 'spheroidite', 'network')
x_p1_2, y_p1_2 = get_XY(f_128_mean, y_train, 'spheroidite', 'network')
x_p1_3, y_p1_3 = get_XY(f_256_mean, y_train, 'spheroidite', 'network')
x_p1_4, y_p1_4 = get_XY(f_512a_mean, y_train, 'spheroidite', 'network')
x_p1_5, y_p1_5 = get_XY(f_512b_mean, y_train, 'spheroidite', 'network')

print(1-cross_val_score(clf, x_p1_1, y_p1_1, cv=10).mean())
print(1-cross_val_score(clf, x_p1_2, y_p1_2, cv=10).mean())
print(1-cross_val_score(clf, x_p1_3, y_p1_3, cv=10).mean())
print(1-cross_val_score(clf, x_p1_4, y_p1_4, cv=10).mean())
print(1-cross_val_score(clf, x_p1_5, y_p1_5, cv=10).mean())
print()

x_p2_1, y_p2_1 = get_XY(f_64_mean, y_train, 'spheroidite', 'pearlite')
x_p2_2, y_p2_2 = get_XY(f_128_mean, y_train, 'spheroidite', 'pearlite')
x_p2_3, y_p2_3 = get_XY(f_256_mean, y_train, 'spheroidite', 'pearlite')
x_p2_4, y_p2_4 = get_XY(f_512a_mean, y_train, 'spheroidite', 'pearlite')
x_p2_5, y_p2_5 = get_XY(f_512b_mean, y_train, 'spheroidite', 'pearlite')

print(1-cross_val_score(clf, x_p2_1, y_p2_1, cv=10).mean())
print(1-cross_val_score(clf, x_p2_2, y_p2_2, cv=10).mean())
print(1-cross_val_score(clf, x_p2_3, y_p2_3, cv=10).mean())
print(1-cross_val_score(clf, x_p2_4, y_p2_4, cv=10).mean())
print(1-cross_val_score(clf, x_p2_5, y_p2_5, cv=10).mean())
print()

x_p3_1, y_p3_1 = get_XY(f_64_mean, y_train, 'pearlite', 'network')
x_p3_2, y_p3_2 = get_XY(f_128_mean, y_train, 'pearlite', 'network')
x_p3_3, y_p3_3 = get_XY(f_256_mean, y_train, 'pearlite', 'network')
x_p3_4, y_p3_4 = get_XY(f_512a_mean, y_train, 'pearlite', 'network')
x_p3_5, y_p3_5 = get_XY(f_512b_mean, y_train, 'pearlite', 'network')

print(1-cross_val_score(clf, x_p3_1, y_p3_1, cv=10).mean())
print(1-cross_val_score(clf, x_p3_2, y_p3_2, cv=10).mean())
print(1-cross_val_score(clf, x_p3_3, y_p3_3, cv=10).mean())
print(1-cross_val_score(clf, x_p3_4, y_p3_4, cv=10).mean())
print(1-cross_val_score(clf, x_p3_5, y_p3_5, cv=10).mean())
print()


# Test errors with best feature set

#f_512b_mean_test = [np.mean(feature_extractor_512b.predict(x), axis=(1,2)) for x in x_test]
#np.save('f_512b_mean_test', f_512b_mean_test)
f_512b_mean_test = np.load('f_512b_mean_test.npy')
f_512b_mean_test = np.squeeze(f_512b_mean_test, axis=(1,))

x_pl_test, y_pl_test = get_XY(f_512b_mean_test, y_test, 'pearlite', '')
x_nw_test, y_nw_test = get_XY(f_512b_mean_test, y_test, 'network', '')
x_sp_test, y_sp_test = get_XY(f_512b_mean_test, y_test, 'spheroidite', '')

#x_p2_test, y_p2_test = get_XY(f_512b_mean_test, y_test, 'spheroidite', 'network')
#x_p3_test, y_p3_test = get_XY(f_512b_mean_test, y_test, 'spheroidite', 'network')

multilabel_cl = OneVsOneClassifier(clf)



multilabel_cl.fit(f_512b_mean, y_train)
print(1-multilabel_cl.score(x_pl_test, y_pl_test))
print(1-multilabel_cl.score(x_nw_test, y_nw_test))
print(1-multilabel_cl.score(x_sp_test, y_sp_test))
print()



#x_p1_5, y_p1_5 = get_XY(f_512b_mean, y_train, 'spheroidite', 'network')
clf.fit(x_p1_5, y_p1_5)
print(1-clf.score(x_sp_test, y_sp_test))
print(1-clf.score(x_nw_test, y_nw_test))
print()

#x_p1_5, y_p1_5 = get_XY(f_512b_mean, y_train, 'pearlite', 'network')
clf.fit(x_p2_5, y_p2_5)
print(1-clf.score(x_sp_test, y_sp_test))
print(1-clf.score(x_pl_test, y_pl_test))
print()

clf.fit(x_p3_5, y_p3_5)
print(1-clf.score(x_pl_test, y_pl_test))
print(1-clf.score(x_nw_test, y_nw_test))
print()





