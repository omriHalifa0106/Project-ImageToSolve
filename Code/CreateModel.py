import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import model_from_json

sdir=r'C:\\Users\\halif\\programming\ProjectImageToSolve\\Code\\dataset'
classlist=os.listdir(sdir)
filepaths=[]
labels=[]
classes=[]

for classDir in classlist:
    classpath=os.path.join(sdir, classDir)
    if os.path.isdir(classpath):
        classes.append(classDir)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            if os.path.isfile(fpath):
                filepaths.append(fpath)
                labels.append(classDir)



fseries=pd.Series(filepaths, name='filepaths')
Lseries=pd.Series (labels, name='labels')
df=pd.concat([fseries, Lseries], axis=1)
print (df['labels'].value_counts()) # check balance of dataset - it is reasonably balanced

train_split=0.9
test_split=0.05
dummy_split=test_split/(1-train_split)
train_df, dummy_df=train_test_split(df, train_size=train_split, shuffle=True, random_state = 123)
test_df, valid_df=train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)

def scalar(img):
    return img/127.5-1 # scale pixels between -1 and + 1

gen=ImageDataGenerator(preprocessing_function=scalar)
train_gen=gen.flow_from_dataframe(train_df, x_col= 'filepaths', y_col='labels', target_size=(128,128), class_mode='categorical',
                                  color_mode='rgb', shuffle=False)#,validate_filenames=False)
test_gen=gen.flow_from_dataframe(test_df, x_col= 'filepaths', y_col='labels', target_size=(128,128), class_mode='categorical',
                                  color_mode='rgb', shuffle=False)#,validate_filenames=False)
valid_gen=gen.flow_from_dataframe(valid_df, x_col= 'filepaths', y_col='labels', target_size=(128,128), class_mode='categorical',
                                  color_mode='rgb', shuffle=False)#,validate_filenames=False)




base_model=tf.keras.applications.MobileNetV2( include_top=False, input_shape=(128,128,3), pooling='max', weights='imagenet')
x=base_model.output
x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
x = Dense(1024, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006) ,activation='relu', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=123))(x)
x=Dropout(rate=.3, seed=123)(x)
output=Dense(len(classes), activation='softmax',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
model=Model(inputs=base_model.input, outputs=output)


model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
estop=tf.keras.callbacks.EarlyStopping( monitor="val_loss",  patience=4, verbose=1,restore_best_weights=True)
rlronp=tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss",factor=0.5, patience=1, verbose=1)
history=model.fit(x=train_gen,  epochs=12, verbose=1, callbacks=[estop, rlronp],  validation_data=valid_gen,
               validation_steps=None,  shuffle=False,  initial_epoch=0)


model.summary()
acc=model.evaluate (test_gen, verbose=1)[1]*100
print ('Model Accuracy on Test Set: ', acc)
preds=model.predict(test_gen)
y_pred=[]
y_true=[]
for i, p in enumerate(preds):
    y_pred.append(np.argmax(p))
    y_true.append(test_gen.labels[i])
y_pred=np.array(y_pred)
y_true=np.array(y_true)

cm = confusion_matrix(y_true, y_pred )
length=len(classes)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
plt.xticks(np.arange(length)+.5, classes, rotation= 90)
plt.yticks(np.arange(length)+.5, classes, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# plot accuracy and loss
def plotgraph(epochs, accuracy, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, accuracy, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
# Accuracy curve
plotgraph(epochs, acc, val_acc)

# serialize weights to HDF5
model.save_weights("modelMathSolve.h5")
print("Saved model to disk")