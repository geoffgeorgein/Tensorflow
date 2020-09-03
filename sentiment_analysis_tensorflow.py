#!/usr/bin/env python
# coding: utf-8



from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import LambdaCallback
import numpy as np

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)


class_names=['negative','positive']


word_index=imdb.get_word_index()

print(word_index['me'])

reverse_word_index=dict((value,key) for key,value in word_index.items())


def decode(review):
    text=" "
    for i in review:
        text=text+reverse_word_index[i]
        text+=" "
    return text



print(len(x_train[0])) #different reviews have different length
print(len(x_train[6]))


#padding with the since it doesn't carry any information
x_train=pad_sequences(x_train,value=word_index['the'],padding='post',maxlen=256)
x_test=pad_sequences(x_test,value=word_index['the'],padding='post',maxlen=256)

print(len(x_train[0]))
print(len(x_test[6]))



decode(x_train[0])


model=Sequential([Embedding(10000,16),  #Since it has 16 features
                 GlobalAveragePooling1D(),
                 Dense(16,activation='relu'),
                 Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.summary()


simple_log=LambdaCallback(on_epoch_end=lambda e,l:print(e,end='.'))

E=20 #no. of epochs

h=model.fit(x_train,y_train,
           validation_split= 0.2,
           epochs=E,
           callbacks=[simple_log],
           verbose=False)


# # Predictions and Evaluation

#plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(range(E),h.history['acc'],label='Training')
plt.plot(range(E),h.history['val_acc'],label='Validation')
plt.legend()
plt.show()


loss,acc=model.evaluate(x_test,y_test)
print('Test accuracy',acc*100)


p=model.predict(np.expand_dims(x_test[5],axis=5))
print(class_names[np.argmax(p[5])])



decode(x_test[5])
