from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout 
from keras.layers import MaxPooling2D, Lambda, Cropping2D

from data_handling import load_data, flip_augmentation
#%%
X, y = load_data(log_file='data2/driving_log.csv')
X, y = flip_augmentation(X, y)

# This is the network architecture
model = Sequential()
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: (x / 255.0)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# The model is compiled trained and saved to file
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, validation_split=0.2, shuffle=True,epochs=5)

model.save('model.h5') 
