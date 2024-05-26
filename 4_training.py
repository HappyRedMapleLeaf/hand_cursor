import tensorflow as tf
import cv2
import numpy as np
import random

with open("data.txt", "r") as f:
    data = f.readlines()
    f.close()

boxsize = 35
    
train_outputs_hand = [] # whether or not hand is there
train_inputs_hand = []

train_outputs_xy = [] #position of hand if it is there
train_inputs_xy = []

train_outputs_fist = [] #if palm or fist
train_inputs_fist = []

i = 0
for line in data:
    line = line.strip()
    line = line.split(",")

    img = cv2.imread(f"data/d{i}.png")
    img = img / 255.0  # normalize to 0 to 1
    img = np.transpose(img, (1, 0, 2)) # flip axes

    if line[0] == "1":
        # 1/6 chance to add the image to the hand detection to balance the dataset
        if random.randint(0, 5) == 0:
            train_inputs_hand.append(img)
            train_outputs_hand.append(int(line[0]))
        
        train_inputs_xy.append(img)
        x = int(line[1])
        y = int(line[2])
        train_outputs_xy.append([x / 160.0, y / 90.0])

        # add randomness into center of hand box
        x += random.randint(-10, 10)
        y += random.randint(-10, 10)
        x = min(max(x, boxsize // 2), 159 - boxsize // 2)
        y = min(max(y, boxsize // 2), 89 - boxsize // 2)

        # crop boxsize x boxsize area around x and y
        box = img[x - boxsize // 2:x + boxsize // 2 + 1, y - boxsize // 2:y + boxsize // 2 + 1]

        train_inputs_fist.append(box)
        train_outputs_fist.append(int(line[3]))
    else:
        train_inputs_hand.append(img)
        train_outputs_hand.append(int(line[0]))
    i += 1




# IF THERE IS A HAND
# Shuffle the dataset and split it into batches
dataset_hand = tf.data.Dataset.from_tensor_slices((train_inputs_hand, train_outputs_hand))
dataset_hand = dataset_hand.shuffle(len(train_inputs_hand)).batch(24)

model_hand = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 90, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_hand.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
model_hand.fit(dataset_hand, epochs=20)
model_hand.save("model_hand.keras")




# WHERE IS THE HAND
dataset_xy = tf.data.Dataset.from_tensor_slices((train_inputs_xy, train_outputs_xy))
dataset_xy = dataset_xy.shuffle(len(train_inputs_xy)).batch(24)

model_xy = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 90, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model_xy.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
model_xy.fit(dataset_xy, epochs=25)
model_xy.save("model_xy.keras")




# IS IT A FIST OR A PALM
dataset_fist = tf.data.Dataset.from_tensor_slices((train_inputs_fist, train_outputs_fist))
dataset_fist = dataset_fist.shuffle(len(train_inputs_fist)).batch(24)

model_fist = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(boxsize, boxsize, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_fist.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
model_fist.fit(dataset_fist, epochs=25)
model_fist.save("model_fist.keras")