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

for i in range(len(train_inputs_fist)):
    # draw image to screen
    img = np.copy(train_inputs_fist[i])
    img = np.transpose(img, (1, 0, 2))
    img = (img * 255).astype('uint8')
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    