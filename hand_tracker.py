import tensorflow as tf
import cv2
import numpy as np
from collections import deque

model_hand = tf.keras.saving.load_model("model_hand.keras")
model_xy = tf.keras.saving.load_model("model_xy.keras")
model_fist = tf.keras.saving.load_model("model_fist.keras")

WIDTH = 640
HEIGHT = 360

boxsize = 35

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

xdeque = deque(maxlen=5)
ydeque = deque(maxlen=5)

while(True):
    vid.read()
    ret, frame = vid.read()

    img = cv2.resize(frame, (160, 90), interpolation= cv2.INTER_LINEAR)
    img = img / 255.0  # normalize to 0 to 1
    img = np.transpose(img, (1, 0, 2)) # flip axes

    hand = model_hand.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    if hand > 0.5:
        coords = model_xy.predict(np.expand_dims(img, axis=0), verbose=0)[0]

        xdeque.append(min(max(coords[0], 0), 1))
        ydeque.append(min(max(coords[1], 0), 1))

        handX = np.median(xdeque)
        handY = np.median(ydeque)

        x = int(handX * 160)
        y = int(handY * 90)

        # keep box in frame
        x = min(max(x, boxsize // 2), 159 - boxsize // 2)
        y = min(max(y, boxsize // 2), 89 - boxsize // 2)

        # crop boxsize x boxsize area around x and y
        box = img[x - boxsize // 2:x + boxsize // 2 + 1, y - boxsize // 2:y + boxsize // 2 + 1]
        fist = model_fist.predict(np.expand_dims(box, axis=0), verbose=0)[0][0]
        screenX = int(handX * WIDTH)
        screenY = int(handY * HEIGHT)

        # complication: moving blurry hand is typically detected as a fist so for now
        # we do a quick solution called just move the hand slowly
        if fist > 0.5:
            frame = cv2.circle(frame, (screenX, screenY), 5, (0, 255, 0), -1)
        else:
            frame = cv2.circle(frame, (screenX, screenY), 5, (255, 0, 0), -1)
    else:
        frame = cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 5, (0, 0, 255), -1)
    
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
vid.release() 
cv2.destroyAllWindows()