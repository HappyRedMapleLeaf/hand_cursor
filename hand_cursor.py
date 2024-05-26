import tensorflow as tf
import cv2
import numpy as np
from collections import deque
import win32api, win32con

down = False

def move(mx,my):
    x = mx * 1.1 - 0.05
    y = my * 1.2 - 0.1
    x = min(max(x, 0), 1)
    y = min(max(y, 0), 1)
    x = -x + 1

    # https://stackoverflow.com/questions/1181464/controlling-mouse-with-python
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(x*65535.0), int(y*65535.0))
    # win32api.SetCursorPos((int(x * 1920), int(y * 1080)))

def click():
    global down
    if not down:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
        down = True

def unclick():
    global down
    if down:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
        down = False

clicked_history = deque(maxlen=3)

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
        move(handX,handY)

        clicked_history.append(round(fist))
        if sum(clicked_history) == clicked_history.maxlen:
            click()
        elif sum(clicked_history) == 0:
            unclick()

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