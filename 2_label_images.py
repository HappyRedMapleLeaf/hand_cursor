import cv2
import os.path

WIDTH = 1280
HEIGHT = 720

stop = False

frameno = 0

def handle_mouse(event,x,y,flags,param):
    global stop, frame, frameno
    if event == cv2.EVENT_LBUTTONUP:
        with open("data.txt", "a") as f:
            f.write(f"1,{x // 8},{y // 8},0\n") # open palm
            f.close()
        next_frame()
    if event == cv2.EVENT_RBUTTONUP:
        with open("data.txt", "a") as f:
            f.write(f"1,{x // 8},{y // 8},1\n") # closed fist
            f.close()
        next_frame()

def next_frame():
    global stop, frame, frameno
    if not os.path.isfile(f"data/d{frameno}.png"):
            stop = True
            return
    frame = cv2.imread(f"data/d{frameno}.png")
    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation= cv2.INTER_LINEAR)
    cv2.imshow('frame', frame)
    frameno += 1

next_frame()

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',handle_mouse)

while(not stop):
    key = cv2.waitKey(1) & 0xFF
    if key == 32: # space
        with open("data.txt", "a") as f:
            f.write(f"0,0,0,0\n")
            f.close()
        next_frame()
    if key == ord('q'):
        break
  
cv2.destroyAllWindows() 
