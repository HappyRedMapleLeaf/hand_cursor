import cv2

WIDTH = 640
HEIGHT = 360

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

ret, frame = vid.read()
cv2.imshow('frame', frame)

defx = WIDTH // 2
defy = HEIGHT // 2

frameno = 0

while(True):
    vid.read()
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        save = cv2.resize(frame, (WIDTH // 4, HEIGHT // 4), interpolation= cv2.INTER_LINEAR)
        cv2.imwrite(f"data/d{frameno}.png", save)
        frameno += 1
        pass
    elif key == ord('q'):
        break
  
vid.release() 
cv2.destroyAllWindows() 
