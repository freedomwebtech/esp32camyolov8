import urllib.request
import time
import numpy as np
import cv2

url = 'http://192.168.0.104/800x600.jpg'

# Function to adjust brightness
def adjust_brightness(image, factor=1.2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    image = cv2.imdecode(imgNp, -1)

    # Adjust brightness
    image = adjust_brightness(image)

    cv2.imshow('IP Camera Stream', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
