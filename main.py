import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
from picamera.array import PiRGBArray
from picamera import PiCamera
import matplotlib.pyplot as plt


#Mở camera
camera = PiCamera()
#Cài đặt kích thước cửa sổ
camera.resolution = (640, 480)
#Tốc độ đọc camera
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(640, 480))


# erosion
#def erode(img):
#   kernel = np.ones((5, 5), np.uint8)
#    return cv2.erode(img, kernel, iterations=1)
# opening -- erosion followed by a dilation
#def opening(img):
#    kernel = np.ones((5, 5), np.uint8)
#    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#vòng lặp lấy giữ liệu ảnh từ camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        #cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        #Nhấn 's' trên bàn phím sẽ lấy khung hình cuối cùng
        if key == ord("s"):
             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #chuyển ảnh sang hệ màu xám
             gray = cv2.bilateralFilter(gray, 11, 17, 17) #Lọc nhiễu
             #$gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
             edged = cv2.Canny(gray, 30, 200) #Nhận biết cạnh
             cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#Tìm đường viền
             cnts = imutils.grab_contours(cnts)
             cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]#Sắp xếp Contours từ trái sang phải
             screenCnt = None
             for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 4:
                  screenCnt = approx
                  break
             if screenCnt is None:
               detected = 0
               print ("No contour detected")
             else:
               detected = 1
             if detected == 1:
               cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
             mask = np.zeros(gray.shape,np.uint8)#tạo mặt lạ
             new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
             new_image = cv2.bitwise_and(image,image,mask=mask)#loại bỏ phần thừa
             (x, y) = np.where(mask == 255)
             (topx, topy) = (np.min(x), np.min(y))#Cắt ảnh tách biển số
             (bottomx, bottomy) = (np.max(x), np.max(y))
             Cropped = gray[topx:bottomx+1, topy:bottomy+1]
             Cropped = cv2.threshold(Cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]#Lọc ảnh sau khi cắt
             Cropped = cv2.medianBlur(Cropped,5)
             kernel = np.ones((5,5),np.float32)/10;
             Cropped = cv2.filter2D(Cropped,-1,kernel)#làm mịn ảnh
            # Cropped = cv2.medianBlur(Cropped,5)
            # Cropped = cv2.bilateralFilter(Cropped,9,75,75) #cv2.threshold(Cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
             Cropped = cv2.resize(Cropped, (0, 0), fx = 1.6, fy = 1.6)#phóng to ảnh 
             #Cropped = cv2.opening(Cropped) 
             #custom_config =  '--psm 7 --oem 3 -c tessedit_char_whitelist=/1234'
             text = pytesseract.image_to_string(Cropped,config='-l eng --oem 3 --psm 11' ) #đọc dữ liệu string từ image
             print("Detected Number is:",text)#in biển số xe ra màn hình
            
             cv2.imshow("Frame", image)
             cv2.imshow('Cropped',Cropped)
             cv2.waitKey(0)
             break
cv2.destroyAllWindows()
