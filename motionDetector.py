import cv2, time, numpy


firstFrame = None

#Begin recording using primary webcam
try:
    video = cv2.VideoCapture(0)
except :
    print("Please connect a working webcam to your computer.")

while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)

    #Set firstFrame to the first frame the camera records
    if firstFrame is None:
        firstFrame = gray
        continue

    #DeltaFrame serves as a comparison frame to base future frame differences on
    deltaFrame = cv2.absdiff(firstFrame, gray)

    #Identify pixels that are different than the pixels in deltaFrame
    threshDeltaFrame = cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY )[1]
    threshDeltaFrame = cv2.dilate(threshDeltaFrame, None, iterations=2)

    #Find differences in windows and draw square that identifies frame movements
    (__,cnts,__) = cv2.findContours(threshDeltaFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)



    #cv2.imshow("Capturing", frame)
    #cv2.imshow("Delta", deltaFrame)
    cv2.imshow("Thresh Delta", threshDeltaFrame)
    #cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1)

#Quit image show window when q key is pressed
    if key==ord('q'):
        break

#Stop recording and destroy open windows
video.release()
cv2.destroyAllWindows()
