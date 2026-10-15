import cv2

capture = cv2.VideoCapture("Vehicle_Recognision/cars.mp4")
mlmodel = cv2.CascadeClassifier("Vehicle_Recognision/cars.xml")

while True:
    returnvalue, frames = capture.read()
    #this captures the frames
    grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    #this changes the frames to grey
    cars = mlmodel.detectMultiScale(grey, 1.1, 1)
    #this will detect any size vehicle. the 1.1 is the radius of the circle.
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 5)
        #this frames cars in rectangles when they are found. Frame is the video, x and y are the coordinates, x+w and y+h are the size of the rectangle, the 0, 0, 255 is the color, and 5 is the width of the rectangle.
    cv2.imshow("Vechicle detection", frames)
    key = cv2.waitKey(10)
    if key == 27:
    #27 is the espcape key
        break
cv2.destroyAllWindows()