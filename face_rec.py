import cv2
import sys

'''define face_cascade variable'''
'''load the classifier,add the path of cascade classifier which does classification'''
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

'''frame for capturing frame by frame when video capture is on'''
while True:
    '''capture frame by frame'''

    retval,frame = video_capture.read()

    '''convert the captured image to grayscale'''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''detects specific features specified in the haarcascade by using these 3 parameters'''

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)

    )

    '''for each face we draw the rectangle for some height and width'''
    # Draw a rectangle around recognized faces and color for the box is specified as (50,50,200)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 200), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the camera view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()




