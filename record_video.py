import picamera     # Importing the library for camera module
from time import sleep  # Importing sleep from time library to add delay in program

camera = picamera.PiCamera()    # Setting up the camera
camera.resolution = (640, 480) 
camera.start_preview()      # You will see a preview window while recording
camera.start_recording('/home/pi/Videos/liftarm_real-yijin.h264') # Video will be saved at desktop
camera.wait_recording(30)
camera.stop_preview()
