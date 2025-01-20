from teachable_machine import TeachableMachine

from tkinter import *
from PIL import Image, ImageTk
import cv2

my_model = TeachableMachine(model_path='keras_model.h5', model_type='h5')


# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("600x350")
def show_frames():
   cap= cv2.VideoCapture(0)# capture frame from webcam
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   cv2.imwrite("result.jpg",cv2image)#writes image in currentfolder
   img = Image.fromarray(cv2image)
   img = img.resize(size= (250,200))
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   cap.release()
   result = my_model.classify_image("result.jpg")
   print('highest_class_id:', result['highest_class_id'])
   print('all_predictions:', result['all_predictions'])
# Create a Label to capture the Video frames
img = ImageTk.PhotoImage(Image.open("empty.jpg"))
label =Label(win,image = img)
label.grid(row=0, column=0)
button = Button(win,text = "Capture",command = show_frames)
button.grid(row = 0, column = 1)

# Define function to show frame

 
win.mainloop()
