# Import necessary libraries
import tkinter as tk # For GUI elements
import os  # For file operations
import csv # To handle CSV file operations
import cv2 # OpenCV for image processing
import numpy as np # For numerical computations
from PIL import Image # For image processing
import pandas as pd # To handle dataframes and attendance records
import datetime # For date and time operations
import time # For time operations
from tkinter import messagebox # For GUI message boxes
import unicodedata  # For validating numeric input


# Initialize the main window
window = tk.Tk() # Create the main Tkinter window
window.title("Attendance System") # The title of the window
window.configure(background='black') # the background color of the window 

 # Set the weight of the columns
window.columnconfigure(0, weight=1) 
window.columnconfigure(1, weight=1) 
window.columnconfigure(2, weight=1) 


# Create and place the attendance portal title label
label_portal_name = tk.Label(window, text="BATH SPA UNIVERSITY ATTENDANCE  PORTAL", bg="black", fg="white", width=40, height=1, font=('Times New Roman', 35, 'bold underline'))
label_portal_name.grid(row=0, column=0, columnspan=3, pady=20)

# University ID label and entry
university_id_label = tk.Label(window, text="Enter Your University ID", width=20, height=2, fg="white", bg="black", font=('Times New Roman', 25, 'bold'))
university_id_label.grid(row=1, column=0, padx=20, pady=10)

university_id_entry = tk.Entry(window, width=30, bg="white", fg="red", font=('Times New Roman', 15, 'bold'))
university_id_entry.grid(row=2, column=0, padx=20, pady=10)

# Student name label and entry
name_title = tk.Label(window, text="Enter Your Name", width=20, fg="white", bg="black", height=2, font=('Times New Roman', 25, 'bold'))
name_title.grid(row=1, column=1, padx=20, pady=10)

Student_name_entry= tk.Entry(window, width=30, bg="white", fg="blue", font=('Times New Roman', 15, 'bold'))
Student_name_entry.grid(row=2, column=1, padx=20, pady=10)

# Notification label and message
notification_title = tk.Label(window, text="Notification", width=20, fg="white", bg="black", height=2, font=('Times New Roman', 25, 'bold'))
notification_title.grid(row=1, column=2, padx=20, pady=10)

notification_message_title = tk.Label(window, text="", bg="white", fg="blue", width=30, height=1, activebackground="white", font=('Times New Roman', 15, 'bold'))
notification_message_title.grid(row=2, column=2, padx=20, pady=10)

# Attendance label and message
attendance_title = tk.Label(window, text="Attendance", width=10, fg="white", bg="green", height=2, font=('Times New Roman', 30, 'bold'))
attendance_title.grid(row=3, column=0, columnspan=3, pady=20)

attendance_message_label = tk.Label(window, text="", fg="green", bg="black", activeforeground="green", width=60, height=4, font=('times', 15, 'bold'), highlightthickness=2, highlightbackground="white")
attendance_message_label.grid(row=4, column=0, columnspan=3, pady=20)
 
 # Function to clear the university ID entry field
def clearUniversityIdEntry():
    university_id_entry.delete(0, 'end') # Clear the content of the university ID entry field
    notifi_result = "" # Clear any notification message
    notification_message_title.configure(text= notifi_result)

# Function to clear the student name entry field
def clear_Entry_Name():
    Student_name_entry.delete(0, 'end') # Clear the content of the student name entry field
    notifi_result = "" # Clear any notification message
    notification_message_title.configure(text= notifi_result)    
    


 # Function to validate if a string is numeric   
def id_numeric(s):
    try:
        float(s) # Try converting the string to a float
        return True
    except ValueError:
        pass
 
    try:
        unicodedata.numeric(s) # Check if the string is a numeric character
        return True
    except (TypeError, ValueError):
        pass
 
    return False # Return False if neither condition is met

 
# Function to capture images of the students
def TakePictures():        
    Student_id=(university_id_entry.get()) # Get the university ID entered by the student
    student_name=(Student_name_entry.get()) # Get the name entered by the student

    # Validate inputs
    if not Student_id:
        notifi_result="Please enter ID" # Display a notification message if the university ID is not entered
        notification_message_title.configure(text = notifi_result)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter your ID number properly , press yes if you understood",icon = 'warning')
        
    elif not student_name:
        notifi_result ="Please enter Name" # Display a notification message if the student name is not entered
        notification_message_title.configure(text = notifi_result)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter your name properly , press yes if you understood",icon = 'warning')
        
        
    elif(id_numeric(Student_id) and student_name.isalpha()):
            camera = cv2.VideoCapture(0) # Open the camera
            harcascade_Path = "haarcascade_frontalface_default.xml"  # Path to the Haar Cascade face detection model
            face_recognizer=cv2.CascadeClassifier(harcascade_Path) # Load the Haar Cascade face detection model
            sample_Number=0 # Initialize the sample number to 0

            while(True):
                ret, image_frame_captured = camera.read() # Capture the image from the camera
                gray = cv2.cvtColor(image_frame_captured, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
                faces = face_recognizer.detectMultiScale(gray, 1.3, 5) # Detect faces in the image

                # Process each detected face
                for (x,y,w,h) in faces:
                    cv2.rectangle(image_frame_captured,(x,y),(x+w,y+h),(255,0,0),2) 

                    #incrementing sample number 
                    sample_Number=sample_Number+1

                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ "+student_name +"."+Student_id +'.'+ str(sample_Number) + ".jpg", gray[y:y+h,x:x+w])

                    #display the frame
                    cv2.imshow('frame',image_frame_captured)

                # Stop if 'q' is pressed or 60 samples are captured
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
    
                # break if the sample number is morethan 100
                elif sample_Number>60:
                    break
            camera.release() # Release the camera
            cv2.destroyAllWindows() # Close OpenCV windows
            notifi_result = "Images Saved for ID : " + Student_id +" Name : "+ student_name
            row = [Student_id , student_name]

            # Save student details to a CSV file
            with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            notification_message_title.configure(text= notifi_result)

      # Handle invalid inputs        
    else:
        if(id_numeric(Student_id)):
            notifi_result = "Enter Alphabetical Name"
            notification_message_title.configure(text= notifi_result)
        if(student_name.isalpha()):
            notifi_result = "Enter Numeric Id"
            notification_message_title.configure(text= notifi_result)
            
 # Function to train the face recognition model using captured images   
def TrainPictures():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Initialize the LBPH face recognizer
    faces,Student_id = getImagesAndLabels("TrainingImage") # Get face data and corresponding IDs
    recognizer.train(faces, np.array(Student_id)) # Train the model using faces and IDs
    recognizer.save("TrainingImageLabel\Trainner.yml") # Save the trained model to a file
    
    # Notify the user that training is complete
    notifi_result = f"Image Trained"
    clearUniversityIdEntry() # Clear the university ID field
    clear_Entry_Name() # Clear the name field
    notification_message_title.configure(text=notifi_result) # Display the notification message
    tk.messagebox.showinfo('Completed', f'The model has been trained successfully.')
    
# Function to load images and their corresponding labels
def getImagesAndLabels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]  # Get paths of all images in the directory
    
    faces=[] # List to store face images

    Ids=[] # List to store IDs corresponding to the faces

    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        # Converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        # Getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id) # Append the ID to the Ids list       
    return faces,Ids # Return the faces and Ids lists


# Function to track faces and mark attendance
def TrackPictures():
    recognizer = cv2.face.LBPHFaceRecognizer_create() # Initialize the LBPH face recognizer
    recognizer.read("TrainingImageLabel\Trainner.yml") # Load the trained model
    harcascadePath = "haarcascade_frontalface_default.xml" # Path to the Haar Cascade face detection model
    faceCascade = cv2.CascadeClassifier(harcascadePath);  # Load the face detection model    
    df=pd.read_csv("StudentDetails\StudentDetails.csv") # Load the student details from the CSV file
    camera = cv2.VideoCapture(0) # Open the camera
    font = cv2.FONT_HERSHEY_SIMPLEX # Define the font for text        
    col_names =  ['Id','Name','Date','Time'] # Define column names for the attendance data
    attendanceData = pd.DataFrame(columns = col_names) # Create a dataframe to store attendance data


    while True:
        ret, im =camera.read() # Read the image from the camera
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        faces=faceCascade.detectMultiScale(gray, 1.2,5) # Detect faces in the image  

        # Process each detected face
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2) # Draw a rectangle around the face
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w]) # Predict the ID of the face  

            if(conf < 50): # If the confidence is less than 50 (higher accuracy), mark the attendance
                ts = time.time() # Get the current timestamp      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') # Get the current date
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S') # Get the current time
                aa=df.loc[df['Id'] == Id]['Name'].values # Get the name corresponding to the ID
                tt=str(Id)+"-"+aa
                attendanceData.loc[len(attendanceData)] = [Id,aa,date,timeStamp] # Add the attendance record to the dataframe

        
                
            else:
                 # Handle unknown faces
                Id='Unknown'                
                tt=str(Id) 

            # If the confidence is greater than 75, save the image of the unknown face 
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1 
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendanceData=attendanceData.drop_duplicates(subset=['Id'],keep='first') # Remove duplicate entries from the attendance data  
        cv2.imshow('im',im) # Display the frame
        if (cv2.waitKey(1)==ord('q')): # Exit if 'q' is pressed
            break
    # Save the attendance data to a CSV file    
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendanceData.to_csv(fileName,index=False)

    camera.release() # Release the camera
    cv2.destroyAllWindows() # Close OpenCV windows

    # Notify the user that attendance has been marked
    notifi_result=attendanceData
    attendance_message_label.configure(text= notifi_result)
    notifi_result = "Attendance Taken"
    notification_message_title.configure(text= notifi_result)
    tk.messagebox.showinfo('Completed','Congratulations ! Your attendance has been successfully marked for the day!!')

# Function to quit the application   
def quit_window():
    MsgBox = tk.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application?', icon='warning')
    if MsgBox == 'yes':
        tk.messagebox.showinfo("Hello", "Thank You. Have a nice day!!")
        window.destroy() # Close the window


# Create buttons for capturing images, training the model, and marking attendance   
takePic = tk.Button(window, text="Capture Image", command=TakePictures, fg="white", bg="blue", width=25, height=2, activebackground="white", font=('Times New Roman', 15, 'bold'))
takePic.grid(row=5, column=0, padx=20, pady=20)

trainPic = tk.Button(window, text="Train Model", command=TrainPictures, fg="white", bg="blue", width=25, height=2, activebackground="white", font=('Times New Roman', 15, 'bold'))
trainPic.grid(row=5, column=1, padx=20, pady=20)

trackPic = tk.Button(window, text="Mark Attendance", command=TrackPictures, fg="white", bg="red", width=25, height=2, activebackground="white", font=('Times New Roman', 15, 'bold'))
trackPic.grid(row=5, column=2, padx=20, pady=20)

quitWindow = tk.Button(window, text="QUIT", command=quit_window, fg="white", bg="red", width=10, height=2, activebackground="pink", font=('Times New Roman', 15, 'bold'))
quitWindow.grid(row=6, column=1, pady=20)
 
 # Run the main window
window.mainloop()