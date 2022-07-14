#import libraries
# ----------------------------------------------------------
from fileinput import filename
import os
from os.path import dirname, join
from pathlib import Path
from colored import fg
from tkinter.font import BOLD
import cv2
import json
import math
import numpy as np
import tkinter as tk
from tkinter.messagebox import askyesno, askquestion
from tkinter import messagebox
from tkinter import filedialog
from tkinter.constants import INSERT
# ----------------------------------------------------------

# //TODO: Defined the Sub class
# ----------------------------------------------------------
class Window(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        # //* sets the title of the Toplevel widget
        self.title("Results")

        # //* sets the geometry of toplevel
        self.geometry("400x450")

        # //*Resizing widgets when expanding the window
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # //* Button for closing
        self.button1 = tk.Button(
            self, text="Exit", width=5, command=self.destroy)
        self.button1.grid(row=2, column=1, pady=20)

# ----------------------------------------------------------

# //TODO: Defined the Main class
# ----------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.File = None
        # //* configure the root window
        self.geometry("365x195")
        self.title('Body size')
        self.config(background="#f2f2f2")

        # //*Resizing widgets when expanding the window
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # //* Label widgets
        self.label1 = tk.Label(self, text="Upload your file", fg="blue")
        self.label1.grid(row=0, column=0, padx=10, pady=20, sticky='w')
        
        self.label2 = tk.Label(self, text="Gender")
        self.label2.grid(row=1, column=0, padx=10, pady=20, sticky='w')
        
        self.label3 = tk.Label(self, text="Enter your height (cm)")
        self.label3.grid(row=2, column=0, padx=10, pady=20, sticky='w')
        
        # //* Button widgets
        self.button1 = tk.Button(self, text="Browse Files", command=self.upload, activeforeground='red')
        self.button1.grid(row=0, column=1, padx=10, pady=20)
        
        self.button2 = tk.Button(self, text="Exit", command=self.exit, activeforeground='red')
        self.button2.grid(row=0, column=2, padx=10, pady=20)
        
        self.button3 = tk.Button(self, text="Predict",command=self.predict, activeforeground='red')
        self.button3.grid(row=2, column=2, padx=10, pady=20)

        # //* Entry widgets
        self.entry1 = tk.Entry(self)
        self.entry1.grid(row=1, column=1, padx=20, pady=20)
        
        self.entry2 = tk.Entry(self)
        self.entry2.grid(row=2, column=1, padx=20, pady=20)
        
        # //* Initiliaze the variable
        self.filename_var = str()
        
# ----------------------------------------------------------

    # //TODO: Defined all the necessaries functions
# ----------------------------------------------------------

    # //* Exit function that closes the program

    def exit(self):
        answer = askyesno(title='Confirmation',
                          message='Are you sure that you want to quit?')
        if answer == True:
            self.destroy()
            
    # //* Browse function to open files in the file explorer window

    def upload(self):
        self.filename_var = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=(("JPEG files", "*.jpeg*"),
                                                                                                ("JPG files", "*.jpg*"),
                                                                                                ("PNG files", "*.png*")))
        if self.filename_var != '':
           self.label1.configure(text=os.path.basename(self.filename_var))


    # //* Function to predict the body size dimension

    def predict(self):
        # //* join and dirname methods allow us to read the file
        protoPath = join(dirname(__file__), "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        modelPath = join(dirname(__file__), "pose/mpi/pose_iter_160000.caffemodel")

        bodyNet = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        
        filename = self.filename_var
        if filename == '':
            messagebox.showwarning('warning', 'Please upload your file!')
        else:
            frame = cv2.imread(filename)
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            edge = cv2.Canny(blurred, 100, 200)

            # //* Specify the input image dimensions
            inWidth = 368
            inHeight = 368

            # //* Prepare the frame to be fed to the network
            inpBlob = cv2.dnn.blobFromImage(
                frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            # //* Set the prepared object as the input blob of the network
            bodyNet.setInput(inpBlob)
            output = bodyNet.forward()

            H = output.shape[2]
            W = output.shape[3]

            # //* Empty list to store the detected keypoints
            frameHeight, frameWidth, chanel = frame.shape
            points = []
            for i in range(15):
                # //* confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # //* Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # //* Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H
                threshold = 0
                if prob > threshold:
                    # //* Add the point to the list if the probability is greater than the threshold
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            body = {}
            body['Head'] = points[0]
            body['Neck'] = points[1]
            body['Right Shoulder'] = points[2]
            body['Right Elbow'] = points[3]
            body['Right Wrist'] = points[4]
            body['Left Shoulder'] = points[5]
            body['Left Elbow'] = points[6]
            body['Left Wrist'] = points[7]
            body['Right Hip'] = points[8]
            body['Right Knee'] = points[9]
            body['Right Ankle'] = points[10]
            body['Left Hip'] = points[11]
            body['Left Knee'] = points[12]
            body['Left Ankle'] = points[13]
            body['Chest'] = points[14]

            def measurement(a, b):
                return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

            def distance(a, b, x):
                # return (a/b*x)
                return (a/b*x)*2.54

            high = body['Left Ankle'][1] - body['Head'][1]
            head = measurement(body['Head'], body['Neck'])
            shoulder = measurement(body['Right Shoulder'], body['Left Shoulder'])
            right_elbow = measurement(body['Right Shoulder'], body['Right Elbow'])
            left_elbow = measurement(body['Left Shoulder'], body['Left Elbow'])
            right_wrist = measurement(body['Right Shoulder'], body['Right Elbow']) + \
                measurement(body['Right Elbow'], body['Right Wrist'])
            left_wirst = measurement(body['Left Shoulder'], body['Left Elbow']) + \
                measurement(body['Left Elbow'], body['Left Wrist'])
            hip = measurement(body['Right Hip'], body['Left Hip'])
            right_knee = measurement(body['Right Hip'], body['Right Knee'])
            left_knee = measurement(body['Left Hip'], body['Left Knee'])
            right_ankle = measurement(body['Right Hip'], body['Right Knee']) + \
                measurement(body['Right Knee'], body['Right Ankle'])
            left_ankle = measurement(body['Left Hip'], body['Left Knee']) + \
                measurement(body['Left Knee'], body['Left Ankle'])
            body['Right Hip'] = [body['Right Hip']
                                [0] - hip/2, body['Right Hip'][1]]
            body['Left Hip'] = [body['Left Hip'][0] + hip/2, body['Left Hip'][1]]
            points[8] = body['Right Hip']
            points[11] = body['Left Hip']
            chest = body['Neck'][1] + (body['Chest'][1]-body['Neck'][1])/2
            chest_right = [body['Right Shoulder'][0] +
                        (body['Right Hip'][0]-body['Right Shoulder'][0])/2, chest]
            chest_left = [body['Left Hip'][0] +
                        (body['Left Shoulder'][0]-body['Left Hip'][0])/2, chest]
            waist = body['Chest'][1] + (body['Right Hip'][1]-body['Chest'][1])/2
            waist_right = [int(body['Right Hip'][0]+(body['Chest']
                                                    [0]-body['Right Hip'][0])/2), int(waist)]
            waist_left = [int(body['Chest'][0]+(body['Left Hip']
                                                [0]-body['Chest'][0])/2), int(waist)]
            # print(np.sum(edge[176, 300:310]))

            while(np.sum(edge[waist_right[0], waist_right[1]-10:waist_right[1]+10]) == 0):
                waist_right[0] -= 1

            points.append(waist_right)
            points.append(chest_left)
            points.append(waist_left)
            points.append(waist_left)
            # print(waist_left, waist_left)
            hip = measurement(body['Right Hip'], body['Left Hip'])
            chest = measurement(chest_right, chest_left)
            waist = measurement(waist_right, waist_left)

            l1 = range(19)
            l = [0, 1, 3, 5, 6, 7, 9, 10, 15]
            for i in l1:
                cv2.circle(frame, (int(points[i][0]), int(
                    points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(
                    points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            # //* Different functions that show a warning
            
            # //* Conditions that show warnings or validate the entry
            gender = self.entry1.get()
            high_cm = self.entry2.get()
            try:
                gender = self.entry1.get()
                high_cm = self.entry2.get()
                if len(gender) == 0 or len(gender) == 0:
                    messagebox.showwarning('warning', 'Field(s) cannot be empty! Please fill the field(s).')
                elif int(high_cm) < 30 or int(high_cm) > 230:
                    messagebox.showwarning('warning', 'The height should be between 30 to 230 cm.\nPlease try again.')
                elif gender == "female" or gender == "male":
                    if gender == "female":
                        # //* Open a new window
                        window = Window(self)

                        # //* Routes enteries values in the window
                        window.grab_set()

                        fact_high = int(high_cm) - 5

                        # //* Create a TextBox
                        self.TextBox = tk.Text(window, width=56, font=("BOLD", 13))
                        self.TextBox.grid(row=1, column=1, padx=20,
                                        pady=20, ipady=10, sticky="NSEW")

                        # //* Insert values in the TextBox
                        my_height = 'Height: '+str(high_cm)+' cm\n'
                        self.TextBox.insert(INSERT, my_height)

                        chest_size = 'Chest size: ' + \
                            str(distance(chest, high, fact_high))+' cm\n'
                        self.TextBox.insert(INSERT, chest_size)

                        hip_size = 'Hip size: ' + \
                            str(distance(hip, high, fact_high))+' cm\n'
                        self.TextBox.insert(INSERT, hip_size)

                        waist_size = 'Waist size: ' + \
                            str(distance(waist, high, fact_high))+' cm\n'
                        self.TextBox.insert(INSERT, waist_size)

                        width = round(frameWidth/frameHeight*480)
                        height = 480
                        dim = (width, height)
                        resized = cv2.resize(
                            frame, dim, interpolation=cv2.INTER_AREA)

                        # //* Break line(new line)
                        newLine = '\n'
                        self.TextBox.insert(INSERT, newLine)

                        # //* Body size list
                        chestString = str(distance(chest, high, fact_high))
                        chestFloat = list(map(float, chestString.split(', ')))
                        chestInt = list(map(int, chestFloat))

                        hipString = str(distance(hip, high, fact_high))
                        hipFloat = list(map(float, hipString.split(', ')))
                        hipInt = list(map(int, hipFloat))

                        waistString = str(distance(waist, high, fact_high))
                        waistFloat = list(map(float, waistString.split(', ')))
                        waistInt = list(map(int, waistFloat))

                        # //* Clothing size for women
                        # //* Jackets, coats, blouses, polos and sweats
                        gender_female = "You are a woman and your sizes are the following: " + '\n'
                        self.TextBox.insert(INSERT, gender_female)
                        womenTop1 = {
                            '36' : [82, 83, 84, 85, 86],
                            '38' : [86, 87, 88, 89, 90],
                            '40' : [90, 91, 91, 92, 93, 94],
                            '42' : [94, 95, 96, 97, 98],
                            '44' : [98, 99, 100, 101, 102],
                            '46' : [102, 103, 104, 105, 106, 107],
                            '48' : [107, 108, 109, 110, 111, 112, 113],
                            '50' : [113, 114, 115, 116, 117, 118, 119]
                        }
                        womenTop1_List = [key for ele in chestInt
                                        for key, val in womenTop1.items() if ele in val]
                        womenTop1_Size = "Jackets, coats, blouses, polos and sweats = " + \
                            '-'.join(map(str, womenTop1_List)) + '\n'
                        self.TextBox.insert(INSERT, womenTop1_Size)

                        # //* T-shirts, pulls and vests
                        womenTop2 = {
                            'S' : [82-88],
                            'M' : [88-94],
                            'L' : [94-102],
                            'XL' : [102, 103, 104, 105, 106, 107, 108, 109, 110],
                            '2XL' : [110, 111, 112, 113, 114, 115, 116, 117, 118]
                        }
                        womenTop2_List = [key for ele in chestInt
                                        for key, val in womenTop2.items() if ele in val]
                        womenTop2_Size = "T-shirts, pulls and vests = " + \
                            '-'.join(map(str, womenTop2_List)) + '\n'
                        self.TextBox.insert(INSERT, womenTop2_Size)

                        # //* Trousers, jeans, shorts and bermuda shorts
                        womenUnderwear = {
                            '38' : [75, 76, 77, 78, 79],
                            '40' : [79, 80, 81, 82, 83],
                            '42' : [83, 84, 85, 86, 87],
                            '44' : [87, 88, 89, 90, 91],
                            '46' : [91, 92, 93, 94, 95],
                            '48' : [95, 96, 97, 98, 99, 100, 101],
                            '50' : [101, 102, 103, 104, 105],
                            '52' : [105, 106, 107, 108, 109]
                        }
                        womenUnderwear_List = [key for ele in waistInt and hipInt
                                            for key, val in womenUnderwear.items() if ele in val]
                        womenUnderwear_Size = "Trousers, jeans, shorts and bermuda shorts = " + \
                            '-'.join(map(str, womenUnderwear_List))
                        self.TextBox.insert(INSERT, womenUnderwear_Size)
                            
                        # //* Lists directory contents before saving new images
                        Green = fg('green')
                        White = fg('white')
                        filedirectory = (
                            r"C:/Users/Salem/Desktop/Body-App/images/results")
                        os.chdir(filedirectory)
                        print()
                        print(White + "List of files available before saving function is executed:")
                        print(Green + str(os.listdir(filedirectory)))
                        print()

                        d = 0
                        while os.path.isfile('saveImage%d.jpg' % d):
                            d += 1

                        saved_image = cv2.imwrite(
                            "C:/Users/Salem/Desktop/Body-App/images/results/saveImage%d.jpg" % d, frame)

                        # //* Lists directory contents after saving new images
                        Red = fg('red')
                        print(White + "List of files available after saving function is executed:")
                        print(Red + str(os.listdir(filedirectory)))
                    else:
                        # //* Open a new window
                        window = Window(self)

                        # //* Routes enteries values in the window
                        window.grab_set()

                        fact_high = int(high_cm) - 5

                        # //* Create a TextBox
                        self.TextBox = tk.Text(window, width=56, font=("BOLD", 13))
                        self.TextBox.grid(row=1, column=1, padx=20,
                                        pady=20, ipady=10, sticky="NSEW")

                        # //* Insert values in the TextBox
                        my_height = 'Height: '+str(high_cm)+' cm\n'
                        self.TextBox.insert(INSERT, my_height)

                        chest_size = 'Chest size: ' + \
                            str(distance(chest, high, fact_high))+' cm\n'
                        self.TextBox.insert(INSERT, chest_size)

                        hip_size = 'Hip size: ' + \
                            str(distance(hip, high, fact_high))+' cm\n'
                        self.TextBox.insert(INSERT, hip_size)

                        waist_size = 'Waist size: ' + \
                            str(distance(waist, high, fact_high))+' cm\n'
                        self.TextBox.insert(INSERT, waist_size)

                        width = round(frameWidth/frameHeight*480)
                        height = 480
                        dim = (width, height)
                        resized = cv2.resize(
                            frame, dim, interpolation=cv2.INTER_AREA)

                        # //* Break line(new line)
                        newLine = '\n'
                        self.TextBox.insert(INSERT, newLine)

                        # //* Body size list
                        chestString = str(distance(chest, high, fact_high))
                        chestFloat = list(map(float, chestString.split(', ')))
                        chestInt = list(map(int, chestFloat))

                        hipString = str(distance(hip, high, fact_high))
                        hipFloat = list(map(float, hipString.split(', ')))
                        hipInt = list(map(int, hipFloat))

                        waistString = str(distance(waist, high, fact_high))
                        waistFloat = list(map(float, waistString.split(', ')))
                        waistInt = list(map(int, waistFloat))

                        # //* Clothing size for men
                        # //* Jackets and coats
                        gender_male = "You are a man and your sizes are the following: " + '\n'
                        self.TextBox.insert(INSERT, gender_male)
                        menTop1 = {
                            'S': [89, 90, 91, 92, 93, 94, 95],
                            'M': [95, 96, 97, 98, 99, 100, 101],
                            'L': [101, 102, 103, 104, 105, 106, 107],
                            'XL': [107, 108, 109, 110, 111, 112, 113],
                            '2XL': [113, 114, 115, 116, 117, 118, 119],
                            '3XL': [119, 120, 121, 122, 123, 124, 125],
                        }
                        menTop1_List = [key for ele in chestInt
                                        for key, val in menTop1.items() if ele in val]
                        menTop1_Size = "Jackets and coats = " + \
                            '-'.join(map(str, menTop1_List)) + '\n'
                        self.TextBox.insert(INSERT, menTop1_Size)

                        # //* T-shirts, polos and sweaters
                        menTop2 = {
                            'S': [90, 91, 92, 93, 94],
                            'M': [94, 95, 96, 97, 98],
                            'L': [98, 99, 100, 101, 102, 103],
                            'XL': [103, 104, 105, 106, 107, 108, 109],
                            '2XL': [109, 110, 111, 112, 113, 114, 115],
                            '3XL': [115, 116, 117, 118, 119, 120],
                            '4XL': [120, 121, 122, 123, 124, 125, 126, 127]
                        }
                        menTop2_List = [key for ele in chestInt
                                        for key, val in menTop2.items() if ele in val]
                        menTop2_Size = "T-shirts, polo and sweaters = " + \
                            '-'.join(map(str, menTop2_List)) + '\n'
                        self.TextBox.insert(INSERT, menTop2_Size)

                        # //* Trousers, jeans, shorts and bermuda shorts
                        menUnderwear_hip = {
                            '38': [90, 91, 92, 93, 94],
                            '40': [94, 95, 96, 97, 98],
                            '42': [98, 99, 100, 101, 102],
                            '44': [102, 103, 104, 105, 106],
                            '46': [106, 107, 108, 109, 110],
                            '48': [110, 111, 112, 113, 114],
                            '50': [114, 115, 116, 117, 118],
                            '52': [118, 119, 120, 121, 122]
                        }
                        menUnderwear_waist = {
                            '38': [75, 76, 77, 78, 79],
                            '40': [79, 80, 81, 82, 83],
                            '42': [83, 84, 85, 86, 87],
                            '44': [87, 88, 89, 90, 91],
                            '46': [91, 92, 93, 94, 95],
                            '48': [95, 96, 97, 98, 99, 100, 101],
                            '50': [101, 102, 103, 104, 105],
                            '52': [105, 106, 107, 108, 109]
                        }
                        menUnderwear_List = [key for ele in hipInt
                                            for key, val in menUnderwear_hip.items() if ele in val] + [key for ele in waistInt
                                            for key, val in menUnderwear_waist.items() if ele in val]
                        menUnderwear_Size = "Trousers, jeans, shorts and bermuda shorts = " + \
                            '-'.join(map(str, menUnderwear_List))
                        self.TextBox.insert(INSERT, menUnderwear_Size)

                        # //* Lists directory contents before saving new images
                        Green = fg('green')
                        White = fg('white')
                        filedirectory = (
                            r"C:/Users/Salem/Desktop/Body-App/images/results")
                        os.chdir(filedirectory)
                        print()
                        print(White + "List of files available before saving function is executed:")
                        print(Green + str(os.listdir(filedirectory)))
                        print()

                        d = 0
                        while os.path.isfile('saveImage%d.jpg' % d):
                            d += 1

                        saved_image = cv2.imwrite(
                            "C:/Users/Salem/Desktop/Body-App/images/results/saveImage%d.jpg" % d, frame)

                        # //* Lists directory contents after saving new images
                        Red = fg('red')
                        print(
                            White + "List of files available after saving function is executed:")
                        print(Red + str(os.listdir(filedirectory)))
                else:
                    messagebox.showwarning('warning', 'Please enter female or male as your gender.')
            except ValueError:
                messagebox.showwarning('warning', 'The height cannot be different from an integer value!\nPlease try with a integer value.')
                
# ----------------------------------------------------------


# ----------------------------------------------------------
#//* Execute the graphical user interface
app = App()
app.mainloop()
# ----------------------------------------------------------
