import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from matplotlib.figure import Figure
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

import cv2 

import time


pencere = tk.Tk() 
pencere.configure(bg='black')


def kamera_ac () :
    mesaj = messagebox.showinfo(title = "Bilgilendirme", message = "Kamera açma işlemi başarılı. Tespit ekranı yeni pencere de açılıyor.")
    print(mesaj)

    
    # Load Yolo
    net = cv2.dnn.readNet("model/yolov3.cfg", "model/yolov3.weights")
  
    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading camera
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0


    while True:
        _, frame = cap.read()
        frame_id += 1

        height, width, channels = frame.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    
    cap.release()
    cv2.destroyAllWindows()
       

def foto_ac():
    
    # fotoğraf seçme ve ekleme
    
    file_name = filedialog.askopenfilename()
    print(file_name)
    
    img0 = Image.open(file_name)
    img0 = ImageTk.PhotoImage(img0)
    
    labell = tk.Label(fram1, image = img0)
    labell.image = img0
    labell.pack(padx = 15, pady = 15)
    
    mesaj = messagebox.showinfo(title = "Bilgilendirme", message = "Kamera açma işlemi başarılı. Tespit ekranı yeni pencere de açılıyor.")
    print(mesaj)
    
    
    # Load Yolo
    net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")   # openCV ile modellerimizi okuyoruz.
    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]                     # coco.names içerisindeki nesneleri diziye aktarıyoruz.

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))                 # her nesneye farklı bir renk atar.

    # Loading image
    img = cv2.imread(file_name)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


    cv2.imshow("Tespit Ekranı", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

frame_sol = tk.LabelFrame(pencere, width = 250, height = 280,bg = "gray20")
frame_sol.pack (side = tk.LEFT, fill = tk.BOTH, expand = True, padx = 10, pady = 10)

butn = tk.Button(frame_sol, text = "Fotoğraf Aç", width = 20, height = 3, bg = "yellow",command = foto_ac)
butn.place(x = 250, y=25)

fram1 = tk.LabelFrame (frame_sol, text = "Fotoğraf",  bg = "gray20", width = 500, height = 500)
fram1.place(x = 58, y=120)



frame_sag = tk.LabelFrame(pencere, width = 250, height = 200,bg = "gray20")
frame_sag.pack (side = tk.LEFT, fill = tk.BOTH, expand = True, padx = 10, pady = 10)

'''
butn4 = tk.Button(frame_sag, text = "Kamera Aç",  width = 50, height = 10, bg = "yellow",command=kamera_ac)
butn4.place(x = 150, y=25)

'''
etiket = tk.Label(frame_sag, text = "Canlı tespiti başlatmak için aşağıdaki kamera butonuna tıklayınız",bg = "yellow",height = 3) 
etiket.place(x = 160 , y = 20)

butn4 = tk.Button(frame_sag, text = "Kamera Aç",  width = 200, height = 120, bg = "grey50",command=kamera_ac)
imgg = ImageTk.PhotoImage(file="fotoButon/icons8-camera-96.png")
butn4.config(image = imgg)
butn4.place(x = 225, y=150)

'''
butn5 = tk.Button(frame_sag, text = "Video Yükle",  width = 50, height = 10, bg = "yellow")
butn5.place(x = 150, y=400)
'''

pencere.mainloop()









