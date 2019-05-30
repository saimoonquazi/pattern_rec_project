####################################################################################################################################
#
# Description: This GUI was developed in an attempt to create an interactive portal to display the functionality of the classification #              developed during the course of this project. Tkinter was used for the development of the GUI with minimalistic          #              functionality.                
#
#####################################################################################################################################

# Import the relevant libraries
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import data_file_reader
from tkinter import filedialog
import feature_extractor_GUI
from tkinter import ttk



class Assignment(tk.Frame):
    def Apply(self):
        self.label_imgtype.destroy()
        plt.close('all')
        self.original_image,self.result_image, self.binary_image,self.filtered_image=feature_extractor_GUI.extract_features_prediction(self.filename)
        self.original_image=cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        feature_extractor_GUI.DEV_drawGui(self.original_image, self.result_image, self.binary_image)                                                                                                      
        plt.figure(2)
        plt.plot(self.filtered_image)
        plt.xlabel('Angles(deg)')
        plt.xlabel('Angles(Deg)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Filtered Image with Angle Calculation')
        plt.show()
        self.label_imgtype = tk.Label(root, text="Features Extracted Successfully")              
        self.label_imgtype.config(font=("Times New Roman", 14))                                                        
        self.label_imgtype.grid(row=5, columnspan=2) 
        self.feature_extraction_flag=True
    def browse(self):    
        self.label_imgtype.destroy()
        self.filename = filedialog.askopenfilename()
        self.img = cv2.imread(self.filename, 1)                                                                         
        self.b, self.g, self.r = cv2.split(self.img)                                                                    
        self.img1 = cv2.merge((self.r, self.g, self.b))                                                                 
        self.img2 = Image.fromarray(self.img1)                                                                          
        self.img = self.img2.resize((512,512))                                                                          
        self.canvas.image = ImageTk.PhotoImage(self.img)                                                                
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')                                            
        self.label_imgtype = tk.Label(root, text="The Image Loaded, Features not Extracted")                                            
        self.label_imgtype.config(font=("Times New Roman", 14))                                                         
        self.label_imgtype.grid(row=5, columnspan=2)                                                                    
        
    def classify(self):
        #Check if feature extraction has been done, if not run the function
        if self.feature_extraction_flag is False:
            self.Apply()
        #Check if the model has been trained or not, if not train the model
        self.result_label.destroy() 
        if self.train_flag is False:
            self.model_train()
        filename='features_test.csv'
        data,predict_features,_=data_file_reader.file_reader(filename,'test')
        lda_test_set = self.lda.transform(predict_features)
        prediction=self.clf.predict(lda_test_set)
        if prediction==0:
            self.result_label = tk.Label(root, text="The Image Contains a Natural Scene",wraplength=200)                                                 
            print('The Image Contains a Natural Scene')
        else:
            self.result_label = tk.Label(root, text="The Image Contains Man Made Objects in the Scene",wraplength=200)                                              
            print('The Image Contains Man Made Objects in the Scene')
        self.result_label.config(font=("Times New Roman", 14))                                                           
        self.result_label.place(relx=0.7, rely=0.45, anchor='sw')            

    def model_train(self):
        self.label_training_state.destroy()                                                                                      
        filename='features_train.csv'
        data,features,labels=data_file_reader.file_reader(filename,'train')
        self.svc=SVC(kernel='linear', C=1)
        self.rf=RandomForestClassifier(n_estimators=50, random_state=1)
        self.knn=KNeighborsClassifier(n_neighbors=3)
        self.mv=VotingClassifier(estimators=[('rf', self.rf),('knn',self.knn),('svc',self.svc)], voting='hard')
        self.lda=LDA(n_components=200)
        lda_train_set=self.lda.fit_transform(features,np.ravel(labels))
        if self.comboExample.get() == "Majority Voting":    
            self.clf=self.mv.fit(lda_train_set,np.ravel(labels))
            classifier_label="Model Trained Successfully on Majority Voting"
        elif self.comboExample.get() == "SVC":
            self.clf=self.svc.fit(lda_train_set,np.ravel(labels))
            classifier_label="Model Trained Successfully on SVC"
        elif self.comboExample.get() == "KNN":
            self.clf=self.knn.fit(lda_train_set,np.ravel(labels))
            classifier_label="Model Trained Successfully on KNN"            
        else:
            self.clf=self.rf.fit(lda_train_set,np.ravel(labels))     
            classifier_label="Model Trained Successfully on Random Forest"
        self.label_training_state = tk.Label(root, text=classifier_label,wraplength=200)                                              
        self.label_training_state.config(font=("Times New Roman", 12))                                                           
        self.label_training_state.grid(row=6, column=8) 
        self.train_flag=True                                                                     


    def __init__(self, root):
        tk.Frame.__init__(self, root)
        self.train_flag=False
        self.feature_extraction_flag=False
        # BROWSE
        self.btn_browse = tk.Button(root, text="Browse", command=self.browse)
        self.btn_browse.grid(row=0, column=0)

        # APPLY
        self.btn_feature_extract = tk.Button(root, text="Extract Features", command=self.Apply)
        self.btn_feature_extract.grid(row=0, column=1)

        # Classify Image
        self.btn_classify = tk.Button(root, text="Classify Image", command=self.classify)
        self.btn_classify.grid(row=5, column=5)

        # Train Model
        self.model_train_btn = tk.Button(root, text="Train Model", command=self.model_train)
        self.model_train_btn.grid(row=5, column=8)



        # CANVAS
        self.canvas = tk.Canvas(root, width=800, height=800)
        self.canvas.grid(row=10, columns=10)
        
        self.label_imgtype = tk.Label(root, text="No Image Loaded")              
        self.label_imgtype.config(font=("Times New Roman", 14))                                                         
        self.label_imgtype.grid(row=5, columnspan=2) 

        # Labels
        self.label_training_state = tk.Label(root, text="Not Trained")
        self.label_training_state.grid(row=6, column=8)
        self.result_label = tk.Label(root, text="No Classification")
        self.result_label.config(font=("Times New Roman", 14))  
        self.result_label.place(relx=0.7, rely=0.4, anchor='sw')
        self.course_label = tk.Label(root, text="Pattern Recognition (LOTI.05.046)")
        self.course_label.config(font=("Times New Roman", 16))  
        self.course_label.place(relx=0.34, rely=0.92, anchor='sw')
        self.title_label = tk.Label(root, text="Man made object detection in natural scenes")
        self.title_label.config(font=("Times New Roman", 16))  
        self.title_label.place(relx=0.3, rely=0.95, anchor='sw')
        self.classifier_label = tk.Label(root, text="Choose Classifier")
        self.classifier_label.grid(row=0, column=8)
        self.comboExample = ttk.Combobox(root, 
                            values=[
                                    "SVC", 
                                    "Random Forest",
                                    "KNN",
                                    "Majority Voting"])
        self.comboExample.grid(column=8, row=1)
        self.comboExample.current(3)
        
        self.logo = cv2.imread('tartu_logo.jpg', 1)
        self.b, self.g, self.r = cv2.split(self.logo)                                                                     
        self.logo1 = cv2.merge((self.r, self.g, self.b))                                                                 
        self.logo2 = Image.fromarray(self.logo1)                                                                          
        self.logo = self.logo2.resize((180,180))                                                                          
        self.canvas.image_logo = ImageTk.PhotoImage(self.logo)                                                                                
        self.canvas.create_image(0, 540, image=self.canvas.image_logo, anchor='nw')


# INITIALIZE
root = tk.Tk()
root.wm_title("Pattern Recognition (LOTI.05.046) - Man made object detection in natural scenes")
root.minsize(800,868)
# INITIALIZE CLASS\
Assignment(root).grid()
# LIVE
root.mainloop()