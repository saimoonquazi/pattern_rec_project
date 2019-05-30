#!/usr/bin/env python
# coding: utf-8

#######################################################################################################################################
# This is a helper header script to streamline the feature extraction process by supplying helper functions that can be used if and    # when required. Some of the functions in this file are for plotting and displaying only, and can be ignored in some cases. 
#######################################################################################################################################

#Call Relevant libraries used
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import fftpack
from shutil import copyfile
import time
import os
import csv

######################################################################################################################################
# Function Name: readImage                                 
# Function Inputs : image_name
# Returns: image
# Description: This function takes in an image filename and reads it using OpenCV. The image oobject is then return. It should be noted # that the image filename should be specified if it exists in a specific directory, otherwise it looks for the image in the home       # directory.
#######################################################################################################################################
def readImage(image_name):
    image = cv2.imread(image_name)
    return image

######################################################################################################################################
# Function Name: fft                                 
# Function Inputs : channel
# Returns: magnitude_spectrum
# Description: This function takes in an image object (OpenCV), performs Fast Fourier Transform on it, centers the transformed         # spectrum, then computes the log scale of the magnitude spectrum to get rid of the imaginary component. The function than applies the # high pass filter on the spcectrum. The filtered array is then returned.   #######################################################################################################################################
def fft(channel):
    fft = np.fft.fft2(channel)
    fshift = np.fft.fftshift(fft)
    
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    magnitude_spectrum = highPassFilter(magnitude_spectrum)
    return magnitude_spectrum

######################################################################################################################################
# Function Name: highPassFilter                             
# Function Inputs : input
# Returns: output
# Description: This function is a streamline to performing a high pass filter on the spectrum passed in as an input. The function         # filters out 4% of the total size of the image, maintaining consistency regardless of the size of the image passed in. The size of the # filter is hardcoded and so if the user chooses to change this, they must alter this function.     #######################################################################################################################################
def highPassFilter(input):
    #Create a container for the output image
    output = np.zeros((len(input),len(input[0])))
    
    #Find number of rows and columns of the input image
    rows, cols = len(input), len(input[0])
    #Find the center of the image
    center_row, center_col = int(rows/2), int(cols/2)
    #Compute the sigma point based on the size of the image
    sigma = int(len(input)*0.04)
    
    #Define the mask
    mask_circle = np.ones((rows,cols), np.uint8)

    #Loop through the array from side to side
    for i in range (center_row-sigma,center_row+sigma):
        for j in range (center_col-sigma,center_col+sigma):
            #Compute the euclidean distance from the center to the array position
            distance = np.sqrt((center_row - i)**2 + (center_col - j)**2)
            #If the distance is smaller than sigma value
            if (distance < sigma):
                #Set indexed mask value to 0
                mask_circle[i,j] = 0
    #Apply mask on input and produce the output array
    output= input*mask_circle
    
    return output

######################################################################################################################################
# Function Name: cart2pol                             
# Function Inputs : input
# Returns: angle,binarySpect
# Description: This function takes in a filtered magnitude spectrum array and applies a binary threshold (at 80% of the maximum pixel  # values) to filter out low level values. The # function then computes the angles at which the remaning pixels occurs and counts these # occurances to develop a histogram array of the # occurances (frequency) per angle. It then returns the binary array as well as the   # histogram array.    #######################################################################################################################################
def cart2pol(input):
    #Define containers for Binary array and an output array
    binary = np.zeros((len(input),len(input[0])))
    binarySpect = np.zeros((len(input),len(input[0])))
    
    #Define array to store the angles
    angle = np.zeros((360,1))
    #Find the threshold value by computing it to to 80% of the maximum pixel value of the input
    thrsh = np.amax(input)*0.8
    
    #Loop through the image
    for i in range (0,len(input)):
        for j in range (0,len(input[0])):
            #Apply threshold
            if (input[i,j] > thrsh):
                binarySpect[i,j] = input[i,j]
                binary[i,j] = 1
    
    # Find the center point of the image
    x0 = int(len(input)/2)
    y0 = int(len(input[0])/2)
    
    # Loop through the image
    for i in range (0,len(input)):
        for j in range (0,len(input[0])):
            # If the pixel is not filtered out by the threshold
            if (binary[i,j]==1):
                #Calculate the angle at which the pixel is at
                ind=int(np.arctan2(y0-j,x0-i) * 180 / np.pi)+90
                #Count one for the occurance at this angle
                angle[ind]=angle[ind]+1
    
    return angle, binarySpect

######################################################################################################################################
# Function Name: meanFilterHistogram                            
# Function Inputs : input, kernal_size
# Returns: output
# Description: This function takes in an array of the histogram of occurances at angles and applies a simple box filter of the kernal  # size passed as an input #######################################################################################################################################
def meanFilterHistogram(input, kernel_size):
    output = input.copy()
    n = int(kernel_size/2)
    for i in range (kernel_size,len(input)-kernel_size):
        output[i] = (input[i-n:i+n].sum())/kernel_size 
   
    return output

######################################################################################################################################
# Function Name: DEV_drawGui                            
# Function Inputs : original_image,fft,result_image
# Returns: None
# Description: This is a helper function to streamline the plotting of the input image objects onto the designated figure platform.    # The function is used for troubleshooting and visual representation of the process but is not compulsory of the operation of the      # feature extraction process. #######################################################################################################################################
def DEV_drawGui(original_image, fft, result_image):
    plt.subplot(131),plt.imshow(original_image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(fft, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(result_image, cmap = 'gray')
    plt.title('Filtered Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return 1

######################################################################################################################################
# Function Name: train_data_gen                            
# Function Inputs : directory,label,output_file
# Returns: None
# Description: This function takes in a directory path and loops through the images within and appends it to the specified otuput file # with the feature space computed using the functions above. The output filename is hardcoded and based on the use, the correct label  # is appended ot the start of the feature information (0 for natural scene images and 1 for scenes with man-made objects within). This    # functionality is hardcoded and must be changed if this function is to be used for other applciations. #######################################################################################################################################
def train_data_gen(directory,cls,output_file):
    #Record the start time
    start=time.time()
    #Loop through the images in the directory
    for filename in os.listdir(directory):
        # Make a copy of the current image to the working directory
        copyfile(directory+'/'+filename, filename)
        image_name = filename
        print('Reading...:'+filename)
        #Read the image using the appropriate function
        original_image = readImage(image_name)
        #Define an output container
        result_array = np.zeros_like(original_image)
        #Convert the image to grayscale
        gray_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2GRAY)
        #Execute the fft function to get back the filtered magnitude spectrum
        result_array = fft(gray_img)
        #Convert the ouput to an image object
        result_image = Image.fromarray(result_array)
        #Use the cart2pol function described above to get back the angles histogram and the binary image array
        angles, binary_image = cart2pol(result_array)
        #Apply the box filter to the histogram
        filteredAngles = meanFilterHistogram(angles,7)  
        #Definition of the appropriate label (hardcoded)
        label=np.array([cls])
        #Append the labels and ensure that the shape is righte before writing to CSV file
        feature_print=np.concatenate((label,np.ravel(filteredAngles)))
        feature_print=feature_print.reshape((361,1))
    
        #Open the features_train.csv file and append the data as a row in the file with tab delimiter     
        with open(output_file, 'a') as csvFile:
            writer = csv.writer(csvFile,delimiter ='\t')
            writer.writerows(feature_print.T)
        csvFile.close()
        
        #Plot required information
        plt.plot(angles)
        plt.show()
        plt.plot(filteredAngles)
        #plt.savefig(filename+'_hist'+'.png')
        plt.show()
        DEV_drawGui(original_image, result_image, Image.fromarray(binary_image))
        #Once done, remove the image copied  
        os.remove(filename) 
    #Once all the images have been processed, stop the timer and print the total time taken to complete the loop    
    end=time.time()
    print('Execution_time: %f'%(end-start))
    return 1

######################################################################################################################################
# Function Name: extract_features_predictions                            
# Function Inputs : filename
# Returns: None
# Description: This function caters to the need of taking in only 1 image to extract features from. The process is the same as the     # train_data_gen function, but only differs by reading only a single image to extract the features and does not append a label to it.  # This function also always operates on a feature_test.csv file.  #######################################################################################################################################
def extract_features_prediction(filename):
    # Take the filename passed as an argument and read the image
    image_name = filename
    print('Reading...:'+filename)
    original_image = readImage(image_name)
    
    #Define a container for the output
    result_array = np.zeros_like(original_image)
    
    #Convert the image to grayscale and compute the filtered magnitude spectrum of the image and convert it to an image object
    gray_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2GRAY)
    result_array = fft(gray_img)
    result_image = Image.fromarray(result_array)
    
    #Obtain the histogram and binary image array
    angles, binary_image = cart2pol(result_array)
    
    #Apply box filter to the histogram
    filteredAngles = meanFilterHistogram(angles,7)  
    
    #Optional, plot the images for visualization
    DEV_drawGui(original_image, result_image, Image.fromarray(binary_image))
    
    #Open the features_test.csv file and write the features as rows
    with open('features_test.csv', 'a') as csvFile:
        writer = csv.writer(csvFile,delimiter ='\t')
        writer.writerows(filteredAngles.T)
    csvFile.close()
    
    #Plot the appropriate results
    plt.plot(angles)
    plt.show()
    plt.plot(filteredAngles)
    #plt.savefig(filename+'_hist'+'.png')
    plt.show()
    return 1



