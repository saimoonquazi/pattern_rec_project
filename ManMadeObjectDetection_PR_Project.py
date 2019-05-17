import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy import fftpack

def readImage(image_name):
    image = Image.open(image_name)
    return image
##

def fft(channel):
    fft = np.fft.fft2(channel)
    fshift = np.fft.fftshift(fft)
    
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum
##

def cart2pol(input):
    
    binary = np.zeros((len(input),len(input[1])))
    binarySpect = np.zeros((len(input),len(input[1])))
    #angle = np.zeros((len(input),len(input[1])))
    angle = np.zeros((360,1))
    thrsh = 220
    print(input[5,5])
    for i in range (0,len(input)):
        for j in range (0,len(input[0])):
            if (input[i,j] > thrsh):
                binarySpect[i,j] = input[i,j]
                binary[i,j] = 1
                
    ##
    
    #binary[5,5]=1
    #Coodrinate origin
    x0 = int(len(input)/2)
    y0 = int(len(input[0])/2)
    print(x0)
    print(y0)  
    for i in range (0,len(input)):
        for j in range (0,len(input[0])):
            if (binary[i,j]==1):
                ind=int(np.arctan2(y0-j,x0-i) * 180 / np.pi)+90
                #print(ind)
                angle[ind]=angle[ind]+1
    
    print(angle)
    plt.plot(angle)
    plt.show()
    return binarySpect
##

# def image2array(img):
    # return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
# ##

def DEV_drawGui(original_image, result_image):
    plt.subplot(121),plt.imshow(original_image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(result_image, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return 1
##

def main():
    image_name = "house.png"
    original_image = readImage(image_name)
    channels = original_image.split()
    
    result_array = np.zeros_like(original_image)
    
    gray_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2GRAY)
    result_array = fft(gray_img)
    result_image = Image.fromarray(result_array)
        
    binary_image = Image.fromarray(cart2pol(result_array))
    
    DEV_drawGui(original_image, binary_image)
    
    return 1
##

main()
