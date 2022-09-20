#TODO find a way to automatically install dependencies

import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import math
import os

#TODO maybe make a Kernel class (overkill, but nice to do)
def generate_kernel(s):
    """
    Generates an n x n Gaussian kernel matrix depending on the standard deviation.
    n is given as 4 times the standard deviation, while guaranteed to be odd.
    Inputs:
    s (float) - The standard deviation.
    Returns:
    kernel (numpy.ndarray) - The kernel, covering at least 4 standard deviations.
    n (int) - the length of the square kernel.
    """

    # The kernel is at least 4 standard deviations, can be changed later
    # n is the side length
    n = math.ceil(4*s)

    # Ensure odd sized kernel
    if (n % 2) != 1:
        n = n+1

    # Matrix corresponding to the y coordinates of the kernel
    kernel_y = []
    for i in range((n//2), -(n//2 + 1), -1):
        kernel_y.append(np.ones(n)*i)
    kernel_y = np.array(kernel_y) # Convert Python list to numpy array

    # We simply transpose our y matrix, then negate for our kernel's x coordinates
    kernel_x = -1*(np.transpose(kernel_y))

    # Obtain kernel with 2D Gaussian distribution
    kernel_xy = kernel_x*kernel_x + kernel_y*kernel_y
    kernel = 1/(2*math.pi*(s*s))*np.exp(-kernel_xy/(2*(s*s)))

    return kernel, n


def process_image(img, k, n):
    """
    Convolves the kernel with the image in RGB.
    Inputs:
    img (numpy.ndarray) - The image, as a matrix.
    k - (numpy.ndarray) - The kernel, as a matrix.
    n (int) - the side length of the square kernel.
    Returns:
    image_blur (numpy.ndarray) - The blurred image as a matrix.
    """

    # Retain original image size by zero padding up to the kernel radius
    image_pad = np.pad(img, pad_width = [(n//2, n//2),(n//2, n//2),(0,0)], mode = 'constant')
    # Initialize empty image for population
    image_blur = np.zeros_like(img)

    k = np.expand_dims(k, axis = -1) # Cast (n,n) array to (n,n,1) array

    # Navigate kernel through nonzero elements of padded image, then convolve with image
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            image_blur[y,x] = np.sum(k*image_pad[y:y + k.shape[0], x:x + k.shape[1]], axis = (0, 1))
    
    return image_blur


def main():
    """
    Main function, gathers inputs, calls functions, and saves output image as a file.
    Inputs:
    None
    Returns:
    None
    """

    # Print local files in current working directory
    print("\nFiles located in current working directory {}:".format(str(os.getcwd())))
    for file in os.listdir():
        print(file)

    while True:
        try:
            filename = input('\nEnter an image file: ')
            image = asarray(Image.open(filename)) # Open image, then convert to matrix
            image_3channel = image[:,:,:3] # We do not need to deal with alpha values, if present
            sdev = float(input("Enter the standard deviation (higher values blur the image more): "))

            if sdev > 0:                
                kernel, n = generate_kernel(sdev)
                print("Computing...")
                array_blur = process_image(image_3channel, kernel, n)

                image_blur = Image.fromarray(array_blur) # Convert from array back to image
                savedfilename = 'blurred_sdev=' + str(sdev) + '_' + filename
                image_blur.save(savedfilename) # Save as new image

                print("Image saved as {} in {}".format(savedfilename, str(os.getcwd())))
    
                break

            elif sdev <= 0:
                print("Please enter a positive, nonzero float")

        except PIL.UnidentifiedImageError: # Unsupported image
            print("Please enter an image file")

        except FileNotFoundError: # No file found
            print("File not found, please try again")

        except ValueError: # Invalid standard deviation
            print("Enter a valid number")
            image.close()
  
    pass
            

if __name__ == "__main__":
    main()