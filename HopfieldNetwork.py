import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_prefix = 'pattern' # prefix for image files
side = 50 # side length of the image
num_patterns = 5
n = side * side # number of neurons (pixels)
imgs=[]

# image processing functions
def load_img(path, side): # load and process images into a binary array, where pixels are represented as 1 or -1
    img = Image.open(path)
    img = img.resize((side, side))
    img = img.convert('1') # convert image to black and white
    img = 2 * np.array(img, int) - 1 # map pixel values to {1, -1}
    return img.flatten() # return the processed image as a 1D array

def show_array(img_array): # visualize images (array)
    side = int(np.sqrt(img_array.shape[0])) # calculate the side length of the image
    img_array = img_array.reshape((side, side)) # reshape the array into a 2D image
    plt.figure(figsize=(3, 3)) # create a new figure
    plt.imshow(img_array) # display the image
    plt.axis('off') # hide the axis
    plt.show() # show the figure

def show_multiple_arrays(img_arrays): # visualize multiple images (arrays)
    fig = plt.figure(figsize=(3, 3))
    for i in range(len(img_arrays)):
        side = int(np.sqrt(img_arrays[i].shape[0]))
        plt.subplot(1, len(img_arrays), i+1)
        plt.imshow(img_arrays[i].reshape((side, side)))
        plt.axis('off')
    plt.show()

def modify_img(n, img): # introduce noise or modifications to an image, for testing the network’s ability to reconstruct it
  for i in range(n): # make 1/2 of image negative
    if i > n/2-1:
      img[i] = -1
  return img

# training. calculate weigths matrix.
def calculate_w(img): # create a weight matrix using the Hebbian learning rule based on the outer product of the image vector.
    # w = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(n):
    #         if i!=j:
    #             w[i,j]=img[i]*img[j] # Hebbian learning rule
    w = np.outer(img, img)
    np.fill_diagonal(w, 0) # set diagonal to 0, because we don't want to reinforce the neuron's own activation
    return w

# inference. reconstruct image
def reconstructed_image(n, w, state): # use the weight matrix to reconstruct an image from a modified or noisy version
    # for i in range(n):
    #     sum = 0
    #     for j in range(n):
    #         sum += w[i,j]*state[j]
    #     state[i] = 1 if sum>0 else -1
    state = np.sign(np.dot(w, state)) # vectorized implementation for efficiency
    return state


# memorize images
for i in range(1, num_patterns+1): # memorize images from 1 to 5
    imgs.append(load_img(f'{image_prefix}{i}.png', side))
print('memorized images:')
show_multiple_arrays(imgs)

# calculate weights matrix. a one-shot learning process
w = np.zeros((n,n)) # initialize weight matrix
for p in range(len(imgs)): # iterate over all patterns
    w += calculate_w(imgs[p]) # accumulate weight matrices for each pattern
w /= len(imgs) # normalize by number of neurons

# init a perturbation
#state = modify_img(n, load_img(f'{image_prefix}2.png', side)) # modified image
state = np.random.choice([-1, 1], size=n) # modify random pixels
print('init state:')
show_array(state)

# reconstruct image
state = reconstructed_image(n, w, state)
print('reconstructed image:')
show_array(state)