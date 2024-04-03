
import numpy as np
import tifffile as tiff
from queue import Queue

# Load the TIFF stack
image = tiff.imread('/Users/nairaaltunkeser/Desktop/Ilastik_stack.tif')
# Starting pixel
start_pixel = (34, 378, 297)  # (z,y,x)

def is_valid(z, y, x, visited, image): 
    Z, Y, X = image.shape
    return (0 <= x < X and 0 <= y < Y and 0 <= z < Z and image[z][y][x] == 1 and not visited[z][y][x])

# BFS function
def search(start_pixel, image):
    # All possible moves from a cell
    moves = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    #moves = [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)] 


    visited = np.zeros(image.shape, dtype=bool) # False for each pixel initially
    Z, Y, X = image.shape
    component = []
    queue = Queue()

    # Starting from the start_pixel
    queue.put(start_pixel)
    visited[start_pixel] = True
    component.append(start_pixel)

    while not queue.empty():
        pixel = queue.get()
     
        # Checking all possible moves
        for move in moves:
            z = pixel[0] + move[0]
            y = pixel[1] + move[1]
            x = pixel[2] + move[2]
            
            if is_valid(z, y, x, visited, image):
                queue.put((z, y, x))
                visited[z][y][x] = True
                component.append((z, y, x))
    
    return component

connected_component = search(start_pixel, image)

#To save the results 

# Create an empty image of float32
output_image = np.zeros(image.shape, dtype=np.float32)

# Fill in the connected component
for pixel in connected_component:
    output_image[int(pixel[0]), int(pixel[1]), int(pixel[2])] = 1.0

# Save the output image as a TIFF file
tiff.imsave('/Users/nairaaltunkeser/Desktop/Output.tif', output_image)
