import os
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import feature
from skimage.morphology import disk, square, erosion, dilation, binary_opening, binary_closing
from skimage.measure import find_contours
from skimage import filters
from scipy import ndimage as ndi
from skimage.draw import polygon, line
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
from matplotlib import cm
import matplotlib.patches as patches

def normalize(frame, plot=False):

    norm_frame = frame/np.mean(frame)

    if plot is True:
        plt.figure()
        plt.imshow(norm_frame)

    return norm_frame

def get_blue_arrow(frame, plot=False):
    frame_copy = frame.copy()

    blue_mask = 255*(frame_copy[:,:,0] < 0.4)*(frame_copy[:,:,1] < 0.4)*(frame_copy[:,:,2] > 0.6)

    frame_copy = blue_mask.astype(int)
    frame_copy = cv2.medianBlur(np.array(frame_copy,dtype = np.uint8), 3)
    
    if plot is True:
        plt.figure()
        plt.imshow(frame_copy, cmap="gray")

    return frame_copy

def get_red_black(frame, plot=False):
    frame_copy = frame.copy()

    black_mask = 255*(frame_copy[:,:,1] < 0.7)*(frame_copy[:,:,2] < 0.7)

    frame_copy = black_mask.astype(int)
    
    if plot is True:
        plt.figure()
        plt.imshow(frame_copy, cmap="gray")

    return frame_copy

def object_center(frame, plot=False):
    frame_copy = frame.copy()
    x_center = 0
    y_center = 0
    nb_point = 0

    for x in range(frame_copy.shape[0]):
        for y in range(frame_copy.shape[1]):
            if frame_copy[x,y] > 0:
                nb_point += 1
                x_center += x
                y_center += y

    if nb_point == 0:
        print("No points found")
        return 0

    x_center /= nb_point
    y_center /= nb_point

    if plot is True:
        plt.figure()
        plt.imshow(frame_copy, cmap="gray")
        plt.scatter(y_center,x_center, s = 10, c='red', marker='o')

    return x_center, y_center

def remove_robot(frame, arrow_center,width, plot=False):
    
    frame_copy = frame.copy()

    width = np.round(width)
    arrow_center = np.array(np.round([arrow_center[0],arrow_center[1]]), dtype = np.uint32)

    if arrow_center[0] - width < 0:
        x_start = 0
        x_end = arrow_center[0] + width
    elif arrow_center[0] + width > frame.shape[0]:
        x_start = arrow_center[0] - width 
        x_end = frame.shape[0]
    else:
        x_start = arrow_center[0] - width
        x_end = arrow_center[0] + width 

    if arrow_center[1] - width < 0:
        y_start = 0
        y_end = arrow_center[1] + width
    elif arrow_center[1] + width > frame.shape[1]:
        y_start = arrow_center[1] - width
        y_end = frame.shape[1]
    else:
        y_start = arrow_center[1] - width
        y_end = arrow_center[1] + width 

    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            frame_copy[x][y] = 0
        
    if plot is True:
        plt.figure()
        plt.imshow(frame_copy, cmap="gray")

    return frame_copy

def find_edges(frame, plot=False):

    frame_copy = frame.copy()

    edge_sobel = filters.sobel(frame_copy)
    edge_sobel *= 255/edge_sobel.max()
    edge_sobel = edge_sobel.astype(int)

    if plot is True:
        plt.figure()
        plt.imshow(edge_sobel)
    
    return edge_sobel

def remove_lines(frame, width, plot=False):
        
    frame_copy = frame.copy()
    frame_plot = frame.copy()
    # rr, cc = line(100, 0, 400, 719)
    # frame_copy[rr, cc] = 256

    # rr, cc = line(400, 0, 0, 719)
    # frame_copy[rr, cc] = 256

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(frame_copy, theta=tested_angles)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

        if angle > -np.pi/4 and angle < np.pi/4 :
            x0, x1 = 0, frame_copy.shape[0]
            y0, y1 = (dist - np.array([x0,x1]) * np.sin(angle))/np.cos(angle)

            r = np.array([x0 - width * np.sin(angle), x0 + width * np.sin(angle), x1 + width * np.sin(angle), x1 - width * np.sin(angle)])
            c = np.array([y0 - width * np.cos(angle), y0 + width * np.cos(angle), y1 + width * np.cos(angle),y1 - width * np.cos(angle)])
            r[r <= 0] = 0
            c[c <= 0] = 0
            r[r >= 480] = 480
            c[c >= 720] = 720
            r = r.astype(int)
            c = c.astype(int)

            rr, cc = polygon(r, c)
            frame_copy[rr, cc] = 0
            frame_plot[rr, cc] = 128
                    

        else:
            y0, y1 = 0, frame_copy.shape[1]
            x0, x1 = (dist - np.array([y0,y1]) * np.cos(angle)) / np.sin(angle)
            r = np.array([x0 - width * np.sin(angle), x0 + width * np.sin(angle), x1 + width * np.sin(angle),x1 - width * np.sin(angle)])
            c = np.array([y0 - width * np.cos(angle), y0 + width * np.cos(angle), y1 + width * np.cos(angle),y1 - width * np.cos(angle) ])
            r[r <= 0] = 0
            c[c <= 0] = 0
            r[r >= 480] = 480
            c[c >= 720] = 720
            r = r.astype(int)
            c = c.astype(int)
            rr, cc = polygon(r, c)
            frame_copy[rr, cc] = 0
            frame_plot[rr, cc] = 128

    if plot is True:
        # Generating figure 1
        fig, ax = plt.subplots(3, 2, figsize=(30,30))
        #ax = axes.ravel()

        ax[0,0].imshow(frame, cmap=cm.gray)
        ax[0,0].set_title('Input image')
        ax[0,0].set_axis_off()

        ax[0,1].imshow(np.log(1 + h),
                    extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                    cmap=cm.gray, aspect=1/1.5)
        ax[0,1].set_title('Hough transform')
        ax[0,1].set_xlabel('Angles (degrees)')
        ax[0,1].set_ylabel('Distance (pixels)')
        ax[0,1].axis('image')

        
        ax[1,0].imshow(frame_copy, cmap=cm.gray)
        origin = np.array((0, frame_copy.shape[1]))

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[1,0].plot(origin, (y0, y1), '-r')
    
        ax[1,0].set_xlim(origin)
        ax[1,0].set_ylim((frame_copy.shape[0], 0))

        ax[1,0].set_title('Detected lines')
        ax[1,0].set_axis_off()

        ax[1,1].imshow(frame_plot, cmap=cm.gray)
        ax[1,1].set_axis_off()
        ax[1,1].set_title('Polygon covering lines')

        ax[2,0].imshow(frame_copy, cmap=cm.gray)
        ax[2,0].set_axis_off()
        ax[2,0].set_title('Image without lines')
        ax[2,1].set_axis_off()

        plt.tight_layout()
        plt.savefig("Hough.pdf")
        plt.show()

    return frame_copy

def segmentation(frame, width, adjust, plot=False):

    frame_copy = frame.copy()

    if frame_copy.shape[0] % width != 0 or frame_copy.shape[1] % width != 0:
        print('frame shape not a multiple of width')
        return 0

    mass_centers = []

    for x_step in range(int(frame_copy.shape[0] / width)):
        for y_step in range(int(frame_copy.shape[1] / width)):
            
            square = frame_copy[width * x_step : width * (x_step + 1), width * y_step : width * (y_step + 1)]
            total = np.sum(square > 0)
            best_adj = (0,0) 

            if  total <= 50:
                continue

            for x_adj in range(-int(adjust), int(adjust)):
                for y_adj in range(-int(adjust), int(adjust)):
                    square = frame_copy[width * x_step + x_adj: width * (x_step + 1) + x_adj, width * y_step + y_adj: width * (y_step + 1) + y_adj]
                    total2 = np.sum( square > 0) 
                    if  total2 > total:
                        total = total2
                        best_adj = (x_adj,y_adj) 
                        final_square = square

            final_square = frame_copy[width * x_step + best_adj[0]: width * (x_step + 1) + best_adj[0], width * y_step  + best_adj[1]: width * (y_step + 1) + best_adj[1]]
            square_mass_center = object_center(final_square)   
            mass_center = np.array(square_mass_center) + np.array([width * x_step,width * y_step]) + np.array(best_adj)
            mass_center = np.round(mass_center).astype(int)

        
            if mass_center.tolist() in mass_centers:
                continue
            else:
                mass_centers.append(mass_center.tolist())
    
    mass_centers = np.array(mass_centers)

    if plot is True:
        fig,ax = plt.subplots(1)
        ax.imshow(frame_copy, cmap = 'gray')
        ax.scatter(mass_centers[:,1], mass_centers[:,0], color='r')
        plt.show()
    
    return mass_centers
        
def get_images(frame, mass_centers, sub_image_size, plot= False):

    frame_copy = frame.copy() 
    images = np.empty([mass_centers.shape[0],sub_image_size[0],sub_image_size[1]])

    i = 0
    for center in mass_centers:
        images[i,:,:] = frame_copy[int(center[0] - sub_image_size[0]/2): int(center[0] + sub_image_size[0]/2), int(center[1] - sub_image_size[1]/2): int(center[1] + sub_image_size[1]/2)]
        if plot is True:
            fig,ax = plt.subplots(1)
            ax.imshow(images[i,:,:])
            plt.show()
        i += 1
    return images    

def get_divide_sign(images, plot=False):

    for im in range(images.shape[0]):
        contours = find_contours(images[im],128,'high')
        if len(contours) == 3:
            image = images[im]
            image_idx = im
            if plot is True:
                plt.figure()
                plt.imshow(image, cmap= "gray")
                for contour in range(len(contours)):
                    plt.scatter(contours[contour][:,1],contours[contour][:,0])
            break

    if "image" in locals():
        return image, image_idx
    else:
        print("Image with 3 countours not found")
        return 0

def main_segmentation(frame, plot=False):

    frame_copy = frame.copy()
    norm_frame = normalize(frame_copy)   

    print(norm_frame)
    blue_arrow = get_blue_arrow(norm_frame)

    arrow_center = object_center(blue_arrow, plot=False)

    signs = get_red_black(norm_frame,plot=False)

    signs_without_robot = remove_robot(signs, arrow_center, width = 70)

    if plot is True:
        fig, ax = plt.subplots(3, 2, figsize=(30, 30))
        
        ax[0,0].set_title('First original frame')
        ax[0,0].imshow(frame, cmap='gray')
        ax[0,0].set_axis_off()

        ax[0,1].set_title('Normalized colours')
        ax[0,1].imshow(norm_frame, cmap='gray')
        ax[0,1].set_axis_off()

        ax[1,0].set_title('Blue arrow with center of gravity')
        ax[1,0].imshow(blue_arrow, cmap="gray")
        ax[1,0].scatter(arrow_center[1],arrow_center[0], s = 10, c='red', marker='o')
        ax[1,0].set_axis_off()

        ax[1,1].set_title('Black (and red) threshold')
        ax[1,1].imshow(signs, cmap="gray")
        ax[1,1].set_axis_off()

        ax[2,0].set_title('Image without robot')
        ax[2,0].imshow(signs_without_robot, cmap="gray")
        ax[2,0].set_axis_off()
        ax[2,1].set_axis_off()

        plt.tight_layout()
        plt.savefig("Signs.pdf")
        plt.show()

    edge_sobel = find_edges(signs_without_robot, plot=False)

    clean_contours = remove_lines(edge_sobel, width = 5, plot=plot)

    mass_centers = segmentation(clean_contours, width = 40, adjust = 40, plot=False)

    segmented_images = get_images(signs, mass_centers, sub_image_size = (40,40), plot=False)

    

    if plot is True and segmented_images.shape[0] <= 10:
        fig, ax = plt.subplots(5, 3, figsize=(30, 30))
        
        for i in range(segmented_images.shape[0]):
            ax[int(i/2),i%2].imshow(segmented_images[i,:,:], cmap='gray')
            ax[int(i/2),i%2].axis('off')
       
        
        ax[0,2].imshow(frame_copy, cmap = 'gray')
        ax[0,2].scatter(mass_centers[:,1], mass_centers[:,0], color='r')
        
        for i in range(1,5):
            ax[i,2].set_axis_off()
    
        plt.tight_layout()
        plt.savefig("Segmented_images.pdf")
        plt.show()

    return segmented_images


####################################
data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'robot_parcours_1'
data_path = os.path.join(data_base_path, data_folder + '.avi')
############################################

# Create a video capture object to read videos
cap = cv2.VideoCapture(data_path)   
 

# Read first frame
success, frame1 = cap.read()
images = main_segmentation(frame1, plot=False)
image, image_idx = get_divide_sign(images,plot = False)

def get_sign(robot_center, mass_centers):

    dist = sys.maxsize

    for center in mass_centers:
        temp_dist = np.linalg.norm(center-robot_center)

        if temp_dist < dist:
            dist = temp_dist
            closest_center = center

    if temp_dist < 20:
        close_enough = True
        
    else:
        close_enough = False
    

    return close_enough, closest_center

