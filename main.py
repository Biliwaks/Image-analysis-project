import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from skimage.measure import find_contours
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm
import skimage.io
import mahotas
from skimage.draw import polygon
from skimage.transform import resize, rotate
import tarfile


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
    """ This function removes the black noise created by the robot's shadow by putting
       to zero all pixels in a square of size 'width' centered on the robot """
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
    """ This function get the edges of the frame by applying a sobel filter """
    frame_copy = frame.copy()

    edge_sobel = filters.sobel(frame_copy)
    edge_sobel *= 255/edge_sobel.max()
    edge_sobel = edge_sobel.astype(int)

    if plot is True:
        plt.figure()
        plt.imshow(edge_sobel)
    
    return edge_sobel

def remove_lines(frame, width, plot=False):
    """ This function removes all unwanted lines resulting from the sobel filtering by 
        applaying a hough transform on the image. Shortest distance 'dist' from the origin to the 
        line as well as the angle is found. The equation x*cos(angle)+y*sin(angle) = dist is used
        to find points on the line that intersects the image's border. Then, a rectangle (polygon)
        is placed over the line with a certain 'width' to remove the line. We use two different 
        line angle conditions to avoid having huge values for the coordinates and optimize
        computational time. """

    frame_copy = frame.copy()
    frame_plot = frame.copy()

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(frame_copy, theta=tested_angles)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

        if angle > -np.pi/4 and angle < np.pi/4 :
            x0, x1 = 0, frame_copy.shape[0]
            y0, y1 = (dist - np.array([x0,x1]) * np.sin(angle))/np.cos(angle)

            # Get coordinates of the polygon
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
                    
        # Second angle condition
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
    """ This function translates a square of side width until it finds a sufficient amount of pixels.
        Once it has sufficient pixels, it adjusts the square to capture the maximum pixels. Finally,
        it looks for the center of mass of these pixels and finds the center of mass of the digit
        or operator  in the image referential"""
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

           
            
            
            

            if  total <= 30:
                continue
            
            print(total)
            fig,ax = plt.subplots(1)
            ax.imshow(square, cmap = 'gray')
            plt.show()

            for x_adj in range(-int(adjust), int(adjust)):
                for y_adj in range(-int(adjust), int(adjust)):
                    square = frame_copy[width * x_step + x_adj: width * (x_step + 1) + x_adj, width * y_step + y_adj: width * (y_step + 1) + y_adj]
                    total2 = np.sum( square > 0) 
                    if  total2 > total:
                        total = total2
                        best_adj = (x_adj,y_adj) 
                        final_square = square

            
            final_square = frame_copy[width * x_step + best_adj[0]: width * (x_step + 1) + best_adj[0], width * y_step  + best_adj[1]: width * (y_step + 1) + best_adj[1]]
            fig,ax = plt.subplots(1)
            ax.imshow(final_square, cmap = 'gray')
            plt.show()
            
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
        return image_idx
    else:
        print("Image with 3 countours not found")
        return 0

def get_equal_sign(images, plot=False):

    for im in range(images.shape[0]):
        contours = find_contours(images[im],128,'high')
        if len(contours) == 2:
            image = images[im]
            image_idx = im
            if plot is True:
                plt.figure()
                plt.imshow(image, cmap= "gray")
                for contour in range(len(contours)):
                    plt.scatter(contours[contour][:,1],contours[contour][:,0])
            break

    if "image" in locals():
        return image_idx
    else:
        print("Image with 3 countours not found")
        return 0

def get_sign(robot_center, mass_centers):

    dist = sys.maxsize
    idx = 0

    for center in mass_centers:

        temp_dist = np.linalg.norm(center-np.array(robot_center))
        
        if temp_dist < dist:
            dist = temp_dist
            closest_center_idx = idx

        idx += 1

    print('dist: ',dist)
    if dist < 30:
        close_enough = True
        
    else:
        close_enough = False
    
    return close_enough, closest_center_idx

def main_segmentation(frame, plot=False):

    frame_copy = frame.copy()
    norm_frame = normalize(frame_copy)   

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

    mass_centers = segmentation(clean_contours, width = 40, adjust = 30, plot=True)

    segmented_images = get_images(signs_without_robot, mass_centers, sub_image_size = (30,30), plot=False)

    

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

    return segmented_images, mass_centers


####################################
#%%
def get_robot(frame):
    #BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Compute mask from HSV graph with complementary color
    #ok best blue HSV, to tune if bad brightness
    lower_red = np.array([150, 100,80])
    upper_red = np.array([200,255,255])

    # Blue Mask Threshold
    maskb = cv2.inRange(hsv, lower_red, upper_red)

    #Blue blob, threshold and construct de blob
    ret2,thresh2 = cv2.threshold(maskb,100,150,0)
    contours2,hierarchy2 = cv2.findContours(thresh2, 1, 2)
    thresh2 = cv2.erode(thresh2, None, iterations=2)
    thresh2 = cv2.dilate(thresh2, None, iterations=4)



    #Center computation, don't work if colors aren't detected
    M1 = cv2.moments(thresh2)
    if M1["m00"] != 0 :
        cx = int(M1["m10"] / M1["m00"])
        cy = int(M1["m01"] / M1["m00"])
    else : print ("No robot")


    
    return (cx, cy)
    
#%%

def zernike(img):
    zer_list=[]
    for images in img:
        feature = mahotas.features.zernike_moments(images, 10)
        zer_list.append([feature])
    zer_vect=np.array(zer_list)
    return zer_vect

#%%


#source : https://hal.inria.fr/hal-01656212/document
#%%computes the deviation of an image to a model
def deviation(image_zk,model):
    sum=0
    for i in range(np.size(model,1)):
        sum+=(image_zk[i]-model[0,i])**2        
    sigma=3*np.sqrt(sum/np.size(model,1))
    return sigma
#the lowest deviation is the most probable class, this compares all the deviations
def min_deviation(image_zk,zero,one,two,three,four,five,six,seven,eight,plus,div,times,equal):
    deviations=[deviation(image_zk,zero),deviation(image_zk,one),deviation(image_zk,two),deviation(image_zk,three),deviation(image_zk,four),deviation(image_zk,five),deviation(image_zk,six),deviation(image_zk,seven),deviation(image_zk,eight),deviation(image_zk,plus),deviation(image_zk,div),deviation(image_zk,times),deviation(image_zk,equal)]
    return deviations.index(min(deviations))

#%% compute the Zernike coefficients of the input image and runs the comparison routine
def classify(image):
    image_zk=mahotas.features.zernike_moments(image, 10)
    image_zk = image_zk.reshape(len(image_zk),-1)
    label=min_deviation(image_zk,zero,one,two,three,four,five,six,seven,eight,plus,div,times,equal)
    return label

#%%
def equation(list_path):
    list_char = []
    table_char = ["0","1","2","3","4","5","6","7","8","+","/","*","=","-"]
    
    for k in range(len(list_path)): #Separate "1" and "-"
        if list_path[k]==1 and k%2!=0 : list_path[k]=13
        
    for k in range(len(list_path)):
        idx = list_path[k]
        list_char.append(table_char[idx]) #List of operands as char

    nb_elements = len(list_path)

    if len(list_path)>0 and list_path[nb_elements-1] == 12 : #If we have equal
        #Calculer
        result = eval("".join(list_char[:-1])) #Calculation
        list_char.append(str(result)) #Add result to the list
 
    return "".join(list_char) #OUTPUT TO PRINT

#%%
#data_base_path = os.path.join(os.pardir, 'data')
'''data_base_path = os.pardir
data_folder = 'dataset'
tar_path = os.path.join(data_base_path, data_folder + '.tar.gz')
with tarfile.open(tar_path, mode='r:gz') as tar:
    tar.extractall(path=data_base_path)

#  Load data
im_path = os.path.join(data_base_path, data_folder)
im_names = [nm for nm in os.listdir(im_path) if '.png' in nm]  # make sure to only load .png
im_names.sort(key=lambda x: int(os.path.splitext(x)[0]))  # sort file names in ascending order
ic = skimage.io.imread_collection([os.path.join(im_path, nm) for nm in im_names])
im = skimage.io.concatenate_images(ic)
'''
#%%
'''labels=np.zeros(89034)
labels[0:5921]=0
labels[5922:12919]=10 # minus
labels[12920:19916]=11 #times
labels[19917:26916]=12 #div
labels[26917:33915]=13 #equal
labels[33916:40656]=1 #1
labels[40657:46614]=2
labels[46615:52744]=3
labels[52745:58585]=4
labels[58586:64003]=5
labels[64004:69920]=6
labels[69921:76183]=7
labels[76184:82033]=8 
labels[82034:89033]=9 #plus'''

#zernike_descr=zernike(im)

#this imports the pre-computed zernike coefficients to save time
zernike_descr=np.load('zernike_descr.npy') 

#compute the models for each digit, the model is the mean of the zernike coefs
zero=np.mean(zernike_descr[0:5921],0)
one=np.mean(zernike_descr[33916:40656],0)
two=np.mean(zernike_descr[40657:46614],0)
three=np.mean(zernike_descr[46615:52744],0)
four=np.mean(zernike_descr[52745:58585],0)
five=np.mean(zernike_descr[58586:64003],0)
six=np.mean(zernike_descr[64004:69920],0)
seven=np.mean(zernike_descr[69921:76183],0)
eight=np.mean(zernike_descr[76184:82033],0)

plus=np.mean(zernike_descr[82034:89033],0) #9
div=np.mean(zernike_descr[19917:26916],0) #10
times=np.mean(zernike_descr[12920:19916],0) #11
equal=np.mean(zernike_descr[26917:33915],0) #12

#%%
####################################
data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'robot_parcours_1'
data_path = os.path.join(data_base_path, data_folder + '.avi')
############################################
# Create a video capture object to read videos
cap = cv2.VideoCapture(data_path)   
 
# Read first frame
success, frame_init = cap.read()
width, height = frame_init.shape[1], frame_init.shape[0]

#frame_init_copy = normalize(frame_init)
#arrow_frame = get_blue_arrow(frame_init_copy,True)
#cx, cy = object_center(arrow_frame)
cx, cy = get_robot(frame_init)


# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

images, mass_centers = main_segmentation(frame_init, plot=False)

label = np.empty(images.shape[0])



for im in range(images.shape[0]):
    image_resized = resize(images[im], (28, 28),anti_aliasing=True)
    label[im] = classify(image_resized)

image_idx = get_divide_sign(images,plot = True)
label[image_idx] = 10

image_idx = get_equal_sign(images,plot = True)
label[image_idx] = 12
    
######Faire les calculs d'init avec la première fram ici

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 2.0, (width,height))
robot_posx = []
robot_posy = []
list_path = []
center_done = []
frame_count = 0

out.write(frame_init)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame_count += 1
    ########################Faire les opérations avec les frame pendant la vidéo

    #Get robot position and store
    #frame_copy = normalize(frame)
    #arrow_frame = get_blue_arrow(frame_copy)
    #cx, cy = object_center(arrow_frame)
    cx, cy = get_robot(frame)
    robot_posx.append(int(cx))
    robot_posy.append(int(cy))
    
    close_enough, closest_center_idx = get_sign((cy,cx), mass_centers)
    
    if close_enough == True:
        if closest_center_idx not in center_done:
            list_path.append(int(label[closest_center_idx]))
            center_done.append(closest_center_idx)
    print('close_enough: ', close_enough)
    print('center_done: ', center_done)
    print('closest_center_idx: ', closest_center_idx)
    #Draw line
    for i in range(frame_count-1) :
        if i>0 :
            cv2.line(frame, (robot_posx[i-1], robot_posy[i-1]), (robot_posx[i], robot_posy[i]), (0, 255, 0), thickness=2)
 
    print('list_path', list_path)
    
    equation_string = equation(list_path)
    
    #Write text
    position = (int(width/10),height-50)
    cv2.putText(
         frame, #numpy array on which text is written
         equation_string, #text
         position, #position at which writing has to start
         cv2.FONT_HERSHEY_SIMPLEX, #font family
         2, #font size
         (20, 255, 20, 20), #font color
         3) #font stroke
        
    out_frame=frame
    


    plt.figure()
    plt.imshow(out_frame)
    plt.show()
    
    #Save video
    out.write(out_frame)


# When everything done, release the capture
print("stop")
cap.release()
out.release()
cv2.destroyAllWindows()
    

for i in range(images.shape[0]):

    plt.figure()
    plt.imshow(images[i],cmap='gray')
    plt.show()

# %%
