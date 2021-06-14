import cv2 
import cv2.aruco as aruco
import numpy as np
import os

def aug_img_path(path):
    my_images = os.listdir(path)
    dic={}
    #print(my_images)
    for img_path in my_images:
        key = int(os.path.splitext(img_path)[0])
        
        img_aug = cv2.imread(f'{path}/{img_path}')
        dic[key] = img_aug
        #print(dic)
    return dic


# Method to detect Marker and extract the Marker ID

def find_aruco_markers(img , markersize=6 , totalmarkers=250 , draw = True):
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # define so that our detector will look for the dictionary format
    # so we have to define 6 x 6 with 250 ids 
    # simple way 
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # this is the default value so we are changing to our dimension
    key = getattr(aruco,f'DICT_{markersize}X{markersize}_{totalmarkers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_parameter = aruco.DetectorParameters_create()
    bbox , ids , invalid = aruco.detectMarkers(imggray,aruco_dict,parameters=aruco_parameter)

    # print(ids)
    if draw:
        # this method will put a bbox to the detected marker
        aruco.drawDetectedMarkers(img,bbox)
    return [bbox , ids]



def augment_marker(bbox , ids , img , img_aug , draw_id=True):
    tl = bbox[0][0][0], bbox[0][0][1]  # top left
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]  # bottom left
    bl = bbox[0][3][0], bbox[0][3][1]

    h , w , c = img_aug.shape 

    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])

    matrix, _ = cv2.findHomography(pts2,pts1)
    imgout = cv2.warpPerspective(img_aug , matrix , (img.shape[1],img.shape[0]))
    # here the above imgout will just wrapy the marker and make the background full black
    # so now just want to overlay the marker part and make the environment real 
    # step 1 : making the marker area alone black 
    cv2.fillConvexPoly(img , pts1.astype(int),(0,0,0))
    imgout = img + imgout
    id = list(ids)
    # print(int(tl))
    # print(id)
    x1 = int(tl[0])
    y1 = int(tl[1])

    if draw_id:
        cv2.putText(imgout,str(id),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),1)
    return imgout



def main():
    cap = cv2.VideoCapture(0)
    # img_aug = cv2.imread('/home/ganesh/ar/markers/55.jpg')  # img name should only have ID 
    

    aug_img_dict =aug_img_path('/home/ganesh/ar/markers')
    while True:
        success,img = cap.read()
        aruco_marker_founded = find_aruco_markers(img)
        #print(aruco_marker_founded)
        
        #find_aruco_markers(img)

        # checking if bbox is empty ... if not loop over the ids and create augment for the id
        if len(aruco_marker_founded[0]) != 0:
            for bbox , ids in zip(aruco_marker_founded[0],aruco_marker_founded[1]):
                # print(f'ID is {ids}')
                try:
                    img = augment_marker(bbox , ids , img , aug_img_dict[int(ids)])
                except KeyError:
                    print(f"Key ID {ids} not found")

        
        cv2.imshow("Image",img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()

