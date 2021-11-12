from hw2_functions import *

if __name__ == "__main__":
    # path_image = r'FaceImages\Face1.tif'
    # face1 = cv2.imread(path_image)
    # face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    # path_image = r'FaceImages\Face2.tif'
    # face2 = cv2.imread(path_image)
    # face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    # getImagePts(face1_gray,face2_gray,"varname1","varname2",3)
    pointsSet1=np.load('varname1.npy')
    pointsSet2=np.load('varname2.npy')
    T=findProjectiveTransform(pointsSet1,pointsSet2)
