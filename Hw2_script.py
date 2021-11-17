from hw2_functions import *

if __name__ == "__main__":
    path_image = r'FaceImages\Face1.tif'
    face1 = cv2.imread(path_image)
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    path_image = r'FaceImages\Face2.tif'
    face2 = cv2.imread(path_image)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    # getImagePts(face1_gray,face2_gray,"varname1","varname2",12)
    pointsSet1=np.load('varname1.npy')
    pointsSet2=np.load('varname2.npy')
    # T=findAffineTransform(pointsSet1,pointsSet2)
    # image =mapImage(face1_gray,T,[200,250])
    # plt.figure()
    # plt.title("our shit")
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    ims=createMorphSequence(face1_gray,pointsSet1,face2_gray,pointsSet2,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],1)
    for i in range(len(ims)):
        plt.figure()
        plt.title("our shit")
        plt.imshow(ims[i], cmap='gray', vmin=0, vmax=255)
        plt.show()
    writeMorphingVideo(ims,'oushitVideo')