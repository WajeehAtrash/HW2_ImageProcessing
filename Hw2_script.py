import numpy as np

from hw2_functions import *

if __name__ == "__main__":
    print("a---------------------------------------------")
    path_image = r'FaceImages\Face1.tif'
    face1 = cv2.imread(path_image)
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    path_image = r'FaceImages\Face2.tif'
    face2 = cv2.imread(path_image)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    imagepts1=np.load("imPoints1_sectionA.npy")
    imagepts2 = np.load("imPoints2_sectionA.npy")
    ims=createMorphSequence(face1_gray,imagepts1,face2_gray,imagepts2,np.linspace(0,1,100),0)
    writeMorphingVideo(ims,"face1_face2")
    print("b---------------------------------------------")
    path_image = r'lined.tif'
    card = cv2.imread(path_image)
    card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    path_image = r'sloped.tif'
    sloped_card = cv2.imread(path_image)
    sloped_card = cv2.cvtColor(sloped_card, cv2.COLOR_BGR2GRAY)
    # getImagePts(card,sloped_card,'points_b1','points_b2',4)
    imagepts1=np.load('imPoints1_sectionB.npy')
    imagepts2=np.load('imPoints2_sectionB.npy')
    T_projective=findProjectiveTransform(imagepts2,imagepts1)
    projective_image=mapImage(sloped_card,T_projective,card.shape)
    T_Affine=findAffineTransform(imagepts2,imagepts1)
    affined_image=mapImage(sloped_card,T_Affine,card.shape)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(sloped_card, cmap='gray', vmin=0, vmax=255)
    plt.title("source")
    plt.subplot(1, 3, 2)
    plt.imshow(projective_image, cmap='gray', vmin=0, vmax=255)
    plt.title("projective transforme")
    plt.subplot(1, 3, 3)
    plt.imshow(affined_image, cmap='gray', vmin=0, vmax=255)
    plt.title("affine transform")
    print('c---------------------------------------------')
    print('i---------------------------------------------')
    path_image = r'FaceImages\Face1.tif'
    face1 = cv2.imread(path_image)
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    path_image = r'FaceImages\Face2.tif'
    face2 = cv2.imread(path_image)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    # getImagePts(face1_gray, face2_gray, 'imPoints1_sectionC', 'imPoints2_sectionC', 4)
    imagepts1 = np.load("imPoints1_sectionC.npy")
    imagepts2 = np.load("imPoints2_sectionC.npy")
    ims_low = createMorphSequence(face1_gray, imagepts1, face2_gray, imagepts2, np.linspace(0, 1, 2),0)
    # getImagePts(face1_gray, face2_gray, 'imPoints1_sectionC_high', 'imPoints2_sectionC_high', 10)
    imagepts1 = np.load("imPoints1_sectionC_high.npy")
    imagepts2 = np.load("imPoints2_sectionC_high.npy")
    ims_high = createMorphSequence(face1_gray, imagepts1, face2_gray, imagepts2, np.linspace(0, 1, 2),0)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(ims_low[1], cmap='gray', vmin=0, vmax=255)
    plt.title("low points number")
    plt.subplot(1, 3, 2)
    plt.imshow(ims_high[1], cmap='gray', vmin=0, vmax=255)
    plt.title("high points number")
    print('ii--------------------------------------------')
    path_image = r'FaceImages\Face1.tif'
    face1 = cv2.imread(path_image)
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    path_image = r'FaceImages\Face2.tif'
    face2 = cv2.imread(path_image)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    # getImagePts(face1_gray, face2_gray, 'imPoints1_sectionCI_LOC', 'imPoints2_sectionCI_LOC', 5)
    imagepts1 = np.load("imPoints1_sectionCI_LOC.npy")
    imagepts2 = np.load("imPoints2_sectionCI_LOC.npy")
    ims_low = createMorphSequence(face1_gray, imagepts1, face2_gray, imagepts2, np.linspace(0, 1, 2), 0)
    # getImagePts(face1_gray, face2_gray, 'imPoints1_sectionCI2_LOC', 'imPoints2_sectionCI2_LOC', 5)
    imagepts1 = np.load("imPoints1_sectionCI2_LOC.npy")
    imagepts2 = np.load("imPoints2_sectionCI2_LOC.npy")
    ims_high = createMorphSequence(face1_gray, imagepts1, face2_gray, imagepts2, np.linspace(0, 1, 2), 0)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(ims_low[1], cmap='gray', vmin=0, vmax=255)
    plt.title("same points location")
    plt.subplot(1, 3, 2)
    plt.imshow(ims_high[1], cmap='gray', vmin=0, vmax=255)
    plt.title("different points location")
    plt.show()