import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt



def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


def createMorphSequence (im1, im1_pts, im2, im2_pts, t_list, transformType):
    if transformType:
        #TODO: projective transforms
        T12=findProjectiveTransform(im1_pts,im2_pts)
        T21=findProjectiveTransform(im2_pts,im1_pts)
    else:
        # TODO: affine transforms
        T12 = findAffineTransform(im1_pts, im2_pts)
        T21 = findAffineTransform(im2_pts, im1_pts)
    ims = []
    for t in t_list:
        # TODO: calculate nim for each t
        T12_t=(1-t)*np.eye(3)+t*T12
        T21_t=(1-t)*T21+t*np.eye(3)
        im1_row,im1_col=im1.shape
        im2_row, im2_col = im2.shape
        newIm1=mapImage(im1,T12_t,[im2_row,im2_col])
        newIm2=mapImage(im2,T21_t,[im1_row,im1_col])
        nim=t*newIm1+(1-t)*newIm2
        ims.append(nim)
    return ims



def mapImage(im, T, sizeOutIm):

    im_new = np.zeros(sizeOutIm)
    im_new_rows,im_new_cols=sizeOutIm[0],sizeOutIm[1]
    T_inv=np.linalg.pinv(T)
    # create meshgrid of all coordinates in new image [x,y]
    mesh_gridx, mesh_gridy =np.meshgrid(np.arange(0,im_new_rows, 1), np.arange(0, im_new_cols, 1))
    # add homogenous coord [x,y,1]
    mesh_gridx = mesh_gridx.ravel()
    mesh_gridy = mesh_gridy.ravel()
    ones_vec = np.ones((1, sizeOutIm[0] * sizeOutIm[1]))
    hom_coordinates=np.vstack((mesh_gridx,mesh_gridy,ones_vec))

    # calculate source coordinates that correspond to [x,y,1] in new imag
    mapped_coordinate=np.matmul(T_inv,hom_coordinates)
    mapped_coordinate[0,:]=mapped_coordinate[0,:]/mapped_coordinate[2,:]
    mapped_coordinate[1, :] = mapped_coordinate[1, :] / mapped_coordinate[2, :]
    mapped_coordinate = np.delete(mapped_coordinate, 2, 0)
    # find coordinates outside range and delete (in source and target)
    to_delete = []
    for i in range(im_new_rows * im_new_cols):
        if mapped_coordinate[0][i] < 0 or mapped_coordinate[0][i] > im_new_rows-1 or mapped_coordinate[1][i] < 0 or mapped_coordinate[1][i] > im_new_cols-1:
            to_delete.append(i)
    to_delete=np.array(to_delete)
    coordinates = np.delete(hom_coordinates, 2, 0)
    if len(to_delete)>0:
        mapped_coordinate=np.delete(mapped_coordinate,to_delete,axis=1)
        coordinates=np.delete(coordinates,to_delete,axis=1)
    # interpolate - bilinear
    #______________________________________ TODO:
    x_cordinates=mapped_coordinate[0,:]
    y_cordinates=mapped_coordinate[1,:]
    x_left=np.floor(x_cordinates).astype(int)
    x_right=np.ceil(x_cordinates).astype(int)
    y_top=np.floor(y_cordinates).astype(int)
    y_bottom=np.ceil(y_cordinates).astype(int)
    delta_x=x_cordinates-x_left
    delta_y=y_cordinates-y_top
    NW=im[x_left,y_top]
    SW=im[x_left,y_bottom]
    NE=im[x_right,y_top]
    SE=im[x_right,y_bottom]
    S=SE*delta_x+SW*(1-delta_x)
    N=NE*delta_x+NW*(1-delta_x)
    v=N*delta_y+S*(1-delta_y)
    #_____________________________________
    # apply corresponding coordinates
    # new_im [ target coordinates ] = old_im [ source coordinates ]
    orig_x=coordinates[0,:]
    orig_x=orig_x.astype(int)
    orig_y = coordinates[1, :]
    orig_y=orig_y.astype(int)
    for i in range(len(orig_x)):
        im_new[orig_x[i],orig_y[i]]=v[i]
    return im_new



def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    x=np.zeros((2*N,8))
    x_tag=np.zeros((2*N))
	# iterate iver points to create x , x'
    for i in range(0, N):
        x_tag[2 * i]=pointsSet2[i][0]
        x_tag[2 * i + 1] = pointsSet2[i][1]
        x[2 * i][0] = pointsSet1[i][0]
        x[2 * i][1] = pointsSet1[i][1]
        x[2 * i][4] = 1
        x[2 * i][6] = -1 * pointsSet1[i][0] * pointsSet2[i][0]
        x[2 * i][7] = -1 * pointsSet1[i][1] * pointsSet2[i][0]
        #-----------------------------------------------------
        x[2 * i + 1][2] = pointsSet1[i][0]
        x[2 * i + 1][3] = pointsSet1[i][1]
        x[2 * i + 1][5] = 1
        x[2 * i + 1][6] = -1 * pointsSet1[i][0] * pointsSet2[i][1]
        x[2 * i + 1][7] = -1 * pointsSet1[i][1] * pointsSet2[i][1]
    T=np.matmul(np.linalg.pinv(x),x_tag)
    ones=np.ones(1)
    T=np.hstack((T,ones))
    T=T.reshape((3,3))
    temp = T[0][2]  # C
    T[0][2] = T[1][1]
    T[1][1] = T[1][0]
    T[1][0] = temp
    # calculate T - be careful of order when reshaping it
    return T


def findAffineTransform(pointsSet1, pointsSet2):
    T = np.zeros((3, 3))
    N = pointsSet1.shape[0]
    x=np.zeros((N*2,6))
    x_tag=np.zeros(N*2)
	# iterate iver points to create x , x'
    for i in range(0, N):
        x_tag[i*2] = pointsSet2[i][0]
        x_tag[i*2+1] = pointsSet2[i][1]
        x[i*2][0] = pointsSet1[i][0]
        x[i*2][1] = pointsSet1[i][1]
        x[i * 2][4] =1
        x[i*2+1][2] = pointsSet1[i][0]
        x[i*2+1][3] = pointsSet1[i][1]
        x[i * 2 + 1][5] = 1

    # calculate T - be careful of order when reshaping it
    T=  np.matmul(np.linalg.pinv(x),x_tag)
    zeros = np.zeros((1, 3))
    zeros[0][2]=1
    T=T.reshape((2,3))
    T=np.vstack((T,zeros))
    temp =T[0][2]#C
    T[0][2]=T[1][1]
    T[1][1]=T[1][0]
    T[1][0]=temp
    return T



def getImagePts(im1, im2,varName1,varName2, nPoints):
    plt.figure()
    plt.title("first image points selection")
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    imagePts1 = plt.ginput(n=nPoints,show_clicks=True)
    imagePts1 =[(t[1], t[0]) for t in imagePts1]
    plt.figure()
    plt.title("second image points selection")
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    imagePts2 = plt.ginput(n=nPoints,show_clicks=True)
    imagePts2 = [(t[1], t[0]) for t in imagePts2]
    ones=np.ones((nPoints,1))
    imagePts1 = np.round(imagePts1)
    imagePts2 = np.round(imagePts2)
    imagePts1 = np.hstack((imagePts1,ones))
    imagePts2 = np.hstack((imagePts2,ones))
    np.save(varName1+".npy", imagePts1)
    np.save(varName2+".npy", imagePts2)

