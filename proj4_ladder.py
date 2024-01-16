import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def feature_detector(image_1,image_2):
    gray_image1 = cv.cvtColor(image_1,cv.COLOR_BGR2GRAY)
    gray_image2 = cv.cvtColor(image_2,cv.COLOR_BGR2GRAY)
    sift=cv.SIFT_create()
    kp1,des1=sift.detectAndCompute(gray_image1,None)
    kp2,des2=sift.detectAndCompute(gray_image2,None)
    bf=cv.BFMatcher()
    match=bf.knnMatch(des1,des2,k=2)
    good=[]
    cam1_points=[]
    cam2_points=[]
    for m,n in match:
        if m.distance<0.5*n.distance:
            good.append(m)
    for j in good:
        (pts_1_x,pts_1_y)=kp1[j.queryIdx].pt 
        (pts_2_x,pts_2_y)=kp2[j.trainIdx].pt
        cam1 = (pts_1_x,pts_1_y)
        cam2 = (pts_2_x,pts_2_y)
        cam1_points.append(cam1)
        cam2_points.append(cam2)

    cam1_pts=np.array(cam1_points)
    cam2_pts=np.array(cam2_points)
    return cam1_pts,cam2_pts

def fundemental_matrix(s, c):
    A_matrix = np.zeros((8,9),np.float32)
    for i in range(8):
        A_matrix[i][0] = campoints1[s[c]][0]*campoints2[s[c]][0]
        A_matrix[i][1] = campoints1[s[c]][0]*campoints2[s[c]][1]
        A_matrix[i][2] = campoints1[s[c]][0]
        A_matrix[i][3] = campoints1[s[c]][1]*campoints2[s[c]][0]
        A_matrix[i][4] = campoints1[s[c]][1]*campoints2[s[c]][1]
        A_matrix[i][5] = campoints1[s[c]][1]
        A_matrix[i][6] = campoints2[s[c]][0]
        A_matrix[i][7] = campoints2[s[c]][1]
        A_matrix[i][8] = 1
        c += 1
        # print("The A matrix is:")
        # print(A_matrix.shape)
        # break
    _,_,V_T = np.linalg.svd(A_matrix)
    # print(V_T)
    Fundemental_Matrix = V_T[-1]
    # print(Fundemental_Matrix)
    Fundemental_Matrix = np.reshape(Fundemental_Matrix,(3,3))
    # print("fundemental matrix")
    # print(Fundemental_Matrix)
    U,S,R = np.linalg.svd(Fundemental_Matrix)
    S[2]=0
    # singularDiagonalMatrix = np.array([[S[0],0,0],[0,S[1],0],[0,0,0]])
    funmat = U @ np.diag(S) @ R
    # print("reformed fundemental matrix")
    # print(funmat)
    return funmat

def ransac(campts1,campts2):
    max_iterations = 5000
    current_iteration = 0
    np.random.seed(0)
    sample = np.random.choice(len(campoints2),max_iterations*8)
    column = 0

    while current_iteration <max_iterations:  
        F_matrix = fundemental_matrix(sample, column)
        # print(campoints1.shape)
        one_vector_column = np.ones((len(campoints2),1))
        # print(one_vector_column)
        campts_1 = np.hstack([campts1,one_vector_column])
        # print(campoints1)
        campts_2 = np.hstack([campts2,one_vector_column])
        compute_Error_matrix = campts_2@F_matrix@campts_1.T
        Error_matrix = abs(np.diag(compute_Error_matrix))
        # print(Error_matrix) 
        inliers = []
        for i in Error_matrix:
            if(i < 0.5):
                inliers.append(i)
        if len(inliers) >1000:
            print(len(inliers))
            return F_matrix
        current_iteration+=1
        column += 8

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    
    
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    # image_1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
    r,c,_ = image_1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,(int(pt1[0]),int(pt1[1])),5,color,-1)
        img2 = cv.circle(img2,(int(pt2[0]),int(pt2[1])),5,color,-1)
    return img1,img2

def triangulation(Rotation,Translation,Intrinsic,pts1,pts2):
    P1 = np.dot(Intrinsic, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(Intrinsic, np.hstack((Rotation, Translation.reshape(-1, 1))))
    Sample1 = pts1[0:4]
    Sample2 = pts2[0:4]
    Triang_matrix = cv.triangulatePoints(P1, P2, np.asarray(Sample1).T, np.asarray(Sample2).T)
    Triang_matrix = Triang_matrix/Triang_matrix[-1]
    T = np.sum(Triang_matrix[2] > 0)
    return T



def Intrinsic_parameters():
    W_mat = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u,s,v = np.linalg.svd(Essential_Matrix)
    Rotation_1 = u@W_mat@v
    Rotation_2 = u@W_mat.T@v
    U3 = u[:,2]
    Translation1 = U3
    Translation2 = -U3
    K1 = triangulation(Rotation_1,Translation1,Intrinsic,campoints1,campoints2)
    K2 = triangulation(Rotation_1,Translation2,Intrinsic,campoints1,campoints2)
    K3 = triangulation(Rotation_2,Translation1,Intrinsic,campoints1,campoints2)
    K4 = triangulation(Rotation_2,Translation2,Intrinsic,campoints1,campoints2)
    maximum_K = max(K1, K2, K3, K4)
    if K1 == maximum_K:
        return Rotation_1, Translation1
    elif K2 == maximum_K:
        return Rotation_1, Translation2
    elif K3 == maximum_K:
        return Rotation_2, Translation1
    return Rotation_2, Translation2

def squared_distances(i, j, l):
    distance = 0
    for a in range(-1, 2):
        for b in range(-1, 2):
            distance += (int(left_rectified_gray[i+a][j+b]) -int(right_rectified_gray[i+a][b+j-l]))**2
    return distance

image_1 = cv.imread("ladderim0.png")
image_2 = cv.imread("ladderim1.png")
image1 = image_1.copy()
image2 = image_2.copy()
campoints1,campoints2 = feature_detector(image_1,image_2)
Fmatrix = ransac(campoints1,campoints2)
#print("Fundemental Matrix")
# print(Fmatrix)
Fmatrix = Fmatrix/Fmatrix[2][2]
print("Fundemental Matrix")
print(Fmatrix)
Intrinsic=np.array([[1734.16,0,333.49],[0,1734.16,958.05],[0,0,1]])
Essential_Matrix = Intrinsic.T@Fmatrix@Intrinsic
print("Essential Matrix")
print(Essential_Matrix)
Rotation,Translation = Intrinsic_parameters()
print("Roation Matrix:")
print(Rotation)
print("Translation:")
print(Translation)
l1 = cv.computeCorrespondEpilines(campoints2.reshape(-1,1,2), 2,Fmatrix)
l1 = l1.reshape(-1,3)
image_5,image_6 = drawlines(image1,image2,l1,campoints1,campoints2)
l2 = cv.computeCorrespondEpilines(campoints1.reshape(-1,1,2), 1,Fmatrix)
l2 = l2.reshape(-1,3)
image_3,image_4 = drawlines(image2,image1,l2,campoints2,campoints1)

imageSize = [1080,1920]
success,H1,H2 = cv.stereoRectifyUncalibrated(campoints1, campoints2, Fmatrix, imageSize)
print("H1 Homography Matrix")
print(H1)
print("H2 Homography Matrix")
print(H2)
left_epilines = cv.warpPerspective(image1,H1,[1080,1920])
right_epilines = cv.warpPerspective(image2,H2,[1080,1920])
left_rectified = cv.warpPerspective(image_1,H1,[1080,1920])
right_rectified = cv.warpPerspective(image_2,H2,[1080,1920])
cv.imshow("left_epilines",left_epilines)
cv.waitKey(0)
cv.imshow("right_epilines",right_epilines)
cv.waitKey(0)
cv.imshow("left_rectified",left_rectified)
cv.waitKey(0)
cv.imshow("right_rectified",right_rectified)
cv.waitKey(0)
cv.destroyAllWindows()
left_rectified_gray = cv.cvtColor(left_rectified,cv.COLOR_BGR2GRAY)
right_rectified_gray = cv.cvtColor(right_rectified,cv.COLOR_BGR2GRAY)
height1,width1 = left_rectified_gray.shape
disparity_error = np.zeros(shape=(width1,height1))
Depth_eval = np.zeros(shape=(width1,height1))
for i in np.arange(3, height1-3):
    for j in np.arange(3, width1-3):
        MinSqDist = math.inf
        for l in np.arange(0, 20, 3):
            # x, y = 0, 0
            dist = squared_distances(i, j, l)
            MinSqDist = min(MinSqDist, dist)
            if dist == MinSqDist:
                disp = l
        disparity_error[j][i] = disp
        if int(disp) == 0:
            disp = 0.0000000001
            Depth_eval[j][i] = 1734.16*228.38/disp
        else:
            Depth_eval[j][i] = 1734.16*228.38/disp
print("Minimun Disparity Error:")
print(np.max(disparity_error))
print("Maximum Disparity Error:")
print(np.min(disparity_error))
# disparity_normalized = cv.normalize(
#     disparity_error.T, None, 27, 85, cv.NORM_MINMAX, cv.CV_8U)
# disparity_colormap = cv.applyColorMap(disparity_normalized, cv.COLORMAP_JET)
# cv.imshow("disparity_grayscale.png", disparity_normalized)
# cv.imshow("color_map.png", disparity_colormap)
depth_normalized = cv.normalize(
    Depth_eval.T, None, 27, 85
    , cv.NORM_MINMAX, cv.CV_8U)
depth_colormap = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
cv.imshow("depth_grayscale.png", depth_normalized)
cv.imshow("color_map.png", depth_colormap)
cv.waitKey() & 0xFF == ord("q")
cv.destroyAllWindows()

