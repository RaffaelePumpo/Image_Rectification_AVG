import numpy as np
import cv2
from matplotlib import pyplot as plt


def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])

filename = 'fig1_6c__.jpg'
img = cv2.imread(filename)

plt.imshow(img, cmap='gray')
pts = np.asarray(plt.ginput(4, timeout=-1))
plt.show()

print('chosen coord: ', pts)  # each row is a point

plt.plot(*zip(*pts), marker='o', color='r', ls='')
plt.imshow(img)
plt.show()

points = np.array([
    [1053.01298701,  460.10606061],
    [1364.7012987,   641.92424242],
    [1057.34199134,  867.03246753],
    [ 745.65367965,  641.92424242]
])

#points = pts # to overwrite the chosen points


'''
Affine rectification
'''
print('\n-------- Task 1: Affine rectification --------')
pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)  # convert chosen pts to homogeneous coordinate
print('Task 1.1: Identify image of the line at inf on projective plane')

print(pts_homo[0,:], pts_homo[1,:])
# choose the first horizontal line
hor_0 =  np.cross(pts_homo[0,:],pts_homo[1,:])

# the 2nd horizontal line
hor_1 = np.cross(pts_homo[2,:], pts_homo[3,:])

# first ideal point
pt_ideal_0 =  np.cross(hor_0, hor_1)
pt_ideal_0 /= pt_ideal_0[-1]  # normalize
print('@Task 1.1: first ideal point: ', pt_ideal_0)

# the 1st vertical line
ver_0 = np.cross(pts_homo[0,:], pts_homo[3,:])

# the 2nd vertical line
ver_1 = np.cross(pts_homo[1,:], pts_homo[2,:])

# 2nd ieal point
pt_ideal_1 = np.cross(ver_0,ver_1)
pt_ideal_1 /= pt_ideal_1[-1]
print('@Task 1.1: second ideal point: ', pt_ideal_1)

# image of line at inf
l_inf = np.cross(pt_ideal_0, pt_ideal_1)
l_inf /= l_inf[-1]
print('@Task1.1: line at infinity: ', l_inf)

print('Task 1.2: Construct the projectivity that affinely rectify image')
H = np.array([ [1, 0, 0],
              [0 ,1, 0],
              [l_inf[0], l_inf[1], l_inf[2]]

])



print('@Task 1.2: image of line at inf on affinely rectified image: ', (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

H_E = euclidean_trans(np.deg2rad(0), 50, 250)

affine_img = cv2.warpPerspective(img, H_E @ H, (img.shape[1], img.shape[0]))

print(H_E@H)
affine_pts = (H_E @ H @ pts_homo.T).T

for i in range(affine_pts.shape[0]):
    affine_pts[i] /= affine_pts[i, -1]

plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(affine_img)
plt.show()
print('-------- End of Task 1 --------\n')

'''
Task 2: Metric rectification
'''
print('\n-------- Task 2: Metric rectification --------')
print('Task 2.1: transform 4 chosen points from projective image to affine image')

print(affine_pts)
# image of first horizontal line on affine plane
aff_hor_0 = np.cross(affine_pts[0,:].T, affine_pts[1,:])
# image of 2nd horizontal line on affine plane
aff_hor_1 = np.cross(affine_pts[2,:], affine_pts[3,:]) 

# image of first vertical line on affine plane
aff_ver_0 = np.cross(affine_pts[0,:], affine_pts[3,:]) 
#image of 2nd vertical line on affine plane
aff_ver_1 = np.cross(affine_pts[1,:], affine_pts[2,:]) 

aff_hor_0 /= aff_hor_0[-1]
aff_hor_1 /= aff_hor_1[-1]
aff_ver_0 /= aff_ver_0[-1]
aff_ver_1 /= aff_ver_1[-1]
print('@Task 2.1: first chosen point coordinate')
print('\t\t on projective image: ', pts_homo[0])
print('\t\t on affine image: ', affine_pts[0])
print(affine_pts)



print('Task 2.2: construct constraint matrix C to find vector s')
C0 = np.array([aff_hor_0[0]*aff_ver_0[0], aff_hor_0[0]*aff_ver_0[1] + aff_hor_0[1]*aff_ver_0[0], aff_hor_0[1]*aff_ver_0[1]])

C1 = np.array([aff_hor_1[0]*aff_ver_1[0], aff_hor_1[0]*aff_ver_1[1]+ aff_hor_1[1]*aff_ver_1[0], aff_hor_1[1]*aff_ver_1[1]])

C = np.vstack([C0, C1])
print('@Task 2.2: constraint matrix C:\n', C)

print('Task 2.3: Find s by looking for the kernel of C (hint: SVD)')
#Apply SVD decomposition to C 
u,e,v = np.linalg.svd(C)
print(v)

s = v.T[:,-1]


print('@Task 2.3: s = ', s)
print('@Task 2.3: C @ s = \n', C @ s.reshape(-1, 1))
mat_S = np.array([
    [s[0], s[1]],
    [s[1], s[2]],
])
print('@Task 2.3: matrix S:\n', mat_S)

print('Task 2.4: Find the projectivity that do metric rectificaiton')
e,q = np.linalg.eig(mat_S)
print(q, e)
K = q * np.sqrt(np.diag(e))
print(K)
# Writing H
Kinv = np.linalg.inv(K)
H = np.array([ 
    [Kinv[0,0] , Kinv[0,1], 0],
    [Kinv[1,0], Kinv[1,1], 0] ,
    [0, 0, 1],
     ])
print(H)
aff_dual_conic = np.array([
    [s[0], s[1], 0],
    [s[1], s[2], 0],
    [0, 0, 0]
])
print('@Task 2.3: image of dual conic on metric rectified image: ', H @ aff_dual_conic @ H.T)

H_E = euclidean_trans(np.deg2rad(0), 30, 80)
H_fin = H_E @ H
eucl_img = cv2.warpPerspective(affine_img, H_fin, (img.shape[1], img.shape[0]))

eucl_pts = (H_fin @ affine_pts.T).T
for i in range(eucl_pts.shape[0]):
    eucl_pts[i] /= eucl_pts[i, -1]
plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(eucl_img)
plt.show()
