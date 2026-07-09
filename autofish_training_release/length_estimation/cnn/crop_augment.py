import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

#Source: https://stackoverflow.com/questions/75388906/how-to-rotate-and-translate-an-image-with-opencv-without-losing-off-screen-data

def get_rotation_mat(image, angle, cx, cy):
    w, h = (image.shape[1], image.shape[0])
    #cx, cy = (w//2,h//2)

    M = cv2.getRotationMatrix2D((cx, cy), -1*angle, 1.0)
    #rotated = cv2.warpAffine(image, M, (w,h))
    return M

def get_translation_mat(d_x, d_y):
    M = np.float64([
        [1, 0, d_x],
        [0, 1, d_y]
    ])

    #return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return M

def chain_affine_transformation_mats(M0, M1):
    """
    Chaining affine transformations given by M0 and M1 matrices.
    M0 - 2x3 matrix applying the first affine transformation (e.g rotation).
    M1 - 2x3 matrix applying the second affine transformation (e.g translation).
    The method returns M - 2x3 matrix that chains the two transformations M0 and M1 (e.g rotation then translation in a single matrix).
    """
    T0 = np.vstack((M0, np.array([0, 0, 1])))  # Add row [0, 0, 1] to the bottom of M0 ([0, 0, 1] applies last row of eye matrix), T0 is 3x3 matrix.
    T1 = np.vstack((M1, np.array([0, 0, 1])))  # Add row [0, 0, 1] to the bottom of M1.
    T = T1 @ T0  # Chain transformations T0 and T1 using matrix multiplication.
    M = T[0:2, :]  # Remove the last row from T (the last row of affine transformations is always [0, 0, 1] and OpenCV conversion is omitting the last row).
    return M

def crop_augment(image, max_augs=4, retain_biggest=True):
    mask = np.argwhere(image != 0)
    center = np.mean(mask, axis=0)[:2]

    height = np.max(mask[:,0])-np.min(mask[:,0])
    width = np.max(mask[:,1])-np.min(mask[:,1])

    new_img = copy.deepcopy(image)
    num_augs = np.random.randint(low=0.0, high=max_augs+1)
    for k in range(num_augs):
        angle = np.random.uniform(low=0.0, high=360.0)
        d_x = np.random.uniform(low=-(height*0.4), high=(height*0.4))
        d_y = np.random.uniform(low=-(width*0.4), high=(width*0.4))

        rotationM = get_rotation_mat(image, angle, cx=center[1], cy=center[0])  # Compute rotation transformation matrix
        translationM = get_translation_mat(d_x, d_y)  # Compute translation transformation matrix

        M = chain_affine_transformation_mats(translationM, rotationM)

        transformed_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        new_img[transformed_image != 0] = 0

    if(retain_biggest):
        thresh = np.zeros((new_img.shape[0],new_img.shape[1])).astype(np.uint8)
        thresh[new_img[:,:,0] > 0] = 255
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)

        c_mask = np.zeros_like(new_img)
        #cv2.drawContours(c_mask, c, -1, (255,255,255), thickness=-1)
        cv2.fillPoly(c_mask, pts =[c], color=(255,255,255))

        new_img[c_mask == 0] = 0

        # if(self.bbox_input):
        #     mask = np.where(image != 0)
        #     bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        
    return new_img, new_bbox, new_mask



if __name__ == "__main__":
    path = "01-00001.png"
    image = cv2.imread(path)

    for i in range(1000):
        new_img = crop_augment(image)
        plt.imshow(new_img)
        plt.show()
