import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

class Checkerboard():
    def __init__(self, img_path, calibrated_cam=None, undistort=False):
        self.calibrated_cam = calibrated_cam
        self.undistort = undistort
        self.square_size_mm = 40.0
        self.checker_size = (9,6)
        self.img_path = img_path
        self.img = self.load_img(img_path)
        self.obj_pts = self.setup_obj_points()
        self.img_pts = self.detect_img_points()
        self.Rs = None
        self.t = None

        # Estimate extrinsics
        if(self.img_pts is not None):
            if(undistort):
                dist = None
            else:
                dist = self.calibrated_cam.dist
            
            ret, self.Rs, self.t = cv2.solvePnP(self.obj_pts,
                                                self.img_pts,
                                                self.calibrated_cam.cam_mat,
                                                dist)
            

    def load_img(self, path):
        img = cv2.imread(self.img_path)

        if(self.undistort and self.calibrated_cam is not None):
            img = cv2.undistort(img,
                                self.calibrated_cam.cam_mat,
                                self.calibrated_cam.dist,
                                None,
                                self.calibrated_cam.cam_mat)
        
        return img

            
    # Try to shift the obj points to account for checkerboard thickness
    # and recalculate image points and the homography
    def get_shifted_img_points(self, cb_thickness):
        if(self.Rs is None or self.t is None):
            return None
        
        shifted_obj_pts = copy.deepcopy(self.obj_pts)
        shifted_obj_pts[:,2] = -cb_thickness # in mm

        img_pts_proj, _ = cv2.projectPoints(shifted_obj_pts,
                                            self.Rs,
                                            self.t,
                                            self.calibrated_cam.cam_mat,
                                            self.calibrated_cam.dist)


        # Update img points and homography
        #self.img_pts = img_pts_proj
        #self.homography = self.calc_homography()        
        return img_pts_proj


    def setup_obj_points(self):
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
        obj_pts = np.zeros((self.checker_size[0]*self.checker_size[1],3), np.float32)
        obj_pts[:,:2] = np.mgrid[0:self.checker_size[0],0:self.checker_size[1]].T.reshape(-1,2)
        obj_pts *= self.square_size_mm
        return obj_pts


    # Find the chess board corners
    def detect_img_points(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.checker_size[0], self.checker_size[1]), None)
        if(ret):
            return corners
        return None

    def create_transform_matrix(self, R, t):
        T = np.eye(4)
        T[:3,:3],_ = cv2.Rodrigues(R)
        T[:3,-1] = t.reshape(3,)
        return T

    def transform_3d_points(self, pts, R, t):
        # Get transformation
        T = self.create_transform_matrix(R,t)

        # Convert to homogenous coords
        pts_homo = np.ones((pts.shape[0], pts.shape[1]+1))
        pts_homo[:,:3] = pts

        # Apply transformation
        transformed = np.dot(T, np.transpose(pts_homo))
        transformed = np.transpose(transformed)

        # From homogenous to non-homogenous
        w = transformed[:,-1].reshape(-1,1)
        transformed = transformed / w
        return transformed[:,:3]

    def get_obj_pts_camera_coords(self):
        if(self.Rs is None or self.t is None):
            return None
        transformed_pts = self.transform_3d_points(self.obj_pts, self.Rs, self.t)
        return transformed_pts


    

    # def get_pixels_per_cm(self):
    #     if(self.img_pts is None):
    #         print("FAILED TO INIT IMAGE POINTS!!")
    #         return

    #     # Extract corners
    #     c1_img = np.array(self.img_pts[0][0])
    #     c2_img = np.array(self.img_pts[-1][0])

    #     c1_obj = np.array(self.obj_pts[0])
    #     c2_obj = np.array(self.obj_pts[-1])

    #     # Calc lengths
    #     dist_img = np.linalg.norm(c1_img - c2_img)
    #     dist_obj = np.linalg.norm(c1_obj - c2_obj)

    #     pixels_per_mm = dist_img / dist_obj
    #     pixels_per_cm = pixels_per_mm*10.0
    #     return pixels_per_cm

    # def get_length_naive(self, pts):
    #     if(self.calibrated_cam is not None):
    #         img_pts = pts.reshape(1,-1,2)
    #         img_pts = self.calibrated_cam.undistort_points(img_pts).reshape(-1,2)
    #         img_dist = np.sum(np.linalg.norm((img_pts[1:] - img_pts[:-1]), axis=1))
    #     return img_dist / self.get_pixels_per_cm()

    # def get_length_homography(self, pts, undistort=False):
    #     # Map points from image coordinates to checkerboard
    #     img_pts = copy.deepcopy(pts).reshape(1,-1,2)

    #     if(self.calibrated_cam is not None and undistort):
    #         img_pts = self.calibrated_cam.undistort_points(img_pts).reshape(1,-1,2)

    #     # Get length in cm
    #     obj_pts = cv2.perspectiveTransform(img_pts, self.homography).reshape(-1,2)
    #     obj_dist = np.sum(np.linalg.norm((obj_pts[1:] - obj_pts[:-1]), axis=1))
    #     obj_dist /= 10.0 # convert from mm to cm!

    #     # Get length in pixels
    #     img_pts = img_pts.reshape(-1,2)
    #     img_dist = np.sum(np.linalg.norm((img_pts[1:] - img_pts[:-1]), axis=1))
    #     return obj_dist, img_dist

    # def get_reprojection_error(self, pts):
    #     img_pts = copy.deepcopy(pts).reshape(1,-1,2)

    #     # Undistort points
    #     if(self.calibrated_cam is not None):
    #         img_pts = self.calibrated_cam.undistort_points(img_pts).reshape(1,-1,2)

    #     org_img_pts = copy.deepcopy(img_pts)

    #     # Project img pts to obj pts and back again
    #     obj_pts = cv2.perspectiveTransform(img_pts, self.homography) #.reshape(-1,2)
    #     reproj_img_pts = cv2.perspectiveTransform(obj_pts, np.linalg.inv(self.homography)) #.reshape(-1,2)

    #     # Calc reprojection error
    #     error = np.mean(np.linalg.norm(org_img_pts-reproj_img_pts, axis=1))
    #     return error


    # def calc_homography(self):
    #     src_pts = self.img_pts.reshape(-1,1,2)
    #     dst_pts = self.obj_pts[:,:2].reshape(-1,1,2)
    #     homo, inliers = cv2.findHomography(src_pts, dst_pts)
    #     return homo


    # def get_mean_img_points(self):
    #     if(self.img_pts is None):
    #         self.img_pts = self.setup_img_points()
    #     return np.mean(self.img_pts, axis=0)[0]

    # def get_dist2pts(self, pts):
    #     fish_mean = np.mean(pts, axis=0)[0]
    #     checkerboard_mean = self.get_mean_img_points()
    #     return np.linalg.norm(fish_mean - checkerboard_mean)
