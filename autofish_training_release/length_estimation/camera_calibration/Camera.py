import numpy as np
import cv2
import glob
import argparse
import copy
import json

# https://github.com/leomariga/pyRANSAC-3D?tab=readme-ov-file
import pyransac3d as pyrsc

class Camera():
# Based on:
# https://docs.opencv.org/4.5.5/dc/dbb/tutorial_py_calibration.html

    def __init__(self):
        self.square_size_mm = 40.0
        self.checker_size = (9,6)
        self.term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.cam_mat = None
        self.dist = None
        self.cam_mat_optim = None
        self.roi = None
        self.obj_pts = None
        self.img_pts = None

        self.Rs = None
        self.ts = None

        self.homography = None
        self.length_undistorted = False
        self.plane_thresh = 2.5 #mm

    def load_from_json(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

            self.cam_mat = np.eye(3)
            self.cam_mat[0][0] = data["fx"]
            self.cam_mat[1][1] = data["fy"]
            self.cam_mat[0][2] = data["cx"]
            self.cam_mat[1][2] = data["cy"]

            self.dist = np.zeros((1,5))
            self.dist[0,:] = data["dist_coeff"]


    # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    def project_point2plane(self, point, plane_eq):
        n = np.array(plane_eq[:3])
        d = np.array(plane_eq[3:])
        p = np.array(point).reshape(-1,3)

        dot = np.dot(p,n)+d

        n_dot = n*dot.reshape(-2,1)
        p_plane = p - n_dot
        return p_plane

    # Calc homograpy for length estimation
    # based on all the detected checkerboards
    def calc_homography(self):
        # Fit plane to all checkerboards in camera space
        plane = pyrsc.Plane()
        cam_pts = self.get_obj_pts_camera_coords()
        img_pts = copy.deepcopy(self.img_pts)
        points = np.concatenate(cam_pts, axis=0 )
        plane_eq, plane_inliers = plane.fit(points, self.plane_thresh)

        # Offset to account for checkerboard thickness
        #plane_eq[-1] += 22.0

        # Indentify inliers using a mask
        mask = np.zeros(points.shape[0], dtype=int)
        mask[plane_inliers] = 1
        mask = mask.reshape(len(cam_pts),-1)

        # Project inliers to plane
        # and find img inliers
        inliers_plane = np.empty((0,3))
        inliers_img = np.empty((0,2))

        for k,m in enumerate(mask):
            indices = np.argwhere(m>0)
            curr_cam_p = cam_pts[k][indices]
            if(len(curr_cam_p) == 0):
                continue

            plane_proj = self.project_point2plane(curr_cam_p, plane_eq)

            #inliers_plane = np.concatenate((inliers_plane, plane_proj), axis=0)

            # img_pts_proj, _ = cv2.projectPoints(plane_proj,
            #                                     np.eye(3), #self.Rs[k],
            #                                     np.zeros(3), #self.ts[k],
            #                                     self.cam_mat,
            #                                     self.dist)

            # print("org: ", img_pts[k][indices])
            # print("new: ", img_pts_proj)

            #curr_img_pts = img_pts_proj.reshape(-1,2)
            curr_img_pts = img_pts[k][indices].reshape(-1,2)
            inliers_img = np.concatenate((inliers_img, curr_img_pts), axis=0)


            from skspatial.objects import Line, Plane
            from skspatial.plotting import plot_3d

            ## Calc ray-plane intersection
            # Calc rays
            print("calc rays!")
            undist_pts = cv2.undistortPoints(curr_img_pts, self.cam_mat, self.dist) #, P=self.cam_mat)
            print(undist_pts.shape)
            rays = np.ones((len(undist_pts),3))
            rays[:,:2] = undist_pts.reshape(-1,2)
            print(rays.shape)
            print("before; ", rays)
            rays /= np.linalg.norm(rays,axis=1).reshape(-1,1)
            print("after; ", rays)

            plane = Plane(point=plane_proj[0], normal=plane_eq[:3])

            for r in rays:
                line = Line(point=[0,0,0], direction=r)
                plane_reproj = plane.intersect_line(line).reshape(1,3)
                inliers_plane = np.concatenate((inliers_plane, plane_reproj), axis=0)

        # Calc homography
        src_pts = inliers_img.reshape(-1,1,2)
        if(self.length_undistorted):
            src_pts = self.undistort_points(src_pts)
        dst_pts = inliers_plane[:,:2].reshape(-1,1,2)

        print("src: ", src_pts.shape)
        print("dst: ", dst_pts.shape)

        homo, inliers = cv2.findHomography(src_pts, dst_pts)
        return homo


    def get_length_homography(self, pts):
        if(self.homography is None):
            self.homography = self.calc_homography()

        # Map points from image coordinates to checkerboard
        img_pts = copy.deepcopy(pts).reshape(1,-1,2)

        # Undistort if selected
        if(self.length_undistorted):
            img_pts = self.undistort_points(img_pts).reshape(1,-1,2)

        # Get length in cm
        obj_pts = cv2.perspectiveTransform(img_pts, self.homography).reshape(-1,2)
        obj_dist = np.sum(np.linalg.norm((obj_pts[1:] - obj_pts[:-1]), axis=1))
        obj_dist /= 10.0 # convert from mm to cm!

        # Get length in pixels
        img_pts = img_pts.reshape(-1,2)
        img_dist = np.sum(np.linalg.norm((img_pts[1:] - img_pts[:-1]), axis=1))
        return obj_dist, img_dist



    def setup_obj_points(self):
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
        obj_pts = np.zeros((self.checker_size[0]*self.checker_size[1],3), np.float32)
        obj_pts[:,:2] = np.mgrid[0:self.checker_size[0],0:self.checker_size[1]].T.reshape(-1,2)
        obj_pts *= self.square_size_mm
        return obj_pts

    def calc_reproj_error(self, obj_pts, img_pts, cam_mat, dist, Rs, ts):
        all_errors = []

        for i in np.arange(len(obj_pts)):
            img_pts_proj, _ = cv2.projectPoints(obj_pts[i], Rs[i], ts[i], cam_mat, dist)
            error = cv2.norm(img_pts[i], img_pts_proj, cv2.NORM_L2)/len(img_pts_proj)
            all_errors.append(error)
        return np.mean(np.array(all_errors))

    def undistort_img(self, img):
        # Undistort image
        if(self.cam_mat_optim is not None):
            img_undist = cv2.undistort(img, self.cam_mat, self.dist, None, self.cam_mat_optim)

            # Crop and return image
            x, y, w, h = self.roi
            return img_undist[y:y+h, x:x+w]
        else:
            img_undist = cv2.undistort(img, self.cam_mat, self.dist, None, self.cam_mat)
            return img_undist


    def undistort_points(self, points):
        pts = points.copy() # make sure not to overwrite original points
        if(self.cam_mat_optim is not None):
            undist_pts = cv2.undistortPoints(pts, self.cam_mat, self.dist, P=self.cam_mat_optim)

            # Correct for ROI after optimized camera matrix
            for i in np.arange(len(undist_pts)):
                undist_pts[i][0][0] -= self.roi[0]
                undist_pts[i][0][1] -= self.roi[1]
        else:
            undist_pts = cv2.undistortPoints(pts, self.cam_mat, self.dist, P=self.cam_mat)

        return undist_pts

    # Based on:
    # https://answers.opencv.org/question/148670/re-distorting-a-set-of-points-after-camera-calibration/
    def redistort_points(self, points):
        pts = points.copy() # make sure not to overwrite original points

        # Undo ROI correction
        for i in np.arange(len(pts)):
            pts[i][0][0] += self.roi[0]
            pts[i][0][1] += self.roi[1]

        # Re-distort points
        if(self.cam_mat_optim is not None):
            pts_normalized = cv2.undistortPoints(pts, self.cam_mat_optim, None)
        else:
            pts_normalized = cv2.undistortPoints(pts, self.cam_mat, None)
        pts_homo = cv2.convertPointsToHomogeneous(pts_normalized)
        rtemp = ttemp = np.array([0,0,0], dtype='float32')
        pts_dist, _ = cv2.projectPoints(pts_homo, rtemp, ttemp, self.cam_mat, self.dist)
        return pts_dist

    def calibrate(self, img_path, eval_error=False, max_imgs=None, optimize_cam=False):
        # Prepare object points
        obj_pts = self.setup_obj_points()

        # Process all calibration images
        images = glob.glob(img_path + "*.png")

        if(len(images) == 0):
            images = glob.glob(img_path + "*.jpg")

        if(max_imgs is not None):
            images = images[:max_imgs]

        print(images)
            
        all_obj_pts = []
        all_img_pts = []
        for p in images:
            print("Calibration - processing:")
            print(p)
            img = cv2.imread(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[::4,::4]

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.checker_size[0],
                                                            self.checker_size[1]), None)

            # If found, add object points and image points (after refining them)
            if(ret):
                all_obj_pts.append(obj_pts)
                corners = corners*4
                corners_sub = cv2.cornerSubPix(gray,corners, (5,5), (-1,-1), self.term_crit)
                all_img_pts.append(corners_sub)

                #cv2.drawChessboardCorners(img, self.checker_size, corners_sub, ret)
                #cv2.imshow('img', img[::2,::2,:])
                #cv2.waitKey(0)

        # Run the camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_pts, all_img_pts,
                                                           gray.shape[::-1], None, None,
                                                           flags=cv2.CALIB_USE_LU)


        if(ret):
            self.cam_mat = mtx
            self.dist = dist
            #self.Rs = rvecs
            #self.ts = tvecs
            self.obj_pts = all_obj_pts
            self.img_pts = all_img_pts

            # Optimize (?) camera matrix - SHOULD WE BE DOING THIS?!?!
            if(optimize_cam):
                h, w = img.shape[:2]
                self.cam_mat_optim, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_mat,
                                                                             self.dist,
                                                                             (w,h), 1, (w,h))
            # Calc re-projection error if specified
            error = self.calc_reproj_error(all_obj_pts, all_img_pts, mtx, dist, rvecs, tvecs)
            print("Re-projection avg error: {0} pixels".format(error))

            self.Rs = rvecs
            self.ts = tvecs
            return error

        return

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

        transformed = []
        for k,curr_pts in enumerate(self.obj_pts):
            curr_R = self.Rs[k]
            curr_t = self.ts[k]

            transformed_pts = self.transform_3d_points(curr_pts, curr_R, curr_t)
            transformed.append(transformed_pts)
        return transformed





if __name__ == "__main__": # Just for debugging purposes
    cal_images_path = "test-sample/jai/checkerboards/"

    # Test calibration
    cam = Camera()
    cam.calibrate(cal_images_path, eval_error=True)

    # Test undistort image
    img_path = "test-sample/jai/rgb/00017.png"
    img = cv2.imread(img_path)
    img_undist = cam.undistort_img(img)

    cv2.imwrite(img_path.replace(".png","_undist.png"), img_undist)

    # Test undistort / re-distort points
    img_path = "test-sample/jai/rgb/00017.png"
    img = cv2.imread(img_path)
    #pts = np.array([[[654.0, 1559.0]]]) # fish eye
    pts = np.array([[[1945.0, 28.0], # ruler eye
                     [654.0, 1559.0]]]) # fish eye
    pts_undist = cam.undistort_points(pts)
    print("org: ", pts)
    print("undistorted: ", pts_undist)
    print("re-distorted: ", cam.redistort_points(pts_undist))
