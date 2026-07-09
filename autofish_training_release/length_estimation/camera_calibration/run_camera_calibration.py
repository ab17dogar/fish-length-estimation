import cv2
import argparse
import numpy as np
import os
import glob
import json
import random
import pyransac3d as pyrsc
from Camera import Camera
from Checkerboard import Checkerboard
import matplotlib.pyplot as plt

max_cal_images = 20

def save2png(input_path, output_path):
    img = cv2.imread(input_path, -1)

    clahe = cv2.createCLAHE()
    img2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(output_path, img2)


def process_group(autofish_dir, group_no):
    group_path = os.path.join(autofish_dir, f"group_{group_no}")
    print(f"Processing: {group_path}")

    cam_imgs = glob.glob(os.path.join(group_path,"calibration/jai/*.png"))
    #cam_imgs.sort()
    random.shuffle(cam_imgs)

    # prepare output_dir
    png_dir = os.path.join(autofish_dir, "camera_calibration", f"group_{group_no}")
    if not os.path.exists(png_dir): # create if it does not exists
        os.makedirs(png_dir)
    else: # remove old files if it exists
        for f in glob.glob(os.path.join(png_dir, "*")):
            os.remove(f)

    # process camera images
    for i,cam_img_path in enumerate(cam_imgs):
        print(f" img: {i+1}/{len(cam_imgs)}")
        # convert img
        png_path = os.path.join(png_dir, "{:05d}.png".format(i+1))
        save2png(cam_img_path, png_path)
        if(i+1 == max_cal_images):
            break

def calculate_intrinsic(autofish_dir, cal_output_path):
    cam = Camera()
    cal_path = os.path.join(autofish_dir, "camera_calibration/group_*/")
    error = cam.calibrate(cal_path, eval_error=True)

    print(cam.cam_mat)
    print(cam.dist)

    cam_cal = {}
    cam_cal["square_size_mm"] = cam.square_size_mm
    cam_cal["reproj_error"] = error

    cam_cal["fx"] = cam.cam_mat[0][0]
    cam_cal["fy"] = cam.cam_mat[1][1]
    cam_cal["cx"] = cam.cam_mat[0][2]
    cam_cal["cy"] = cam.cam_mat[1][2]

    cam_cal["dist_coeff"] = list(cam.dist[0])

    out_path = os.path.join(cal_output_path, "intrinsic_params.json")
    with open(out_path, "w") as f:
        json.dump(cam_cal, f, indent=4)
    return cam



def get_group_from_path(path):
    elements = path.split("/")
    for e in elements:
        if("group_" in e):
            group_no = int(e.replace("group_",""))
            return group_no
    return None

def calculate_plane_eq(autofish_dir, cal_output_path, undistort, cb_thickness=0.0, visualize=False):    
    # Load calibrated camera
    cam = Camera()
    cam.load_from_json(os.path.join(cal_output_path, "intrinsic_params.json"))

    # Find all group folders
    group_paths = glob.glob(os.path.join(autofish_dir, "camera_calibration/group_*/"))

    plane_equations = {}
    homographies = {}
    for p in group_paths:
        print("Finding plane for ", p)
        group_no = get_group_from_path(p)

        # Init checkerboards
        cb_imgs = glob.glob(p + "*.png")

        # Find points
        img_pts = [] # 2D pixel in pixels
        obj_pts = [] # 3D points in relation to camera
        for img_path in cb_imgs:
            print("Processing CB: ", img_path)
            curr_cb = Checkerboard(img_path, calibrated_cam=cam, undistort=undistort)

            curr_obj_pts = curr_cb.get_obj_pts_camera_coords()
            curr_img_pts = curr_cb.get_shifted_img_points(cb_thickness)

            if(visualize):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.figure()
                plt.imshow(img_rgb)
                plt.scatter(curr_img_pts.reshape(-1,2)[:,0], curr_img_pts.reshape(-1,2)[:,1])
                plt.savefig(f"detected_image_points_{len(img_pts)}.png", dpi=600)
            
            if(curr_obj_pts is not None and curr_img_pts is not None):
                obj_pts.append(curr_obj_pts)
                img_pts.append(curr_img_pts)

        # Find plane to the points
        plane = pyrsc.Plane()
        obj_points = np.concatenate(obj_pts, axis=0 )
        best_eq, best_inliers = plane.fit(obj_points, 1.0)


        # Visualize plane
        # source: https://stackoverflow.com/questions/36060933/plot-a-plane-and-points-in-3d-simultaneously
        if(visualize):
            normal = np.array(best_eq[:3])
            d = best_eq[-1]

            # create x,y
            xx, yy = np.meshgrid(range(-500,500), range(-500,500))

            # calculate corresponding z
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
            
            # Create the figure
            fig = plt.figure()

            # Add an axes
            ax = fig.add_subplot(111,projection='3d')
            #plt3d = plt.figure().gca(projection='3d')

            #ax.hold(True)

            #and i would like to plot this point :
            for pts in obj_pts: #[10:]:
                ax.scatter(pts[:,0] , pts[:,1] , pts[:,2]) #, s=2) #,  color='green')
            #plt.show()
            ax.scatter([0], [0], [0], marker="x", color="black")
            ax.plot_surface(xx, yy, z, alpha=0.2)

            ax.invert_zaxis()
            plt.savefig("fitted-plane.png", dpi=600)            

        # Find homography       
        img_points = np.concatenate(img_pts, axis=0)
        src_pts = img_points[best_inliers].reshape(-1,1,2)
        dst_pts = obj_points[best_inliers][:,:2].reshape(-1,1,2)
        homo, inliers = cv2.findHomography(src_pts, dst_pts)
        

        if(best_eq[-1] < 0):
            best_eq = [b*-1.0 for b in best_eq]

        print(" plane eq: ", best_eq)
        plane_equations[f"group_{group_no}"] = best_eq

        print(" homography: ", homo)
        homographies[f"group_{group_no}"] = homo.tolist()        

    # Save output
    os.makedirs(cal_output_path, exist_ok=True)
        
    out_path = os.path.join(cal_output_path, "plane_params.json")
    with open(out_path, "w") as f:
      json.dump(plane_equations, f, indent=4)

    out_path = os.path.join(cal_output_path, "homographies.json")
    with open(out_path, "w") as f:
      json.dump(homographies, f, indent=4)


if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--autofish_dir", help="Path to the directory with all the AutoFish group dirs.")

    parser.add_argument('--undistort', dest='undistort', action='store_true')
    parser.add_argument('--no-undistort', dest='undistort', action='store_false')
    parser.set_defaults(undistort=True)

    parser.add_argument("--cb_thickness", type=float, default=0.0,
                        help="Thickness (mm) of the checkerboard") # not used
    
    
    arguments = parser.parse_args()
    p = parser.parse_args()

    cal_output_path = os.path.join("camera_calibration", "output",
                                   f"undistort-{p.undistort}")
    os.makedirs(cal_output_path, exist_ok=True)

    # Calculate and save the camera intrinsics
    if(not os.path.isfile(os.path.join(cal_output_path, "intrinsic_params.json"))):
        calculate_intrinsic(p.autofish_dir, cal_output_path)
    else:
        print("Intrinsic camera params already exists - skipping...")

    # # Calculate the plane of the checkerboards for each group
    # calculate_plane_eq(p.autofish_dir, cal_output_path,
    #                    undistort=p.undistort,
    #                    cb_thickness=p.cb_thickness)
