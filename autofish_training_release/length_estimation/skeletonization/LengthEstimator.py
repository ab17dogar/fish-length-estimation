import numpy as np
from skimage.morphology import skeletonize, medial_axis
import copy
import cv2

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

class LengthEstimator():
    @classmethod
    def skeletonize(cls, binary_mask, method, use_bb=True):

        if(use_bb):
            bb = bbox(binary_mask)
            mask = copy.deepcopy(binary_mask[bb[0]:bb[1],bb[2]:bb[3]])
        else:
            mask = copy.deepcopy(binary_mask)


        if method == "zhang":
            skeleton = skeletonize(mask, method='zhang')
            res = skeleton.astype(np.uint8)*255
        elif method == "lee":
            skeleton = skeletonize(mask, method='lee')
            res = skeleton.astype(np.uint8)*255
        elif method == "median":
            skeleton = medial_axis(mask, return_distance=False)
            res = skeleton.astype(np.uint8)*255

        if(use_bb):
            full_res = copy.deepcopy(binary_mask)
            full_res[bb[0]:bb[1],bb[2]:bb[3]] = res
            res = full_res
        return res


    @classmethod
    def fit_polynomial(cls, independent_var, dependent_var, poly_degree):
        """
        The dependent variable is the one that depends on the value of some other number.
        For example, if y = x+3, then the value y depends on the value of x.
        Another way to put it is the dependent variable is the output value and the independent variable is the input value.
        """
        vandermonde = np.vander(independent_var, poly_degree+1, increasing=True)
        vandermonde_pinv = np.linalg.pinv(vandermonde)
        coefficients = vandermonde_pinv@dependent_var
        return np.flip(coefficients)

    @classmethod
    #TODO The slowest part of the code is skeletonization, where "median" is the slowest method. "zhang" and "lee" are comparable.
    def get_poly_fit(cls, binary_mask, skeleton_method="zhang", degree=3, subsample_skeleton=None, subsample_fit=None, clip=True):
        def compute_squared_error(gt, predicted):
            squared_diff = (gt - predicted) ** 2
            return np.sum(squared_diff)

        skeleton = cls.skeletonize(np.ascontiguousarray(binary_mask), skeleton_method, use_bb=True)
        skeleton = np.argwhere(skeleton==255)
        if subsample_skeleton != None:
            skeleton = cls.subsample_uniform(points=skeleton, to_keep=subsample_skeleton, is_data_transposed=True)

        fish = np.argwhere(binary_mask==1)
        #Compute solution for P(x) = y
        coefficients = cls.fit_polynomial(independent_var=skeleton[:, 1], dependent_var=skeleton[:, 0], poly_degree=degree)
        skeleton_fit = np.polyval(coefficients, skeleton[:, 1])
        error = compute_squared_error(skeleton[:, 0], skeleton_fit)
        x_range = np.arange(np.min(fish[:, 1]), np.max(fish[:, 1])+1) #+1 to also include the max value itself
        y_fit = np.polyval(coefficients, x_range)
        curve_points = np.asarray([x_range, y_fit]).T
        first_solution = [curve_points, error, coefficients, skeleton]

        #Compute solution for P(y) = x
        coefficients = cls.fit_polynomial(independent_var=skeleton[:, 0], dependent_var=skeleton[:, 1], poly_degree=degree)
        skeleton_fit = np.polyval(coefficients, skeleton[:, 0])
        error = compute_squared_error(skeleton[:, 1], skeleton_fit)
        y_range = np.arange(np.min(fish[:, 0]), np.max(fish[:, 0])+1)
        x_fit = np.polyval(coefficients, y_range)
        curve_points = np.asarray([x_fit, y_range]).T
        second_solution = [curve_points, error, coefficients, skeleton]

        if first_solution[1] < second_solution[1]:
            solution = first_solution
        else:
            solution = second_solution

        if subsample_fit != None:
            solution[0] = cls.subsample_uniform(points=solution[0], to_keep=subsample_fit, is_data_transposed=True)

        if clip:
            hull = cls.compute_convex_hull(binary_mask)
            solution[0] = cls.clip_to_region(points=solution[0], region=hull, is_data_transposed=True)

        return solution


    @classmethod
    def clip_to_region(cls, points, region, is_data_transposed):
        if not is_data_transposed:
            points = points.T

        mask = np.zeros(points.shape[0], dtype=bool)
        results = np.array([cv2.pointPolygonTest(region, tuple(point), measureDist=False) >= 0 for point in points])
        mask = mask | results

        clipped_points = points[mask]
        if not is_data_transposed:
            clipped_points = clipped_points.T
        return clipped_points


    @classmethod
    def compute_rotated_bounding_box(cls, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_contours = np.concatenate(contours, axis=0)
        rect = cv2.minAreaRect(merged_contours)
        # Convert the rotated rectangle to a box2D (a rect with width, height, angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box


    @classmethod
    def compute_convex_hull(cls, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_contours = np.concatenate(contours, axis=0)
        convex_hull = cv2.convexHull(merged_contours)
        return convex_hull


    @classmethod
    def subsample_random(cls, points, to_keep, is_data_transposed, keep_end_points=True):
        assert type(is_data_transposed) == bool, "Parameter 'is_data_transposed' must be a bool type."
        if not is_data_transposed:
            points = points.T
        number_of_points=points.shape[0]

        if isinstance(to_keep, float):
            assert 0 < to_keep <= 1.0, f"Parameter 'to_keep' must be in range from [0,1>"
            number_of_points_to_keep = int(number_of_points*to_keep)
        elif isinstance(to_keep, int):
            assert 0 < to_keep <= number_of_points, f"Parameter 'to_keep' must be in range from [0, number_of_points>"
            number_of_points_to_keep = to_keep
        else:
            print(f"Parameter 'to_keep' must be either a float or an int")
            exit()

        if not keep_end_points:
            all_indices = np.arange(0, number_of_points)
            rnd_indices = np.random.choice(all_indices, number_of_points_to_keep, replace=False)
            rnd_indices = np.sort(rnd_indices)
        else:
            all_indices = np.arange(1, number_of_points-2) #randomly sample indices, but preserve end points
            rnd_indices = np.random.choice(all_indices, number_of_points_to_keep-2, replace=False)
            rnd_indices = np.append(rnd_indices, [0, number_of_points-1]) #append indices of end points
            rnd_indices = np.sort(rnd_indices)

        sampled_points = points[rnd_indices, :]
        if not is_data_transposed:
            sampled_points = sampled_points.T

        return sampled_points


    @classmethod
    def subsample_uniform(cls, points, to_keep, is_data_transposed):
        assert 0 < to_keep <= 1.0, f"Parameter 'to_keep' must be in range from [0,1>"
        assert type(is_data_transposed) == bool, "Parameter 'is_data_transposed' must be a bool type."
        if not is_data_transposed:
            points = points.T

        number_of_points=points.shape[0]
        number_of_points_to_keep = int(number_of_points*to_keep)
        sampled_indices = np.linspace(0, number_of_points-1, number_of_points_to_keep, dtype=int)
        sampled_points = points[sampled_indices, :]

        if not is_data_transposed:
            sampled_points = sampled_points.T
        return sampled_points
