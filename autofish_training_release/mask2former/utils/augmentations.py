from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog
#from detectron2.data.transforms import Augmentation
#from fvcore.transforms.transform import Transform
import numpy as np
#import cv2

#TODO Logic to ensure test resize scale and train resize scale results in similar image dimensions if jai and rs datasets are used. Currently, a user is expected to compute the scale
def get_augmentations(configuration_file):
    #train augmentations
    train_dataset = DatasetCatalog.get(configuration_file['train_dataset'][0])
    train_img_height, train_img_width = train_dataset[0]['height'], train_dataset[0]['width']
    train_augmentations = _create_augmentations_list(configuration_file['train_augmentations'], train_img_height, train_img_width)
    
    #test augmentations
    test_dataset = DatasetCatalog.get(configuration_file['test_dataset'][0])
    test_img_height, test_img_width = test_dataset[0]['height'], test_dataset[0]['width']

    test_augmentations = _create_augmentations_list(configuration_file['test_augmentations'], test_img_height, test_img_width)
    return train_augmentations, test_augmentations


def _create_augmentations_list(yaml_augs, img_height, img_width):
    augs = []
    
    if yaml_augs.get('hFlip') and yaml_augs['hFlip']:
        augs.append(T.RandomFlip(prob=yaml_augs['hFlip_prob'], horizontal=True, vertical=False))
    
    if yaml_augs.get('vFlip') and yaml_augs['vFlip']: 
        augs.append(T.RandomFlip(prob=yaml_augs['vFlip_prob'], horizontal=False, vertical=True))

    if yaml_augs.get('rBrightness') and yaml_augs['rBrightness']:
        augs.append(T.RandomBrightness(intensity_min=yaml_augs['bright_min'], intensity_max=yaml_augs['bright_max']))
    
    if yaml_augs.get('rContrast') and yaml_augs['rContrast']:
        augs.append(T.RandomContrast(intensity_min=yaml_augs['cont_min'], intensity_max=yaml_augs['cont_max']))
    
    if yaml_augs.get('rSaturation') and yaml_augs['rSaturation']:
        augs.append(T.RandomSaturation(intensity_min=yaml_augs['sat_min'], intensity_max=yaml_augs['sat_max']))
    
    if yaml_augs.get('cropToJaiAspectRatio') and yaml_augs['cropToJaiAspectRatio']['crop']:
        """
        augs.append(T.CropTransform(
            x0=626, 
            w=1294, 
            y0=0, 
            h=1080, 
            orig_w=1920, 
            orig_h=1080
        ))
        """
        #augs.append(CropRSImagesAugmentation())
        #aspect_ratio = augs[-1].aspect_ratio
        #img_target_shape = (
        #    int(img_height*yaml_augs['cropToJaiAspectRatio']['scale']), 
        #    int(img_height*aspect_ratio*yaml_augs['cropToJaiAspectRatio']['scale'])
        #    ) 
        #augs.append(T.Resize(shape=img_target_shape))

    if yaml_augs.get('resize') and yaml_augs['resize']:
        img_target_shape = (int(img_height*yaml_augs['scale']), int(img_width*yaml_augs['scale'])) 
        augs.append(T.Resize(shape=img_target_shape))
    
    if yaml_augs.get('rCrop') and yaml_augs['rCrop']['crop']:
        augs.append(T.RandomCrop(yaml_augs['rCrop']['type'], crop_size=tuple(yaml_augs['rCrop']['crop_size'])))

    return augs

#crop RS images to aspect ratio of Jai images
#implementation - crop a square with the same aspect ratio as Jai, new rs_img_width calculated as jai_aspect_ration*rs_img_height -> (2464*1080)/2056
#distance_from_right -> distance of the cropped square from the right border of the image
class CropRSImagesAugmentation(T.Augmentation):
    def __init__(self, aspect_ratio=2464/2056, distance_from_right=0):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        aspect_ratio_adjusted_img_width = int(image.shape[0]*self.aspect_ratio)
        crop_start = image.shape[1] - aspect_ratio_adjusted_img_width - self.distance_from_right
        crop_end = image.shape[1]- self.distance_from_right
        print(f"x0={crop_start} w={crop_end-crop_start} orig_w={image.shape[1]} orig_h={image.shape[0]}")
        return T.CropTransform(x0=crop_start, w=crop_end-crop_start, y0=0, h=1080, orig_w=image.shape[1], orig_h=image.shape[0])

class CropRSImageTransform(T.Transform):
    def __init__(self, image_shape, aspect_ratio, distance_from_right):
        #self.aspect_ratio = aspect_ratio
        #distance = distance_from_right
        #self.aspect_ratio_adjusted_img_width = int(image_shape[0]*aspect_ratio)
        #self.crop_start = image_shape[1] - self.aspect_ratio_adjusted_img_width - distance_from_right
        #self.crop_end = image_shape[1]- distance_from_right
        #self.padding = (
        #    self.crop_start,
        #    image_shape[1]-self.crop_end
        #)
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img) -> np.ndarray:
        #img_width = int(img.shape[0]*self.aspect_ratio)
        #crop_start_width = img.shape[1] - img_width - self.distance
        #crop_end_width = img.shape[1]- self.distance
        #cropped_image = img[0:img.shape[0], crop_start_width:crop_end_width]
        #  
        aspect_ratio_adjusted_img_width = int(self.image_shape[0]*self.aspect_ratio)
        crop_start = self.image_shape[1] - aspect_ratio_adjusted_img_width - self.distance_from_right
        crop_end = self.image_shape[1]- self.distance_from_right
        cropped_image = img[0:img.shape[0], crop_start:crop_end] 
        return cropped_image

    def apply_coords(self, coords) -> np.ndarray:
        aspect_ratio_adjusted_img_width = int(self.image_shape[0]*self.aspect_ratio)
        crop_start = self.image_shape[1] - aspect_ratio_adjusted_img_width - self.distance_from_right
        print(f"apply_coords {coords}")
        coords[:, 1] = coords[:, 1]
        coords[:, 0] = coords[:, 0] - crop_start 
        return coords
    
    def apply_segmentation(self, segmentation) -> np.ndarray:
        segmentation = self.apply_image(segmentation)
        return segmentation
    
    def inverse(self) -> T.Transform:
        aspect_ratio_adjusted_img_width = int(self.image_shape[0]*self.aspect_ratio)
        crop_start = self.image_shape[1] - aspect_ratio_adjusted_img_width - self.distance_from_right
        crop_end = self.image_shape[1]- self.distance_from_right
        padding = (
            crop_start,
            self.image_shape[1]-crop_end
        )
        return InverseCropRSImageTransform(padding, crop_start)
    

class InverseCropRSImageTransform(T.Transform):
    def __init__(self, padding, crop_start):
        #self.padding = padding
        #self.crop_start = crop_start
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img) -> np.ndarray:
        # Pad the image to the left to undo the cropping
        padded_image = np.pad(img, ((0, 0), self.padding, (0, 0)), mode='constant', constant_values=(122, 122))
        return padded_image

    def apply_coords(self, coords) -> np.ndarray:
        coords[:, 1] = coords[:, 1]
        coords[:, 0] = coords[:, 0] + self.crop_start
        return coords

    def apply_segmentation(self, segmentation) -> np.ndarray:
        return self.apply_image(segmentation)

    def inverse(self) -> T.Transform:
        # The inverse of the InverseCropRSImageTransform is the original CropRSImageTransform
        return T.NoOpTransform()



"""
import cv2
import numpy as np
class WatermarkAugmentation(Augmentation):
    #Add a watermark to the bottom-right corner of the image.
    #Args:
    #    watermark_text (str): The text to be used as the watermark.
    def __init__(self, watermark_text="detectron2"):
        super().__init__()
        self.watermark_text = watermark_text

    def get_transform(self, image):
        return WatermarkTransform(image.shape[:2], self.watermark_text)
    

class WatermarkTransform(T.Transform):
    def __init__(self, image_shape, watermark_text):
        self.image_shape = image_shape
        self.watermark_text = watermark_text

    def apply_image(self, img):
        watermark_img = self.generate_watermark(img)
        return cv2.addWeighted(img, 1, watermark_img, 0.5, 0)

    def apply_coords(self, coords):
        return coords

    def generate_watermark(self, img):
        watermark_img = np.copy(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_right_corner = (self.image_shape[1] - 150, self.image_shape[0] - 20)
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)  # White color

        cv2.putText(
            watermark_img, self.watermark_text, bottom_right_corner, font,
            font_scale, font_color, font_thickness, cv2.LINE_AA
        )
        return watermark_img
""" 
