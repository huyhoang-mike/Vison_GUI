import albumentations as A
import cv2
import numpy as np

class Algorithm:
    def rotate(self, limit, border):
        if border == "Constant":
            border_mode = 1
        elif border == "Replicate":
            border_mode = 2
        elif border == "Reflect":
            border_mode = 3
        transforms = A.Compose([
                A.augmentations.geometric.rotate.Rotate(limit=limit, border_mode=border_mode, p=1),
            ])
        return transforms

    def rotateOD(self, limit, border):
        if border == "Constant":
            border_mode = 1
        elif border == "Replicate":
            border_mode = 2
        elif border == "Reflect":
            border_mode = 3
        transforms = A.Compose([
                A.augmentations.geometric.rotate.Rotate(limit=limit, border_mode=border_mode, p=1),
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms

    def Rcrop(self, width, height):
        transforms = A.Compose([
                A.RandomCrop(width=width, height=height)
            ])
        return transforms
    
    def RcropOD(self, width, height):
        transforms = A.Compose([
                A.RandomCrop(width=width, height=height)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def Mcrop(self, xmin, ymin, xmax, ymax):
        transforms = A.Compose([
                A.Crop(x_min=int(xmin), y_min=int(ymin), x_max=int(xmax), y_max=int(ymax))
            ])
        return transforms
    
    def McropOD(self, xmin, ymin, xmax, ymax):
        transforms = A.Compose([
                A.Crop(x_min=int(xmin), y_min=int(ymin), x_max=int(xmax), y_max=int(ymax))
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms

    def resize(self, width, height):
        transforms = A.Compose([
                A.Resize(p=1, height=height, width=width)
            ])
        return transforms
    
    def resizeOD(self, width, height):
        transforms = A.Compose([
                A.Resize(p=1, height=height, width=width)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms

    def hflip(self):
        transforms = A.Compose([
                A.HorizontalFlip(p=1)
            ])
        return transforms
    
    def hflipOD(self):
        transforms = A.Compose([
                A.HorizontalFlip(p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms

    def vflip(self):
        transforms = A.Compose([
                A.VerticalFlip(p=1)
            ])
        return transforms
    
    def vflipOD(self):
        transforms = A.Compose([
                A.VerticalFlip(p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def randomScale(self, scale_limit, interpolation):
        transforms = A.Compose([
                A.RandomScale(p=1, scale_limit=scale_limit, interpolation=interpolation)
            ])
        return transforms
    
    def randomScaleOD(self, scale_limit, interpolation):
        transforms = A.Compose([
                A.RandomScale(p=1, scale_limit=scale_limit, interpolation=interpolation)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def Blur(self, blur_limit):
        transforms = A.Compose([
                A.Blur(always_apply=False, p=1)
            ])
        return transforms
    
    def BlurOD(self, blur_limit):
        transforms = A.Compose([
                A.Blur(p=1, blur_limit=blur_limit)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def HueSaturationValue(self,hue_shift_limit,sat_shift_limit,val_shift_limit):
        transforms = A.Compose([
                A.HueSaturationValue(p=1, hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit)
            ])
        return transforms
    
    def HueSaturationValueOD(self,hue_shift_limit,sat_shift_limit,val_shift_limit):
        transforms = A.Compose([
                A.HueSaturationValue(p=1, hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms

    def RandomBrightnessContrast(self,brightness_limit,contrast_limit ):
        transforms = A.Compose([
                A.RandomBrightnessContrast(p=1, brightness_limit=brightness_limit, contrast_limit=contrast_limit)
            ])
        return transforms
    
    def RandomBrightnessContrastOD(self,brightness_limit,contrast_limit ):
        transforms = A.Compose([
                A.RandomBrightnessContrast(p=1, brightness_limit=brightness_limit, contrast_limit=contrast_limit)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def Solarize(self, threshold):
        transforms = A.Compose([
                A.Solarize(p=1, threshold=threshold)
            ])
        return transforms
    
    def SolarizeOD(self, threshold):
        transforms = A.Compose([
                A.Solarize(p=1, threshold=threshold)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def RandomGamma(self, gamma_limit):
        transforms = A.Compose([
                A.RandomGamma(p=1, gamma_limit=gamma_limit)
            ])
        return transforms
    
    def RandomGammaOD(self, gamma_limit):
        transforms = A.Compose([
                A.RandomGamma(p=1, gamma_limit=gamma_limit)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def GaussNoise(self, var_limit, mean):
        transforms = A.Compose([
                A.GaussNoise(p=1, var_limit=var_limit, mean=mean)
            ])
        return transforms
    
    def GaussNoiseOD(self, var_limit, mean):
        transforms = A.Compose([
                A.GaussNoise(p=1, var_limit=var_limit, mean=mean)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
  
    def ChannelShuffle(self):
        transforms = A.Compose([
                A.ChannelShuffle(p=1)
            ])
        return transforms
    
    def ChannelShuffleOD(self):
        transforms = A.Compose([
                A.ChannelShuffle(p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def CoarseDropout(self, min_holes, max_holes, min_height, max_height, min_width, max_width, fill_value):
        transforms = A.Compose([
                A.CoarseDropout(p=1, min_holes=int(min_holes), max_holes=int(max_holes), min_height=int(min_height), 
                                max_height=int(max_height), min_width=int(min_width), max_width=int(max_width), fill_value=fill_value)
            ])
        return transforms
    
    def CoarseDropoutOD(self, min_holes, max_holes, min_height, max_height, min_width, max_width, fill_value):
        transforms = A.Compose([
                A.CoarseDropout(p=1, min_holes=int(min_holes), max_holes=int(max_holes), min_height=int(min_height), 
                                max_height=int(max_height), min_width=int(min_width), max_width=int(max_width), fill_value=fill_value)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        return transforms
    
    def multiTransform(self, param_array):
        transforms = A.Compose([
                A.augmentations.geometric.rotate.Rotate(limit=param_array["rotate_limit"], border_mode=param_array["border_mode"], p=1), 
                A.RandomCrop(width=param_array["width_crop"], height=param_array["height_crop"]),
                A.Resize(p=1, height=param_array["height_resize"], width=param_array["width_resize"]),
                A.Solarize(p=1, threshold=param_array["threshold"])
            ])
        return transforms
    
    def multiTransformF(self, param_array):
        transforms = A.Compose([
                A.augmentations.geometric.rotate.Rotate(limit=param_array["rotate_limit"], 
                                                        border_mode=param_array["border_mode"], p=1),
                A.RandomCrop(width=param_array["width_crop"], height=param_array["height_crop"]),
                A.Crop(x_min=int(param_array["xmin"]), y_min=int(param_array["ymin"]), 
                                        x_max=int(param_array["xmax"]), y_max=int(param_array["ymax"])),
                A.Resize(p=1, height=param_array["height_resize"], width=param_array["width_resize"]),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomScale(p=1, scale_limit=param_array["scale_limit"], interpolation=param_array["interpolation"]),
                A.ZoomBlur(p=1, max_factor=param_array["max_factor"], step_factor=param_array["step_factor"]),
                A.HueSaturationValue(p=1, hue_shift_limit=param_array["hue_shift_limit"], 
                                     sat_shift_limit=param_array["sat_shift_limit"], val_shift_limit=param_array["val_shift_limit"]),
                A.RandomBrightnessContrast(p=1, brightness_limit=param_array["brightness_limit"], contrast_limit=param_array["contrast_limit"]),
                A.Solarize(p=1, threshold=param_array["threshold"])
            ])
        return transforms
    
    def scale_contrast(self, mean_shift, contrast_scaling, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray_img)
        std_dev = np.std(gray_img)
        normalized_img = (gray_img - mean_val) / std_dev
        # Modify the constants for contrast scaling and mean shift
        scaled_contrast_img = mean_shift + contrast_scaling * normalized_img
        scaled_contrast_img = np.clip(scaled_contrast_img, 0, 255).astype(np.uint8)
        scaled_contrast_img = cv2.cvtColor(scaled_contrast_img, cv2.COLOR_GRAY2RGB)
        return scaled_contrast_img