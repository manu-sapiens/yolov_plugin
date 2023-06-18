print("---------- Yolov Integration ----------------")
# ---------------------------------------------------
# --------------- Yolov -----------------------------
# ---------------------------------------------------
from utils.omni_utils_misc import omni_get_env, log_decorator
from utils.omni_utils_gpu import GetTorchDevice
from utils.omni_utils_masks import process_mask, save_image_or_mask

OMNI_TEMP_FOLDER = omni_get_env("OMNI_TEMP_FOLDER")
OMNI_CHECKPOINT_FOLDER = omni_get_env("OMNI_CHECKPOINT_FOLDER")

import os
from utils.omni_utils_http import CdnHandler, CdnResponse
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Any, List
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLO

from Plugins.yolov_plugin.yolov_definitions import YolovSegment_Input, YolovSegment_Response

@log_decorator
async def integration_YolovSegment_Post(input: YolovSegment_Input):
    cdn = CdnHandler()
    if True: #try:
        cdn.announcement()
        print("------------- post_yolov_segment ------------------")
        print(f"input = {input}")

        categories_csv = input.categories_csv
        minimum_confidence = input.minimum_confidence

        model = input.model

        output_merged = input.output_merged 
        invert_mask = input.invert_mask
        output_alpha = input.output_alpha
        output_mask = input.output_mask


        categories = categories_csv.split(", ")

        print(f"categories = {categories}")
        print(f"minimum_confidence = {minimum_confidence}")
        print(f"output_merged = {output_merged}")
        print(f"output_alpha = {output_alpha}")   
        print(f"invert_mask = {invert_mask}")
        print(f"output_mask = {output_mask}")

        # todo: convert categories into list of number (again :( )) to be passed directly into model.predict

        input_cdns = [input.images[0]] # !!!!!!  
        input_filenames = await cdn.download_files_from_cdn(input_cdns)
        input_filename = input_filenames[0]
        print(f"input_filename = {input_filename}")
        # todo: we only support one image at a time for now as we already return an array of images and json for each processed image.
        # and I don't want to deal with arrays of array in the Designer (for now)


        retina_masks = True
        yolov_model_path = os.path.join(OMNI_CHECKPOINT_FOLDER, model)  #'yolov8s.pt'
        yolov_model = YOLO(yolov_model_path)
        # todo: for optimization, we could use the base model (e.g. yolov8s.pt) instead of the segmentation model (e.g. yolov8s-segment.pt) when doing json-only output


        device = GetTorchDevice() # support CUDA if present


        # read the categories
        yolov_category_names = yolov_model.names
        #print(f"yolov_category_names = \n{yolov_category_names}")

        # convert categories into class_ids
        use_all_categories = False
        if len(categories) == 0 or "all" in categories: use_all_categories = True

        categories_dict = {}
        class_ids = None
        if not use_all_categories:

            for key, value in yolov_category_names.items():
                categories_dict[value] = key
            #
            class_ids= []
            for category in categories:
                if category in categories_dict:
                    class_ids.append(categories_dict[category])
                #
            #
        #
            
        print(f"class_ids = {class_ids}")

        yolov_results = None
        result_filenames = []  # To hold the filenames of the saved masks

        image_bgr = cv2.imread(input_filename)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        yolov_results = yolov_model.predict(
            image_rgb, 
            device=device, 
            retina_masks=retina_masks, 
            conf=minimum_confidence, 
            classes=class_ids)        
        
        max_x = image_bgr.shape[0]
        max_y = image_bgr.shape[1]  

        if yolov_results != None and len(yolov_results) > 0:
    
            yolov_result= yolov_results[0]
            all_masks = yolov_result.masks

            raw_yolov_detections = sv.Detections.from_yolov8(yolov_results[0])
            print(f"raw_yolov_detections = \n{raw_yolov_detections}")
            
            use_all_categories = False
            if len(categories) == 0 or "all" in categories: use_all_categories = True

            image_bgr = cv2.imread(input_filename)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            base_name ="yolov_"+cdn.generate_random_name()

            merged_mask = None
            result_filenames = []
            category_count = {}
            result_detections = []
            for xyxy, confidence, class_id, mask in zip(raw_yolov_detections.xyxy, raw_yolov_detections.confidence, raw_yolov_detections.class_id, all_masks):

                category = yolov_category_names[class_id]
                print(f"\n----------\nCategory = {category}, class_id = {class_id}, confidence = {confidence}, xyxy = {xyxy}")

                # Extract bounding box coordinates
                x1, y1, x2, y2 = xyxy.tolist()
                category = yolov_model.names[class_id]


                # Build dictionary for the detection
                detection_dict = {
                    'xyxy': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'category': str(category)
                }

                # Append the detection dictionary to the json_array
                result_detections.append({"detection":detection_dict})

                mask_data = mask.data  # raw masks tensor, (N, H, W) or masks.masks 
                # Convert the tensor to a numpy array and rescale to 0-255
                mask_data_numpy = (mask_data.numpy() * 255).astype(np.uint8)[0] # --------- added [0]
                mask_data_numpy_bool = mask_data_numpy > 0 # Set all non-zero values to True

                filename, merged_mask = process_mask(mask_data_numpy_bool, merged_mask, category, output_mask, output_merged, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name)
                print(f"identified: {filename}")
                result_filenames.append(filename)
            
                if output_merged:
                    print("Processing everthing mask")
                    print(f"merged_mask.shape = {merged_mask.shape}")
                    everything_filename = save_image_or_mask(merged_mask, "all", output_mask, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name+"_everything")
                    result_filenames = [everything_filename]
                #
            #   

        results_cdns = []
        if len(result_filenames) > 0:
            results_cdns = await cdn.upload_files_to_cdn(result_filenames)
            # delete the results files from the local storage
            cdn.delete_temp_files(result_filenames)
        #

        response = YolovSegment_Response(passthrough_array=input.images, media_array=results_cdns, json_array=result_detections)

        #print(f"response = {response}")
        print("\n-----------------------\n")
        return response
    else: #except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
