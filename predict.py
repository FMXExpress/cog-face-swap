# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
from typing import Union, List
import zipfile
from pathlib import Path as PathObj
import re

from cog import BasePredictor, Path, Input
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
import gfpgan


# Check the version of insightface
assert insightface.__version__>='0.7'

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Prepare the Face Analysis app
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 means GPU/CUDA support
        # Load the face swapper model
        self.swapper = insightface.model_zoo.get_model('./checkpoints/inswapper_128.onnx', download=True, download_zip=True)
        # Load GFPGAN upscaler
        self.face_enhancer = gfpgan.GFPGANer(model_path='./checkpoints/GFPGANv1.4.pth', upscale=1)

    def get_single_face(self, img_data: np.ndarray):
        '''get and return the largest single face in a image'''
        analysed = self.face_analyzer.get(img_data)
        if not analysed:
            print("Error: No face found")
            return None
        largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return largest

    def detect_faces(self, target_image: np.ndarray):
        '''detect faces in a image and return the bounding boxes in original sorted order'''
        # img_data = ins_get_image(target_image)
        faces = self.face_analyzer.get(target_image)
        print(f"face num:{len(faces)}")
        # faces = sorted(faces, key = lambda x : x.bbox[0])

        if faces:
            return [face.bbox for face in faces]
        else:
            return None
    
    def swap_faces(self, target_image: np.ndarray, model_faces: list[np.ndarray]):
        '''swap faces in a image and return the swapped image'''
        # Get all faces in target image
        target_faces = self.face_analyzer.get(target_image)
        if not target_faces:
            raise ValueError(f"Error: No face found in target image")

        # append 'null' value to model_faces to make sure the number of faces in target image and model images are the same        
        if len(model_faces) < len(target_faces):
            model_faces.extend([None] * (len(target_faces) - len(model_faces)))
        # drop the extra faces in model_faces if len(model_faces) > len(target_faces)
        elif len(model_faces) > len(target_faces):
            model_faces = model_faces[:len(target_faces)]
        assert len(model_faces) == len(target_faces)

        # swap face one by one, skip if model_faces[i] is None
        result = target_image.copy()
        for idx, face in enumerate(model_faces):
            if face is None:
                continue
            src_face = self.get_single_face(face)
            if not src_face:
                print(f"Error: No face found in model image {idx}")
                continue
            result = self.swapper.get(result, target_faces[idx], src_face, paste_back=True)

        # Enhance the swapped image by GFPGAN
        _, _, enhanced_image = self.face_enhancer.enhance(result, paste_back=True)
        return enhanced_image
    
    def concat_swap_faces(self, target_image: np.ndarray, model_faces: list[np.ndarray]):
        '''swap faces in a image and return the horizentally concatenated swapped faces'''
        # Get all faces in target image
        target_faces = self.face_analyzer.get(target_image)
        if not target_faces:
            raise ValueError(f"Error: No face found in target image")

        # append 'null' value to model_faces to make sure the number of faces in target image and model images are the same
        if len(model_faces) < len(target_faces):
            model_faces.extend([None] * (len(target_faces) - len(model_faces)))
        # drop the extra faces in model_faces if len(model_faces) > len(target_faces)
        elif len(model_faces) > len(target_faces):
            model_faces = model_faces[:len(target_faces)]
        assert len(model_faces) == len(target_faces)

        # swap each target face areas with corresponding source face and append new faces as a list
        results = []
        for idx, face in enumerate(target_faces):
            if model_faces[idx] is None:
                continue
            src_face = self.get_single_face(model_faces[idx])
            if not src_face:
                print(f"Error: No face found in model image {idx}")
                continue
            _img, _ = self.swapper.get(target_image, face, src_face, paste_back=False)
            results.append(_img)
        results = np.concatenate(results, axis=1)
        return results
    
    def preprocess_image(self, model_faces_list: list[Path], target_image: Path, detect_mode: bool) -> Union[np.ndarray, List[np.ndarray]]:
        '''preprocess images and run inference'''
        # 1.preprocess by opencv, convert Path images to np.ndarray
        target = cv2.imread(str(target_image))
         # Check if image was successfully loaded
        if target is None:
            raise ValueError(f"Could not open or read the target image")
        src_faces = []
        if not detect_mode:
            for face_path in model_faces_list:
                # Check if path is valid and file exists
                if not face_path or not face_path.is_file():
                    src_faces.append(None)
                    continue
                src_faces.append(cv2.imread(str(face_path)))
        return target, src_faces

    def inference(self, target: np.ndarray, src_faces: list[np.ndarray], detect_mode: bool):
        try:
            # 2.inference
            if detect_mode:
                # detect faces
                face_bboxes = self.detect_faces(target)
                if not face_bboxes:
                    raise ValueError(f"Error: No face found in target image")
                return face_bboxes

            else:
                # swap multiple faces in target image
                swapped_image = self.swap_faces(target, src_faces)
                return swapped_image
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def predict(
        self,
        target_image: Path = Input(
            description="Faces in target image would be changed.",
        ),
        model_faces: Path = Input(
            description="Model Faces would be swapped into target image.", 
            default=None,
        ),
        inference_mode: str = Input(
            default="swap",
            choices=["swap", "detect"],
            description="Face swap mode or detection mode. Default is swap.",
        ),
    ) -> Union[List[tuple], Path]:
        """Run a single prediction on the model"""
        # Check if inference_mode is detect or swap
        detect_mode = inference_mode == "detect"
        if not detect_mode and model_faces is None:
            raise ValueError(f"model_faces is required when inference_mode is swap")
        
        if detect_mode:
            print(f"Running face detection on {target_image.name}")
            target, _ = self.preprocess_image([], target_image, detect_mode)
            face_bboxes = self.inference(target, [], detect_mode)
            return face_bboxes
        
        # Create a temporary directory that will last throughout the function
        temp_dir = tempfile.TemporaryDirectory()

        # check file type of model_faces. uncompress it if it is zip file, or keep it as a single image file
        model_faces_list = []
        print(f"model_faces: {model_faces.name}")
        if model_faces.suffix == ".zip":
            with zipfile.ZipFile(model_faces, 'r') as zip_ref:
                zip_ref.extractall(temp_dir.name)
                # the mapping of file names to a list with None for missing indices based on the numeric sequence in the filenames
                # Extracted files
                extracted_files = {int(re.search(r'\d+', file).group()): file for file in zip_ref.namelist() if re.search(r'\d+', file)}
                # Highest index based on the filenames
                max_index = max(extracted_files.keys(), default=0)
                # Map to file path list with None for missing indices
                model_faces_list = [PathObj(temp_dir.name, extracted_files[i]) if i in extracted_files else None for i in range(1, max_index + 1)]

        elif model_faces.suffix in [".png", ".jpg", "jpeg"]:
            model_faces_list = [model_faces]
        else:
            raise ValueError(f"Error: model_faces should be a zip file or a single image file")
        
        # 2.preprocess and inference
        target, src_faces = self.preprocess_image(model_faces_list, target_image, detect_mode)
        swapped_image = self.inference(target, src_faces, detect_mode)

        # At the end of the function, cleanup the temporary directory and uncompressed file
        temp_dir.cleanup()

        # 3.postprocess
        # Save the image to a temporary file
        # This file will automatically be deleted by Cog after it has been returned.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            cv2.imwrite(temp_file.name, swapped_image)
            temp_path = temp_file.name
            print(f" swapped image file is saved to temp_path: {temp_path}\n")

        return Path(temp_path)
        

