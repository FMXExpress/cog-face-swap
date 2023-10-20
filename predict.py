# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
from typing import Optional, Union
import time

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
        self.face_enhancer = gfpgan.GFPGANer(model_path='./checkpoints/GFPGANv1.4.pth', upscale=2)

    def get_single_face(self, img_data):
        '''get and return the largest single face in a image'''
        analysed = self.face_analyzer.get(img_data)
        if not analysed:
            print("Error: No face found")
            return None
        largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return largest

    def detect_faces(self, target_image):
        '''detect faces in a image and return the bounding boxes in original sorted order'''
        # img_data = ins_get_image(target_image)
        faces = self.face_analyzer.get(target_image)
        print(f"face num:{len(faces)}")
        # faces = sorted(faces, key = lambda x : x.bbox[0])

        if faces:
            return [face.bbox for face in faces]
        else:
            return None
    
    def swap_faces(self, target_image: np.ndarray, model_faces: list):
        '''swap faces in a image and return the swapped image'''
        # Get all faces in target image
        target_faces = self.face_analyzer.get(target_image)
        if not target_faces:
            raise ValueError(f"Error: No face found in target image")

        # append 'null' value to model_faces to make sure the number of faces in target image and model images are the same        
        if len(model_faces) < len(target_faces):
            model_faces.extend([None] * (len(target_faces) - len(model_faces)))
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

    def predict(
        self,
        target_image: Path = Input(
            description="Faces in target image would be changed.",
        ),
        model_faces: list[Path] = Input(
            description="Model Faces would be swapped into target image.", 
            default=None,
        ),
        inference_mode: str = Input(
            default="swap",
            choices=["swap", "detect"],
            description="Face swap mode or detection mode. Default is swap.",
        ),
    ) -> Union(list[tuple], list[Path]):
        """Run a single prediction on the model"""
        # Check if inference_mode is detect or swap
        if inference_mode == "detect":
            detect_mode = True
        else: 
            detect_mode = False
            if model_faces is None:
                raise ValueError(f"model_faces is required when inference_mode is swap")

        try:
            # 1.preprocess by opencv, convert Path images to np.ndarray
            target = cv2.imread(str(target_image))
            src_faces = []
            if not detect_mode:
                for face in model_faces:
                    if not face:
                        src_faces.append(None)
                        continue
                    src_faces.append(cv2.imread(str(face)))

            # Check if image was successfully loaded
            if target is None:
                raise ValueError(f"Could not open or read the target image")

            # 2.inference
            # detect faces
            if detect_mode:
                face_bboxes = self.detect_faces(target_image)
                if not face_bboxes:
                    raise ValueError(f"Error: No face found in target image")
                return face_bboxes

            # swap multiple faces in target image
            else:
                swapped_image = self.swap_faces(target_image, src_faces)

                # 3.postprocess
                # Save the image to a temporary file
                # This file will automatically be deleted by Cog after it has been returned.
                '''out_path = Path(tempfile.mkdtemp()) / f"{str(int(time.time()))}.png"
                cv2.imwrite(str(out_path), swapped_image)
                return out_path'''
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, swapped_image)
                    temp_path = temp_file.name
                    print(f" swapped image file is saved to temp_path: {temp_path}\n")

                return Path(temp_path)
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")


