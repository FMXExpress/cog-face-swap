# utf-8
import tempfile
import dlib
from cog import BasePredictor, Path, Input
import os

from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


LANDMARKS_DETECTOR = LandmarksDetector("./checkpoints/shape_predictor_68_face_landmarks.dat")

def align_image(raw_img_path, aligned_face_dir):
    output_list = []
    if not os.path.exists(aligned_face_dir):
        os.makedirs(aligned_face_dir)
    for i, face_landmarks in enumerate(LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1):
        face_img_name = '%s_%02d.png' % (os.path.splitext(raw_img_path)[0], i)
        print(f"face name{i}: {face_img_name}")
        aligned_face_path = os.path.join(aligned_face_dir, face_img_name)
        image_align(raw_img_path, aligned_face_path, face_landmarks)
        output_list.append(aligned_face_path)
    return output_list


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input source image.",
        ),
    ) -> list[Path]:
        out_path_dir = "aligned_image"
        output = align_image(str(image), str(out_path_dir))

        return output

