import os
import shutil
from config import *

import cv2 as cv
import sqlite3
import pycolmap

print(pycolmap.has_cuda)
DB_PATH = "database.db"
FRAME_OVERLAP = 10
SIFT_MAX_NUM_FEATURES = 2048
FRAME_PATH = "res/output/images"
class COLMAP_Processor():
    def __init__(self, ):
        ...

    def extract_images(self,video_path: str, frames_modulo:int = 16):
        # frame_path = "/".join(video_path.split("/")[:-1]) + "/extracted_frames/"
        assert os.path.exists(video_path), f"video path {video_path} does not exist"
        assert video_path.split(".")[-1].lower() in ["mp4", "avi", "mov"], "unsupported video format"
        frame_path = FRAME_PATH
        os.makedirs(frame_path, exist_ok=True)
        vid = cv.VideoCapture(video_path)

        if not vid.isOpened():
            raise IOError("Couldn't open video file")

        ret, frame = vid.read()

        i,c = 0,0
        while ret:
            if (i+1) % frames_modulo == 0:
                cv.imwrite(os.path.join(frame_path, f"frame_{c:05d}.png"), frame)
                c+=1
            ret, frame = vid.read()
            i += 1;


        vid.release()
        return frame_path


    def feature_extract_and_match(self, images_path:str, mode = "exhaustive"):
        extract_options = pycolmap.FeatureExtractionOptions()       # sfittFeatureExtractionOptions is inside of featureextractoptions
        extract_options.use_gpu = True
        extract_options.sift.max_num_features = SIFT_MAX_NUM_FEATURES
        # extract_options.max_num_features = SIFT_MAX_NUM_FEATURES
        pycolmap.extract_features(
            database_path=DB_PATH,
            image_path=images_path,
            camera_mode=pycolmap.CameraMode.AUTO, extraction_options=extract_options
        )
        match_options = pycolmap.FeatureMatchingOptions()
        match_options.use_gpu = True
        if mode == "exhaustive":
            # O(N^2) - full comparison with all frames.
            # takes lots of time
            pycolmap.match_exhaustive(
                database_path=DB_PATH,
                matching_options=match_options
            )
        elif mode == "sequential":
            # this is only comparing frame-ids within the overlap.
            # i.e. frame-5 only compares with frame 0-10 if overlap=5
            # O(N*overlap) \sim O(N)
            seq_options = pycolmap.SequentialPairingOptions()
            seq_options.overlap = FRAME_OVERLAP
            pycolmap.match_sequential(
                database_path=DB_PATH,
                matching_options=match_options,
                pairing_options=seq_options,
            )
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("select count(*) from two_view_geometries")
            matches = cursor.fetchone()[0]
        print(f"found {matches} matches")

    def reconstruct(self,images_path: str, output_path: str = "sparse/"):
        os.makedirs(output_path, exist_ok=True)
        # inc_pipeline_opts = pycolmap.IncrementalPipelineOptions()
        # inc_pipeline_opts.use_gpu = True
        reconstruction = pycolmap.incremental_mapping(
            database_path=DB_PATH,
            image_path=images_path,
            output_path=output_path,
            # mapping_options=inc_pipeline_opts
        )

        return reconstruction

    @staticmethod
    def load_reconstruction(path:str):
        reconst = pycolmap.Reconstruction(path)

        return reconst

    def create_colmap(self,video_path:str, frames_modulo=20, mode="exhaustive"):
        tmp_frame_path = self.extract_images(video_path, frames_modulo=frames_modulo)
        self.feature_extract_and_match(tmp_frame_path, mode=mode)
        self.reconstruct(tmp_frame_path, output_path="res/output/sparse/")

        # dont remove, we need it for the latter gsplat generation.
        # shutil.rmtree(tmp_frame_path)


    def clean_up(self):
        # should be called at the start of a new run/ end of a run, cleans up all the code
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        #TODO: make this variable
        # path1 = "res/output/images"
        # path2 = "res/output/sparse"
        # if os.path.exists(path1):
        #     shutil.rmtree(path1)
        # if os.path.exists(path2):
        #     shutil.rmtree(path2)


# cp = COLMAP_Processor()
# cp.create_colmap("res/input/test_video.MOV", frames_modulo=10, mode="sequential")
# reconst = COLMAP_Processor.load_reconstruction(path="res/output/sparse/0")


