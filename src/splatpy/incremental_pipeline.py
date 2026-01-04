import os
import shutil
from .config import *

import cv2 as cv
import sqlite3
import pycolmap



# DB_PATH = ".tmp/database.db"
FRAME_OVERLAP = 20
SIFT_MAX_NUM_FEATURES = 512#8192
# FRAME_PATH = "res/output/images"
# OUTPUT_PATH_SPARSE = "res/output/sparse"

class COLMAP_Processor():
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.frame_path         = os.path.join(self.save_dir, "images")
        self.output_path_sparse = os.path.join(self.save_dir, "sparse")
        self.db_path            = os.path.join(self.save_dir, "database.db")
        os.makedirs(self.frame_path, exist_ok=True)
        os.makedirs(self.output_path_sparse, exist_ok=True)

    def extract_images(self,video_path: str, frames_modulo:int = 16,):
        # frame_path = "/".join(video_path.split("/")[:-1]) + "/extracted_frames/"
        assert os.path.exists(video_path), f"video path {video_path} does not exist"
        assert video_path.split(".")[-1].lower() in ["mp4", "avi", "mov"], "unsupported video format"

        os.makedirs(self.frame_path, exist_ok=True)
        vid = cv.VideoCapture(video_path)

        if not vid.isOpened():
            raise IOError("Couldn't open video file")

        ret, frame = vid.read()

        i,c = 0,0
        while ret:
            if (i+1) % frames_modulo == 0:
                cv.imwrite(os.path.join(self.frame_path, f"frame_{c:05d}.png"), frame)
                c+=1
            ret, frame = vid.read()
            i += 1;


        vid.release()
        return self.frame_path


    def feature_extract_and_match(
        self,
        images_path:str,
        mode:str="exhaustive",
        sift_num_max_features:int=4096,
    ):
        extract_options = pycolmap.FeatureExtractionOptions()       # sfittFeatureExtractionOptions is inside of featureextractoptions
        extract_options.use_gpu = True
        extract_options.sift.max_num_features = sift_num_max_features

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        pycolmap.extract_features(
            database_path=self.db_path,
            image_path=images_path,
            camera_mode=pycolmap.CameraMode.AUTO, extraction_options=extract_options
        )
        match_options = pycolmap.FeatureMatchingOptions()
        match_options.use_gpu = True
        if mode == "exhaustive":
            # O(N^2) - full comparison with all frames.
            # takes lots of time
            pycolmap.match_exhaustive(
                database_path=self.db_path,
                matching_options=match_options
            )
        elif mode == "sequential":
            # this is only comparing frame-ids within the overlap.
            # i.e. frame-5 only compares with frame 0-10 if overlap=5
            # O(N*overlap) \sim O(N)
            seq_options = pycolmap.SequentialPairingOptions()
            seq_options.overlap = FRAME_OVERLAP
            pycolmap.match_sequential(
                database_path=self.db_path,
                matching_options=match_options,
                pairing_options=seq_options,
            )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("select count(*) from two_view_geometries")
            matches = cursor.fetchone()[0]
        print(f"found {matches} matches")

    def reconstruct(self,images_path: str,):
        os.makedirs(self.output_path_sparse, exist_ok=True)
        # inc_pipeline_opts = pycolmap.IncrementalPipelineOptions()
        # inc_pipeline_opts.use_gpu = True
        reconstruction = pycolmap.incremental_mapping(
            database_path=self.db_path,
            image_path=images_path,
            output_path=self.output_path_sparse,
            # mapping_options=inc_pipeline_opts
        )
        return reconstruction

    @staticmethod
    def load_reconstruction(path:str):
        reconst = pycolmap.Reconstruction(path)

        return reconst

    def create_colmap(
        self,
        video_path:str,
        frames_modulo=20,
        mode="exhaustive",
        sift_num_max_features:int=4096,
    ):
        tmp_frame_path = self.extract_images(video_path, frames_modulo=frames_modulo)
        self.feature_extract_and_match(tmp_frame_path, mode=mode, sift_num_max_features=sift_num_max_features)
        self.reconstruct(tmp_frame_path,)
        # dont remove, we need it for the latter gsplat generation.
        # shutil.rmtree(tmp_frame_path)


    def clean_up(self):
        # should be called at the start of a new run/ end of a run, cleans up all the code
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        #TODO: make this dynamic
        if os.path.exists(self.frame_path):
            shutil.rmtree(self.frame_path)
        if os.path.exists(self.output_path_sparse):
            shutil.rmtree(self.output_path_sparse)
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def __del__(self):
        print("in delete function")
        self.clean_up()



