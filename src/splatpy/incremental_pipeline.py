import os
import shutil
from .config import *

import torch
import cv2 as cv
import sqlite3
import pycolmap
from pytorch_msssim import ms_ssim


FRAME_OVERLAP = 20
DEBUGGING   = False
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


    def _get_frames(self, video_path:str, frames_ids: list[int]) -> list[cv.typing.MatLike]:
        """
        Iterates over the video and extracts the specified frame ids.
        """
        vid = cv.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open video file")
        ret, frame = vid.read()
        i = 0

        selected_frames = []

        while ret:
            if i in frames_ids:
                selected_frames.append(frame)
            ret, frame = vid.read()
            i += 1
        return selected_frames


    def _save_frames(self, frames: list[cv.typing.MatLike]):
        """
        Writes frames to the frame path defined in 'self.frame_path'
        """
        for c, frame in enumerate(frames):
            cv.imwrite(os.path.join(self.frame_path, f"frame_{c:05d}.png"), frame)

    def _remove_similar_frames(self, frames:list[cv.typing.MatLike],  threshold:float):
        assert 0.0 <= threshold <= 1.0, "threshold must be between 0 and 1"

        different_frames = []
        buffer_tensors   = []

        for current_frame in frames:
            current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2RGB)
            # frame is type numpy.ndarray with shape (h, w, 3), torch needs
            curren_frame_tensor = torch.tensor(current_frame).permute(2, 0, 1).unsqueeze(0).float()  # shape (1, 3, h,w)
            if len(different_frames) == 0:
                different_frames.append(current_frame)
                buffer_tensors.append(curren_frame_tensor)
            else:
                last_frame = different_frames[-1]
                last_frame_tensor = buffer_tensors[-1]

                similarity = ms_ssim(curren_frame_tensor, last_frame_tensor, data_range=255, size_average=True)


                if similarity <= threshold:
                    different_frames.append(current_frame)
                    buffer_tensors.append(curren_frame_tensor)


        #final comparison: first vs middle
        # sim = ms_ssim(
        #     buffer_tensors[0], buffer_tensors[len(buffer_tensors)//2], data_range=255, size_average=True
        # )
        # print(f"similarity first vs middle: {sim}")

        print(f"removed {len(frames) - len(different_frames)} similar frames, {len(different_frames)} frames remain")
        return different_frames





    def extract_images(
        self,
        video_path: str,
        extraction_rate: float = 0.10,
    ):
        # frame_path = "/".join(video_path.split("/")[:-1]) + "/extracted_frames/"
        assert os.path.exists(video_path), f"video path {video_path} does not exist"
        assert video_path.split(".")[-1].lower() in ["mp4", "avi", "mov"], "unsupported video format"

        os.makedirs(self.frame_path, exist_ok=True)
        vid = cv.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open video file")
        length = vid.get(cv.CAP_PROP_FRAME_COUNT)


        if length == 0:
            if DEBUGGING:
                print(f"Video has 0 frames, skipping extraction")
            vid.release()
            return self.frame_path

        # calculate based on extraction rate.
        target_num_frames = max(1, int(length * extraction_rate))
        if DEBUGGING:
            print(f"Video has {int(length)} frames, extracting {target_num_frames} ({extraction_rate*100:.1f}%)")

        #add warning if too few frames
        if target_num_frames < 10:
            #proper parning
            print(f"Warning: Extracting only {target_num_frames} frames from video with {int(length)} frames. Consider increasing extraction rate.")

        # store laplacian variances for all frames
        vars = []
        ret,frame = vid.read()

        i, c = 0,0
        while ret:
            #https://stackoverflow.com/a/48321095
            # laplacian => edge detector; var(laplacian) => quantification of edges.
            vars.append(cv.Laplacian(frame, cv.CV_64F).var())
            i += 1
            ret, frame = vid.read()
        vid.release()

        assert len(vars) == length, "length mismatch in variance computation"
        bucket_size = length // target_num_frames
        bucket_higests = []
        current_bucket = 0
        current_highest_in_bucket = 0
        for i in range(1, int(length)):
            if vars[i] > vars[current_highest_in_bucket]:
                current_highest_in_bucket = i

            if i % bucket_size == 0:
                bucket_higests.append(current_highest_in_bucket)
                current_bucket +=1
                current_highest_in_bucket = i

        # last bucket
        if current_highest_in_bucket not in bucket_higests:
            bucket_higests.append(current_highest_in_bucket)


        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        frames = self._get_frames(video_path, bucket_higests)

        # Handle case where we couldn't extract all requested frames
        # (can happen with real videos due to frame access issues)
        if len(frames) < len(bucket_higests):
            if DEBUGGING:
                print(f"Warning: Could only extract {len(frames)} frames out of {len(bucket_higests)} requested")
        #remove frames that are too similar, based on ms_ssim
        different_frames = self._remove_similar_frames(frames, threshold=0.7)


        self._save_frames(different_frames)

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
        video_path: str,
        extraction_rate: float = 0.10,
        mode: str = "exhaustive",
        sift_num_max_features: int = 4096,
    ):
        tmp_frame_path = self.extract_images(video_path, extraction_rate=extraction_rate)
        self.feature_extract_and_match(tmp_frame_path, mode=mode, sift_num_max_features=sift_num_max_features)
        self.reconstruct(tmp_frame_path,)
        # dont remove, we need it for the latter gsplat generation.
        # shutil.rmtree(tmp_frame_path)


    def clean_up(self):
        if DEBUGGING:
            pass
        else:
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



