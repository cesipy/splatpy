from splatpy import video_to_splat, video_to_splat_advanced

def main():
    video_path: str = "res/input/house.mp4"

    output = video_to_splat(video_path, quality="low")



if __name__ == "__main__":
    main()


