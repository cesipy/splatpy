from splatpy import video_to_splat

def main():
    video_path: str = "res/input/bishopstone.mp4"


    output = video_to_splat(video_path, quality="medium")
    print(f"\nGaussian Splat saved to: {output}")


if __name__ == "__main__":
    main()


