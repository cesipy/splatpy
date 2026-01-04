from splatpy import video_to_splat, video_to_splat_advanced

def main():
    video_path: str = "res/input/house.mp4"
    output = video_to_splat_advanced(video_path, training_steps=1000  )

    output = video_to_splat(video_path, quality="low")
    print(f"\nGaussian Splat saved to: {output}")


if __name__ == "__main__":
    main()


