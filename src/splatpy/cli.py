import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        prog="splatpy",
        description="Convert a video to a 3D Gaussian Splat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
quality presets:
  test    quick pipeline check     (1k steps,  5%% frames)
  low     fast preview             (7k steps,  5%% frames)
  medium  balanced quality         (15k steps, 10%% frames)  [default]
  high    high quality             (28k steps, 15%% frames)
  ultra   maximum quality          (35k steps, 30%% frames)

examples:
  splatpy video.mp4
  splatpy video.mp4 --quality high --output results/
  splatpy video.mp4 --no-orbit
        """,
    )

    parser.add_argument(
        "video",
        help="path to input video file (.mp4, .avi, or .mov)",
    )
    parser.add_argument(
        "--quality", "-q",
        default="medium",
        choices=["test", "low", "medium", "high", "ultra"],
        help="quality preset (default: medium)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        metavar="DIR",
        help="output directory (default: results/)",
    )
    parser.add_argument(
        "--no-orbit",
        action="store_true",
        help="skip rendering the 360° orbit video",
    )
    parser.add_argument(
        "--orbit-frames",
        type=int,
        default=240,
        metavar="N",
        help="number of frames in orbit video (default: 240)",
    )
    parser.add_argument(
        "--depth-prior",
        action="store_true",
        help="use Depth Anything V2 to densify the point cloud (requires: pip install transformers accelerate)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    from .api import video_to_splat

    try:
        output_path = video_to_splat(
            video_path=args.video,
            quality=args.quality,
            output_dir=args.output,
            render_orbit=not args.no_orbit,
            orbit_frames=args.orbit_frames,
            use_depth_prior=args.depth_prior,
        )

        print(f"\nDone! Splat saved to: {output_path}")
        print("\nTo view your splat, drag and drop the .ply file into:")
        print("  https://supersplat.playcanvas.com")

    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nerror: {e}", file=sys.stderr)
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()
