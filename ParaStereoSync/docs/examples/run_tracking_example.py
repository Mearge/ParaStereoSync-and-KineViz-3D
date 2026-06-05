"""
Run a simple tracking example using `eccTrackVideo`.

Edit the paths below to point to your video and template CSV files.
This script is a lightweight template and will not run if required dependencies or files are missing.
"""
import os
import sys

VIDEO_PATH = r"/path/to/your/video_cam1.mp4"
TEMPLATE_CSV = r"/path/to/your/templates_cam1.csv"
OUTPUT_CSV = r"/path/to/output/cam1_tracks.csv"

def main():
    try:
        from eccTrackVideo_Multistep_warp_guess import eccTrackVideo
    except Exception as e:
        print("Could not import eccTrackVideo. Ensure you run this from repository root and have installed dependencies.")
        print(e)
        sys.exit(1)

    params = dict(
        videoFilepath=VIDEO_PATH,
        tmpltFilepath=TEMPLATE_CSV,
        tmpltFrameId=0,
        frameRange=None,
        tmpltRange=None,
        mTable=None,
        saveFilepath=OUTPUT_CSV,
    )

    print("Starting tracking with params:\n", params)
    eccTrackVideo(**params)
    print("Tracking finished. Output saved to:", OUTPUT_CSV)

if __name__ == '__main__':
    main()
