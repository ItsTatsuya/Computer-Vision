# SIFT Image Matching

This project implements SIFT (Scale-Invariant Feature Transform) based image matching to find and visualize corresponding keypoints between two images.

## Features

- **SIFT Feature Detection**: Detects keypoints in both images
- **Descriptor Matching**: Uses FLANN-based matcher with Lowe's ratio test
- **Homography Estimation**: Finds geometric transformation using RANSAC
- **Visualization**: Creates a side-by-side comparison with matched keypoints

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your two images in the same directory as the script (or provide full paths)
2. Update the image paths in `sift_matching.py`:
   ```python
   image1_path = 'your_image1.jpg'
   image2_path = 'your_image2.jpg'
   ```
3. Run the script:
   ```bash
   python sift_matching.py
   ```

## Output

- Console output with keypoint counts and match statistics
- Visualization displayed in a matplotlib window
- Saved image file: `sift_matches.jpg`

## How It Works

1. **Load Images**: Reads two images and converts to grayscale
2. **SIFT Detection**: Detects keypoints and computes descriptors for both images
3. **Matching**: Uses FLANN matcher to find the 2 nearest neighbors for each descriptor
4. **Ratio Test**: Filters matches using Lowe's ratio test (default: 0.75)
5. **Homography**: Estimates geometric transformation using RANSAC
6. **Visualization**: Draws lines connecting matched keypoints

## Parameters

You can adjust these parameters in the code:

- `ratio_threshold`: Lowe's ratio test threshold (default: 0.75, lower = stricter)
- `min_match_count`: Minimum matches required for homography (default: 10)
- `RANSAC threshold`: Maximum reprojection error in pixels (default: 5.0)
