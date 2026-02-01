import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(image1_path, image2_path):
    """Load two images from file paths."""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None:
        raise ValueError(f"Could not load image from {image1_path}")
    if img2 is None:
        raise ValueError(f"Could not load image from {image2_path}")

    # Convert to grayscale for SIFT detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1, img2, gray1, gray2


def detect_and_compute_sift(gray1, gray2):
    """Detect keypoints and compute SIFT descriptors."""
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    print(f"Image 1: Found {len(keypoints1)} keypoints")
    print(f"Image 2: Found {len(keypoints2)} keypoints")

    return keypoints1, descriptors1, keypoints2, descriptors2


def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.75):
    """Match descriptors using FLANN-based matcher with ratio test."""
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find k=2 best matches for each descriptor
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    print(f"Found {len(good_matches)} good matches after ratio test")

    return good_matches


def visualize_matches(img1, keypoints1, img2, keypoints2, matches, output_path='matches.jpg'):
    """Visualize matched keypoints between two images."""
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Display using matplotlib
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Feature Matching - {len(matches)} matches')
    plt.axis('off')
    plt.tight_layout()

    # Save the result
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.show()


def find_homography(keypoints1, keypoints2, good_matches, min_match_count=10):
    """Find homography matrix and draw inlier matches."""
    if len(good_matches) < min_match_count:
        print(f"Not enough matches found - {len(good_matches)}/{min_match_count}")
        return None, None

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Count inliers
    inliers = mask.ravel().tolist()
    inlier_count = sum(inliers)
    print(f"Found {inlier_count} inliers out of {len(good_matches)} matches")

    return M, mask


def main():
    """Main function to run SIFT image matching."""
    # Specify your image paths here
    image1_path = 'data/sift-image-1.jpeg'
    image2_path = 'data/sift-image-2.jpeg'

    print("=" * 60)
    print("SIFT Image Matching")
    print("=" * 60)

    # Load images
    print("\n1. Loading images...")
    img1, img2, gray1, gray2 = load_images(image1_path, image2_path)

    # Detect and compute SIFT features
    print("\n2. Detecting SIFT features...")
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_and_compute_sift(gray1, gray2)

    # Match descriptors
    print("\n3. Matching descriptors...")
    good_matches = match_descriptors(descriptors1, descriptors2, ratio_threshold=0.75)

    # Find homography (optional - for robustness)
    print("\n4. Finding homography...")
    M, mask = find_homography(keypoints1, keypoints2, good_matches)

    # Visualize matches
    print("\n5. Visualizing matches...")
    visualize_matches(img1, keypoints1, img2, keypoints2, good_matches, 'sift_matches.jpg')

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
