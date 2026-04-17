\# Vision Dataset Specification



\## Classes

1\. \*\*ambulance\*\* (priority: detect vehicle body)

2\. \*\*lightbar\*\* (optional: detect emergency lights on top)



\## Image Requirements

\- \*\*Resolution\*\*: Minimum 640x640, recommended 1280x720+

\- \*\*Format\*\*: JPG/PNG

\- \*\*Quality\*\*: No heavy compression

\- \*\*Diversity\*\*: 

&nbsp; - Multiple ambulance types (box, van, SUV)

&nbsp; - Various angles (front, side, rear, 3/4 view)

&nbsp; - Different distances (10m - 100m)

&nbsp; - Day and night conditions

&nbsp; - Clear and adverse weather



\## Annotation Format

YOLO format: `<class\_id> <x\_center> <y\_center> <width> <height>`

\- Normalized coordinates \[0, 1]

\- One annotation per line in `.txt` file

\- Class IDs: 0=ambulance, 1=lightbar



\## Data Sources

1\. \*\*Google Open Images\*\*: Search "ambulance"

2\. \*\*Roboflow Universe\*\*: Pre-labeled ambulance datasets

3\. \*\*YouTube Videos\*\*: Extract frames from dash-cam footage

4\. \*\*Custom Recording\*\*: Smartphone/dash-cam (with permission)

5\. \*\*COCO Dataset\*\*: Some emergency vehicle images

6\. \*\*Kaggle Datasets\*\*: "Emergency Vehicles", "Traffic CCTV"



\## Minimum Dataset Size

\- \*\*Quick test\*\*: 100 images (50 train, 25 val, 25 test)

\- \*\*Production\*\*: 1000+ images (700 train, 150 val, 150 test)

\- \*\*Optimal\*\*: 5000+ images with diverse conditions



\## Augmentation Strategy

Applied during training:

\- \*\*Motion blur\*\*: Simulate moving vehicles (kernel 15-25)

\- \*\*Fog/Haze\*\*: Reduce visibility (alpha 0.3-0.7)

\- \*\*Rain\*\*: Add rain streaks

\- \*\*Sun glare\*\*: Bright spots and lens flare

\- \*\*Night\*\*: Reduce brightness, add noise

\- \*\*Standard\*\*: Flip, rotate, scale, crop, HSV jitter



\## Quality Checks

\- All bounding boxes visible and tight

\- No overlapping duplicates

\- Correct class labels

\- Images readable and clear enough for detection



