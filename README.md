# Video Processing and Paragraph Extraction

A Python and OpenCV project that combines two digital image processing tasks: editing video with face blurring and visual effects, and extracting paragraphs from scanned document images using histogram projection.

## Features

### Video Processing
- **Face blurring**: detects faces in each frame and blurs them to protect identity
- **Brightness detection**: measures the brightness of frames in the video
- **Watermarking**: adds a watermark to the video
- **Overlay**: places an image or graphic on top of the video frames
- **End-screen**: appends a closing screen to the finished video

### Paragraph Extraction
- Takes a scanned image of a document as input
- Uses **histogram projection** to find the gaps between lines and paragraphs
- Splits the page into separate paragraph images

## How Histogram Projection Works

The scanned page is converted to a binary image. Pixel counts are then summed along each row (horizontal projection) to create a histogram. Rows with text produce high values, while the blank spaces between paragraphs produce values close to zero. Wide runs of empty rows mark the boundaries between paragraphs, and the page is cropped at those boundaries.

## Tech Stack

- Python 3
- OpenCV
- NumPy

## Getting Started

### Prerequisites

Make sure Python 3 is installed, then install the dependencies:

```bash
pip install opencv-python numpy
```

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Usage

Run the video processing script:

```bash
python video_processing.py
```

Run the paragraph extraction script:

```bash
python paragraph_extraction.py
```

Update the input file paths at the top of each script to point to your own video and scanned image.

## Project Structure

```
.
├── video_processing.py        # Face blur, brightness, watermark, overlay, end-screen
├── paragraph_extraction.py    # Histogram projection paragraph splitting
├── samples/                   # Example input and output files
└── README.md
```

## What I Learned

- Applying face detection and spatial filtering to video frames with OpenCV
- Working with frames, codecs, and video writers in Python
- Using projection profiles to segment document layouts
- Collaborating on a shared codebase while following consistent coding conventions

## Authors

Group project for CSC2014, Sunway University.

- Decent ([GitHub profile link])

## License

This project was created for educational purposes.
