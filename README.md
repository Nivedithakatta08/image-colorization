# Image Colorization

Converts grayscale images into colorized versions using a deep learning model built with PyTorch.

## What it does

Takes a black-and-white image as input and predicts realistic colors for it, outputting a colorized version. Built as a way to explore image-to-image translation and computer vision fundamentals.

## Project structure

```
image-colorization/
├── colorize.py         # Core grayscale-to-color conversion script
├── requirements.txt    # Dependencies
└── README.md
```

## Tech stack

- PyTorch — model training and inference
- OpenCV, NumPy — image preprocessing
- Streamlit — demo interface

## Getting started

Clone the repo:

```bash
git clone https://github.com/Nivedithakatta08/image-colorization.git
cd image-colorization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run it:

```bash
python colorize.py --input path/to/image.jpg
```

## Status

Work in progress — currently a starter script. Next steps: train on a proper dataset, evaluate output quality, and wrap it in a usable demo.

## License

MIT
