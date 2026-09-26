# 🏛️ Lighthouse 3D Reconstruction

A Python-based computer vision project that reconstructs a **3D representation of a real-world scene from multiple 2D photographs**.

The project uses feature detection, feature matching, camera pose estimation, triangulation, Structure-from-Motion (SfM), point-cloud processing, and surface reconstruction to generate a 3D representation of the captured space.

---

## 📌 Project Overview

The goal of this project is to take multiple photographs of the same physical space from different viewpoints and reconstruct the scene in 3D.

The pipeline processes the photographs and performs:

1. Image loading and preprocessing
2. SIFT feature detection
3. Feature matching
4. Camera pose estimation
5. Essential matrix estimation
6. 3D point triangulation
7. Incremental Structure-from-Motion
8. Point-cloud generation
9. Point-cloud filtering
10. Surface/mesh reconstruction
11. 3D visualization

The final result is a reconstructed **3D point cloud and mesh** of the captured scene.

---

## ✨ Features

- 📷 Multi-image 3D reconstruction
- 🔍 SIFT feature detection
- 🔗 Feature matching using OpenCV
- 📐 Essential matrix estimation
- 📍 Camera pose recovery
- 🧭 Incremental Structure-from-Motion (SfM)
- 📊 PnP-based camera pose estimation
- 🔺 3D point triangulation
- ☁️ Point-cloud generation
- 🧹 Point-cloud outlier filtering
- 📉 Voxel downsampling
- 🏗️ Poisson surface reconstruction
- 💾 PLY point-cloud export
- 🖥️ 3D reconstruction visualization

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Computer vision and image processing |
| OpenCV-Contrib | SIFT feature detection |
| Open3D | Point-cloud and 3D processing |
| NumPy | Numerical computation |
| SciPy | Scientific computing |
| Matplotlib | Visualization |
| tqdm | Progress tracking |

---

# 📂 Project Structure

```text
lighthouse-3d-reconstruction/
│
├── pipeline.py
├── getfocal.py
├── README.md
├── .gitignore
│
└── images/
    ├── image001.jpg
    ├── image002.jpg
    ├── image003.jpg
    └── ...
```

> **Important:** The `images` folder and original image dataset are not included in this GitHub repository.

---

# 📸 Dataset / Input Images

## ⚠️ The dataset is NOT included

The original image dataset used during development is **not uploaded to this repository**.

Anyone who wants to run this project must provide their **own image dataset**.

Create an `images` folder inside the project directory:

```text
lighthouse-3d-reconstruction/
│
├── pipeline.py
├── getfocal.py
├── README.md
├── .gitignore
│
└── images/
    ├── image001.jpg
    ├── image002.jpg
    ├── image003.jpg
    └── ...
```

Place your photographs inside the `images` folder.

### Recommended image characteristics

For better reconstruction results:

- Use multiple photographs of the same physical space.
- Capture the scene from different viewpoints.
- Maintain sufficient overlap between photographs.
- Avoid excessive motion blur.
- Use reasonably clear images.
- Make sure important parts of the scene appear in multiple images.
- Avoid very large changes in viewpoint between consecutive images.
- Capture enough images to provide good coverage of the scene.

### Important

The reconstruction quality depends heavily on the input photographs.

**Users are responsible for providing their own suitable image dataset.**

---

# ⚙️ Installation

## 1. Install Python

Install Python on your computer if it is not already installed.

Verify the installation:

```bash
python --version
```

---

## 2. Install Required Libraries

Open a terminal inside the project folder and run:

```bash
python -m pip install opencv-contrib-python open3d numpy matplotlib tqdm scipy
```

---

# 🚀 How to Run the Project

## Step 1 — Clone the Repository

Clone the GitHub repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Then enter the project folder:

```bash
cd lighthouse-3d-reconstruction
```

---

## Step 2 — Add Your Own Images

Create an `images` folder:

```text
lighthouse-3d-reconstruction/
└── images/
```

Put your own photographs inside it.

Example:

```text
images/
├── image001.jpg
├── image002.jpg
├── image003.jpg
├── image004.jpg
├── image005.jpg
└── ...
```

---

## Step 3 — Run the Pipeline

Run:

```bash
python pipeline.py
```

The program will process the images and attempt to reconstruct the scene in 3D.

---

# 🔄 Reconstruction Pipeline

The overall workflow is:

```text
                 INPUT IMAGES
                      │
                      ▼
             Image Loading
                      │
                      ▼
          Image Preprocessing
                      │
                      ▼
          SIFT Feature Detection
                      │
                      ▼
           Feature Matching
                      │
                      ▼
        Essential Matrix Estimation
                      │
                      ▼
          Camera Pose Recovery
                      │
                      ▼
          3D Point Triangulation
                      │
                      ▼
       Incremental Structure-from-Motion
                      │
                      ▼
            Global Point Cloud
                      │
                      ▼
          Outlier Point Filtering
                      │
                      ▼
            Voxel Downsampling
                      │
                      ▼
       Poisson Surface Reconstruction
                      │
                      ▼
              3D MODEL
```

---

# 🧠 How It Works

## 1. Image Loading

The pipeline loads photographs from the `images` directory.

The photographs should represent the same physical scene from different viewpoints.

---

## 2. SIFT Feature Detection

SIFT features are detected in the images.

These features provide distinctive points that can be identified across multiple photographs.

---

## 3. Feature Matching

Features from different images are matched to determine which points correspond to the same physical locations in the scene.

---

## 4. Camera Pose Estimation

The system estimates the relative position and orientation of the cameras using matched feature points.

---

## 5. 3D Triangulation

Corresponding points observed from multiple camera positions are triangulated to estimate their 3D coordinates.

---

## 6. Incremental Structure-from-Motion

Additional images are progressively added to the reconstruction.

Camera poses and 3D points are estimated as more images are incorporated.

---

## 7. Point Cloud Processing

The reconstructed 3D points are processed to remove noisy or isolated points.

The point cloud can then be downsampled for easier visualization and processing.

---

## 8. Surface Reconstruction

The processed point cloud can be used to generate a surface mesh using Poisson reconstruction.

---

# 📤 Output

The pipeline can generate reconstruction results in the `outputs` directory.

Typical output structure:

```text
outputs/
│
├── depth_maps/
│
├── point_cloud/
│   ├── lighthouse_clean.ply
│   └── camera_poses.txt
│
├── mesh/
│   └── lighthouse_mesh.ply
│
└── screenshots/
    └── complete_3d_reconstruction.png
```

---

# ☁️ Point Cloud

The `.ply` point-cloud file contains reconstructed 3D points representing the scene.

Example:

```text
lighthouse_clean.ply
```

The point cloud can be opened using software that supports the PLY format.

---

# 🏗️ 3D Mesh

The reconstructed point cloud can be converted into a surface mesh.

Example:

```text
lighthouse_mesh.ply
```

The mesh provides a surface representation of the reconstructed scene.

---

# 📍 Camera Poses

The pipeline can also save estimated camera poses:

```text
camera_poses.txt
```

These represent the estimated camera positions and orientations used during reconstruction.

---

# 👁️ Visualization

The generated `.ply` files can be viewed using 3D visualization software that supports the PLY format.

Open3D can also be used for point-cloud and mesh visualization.

---

# 🔧 Configuration

Important parameters can be adjusted inside:

```text
pipeline.py
```

Examples include:

```python
MAX_IMAGES
RESIZE_WIDTH
SIFT_FEATURES
RATIO_TEST
```

These parameters control aspects such as:

- Number of images processed
- Image resolution
- Number of detected features
- Feature matching threshold

Parameter values may need to be adjusted depending on the input dataset and computer hardware.

---

# ⚠️ Limitations

3D reconstruction depends strongly on the quality and coverage of the input photographs.

The following conditions can reduce reconstruction quality:

- Insufficient image overlap
- Motion blur
- Poor lighting
- Reflective surfaces
- Transparent surfaces
- Repetitive textures
- Very large camera movements
- Too few photographs
- Areas that are not visible in any photograph

### Important

A surface that is never visible in any input photograph cannot be reliably reconstructed from those photographs alone.

Therefore, capturing the scene from multiple viewpoints is important for obtaining better 3D coverage.

---

# 💻 Hardware Considerations

3D reconstruction can require significant computational resources.

Processing time and memory usage depend on:

- Number of images
- Image resolution
- Number of detected features
- Number of reconstructed points
- Computer RAM
- CPU performance

For larger datasets, reducing image resolution or processing fewer images may help reduce memory usage.

---

# 📚 Applications

The techniques used in this project can be applied to:

- Computer Vision
- Photogrammetry
- 3D Mapping
- Architecture
- Digital Heritage Documentation
- Robotics
- Virtual Reality
- Augmented Reality
- Digital Twins
- 3D Scene Reconstruction
- Research and Education

---

# 🎯 Project Objective

The primary objective of this project is to demonstrate how computer vision and multi-view geometry can be used to reconstruct a 3D representation of a real-world environment from ordinary photographs.

The project combines multiple computer vision techniques into a single reconstruction pipeline.

---

# 🔮 Future Improvements

Possible future improvements include:

- Bundle adjustment
- Improved camera calibration
- Dense multi-view stereo
- Better feature tracking
- Improved camera pose optimization
- Advanced point-cloud filtering
- Higher-quality mesh generation
- Texture mapping
- GPU acceleration
- Automatic image quality assessment
- Better handling of large image datasets

---

# 📋 Requirements

```text
Python 3.x
OpenCV
OpenCV-Contrib
Open3D
NumPy
SciPy
Matplotlib
tqdm
```

Install them with:

```bash
python -m pip install opencv-contrib-python open3d numpy matplotlib tqdm scipy
```

---

# 📁 Dataset Notice

**The image dataset is intentionally not included in this repository.**

To use this project, users must provide their own photographs of a scene.

The input photographs should be placed inside:

```text
images/
```

The repository contains the reconstruction code and supporting files, while the input dataset must be supplied separately by the user.

---

# 👨‍💻 Author

This project was developed as a computer vision and 3D reconstruction project using Python, OpenCV, and Open3D.

---

# 📄 License

This project is intended for educational and research purposes.
