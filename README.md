# Photogrammetry
<img src="Photogrammetry/Resources/Icons/Photogrammetry.png">

An extension to preprocess (mask) large collections of photographs which then can be processed to construct 3D models with texture. 
See [Zhang and Maga (2023) An Open-Source Photogrammetry Workflow for Reconstructing 3D Models](https://academic.oup.com/iob/article/5/1/obad024/7221338) on how to take pictures of specimens using a low-cost step-up and optionally use Aruco markers to obtain the physical scale of the object. 

## Prerequisites
### Running on MorphoCloud
There are no prerequisites if [you are using MorphoCloud On Demand](https://instances.morpho.cloud). All necessary libraries are preloaded. 

### Running Locally on your computer
We suggest using the MorphoCloud On Demand service to run the Photogrammetry extension. There are few reasons for that. First and foremost, GPUs provided (Nvidia A100) on MorphoCloud do really accelerate the workflow both for masking and 3D model reconstruction. The typical workflow from start to end using the provided sample data below would take about 60-70 minutes on the MorphoCloud. It will be significantly longer on your own computer, unless you have a very high end CPU with dozens of cores, and a powerful GPU.  Additionally, it will require you [to install the Docker container](https://docs.docker.com/engine/install/), and [the Nvidia container toolkit if you are using Nvidia GPUs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your computer, both of which requires having admin access to your computer. 

Due to these complexities of installing Docker, Photogrammetry extension currently not provided as part of the Slicer Extension Catalogue for MacOS and Windows, only Linux is supported. But if you are an advanced user and successfully installed docker, you can manually install the extension by:

1. Downloading this repository (use the green code button and choose Download ZIP). Alternatively, you can clone the repository via git.
2. Uncompress the zip file on your desktop
3. Drag and drop the SlicerPhotogrammetry (might also be called SlicerPhotogrammetry-main) folder to the Slicer application window, and choose "Add Python Scripted Modules to the Application" option from the dropdown menu of the popup message, and enable both Photogrammetry and ClusterPhotos.



## Sample Data
Unprocessed photographs from [15 mountain beavers used in the Zhang and Maga, 2022 paper can be downloaded from here:](https://seattlechildrens1.box.com/v/PhotogrammetrySampleData)

Use this link to download one specimen (UWBM 82409) used in the tutorials and other documentation: 
https://app.box.com/shared/static/z8pypqqmel8pv4mp5k01philfrqep8xm.zip

# User Guide

This document will guide you through the **Photogrammetry** module's features, from loading models and preparing masks to running reconstructions through ODM. 

## Table of Contents
1. [Overview](#1-overview)
2. [What is SAM?](#2-what-is-sam)
3. [Choosing the SAM Model Variant](#3-choosing-the-sam-model-variant)
4. [Setting Up Input/Output Folders](#4-setting-up-inputoutput-folders)
5. [Processing Folders and Browsing Image Sets](#5-processing-folders-and-browsing-image-sets)
6. [Masking Workflows](#6-masking-workflows)
   - [Batch Masking (All Images)](#a-batch-masking-all-images)
   - [Single Image Masking](#b-single-image-masking)
7. [Mask Resolution and Performance](#7-mask-resolution-and-performance)
8. [Monitoring Masking Progress](#8-monitoring-masking-progress)
9. [Managing WebODM](#9-managing-webodm)
   - [Installing/Pulling and Launching NodeODM](#91-installingpulling-and-launching-nodeodm)
   - [Stopping the NodeODM Container](#91-installingpulling-and-launching-nodeodm)
10. [Find-GCP and Marker Detection](#10-find-gcp-and-marker-detection)
11. [Configuring and Running WebODM Tasks](#11-configuring-and-running-webodm-tasks)
12. [Saving and Restoring Tasks](#12-saving-and-restoring-tasks)
13. [Importing the Final 3D Model into Slicer](#13-importing-the-final-3d-model-into-slicer)

---

## 1. Overview

**Photogrammetry** is a 3D Slicer module designed to help users transform large sets of photographs into a single 3D model using **photogrammetry**. The module integrates:
- **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** for efficient masking of each image (removing background, highlighting your object).
- **[WebODM / NodeODM](https://www.opendronemap.org/webodm/)** for aerial or close-range photogrammetry reconstruction.
- **Additional** tools for generating Ground Control Point (GCP) data (optional, but necessary if you want your models to have accurate physical dimensions).

### What the Module Allows You To Do
1. **Run Segment Anything Model (SAM)** conveniently in 3D Slicer.  
2. **Batch-mask or individually mask** sets of images using bounding boxes and optional inclusion/exclusion points to guide segmentation.  
3. **Generate GCP data** (if you have marker coordinates) to obtain accurate physical dimensions of the specimens.  
4. **Interface with WebODM** to launch reconstructions locally or on a remote node.  
5. **Download the final 3D model** back into Slicer for further inspection and visualization.

---

## 2. What is SAM?

**SAM (Segment Anything Model)** is a state-of-the-art segmentation model by Meta. It can segment objects in images with minimal user input:
- **Bounding boxes** around the object you wish to mask, and
- **Inclusion (green) or Exclusion (red) points** indicating details SAM might miss or include erroneously.

SAM supports several variants (ViT-base, ViT-large, ViT-huge) differing in file size, GPU memory demands, and inference speed. 

---

## 3. Choosing the SAM Model Variant

Upon first opening the **Photogrammetry** module, you will see a dropdown labeled **SAM Variant**. The three typical variants are:

1. **ViT-base (~376 MB):**  
   - Fastest inference, least GPU memory usage. Good if you have limited GPU or use CPU.
2. **ViT-large (~1.03 GB):**  
   - Balanced in speed and memory usage.  
3. **ViT-huge (~2.55 GB):**  
   - Highest accuracy potential, but requires significant GPU RAM and more time.

### Loading the Chosen Model
- Select the variant that fits your resources (e.g., **ViT-base** if you're unsure).
- Click **Load Model**.  
- Wait for the download if you haven't already downloaded the weights.  
- Once loaded, you can proceed with folder processing.

> **Tip:** If you have a strong GPU (e.g., 8GB VRAM or more), you can try **ViT-large** or **ViT-huge**. If you are CPU-only or have ~4GB VRAM or less, **ViT-base** is safer.

---

## 4. Setting Up Input/Output Folders

1. **Input Folder:** Should contain **multiple subfolders** (each subfolder is one "image set"). For example:
   - `Beaver_Skull_3_Images`
     - `Set1` (all photographs of the object taken in a first orientation -e.g., top view)
     - `Set2` (all photographs of the object taken in a second orientation, -e.g., bottom view)
     - etc.
Keeping similar orientations in sets help with the workflow associated with masking the background (see below).

2. **Output Folder:** A separate folder you create for masked images and where final results will be placed.

Use the **Directory** pickers in the module UI to select these paths. SlicerPhotogrammetry stores your selection so you don't have to re-pick them every time. Just remember to change them for each new reconstruction project.

---

## 5. Processing Folders and Browsing Image Sets

Once you've loaded a SAM model and chosen valid **Input** and **Output** directories:

- Click **Process Folders**.  
  - The module will scan each subfolder in the Input folder.  
  - Any recognized image (`*.jpg`, `*.png`, etc.) will be listed and ready for masking.  
- A **progress bar** indicates the scanning progress.  
- After processing, you can pick any subfolder (image set) in the **Image Set** dropdown to inspect or mask.

### Navigating Sets
- The **Image Set** combobox shows each subfolder name.
- Select a set to see its images.
- You can switch sets at any time each set's state (masked or not) is preserved.

### Image List
- The **Image List** table shows the filename and image number for each image in the selected image set.
- You can click on each image in this to load and view the image, and it's related mask, if present.
- Unmasked images appear red on this list but will change to green once masked.

---

## 6. Masking Workflows

Masking removes background and keeps only the foreground object. SlicerPhotogrammetry provides **two** main masking workflows:

1. **Batch Masking:** For when you want to define a single bounding box (ROI) that applies across **all** images in a set.  
2. **Single Image Masking:** For fine-tuning individual images or if batch mode didn't capture the object well enough.

### (A) Batch Masking (All Images)

1. **Select a Set** from the dropdown.
2. Click **Place/Adjust ROI for All Images**.  
   - This removes any previously created masks for that set.
   - Puts you in a special ****global bounding box**** mode.
3. Slicer will prompt you to **place a bounding box** that should encompass the object in every image.  
   - Use the 2D slice viewer (**Red**) to **drag and resize** the ROI, ensuring the bounding region covers the entire object.  
   - Switch between images with the **\<** and **\>** buttons to verify the bounding box is suitable for all angles.
4. When satisfied, click **Finalize ROI and Mask All Images**.  
   - SAM processes each image. If you checked **quarter** or **half** resolution (see [Mask Resolution](#7-mask-resolution-and-performance)), it speeds up processing at the cost of fine detail.  
   - A **progress bar** updates you on how many images have been masked and provides an estimate of how long remains.

Once batch masking completes, **all images** in that set will have a masked `.jpg` color output and `_mask.jpg` binary mask in the Output folder.  

---

### (B) Single Image Masking

Use this when you want more precise masks or to correct images from the batch approach:

1. **Navigate** to the problematic image using the **\<** and **\>** buttons.  
2. Click **Place Bounding Box** to define a bounding ROI around that single image's object. 
   - **Note:** If the image was already masked, you'll be prompted to remove the existing mask before placing a new ROI.
3. **Optionally** place **inclusion (green)** or **exclusion (red)** fiducial points:
   - **Add Inclusion Points**: Mark parts of the object that were incorrectly excluded.
   - **Add Exclusion Points**: Mark background areas or clutter incorrectly included.  
   - After choosing one of these modes, click anywhere in the 2D viewer to drop points.
   - **Click 'Stop Adding'** once you're done placing those points.
4. Finally, click **Mask Current Image**. SAM uses the bounding box plus your points to refine the mask.

> You can also **clear** all points or remove them individually if you need to restart.  

**Result:** A masked image pair (`.jpg` + `_mask.jpg`) is saved to your Output folder. The 2D viewer will shows the masked version in the 'Red2' slice viewer as a preview.

---

## 7. Mask Resolution and Performance

Beneath the image set tools, you'll find these **radio buttons**:
- **Full resolution (1.0)**
- **Half resolution (0.5)**
- **Quarter resolution (0.25)**

Choosing **half** or **quarter** can speed up masking, especially on CPU. The trade-off is lower detail near object boundaries.

> **Tip**: If you have a powerful GPU, you can keep **full resolution** for maximum detail. If you notice slow performance or run out of memory, switch to a lower resolution.

---

## 8. Monitoring Masking Progress

- A label like **'Masked: 3/20'** appears, telling you how many images have a finalized mask.
- For batch mode, a **dedicated progress bar** shows the overall time remaining and a per-image estimate.
- For single-image mode, there is no progress bar, but Slicer will show a brief 'processing' message.

---

## 9. Managing WebODM

Once images are masked, you can switch to the **'WebODM'** tab to handle the reconstruction process.

### 9.1 Installing/Pulling and Launching NodeODM
The webODM tab includes tools:
1. **Launch WebODM**  
   - Checks for Docker.  
   - Pulls the `opendronemap/nodeodm:gpu` image if not present.  
   - Attempts to start the container on **port 3002** with GPU enabled.  
2. **Stop Node** 
   - Stops any running container on port 3002.

> You can ignore these if you already have a local NodeODM instance running. In that case, just enter its IP and port.

**Node IP** defaults to `127.0.0.1` with **port** `3002`. Unless you have specialized Docker port forwarding or remote server usage, stick with these.

---

## 10. Find-GCP and Marker Detection

This module also supports generating a single **GCP (Ground Control Points) file** if you have images containing ArUco markers and a known coordinate file:
- **Clone Find-GCP**: Downloads the [Find-GCP](https://github.com/zsiki/Find-GCP) Python script into your Slicer environment.  
- Provide your **Find-GCP Script** path (auto-filled if you clicked 'Clone') and your **GCP Coord File**.  
- Pick the **ArUco Dictionary ID** you used when printing/generating your markers. Commonly `2` or `4`.
- Click **Generate Single Combined GCP File**. The script will produce `combined_gcp_list.txt` in your Output folder, merging GCP references from all sets of images.

> **What is a GCP Coord File?**  
> It lists real-world coordinates for each marker ID, ensuring the final reconstruction is positioned accurately in 3D space (e.g., for georeferencing).

---

## 11. Configuring and Running WebODM Tasks

Below the **Find-GCP** panel, you'll see **WebODM Task** options:

- **Node IP** and **Node Port**: Your local or remote NodeODM service (default `127.0.0.1:3002`).
- A list of advanced parameters under 'Launch WebODM Task':
  - **ignore-gsd, matcher-neighbors, mesh-octree-depth, mesh-size,** etc.
  - Each parameter has a **tooltip**, hover over it to see more details.
- **max-concurrency**: Number of parallel processes used by the reconstruction pipeline.
- **Dataset name**: A friendly label for your reconstruction.

**Steps to run**:
1. Confirm **all sets** are fully masked (the module checks for `_mask.jpg` files).  
2. Adjust parameters or accept defaults (tuned for typical scenarios).  
3. Click **Run WebODM Task With Selected Parameters**.

The module then:
- Uploads your masked images (and optional `combined_gcp_list.txt`) to NodeODM.
- Creates a new WebODM task using these parameters.

**Task Monitoring**:
- A console log appears in the **Console Log** box. 
- You can click **Stop Monitoring** if you want to disconnect from real-time updates (the task continues on NodeODM in the background).

---

## 12. Saving and Restoring Tasks

You can **Save Task** at any point (even while a task is running). This creates a JSON file containing:
- Your folder paths
- Masking states for all sets
- WebODM parameters
- (Optional) Output directory of the WebODM results

Later, **Restore Task** re-loads everything exactly as it was, so you can continue or re-run the reconstruction.  

> **Recommendation**: Save the task JSON into the same **Output Folder** you used for the masked images, so you have everything in one place.

---

## 13. Importing the Final 3D Model into Slicer

When WebODM finishes:
1. The module automatically downloads the result into a folder named something like `WebODM_<hash>` under your Output folder.
2. Click **Import WebODM Model** to load the `odm_textured_model_geo.obj` (or equivalent) back into Slicer.
3. Slicer will switch to a **3D layout** showing you the reconstructed mesh.  

From there, you can continue analyzing or refining in Slicer, or export to other software.

## 14. Video tutorial for Photogrammetry 
https://www.youtube.com/watch?v=YRHlb0dGyNc&t=9s

## FUNDING 
Photogrammetry extension is supported by grants (DBI/2301405, OAC/2118240) from National Science Foundation to AMM (Seattle Children's Research Institute) 

## ACKNOWLEDGEMENT
Photogrammety extension uses the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to segment foreground object. For stereophotogrammetic reconstuction the extension uses the [pyODM package from the Open Drone Map project](https://github.com/OpenDroneMap/PyODM).
