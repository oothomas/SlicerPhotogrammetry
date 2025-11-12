#
# Photogrammetry.py
#
# COMPLETE MODULE CODE WITH NEW TOOLTIP FEATURE FOR WEBODM PARAMETERS
#

import os
import sys
import stat
import qt
import ctk
import vtk
import slicer
import shutil
import numpy as np
import logging
import time  # for timing
import hashlib  # used for generating a short hash
import subprocess  # for new Docker commands
import json  # NEW >> For saving/restoring reconstructions
from slicer.ScriptedLoadableModule import *
from slicer.util import extractArchive
from typing import List
import types


def convert_numpy_types(obj):
    """
    Recursively convert any NumPy numeric types to native Python
    types so json.dumps can handle them.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(x) for x in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # or convert to list
    else:
        return obj


class Photogrammetry(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Photogrammetry"
        self.parent.categories = ["SlicerMorph.Photogrammetry"]
        self.parent.dependencies = []
        self.parent.contributors = ["Oshane Thomas (SCRI), Murat Maga (SCRI)"]
        self.parent.helpText = """Photogrammetry is a 3D Slicer module designed to streamline the process of 
        photogrammetry reconstruction. This module integrates the Segment Anything Model (SAM) for semi-automatic 
        image masking and provides seamless connectivity to WebODM for generating high-quality 3D reconstructions 
        from photographs. Key features include:

        - **Batch and Single Image Masking**: Easily mask objects in images using bounding boxes, inclusion/exclusion 
        points, and SAM. - **WebODM Integration**: Launch, manage, and monitor reconstruction tasks directly within 
        the module. - **Find-GCP Support**: Generate combined Ground Control Point (GCP) lists to improve 
        reconstruction accuracy. - **Customizable Parameters**: Fine-tune WebODM settings for optimal performance and 
        reconstruction quality.

        The module is designed to handle large datasets, offering resolution settings to optimize for GPU or 
        CPU-based workflows. Users can save and restore tasks, making the workflow efficient for both small and 
        large-scale projects. Whether you're a researcher, educator, or enthusiast, SlicerPhotogrammetry provides an 
        accessible way to turn images into 3D models."""

        self.parent.acknowledgementText = """This module was developed with support from the National Science 
        Foundation under grants DBI/2301405 and OAC/2118240 awarded to AMM at Seattle Children's Research Institute. 
        """

        # Suppress VTK warnings globally
        # vtk.vtkObject.GlobalWarningDisplayOff()

        slicer.photogrammetryLO = """
        <layout type="horizontal" split="true">
          <item>
            <view class="vtkMRMLSliceNode" singletontag="Red">
              <property name="orientation" action="default">Axial</property>
              <property name="viewlabel" action="default">R</property>
              <property name="viewcolor" action="default">#F34A33</property>
            </view>
          </item>
          <item>
            <view class="vtkMRMLSliceNode" singletontag="Red2">
              <property name="orientation" action="default">Axial</property>
              <property name="viewlabel" action="default">R2</property>
              <property name="viewcolor" action="default">#F34A33</property>
            </view>
          </item>
        </layout>
        """


class PhotogrammetryWidget(ScriptedLoadableModuleWidget):
    """
    Manages UI for:
     - SAM model loading,
     - Folder processing,
     - Bbox/masking steps,
     - EXIF + color/writing,
     - Creating _mask.png for webODM,
     - Creating single combined GCP file for all sets,
     - Non-blocking WebODM tasks (using pyodm),
     - Checking/Installing/Re-launching WebODM on port 3002 with GPU support,
     - Inclusion/Exclusion point marking for SAM,
     - A "Mask All Images In Set" workflow that removes existing masks,
       lets you place an ROI bounding box for the entire set, then finalize for all,
     - Importing a completed WebODM model as an OBJ and switching to a 3D layout,
     - NEW >> Saving/Restoring WebODM tasks (via JSON).
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

        self.mainTabWidget = None
        self.imageIndexLabel = None
        self.logic = None
        self.vtkLogFilter = None
        self.logger = None

        # small in-memory cache
        self.imageCache = {}
        self.maxCacheSize = 64

        self.downsampleFactor = 0.15  # e.g. 15% for slice display usage

        # Master nodes
        self.masterVolumeNode = None
        self.masterLabelMapNode = None
        self.masterMaskedVolumeNode = None
        self.emptyNode = None

        self.boundingBoxFiducialNode = None

        self.setStates = {}
        self.currentSet = None
        self.imagePaths = []
        self.currentImageIndex = 0

        self.imageStates = {}

        self.createdNodes = []
        self.currentBboxLineNodes = []
        self.boundingBoxRoiNode = None

        self.buttonsToManage = []
        self.prevButton = None
        self.nextButton = None

        # Single "Mask Current Image" button
        self.maskCurrentImageButton = None

        self.maskAllImagesButton = None
        self.finalizeAllMaskButton = None  # <-- NEW button
        self.finalizingROI = False
        self.maskAllProgressBar = None
        self.processButton = None
        self.imageSetComboBox = None
        self.placeBoundingBoxButton = None
        self.outputFolderSelector = None
        self.masterFolderSelector = None
        self.processFoldersProgressBar = None
        self.previousButtonStates = {}

        # Model selection UI
        self.samVariantCombo = None
        self.loadModelButton = None
        self.modelLoaded = False

        # GCP
        self.findGCPScriptSelector = None
        self.generateGCPButton = None
        self.gcpCoordFileSelector = None
        self.arucoDictIDSpinBox = None
        self.gcpListContent = ""
        self.gcpCoordFilePath = ""

        # "Clone Find-GCP" button
        self.cloneFindGCPButton = None

        # WebODM: simplified to two main buttons plus extra UI
        self.nodeIPLineEdit = None
        self.nodePortSpinBox = None
        self.launchWebODMTaskButton = None
        self.webodmLogTextEdit = None
        self.stopMonitoringButton = None
        self.lastWebODMOutputLineIndex = 0

        # Replaces older "Check/Install/Relaunch" with new "Launch" + "Stop Node"
        self.launchWebODMButton = None
        self.stopWebODMButton = None

        # Button to import WebODM model (OBJ) into Slicer as a 3D model
        self.importModelButton = None

        # ------------------
        # 1) Baseline params (some remain constant, others were removed to vary)
        # ------------------
        self.baselineParams = {
            "orthophoto-resolution": 0.3,
            "skip-orthophoto": True,
            "texturing-single-material": True,
            "use-3dmesh": True,
        }

        # ------------------
        # 2) Factor levels, including newly added parameters
        # ------------------
        self.factorLevels = {
            "ignore-gsd": [False, True],
            "matcher-neighbors": [16, 0, 8, 12, 24],
            "mesh-octree-depth": [12, 13, 14],
            "mesh-size": [300000, 500000, 750000, 1000000],
            "min-num-features": [50000, 10000, 20000],
            "pc-filter": [1, 2, 3, 4, 5],
            "depthmap-resolution": [3072, 2048, 4096, 8192],
            "matcher-type": ["bruteforce", "bow", "flann"],
            "feature-type": ["dspsift", "akaze", "hahog", "orb", "sift"],
            "feature-quality": ["ultra", "medium", "high"],
            "pc-quality": ["high", "medium", "ultra"],
            "optimize-disk-space": [True, False],
            "rerun": ["openmvs", "dataset", "split", "merge", "opensfm"],
            "no-gpu": [False, True],
        }
        self.factorComboBoxes = {}

        # We'll create separate UI elements for:
        #  - max-concurrency (QSpinBox, 16..256)
        #  - name (QLineEdit, default "SlicerReconstruction")
        self.maxConcurrencySpinBox = None
        self.datasetNameLineEdit = None

        self.maskedCountLabel = None

        # We register a new custom layout ID
        self.layoutId = 1003

        # WebODM installation
        self.webODMManager = None
        self.webODMLocalFolder = None

        # We store references to 3 radio buttons for resolution
        self.radioFull = None
        self.radioHalf = None
        self.radioQuarter = None

        #
        # NEW: Inclusion/Exclusion points
        #
        self.exclusionPointNode = None
        self.exclusionPointAddedObserverTag = None

        self.inclusionPointNode = None
        self.inclusionPointAddedObserverTag = None

        self.addInclusionPointsButton = None
        self.addExclusionPointsButton = None
        self.stopAddingPointsButton = None
        self.clearPointsButton = None

        # NEW: Keep track of whether we're in a special "mask all images" mode
        self.globalMaskAllInProgress = False

        # NEW >> Additional references to handle Save/Restore of tasks
        self.saveTaskButton = None
        self.restoreTaskButton = None

        self.iconGreen = self.createColoredIcon(qt.QColor(0, 200, 0))
        self.iconRed = self.createColoredIcon(qt.QColor(200, 0, 0))

    def setup(self):
        """
        Sets up the module GUI and logic, including:

        - Model loading UI
        - Image set processing UI
        - Masking controls
        - WebODM management & launching
        - Factor combos for WebODM tasks
        - GCP generation
        - Import model button (new)
        - SAVE/RESTORE task feature (new).
        """

        ScriptedLoadableModuleWidget.setup(self)
        self.load_dependencies()
        self.logic = PhotogrammetryLogic()

        self.setupLogger()
        self.layout.setAlignment(qt.Qt.AlignTop)
        self.createCustomLayout()  # Register and create our layout

        #
        # Create a QTabWidget to hold two tabs
        #
        self.mainTabWidget = qt.QTabWidget()
        self.layout.addWidget(self.mainTabWidget)

        # Tab 1: "Image Masking"
        tab1Widget = qt.QWidget()
        tab1Layout = qt.QVBoxLayout(tab1Widget)
        tab1Widget.setLayout(tab1Layout)

        # Tab 2: "WebODM"
        tab2Widget = qt.QWidget()
        tab2Layout = qt.QVBoxLayout(tab2Widget)
        tab2Widget.setLayout(tab2Layout)

        self.mainTabWidget.addTab(tab1Widget, "Image Masking")
        self.mainTabWidget.addTab(tab2Widget, "NodeODM")

        #
        # (A) Main Collapsible: Import Image Sets
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Import Image Sets"
        tab1Layout.addWidget(parametersCollapsibleButton)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.samVariantCombo = qt.QComboBox()
        self.samVariantCombo.addItem("ViT-base (~376 MB)")
        self.samVariantCombo.addItem("ViT-large (~1.03 GB)")
        self.samVariantCombo.addItem("ViT-huge (~2.55 GB)")
        parametersFormLayout.addRow("SAM Variant:", self.samVariantCombo)

        self.loadModelButton = qt.QPushButton("Load Model")
        parametersFormLayout.addWidget(self.loadModelButton)
        self.loadModelButton.connect('clicked(bool)', self.onLoadModelClicked)

        self.masterFolderSelector = ctk.ctkDirectoryButton()
        savedMasterFolder = slicer.app.settings().value("Photogrammetry/masterFolderPath", "")
        if os.path.isdir(savedMasterFolder):
            self.masterFolderSelector.directory = savedMasterFolder
        parametersFormLayout.addRow("Input Folder:", self.masterFolderSelector)

        self.outputFolderSelector = ctk.ctkDirectoryButton()
        savedOutputFolder = slicer.app.settings().value("Photogrammetry/outputFolderPath", "")
        if os.path.isdir(savedOutputFolder):
            self.outputFolderSelector.directory = savedOutputFolder
        parametersFormLayout.addRow("Output Folder:", self.outputFolderSelector)

        self.processButton = qt.QPushButton("Process Folders")
        parametersFormLayout.addWidget(self.processButton)
        self.processButton.connect('clicked(bool)', self.onProcessFoldersClicked)

        self.processFoldersProgressBar = qt.QProgressBar()
        self.processFoldersProgressBar.setVisible(False)
        parametersFormLayout.addWidget(self.processFoldersProgressBar)

        self.maskedCountLabel = qt.QLabel("0/0 Masked")
        parametersFormLayout.addRow("Overall Progress:", self.maskedCountLabel)

        self.imageSetComboBox = qt.QComboBox()
        self.imageSetComboBox.enabled = False
        parametersFormLayout.addRow("Image Set:", self.imageSetComboBox)
        self.imageSetComboBox.connect('currentIndexChanged(int)', self.onImageSetSelected)

        # Create a QTableWidget with 2 columns: "Index" and "Filename"
        self.imageTable = qt.QTableWidget()
        self.imageTable.setColumnCount(2)
        self.imageTable.setHorizontalHeaderLabels(["Image", "Filename"])
        self.imageTable.verticalHeader().hide()  # Hide the built-in row number labels
        self.imageTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.imageTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.imageTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.imageTable.setMinimumHeight(200)  # Optional: ensures some visible space
        # Make the second column stretch to fill the width:
        self.imageTable.horizontalHeader().setStretchLastSection(True)

        # Add the table widget to the form layout
        parametersFormLayout.addRow("Image List:", self.imageTable)

        # Connect the cellClicked signal to a slot we'll define:
        self.imageTable.cellClicked.connect(self.onImageTableCellClicked)

        # Group box for resolution selection
        resGroupBox = qt.QGroupBox("Masking Resolution")
        resLayout = qt.QVBoxLayout(resGroupBox)

        self.radioFull = qt.QRadioButton("Full resolution (1.0)")
        self.radioFull.setChecked(True)  # default
        self.radioHalf = qt.QRadioButton("Half resolution (0.5)")
        self.radioQuarter = qt.QRadioButton("Quarter resolution (0.25)")

        resLayout.addWidget(self.radioFull)
        resLayout.addWidget(self.radioHalf)
        resLayout.addWidget(self.radioQuarter)

        parametersFormLayout.addWidget(resGroupBox)

        navLayout = qt.QGridLayout()

        self.prevButton = qt.QPushButton("<")
        self.prevButton.setToolTip("Go to the previous image")

        self.nextButton = qt.QPushButton(">")
        self.nextButton.setToolTip("Go to the next image")

        self.imageIndexLabel = qt.QLabel("Image 1")
        self.imageIndexLabel.setAlignment(qt.Qt.AlignCenter)

        self.prevButton.enabled = False
        self.nextButton.enabled = False

        navLayout.addWidget(self.prevButton, 0, 0)
        navLayout.addWidget(self.imageIndexLabel, 0, 1)
        navLayout.addWidget(self.nextButton, 0, 2)
        parametersFormLayout.addRow("    ", navLayout)

        self.prevButton.connect('clicked(bool)', self.onPrevImage)
        self.nextButton.connect('clicked(bool)', self.onNextImage)

        #
        # "Batch Masking" row with "Mask All" + "Finalize All" buttons
        #
        maskAllLayout = qt.QHBoxLayout()
        self.maskAllImagesButton = qt.QPushButton("Place/Adjust ROI for All Images")
        self.maskAllImagesButton.enabled = False
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)
        maskAllLayout.addWidget(self.maskAllImagesButton)

        self.finalizeAllMaskButton = qt.QPushButton("Finalize ROI and Mask All Images")
        self.finalizeAllMaskButton.enabled = False
        self.finalizeAllMaskButton.connect('clicked(bool)', self.onFinalizeAllMaskClicked)
        maskAllLayout.addWidget(self.finalizeAllMaskButton)

        parametersFormLayout.addRow("Mask Batch:", maskAllLayout)

        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        self.maskAllProgressBar.setTextVisible(True)
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        #
        # NEW: Inclusion/Exclusion points UI
        #
        pointsButtonsLayout = qt.QHBoxLayout()

        self.addInclusionPointsButton = qt.QPushButton("Add Inclusion Points")
        self.addInclusionPointsButton.enabled = False
        self.addInclusionPointsButton.connect('clicked(bool)', self.onAddInclusionPointsClicked)
        pointsButtonsLayout.addWidget(self.addInclusionPointsButton)

        self.addExclusionPointsButton = qt.QPushButton("Add Exclusion Points")
        self.addExclusionPointsButton.enabled = False
        self.addExclusionPointsButton.connect('clicked(bool)', self.onAddExclusionPointsClicked)
        pointsButtonsLayout.addWidget(self.addExclusionPointsButton)

        self.stopAddingPointsButton = qt.QPushButton("Stop Adding")
        self.stopAddingPointsButton.enabled = False
        self.stopAddingPointsButton.connect('clicked(bool)', self.onStopAddingPointsClicked)
        pointsButtonsLayout.addWidget(self.stopAddingPointsButton)

        self.clearPointsButton = qt.QPushButton("Clear Points")
        self.clearPointsButton.enabled = False
        self.clearPointsButton.connect('clicked(bool)', self.onClearPointsClicked)
        pointsButtonsLayout.addWidget(self.clearPointsButton)

        parametersFormLayout.addRow(" ", pointsButtonsLayout)

        singleMaskingLayout = qt.QHBoxLayout()

        self.placeBoundingBoxButton = qt.QPushButton("Place Bounding Box")
        self.placeBoundingBoxButton.enabled = False
        singleMaskingLayout.addWidget(self.placeBoundingBoxButton)
        self.placeBoundingBoxButton.connect('clicked(bool)', self.onPlaceBoundingBoxClicked)

        self.maskCurrentImageButton = qt.QPushButton("Mask Current Image")
        self.maskCurrentImageButton.enabled = False
        singleMaskingLayout.addWidget(self.maskCurrentImageButton)
        self.maskCurrentImageButton.connect('clicked(bool)', self.onMaskCurrentImageClicked)

        parametersFormLayout.addRow("Mask Image:", singleMaskingLayout)
        # Save references for enabling/disabling
        self.buttonsToManage = [
            self.masterFolderSelector,
            self.outputFolderSelector,
            self.processButton,
            self.imageSetComboBox,
            self.placeBoundingBoxButton,
            self.maskCurrentImageButton,
            self.maskAllImagesButton,
            self.prevButton,
            self.nextButton
        ]
        for btn in self.buttonsToManage:
            if isinstance(btn, qt.QComboBox):
                btn.setEnabled(False)
            else:
                btn.enabled = False

        #
        # NEW >> Save/Restore Collapsible
        #
        saveRestoreCollapsible = ctk.ctkCollapsibleButton()
        saveRestoreCollapsible.text = "Save/Restore Reconstruction Task"
        tab1Layout.addWidget(saveRestoreCollapsible)

        saveRestoreLayout = qt.QFormLayout(saveRestoreCollapsible)

        self.saveTaskButton = qt.QPushButton("Save Task")
        self.restoreTaskButton = qt.QPushButton("Restore Task")
        buttonsRow = qt.QHBoxLayout()
        buttonsRow.addWidget(self.saveTaskButton)
        self.saveTaskButton.enabled = False
        self.restoreTaskButton.enabled = False
        buttonsRow.addWidget(self.restoreTaskButton)
        saveRestoreLayout.addRow(buttonsRow)

        self.saveTaskButton.connect('clicked(bool)', self.onSaveTaskClicked)
        self.restoreTaskButton.connect('clicked(bool)', self.onRestoreTaskClicked)

        tab1Layout.addStretch(1)
        # END NEW >>

        #
        # (B) Manage WebODM (Install/Launch) Collapsible
        #
        manageWODMCollapsibleButton = ctk.ctkCollapsibleButton()
        manageWODMCollapsibleButton.text = "Manage WebODM (Install/Launch)"
        tab2Layout.addWidget(manageWODMCollapsibleButton)
        manageWODMFormLayout = qt.QFormLayout(manageWODMCollapsibleButton)

        buttonRow = qt.QHBoxLayout()
        self.launchWebODMButton = qt.QPushButton("Launch NodeODM")
        self.stopWebODMButton = qt.QPushButton("Stop Node")

        buttonRow.addWidget(self.launchWebODMButton)
        buttonRow.addWidget(self.stopWebODMButton)
        manageWODMFormLayout.addRow(buttonRow)

        #
        # (C) Find-GCP Collapsible
        #
        webODMCollapsibleButton = ctk.ctkCollapsibleButton()
        webODMCollapsibleButton.text = "Find-GCP"
        tab2Layout.addWidget(webODMCollapsibleButton)
        webODMFormLayout = qt.QFormLayout(webODMCollapsibleButton)

        self.cloneFindGCPButton = qt.QPushButton("Clone Find-GCP")
        webODMFormLayout.addWidget(self.cloneFindGCPButton)
        self.cloneFindGCPButton.connect('clicked(bool)', self.onCloneFindGCPClicked)

        self.findGCPScriptSelector = ctk.ctkPathLineEdit()
        self.findGCPScriptSelector.filters = ctk.ctkPathLineEdit().Files
        self.findGCPScriptSelector.setToolTip("Select path to Find-GCP.py script.")
        webODMFormLayout.addRow("Find-GCP Script:", self.findGCPScriptSelector)

        savedFindGCPScript = slicer.app.settings().value("Photogrammetry/findGCPScriptPath", "")
        if os.path.isfile(savedFindGCPScript):
            self.findGCPScriptSelector.setCurrentPath(savedFindGCPScript)
        self.findGCPScriptSelector.connect('currentPathChanged(QString)', self.onFindGCPScriptChanged)

        self.gcpCoordFileSelector = ctk.ctkPathLineEdit()
        self.gcpCoordFileSelector.filters = ctk.ctkPathLineEdit().Files
        self.gcpCoordFileSelector.setToolTip("Select GCP coordinate file (required).")
        webODMFormLayout.addRow("GCP Coord File:", self.gcpCoordFileSelector)

        self.arucoDictIDSpinBox = qt.QSpinBox()
        self.arucoDictIDSpinBox.setMinimum(0)
        self.arucoDictIDSpinBox.setMaximum(99)
        self.arucoDictIDSpinBox.setValue(2)
        webODMFormLayout.addRow("ArUco Dictionary ID:", self.arucoDictIDSpinBox)

        self.generateGCPButton = qt.QPushButton("Generate Single Combined GCP File (All Sets)")
        webODMFormLayout.addWidget(self.generateGCPButton)
        self.generateGCPButton.connect('clicked(bool)', self.onGenerateGCPClicked)
        self.generateGCPButton.setEnabled(True)

        #
        # (D) Launch WebODM Task Collapsible
        #
        webodmTaskCollapsible = ctk.ctkCollapsibleButton()
        webodmTaskCollapsible.text = "Launch WebODM Task"
        tab2Layout.addWidget(webodmTaskCollapsible)
        webodmTaskFormLayout = qt.QFormLayout(webodmTaskCollapsible)

        self.nodeIPLineEdit = qt.QLineEdit("127.0.0.1")
        self.nodeIPLineEdit.setToolTip("Enter the IP address of the NodeODM instance (e.g. 127.0.0.1).")
        webodmTaskFormLayout.addRow("Node IP:", self.nodeIPLineEdit)

        self.nodePortSpinBox = qt.QSpinBox()
        self.nodePortSpinBox.setMinimum(1)
        self.nodePortSpinBox.setMaximum(65535)
        self.nodePortSpinBox.setValue(3002)
        self.nodePortSpinBox.setToolTip("Port number on which NodeODM is listening. Commonly 3001 or 3002.")
        webodmTaskFormLayout.addRow("Node Port:", self.nodePortSpinBox)

        # ----------------
        # NEW >> Tooltips for each WebODM parameter
        # ----------------
        parameterTooltips = {
            "ignore-gsd": (
                "Ignore Ground Sampling Distance (GSD). A memory/processor-hungry setting if true.\n"
                "Ordinarily, GSD caps maximum resolution. Use with caution.\nDefault: False"
            ),
            "matcher-neighbors": (
                "Perform image matching with the nearest images based on GPS exif data.\n"
                "Set to 0 to match by triangulation.\nDefault: 0"
            ),
            "mesh-octree-depth": (
                "Octree depth used in mesh reconstruction. Increase for more vertices.\n"
                "Typical range 8-12.\nDefault: 11"
            ),
            "mesh-size": (
                "Max vertex count for the output mesh.\nDefault: 200000"
            ),
            "min-num-features": (
                "Minimum number of features to extract per image.\n"
                "Higher values can help with low-overlap areas but increase processing.\nDefault: 10000"
            ),
            "pc-filter": (
                "Filters the point cloud by removing outliers.\n"
                "Value = # of standard deviations from local mean.\nDefault: 5"
            ),
            "depthmap-resolution": (
                "Sets the resolution for depth maps.\n"
                "Higher values = more detail, but more memory/time.\nTypical range 2048..8192.\nDefault: 2048"
            ),
            "matcher-type": (
                "Matcher algorithm: bruteforce, bow, or flann.\n"
                "FLANN is slower but stable, BOW is faster but might miss matches,\n"
                "BRUTEFORCE is slow but robust.\nDefault: flann"
            ),
            "feature-type": (
                "Keypoint/descriptor algorithm: akaze, dspsift, hahog, orb, sift.\n"
                "Default: dspsift"
            ),
            "feature-quality": (
                "Feature extraction quality: ultra, high, medium, low, lowest.\n"
                "Higher quality = better features, but slower.\nDefault: high"
            ),
            "pc-quality": (
                "Point cloud quality: ultra, high, medium, low, lowest.\n"
                "Higher = denser cloud, more resources.\nDefault: medium"
            ),
            "optimize-disk-space": (
                "Delete large intermediate files to reduce disk usage.\n"
                "Prevents partial pipeline restarts.\nDefault: False"
            ),
            "rerun": (
                "Rerun only a specific pipeline stage and stop.\n"
                "Options: dataset, split, merge, opensfm, openmvs, etc.\n"
                "Default: (none)"
            ),
            "no-gpu": (
                "Disable GPU usage even if available.\nDefault: False"
            ),
        }
        # ----------------

        for factorName, levels in self.factorLevels.items():
            combo = qt.QComboBox()
            for val in levels:
                combo.addItem(str(val))

            # Assign the tooltip text for each parameter from the dictionary above:
            if factorName in parameterTooltips:
                combo.setToolTip(parameterTooltips[factorName])
            else:
                combo.setToolTip(f"Parameter '{factorName}' is not documented in the tooltips dictionary.")

            self.factorComboBoxes[factorName] = combo
            webodmTaskFormLayout.addRow(f"{factorName}:", combo)

        self.maxConcurrencySpinBox = qt.QSpinBox()
        self.maxConcurrencySpinBox.setRange(16, 256)
        self.maxConcurrencySpinBox.setValue(16)
        self.maxConcurrencySpinBox.setToolTip(
            "Maximum number of processes used by WebODM.\n"
            "Higher values = faster but more memory usage."
        )
        webodmTaskFormLayout.addRow("max-concurrency:", self.maxConcurrencySpinBox)

        self.datasetNameLineEdit = qt.QLineEdit("SlicerReconstruction")
        self.datasetNameLineEdit.setToolTip(
            "Name of the dataset in WebODM.\nThis will be the reconstruction folder label."
        )
        webodmTaskFormLayout.addRow("name:", self.datasetNameLineEdit)

        self.launchWebODMTaskButton = qt.QPushButton("Run NodeODM Task With Selected Parameters (non-blocking)")
        webodmTaskFormLayout.addWidget(self.launchWebODMTaskButton)
        self.launchWebODMTaskButton.setEnabled(False)
        self.launchWebODMTaskButton.connect('clicked(bool)', self.onRunWebODMTask)

        self.webodmLogTextEdit = qt.QTextEdit()
        self.webodmLogTextEdit.setReadOnly(True)
        webodmTaskFormLayout.addRow("Console Log:", self.webodmLogTextEdit)

        self.stopMonitoringButton = qt.QPushButton("Stop Monitoring")
        self.stopMonitoringButton.setEnabled(False)
        webodmTaskFormLayout.addWidget(self.stopMonitoringButton)

        self.importModelButton = qt.QPushButton("Import Reconstructed Model")
        tab2Layout.addWidget(self.importModelButton)

        tab2Layout.addStretch(1)

        # Connect new simplified button signals
        self.launchWebODMButton.connect('clicked(bool)', self.onLaunchWebODMClicked)
        self.stopWebODMButton.connect('clicked(bool)', self.onStopNodeClicked)
        self.stopMonitoringButton.connect('clicked(bool)', self.onStopMonitoring)
        self.importModelButton.connect('clicked(bool)', self.onImportModelClicked)

        self.createMasterNodes()

        modulePath = os.path.dirname(slicer.modules.photogrammetry.path)
        self.webODMLocalFolder = os.path.join(modulePath, 'Resources', 'WebODM')

        # Ensure the folder exists with proper permissions
        self.ensure_webodm_folder_permissions()

        self.webODMManager = SlicerWebODMManager(widget=self)
        self.initializeInclusionMarkupsNode()
        self.initializeExclusionMarkupsNode()

        self.addLayoutButton(self.layoutId, "Double Red Viewport",
                             "Custom Layout for Photogrammetry Module",
                             "red_squared_lo_icon.png", slicer.photogrammetryLO)

    def createColoredIcon(self, color, size=16):
        pixmap = qt.QPixmap(size, size)
        pixmap.fill(color)
        return qt.QIcon(pixmap)

    def isSetFullyMasked(self, setName):
        setInfo = self.setStates.get(setName, None)
        if not setInfo:
            return False
        # If any image is not masked, return False
        for _, imageState in setInfo["imageStates"].items():
            if imageState["state"] != "masked":
                return False
        return True

    def onImageTableCellClicked(self, row, column):
        """
        When the user clicks a table row, jump directly to that image.
        """
        self.currentImageIndex = row  # zero-based
        self.updateVolumeDisplay()

    def updateImageTable(self):
        """
        Refreshes the table rows based on self.imagePaths and self.imageStates.
        Red highlight = unmasked, Green highlight = masked.
        """
        if not self.imagePaths:
            self.imageTable.setRowCount(0)
            return

        numImages = len(self.imagePaths)
        self.imageTable.setRowCount(numImages)

        for rowIndex in range(numImages):
            # We assume one-based indexing for display
            displayIndex = rowIndex + 1
            imagePath = self.imagePaths[rowIndex]
            baseName = os.path.basename(imagePath)

            # Create QTableWidgetItems for the two columns
            indexItem = qt.QTableWidgetItem(str(displayIndex))
            fileItem = qt.QTableWidgetItem(baseName)

            # Check if masked or not
            st = self.imageStates[rowIndex]["state"]
            if st == "masked":
                # green
                color = qt.QColor(200, 255, 200)
            else:
                # "none" or "bbox" => not fully masked => red
                color = qt.QColor(255, 200, 200)

            brush = qt.QBrush(color)
            indexItem.setBackground(brush)
            fileItem.setBackground(brush)

            self.imageTable.setItem(rowIndex, 0, indexItem)
            self.imageTable.setItem(rowIndex, 1, fileItem)

        # Optional: auto-resize the first column for the index
        # self.imageTable.resizeColumnToContents(0)

    def ensure_webodm_folder_permissions(self):
        """
        Create the WebODM folder if it doesn't exist and set permissions.
        """
        try:
            if not os.path.exists(self.webODMLocalFolder):
                os.makedirs(self.webODMLocalFolder)
            os.chmod(self.webODMLocalFolder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logging.info(f"WebODM folder created and permissions set: {self.webODMLocalFolder}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to create or set permissions for WebODM folder:\n{str(e)}")

    def load_dependencies(self):
        """
        Ensure all needed Python dependencies are installed.
        """
        import slicer

        try:
            import OBJFile
        except ModuleNotFoundError:
            slicer.util.messageBox("OBJFile from the SlicerMorph Extension is required. "
                                   "Please install SlicerMorph from the Extensions Manager.")

        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.messageBox("Photogrammetry requires the PyTorch extension. "
                                   "Please install it from the Extensions Manager.")
        torchLogic = None
        try:
            import PyTorchUtils
            torchLogic = PyTorchUtils.PyTorchUtilsLogic()
            if not torchLogic.torchInstalled():

                if not slicer.util.confirmOkCancelDisplay(
                        f"This module requires installation of additional Python packages. Installation needs network "
                        f"connection and may take several minutes. Click OK to proceed.",
                        "Confirm Python package installation"
                ):
                    raise InstallError("User cancelled.")

                logging.debug('Installing PyTorch...')
                torch = torchLogic.installTorch(askConfirmation=True, forceComputationBackend='cu124')
                if torch:
                    restart = slicer.util.confirmYesNoDisplay(
                        "Pytorch dependencies have been installed. A restart of 3D Slicer is needed. Restart now?"
                    )
                    if restart:
                        slicer.util.restart()
                if torch is None:
                    slicer.util.messageBox('PyTorch must be installed manually to use this module.')
        except Exception:
            pass

        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
        except ImportError:
            slicer.util.pip_install("Pillow")
            from PIL import Image
            from PIL.ExifTags import TAGS

        try:
            import cv2
            if not hasattr(cv2, 'xfeatures2d'):
                raise ImportError("opencv-contrib-python is not properly installed")
        except ImportError:
            slicer.util.pip_install("opencv-python")
            slicer.util.pip_install("opencv-contrib-python")
            import cv2

        try:
            import segment_anything
        except ImportError:
            import os
            from slicer.util import downloadFile, extractArchive, pip_install

            # 1) Decide where to put the downloaded ZIP locally:
            modulePath = os.path.dirname(slicer.modules.photogrammetry.path)
            resourcesFolder = os.path.join(modulePath, "Resources")
            if not os.path.isdir(resourcesFolder):
                os.makedirs(resourcesFolder)

            # 2) Direct download link
            url = "https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip"

            try:
                pip_install(url)
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to pip-install segment-anything from {localExtractDir}:\n{str(e)}")
                raise

            # 3) Finally, import again
            import segment_anything

        try:
            import pyodm
        except ImportError:
            slicer.util.pip_install("pyodm")
            import pyodm

        try:
            import matplotlib
        except ImportError:
            slicer.util.pip_install("matplotlib")
            import matplotlib

        from segment_anything import sam_model_registry, SamPredictor

    def createCustomLayout(self):
        """
        Register a new custom layout that includes side-by-side Red and Red2 slices.
        Also make it available in the Slicer layout selector.
        """

        if not slicer.app.layoutManager().layoutLogic().GetLayoutNode().SetLayoutDescription(self.layoutId,
                                                                                             slicer.photogrammetryLO):
            slicer.app.layoutManager().layoutLogic().GetLayoutNode().AddLayoutDescription(self.layoutId,
                                                                                          slicer.photogrammetryLO)

        slicer.app.layoutManager().setLayout(self.layoutId)

    def addLayoutButton(self, layoutID, buttonAction, toolTip, imageFileName, layoutDiscription):
        layoutManager = slicer.app.layoutManager()
        layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(layoutID, layoutDiscription)

        viewToolBar = slicer.util.mainWindow().findChild('QToolBar', 'ViewToolBar')
        layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
        layoutSwitchActionParent = layoutMenu
        layoutSwitchAction = layoutSwitchActionParent.addAction(buttonAction)  # add inside layout list

        moduleDir = os.path.dirname(slicer.util.modulePath(self.__module__))
        iconPath = os.path.join(moduleDir, 'Resources/Icons', imageFileName)
        layoutSwitchAction.setIcon(qt.QIcon(iconPath))
        layoutSwitchAction.setToolTip(toolTip)
        layoutSwitchAction.connect('triggered()',
                                   lambda layoutId=layoutID: slicer.app.layoutManager().setLayout(layoutId))
        layoutSwitchAction.setData(layoutID)

    def setupLogger(self):
        class VTKLogFilter(logging.Filter):
            def filter(self, record):
                suppressed_messages = [
                    "Input port 0 of algorithm vtkImageMapToWindowLevelColors",
                    "Input port 0 of algorithm vtkImageThreshold",
                    "Initialize(): no image data to interpolate!"
                ]
                return not any(msg in record.getMessage() for msg in suppressed_messages)

        self.logger = logging.getLogger('VTK')
        self.vtkLogFilter = VTKLogFilter()
        self.logger.addFilter(self.vtkLogFilter)

    def createMasterNodes(self):
        if self.masterVolumeNode and slicer.mrmlScene.IsNodePresent(self.masterVolumeNode):
            slicer.mrmlScene.RemoveNode(self.masterVolumeNode)
        self.masterVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", "ImageVolume")
        self.masterVolumeNode.CreateDefaultDisplayNodes()

        if self.masterLabelMapNode and slicer.mrmlScene.IsNodePresent(self.masterLabelMapNode):
            slicer.mrmlScene.RemoveNode(self.masterLabelMapNode)
        self.masterLabelMapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "MaskOverlay")
        self.masterLabelMapNode.CreateDefaultDisplayNodes()

        if self.masterMaskedVolumeNode and slicer.mrmlScene.IsNodePresent(self.masterMaskedVolumeNode):
            slicer.mrmlScene.RemoveNode(self.masterMaskedVolumeNode)
        self.masterMaskedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", "MaskVolume")
        self.masterMaskedVolumeNode.CreateDefaultDisplayNodes()

        self.emptyNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "NullVolume")
        self.emptyNode.CreateDefaultDisplayNodes()
        import numpy as np
        dummy = np.zeros((1, 1), dtype=np.uint8)
        slicer.util.updateVolumeFromArray(self.emptyNode, dummy)

        self.masterVolumeNode.GetDisplayNode().SetInterpolate(False)
        self.masterMaskedVolumeNode.GetDisplayNode().SetInterpolate(False)

        redSliceLogic = slicer.app.layoutManager().sliceWidget('Red').sliceLogic()
        redSliceNode = redSliceLogic.GetSliceNode()
        redSliceNode.SetUseLabelOutline(False)

        red2SliceLogic = slicer.app.layoutManager().sliceWidget('Red2').sliceLogic()
        red2SliceNode = red2SliceLogic.GetSliceNode()
        red2SliceNode.SetUseLabelOutline(True)

        lm = slicer.app.layoutManager()
        redComp = lm.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComp.SetBackgroundVolumeID(self.masterVolumeNode.GetID())
        redComp.SetLabelVolumeID(self.masterLabelMapNode.GetID())

        red2Comp = lm.sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Comp.SetBackgroundVolumeID(self.masterMaskedVolumeNode.GetID())

    def onFindGCPScriptChanged(self, newPath):
        slicer.app.settings().setValue("Photogrammetry/findGCPScriptPath", newPath)

    def updateMaskedCounter(self):
        totalImages = 0
        maskedCount = 0
        for setName, setData in self.setStates.items():
            totalImages += len(setData["imagePaths"])
            for _, info in setData["imageStates"].items():
                if info["state"] == "masked":
                    maskedCount += 1
        self.maskedCountLabel.setText(f"{maskedCount}/{totalImages} Masked")

    def onLoadModelClicked(self):
        variant = self.samVariantCombo.currentText
        variantInfo = {
            "ViT-huge (~2.55 GB)": {
                "filename": "sam_vit_h_4b8939.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "registry_key": "vit_h"
            },
            "ViT-large (~1.03 GB)": {
                "filename": "sam_vit_l_0b3195.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "registry_key": "vit_l"
            },
            "ViT-base (~376 MB)": {
                "filename": "sam_vit_b_01ec64.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "registry_key": "vit_b"
            }
        }
        info = variantInfo[variant]
        success = self.logic.loadSAMModel(
            variant=info["registry_key"],
            filename=info["filename"],
            url=info["url"]
        )
        if success:
            slicer.util.infoDisplay(f"{variant} model loaded successfully.")
            self.modelLoaded = True
            self.processButton.setEnabled(True)
            self.masterFolderSelector.setEnabled(True)
            self.outputFolderSelector.setEnabled(True)
            self.samVariantCombo.setEnabled(False)
            self.loadModelButton.setEnabled(False)
            self.restoreTaskButton.setEnabled(True)
        else:
            slicer.util.errorDisplay("Failed to load the model. Check logs.")

    def updateSetComboBoxIcons(self):
        """
        Loop over each combo box item, check if that set is fully masked,
        and set its icon green/red accordingly.
        """
        for i in range(self.imageSetComboBox.count):
            setName = self.imageSetComboBox.itemText(i)
            icon = self.iconGreen if self.isSetFullyMasked(setName) else self.iconRed
            self.imageSetComboBox.setItemIcon(i, icon)

    def onProcessFoldersClicked(self):
        """
        Automatically load every subfolder (set) under the master folder into self.setStates.
        Then check for pre-existing masks in each set, so updateMaskedCounter() displays the
        global total across all sets.
        """
        if not self.modelLoaded:
            slicer.util.warningDisplay("Please load a SAM model before processing folders.")
            return

        if self.anySetHasProgress():
            if not slicer.util.confirmYesNoDisplay(
                    "All progress made so far will be lost. Proceed?"
            ):
                return
            self.clearAllData()

        masterFolderPath = self.masterFolderSelector.directory
        outputFolderPath = self.outputFolderSelector.directory
        if not os.path.isdir(masterFolderPath):
            slicer.util.errorDisplay("Please select a valid master folder.")
            return
        if not os.path.isdir(outputFolderPath):
            slicer.util.errorDisplay("Please select a valid output folder.")
            return

        slicer.app.settings().setValue("Photogrammetry/masterFolderPath", masterFolderPath)
        slicer.app.settings().setValue("Photogrammetry/outputFolderPath", outputFolderPath)

        # Prepare subfolders
        subfolders = [f for f in os.listdir(masterFolderPath)
                      if os.path.isdir(os.path.join(masterFolderPath, f))]
        subfolders = sorted(subfolders)

        self.imageSetComboBox.clear()
        self.processFoldersProgressBar.setVisible(True)
        self.processFoldersProgressBar.setRange(0, len(subfolders))
        self.processFoldersProgressBar.setValue(0)

        for idx, sf in enumerate(subfolders):
            #self.imageSetComboBox.addItem(sf)

            setFolderPath = os.path.join(masterFolderPath, sf)
            imagePaths = self.logic.get_image_paths_from_folder(setFolderPath)

            if len(imagePaths) == 0:
                # You can skip empty sets or leave them as-is
                logging.info(f"No images found in subfolder: {sf}")
                # We'll store it anyway, but 'imagePaths' will be empty
            exifMap = {}
            imageStates = {}
            for i, path in enumerate(imagePaths):
                exif_bytes = self.getEXIFBytes(path)
                exifMap[i] = exif_bytes
                imageStates[i] = {
                    "state": "none",
                    "bboxCoords": None,
                    "maskNodes": None
                }

            # Add the new set to self.setStates
            self.setStates[sf] = {
                "imagePaths": imagePaths,
                "imageStates": imageStates,
                "exifData": exifMap
            }

            # Check any pre-existing masks for *this* set
            # We temporarily set 'currentSet' so checkPreExistingMasks()
            # knows where to look on disk.
            self.currentSet = sf
            self.imagePaths = imagePaths
            self.imageStates = imageStates
            self.checkPreExistingMasks()

            # At this point, self.imageStates is updated, so we can compute isSetFullyMasked(sf).
            iconToUse = self.iconGreen if self.isSetFullyMasked(sf) else self.iconRed
            self.imageSetComboBox.addItem(iconToUse, sf)

            self.processFoldersProgressBar.setValue(idx + 1)
            slicer.app.processEvents()

        self.processFoldersProgressBar.setVisible(False)

        # After loading all sets, select the *first* set in the combo (if any exist)
        if self.imageSetComboBox.count > 0:
            self.imageSetComboBox.setCurrentIndex(0)
            self.currentSet = self.imageSetComboBox.currentText
            # Because we've already loaded it, simply restoreSetState to update the UI
            self.restoreSetState(self.currentSet)
        else:
            # If no subfolders found or no images found, just clear
            self.currentSet = None
            self.imagePaths = []
            self.imageStates = {}

        # Finally, update the global masked counter (all sets are loaded in self.setStates now)
        self.updateMaskedCounter()

        # Enable the combo box, so user can switch sets if they want
        self.imageSetComboBox.enabled = True

    def anySetHasProgress(self):
        for _, setData in self.setStates.items():
            for _, stInfo in setData["imageStates"].items():
                if stInfo["state"] in ["bbox", "masked"]:
                    return True
        return False

    def clearAllData(self):
        self.setStates = {}
        self.currentSet = None
        self.imagePaths = []
        self.currentImageIndex = 0
        self.imageSetComboBox.clear()
        self.imageSetComboBox.enabled = False

        self.placeBoundingBoxButton.enabled = False
        self.maskCurrentImageButton.enabled = False
        self.maskAllImagesButton.enabled = False
        self.prevButton.enabled = False
        self.nextButton.enabled = False
        self.imageIndexLabel.setText("Image 1")

        self.launchWebODMTaskButton.setEnabled(False)
        self.maskedCountLabel.setText("0/0 Masked")

        self.imageCache.clear()
        self.resetMasterVolumes()

    def resetMasterVolumes(self):
        import numpy as np
        emptyArr = np.zeros((1, 1), dtype=np.uint8)
        slicer.util.updateVolumeFromArray(self.masterVolumeNode, emptyArr)
        slicer.util.updateVolumeFromArray(self.masterLabelMapNode, emptyArr)
        slicer.util.updateVolumeFromArray(self.masterMaskedVolumeNode, emptyArr)

    def onImageSetSelected(self, index):
        if index < 0:
            return
        self.saveCurrentSetState()

        self.currentSet = self.imageSetComboBox.currentText
        if self.currentSet in self.setStates:
            self.restoreSetState(self.currentSet)
        else:
            masterFolderPath = self.masterFolderSelector.directory
            setFolderPath = os.path.join(masterFolderPath, self.currentSet)
            self.imagePaths = self.logic.get_image_paths_from_folder(setFolderPath)
            if len(self.imagePaths) == 0:
                slicer.util.warningDisplay("No images found in this set.")
                return

            self.imageCache.clear()

            self.imageStates = {}
            exifMap = {}
            for i, path in enumerate(self.imagePaths):
                exif_bytes = self.getEXIFBytes(path)
                exifMap[i] = exif_bytes
                self.imageStates[i] = {
                    "state": "none",
                    "bboxCoords": None,
                    "maskNodes": None
                }

            self.currentImageIndex = 0
            self.setStates[self.currentSet] = {
                "imagePaths": self.imagePaths,
                "imageStates": self.imageStates,
                "exifData": exifMap
            }
            self.checkPreExistingMasks()

            self.updateVolumeDisplay()
            self.placeBoundingBoxButton.enabled = True
            self.maskCurrentImageButton.enabled = False
            self.maskAllImagesButton.enabled = False
            self.prevButton.enabled = (len(self.imagePaths) > 1)
            self.nextButton.enabled = (len(self.imagePaths) > 1)

        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()
        self.enableMaskAllImagesIfPossible()

    def getEXIFBytes(self, path):
        from PIL import Image
        try:
            im = Image.open(path)
            return im.info.get("exif", b"")
        except:
            return b""

    def checkPreExistingMasks(self):
        import cv2
        import numpy as np

        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        if not os.path.isdir(setOutputFolder):
            return

        for i, imgPath in enumerate(self.imagePaths):
            stInfo = self.imageStates[i]
            if stInfo["state"] == "masked":
                continue
            baseName = os.path.splitext(os.path.basename(imgPath))[0]
            maskFile = os.path.join(setOutputFolder, f"{baseName}_mask.jpg")
            if os.path.isfile(maskFile):
                maskBGR = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
                if maskBGR is not None and np.any(maskBGR > 127):
                    stInfo["state"] = "masked"
                    yInds, xInds = np.where(maskBGR > 127)
                    if len(xInds) > 0 and len(yInds) > 0:
                        x_min, x_max = xInds.min(), xInds.max()
                        y_min, y_max = yInds.min(), yInds.max()
                        stInfo["bboxCoords"] = (x_min, y_min, x_max, y_max)
                    stInfo["maskNodes"] = None

    def saveCurrentSetState(self):
        if self.currentSet is None or self.currentSet not in self.setStates:
            return
        self.setStates[self.currentSet]["imageStates"] = self.imageStates

    def restoreSetState(self, setName):
        setData = self.setStates[setName]
        self.imagePaths = setData["imagePaths"]
        self.imageStates = setData["imageStates"]
        self.currentImageIndex = 0

        self.imageCache.clear()
        self.updateVolumeDisplay()
        self.placeBoundingBoxButton.enabled = True
        self.refreshButtonStatesBasedOnCurrentState()
        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()
        self.updateImageTable()

        if len(self.imagePaths) > 1:
            self.prevButton.enabled = True
            self.nextButton.enabled = True
        else:
            self.prevButton.enabled = False
            self.nextButton.enabled = False

    def updateVolumeDisplay(self):
        self.imageIndexLabel.setText(f"Image {self.currentImageIndex + 1}")
        if self.currentImageIndex < 0 or self.currentImageIndex >= len(self.imagePaths):
            return

        st = self.imageStates[self.currentImageIndex]["state"]
        self.removeBboxLines()

        colorArrDown = self.getDownsampledColor(self.currentSet, self.currentImageIndex)
        colorArrDownRGBA = colorArrDown[np.newaxis, ...]
        slicer.util.updateVolumeFromArray(self.masterVolumeNode, colorArrDownRGBA)

        if st == "none":
            self.showOriginalOnly()
            self.maskCurrentImageButton.enabled = False
        elif st == "bbox":
            self.showOriginalOnly()
            self.drawBboxLines(self.imageStates[self.currentImageIndex]["bboxCoords"])
            self.maskCurrentImageButton.enabled = True
        elif st == "masked":
            self.showMaskedState(colorArrDown, self.currentImageIndex)
            self.maskCurrentImageButton.enabled = False

        self.enableMaskAllImagesIfPossible()

        lm = slicer.app.layoutManager()
        lm.sliceWidget('Red').sliceLogic().FitSliceToAll()
        lm.sliceWidget('Red2').sliceLogic().FitSliceToAll()

        # If state == "bbox", enable the new inclusion/exclusion points
        self.updatePointButtons()

    def getDownsampledColor(self, setName, index):
        downKey = (setName, index, 'down')
        if downKey in self.imageCache:
            return self.imageCache[downKey]

        fullColor = self.getFullColorArray(setName, index)
        import cv2
        h, w, _ = fullColor.shape
        newW = int(w * self.downsampleFactor)
        newH = int(h * self.downsampleFactor)
        colorDown = cv2.resize(fullColor, (newW, newH), interpolation=cv2.INTER_AREA)

        if len(self.imageCache) >= self.maxCacheSize:
            self.evictOneFromCache()
        self.imageCache[downKey] = colorDown
        return colorDown

    def getFullColorArray(self, setName, index):
        fullKey = (setName, index, 'full')
        if fullKey in self.imageCache:
            return self.imageCache[fullKey]

        path = self.setStates[setName]["imagePaths"][index]
        from PIL import Image
        im = Image.open(path).convert('RGB')
        colorArr = np.array(im)

        # Flip up-down and left-right
        colorArr = np.flipud(colorArr)
        colorArr = np.fliplr(colorArr)

        if len(self.imageCache) >= self.maxCacheSize:
            self.evictOneFromCache()
        self.imageCache[fullKey] = colorArr
        return colorArr

    def evictOneFromCache(self):
        k = next(iter(self.imageCache.keys()))
        del self.imageCache[k]

    def refreshButtonStatesBasedOnCurrentState(self):
        st = self.imageStates[self.currentImageIndex]["state"]
        if st == "none":
            self.maskCurrentImageButton.enabled = False
        elif st == "bbox":
            self.maskCurrentImageButton.enabled = True
        elif st == "masked":
            self.maskCurrentImageButton.enabled = False
        self.enableMaskAllImagesIfPossible()

    def enableMaskAllImagesIfPossible(self):
        if self.currentSet:
            self.maskAllImagesButton.enabled = True

    def showOriginalOnly(self):
        lm = slicer.app.layoutManager()
        redComp = lm.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComp.SetLabelVolumeID(self.emptyNode.GetID())

        red2Comp = lm.sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Comp.SetBackgroundVolumeID(self.emptyNode.GetID())

    def showMaskedState(self, colorArrDown, imageIndex):
        stInfo = self.imageStates[imageIndex]
        if stInfo["state"] != "masked":
            return
        maskDown = self.getMaskFromCacheOrDisk(self.currentSet, imageIndex, downsample=True)
        labelDown = (maskDown > 127).astype(np.uint8)

        slicer.util.updateVolumeFromArray(self.masterLabelMapNode, labelDown)

        cpy = colorArrDown.copy()
        cpy[labelDown == 0] = 0
        cpyRGBA = cpy[np.newaxis, ...]
        slicer.util.updateVolumeFromArray(self.masterMaskedVolumeNode, cpyRGBA)

        lm = slicer.app.layoutManager()
        redComp = lm.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComp.SetLabelVolumeID(self.masterLabelMapNode.GetID())
        redComp.SetLabelOpacity(0.5)

        red2Comp = lm.sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Comp.SetBackgroundVolumeID(self.masterMaskedVolumeNode.GetID())

    def getMaskFromCacheOrDisk(self, setName, index, downsample=False):
        if downsample:
            key = (setName, index, 'mask-down')
            if key in self.imageCache:
                return self.imageCache[key]
        else:
            key = (setName, index, 'mask')
            if key in self.imageCache:
                return self.imageCache[key]

        path = self.setStates[setName]["imagePaths"][index]
        baseName = os.path.splitext(os.path.basename(path))[0]
        maskPath = os.path.join(self.outputFolderSelector.directory, setName, f"{baseName}_mask.jpg")

        from PIL import Image
        maskPil = Image.open(maskPath).convert("L")
        maskArr = np.array(maskPil)
        maskArr = np.flipud(maskArr)
        maskArr = np.fliplr(maskArr)

        if downsample:
            colorDown = self.getDownsampledColor(setName, index)
            hD, wD, _ = colorDown.shape
            import cv2
            maskDown = cv2.resize(maskArr, (wD, hD), interpolation=cv2.INTER_NEAREST)
            if len(self.imageCache) >= self.maxCacheSize:
                self.evictOneFromCache()
            self.imageCache[(setName, index, 'mask-down')] = maskDown
            return maskDown
        else:
            if len(self.imageCache) >= self.maxCacheSize:
                self.evictOneFromCache()
            self.imageCache[(setName, index, 'mask')] = maskArr
            return maskArr

    def onPrevImage(self):
        """
        Move to the previous image in the current set.
        In circular (carousel) mode, if we are on the first image (index 0)
        and the user clicks "prev", we jump to the last image in the set.
        """

        # --- No change needed here. Keep existing checks for no images. ---
        if not self.imagePaths:
            return

        # --- CHANGE START: Carousel logic for prev ---
        if self.currentImageIndex <= 0:
            # We are at the first image (index = 0), so wrap to the last
            self.currentImageIndex = len(self.imagePaths) - 1
        else:
            # Normal "move one back"
            self.currentImageIndex -= 1
        # --- CHANGE END ---

        self.updateVolumeDisplay()

        # The rest is unchanged
        if self.finalizingROI:
            self.maskAllImagesButton.enabled = False
        else:
            self.maskAllImagesButton.enabled = True

    def onNextImage(self):
        """
        Move to the next image in the current set.
        In circular (carousel) mode, if we are on the last image,
        going forward jumps to the first image in the set.
        """

        # --- No change needed here. Keep existing checks for no images. ---
        if not self.imagePaths:
            return

        # --- CHANGE START: Carousel logic for next ---
        if self.currentImageIndex >= len(self.imagePaths) - 1:
            # We are at the last image, wrap to the first
            self.currentImageIndex = 0
        else:
            # Normal "move one forward"
            self.currentImageIndex += 1
        # --- CHANGE END ---

        self.updateVolumeDisplay()

        # The rest is unchanged
        if self.finalizingROI:
            self.maskAllImagesButton.enabled = False
        else:
            self.maskAllImagesButton.enabled = True

    def onMaskCurrentImageClicked(self):
        """
        Merge bounding-box finalization + SAM + negative (exclusion) + positive (inclusion) points if any.
        """
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo:
            slicer.util.warningDisplay("No image state found. Please select a valid image.")
            return

        # If ROI not yet finalized, do so
        if self.boundingBoxRoiNode:
            self.finalizeBoundingBoxAndRemoveROI()

        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo or stInfo["state"] != "bbox":
            slicer.util.warningDisplay("No bounding box defined or finalized for this image. Cannot mask.")
            return

        import numpy as np

        resFactor = self.getUserSelectedResolutionFactor()

        bboxDown = stInfo["bboxCoords"]
        bboxFull = self.downBboxToFullBbox(bboxDown, self.currentSet, self.currentImageIndex)
        colorArrFull = self.getFullColorArray(self.currentSet, self.currentImageIndex)

        # Gather negative points (exclusion)
        negPointsFull = []
        numNeg = self.exclusionPointNode.GetNumberOfControlPoints()
        for i in range(numNeg):
            ras = [0, 0, 0]
            self.exclusionPointNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self.rasToDownsampleIJK(ras, self.masterVolumeNode)
            ptFull = self.downPointToFullPoint(ijk, self.currentSet, self.currentImageIndex)
            negPointsFull.append(ptFull)

        # Gather positive points (inclusion)
        posPointsFull = []
        numPos = self.inclusionPointNode.GetNumberOfControlPoints()
        for i in range(numPos):
            ras = [0, 0, 0]
            self.inclusionPointNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self.rasToDownsampleIJK(ras, self.masterVolumeNode)
            ptFull = self.downPointToFullPoint(ijk, self.currentSet, self.currentImageIndex)
            posPointsFull.append(ptFull)

        import cv2
        opencvFull = self.logic.pil_to_opencv(self.logic.array_to_pil(colorArrFull))
        marker_outputs = self.detect_aruco_bounding_boxes(opencvFull, aruco_dict=cv2.aruco.DICT_4X4_250)

        mask = self.logic.run_sam_segmentation_with_incl_excl(
            colorArrFull, bboxFull, posPointsFull, negPointsFull,
            marker_outputs
        )

        stInfo["state"] = "masked"
        maskBool = (mask > 0)

        # Save the newly computed mask
        self.saveMaskedImage(self.currentImageIndex, colorArrFull, maskBool)

        # Remove inclusion/exclusion points
        self.exclusionPointNode.RemoveAllControlPoints()
        self.inclusionPointNode.RemoveAllControlPoints()

        # Re-display
        self.updateVolumeDisplay()
        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()
        self.updateImageTable()

        # Restore normal button states
        self.restoreButtonStates()
        self.enableMaskAllImagesIfPossible()

        self.imageTable.setEnabled(True)

        # refresh icons for all sets
        self.updateSetComboBoxIcons()

    def saveMaskedImage(self, index, colorArrFull, maskBool):
        from PIL import Image
        setData = self.setStates[self.currentSet]
        exifMap = setData.get("exifData", {})
        exif_bytes = exifMap.get(index, b"")

        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        os.makedirs(setOutputFolder, exist_ok=True)

        cpy = colorArrFull.copy()
        cpy[~maskBool] = 0
        cpy = np.flipud(cpy)
        cpy = np.fliplr(cpy)

        baseName = os.path.splitext(os.path.basename(self.imagePaths[index]))[0]
        colorPngFilename = baseName + ".jpg"
        colorPngPath = os.path.join(setOutputFolder, colorPngFilename)

        colorPil = Image.fromarray(cpy.astype(np.uint8))
        if exif_bytes:
            colorPil.save(colorPngPath, "jpeg", quality=100, exif=exif_bytes)
        else:
            colorPil.save(colorPngPath, "jpeg", quality=100)

        maskBin = (maskBool.astype(np.uint8) * 255)
        maskBin = np.flipud(maskBin)
        maskBin = np.fliplr(maskBin)
        maskPil = Image.fromarray(maskBin, mode='L')
        maskFilename = f"{baseName}_mask.jpg"
        maskPath = os.path.join(setOutputFolder, maskFilename)
        maskPil.save(maskPath, "jpeg")

        maskCacheKey = (self.currentSet, index, 'mask')
        maskCacheKeyDown = (self.currentSet, index, 'mask-down')
        if maskCacheKey in self.imageCache:
            del self.imageCache[maskCacheKey]
        if maskCacheKeyDown in self.imageCache:
            del self.imageCache[maskCacheKeyDown]

    def onMaskAllImagesClicked(self):
        """
        Override old behavior. Removes existing masks, places a single ROI for the entire set, etc.
        """
        self.finalizingROI = True

        if not self.currentSet or self.currentSet not in self.setStates:
            slicer.util.warningDisplay("No current set is selected. Unable to proceed.")
            return

        confirm = slicer.util.confirmYesNoDisplay(
            "This will remove ALL existing masks (and bounding boxes) for the entire set.\n"
            "Continue?"
        )
        if not confirm:
            slicer.util.infoDisplay("Mask All Images canceled.")
            return

        self.removeAllMasksForCurrentSet()

        for idx in range(len(self.imagePaths)):
            self.imageStates[idx]["state"] = "none"
            self.imageStates[idx]["bboxCoords"] = None
            self.imageStates[idx]["maskNodes"] = None

        # self.currentImageIndex = 0
        self.updateVolumeDisplay()

        self.globalMaskAllInProgress = True
        self.startPlacingROIForAllImages()

        self.disableAllUIButNextPrevAndFinalize()

        slicer.util.infoDisplay(
            "You can now place/adjust a bounding box ROI. Then switch images with < or > to see how it fits.\n"
            "When ready, click 'Finalize ROI for All' to mask all images in the set.",
            autoCloseMsec=8000
        )

    def removeAllMasksForCurrentSet(self):
        """
        Removes any existing mask files on disk for the currently selected set.
        """
        if not self.currentSet:
            return

        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        if not os.path.isdir(setOutputFolder):
            return

        for fn in os.listdir(setOutputFolder):
            if fn.lower().endswith("_mask.jpg"):
                try:
                    os.remove(os.path.join(setOutputFolder, fn))
                except Exception as e:
                    logging.warning(f"Failed to remove mask file {fn}: {str(e)}")

    def startPlacingROIForAllImages(self):
        self.removeBboxLines()
        if self.boundingBoxRoiNode:
            slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode)
            self.boundingBoxRoiNode = None

        self.boundingBoxRoiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "BoundingBoxROI_ALL")
        self.boundingBoxRoiNode.CreateDefaultDisplayNodes()
        dnode = self.boundingBoxRoiNode.GetDisplayNode()
        dnode.SetVisibility(True)
        dnode.SetHandlesInteractive(True)

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsROINode")
        selectionNode.SetActivePlaceNodeID(self.boundingBoxRoiNode.GetID())
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self.boundingBoxRoiNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.checkROIPlacementComplete
        )

    def disableAllUIButNextPrevAndFinalize(self):
        self.storeCurrentButtonStates()
        for b in self.buttonsToManage:
            if b in [self.prevButton, self.nextButton]:
                b.enabled = True
            else:
                if isinstance(b, qt.QComboBox):
                    b.setEnabled(False)
                else:
                    b.enabled = False
        self.finalizeAllMaskButton.enabled = True

        self.addInclusionPointsButton.enabled = False
        self.addExclusionPointsButton.enabled = False
        self.clearPointsButton.enabled = False
        self.stopAddingPointsButton.enabled = False
        self.maskAllImagesButton.enabled = False

        self.imageTable.setEnabled(False)

    def onFinalizeAllMaskClicked(self):
        if not self.globalMaskAllInProgress:
            return

        if not self.boundingBoxRoiNode:
            slicer.util.warningDisplay("No ROI to finalize. Please place an ROI first.")
            return

        coordsDown = self.computeBboxFromROI()
        if not coordsDown:
            slicer.util.warningDisplay("Unable to compute bounding box from ROI.")
            return

        self.finalizeAllMaskButton.enabled = False

        self.removeRoiNode()

        negPointsFull = []
        numNeg = self.exclusionPointNode.GetNumberOfControlPoints()
        for i in range(numNeg):
            ras = [0, 0, 0]
            self.exclusionPointNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self.rasToDownsampleIJK(ras, self.masterVolumeNode)
            ptFull = self.downPointToFullPoint(ijk, self.currentSet, self.currentImageIndex)
            negPointsFull.append(ptFull)

        posPointsFull = []
        numPos = self.inclusionPointNode.GetNumberOfControlPoints()
        for i in range(numPos):
            ras = [0, 0, 0]
            self.inclusionPointNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self.rasToDownsampleIJK(ras, self.masterVolumeNode)
            ptFull = self.downPointToFullPoint(ijk, self.currentSet, self.currentImageIndex)
            posPointsFull.append(ptFull)

        n = len(self.imagePaths)

        self.maskAllProgressBar.setVisible(True)
        self.maskAllProgressBar.setTextVisible(True)
        self.maskAllProgressBar.setRange(0, n)
        self.maskAllProgressBar.setValue(0)

        import time
        start_time = time.time()

        resFactor = self.getUserSelectedResolutionFactor()

        for count, idx in enumerate(range(n)):
            self.maskSingleImage(
                idx,
                coordsDown,
                posPointsFull,
                negPointsFull,
                resFactor
            )
            processed = count + 1
            self.maskAllProgressBar.setValue(processed)

            elapsed_secs = time.time() - start_time
            avg = elapsed_secs / processed
            remain = avg * (n - processed)

            def fmt(sec):
                sec = int(sec)
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                if h > 0:
                    return f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    return f"{m:02d}:{s:02d}"

            el_str = fmt(elapsed_secs)
            rm_str = fmt(remain)
            msg = f"Masking image {processed}/{n} | Elapsed: {el_str}, Remain: {rm_str}"
            self.maskAllProgressBar.setFormat(msg)
            slicer.app.processEvents()

        end_time = time.time()
        logging.info(f"Global Set Masking execution time: {end_time - start_time:.6f} seconds")

        slicer.util.infoDisplay("All images in set have been masked successfully.")
        self.maskAllProgressBar.setVisible(False)
        self.updateVolumeDisplay()
        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()
        self.updateImageTable()

        self.restoreButtonStates()
        self.imageTable.setEnabled(True)

        self.finalizingROI = False
        self.globalMaskAllInProgress = False
        self.maskAllImagesButton.enabled = True

        # refresh icons for all sets
        self.updateSetComboBoxIcons()

    def removeRoiNode(self):
        if self.boundingBoxRoiNode:
            dnode = self.boundingBoxRoiNode.GetDisplayNode()
            if dnode:
                dnode.SetHandlesInteractive(False)
            slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode)
            self.boundingBoxRoiNode = None

    def maskSingleImage(self, index, bboxDown, posPointsFull, negPointsFull, resFactor=1.0):
        import numpy as np
        import cv2

        bboxFull = self.downBboxToFullBbox(bboxDown, self.currentSet, index)
        colorArrFull = self.getFullColorArray(self.currentSet, index)

        if abs(resFactor - 1.0) < 1e-5:
            opencvFull = self.logic.pil_to_opencv(self.logic.array_to_pil(colorArrFull))
            marker_outputs = self.detect_aruco_bounding_boxes(opencvFull, aruco_dict=cv2.aruco.DICT_4X4_250)

            mask = self.logic.run_sam_segmentation_with_incl_excl(
                colorArrFull, bboxFull, posPointsFull, negPointsFull,
                marker_outputs
            )
        else:
            H, W, _ = colorArrFull.shape
            newW = int(round(W * resFactor))
            newH = int(round(H * resFactor))

            colorDown = cv2.resize(colorArrFull, (newW, newH), interpolation=cv2.INTER_AREA)

            xMinF, yMinF, xMaxF, yMaxF = bboxFull
            xMinDown = int(round(xMinF * resFactor))
            xMaxDown = int(round(xMaxF * resFactor))
            yMinDown = int(round(yMinF * resFactor))
            yMaxDown = int(round(yMaxF * resFactor))

            posDown = [
                [int(round(px * resFactor)), int(round(py * resFactor))]
                for (px, py) in posPointsFull
            ]
            negDown = [
                [int(round(px * resFactor)), int(round(py * resFactor))]
                for (px, py) in negPointsFull
            ]

            opencvDown = self.logic.pil_to_opencv(self.logic.array_to_pil(colorDown))
            marker_outputs = self.detect_aruco_bounding_boxes(opencvDown, aruco_dict=cv2.aruco.DICT_4X4_250)

            maskDown = self.logic.run_sam_segmentation_with_incl_excl(
                colorDown,
                [xMinDown, yMinDown, xMaxDown, yMaxDown],
                posDown,
                negDown,
                marker_outputs
            )

            mask = cv2.resize(maskDown, (W, H), interpolation=cv2.INTER_NEAREST)

        self.imageStates[index]["state"] = "masked"
        self.imageStates[index]["bboxCoords"] = bboxDown
        self.imageStates[index]["maskNodes"] = None

        maskBool = (mask > 0)
        self.saveMaskedImage(index, colorArrFull, maskBool)

    def onPlaceBoundingBoxClicked(self):
        self.storeCurrentButtonStates()
        if self.globalMaskAllInProgress:
            slicer.util.warningDisplay("You are in 'Mask All Images' mode. Please finalize or cancel that first.")
            self.restoreButtonStates()
            return

        self.disableAllButtonsExceptMask()

        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo:
            return
        s = stInfo["state"]

        if s == "masked":
            if slicer.util.confirmYesNoDisplay(
                    "This image is already masked. Creating a new bounding box will remove the existing mask. Proceed?"
            ):
                self.removeMaskFromCurrentImage()
                self.startPlacingROI()
            else:
                self.restoreButtonStates()
        elif s == "bbox":
            if slicer.util.confirmYesNoDisplay(
                    "A bounding box already exists. Creating a new one will remove it. Proceed?"
            ):
                self.removeBboxFromCurrentImage()
                self.startPlacingROI()
            else:
                self.restoreButtonStates()
        elif s == "none":
            self.startPlacingROI()

    def storeCurrentButtonStates(self):
        self.previousButtonStates = {}
        for b in self.buttonsToManage:
            if isinstance(b, qt.QComboBox):
                self.previousButtonStates[b] = b.isEnabled()
            else:
                self.previousButtonStates[b] = b.enabled

    def restoreButtonStates(self):
        for b, val in self.previousButtonStates.items():
            if isinstance(b, qt.QComboBox):
                b.setEnabled(val)
            else:
                b.enabled = val

    def disableAllButtonsExceptMask(self):
        for b in self.buttonsToManage:
            if b != self.maskCurrentImageButton:
                if isinstance(b, qt.QComboBox):
                    b.setEnabled(False)
                else:
                    b.enabled = False

        self.imageTable.setEnabled(False)

    def startPlacingROI(self):
        self.removeBboxLines()
        if self.boundingBoxRoiNode:
            slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode)
            self.boundingBoxRoiNode = None

        self.boundingBoxRoiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "BoundingBoxROI")
        self.boundingBoxRoiNode.CreateDefaultDisplayNodes()
        dnode = self.boundingBoxRoiNode.GetDisplayNode()
        dnode.SetVisibility(True)
        dnode.SetHandlesInteractive(True)

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsROINode")
        selectionNode.SetActivePlaceNodeID(self.boundingBoxRoiNode.GetID())
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self.boundingBoxRoiNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.checkROIPlacementComplete
        )

        slicer.util.infoDisplay(
            "Draw the ROI and use the handles to adjust it. When done, click 'Mask Current Image' to finalize.",
            autoCloseMsec=7000
        )

        self.addInclusionPointsButton.enabled = True
        self.addExclusionPointsButton.enabled = True
        self.clearPointsButton.enabled = True
        self.maskCurrentImageButton.enabled = True

    def checkROIPlacementComplete(self, caller, event):
        if not self.boundingBoxRoiNode:
            return
        if self.boundingBoxRoiNode.GetControlPointPlacementComplete():
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetPlaceModePersistence(0)
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            slicer.util.infoDisplay(
                "ROI placement complete. You can still edit the ROI or directly click 'Mask Current Image' to finalize.",
                autoCloseMsec=5000
            )

    def removeMaskFromCurrentImage(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo:
            return
        if stInfo["maskNodes"]:
            stInfo["maskNodes"] = None

        self.deleteMaskFile(self.currentImageIndex, self.currentSet)
        stInfo["state"] = "none"
        stInfo["bboxCoords"] = None

        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()
        self.updateImageTable()
        self.updateVolumeDisplay()

    def deleteMaskFile(self, index, setName):
        outputFolder = self.outputFolderSelector.directory
        baseName = os.path.splitext(os.path.basename(self.imagePaths[index]))[0]
        maskPath = os.path.join(outputFolder, setName, f"{baseName}_mask.jpg")
        if os.path.isfile(maskPath):
            try:
                os.remove(maskPath)
            except Exception as e:
                print(f"Warning: failed to remove mask file: {maskPath}, error: {str(e)}")

    def removeBboxFromCurrentImage(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if stInfo:
            stInfo["state"] = "none"
            stInfo["bboxCoords"] = None
            stInfo["maskNodes"] = None
        self.removeBboxLines()

    def finalizeBoundingBoxAndRemoveROI(self):
        if not self.boundingBoxRoiNode:
            return

        coordsDown = self.computeBboxFromROI()
        stInfo = self.imageStates[self.currentImageIndex]
        stInfo["state"] = "bbox"
        stInfo["bboxCoords"] = coordsDown
        stInfo["maskNodes"] = None

        dnode = self.boundingBoxRoiNode.GetDisplayNode()
        if dnode:
            dnode.SetHandlesInteractive(False)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        if self.boundingBoxRoiNode.GetDisplayNode():
            slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode.GetDisplayNode())
        slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode)
        self.boundingBoxRoiNode = None

    def computeBboxFromROI(self):
        roiBounds = [0] * 6
        self.boundingBoxRoiNode.GetBounds(roiBounds)
        p1 = [roiBounds[0], roiBounds[2], roiBounds[4]]
        p2 = [roiBounds[1], roiBounds[3], roiBounds[4]]

        rasToIjkMat = vtk.vtkMatrix4x4()
        self.masterVolumeNode.GetRASToIJKMatrix(rasToIjkMat)

        def rasToIjk(ras):
            ras4 = [ras[0], ras[1], ras[2], 1.0]
            ijk4 = rasToIjkMat.MultiplyPoint(ras4)
            return [int(round(ijk4[0])), int(round(ijk4[1])), int(round(ijk4[2]))]

        p1_ijk = rasToIjk(p1)
        p2_ijk = rasToIjk(p2)
        x_min = min(p1_ijk[0], p2_ijk[0])
        x_max = max(p1_ijk[0], p2_ijk[0])
        y_min = min(p1_ijk[1], p2_ijk[1])
        y_max = max(p1_ijk[1], p2_ijk[1])
        return (x_min, y_min, x_max, y_max)

    def downBboxToFullBbox(self, bboxDown, setName, index):
        x_minD, y_minD, x_maxD, y_maxD = bboxDown

        fullArr = self.getFullColorArray(setName, index)
        downArr = self.getDownsampledColor(setName, index)

        fullH, fullW, _ = fullArr.shape
        downH, downW, _ = downArr.shape

        scaleX = fullW / downW
        scaleY = fullH / downH

        x_minF = int(round(x_minD * scaleX))
        x_maxF = int(round(x_maxD * scaleX))
        y_minF = int(round(y_minD * scaleY))
        y_maxF = int(round(y_maxD * scaleY))
        return (x_minF, y_minF, x_maxF, y_maxF)

    def rasToDownsampleIJK(self, ras, volumeNode):
        rasToIjkMat = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjkMat)
        ras4 = [ras[0], ras[1], ras[2], 1.0]
        ijk4 = rasToIjkMat.MultiplyPoint(ras4)
        i, j = int(round(ijk4[0])), int(round(ijk4[1]))
        return [i, j]

    def downPointToFullPoint(self, ijkDown, setName, index):
        fullArr = self.getFullColorArray(setName, index)
        downArr = self.getDownsampledColor(setName, index)
        fullH, fullW, _ = fullArr.shape
        downH, downW, _ = downArr.shape
        scaleX = fullW / downW
        scaleY = fullH / downH
        xF = int(round(ijkDown[0] * scaleX))
        yF = int(round(ijkDown[1] * scaleY))
        return [xF, yF]

    def removeBboxLines(self):
        for ln in self.currentBboxLineNodes:
            if ln and slicer.mrmlScene.IsNodePresent(ln):
                slicer.mrmlScene.RemoveNode(ln)
        self.currentBboxLineNodes = []

    def drawBboxLines(self, coordsDown):
        if not coordsDown:
            return
        ijkToRasMat = vtk.vtkMatrix4x4()
        self.masterVolumeNode.GetRASToIJKMatrix(ijkToRasMat)

        def ijkToRas(i, j):
            p = [i, j, 0, 1]
            invMat = vtk.vtkMatrix4x4()
            invMat.DeepCopy(ijkToRasMat)
            invMat.Invert()
            ras = invMat.MultiplyPoint(p)
            return [ras[0], ras[1], ras[2]]

        x_min, y_min, x_max, y_max = coordsDown
        p1 = ijkToRas(x_min, y_min)
        p2 = ijkToRas(x_max, y_min)
        p3 = ijkToRas(x_max, y_max)
        p4 = ijkToRas(x_min, y_max)
        lines = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]

        for (start, end) in lines:
            ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            ln.AddControlPoint(start)
            ln.AddControlPoint(end)
            m = ln.GetMeasurement('length')
            if m:
                m.SetEnabled(False)
            dnode = ln.GetDisplayNode()
            dnode.SetLineThickness(0.25)
            dnode.SetSelectedColor(1, 1, 0)
            dnode.SetPointLabelsVisibility(False)
            dnode.SetPropertiesLabelVisibility(False)
            dnode.SetTextScale(0)
            self.currentBboxLineNodes.append(ln)

    def detect_aruco_bounding_boxes(self, cv_img, aruco_dict=None):
        import cv2
        if not aruco_dict:
            aruco_dict = cv2.aruco.DICT_4X4_50

        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_img, dictionary)
        bounding_boxes = []
        if ids is not None:
            for i in range(len(ids)):
                pts = corners[i][0]
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                y_min, y_max = int(y_coords.min()), int(y_coords.max())
                bounding_boxes.append({
                    "marker_id": int(ids[i]),
                    "bbox": (x_min, y_min, x_max, y_max)
                })
        return bounding_boxes

    def assemble_bboxes(self, initial_box_np, marker_outputs, pad=25):
        import numpy as np
        combined_boxes = [initial_box_np]
        for marker_dict in marker_outputs:
            x_min, y_min, x_max, y_max = marker_dict["bbox"]
            x_min_new = x_min - pad
            y_min_new = y_min - pad
            x_max_new = x_max + pad
            y_max_new = y_max + pad
            combined_boxes.append(np.array([x_min_new, y_min_new, x_max_new, y_max_new]))
        return combined_boxes

    def segment_with_boxes(self, image_rgb, boxes, predictor):
        import torch
        import numpy as np
        predictor.set_image(image_rgb)
        h, w, _ = image_rgb.shape
        combined_mask = np.zeros((h, w), dtype=bool)
        for box in boxes:
            with torch.no_grad():
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False
                )
            mask_bool = masks[0].astype(bool)
            combined_mask = np.logical_or(combined_mask, mask_bool)
        return combined_mask

    def onGenerateGCPClicked(self):
        import subprocess
        find_gcp_script = self.findGCPScriptSelector.currentPath
        if not find_gcp_script or not os.path.isfile(find_gcp_script):
            slicer.util.errorDisplay("Please select a valid Find-GCP.py script path.")
            return

        self.gcpCoordFilePath = self.gcpCoordFileSelector.currentPath
        if not self.gcpCoordFilePath or not os.path.isfile(self.gcpCoordFilePath):
            slicer.util.errorDisplay("Please select a valid GCP coordinate file (required).")
            return

        outputFolder = self.outputFolderSelector.directory
        if not outputFolder or not os.path.isdir(outputFolder):
            slicer.util.errorDisplay("Please select a valid output folder.")
            return

        combinedOutputFile = os.path.join(outputFolder, "combined_gcp_list.txt")

        masterFolderPath = self.masterFolderSelector.directory
        if not masterFolderPath or not os.path.isdir(masterFolderPath):
            slicer.util.errorDisplay("Please select a valid master folder.")
            return

        subfolders = [f for f in os.listdir(masterFolderPath)
                      if os.path.isdir(os.path.join(masterFolderPath, f))]
        allImages = []
        for sf in subfolders:
            subFolderPath = os.path.join(masterFolderPath, sf)
            imgs = self.logic.get_image_paths_from_folder(subFolderPath)
            allImages.extend(imgs)

        if len(allImages) == 0:
            slicer.util.warningDisplay("No images found in any subfolder. Nothing to do.")
            return

        dict_id = self.arucoDictIDSpinBox.value
        cmd = [
            sys.executable,
            find_gcp_script,
            "-t", "ODM",
            "-d", str(dict_id),
            "-i", self.gcpCoordFilePath,
            "--epsg", "3857",
            "-o", combinedOutputFile
        ]
        cmd += allImages

        try:
            slicer.util.infoDisplay("Running Find-GCP to produce a combined gcp_list.txt...")
            subprocess.run(cmd, check=True)

            if os.path.isfile(combinedOutputFile):
                with open(combinedOutputFile, "r") as f:
                    self.gcpListContent = f.read()

                slicer.util.infoDisplay(
                    f"Combined GCP list created successfully at:\n{combinedOutputFile}",
                    autoCloseMsec=3500
                )
            else:
                slicer.util.warningDisplay(f"Find-GCP did not produce the file:\n{combinedOutputFile}")

        except subprocess.CalledProcessError as e:
            slicer.util.warningDisplay(f"Find-GCP failed (CalledProcessError): {str(e)}")
        except Exception as e:
            slicer.util.warningDisplay(f"An error occurred running Find-GCP: {str(e)}")

    def generateShortTaskName(self, basePrefix, paramsDict):
        paramItems = sorted(paramsDict.items())
        paramString = ";".join(f"{k}={v}" for k, v in paramItems)
        md5Hash = hashlib.md5(paramString.encode('utf-8')).hexdigest()
        shortHash = md5Hash[:8]
        shortName = f"{basePrefix}_{shortHash}"
        return shortName

    def onRunWebODMTask(self):
        if not self.allSetsHavePhysicalMasks():
            slicer.util.warningDisplay("Not all images have masks. Please mask all sets first.")
            return
        self.webODMManager.onRunWebODMTask()

    def allSetsHavePhysicalMasks(self):
        if not self.setStates:
            return False
        outputRoot = self.outputFolderSelector.directory
        if not os.path.isdir(outputRoot):
            return False
        for setName, setData in self.setStates.items():
            setOutputFolder = os.path.join(outputRoot, setName)
            if not os.path.isdir(setOutputFolder):
                return False
            for imagePath in setData["imagePaths"]:
                baseName = os.path.splitext(os.path.basename(imagePath))[0]
                maskFile = os.path.join(setOutputFolder, f"{baseName}_mask.jpg")
                if not os.path.isfile(maskFile):
                    return False
        return True

    def updateWebODMTaskAvailability(self):
        allSetsMasked = self.allSetsHavePhysicalMasks()
        self.launchWebODMTaskButton.setEnabled(allSetsMasked)

    def onCloneFindGCPClicked(self):
        import os
        import shutil
        import slicer
        from slicer.util import downloadFile, extractArchive

        # -- Paths we will use:
        modulePath = os.path.dirname(slicer.modules.photogrammetry.path)
        resourcesFolder = os.path.join(modulePath, "Resources")
        os.makedirs(resourcesFolder, exist_ok=True)

        # 1) Where to save the downloaded .zip
        zipFilePath = os.path.join(resourcesFolder, "Find-GCP.zip")
        # 2) The name of the folder that GitHub's master.zip will produce
        extractedFolderName = "Find-GCP-master"
        # 3) Full path to that folder after extraction
        cloneFolder = os.path.join(resourcesFolder, extractedFolderName)
        # 4) The script we expect inside that extracted folder
        localGCPFindScript = os.path.join(cloneFolder, "gcp_find.py")

        # Direct download link for the .zip (refs/heads/master)
        url = "https://github.com/zsiki/Find-GCP/archive/refs/heads/master.zip"

        # -- Download the ZIP
        slicer.util.infoDisplay(
            f"Downloading Find-GCP ZIP from:\n{url}\nPlease wait...",
            autoCloseMsec=2000
        )
        try:
            # Overwrite=False by default, but you can specify overwrite=True if you prefer
            downloadFile(url, zipFilePath)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to download Find-GCP zip:\n{str(e)}")
            return

        # -- Check if the folder is already there
        if os.path.isdir(cloneFolder):
            msg = (
                f"The '{extractedFolderName}' folder already exists in the Resources directory.\n"
                "Would you like to delete it and extract again (overwrite)?"
            )
            if not slicer.util.confirmYesNoDisplay(msg):
                # They said "No"; leave existing folder as-is
                slicer.util.infoDisplay("Using existing folder; no changes made.")
                if os.path.isfile(localGCPFindScript):
                    self.findGCPScriptSelector.setCurrentPath(localGCPFindScript)
                    slicer.app.settings().setValue("Photogrammetry/findGCPScriptPath", localGCPFindScript)
                else:
                    slicer.util.warningDisplay(
                        f"Existing folder found, but {localGCPFindScript} does not exist.\n"
                        "Please pick the correct script manually."
                    )
                return
            else:
                try:
                    shutil.rmtree(cloneFolder)
                except Exception as e:
                    slicer.util.errorDisplay(
                        f"Failed to remove existing folder:\n{cloneFolder}\nError: {str(e)}"
                    )
                    return

        # -- Extract the ZIP into Resources/ (NOT into cloneFolder!)
        slicer.util.infoDisplay(
            f"Extracting the Find-GCP ZIP into:\n{resourcesFolder}\nPlease wait...",
            autoCloseMsec=3000
        )
        try:
            extractArchive(zipFilePath, resourcesFolder)
            slicer.util.infoDisplay("Unzipped Find-GCP successfully.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to unzip {zipFilePath}:\n{str(e)}")
            return

        # -- Confirm the script is present
        if not os.path.isfile(localGCPFindScript):
            slicer.util.warningDisplay(
                f"ZIP extracted, but {localGCPFindScript} was not found.\n"
                "Please check the extracted contents or specify the correct script."
            )
            return

        # -- Finally, update your module?s UI and settings
        self.findGCPScriptSelector.setCurrentPath(localGCPFindScript)
        slicer.app.settings().setValue("Photogrammetry/findGCPScriptPath", localGCPFindScript)

    def cleanup(self):
        self.saveCurrentSetState()
        if self.masterVolumeNode and slicer.mrmlScene.IsNodePresent(self.masterVolumeNode):
            slicer.mrmlScene.RemoveNode(self.masterVolumeNode)
        if self.masterLabelMapNode and slicer.mrmlScene.IsNodePresent(self.masterLabelMapNode):
            slicer.mrmlScene.RemoveNode(self.masterLabelMapNode)
        if self.masterMaskedVolumeNode and slicer.mrmlScene.IsNodePresent(self.masterMaskedVolumeNode):
            slicer.mrmlScene.RemoveNode(self.masterMaskedVolumeNode)
        if self.emptyNode and slicer.mrmlScene.IsNodePresent(self.emptyNode):
            slicer.mrmlScene.RemoveNode(self.emptyNode)

        self.masterVolumeNode = None
        self.masterLabelMapNode = None
        self.masterMaskedVolumeNode = None
        self.emptyNode = None

        if self.logger and self.vtkLogFilter:
            self.logger.removeFilter(self.vtkLogFilter)
            self.vtkLogFilter = None
            self.logger = None

    def initializeExclusionMarkupsNode(self):
        existingNode = slicer.mrmlScene.GetFirstNodeByName("ExclusionPoints")
        if existingNode and existingNode.IsA("vtkMRMLMarkupsFiducialNode"):
            self.exclusionPointNode = existingNode
        else:
            self.exclusionPointNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "ExclusionPoints"
            )
            self.exclusionPointNode.CreateDefaultDisplayNodes()

        if self.exclusionPointNode.GetDisplayNode():
            self.exclusionPointNode.GetDisplayNode().SetSelectedColor(1, 0, 0)  # red
            self.exclusionPointNode.GetDisplayNode().SetColor(1, 0, 0)

        self.exclusionPointNode.SetMaximumNumberOfControlPoints(-1)

        if not self.exclusionPointAddedObserverTag:
            self.exclusionPointAddedObserverTag = self.exclusionPointNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onExclusionPointAdded
            )

    def initializeInclusionMarkupsNode(self):
        existingNode = slicer.mrmlScene.GetFirstNodeByName("InclusionPoints")
        if existingNode and existingNode.IsA("vtkMRMLMarkupsFiducialNode"):
            self.inclusionPointNode = existingNode
        else:
            self.inclusionPointNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "InclusionPoints"
            )
            self.inclusionPointNode.CreateDefaultDisplayNodes()

        if self.inclusionPointNode.GetDisplayNode():
            self.inclusionPointNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            self.inclusionPointNode.GetDisplayNode().SetColor(0, 1, 0)

        self.inclusionPointNode.SetMaximumNumberOfControlPoints(-1)

        if not self.inclusionPointAddedObserverTag:
            self.inclusionPointAddedObserverTag = self.inclusionPointNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onInclusionPointAdded
            )

    def onExclusionPointAdded(self, caller, event):
        numPoints = caller.GetNumberOfControlPoints()
        logging.info(f"[ExclusionPoints Debug] A new point was added. Current total = {numPoints}.")

    def onInclusionPointAdded(self, caller, event):
        numPoints = caller.GetNumberOfControlPoints()
        logging.info(f"[InclusionPoints Debug] A new point was added. Current total = {numPoints}.")

    def onAddInclusionPointsClicked(self):
        logging.info("[InclusionPoints Debug] Entering multi-point place mode (inclusion).")
        self.stopAnyActivePlacement()

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.inclusionPointNode.GetID())

        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self.addExclusionPointsButton.enabled = False
        self.stopAddingPointsButton.enabled = True
        self.addInclusionPointsButton.enabled = False

    def onAddExclusionPointsClicked(self):
        logging.info("[ExclusionPoints Debug] Entering multi-point place mode (exclusion).")
        self.stopAnyActivePlacement()

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.exclusionPointNode.GetID())

        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self.addInclusionPointsButton.enabled = False
        self.stopAddingPointsButton.enabled = True
        self.addExclusionPointsButton.enabled = False

    def onStopAddingPointsClicked(self):
        logging.info("[Points Debug] Stopping any place mode for inclusion/exclusion points.")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        self.addInclusionPointsButton.enabled = True
        self.addExclusionPointsButton.enabled = True
        self.stopAddingPointsButton.enabled = False

    def onClearPointsClicked(self):
        if not self.exclusionPointNode and not self.inclusionPointNode:
            return

        msgBox = qt.QMessageBox()
        msgBox.setWindowTitle("Clear Points")
        msgBox.setText("Choose which points you wish to clear:")
        clearExclButton = msgBox.addButton("Exclusion Only", qt.QMessageBox.ActionRole)
        clearInclButton = msgBox.addButton("Inclusion Only", qt.QMessageBox.ActionRole)
        clearBothButton = msgBox.addButton("Both", qt.QMessageBox.ActionRole)
        cancelButton = msgBox.addButton("Cancel", qt.QMessageBox.RejectRole)

        msgBox.exec_()

        clickedButton = msgBox.clickedButton()
        if clickedButton == cancelButton:
            logging.info("Clear points canceled by user.")
            return
        elif clickedButton == clearExclButton:
            self.exclusionPointNode.RemoveAllControlPoints()
            logging.info("Cleared all Exclusion points.")
        elif clickedButton == clearInclButton:
            self.inclusionPointNode.RemoveAllControlPoints()
            logging.info("Cleared all Inclusion points.")
        elif clickedButton == clearBothButton:
            self.exclusionPointNode.RemoveAllControlPoints()
            self.inclusionPointNode.RemoveAllControlPoints()
            logging.info("Cleared all Exclusion and Inclusion points.")

    def stopAnyActivePlacement(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode.GetCurrentInteractionMode() == interactionNode.Place:
            interactionNode.SetPlaceModePersistence(0)
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def updatePointButtons(self):
        st = self.imageStates[self.currentImageIndex]["state"]
        if st == "bbox":
            self.addInclusionPointsButton.enabled = True
            self.addExclusionPointsButton.enabled = True
            self.clearPointsButton.enabled = True
        else:
            self.addInclusionPointsButton.enabled = False
            self.addExclusionPointsButton.enabled = False
            self.clearPointsButton.enabled = False
            self.stopAddingPointsButton.enabled = False

    def getUserSelectedResolutionFactor(self):
        if self.radioHalf.isChecked():
            return 0.5
        elif self.radioQuarter.isChecked():
            return 0.25
        else:
            return 1.0

    # --------------------------------------------------------------------------------
    # NEW >> LAUNCH WEBODM (Combined approach)
    # --------------------------------------------------------------------------------
    def onLaunchWebODMClicked(self):
        proceed = slicer.util.confirmYesNoDisplay(
            "This action will ensure nodeodm:gpu is installed (pull if needed), "
            "stop any running container on port 3002, and launch a new one.\n\n"
            "Proceed?"
        )
        if not proceed:
            slicer.util.infoDisplay("Launch NodeODM canceled by user.")
            return

        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except Exception as e:
            slicer.util.warningDisplay(f"Docker not found or not in PATH.\nError: {str(e)}")
            return

        try:
            check_process = subprocess.run(
                ["docker", "images", "-q", "opendronemap/nodeodm:gpu"],
                capture_output=True,
                text=True,
                check=True
            )
            image_id = check_process.stdout.strip()
            if not image_id:
                slicer.util.infoDisplay("nodeodm:gpu not found, pulling latest (this may take a while).")
                pull_process = subprocess.run(
                    ["docker", "pull", "opendronemap/nodeodm:gpu"],
                    text=True
                )
                if pull_process.returncode != 0:
                    slicer.util.errorDisplay("Failed to pull nodeodm:gpu image. Check logs.")
                    return
                else:
                    slicer.util.infoDisplay("Successfully pulled nodeodm:gpu.")
        except subprocess.CalledProcessError as e:
            slicer.util.errorDisplay(f"Error checking nodeodm:gpu status: {str(e)}")
            return

        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "publish=3002", "--format", "{{.ID}}"],
                capture_output=True, text=True, check=True
            )
            container_ids = result.stdout.strip().split()
            for cid in container_ids:
                if cid:
                    slicer.util.infoDisplay(f"Stopping container {cid} on port 3002...")
                    subprocess.run(["docker", "stop", cid], check=True)
        except Exception as e:
            slicer.util.warningDisplay(f"Error stopping old container(s): {str(e)}")

        local_folder = self.webODMLocalFolder
        if not os.path.isdir(local_folder):
            slicer.util.infoDisplay("Creating local WebODM folder...")
            os.makedirs(local_folder, exist_ok=True)

        slicer.util.infoDisplay("Launching nodeodm:gpu container on port 3002...")
        cmd = [
            "docker", "run", "--rm", "-d",
            "-p", "3002:3000",
            "--gpus", "all",
            "--name", "slicer-webodm-3002",
            "-v", f"{local_folder}:/var/www/data",
            "opendronemap/nodeodm:gpu"
        ]
        try:
            subprocess.run(cmd, check=True)
            slicer.util.infoDisplay("WebODM launched successfully on port 3002.")
            self.nodeIPLineEdit.setText("127.0.0.1")
            self.nodePortSpinBox.setValue(3002)
            slicer.app.settings().setValue("Photogrammetry/WebODMIP", "127.0.0.1")
            slicer.app.settings().setValue("Photogrammetry/WebODMPort", "3002")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to launch WebODM container:\n{str(e)}")

    def onStopNodeClicked(self):
        jobInProgress = (self.webODMManager.webodmTask is not None)

        if jobInProgress:
            proceed = slicer.util.confirmYesNoDisplay(
                "A WebODM task appears to be in progress. Stopping the node now will cancel that task.\n\n"
                "Do you want to continue?"
            )
            if not proceed:
                slicer.util.infoDisplay("Stop Node canceled by user.")
                return

        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "publish=3002", "--format", "{{.ID}}"],
                capture_output=True, text=True, check=True
            )
            container_ids = result.stdout.strip().split()
            if not container_ids or not any(container_ids):
                slicer.util.infoDisplay("No container currently running on port 3002.")
                return

            for cid in container_ids:
                if cid:
                    slicer.util.infoDisplay(f"Stopping container {cid} on port 3002...")
                    subprocess.run(["docker", "stop", cid], check=True)
            slicer.util.infoDisplay("Node successfully stopped.")
        except Exception as e:
            slicer.util.warningDisplay(f"Error stopping container(s): {str(e)}")

    def onStopMonitoring(self):
        self.webODMManager.onStopMonitoring()

    def onImportModelClicked(self):
        self.webODMManager.onImportModelClicked()

    # NEW >> Save and Restore Task
    def onSaveTaskClicked(self):
        """
        Save the current reconstruction setup, including:
         - Input folder, Output folder
         - setStates (minus exifData, to keep size small)
         - WebODM parameters (node IP, port, factor combos, concurrency, dataset name)
         - webODMManager.webodmOutDir (if any)
         - Optionally whether a completed model is present
        """
        masterFolder = self.masterFolderSelector.directory
        outputFolder = self.outputFolderSelector.directory

        if not masterFolder or not os.path.isdir(masterFolder):
            slicer.util.warningDisplay("Cannot save task: Master folder not valid.")
            return
        if not outputFolder or not os.path.isdir(outputFolder):
            slicer.util.warningDisplay("Cannot save task: Output folder not valid.")
            return

        # Prepare setStates copy without exifData
        setStatesCopy = {}
        for k, v in self.setStates.items():
            setStatesCopy[k] = {
                "imagePaths": v["imagePaths"],
                "imageStates": v["imageStates"],
            }

        # Gather factor combos
        chosenFactors = {}
        for factorName, combo in self.factorComboBoxes.items():
            chosenFactors[factorName] = combo.currentText

        # Check if there's a completed model
        modelPath = None
        taskCompleted = False
        if self.webODMManager.webodmOutDir and os.path.isdir(self.webODMManager.webodmOutDir):
            candidateObj = os.path.join(self.webODMManager.webodmOutDir, "odm_texturing", "odm_textured_model_geo.obj")
            if os.path.isfile(candidateObj):
                modelPath = candidateObj
                taskCompleted = True

        dataToSave = {
            "masterFolderPath": masterFolder,
            "outputFolderPath": outputFolder,
            "setStates": setStatesCopy,
            "webodmParams": {
                "nodeIP": self.nodeIPLineEdit.text.strip(),
                "nodePort": self.nodePortSpinBox.value,
                "factors": chosenFactors,
                "maxConcurrency": self.maxConcurrencySpinBox.value,
                "datasetName": self.datasetNameLineEdit.text.strip(),
            },
            "webodmOutDir": self.webODMManager.webodmOutDir,
            "taskCompleted": taskCompleted,
            "modelPath": modelPath,
            # For completeness, store resolution selection:
            "maskingResolution": "full" if self.radioFull.isChecked() else (
                "half" if self.radioHalf.isChecked() else "quarter")
        }

        dataToSave = convert_numpy_types(dataToSave)

        # Let user pick a JSON file path
        fileDialog = qt.QFileDialog()
        fileDialog.setNameFilter("JSON Files (*.json)")
        fileDialog.setDefaultSuffix("json")
        fileDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        if fileDialog.exec_() == qt.QFileDialog.Accepted:
            selectedFile = fileDialog.selectedFiles()[0]
            try:
                with open(selectedFile, 'w') as f:
                    json.dump(dataToSave, f, indent=2)
                slicer.util.infoDisplay(f"Task saved successfully to:\n{selectedFile}")
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to save task:\n{str(e)}")

    def onRestoreTaskClicked(self):
        """
        Load a JSON file, restore:
         - Master folder, Output folder
         - setStates => re-populate UI
         - node IP, port, factor combos, concurrency, dataset name
         - webodmOutDir
         - check if there's a completed model
        """
        fileDialog = qt.QFileDialog()
        fileDialog.setNameFilter("JSON Files (*.json)")
        fileDialog.setAcceptMode(qt.QFileDialog.AcceptOpen)
        if fileDialog.exec_() != qt.QFileDialog.Accepted:
            return
        selectedFile = fileDialog.selectedFiles()[0]
        if not os.path.isfile(selectedFile):
            slicer.util.errorDisplay("Invalid file selected.")
            return

        try:
            with open(selectedFile, 'r') as f:
                loadedData = json.load(f)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load JSON:\n{str(e)}")
            return

        # Basic validation
        requiredKeys = ["masterFolderPath", "outputFolderPath", "setStates", "webodmParams"]
        for rk in requiredKeys:
            if rk not in loadedData:
                slicer.util.errorDisplay(f"JSON missing required key '{rk}'. Invalid file.")
                return

        # 1) Reset everything
        self.clearAllData()

        # 2) Set master folder, output folder
        masterFolder = loadedData["masterFolderPath"]
        outputFolder = loadedData["outputFolderPath"]
        self.masterFolderSelector.directory = masterFolder
        self.outputFolderSelector.directory = outputFolder

        # 3) Re-process folders (onProcessFoldersClicked) to populate subfolders
        #    This sets self.setStates. Then we override with the loaded setStates.
        if os.path.isdir(masterFolder) and os.path.isdir(outputFolder):
            self.onProcessFoldersClicked()

            # Overwrite self.setStates with loaded data
            loadedSetStates = loadedData["setStates"]
            for setName, info in loadedSetStates.items():
                if setName not in self.setStates:
                    # Possibly no matching subfolder in current master folder?
                    continue

                # Convert string keys back to integer keys in "imageStates"
                restoredImageStates = {}
                for strKey, stateVal in info["imageStates"].items():
                    intKey = int(strKey)
                    restoredImageStates[intKey] = stateVal

                info["imageStates"] = restoredImageStates
                self.setStates[setName]["imageStates"] = info["imageStates"]
                self.setStates[setName]["imagePaths"] = info["imagePaths"]

            # 4) Re-check the pre-existing masks for each loaded set
            for setName in self.setStates.keys():
                self.currentSet = setName
                self.imagePaths = self.setStates[setName]["imagePaths"]
                self.imageStates = self.setStates[setName]["imageStates"]
                self.checkPreExistingMasks()

            # 5) Reset combobox to first set (or none if no sets)
            if self.imageSetComboBox.count > 0:
                self.imageSetComboBox.setCurrentIndex(0)
            else:
                self.currentSet = None
                self.imagePaths = []
                self.imageStates = {}

            self.updateMaskedCounter()
            self.updateWebODMTaskAvailability()
        else:
            slicer.util.warningDisplay("Master or output folder does not exist on disk. Partial restore done.")

        # 6) Restore webodm parameters
        wparams = loadedData.get("webodmParams", {})
        nodeIP = wparams.get("nodeIP", "127.0.0.1")
        nodePort = wparams.get("nodePort", 3002)
        factors = wparams.get("factors", {})
        maxConc = wparams.get("maxConcurrency", 16)
        dname = wparams.get("datasetName", "SlicerReconstruction")

        self.nodeIPLineEdit.setText(nodeIP)
        self.nodePortSpinBox.setValue(nodePort)
        self.maxConcurrencySpinBox.setValue(maxConc)
        self.datasetNameLineEdit.setText(dname)

        for factorName, selVal in factors.items():
            if factorName in self.factorComboBoxes:
                combo = self.factorComboBoxes[factorName]
                idx = combo.findText(str(selVal))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        # 7) webodmOutDir
        self.webODMManager.webodmOutDir = loadedData.get("webodmOutDir", None)

        # 8) If there's a completed model
        taskCompleted = loadedData.get("taskCompleted", False)
        modelPath = loadedData.get("modelPath", None)
        if taskCompleted and modelPath and os.path.isfile(modelPath):
            logging.info(f"Restored data suggests task completed. Model path: {modelPath}")
        else:
            logging.info("No completed model found in the restored data (or not found on disk).")

        # 9) Restore masking resolution radio
        res = loadedData.get("maskingResolution", "full")
        if res == "half":
            self.radioHalf.setChecked(True)
        elif res == "quarter":
            self.radioQuarter.setChecked(True)
        else:
            self.radioFull.setChecked(True)

        slicer.util.infoDisplay("Task restored successfully from:\n" + selectedFile)


class SlicerWebODMManager:
    """
    New manager class dedicated to WebODM-related functionality:
     - Checking Docker / WebODM status
     - Installing / Re-installing WebODM
     - Launching a container with GPU support on port 3002
     - Stopping a running node
     - Creating / monitoring a pyodm Task
     - Downloading results on completion
     - Stopping task monitoring
     - Importing the completed model into Slicer
    """

    def __init__(self, widget):
        self.widget = widget
        self.webodmTask = None
        self.webodmOutDir = None
        self.webodmTimer = None
        self.lastWebODMOutputLineIndex = 0

    def onRunWebODMTask(self):
        from pyodm import Node

        node_ip = self.widget.nodeIPLineEdit.text.strip()
        node_port = self.widget.nodePortSpinBox.value
        try:
            node = Node(node_ip, node_port)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to connect to Node at {node_ip}:{node_port}\n{str(e)}")
            return

        masterFolder = self.widget.masterFolderSelector.directory
        if not masterFolder or not os.path.isdir(masterFolder):
            slicer.util.errorDisplay("Master folder is invalid. Aborting.")
            return

        outputFolder = self.widget.outputFolderSelector.directory
        if not outputFolder or not os.path.isdir(outputFolder):
            slicer.util.errorDisplay("Output folder is invalid. Aborting.")
            return

        all_masked_color_jpgs = []
        all_mask_jpgs = []
        for root, dirs, files in os.walk(outputFolder):
            for fn in files:
                lower_fn = fn.lower()
                if lower_fn.endswith(".jpg") and not lower_fn.endswith("_mask.jpg"):
                    all_masked_color_jpgs.append(os.path.join(root, fn))
                elif lower_fn.endswith("_mask.jpg"):
                    all_mask_jpgs.append(os.path.join(root, fn))

        all_jpgs = all_masked_color_jpgs + all_mask_jpgs
        if len(all_jpgs) == 0:
            slicer.util.warningDisplay("No masked .jpg images found in output folder.")
            return

        combinedGCP = os.path.join(outputFolder, "combined_gcp_list.txt")
        files_to_upload = all_jpgs[:]
        if os.path.isfile(combinedGCP):
            files_to_upload.append(combinedGCP)
        else:
            slicer.util.infoDisplay("No combined_gcp_list.txt found. Proceeding without GCP...")

        params = dict(self.widget.baselineParams)

        for factorName, combo in self.widget.factorComboBoxes.items():
            chosen_str = combo.currentText
            if factorName == "ignore-gsd":
                params["ignore-gsd"] = (chosen_str.lower() == "true")
            elif factorName == "optimize-disk-space":
                params["optimize-disk-space"] = (chosen_str.lower() == "true")
            elif factorName == "no-gpu":
                params["no-gpu"] = (chosen_str.lower() == "true")
            else:
                try:
                    val_int = int(chosen_str)
                    params[factorName] = val_int
                except ValueError:
                    params[factorName] = chosen_str

        params["max-concurrency"] = self.widget.maxConcurrencySpinBox.value
        dataset_name = self.widget.datasetNameLineEdit.text.strip()
        if not dataset_name:
            dataset_name = "SlicerReconstruction"
        params["name"] = dataset_name

        prefix = self.widget.datasetNameLineEdit.text.strip() or "SlicerReconstruction"
        shortTaskName = self.widget.generateShortTaskName(prefix, params)

        # shortTaskName = self.widget.generateShortTaskName("SlicerReconstruction", params)
        slicer.util.infoDisplay("Creating WebODM Task (non-blocking). Upload may take time...")

        try:
            self.webodmTask = node.create_task(files=files_to_upload, options=params, name=shortTaskName)
        except Exception as e:
            slicer.util.errorDisplay(f"Task creation failed:\n{str(e)}")
            return

        slicer.util.infoDisplay(f"Task '{shortTaskName}' created successfully. Monitoring progress...")

        self.webodmOutDir = os.path.join(outputFolder, f"WebODM_{shortTaskName}")
        os.makedirs(self.webodmOutDir, exist_ok=True)

        self.widget.webodmLogTextEdit.clear()
        self.widget.stopMonitoringButton.setEnabled(True)

        self.lastWebODMOutputLineIndex = 0

        if self.webodmTimer:
            self.webodmTimer.stop()
            self.webodmTimer.deleteLater()

        self.webodmTimer = qt.QTimer()
        self.webodmTimer.setInterval(5000)
        self.webodmTimer.timeout.connect(self.checkWebODMTaskStatus)
        self.webodmTimer.start()
        self.widget.saveTaskButton.enabled = True

    def onStopMonitoring(self):
        if self.webodmTimer:
            self.webodmTimer.stop()
            self.webodmTimer.deleteLater()
            self.webodmTimer = None
        self.webodmTask = None
        self.widget.stopMonitoringButton.setEnabled(False)
        self.widget.webodmLogTextEdit.append("Stopped monitoring.")

    def checkWebODMTaskStatus(self):
        if not self.webodmTask:
            return
        try:
            info = self.webodmTask.info(with_output=self.lastWebODMOutputLineIndex)
        except Exception as e:
            self.widget.webodmLogTextEdit.append(f"Error retrieving task info: {str(e)}")
            slicer.app.processEvents()
            return

        newLines = info.output or []
        if len(newLines) > 0:
            for line in newLines:
                self.widget.webodmLogTextEdit.append(line)
            self.lastWebODMOutputLineIndex += len(newLines)

        self.widget.webodmLogTextEdit.append(f"Status: {info.status.name}, Progress: {info.progress}%")
        cursor = self.widget.webodmLogTextEdit.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        self.widget.webodmLogTextEdit.setTextCursor(cursor)
        self.widget.webodmLogTextEdit.ensureCursorVisible()
        slicer.app.processEvents()

        if info.status.name.lower() == "completed":
            self.widget.webodmLogTextEdit.append(f"Task completed! Downloading results to {self.webodmOutDir} ...")
            slicer.app.processEvents()
            try:
                self.webodmTask.download_assets(self.webodmOutDir)
                slicer.util.infoDisplay(f"Results downloaded to:\n{self.webodmOutDir}")
            except Exception as e:
                slicer.util.warningDisplay(f"Download failed: {str(e)}")

            if self.webodmTimer:
                self.webodmTimer.stop()
                self.webodmTimer.deleteLater()
                self.webodmTimer = None
            self.webodmTask = None
            self.widget.stopMonitoringButton.setEnabled(False)
        elif info.status.name.lower() in ["failed", "canceled"]:
            self.widget.webodmLogTextEdit.append("Task failed or canceled. Stopping.")
            slicer.app.processEvents()
            if self.webodmTimer:
                self.webodmTimer.stop()
                self.webodmTimer.deleteLater()
                self.webodmTimer = None
            self.webodmTask = None
            self.widget.stopMonitoringButton.setEnabled(False)

    def onImportModelClicked(self):
        if not self.webodmOutDir:
            slicer.util.warningDisplay(
                "No WebODM output directory found. Please run a WebODM reconstruction first."
            )
            return

        objPath = os.path.join(self.webodmOutDir, "odm_texturing", "odm_textured_model_geo.obj")
        if not os.path.isfile(objPath):
            slicer.util.warningDisplay(
                f"No model file found at:\n{objPath}\nMake sure the reconstruction completed successfully."
            )
            return

        try:
            import OBJFile  # Your unmodified module

            # Create a dummy Python object to stand in for the "parent"
            dummyParent = types.SimpleNamespace()

            # Instantiate the custom OBJ reader with the dummy object
            reader = OBJFile.OBJFileFileReader(dummyParent)
            properties = {'fileName': objPath}

            # Attempt to load the textured OBJ
            success = reader.load(properties)
            if success:
                # Now, dummyParent should have loadedNodes set by OBJFileFileReader
                if not hasattr(dummyParent, 'loadedNodes') or not dummyParent.loadedNodes:
                    slicer.util.warningDisplay(
                        "Reader indicated success, but 'loadedNodes' was not set on the parent."
                    )
                    return

                nodeID = dummyParent.loadedNodes[0]
                loadedNode = slicer.mrmlScene.GetNodeByID(nodeID)
                if loadedNode:
                    layoutMgr = slicer.app.layoutManager()
                    layoutMgr.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                    slicer.util.infoDisplay("Imported textured OBJ and switched to 3D layout.")
                else:
                    slicer.util.warningDisplay("Reader indicated success but could not get the node.")
            else:
                slicer.util.warningDisplay("Failed to load textured model. Check logs.")

        except Exception as e:
            slicer.util.warningDisplay(f"Exception while loading textured model: {str(e)}")


class PhotogrammetryLogic(ScriptedLoadableModuleLogic):
    """
    Loads the SAM model, runs segmentation on color arrays.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.predictor = None
        self.sam = None
        import torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA:0 for SAM.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for SAM.")

    def loadSAMModel(self, variant, filename, url):
        try:
            sam_checkpoint = self.check_and_download_weights(filename, url)
            from segment_anything import sam_model_registry, SamPredictor
            self.sam = sam_model_registry[variant](checkpoint=sam_checkpoint)
            self.sam.to(self.device)
            self.predictor = SamPredictor(self.sam)
            return True
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            return False

    @staticmethod
    def check_and_download_weights(filename, weights_url):
        modulePath = os.path.dirname(slicer.modules.photogrammetry.path)
        resourcePath = os.path.join(modulePath, 'Resources', filename)
        if not os.path.isfile(resourcePath):
            slicer.util.infoDisplay(f"Downloading {filename}... This may take a few minutes...", autoCloseMsec=2000)
            try:
                slicer.util.downloadFile(url=weights_url, targetFilePath=resourcePath)
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to download {filename}: {str(e)}")
                raise RuntimeError("Could not download SAM weights.")
        return resourcePath

    @staticmethod
    def get_image_paths_from_folder(folder_path: str, extensions=None) -> List[str]:
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        folder_path = os.path.abspath(folder_path)
        image_paths = []
        if os.path.isdir(folder_path):
            for fn in os.listdir(folder_path):
                ext = os.path.splitext(fn)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(folder_path, fn)
                    if os.path.isfile(full_path):
                        image_paths.append(full_path)
        return sorted(image_paths)

    def run_sam_segmentation_with_incl_excl(self, image_rgb, bounding_box, posPoints, negPoints, marker_outputs=None):
        if not self.predictor:
            raise RuntimeError("SAM model is not loaded.")

        import torch
        import numpy as np

        if marker_outputs is None:
            marker_outputs = []

        mainBox = np.array(bounding_box, dtype=np.int32)
        all_boxes = [mainBox]
        for marker_dict in marker_outputs:
            x_min, y_min, x_max, y_max = marker_dict["bbox"]
            pad = 25
            x_min_new = x_min - pad
            y_min_new = y_min - pad
            x_max_new = x_max + pad
            y_max_new = y_max + pad
            all_boxes.append(np.array([x_min_new, y_min_new, x_max_new, y_max_new], dtype=np.int32))

        allPointCoords = []
        allLabels = []
        if posPoints:
            allPointCoords.extend(posPoints)
            allLabels.extend([1] * len(posPoints))
        if negPoints:
            allPointCoords.extend(negPoints)
            allLabels.extend([0] * len(negPoints))

        allPointCoords = np.array(allPointCoords, dtype=np.float32) if allPointCoords else None
        allLabels = np.array(allLabels, dtype=np.int32) if allLabels else None

        self.predictor.set_image(image_rgb)
        h, w, _ = image_rgb.shape
        combined_mask = np.zeros((h, w), dtype=bool)

        with torch.no_grad():
            for box in all_boxes:
                masks, scores, logits = self.predictor.predict(
                    point_coords=allPointCoords,
                    point_labels=allLabels,
                    box=box,
                    multimask_output=False
                )
                mask_bool = masks[0].astype(bool)
                combined_mask = np.logical_or(combined_mask, mask_bool)

        return combined_mask.astype(np.uint8)

    def array_to_pil(self, colorArr):
        from PIL import Image
        import numpy as np
        return Image.fromarray(colorArr.astype(np.uint8))

    def pil_to_opencv(self, pil_image):
        import cv2
        import numpy as np
        cv_image = np.array(pil_image)
        if cv_image.ndim == 2:
            return cv_image
        elif cv_image.shape[2] == 4:  # RGBA -> BGR
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        else:  # RGB -> BGR
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image
