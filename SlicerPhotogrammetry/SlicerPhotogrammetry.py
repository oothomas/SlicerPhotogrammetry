import os
import sys
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
from slicer.ScriptedLoadableModule import *
from typing import List


class SlicerPhotogrammetry(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlicerPhotogrammetry"
        self.parent.categories = ["SlicerPhotogrammetry"]
        self.parent.dependencies = []
        self.parent.contributors = ["Oshane Thomas"]
        self.parent.helpText = """NA"""
        self.parent.acknowledgementText = """NA"""

        # Suppress VTK warnings globally
        vtk.vtkObject.GlobalWarningDisplayOff()


class SlicerPhotogrammetryWidget(ScriptedLoadableModuleWidget):
    """
    Manages UI for:
     - SAM model loading,
     - Folder processing,
     - Bbox/masking steps,
     - EXIF + color/writing,
     - Creating _mask.png for webODM,
     - Creating single combined GCP file for all sets,
     - Non-blocking WebODM tasks (using pyodm),
     - Shortening WebODM output folder names,
     - Checking/Installing/Re-launching WebODM on port 3002 with GPU support.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

        self.imageIndexLabel = None
        self.logic = None
        self.vtkLogFilter = None
        self.logger = None

        # CHANGED: small in-memory cache
        self.imageCache = {}
        self.maxCacheSize = 64

        self.downsampleFactor = 0.15  # e.g. 15%

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
        self.placingBoundingBox = False

        self.buttonsToManage = []
        self.prevButton = None
        self.nextButton = None

        # CHANGED: We no longer have a separate Done button. We will combine finalizing ROI + masking
        # into a single ?Mask Current Image? button:
        self.maskCurrentImageButton = None

        self.maskAllImagesButton = None
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

        # WebODM
        self.nodeIPLineEdit = None
        self.nodePortSpinBox = None
        self.launchWebODMTaskButton = None
        self.webodmLogTextEdit = None
        self.stopMonitoringButton = None
        self.lastWebODMOutputLineIndex = 0

        # CHANGES: default params
        self.baselineParams = {
            "matcher-type": "bruteforce",
            "orthophoto-resolution": 0.3,
            "skip-orthophoto": True,
            "texturing-single-material": True,
            "use-3dmesh": True,
            "feature-type": "dspsift",
            "feature-quality": "ultra",
            "pc-quality": "ultra",
            "max-concurrency": 32,
        }

        self.factorLevels = {
            "ignore-gsd": [False, True],
            "matcher-neighbors": [0, 8, 16, 24],
            "mesh-octree-depth": [12, 13, 14],
            "mesh-size": [300000, 500000, 750000],
            "min-num-features": [10000, 20000, 50000],
            "pc-filter": [1, 2, 3, 4, 5],
            "depthmap-resolution": [4096, 8192]
        }
        self.factorComboBoxes = {}

        self.maskedCountLabel = None
        self.layoutId = 1003

        # ### CHANGES: New variables for WebODM installation ###
        self.webODMCheckStatusButton = None
        self.webODMInstallButton = None
        self.webODMRelaunchButton = None
        self.webODMLocalFolder = None  # e.g. <ModulePath>/Resources/WebODM

        # NEW: We'll hold a reference to the manager class once it's created.
        self.webODMManager = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.load_dependencies()
        self.logic = SlicerPhotogrammetryLogic()

        self.setupLogger()
        self.layout.setAlignment(qt.Qt.AlignTop)
        self.createCustomLayout()

        #
        # (A) Main Collapsible: Import Image Sets
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Import Image Sets"
        self.layout.addWidget(parametersCollapsibleButton)
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
        savedMasterFolder = slicer.app.settings().value("SlicerPhotogrammetry/masterFolderPath", "")
        if os.path.isdir(savedMasterFolder):
            self.masterFolderSelector.directory = savedMasterFolder
        parametersFormLayout.addRow("Master Folder:", self.masterFolderSelector)

        self.outputFolderSelector = ctk.ctkDirectoryButton()
        savedOutputFolder = slicer.app.settings().value("SlicerPhotogrammetry/outputFolderPath", "")
        if os.path.isdir(savedOutputFolder):
            self.outputFolderSelector.directory = savedOutputFolder
        parametersFormLayout.addRow("Output Folder:", self.outputFolderSelector)

        self.processButton = qt.QPushButton("Process Folders")
        parametersFormLayout.addWidget(self.processButton)
        self.processButton.connect('clicked(bool)', self.onProcessFoldersClicked)

        self.processFoldersProgressBar = qt.QProgressBar()
        self.processFoldersProgressBar.setVisible(False)
        parametersFormLayout.addWidget(self.processFoldersProgressBar)

        self.imageSetComboBox = qt.QComboBox()
        self.imageSetComboBox.enabled = False
        parametersFormLayout.addRow("Image Set:", self.imageSetComboBox)
        self.imageSetComboBox.connect('currentIndexChanged(int)', self.onImageSetSelected)

        self.placeBoundingBoxButton = qt.QPushButton("Place Bounding Box")
        self.placeBoundingBoxButton.enabled = False
        parametersFormLayout.addWidget(self.placeBoundingBoxButton)
        self.placeBoundingBoxButton.connect('clicked(bool)', self.onPlaceBoundingBoxClicked)

        # CHANGED: Remove old Done button references. We combine bounding-box finalization + masking
        # in one step. So no separate ?Done? button in the UI now.

        # NEW: Renamed the old separate "Mask Current Image" button to a single combined step:
        self.maskCurrentImageButton = qt.QPushButton("Mask Current Image")
        self.maskCurrentImageButton.enabled = False
        parametersFormLayout.addWidget(self.maskCurrentImageButton)
        # This new method merges the old ?onDoneClicked? + ?onMaskClicked? logic
        self.maskCurrentImageButton.connect('clicked(bool)', self.onMaskCurrentImageClicked)

        navLayout = qt.QGridLayout()

        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
        prevIconPath = os.path.join(modulePath, 'Resources/Icons', 'Previous.png')
        nextIconPath = os.path.join(modulePath, 'Resources/Icons', 'Next.png')

        # Previous Button with Icon
        self.prevButton = qt.QPushButton("<")
        #prevIcon = qt.QIcon(qt.QPixmap(prevIconPath))
        #self.prevButton.setIcon(prevIcon)
        self.prevButton.setToolTip("Go to the previous image")

        # Previous Button with Icon
        self.nextButton = qt.QPushButton(">")
        #nextIcon = qt.QIcon(qt.QPixmap(nextIconPath))
        #self.nextButton.setIcon(nextIcon)
        self.nextButton.setToolTip("Go to the next image")

        # Image Index Label
        self.imageIndexLabel = qt.QLabel("Image 0")
        self.imageIndexLabel.setAlignment(qt.Qt.AlignCenter)

        self.prevButton.enabled = False
        self.nextButton.enabled = False

        navLayout.addWidget(self.prevButton, 0, 0)
        navLayout.addWidget(self.imageIndexLabel, 0, 1)
        navLayout.addWidget(self.nextButton, 0, 2)

        parametersFormLayout.addRow("    ", navLayout)
        self.prevButton.connect('clicked(bool)', self.onPrevImage)
        self.nextButton.connect('clicked(bool)', self.onNextImage)

        self.maskAllImagesButton = qt.QPushButton("Mask All Images In Set")
        self.maskAllImagesButton.enabled = False
        parametersFormLayout.addWidget(self.maskAllImagesButton)
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)

        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        self.maskAllProgressBar.setTextVisible(True)
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        self.maskedCountLabel = qt.QLabel("Masked: 0/0")
        parametersFormLayout.addRow("Overall Progress:", self.maskedCountLabel)

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
        # (B) Find-GCP
        #
        webODMCollapsibleButton = ctk.ctkCollapsibleButton()
        webODMCollapsibleButton.text = "Find-GCP"
        self.layout.addWidget(webODMCollapsibleButton)
        webODMFormLayout = qt.QFormLayout(webODMCollapsibleButton)

        self.cloneFindGCPButton = qt.QPushButton("Clone Find-GCP")
        webODMFormLayout.addWidget(self.cloneFindGCPButton)
        self.cloneFindGCPButton.connect('clicked(bool)', self.onCloneFindGCPClicked)

        self.findGCPScriptSelector = ctk.ctkPathLineEdit()
        self.findGCPScriptSelector.filters = ctk.ctkPathLineEdit().Files
        self.findGCPScriptSelector.setToolTip("Select path to Find-GCP.py script.")
        webODMFormLayout.addRow("Find-GCP Script:", self.findGCPScriptSelector)

        savedFindGCPScript = slicer.app.settings().value("SlicerPhotogrammetry/findGCPScriptPath", "")
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
        # (C) Launch WebODM Task
        #
        webodmTaskCollapsible = ctk.ctkCollapsibleButton()
        webodmTaskCollapsible.text = "Launch WebODM Task"
        self.layout.addWidget(webodmTaskCollapsible)
        webodmTaskFormLayout = qt.QFormLayout(webodmTaskCollapsible)

        self.nodeIPLineEdit = qt.QLineEdit("127.0.0.1")
        webodmTaskFormLayout.addRow("Node IP:", self.nodeIPLineEdit)

        self.nodePortSpinBox = qt.QSpinBox()
        self.nodePortSpinBox.setMinimum(1)
        self.nodePortSpinBox.setMaximum(65535)
        self.nodePortSpinBox.setValue(3002)
        webodmTaskFormLayout.addRow("Node Port:", self.nodePortSpinBox)

        for factorName, levels in self.factorLevels.items():
            combo = qt.QComboBox()
            for val in levels:
                combo.addItem(str(val))
            self.factorComboBoxes[factorName] = combo
            webodmTaskFormLayout.addRow(f"{factorName}:", combo)

        self.launchWebODMTaskButton = qt.QPushButton("Run WebODM Task With Selected Parameters (non-blocking)")
        webodmTaskFormLayout.addWidget(self.launchWebODMTaskButton)
        self.launchWebODMTaskButton.setEnabled(False)
        self.launchWebODMTaskButton.connect('clicked(bool)', self.onRunWebODMTask)

        self.webodmLogTextEdit = qt.QTextEdit()
        self.webodmLogTextEdit.setReadOnly(True)
        webodmTaskFormLayout.addRow("Console Log:", self.webodmLogTextEdit)

        self.stopMonitoringButton = qt.QPushButton("Stop Monitoring")
        self.stopMonitoringButton.setEnabled(False)
        webodmTaskFormLayout.addWidget(self.stopMonitoringButton)

        ###
        # (D) NEW SECTION: Manage WebODM (Install/Launch)
        ###
        manageWODMCollapsibleButton = ctk.ctkCollapsibleButton()
        manageWODMCollapsibleButton.text = "Manage WebODM (Install/Launch)"
        self.layout.addWidget(manageWODMCollapsibleButton)
        manageWODMFormLayout = qt.QFormLayout(manageWODMCollapsibleButton)

        self.webODMCheckStatusButton = qt.QPushButton("Check WebODM Status on port 3002")
        manageWODMFormLayout.addWidget(self.webODMCheckStatusButton)

        self.webODMInstallButton = qt.QPushButton("Install/Reinstall WebODM (GPU)")
        manageWODMFormLayout.addWidget(self.webODMInstallButton)

        self.webODMRelaunchButton = qt.QPushButton("Relaunch WebODM on Port 3002")
        manageWODMFormLayout.addWidget(self.webODMRelaunchButton)

        self.layout.addStretch(1)
        self.createMasterNodes()

        # Initialize path for local WebODM folder
        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
        self.webODMLocalFolder = os.path.join(modulePath, 'Resources', 'WebODM')

        # ----
        # (E) Initialize the WebODM Manager, hooking up all the WebODM-related signals:
        # ----
        self.webODMManager = SlicerWebODMManager(widget=self)
        self.webODMCheckStatusButton.connect('clicked(bool)', self.webODMManager.onCheckWebODMStatusClicked)
        self.webODMInstallButton.connect('clicked(bool)', self.webODMManager.onInstallWebODMClicked)
        self.webODMRelaunchButton.connect('clicked(bool)', self.webODMManager.onRelaunchWebODMClicked)
        self.stopMonitoringButton.connect('clicked(bool)', self.webODMManager.onStopMonitoring)

    def load_dependencies(self):
        # ALWAYS ADD EXTERNAL IMPORTS HERE
        ##############################################################
        # EXIF helper: read/write using Pillow
        ##############################################################
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.messageBox("SlicerPhotogrammetry requires the PyTorch extension. Please install it from the "
                                   "Extensions Manager.")
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            logging.debug(
                'SlicerPhotogrammetry requires the PyTorch Python package. Installing... (it may take several '
                'minutes)')
            torch = torchLogic.installTorch(askConfirmation=True, forceComputationBackend='cu118')

            if torch:
                # Ask user to restart 3D Slicer
                restart = slicer.util.confirmYesNoDisplay(
                    "Pytorch dependencies has been installed. To apply changes, a restart of 3D Slicer is necessary. "
                    "Would you like to restart now? Click 'YES' to restart immediately or 'NO' if you wish to save your "
                    "work first and restart manually later.")

                if restart:
                    slicer.util.restart()

            if torch is None:
                slicer.util.messageBox('PyTorch extension needs to be installed manually to use this module.')

        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
        except ImportError:
            slicer.util.pip_install("Pillow")
            from PIL import Image
            from PIL.ExifTags import TAGS

        try:
            import cv2
            # Check if contrib modules are available
            if not hasattr(cv2, 'xfeatures2d'):
                raise ImportError("opencv-contrib-python is not properly installed")
        except ImportError:
            slicer.util.pip_install("opencv-python")
            slicer.util.pip_install("opencv-contrib-python")
            import cv2

        try:
            import segment_anything
        except ImportError:
            slicer.util.pip_install("git+https://github.com/facebookresearch/segment-anything.git")
            import segment_anything

        try:
            import git
        except ImportError:
            slicer.util.pip_install("GitPython")
            import git

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
        customLayout = """
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
        layoutMgr = slicer.app.layoutManager()
        layoutNode = layoutMgr.layoutLogic().GetLayoutNode()
        layoutNode.AddLayoutDescription(self.layoutId, customLayout)
        layoutMgr.setLayout(self.layoutId)

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
        red2Comp.SetLabelVolumeID(self.emptyNode.GetID())

    def onFindGCPScriptChanged(self, newPath):
        slicer.app.settings().setValue("SlicerPhotogrammetry/findGCPScriptPath", newPath)

    def updateMaskedCounter(self):
        totalImages = 0
        maskedCount = 0
        for setName, setData in self.setStates.items():
            totalImages += len(setData["imagePaths"])
            for _, info in setData["imageStates"].items():
                if info["state"] == "masked":
                    maskedCount += 1
        self.maskedCountLabel.setText(f"Masked: {maskedCount}/{totalImages}")

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
        else:
            slicer.util.errorDisplay("Failed to load the model. Check logs.")

    def onProcessFoldersClicked(self):
        if not self.modelLoaded:
            slicer.util.warningDisplay("Please load a SAM model before processing folders.")
            return

        if self.anySetHasProgress():
            if not slicer.util.confirmYesNoDisplay("All progress made so far will be lost. Proceed?"):
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

        slicer.app.settings().setValue("SlicerPhotogrammetry/masterFolderPath", masterFolderPath)
        slicer.app.settings().setValue("SlicerPhotogrammetry/outputFolderPath", outputFolderPath)

        self.processFoldersProgressBar.setVisible(True)
        self.processFoldersProgressBar.setValue(0)

        subfolders = [f for f in os.listdir(masterFolderPath) if os.path.isdir(os.path.join(masterFolderPath, f))]
        self.imageSetComboBox.clear()

        if len(subfolders) > 0:
            self.processFoldersProgressBar.setRange(0, len(subfolders))
            for i, sf in enumerate(subfolders):
                self.imageSetComboBox.addItem(sf)
                self.processFoldersProgressBar.setValue(i + 1)
                slicer.app.processEvents()
            self.imageSetComboBox.enabled = True
        else:
            slicer.util.infoDisplay("No subfolders found in master folder.")

        self.processFoldersProgressBar.setVisible(False)

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
        self.imageIndexLabel.setText("Image 0")

        self.launchWebODMTaskButton.setEnabled(False)
        self.maskedCountLabel.setText("Masked: 0/0")

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

    def updateVolumeDisplay(self):
        self.imageIndexLabel.setText(f"Image {self.currentImageIndex}")
        if self.currentImageIndex < 0 or self.currentImageIndex >= len(self.imagePaths):
            return

        st = self.imageStates[self.currentImageIndex]["state"]
        self.removeBboxLines()

        # Retrieve the downsampled color only
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

    def getDownsampledColor(self, setName, index):
        """
        Downsample only the color array (no grayscale).
        """
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

        # Flip up-down and left-right to match how Slicer uses array data
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
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if stInfo and stInfo["state"] == "masked" and stInfo["bboxCoords"] is not None:
            self.maskAllImagesButton.enabled = True
        else:
            self.maskAllImagesButton.enabled = False

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
        if self.currentImageIndex > 0:
            self.currentImageIndex -= 1
            self.updateVolumeDisplay()

    def onNextImage(self):
        if self.currentImageIndex < len(self.imagePaths) - 1:
            self.currentImageIndex += 1
            self.updateVolumeDisplay()

    # --------------------------------------------------------------------------------
    # CHANGED: Instead of "onDoneClicked" + "onMaskClicked," we create one combined
    #          method "onMaskCurrentImageClicked".
    # --------------------------------------------------------------------------------
    def onMaskCurrentImageClicked(self):
        """
        This combines the bounding-box-finalization code and the masking code into a single step.
        If the user has placed an ROI but hasn't finalized it yet, we finalize it (like old onDoneClicked)
        and then proceed to mask (like old onMaskClicked).
        """
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo:
            slicer.util.warningDisplay("No image state found. Please select a valid image.")
            return

        currentState = stInfo["state"]
        # If we do have an ROI node still in the scene, let's finalize bounding box:
        if self.boundingBoxRoiNode:
            self.finalizeBoundingBoxAndRemoveROI()
            # after finalize, we set it to "bbox", so we can proceed

        # Now do the old "onMaskClicked" logic:
        stInfo = self.imageStates.get(self.currentImageIndex, None)  # refresh local reference
        if not stInfo or stInfo["state"] != "bbox":
            slicer.util.warningDisplay("No bounding box defined or finalized for this image. Cannot mask.")
            return

        # Actually perform the SAM masking:
        import numpy as np
        import cv2

        bboxDown = stInfo["bboxCoords"]
        bboxFull = self.downBboxToFullBbox(bboxDown, self.currentSet, self.currentImageIndex)
        colorArrFull = self.getFullColorArray(self.currentSet, self.currentImageIndex)

        opencvFull = self.logic.pil_to_opencv(self.logic.array_to_pil(colorArrFull))
        marker_outputs = self.detect_aruco_bounding_boxes(opencvFull, aruco_dict=cv2.aruco.DICT_4X4_250)

        if len(marker_outputs) == 0:
            mask = self.logic.run_sam_segmentation(colorArrFull, bboxFull)
        else:
            import torch
            all_boxes = self.assemble_bboxes(np.array(bboxFull, dtype=np.int32), marker_outputs, pad=25)
            self.logic.predictor.set_image(colorArrFull)
            combined_mask = np.zeros((colorArrFull.shape[0], colorArrFull.shape[1]), dtype=bool)
            for box in all_boxes:
                with torch.no_grad():
                    masks, _, _ = self.logic.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=False
                    )
                mask_bool = masks[0].astype(bool)
                combined_mask = np.logical_or(combined_mask, mask_bool)
            mask = combined_mask.astype(np.uint8)

        stInfo["state"] = "masked"
        maskBool = (mask > 0)
        self.saveMaskedImage(self.currentImageIndex, colorArrFull, maskBool)
        self.updateVolumeDisplay()
        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()

        self.restoreButtonStates()

        self.enableMaskAllImagesIfPossible()

    def finalizeBoundingBoxAndRemoveROI(self):
        """ Helper function that duplicates the old 'onDoneClicked' bounding box finalization. """
        if not self.boundingBoxRoiNode:
            return

        coordsDown = self.computeBboxFromROI()
        stInfo = self.imageStates[self.currentImageIndex]
        stInfo["state"] = "bbox"
        stInfo["bboxCoords"] = coordsDown
        stInfo["maskNodes"] = None

        # Turn off handles
        dnode = self.boundingBoxRoiNode.GetDisplayNode()
        dnode.SetHandlesInteractive(False)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        # If ROI display node still around, remove it
        if self.boundingBoxRoiNode.GetDisplayNode():
            slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode.GetDisplayNode())
        slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode)
        self.boundingBoxRoiNode = None

    # --------------------------------------------------------------------------------
    # End of merged function
    # --------------------------------------------------------------------------------

    def onMaskAllImagesClicked(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo or stInfo["state"] != "masked" or not stInfo["bboxCoords"]:
            slicer.util.warningDisplay("Current image is not masked or has no bounding box info.")
            return

        bboxDown = stInfo["bboxCoords"]
        self.maskAllProgressBar.setVisible(True)
        self.maskAllProgressBar.setTextVisible(True)
        toMask = [i for i in range(len(self.imagePaths)) if
                  i != self.currentImageIndex and self.imageStates[i]["state"] != "masked"]
        n = len(toMask)
        self.maskAllProgressBar.setRange(0, n)
        self.maskAllProgressBar.setValue(0)

        if n == 0:
            slicer.util.infoDisplay("All images in this set are already masked.")
            self.maskAllProgressBar.setVisible(False)
            return

        start_time = time.time()

        for count, i in enumerate(toMask):
            self.maskSingleImage(i, bboxDown)
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

        slicer.util.infoDisplay("All images in set masked and saved.")
        self.maskAllProgressBar.setVisible(False)
        self.updateVolumeDisplay()
        self.updateMaskedCounter()
        self.updateWebODMTaskAvailability()

    def maskSingleImage(self, index, bboxDown):
        import numpy as np
        import cv2
        import torch

        bboxFull = self.downBboxToFullBbox(bboxDown, self.currentSet, index)
        colorArrFull = self.getFullColorArray(self.currentSet, index)

        opencvFull = self.logic.pil_to_opencv(self.logic.array_to_pil(colorArrFull))
        marker_outputs = self.detect_aruco_bounding_boxes(opencvFull, aruco_dict=cv2.aruco.DICT_4X4_250)

        if len(marker_outputs) == 0:
            mask = self.logic.run_sam_segmentation(colorArrFull, bboxFull)
        else:
            all_boxes = self.assemble_bboxes(np.array(bboxFull, dtype=np.int32), marker_outputs, pad=25)
            self.logic.predictor.set_image(colorArrFull)
            combined_mask = np.zeros((colorArrFull.shape[0], colorArrFull.shape[1]), dtype=bool)
            for box in all_boxes:
                with torch.no_grad():
                    masks, _, _ = self.logic.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=False
                    )
                mask_bool = masks[0].astype(bool)
                combined_mask = np.logical_or(combined_mask, mask_bool)
            mask = combined_mask.astype(np.uint8)

        self.imageStates[index]["state"] = "masked"
        self.imageStates[index]["bboxCoords"] = bboxDown
        self.imageStates[index]["maskNodes"] = None

        maskBool = (mask > 0)
        self.saveMaskedImage(index, colorArrFull, maskBool)

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

    def removeBboxLines(self):
        for ln in self.currentBboxLineNodes:
            if ln and slicer.mrmlScene.IsNodePresent(ln):
                slicer.mrmlScene.RemoveNode(ln)
        self.currentBboxLineNodes = []

    def drawBboxLines(self, coordsDown):
        if not coordsDown:
            return
        ijkToRasMat = vtk.vtkMatrix4x4()
        self.masterVolumeNode.GetIJKToRASMatrix(ijkToRasMat)

        def ijkToRas(i, j):
            p = [i, j, 0, 1]
            ras = ijkToRasMat.MultiplyPoint(p)
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

    def onPlaceBoundingBoxClicked(self):
        self.storeCurrentButtonStates()
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
        """
        Disables all UI elements except the new MaskCurrentImage button,
        which we keep enabled to let user finalize bounding box once done.
        """
        for b in self.buttonsToManage:
            if b != self.maskCurrentImageButton:
                if isinstance(b, qt.QComboBox):
                    b.setEnabled(False)
                else:
                    b.enabled = False

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
            "Draw the ROI and use the handles to adjust it. When done, click 'Mask Current Image' to finalize + mask.",
            autoCloseMsec=6000
        )

        # Our single combined button will finalize + mask once user is satisfied
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

    def removeBboxFromCurrentImage(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if stInfo:
            stInfo["state"] = "none"
            stInfo["bboxCoords"] = None
            stInfo["maskNodes"] = None
        self.removeBboxLines()

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

    def updateWebODMTaskAvailability(self):
        allSetsMasked = self.allSetsHavePhysicalMasks()
        self.launchWebODMTaskButton.setEnabled(allSetsMasked)

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

        subfolders = [f for f in os.listdir(masterFolderPath) if os.path.isdir(os.path.join(masterFolderPath, f))]
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
        """
        We keep the same signature, but the actual WebODM logic is delegated
        to our SlicerWebODMManager. This preserves the UI flow.
        """
        # First replicate the existing check:
        if not self.allSetsHavePhysicalMasks():
            slicer.util.warningDisplay("Not all images have masks. Please mask all sets first.")
            return

        # Now hand off to the manager for the actual logic
        self.webODMManager.onRunWebODMTask()

    def onCloneFindGCPClicked(self):
        import os, shutil
        import slicer

        try:
            import git
        except ImportError:
            slicer.util.pip_install("GitPython")
            import git  # try again

        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
        resourcesFolder = os.path.join(modulePath, 'Resources')
        os.makedirs(resourcesFolder, exist_ok=True)

        cloneFolder = os.path.join(resourcesFolder, "Find-GCP-Repo")
        localGCPFindScript = os.path.join(cloneFolder, "gcp_find.py")

        if os.path.isdir(cloneFolder):
            msg = (
                "The 'Find-GCP-Repo' folder already exists in the Resources directory.\n"
                "Would you like to delete it and clone again (overwrite)?"
            )
            if not slicer.util.confirmYesNoDisplay(msg):
                slicer.util.infoDisplay("Using existing clone; no changes made.")
                if os.path.isfile(localGCPFindScript):
                    self.findGCPScriptSelector.setCurrentPath(localGCPFindScript)
                    slicer.app.settings().setValue("SlicerPhotogrammetry/findGCPScriptPath", localGCPFindScript)
                else:
                    slicer.util.warningDisplay(
                        f"Existing clone found, but {localGCPFindScript} does not exist.\n"
                        "Please pick the correct script manually."
                    )
                return
            else:
                try:
                    shutil.rmtree(cloneFolder)
                except Exception as e:
                    slicer.util.errorDisplay(
                        f"Failed to remove existing cloned folder:\n{cloneFolder}\nError: {str(e)}"
                    )
                    return

        slicer.util.infoDisplay(
            f"Cloning the entire Find-GCP repo to:\n{cloneFolder}\nPlease wait...",
            autoCloseMsec=3000
        )
        try:
            git.Repo.clone_from(
                url="https://github.com/zsiki/Find-GCP.git",
                to_path=cloneFolder
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to clone Find-GCP repo:\n{str(e)}")
            return

        if not os.path.isfile(localGCPFindScript):
            slicer.util.warningDisplay(
                f"Repo cloned, but {localGCPFindScript} was not found.\n"
                "Please check the repo contents or specify the correct script."
            )
            return

        self.findGCPScriptSelector.setCurrentPath(localGCPFindScript)
        slicer.app.settings().setValue("SlicerPhotogrammetry/findGCPScriptPath", localGCPFindScript)

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


class SlicerWebODMManager:
    """
    New manager class dedicated to WebODM-related functionality:
     - Checking Docker / WebODM status
     - Installing / Re-installing WebODM
     - Relaunching a container with GPU support
     - Creating / monitoring a pyodm Task
     - Downloading results on completion
     - Stopping task monitoring
    """

    def __init__(self, widget):
        """
        We store references to the main widget so we can access UI elements,
        directory paths, and shared logic. The widget's 'onRunWebODMTask'
        method calls self.onRunWebODMTask() below, etc.
        """
        self.widget = widget  # SlicerPhotogrammetryWidget instance

        # We'll store our pyodm references here
        self.webodmTask = None
        self.webodmOutDir = None
        self.webodmTimer = None

        # Keep track of the last line index for output
        self.lastWebODMOutputLineIndex = 0

    def onCheckWebODMStatusClicked(self):
        """
        Check if Docker is installed, see if anything is on port 3002,
        and if so, auto-populate IP/port in the UI.
        """
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except Exception as e:
            slicer.util.warningDisplay(
                f"Docker not found or not in PATH.\nError: {str(e)}\nPlease install Docker first."
            )
            return

        from pyodm import Node
        ip_test = "127.0.0.1"
        port_test = 3002
        try:
            test_node = Node(ip_test, port_test)
            info = test_node.info()
            slicer.util.infoDisplay("WebODM node found on 127.0.0.1:3002.\nAuto-populating IP & Port.")
            self.widget.nodeIPLineEdit.setText(ip_test)
            self.widget.nodePortSpinBox.setValue(port_test)
            slicer.app.settings().setValue("SlicerPhotogrammetry/WebODMIP", ip_test)
            slicer.app.settings().setValue("SlicerPhotogrammetry/WebODMPort", str(port_test))
        except Exception:
            slicer.util.infoDisplay("No WebODM node found on port 3002. You can install/launch below.")

    def onInstallWebODMClicked(self):
        """
        Install or re-install WebODM (GPU-based) in widget.webODMLocalFolder.
        For demonstration, we only do docker pull of 'opendronemap/nodeodm:gpu'.
        """
        localFolder = self.widget.webODMLocalFolder
        if os.path.isdir(localFolder):
            msg = (
                f"A WebODM folder already exists at:\n{localFolder}\n"
                "Delete it and reinstall?"
            )
            if not slicer.util.confirmYesNoDisplay(msg):
                slicer.util.infoDisplay("Using existing WebODM directory. No changes made.")
                return
            else:
                try:
                    shutil.rmtree(localFolder)
                except Exception as e:
                    slicer.util.errorDisplay(f"Failed to remove old WebODM folder:\n{str(e)}")
                    return
        os.makedirs(localFolder, exist_ok=True)

        slicer.util.infoDisplay("Pulling WebODM GPU Docker image. This can take a while...")
        try:
            process = subprocess.Popen(
                ["docker", "pull", "opendronemap/nodeodm:gpu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            for line in process.stdout:
                logging.info(line.strip())

            for line in process.stderr:
                logging.error(line.strip())

            return_code = process.wait()
            if return_code == 0:
                slicer.util.infoDisplay("WebODM (GPU) image pulled successfully.")
            else:
                slicer.util.errorDisplay(
                    f"Docker pull failed. Exit code: {return_code}. Check the log for details."
                )
        except Exception as e:
            slicer.util.errorDisplay(f"Docker pull failed: {str(e)}")

    def onRelaunchWebODMClicked(self):
        """
        Stop any container on port 3002, then launch WebODM with GPU support on 3002.
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "publish=3002", "--format", "{{.ID}}"],
                capture_output=True, text=True
            )
            container_ids = result.stdout.strip().split()
            for cid in container_ids:
                if cid:
                    slicer.util.infoDisplay(f"Stopping container {cid} on port 3002...")
                    subprocess.run(["docker", "stop", cid], check=True)
        except Exception as e:
            slicer.util.warningDisplay(f"Error stopping old container(s): {str(e)}")

        local_folder = self.widget.webODMLocalFolder
        try:
            if not os.path.isdir(local_folder):
                slicer.util.infoDisplay("Creating webODM Directory")
                os.makedirs(local_folder, exist_ok=True)

            slicer.util.infoDisplay("Launching new WebODM container on port 3002 with GPU support...")
            cmd = [
                "docker", "run", "--rm", "-d",
                "-p", "3002:3000",
                "--gpus", "all",
                "--name", "slicer-webodm-3002",
                "-v", f"{local_folder}:/webodm_data",
                "opendronemap/nodeodm:gpu"
            ]
            subprocess.run(cmd, check=True)

            slicer.util.infoDisplay("WebODM launched successfully on port 3002.")
            self.widget.nodeIPLineEdit.setText("127.0.0.1")
            self.widget.nodePortSpinBox.setValue(3002)
            slicer.app.settings().setValue("SlicerPhotogrammetry/WebODMIP", "127.0.0.1")
            slicer.app.settings().setValue("SlicerPhotogrammetry/WebODMPort", "3002")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to launch WebODM container:\n{str(e)}")

    def onRunWebODMTask(self):
        """
        Creates and runs the pyodm Task, then starts a QTimer to monitor it.
        """
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
            else:
                try:
                    params[factorName] = int(chosen_str)
                except:
                    params[factorName] = chosen_str

        shortTaskName = self.widget.generateShortTaskName("SlicerReconstruction", params)
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

    def onStopMonitoring(self):
        if self.webodmTimer:
            self.webodmTimer.stop()
            self.webodmTimer.deleteLater()
            self.webodmTimer = None
        self.webodmTask = None
        self.webodmOutDir = None
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
            self.webodmOutDir = None
            self.widget.stopMonitoringButton.setEnabled(False)
        elif info.status.name.lower() in ["failed", "canceled"]:
            self.widget.webodmLogTextEdit.append("Task failed or canceled. Stopping.")
            slicer.app.processEvents()
            if self.webodmTimer:
                self.webodmTimer.stop()
                self.webodmTimer.deleteLater()
                self.webodmTimer = None
            self.webodmTask = None
            self.webodmOutDir = None
            self.widget.stopMonitoringButton.setEnabled(False)


class SlicerPhotogrammetryLogic(ScriptedLoadableModuleLogic):
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
        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
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

    def run_sam_segmentation(self, image_rgb, bounding_box):
        if not self.predictor:
            raise RuntimeError("SAM model is not loaded.")
        import torch
        import numpy as np
        box = np.array(bounding_box, dtype=np.float32)
        with torch.no_grad():
            self.predictor.set_image(image_rgb)
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )
        return masks[0].astype(np.uint8)

    def array_to_pil(self, colorArr):
        from PIL import Image
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
