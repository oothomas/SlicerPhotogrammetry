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
     - Checking/Installing/Re-launching WebODM on port 3002 with GPU support.
     - Inclusion/Exclusion point marking for SAM.
     - NEW: A "Mask All Images In Set" workflow that removes existing masks,
       lets you place an ROI bounding box for the entire set, then finalize for all.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

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

        # default params
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

        # WebODM installation
        self.webODMCheckStatusButton = None
        self.webODMInstallButton = None
        self.webODMRelaunchButton = None
        self.webODMLocalFolder = None

        self.webODMManager = None

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

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.load_dependencies()
        self.logic = SlicerPhotogrammetryLogic()

        self.setupLogger()
        self.layout.setAlignment(qt.Qt.AlignTop)
        self.createCustomLayout()

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
        self.mainTabWidget.addTab(tab2Widget, "WebODM")

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
        savedMasterFolder = slicer.app.settings().value("SlicerPhotogrammetry/masterFolderPath", "")
        if os.path.isdir(savedMasterFolder):
            self.masterFolderSelector.directory = savedMasterFolder
        parametersFormLayout.addRow("Input Folder:", self.masterFolderSelector)

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

        # Group box for resolution selection
        resGroupBox = qt.QGroupBox("Segmentation Resolution")
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

        #
        # We replace the original row for "Mask All Images" with an HBox layout that
        # also contains the new "Finalize ROI for All" button.
        #
        maskAllLayout = qt.QHBoxLayout()
        self.maskAllImagesButton = qt.QPushButton("Mask All Images In Set")
        self.maskAllImagesButton.enabled = False
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)
        maskAllLayout.addWidget(self.maskAllImagesButton)

        # NEW button
        self.finalizeAllMaskButton = qt.QPushButton("Finalize ROI for All")
        self.finalizeAllMaskButton.enabled = False
        self.finalizeAllMaskButton.connect('clicked(bool)', self.onFinalizeAllMaskClicked)
        maskAllLayout.addWidget(self.finalizeAllMaskButton)

        parametersFormLayout.addRow("Batch Masking:", maskAllLayout)

        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        self.maskAllProgressBar.setTextVisible(True)
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        self.maskedCountLabel = qt.QLabel("Masked: 0/0")
        parametersFormLayout.addRow("Overall Progress:", self.maskedCountLabel)


        self.placeBoundingBoxButton = qt.QPushButton("Place Bounding Box")
        self.placeBoundingBoxButton.enabled = False
        parametersFormLayout.addWidget(self.placeBoundingBoxButton)
        self.placeBoundingBoxButton.connect('clicked(bool)', self.onPlaceBoundingBoxClicked)

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

        parametersFormLayout.addRow("Image Masking:", pointsButtonsLayout)

        self.maskCurrentImageButton = qt.QPushButton("Mask Current Image")
        self.maskCurrentImageButton.enabled = False
        parametersFormLayout.addWidget(self.maskCurrentImageButton)
        self.maskCurrentImageButton.connect('clicked(bool)', self.onMaskCurrentImageClicked)

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
        # (B) Manage WebODM (Install/Launch) Collapsible
        #
        manageWODMCollapsibleButton = ctk.ctkCollapsibleButton()
        manageWODMCollapsibleButton.text = "Manage WebODM (Install/Launch)"
        tab2Layout.addWidget(manageWODMCollapsibleButton)
        manageWODMFormLayout = qt.QFormLayout(manageWODMCollapsibleButton)

        self.webODMCheckStatusButton = qt.QPushButton("Check WebODM Status on port 3002")
        manageWODMFormLayout.addWidget(self.webODMCheckStatusButton)

        self.webODMInstallButton = qt.QPushButton("Install/Reinstall WebODM (GPU)")
        manageWODMFormLayout.addWidget(self.webODMInstallButton)

        self.webODMRelaunchButton = qt.QPushButton("Relaunch WebODM on Port 3002")
        manageWODMFormLayout.addWidget(self.webODMRelaunchButton)

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
        # (D) Launch WebODM Task Collapsible
        #
        webodmTaskCollapsible = ctk.ctkCollapsibleButton()
        webodmTaskCollapsible.text = "Launch WebODM Task"
        tab2Layout.addWidget(webodmTaskCollapsible)
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

        tab2Layout.addStretch(1)

        self.createMasterNodes()

        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
        self.webODMLocalFolder = os.path.join(modulePath, 'Resources', 'WebODM')

        self.webODMManager = SlicerWebODMManager(widget=self)
        self.webODMCheckStatusButton.connect('clicked(bool)', self.webODMManager.onCheckWebODMStatusClicked)
        self.webODMInstallButton.connect('clicked(bool)', self.webODMManager.onInstallWebODMClicked)
        self.webODMRelaunchButton.connect('clicked(bool)', self.webODMManager.onRelaunchWebODMClicked)
        self.stopMonitoringButton.connect('clicked(bool)', self.webODMManager.onStopMonitoring)

        # Initialize Markups nodes for Inclusions and Exclusions
        self.initializeInclusionMarkupsNode()
        self.initializeExclusionMarkupsNode()

    #
    # ---------------------------
    # CHANGES BELOW: Markups Nodes for Inclusion and Exclusion
    # ---------------------------
    #
    def initializeExclusionMarkupsNode(self):
        """Create (or retrieve) a single MarkupsFiducialNode used for all exclusion points (red)."""
        existingNode = slicer.mrmlScene.GetFirstNodeByName("ExclusionPoints")
        if existingNode and existingNode.IsA("vtkMRMLMarkupsFiducialNode"):
            self.exclusionPointNode = existingNode
        else:
            self.exclusionPointNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "ExclusionPoints"
            )
            self.exclusionPointNode.CreateDefaultDisplayNodes()

        # Make them red
        if self.exclusionPointNode.GetDisplayNode():
            self.exclusionPointNode.GetDisplayNode().SetSelectedColor(1, 0, 0)  # red
            self.exclusionPointNode.GetDisplayNode().SetColor(1, 0, 0)

        self.exclusionPointNode.SetMaximumNumberOfControlPoints(-1)

        if not self.exclusionPointAddedObserverTag:
            self.exclusionPointAddedObserverTag = self.exclusionPointNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onExclusionPointAdded
            )

    def initializeInclusionMarkupsNode(self):
        """Create (or retrieve) a single MarkupsFiducialNode used for all inclusion points (green)."""
        existingNode = slicer.mrmlScene.GetFirstNodeByName("InclusionPoints")
        if existingNode and existingNode.IsA("vtkMRMLMarkupsFiducialNode"):
            self.inclusionPointNode = existingNode
        else:
            self.inclusionPointNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "InclusionPoints"
            )
            self.inclusionPointNode.CreateDefaultDisplayNodes()

        # Make them green
        if self.inclusionPointNode.GetDisplayNode():
            self.inclusionPointNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            self.inclusionPointNode.GetDisplayNode().SetColor(0, 1, 0)

        self.inclusionPointNode.SetMaximumNumberOfControlPoints(-1)

        if not self.inclusionPointAddedObserverTag:
            self.inclusionPointAddedObserverTag = self.inclusionPointNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onInclusionPointAdded
            )

    def onExclusionPointAdded(self, caller, event):
        """
        Debug callback each time a new exclusion point is placed.
        """
        numPoints = caller.GetNumberOfControlPoints()
        logging.info(f"[ExclusionPoints Debug] A new point was added. Current total = {numPoints}.")

    def onInclusionPointAdded(self, caller, event):
        """
        Debug callback each time a new inclusion point is placed.
        """
        numPoints = caller.GetNumberOfControlPoints()
        logging.info(f"[InclusionPoints Debug] A new point was added. Current total = {numPoints}.")
    #
    # End Markups changes
    # ---------------------------
    #

    def onAddInclusionPointsClicked(self):
        """Enter place mode for the inclusion Markups node. We disable the Exclusion button while active."""
        logging.info("[InclusionPoints Debug] Entering multi-point place mode (inclusion).")

        # Stop if already in place mode for something else
        self.stopAnyActivePlacement()

        # Set up place mode for inclusion node
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.inclusionPointNode.GetID())

        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        # Now manage button states
        self.addExclusionPointsButton.enabled = False
        self.stopAddingPointsButton.enabled = True
        self.addInclusionPointsButton.enabled = False

    def onAddExclusionPointsClicked(self):
        """Enter place mode for the exclusion Markups node. We disable the Inclusion button while active."""
        logging.info("[ExclusionPoints Debug] Entering multi-point place mode (exclusion).")

        # Stop if already in place mode for something else
        self.stopAnyActivePlacement()

        # Set up place mode for exclusion node
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.exclusionPointNode.GetID())

        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        # Now manage button states
        self.addInclusionPointsButton.enabled = False
        self.stopAddingPointsButton.enabled = True
        self.addExclusionPointsButton.enabled = False

    def onStopAddingPointsClicked(self):
        """Stop place mode, restoring normal usage. Both 'Add Inclusion' and 'Add Exclusion' become enabled."""
        logging.info("[Points Debug] Stopping any place mode for inclusion/exclusion points.")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        # Re-enable both "Add Inclusion" and "Add Exclusion"
        self.addInclusionPointsButton.enabled = True
        self.addExclusionPointsButton.enabled = True
        self.stopAddingPointsButton.enabled = False

    def onClearPointsClicked(self):
        """
        Prompt the user: Clear Exclusion? Clear Inclusion? or Clear Both? or Cancel.
        Then do the appropriate clearing.
        """
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
        """
        If either inclusion or exclusion is in place mode, stop it.
        """
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode.GetCurrentInteractionMode() == interactionNode.Place:
            interactionNode.SetPlaceModePersistence(0)
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def updatePointButtons(self):
        """
        Called whenever we want to refresh the state of the Inclusion/Exclusion buttons
        based on whether the current image is in 'bbox' or not.
        """
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

    def load_dependencies(self):
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.messageBox("SlicerPhotogrammetry requires the PyTorch extension. "
                                   "Please install it from the Extensions Manager.")
        torchLogic = None
        try:
            import PyTorchUtils
            torchLogic = PyTorchUtils.PyTorchUtilsLogic()
            if not torchLogic.torchInstalled():
                logging.debug('Installing PyTorch...')
                torch = torchLogic.installTorch(askConfirmation=True, forceComputationBackend='cu118')
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

        # Check if SAM submodules can be imported
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
        self.enableMaskAllImagesIfPossible()

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

    def updateVolumeDisplay(self):
        self.imageIndexLabel.setText(f"Image {self.currentImageIndex}")
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
        # Only enable if we have at least one image with "bbox" or "masked"
        # (meaning we have some bounding region or we can do single image)
        # But the user specifically requested the old approach.
        # We'll simply enable "Mask All" if at least one image is "masked" or "bbox".
        # This matches the original approach.
        #if any(info["state"] in ["masked", "bbox"] for info in self.imageStates.values()):
        self.maskAllImagesButton.enabled = True
        #else:
            #self.maskAllImagesButton.enabled = False

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

        # Gather negative points (exclusion) from node
        negPointsFull = []
        numNeg = self.exclusionPointNode.GetNumberOfControlPoints()
        for i in range(numNeg):
            ras = [0, 0, 0]
            self.exclusionPointNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self.rasToDownsampleIJK(ras, self.masterVolumeNode)
            ptFull = self.downPointToFullPoint(ijk, self.currentSet, self.currentImageIndex)
            negPointsFull.append(ptFull)

        # Gather positive points (inclusion) from node
        posPointsFull = []
        numPos = self.inclusionPointNode.GetNumberOfControlPoints()
        for i in range(numPos):
            ras = [0, 0, 0]
            self.inclusionPointNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self.rasToDownsampleIJK(ras, self.masterVolumeNode)
            ptFull = self.downPointToFullPoint(ijk, self.currentSet, self.currentImageIndex)
            posPointsFull.append(ptFull)

        # We'll do the full-res path here
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

        # Restore normal button states
        self.restoreButtonStates()
        self.enableMaskAllImagesIfPossible()

    #
    # ----------------------------------------------------------------------
    # NEW BATCH MASKING WORKFLOW
    # ----------------------------------------------------------------------
    #
    def onMaskAllImagesClicked(self):
        """
        Now we override the old behavior.
        Instead of prompting for partial overwrites, we do the following:
         1) Ask user to confirm that all existing masks for this set will be removed.
         2) If yes, remove all bounding boxes + masks from this set (states -> none).
         3) Start a new ROI placement for the set (the user can switch images to see the ROI).
         4) All other controls are disabled except next/prev and the new "Finalize ROI for All" button.
         5) The user can reposition the ROI. They can flip through images to see if the bounding box is good enough.
         6) Once they're satisfied, they click "Finalize ROI for All" to do the actual segmentation for every image.
        """
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

        # (1) Remove all bounding boxes and mask files from disk for the entire set
        self.removeAllMasksForCurrentSet()

        # (2) Mark all images as "none"
        for idx in range(len(self.imagePaths)):
            self.imageStates[idx]["state"] = "none"
            self.imageStates[idx]["bboxCoords"] = None
            self.imageStates[idx]["maskNodes"] = None

        # (3) Update display to reflect "none" state
        self.currentImageIndex = 0
        self.updateVolumeDisplay()

        # (4) We start placing an ROI, but for the entire set.
        #     We'll create a single ROI node, let the user place/adjust it.
        self.globalMaskAllInProgress = True
        self.startPlacingROIForAllImages()

        # (5) Disable other UI except next/prev and finalizeAllMaskButton
        self.disableAllUIButNextPrevAndFinalize()

        slicer.util.infoDisplay(
            "You can now place/adjust a bounding box ROI. Then switch images with < or > to see how it fits.\n"
            "When ready, click 'Finalize ROI for All' to mask all images in the set.",
            autoCloseMsec=8000
        )

    def removeAllMasksForCurrentSet(self):
        """
        Removes any existing mask files on disk for the currently selected set
        and cleans up bounding box references in memory.
        """
        if not self.currentSet:
            return

        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        if not os.path.isdir(setOutputFolder):
            return

        # Remove any file that matches "*_mask.jpg"
        for fn in os.listdir(setOutputFolder):
            if fn.lower().endswith("_mask.jpg"):
                try:
                    os.remove(os.path.join(setOutputFolder, fn))
                except Exception as e:
                    logging.warning(f"Failed to remove mask file {fn}: {str(e)}")

    def startPlacingROIForAllImages(self):
        """Creates or re-creates the boundingBoxRoiNode so user can place it for batch masking."""
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

        """Disable everything except next/prev and the finalizeAllMaskButton."""
        for b in self.buttonsToManage:
            if b in [self.prevButton, self.nextButton]:
                b.enabled = True
            else:
                if isinstance(b, qt.QComboBox):
                    b.setEnabled(False)
                else:
                    b.enabled = False
        # The new finalize button is outside buttonsToManage, so manually set it:
        self.finalizeAllMaskButton.enabled = True

        # Also disable the point placement for single images, etc.
        self.addInclusionPointsButton.enabled = False
        self.addExclusionPointsButton.enabled = False
        self.clearPointsButton.enabled = False
        self.stopAddingPointsButton.enabled = False
        self.maskAllImagesButton.enabled = False



    def onFinalizeAllMaskClicked(self):
        """
        Once the user is satisfied with the global bounding box for all images (ROI),
        we finalize the bounding box from the *currently displayed* image?s coordinate space.
        Then we mask all images in the set using that bounding box + any inclusion/exclusion points
        the user might have placed.
        Finally, we exit the "globalMaskAllInProgress" mode and re-enable normal UI.
        """
        if not self.globalMaskAllInProgress:
            return

        if not self.boundingBoxRoiNode:
            slicer.util.warningDisplay("No ROI to finalize. Please place an ROI first.")
            return

        # 1) Finalize the bounding box from the current image (downsample dimension).
        coordsDown = self.computeBboxFromROI()
        if not coordsDown:
            slicer.util.warningDisplay("Unable to compute bounding box from ROI.")
            return

        # Hide ROI from the scene
        self.removeRoiNode()

        # 2) Collect the positive and negative points from the user
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

        # 3) Now proceed to mask all images in the set with that bounding box
        #    and these sets of points
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

        # 4) Re-enable normal UI
        self.restoreButtonStates()
        self.finalizeAllMaskButton.enabled = False
        self.globalMaskAllInProgress = False
        self.maskAllImagesButton.enabled = True

    def removeRoiNode(self):
        if self.boundingBoxRoiNode:
            dnode = self.boundingBoxRoiNode.GetDisplayNode()
            if dnode:
                dnode.SetHandlesInteractive(False)
            slicer.mrmlScene.RemoveNode(self.boundingBoxRoiNode)
            self.boundingBoxRoiNode = None

    #
    # ----------------------------------------------------------------------
    #

    def maskSingleImage(self, index, bboxDown, posPointsFull, negPointsFull, resFactor=1.0):
        import numpy as np
        import cv2

        bboxFull = self.downBboxToFullBbox(bboxDown, self.currentSet, index)
        colorArrFull = self.getFullColorArray(self.currentSet, index)

        if abs(resFactor - 1.0) < 1e-5:
            opencvFull = self.logic.pil_to_opencv(self.logic.array_to_pil(colorArrFull))
            marker_outputs = self.detect_aruco_bounding_boxes(opencvFull, aruco_dict=cv2.aruco.DICT_4X4_250)

            mask = self.logic.run_sam_segmentation_with_incl_excl(
                colorArrFull,
                bboxFull,
                posPointsFull,
                negPointsFull,
                marker_outputs
            )

        else:
            # Downsample approach
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

        from PIL import Image
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

        # remove any old mask entries from the cache so we load the updated version
        maskCacheKey = (self.currentSet, index, 'mask')
        maskCacheKeyDown = (self.currentSet, index, 'mask-down')
        if maskCacheKey in self.imageCache:
            del self.imageCache[maskCacheKey]
        if maskCacheKeyDown in self.imageCache:
            del self.imageCache[maskCacheKeyDown]

    #
    # Single bounding box logic
    #
    def onPlaceBoundingBoxClicked(self):
        self.storeCurrentButtonStates()
        # If user is in the global mask flow, we don't want to do the single bounding box logic.
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
            "Draw the ROI and use the handles to adjust it. When done, click 'Mask Current Image' to finalize + mask. "
            "You can also switch images using < or > to compare before masking.",
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
        """
        Convert a RAS coordinate to IJK in the *downsampled* space (the one used for bounding box).
        We only use X,Y from that IJK for 2D.
        """
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
            # Note: we want the inverse transformation here, because we have
            # RAS->IJK matrix. We need IJK->RAS. So let's invert that matrix:
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
        self.widget = widget
        self.webodmTask = None
        self.webodmOutDir = None
        self.webodmTimer = None
        self.lastWebODMOutputLineIndex = 0

    def onCheckWebODMStatusClicked(self):
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

    def run_sam_segmentation_with_incl_excl(self, image_rgb, bounding_box, posPoints, negPoints, marker_outputs=None):
        """
        Given a color array (H,W,3), bounding box in full coords,
        plus sets of positive 2D points (label=1) and negative 2D points (label=0),
        run the SAM predictor to produce a mask.

        If marker_outputs is given, we create multiple bounding boxes (the main one + around any ArUco markers).
        """
        if not self.predictor:
            raise RuntimeError("SAM model is not loaded.")

        import torch
        import numpy as np

        if marker_outputs is None:
            marker_outputs = []

        # Build the big bounding box set
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

        # Combine points
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
