import os
import sys
import qt
import ctk
import vtk
import slicer
import numpy as np
import torch
from slicer.ScriptedLoadableModule import *
from typing import List


#
# ??????????????????????????????????????????????????????????????????????????????
#   MODULE DECLARATION
# ??????????????????????????????????????????????????????????????????????????????
#
class SlicerPhotogrammetry(ScriptedLoadableModule):
    """
    Basic metadata for the SlicerPhotogrammetry module.
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlicerPhotogrammetry"
        self.parent.categories = ["Examples"]  # or "SlicerPhotogrammetry" if preferred
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name (Institution)"]
        self.parent.helpText = """
        This module demonstrates importing image sets, placing bounding boxes, 
        and running Segment Anything for object masking.
        """
        self.parent.acknowledgementText = """
        This file was originally developed by Your Name.
        """


#
# ??????????????????????????????????????????????????????????????????????????????
#   WIDGET
# ??????????????????????????????????????????????????????????????????????????????
#
class SlicerPhotogrammetryWidget(ScriptedLoadableModuleWidget):
    """
    Main UI for SlicerPhotogrammetry: Folders, bounding box placement,
    single/all image masking, etc.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Preemptively ensure the needed packages are installed
        try:
            import PIL
        except ImportError:
            slicer.util.pip_install("Pillow")

        try:
            import segment_anything
        except ImportError:
            slicer.util.pip_install("git+https://github.com/facebookresearch/segment-anything.git")

        # Logic handles the heavy-lifting
        self.logic = SlicerPhotogrammetryLogic()

        # Per-set data, tracking states and volumes
        self.setStates = {}            # { setName: { "imagePaths": [...], "imageStates": {...}, "originalArrays": [...], "volumeNodes": [...] } }
        self.currentSetName = None
        self.currentImageIndex = 0

        # For bounding box
        self.boundingBoxFiducialNode = None
        self.currentBboxLineNodes = []

        # Layout ID for custom layout (side-by-side slice views)
        self.customLayoutId = 1003

        # UI elements (will be created in setup())
        self.masterFolderSelector = None
        self.outputFolderSelector = None
        self.processFoldersButton = None
        self.processFoldersProgressBar = None
        self.imageSetComboBox = None
        self.placeBoundingBoxButton = None
        self.doneButton = None
        self.maskButton = None
        self.prevButton = None
        self.nextButton = None
        self.imageIndexLabel = None
        self.maskAllImagesButton = None
        self.maskAllProgressBar = None

        # Buttons and states
        self.buttonsToManage = []
        self.previousButtonStates = {}

        # Show loading dialog only on initial load (optional)
        self.loadingDialog = qt.QProgressDialog("Loading SlicerPhotogrammetry...", None, 0, 0, self.parent)
        self.loadingDialog.setWindowModality(qt.Qt.WindowModal)
        self.loadingDialog.show()

    def setup(self):
        """
        Called when the module is opened. Builds the UI layout.
        """
        super().setup()

        # Build a custom 2-slice layout
        self.createCustomLayout()

        #
        # 1) Collapsible: Import Image Sets
        #
        parametersCollapsible = ctk.ctkCollapsibleButton()
        parametersCollapsible.text = "Import Image Sets"
        self.layout.addWidget(parametersCollapsible)
        parametersFormLayout = qt.QFormLayout(parametersCollapsible)

        self.masterFolderSelector = ctk.ctkDirectoryButton()
        self.masterFolderSelector.directory = ""
        parametersFormLayout.addRow("Master Folder:", self.masterFolderSelector)

        self.outputFolderSelector = ctk.ctkDirectoryButton()
        self.outputFolderSelector.directory = ""
        parametersFormLayout.addRow("Output Folder:", self.outputFolderSelector)

        self.processFoldersButton = qt.QPushButton("Process Folders")
        parametersFormLayout.addWidget(self.processFoldersButton)
        self.processFoldersButton.connect('clicked(bool)', self.onProcessFoldersClicked)

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

        self.doneButton = qt.QPushButton("Done")
        self.doneButton.enabled = False
        parametersFormLayout.addWidget(self.doneButton)
        self.doneButton.connect('clicked(bool)', self.onDoneClicked)

        self.maskButton = qt.QPushButton("Mask Current Image")
        self.maskButton.enabled = False
        parametersFormLayout.addWidget(self.maskButton)
        self.maskButton.connect('clicked(bool)', self.onMaskClicked)

        # Navigation
        navHBox = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("<-")
        self.nextButton = qt.QPushButton("->")
        self.imageIndexLabel = qt.QLabel("[Image 0]")
        self.prevButton.enabled = False
        self.nextButton.enabled = False

        navHBox.addWidget(self.prevButton)
        navHBox.addWidget(self.imageIndexLabel)
        navHBox.addWidget(self.nextButton)
        parametersFormLayout.addRow(navHBox)

        self.prevButton.connect('clicked(bool)', self.onPrevImage)
        self.nextButton.connect('clicked(bool)', self.onNextImage)

        self.maskAllImagesButton = qt.QPushButton("Mask All Images")
        self.maskAllImagesButton.enabled = False
        parametersFormLayout.addWidget(self.maskAllImagesButton)
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)

        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        # Store references to control their states
        self.buttonsToManage = [
            self.processFoldersButton,
            self.imageSetComboBox,
            self.placeBoundingBoxButton,
            self.doneButton,
            self.maskButton,
            self.prevButton,
            self.nextButton,
            self.masterFolderSelector,
            self.outputFolderSelector,
            self.maskAllImagesButton
        ]

        self.layout.addStretch(1)

        # Loading complete (optional)
        self.loadingDialog.close()

    def createCustomLayout(self):
        """
        Creates a two-slice layout (Red + Red2) side-by-side.
        """
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
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        layoutNode.AddLayoutDescription(self.customLayoutId, customLayout)
        layoutManager.setLayout(self.customLayoutId)

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   PROCESS FOLDERS
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def onProcessFoldersClicked(self):
        """
        Enumerate subfolders in the master folder, populate combo box.
        If any progress is detected, ask the user to discard it.
        """
        if self.anySetHasProgress():
            if not slicer.util.confirmYesNoDisplay("All progress made so far will be lost. Proceed?"):
                return
            self.clearAllData()

        masterFolder = self.masterFolderSelector.directory
        outputFolder = self.outputFolderSelector.directory

        if not os.path.isdir(masterFolder):
            slicer.util.errorDisplay("Please select a valid master folder.")
            return

        if not os.path.isdir(outputFolder):
            slicer.util.errorDisplay("Please select a valid output folder.")
            return

        # Show progress bar
        self.processFoldersProgressBar.setVisible(True)
        self.processFoldersProgressBar.setValue(0)

        subfolders = [f for f in os.listdir(masterFolder)
                      if os.path.isdir(os.path.join(masterFolder, f))]
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
        """
        Checks if any set has images in 'bbox' or 'masked' states.
        """
        for setName, data in self.setStates.items():
            imageStates = data["imageStates"]
            for idx, s in imageStates.items():
                if s["state"] in ["bbox", "masked"]:
                    return True
        return False

    def clearAllData(self):
        """
        Clears all internal data and UI states, removing any loaded volumes.
        """
        self.setStates.clear()
        self.currentSetName = None
        self.currentImageIndex = 0

        self.removeAllCreatedNodes()

        # Reset UI
        self.imageSetComboBox.clear()
        self.imageSetComboBox.enabled = False
        self.placeBoundingBoxButton.enabled = False
        self.doneButton.enabled = False
        self.maskButton.enabled = False
        self.prevButton.enabled = False
        self.nextButton.enabled = False
        self.maskAllImagesButton.enabled = False
        self.imageIndexLabel.setText("[Image 0]")

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   SET SELECTION & IMAGE LOADING
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def onImageSetSelected(self, index):
        """
        Called when the user selects a new set from the combo box.
        Loads or restores images/volumes for the selected set.
        """
        if index < 0:
            return

        # Save current set to memory
        self.saveCurrentSetState()

        self.currentSetName = self.imageSetComboBox.currentText
        if self.currentSetName in self.setStates:
            # We already processed this set. Restore it.
            self.restoreSetState(self.currentSetName)
        else:
            # Fresh load
            self.loadNewSet(self.currentSetName)

    def loadNewSet(self, setName):
        """
        Reads images from the chosen folder, creates volume nodes, sets up states.
        """
        masterFolder = self.masterFolderSelector.directory
        setFolder = os.path.join(masterFolder, setName)
        imagePaths = self.logic.get_image_paths_from_folder(setFolder)
        if len(imagePaths) == 0:
            slicer.util.warningDisplay("No images found in this set.")
            return

        # Convert images to arrays (grayscale, flipped)
        originalArrays = []
        for path in imagePaths:
            arr = self.logic.loadAndFlipImageToArray(path)
            originalArrays.append(arr)

        # Create volume nodes in the scene
        volumeNodes = self.logic.createVolumeNodesFromArrays(imagePaths, originalArrays)

        # Initialize states
        imageStates = {}
        for i in range(len(imagePaths)):
            imageStates[i] = {"state": "none", "bboxCoords": None, "maskNodes": None}

        # Register this set
        self.setStates[setName] = {
            "imagePaths": imagePaths,
            "imageStates": imageStates,
            "originalArrays": originalArrays,
            "volumeNodes": volumeNodes
        }

        self.currentImageIndex = 0
        self.updateVolumeDisplay()
        self.placeBoundingBoxButton.enabled = True
        self.doneButton.enabled = False
        self.maskButton.enabled = False
        self.prevButton.enabled = (len(imagePaths) > 1)
        self.nextButton.enabled = (len(imagePaths) > 1)
        self.maskAllImagesButton.enabled = False

    def saveCurrentSetState(self):
        """
        Removes volume nodes from the scene but keeps them in memory.
        (So if user switches sets, we don't keep everything loaded.)
        """
        if not self.currentSetName or self.currentSetName not in self.setStates:
            return

        data = self.setStates[self.currentSetName]

        # Remove volumes from scene
        for volNode in data.get("volumeNodes", []):
            if volNode and slicer.mrmlScene.IsNodePresent(volNode):
                slicer.mrmlScene.RemoveNode(volNode)
        data["volumeNodes"] = []

        # Remove any mask volume nodes
        for idx, stateInfo in data["imageStates"].items():
            if stateInfo["state"] == "masked" and stateInfo["maskNodes"]:
                self.logic.removeMaskVolumeNodes(stateInfo["maskNodes"])
                stateInfo["maskNodes"] = None

        # Remove bounding box lines, fiducials
        self.removeAllCreatedNodes()

    def restoreSetState(self, setName):
        """
        Re-creates volume nodes for the chosen set, reattaches masked volumes.
        """
        data = self.setStates[setName]
        imagePaths = data["imagePaths"]
        originalArrays = data["originalArrays"]
        imageStates = data["imageStates"]

        # Create volumes again
        data["volumeNodes"] = self.logic.createVolumeNodesFromArrays(imagePaths, originalArrays)

        # Re-create mask volumes if needed
        for idx, s in imageStates.items():
            if s["state"] == "masked" and s["maskNodes"]:
                self.logic.restoreMaskVolumeNodes(s["maskNodes"])

        self.currentImageIndex = 0
        self.updateVolumeDisplay()
        self.placeBoundingBoxButton.enabled = True
        self.refreshButtonStatesBasedOnCurrentState()

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   NAVIGATION & VOLUME DISPLAY
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def refreshButtonStatesBasedOnCurrentState(self):
        data = self.setStates[self.currentSetName]
        s = data["imageStates"][self.currentImageIndex]
        st = s["state"]

        # The "Done" button only shows if we're placing a bounding box (but haven't placed both points).
        self.doneButton.enabled = (st == "bbox")
        self.maskButton.enabled = (st == "bbox")
        self.enableMaskAllImagesIfPossible()

    def updateVolumeDisplay(self):
        """
        Updates the slice viewers (Red, Red2) to show the appropriate volumes
        (original, label, masked, etc.) for the current image.
        """
        if not self.currentSetName:
            return

        data = self.setStates[self.currentSetName]
        imagePaths = data["imagePaths"]
        volumeNodes = data["volumeNodes"]
        imageStates = data["imageStates"]

        if self.currentImageIndex < 0 or self.currentImageIndex >= len(volumeNodes):
            return

        self.imageIndexLabel.setText(f"[Image {self.currentImageIndex}]")

        # Clear bounding box lines
        self.removeBoundingBoxLines()

        currentVol = volumeNodes[self.currentImageIndex]
        stateInfo = imageStates[self.currentImageIndex]
        st = stateInfo["state"]

        if st == "none":
            # Show original volume only
            self.logic.showOriginalInSliceViews(currentVol, sliceNames=["Red", "Red2"])
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.enableMaskAllImagesIfPossible()
        elif st == "bbox":
            # Show original + bounding box lines
            self.logic.showOriginalInSliceViews(currentVol, sliceNames=["Red", "Red2"])
            bboxCoords = stateInfo["bboxCoords"]
            if bboxCoords:
                self.drawBoundingBoxLines(bboxCoords)
            self.doneButton.enabled = False  # will be re-enabled after user places points
            self.maskButton.enabled = True
            self.enableMaskAllImagesIfPossible()
        elif st == "masked":
            # Show original + label overlay in Red, masked volume in Red2
            maskNodes = stateInfo["maskNodes"]
            self.logic.showMaskedInSliceViews(currentVol, maskNodes, sliceNames=["Red", "Red2"])
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.enableMaskAllImagesIfPossible()

    def enableMaskAllImagesIfPossible(self):
        """
        "Mask All Images" is enabled only if the current image is fully masked and has bbox coords.
        """
        if not self.currentSetName or self.currentSetName not in self.setStates:
            return
        data = self.setStates[self.currentSetName]
        s = data["imageStates"][self.currentImageIndex]
        if s["state"] == "masked" and s["bboxCoords"] is not None:
            self.maskAllImagesButton.enabled = True
        else:
            self.maskAllImagesButton.enabled = False

    def onPrevImage(self):
        if not self.currentSetName:
            return
        if self.currentImageIndex > 0:
            self.currentImageIndex -= 1
            self.updateVolumeDisplay()

    def onNextImage(self):
        if not self.currentSetName:
            return
        data = self.setStates[self.currentSetName]
        if self.currentImageIndex < len(data["volumeNodes"]) - 1:
            self.currentImageIndex += 1
            self.updateVolumeDisplay()

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   BOUNDING BOX
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def onPlaceBoundingBoxClicked(self):
        """
        Switches to "place bounding box" mode. If we already have a box or a mask,
        the user is prompted whether to overwrite.
        """
        data = self.setStates[self.currentSetName]
        s = data["imageStates"][self.currentImageIndex]
        st = s["state"]

        self.storeCurrentButtonStates()
        self.disableAllButtonsExcept(self.doneButton)

        if st == "masked":
            if not slicer.util.confirmYesNoDisplay(
                "This image is already masked. Creating a new bounding box will remove the existing mask. Proceed?"
            ):
                self.restoreButtonStates()
                return
            self.logic.removeMaskVolumeNodes(s["maskNodes"])
            s["maskNodes"] = None
            s["state"] = "none"
            s["bboxCoords"] = None

        elif st == "bbox":
            if not slicer.util.confirmYesNoDisplay(
                "A bounding box already exists. Creating a new one will remove it. Proceed?"
            ):
                self.restoreButtonStates()
                return
            s["state"] = "none"
            s["bboxCoords"] = None
            s["maskNodes"] = None
            self.removeBoundingBoxLines()

        # Start placing bounding box points
        self.startPlacingPoints()

    def startPlacingPoints(self):
        """
        Sets up a fiducial node and changes interaction to "place" mode.
        """
        # Clean up existing lines/fiducials
        self.removeBoundingBoxLines()
        if self.boundingBoxFiducialNode:
            slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode)
            self.boundingBoxFiducialNode = None

        self.boundingBoxFiducialNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "BoundingBoxPoints"
        )
        self.boundingBoxFiducialNode.CreateDefaultDisplayNodes()
        dnode = self.boundingBoxFiducialNode.GetDisplayNode()
        dnode.SetGlyphTypeFromString("Sphere3D")
        dnode.SetUseGlyphScale(False)
        dnode.SetGlyphSize(5.0)
        dnode.SetSelectedColor(1, 0, 0)
        dnode.SetVisibility(True)
        dnode.SetPointLabelsVisibility(True)

        self.boundingBoxFiducialNode.RemoveAllControlPoints()
        self.boundingBoxFiducialNode.RemoveAllObservers()
        self.boundingBoxFiducialNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.onBoundingBoxPointPlaced
        )

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.boundingBoxFiducialNode.GetID())
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self.doneButton.enabled = False
        self.maskButton.enabled = False

    def onBoundingBoxPointPlaced(self, caller, event):
        """
        When user places a fiducial. If we've placed two points, we can finalize.
        """
        if not self.boundingBoxFiducialNode:
            return
        nPoints = self.boundingBoxFiducialNode.GetNumberOfDefinedControlPoints()
        if nPoints == 2:
            slicer.util.infoDisplay("Two points placed. Click 'Done' to confirm bounding box.")
            self.disablePointPlacement()
            self.doneButton.enabled = True

    def disablePointPlacement(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        interactionNode.SetPlaceModePersistence(0)

    def onDoneClicked(self):
        """
        Finalizes bounding box: compute coords in IJK, update state, remove fiducials.
        """
        if not self.boundingBoxFiducialNode or \
           self.boundingBoxFiducialNode.GetNumberOfDefinedControlPoints() < 2:
            slicer.util.warningDisplay("You must place two points first.")
            return

        data = self.setStates[self.currentSetName]
        s = data["imageStates"][self.currentImageIndex]

        # Compute bounding box
        coords = self.logic.computeBoundingBoxFromFiducials(
            self.boundingBoxFiducialNode,
            data["volumeNodes"][self.currentImageIndex]
        )
        s["state"] = "bbox"
        s["bboxCoords"] = coords
        s["maskNodes"] = None

        # Remove fiducial node
        if self.boundingBoxFiducialNode.GetDisplayNode():
            slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode.GetDisplayNode())
        slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode)
        self.boundingBoxFiducialNode = None

        self.doneButton.enabled = False
        self.maskButton.enabled = True
        self.updateVolumeDisplay()

        # Re-enable UI
        self.restoreButtonStates()

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   MASKING (SINGLE IMAGE & ALL IMAGES)
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def onMaskClicked(self):
        """
        Run SAM on the current image using its bounding box, create masked volumes.
        """
        data = self.setStates[self.currentSetName]
        s = data["imageStates"][self.currentImageIndex]
        if s["state"] != "bbox":
            slicer.util.warningDisplay("No bounding box defined for this image.")
            return

        # Convert current volume to 3-channel
        currentVol = data["volumeNodes"][self.currentImageIndex]
        rgbArr = self.logic.volumeToRgbNumpy(currentVol)

        mask = self.logic.run_sam_segmentation(rgbArr, s["bboxCoords"])
        maskNodes = self.logic.createMaskVolumeNodes(currentVol, mask)

        s["state"] = "masked"
        s["maskNodes"] = maskNodes

        # Save masked result to disk
        self.logic.saveMaskedImage(
            setName=self.currentSetName,
            index=self.currentImageIndex,
            imagePaths=data["imagePaths"],
            outputFolder=self.outputFolderSelector.directory,
            grayMasked=maskNodes["grayMasked"]
        )

        # Update display
        self.removeBoundingBoxLines()
        self.updateVolumeDisplay()

    def onMaskAllImagesClicked(self):
        """
        Use the bounding box from the current masked image and apply SAM
        to all other unmasked images in this set.
        """
        data = self.setStates[self.currentSetName]
        s = data["imageStates"][self.currentImageIndex]
        if s["state"] != "masked" or s["bboxCoords"] is None:
            slicer.util.warningDisplay("Current image is not masked or has no bounding box info.")
            return

        bboxCoords = s["bboxCoords"]
        self.maskAllProgressBar.setVisible(True)
        indicesToMask = [
            i for i in range(len(data["imagePaths"]))
            if i != self.currentImageIndex and data["imageStates"][i]["state"] != "masked"
        ]
        self.maskAllProgressBar.setRange(0, len(indicesToMask))
        self.maskAllProgressBar.setValue(0)

        for count, i in enumerate(indicesToMask):
            rgbArr = self.logic.volumeToRgbNumpy(data["volumeNodes"][i])
            mask = self.logic.run_sam_segmentation(rgbArr, bboxCoords)
            maskNodes = self.logic.createMaskVolumeNodes(data["volumeNodes"][i], mask)

            data["imageStates"][i]["state"] = "masked"
            data["imageStates"][i]["bboxCoords"] = bboxCoords
            data["imageStates"][i]["maskNodes"] = maskNodes

            self.logic.saveMaskedImage(
                setName=self.currentSetName,
                index=i,
                imagePaths=data["imagePaths"],
                outputFolder=self.outputFolderSelector.directory,
                grayMasked=maskNodes["grayMasked"]
            )

            self.maskAllProgressBar.setValue(count + 1)
            slicer.app.processEvents()

        slicer.util.infoDisplay("All images masked and saved.")
        self.maskAllProgressBar.setVisible(False)
        self.updateVolumeDisplay()

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   BOUNDING BOX LINES & NODE MANAGEMENT
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def drawBoundingBoxLines(self, bboxCoords):
        """
        Draws line nodes for the bounding box in RAS space.
        """
        if not bboxCoords:
            return
        data = self.setStates[self.currentSetName]
        volumeNode = data["volumeNodes"][self.currentImageIndex]
        x_min, y_min, x_max, y_max = bboxCoords

        ijkToRas = self.logic.ijkToRasFunc(volumeNode)
        p1 = ijkToRas(x_min, y_min)
        p2 = ijkToRas(x_max, y_min)
        p3 = ijkToRas(x_max, y_max)
        p4 = ijkToRas(x_min, y_max)

        lineEndpoints = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        for (start, end) in lineEndpoints:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            lineNode.AddControlPoint(start)
            lineNode.AddControlPoint(end)

            # Hide line measurement
            lengthMeas = lineNode.GetMeasurement("length")
            if lengthMeas:
                lengthMeas.SetEnabled(False)

            dnode = lineNode.GetDisplayNode()
            dnode.SetLineThickness(0.25)
            dnode.SetSelectedColor(1, 1, 0)
            dnode.SetPointLabelsVisibility(False)
            dnode.SetPropertiesLabelVisibility(False)
            dnode.SetTextScale(0)

            self.currentBboxLineNodes.append(lineNode)

    def removeBoundingBoxLines(self):
        """
        Removes any bounding box line nodes from the scene.
        """
        for ln in self.currentBboxLineNodes:
            if ln and slicer.mrmlScene.IsNodePresent(ln):
                slicer.mrmlScene.RemoveNode(ln)
        self.currentBboxLineNodes = []

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   BUTTON STATES
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def storeCurrentButtonStates(self):
        self.previousButtonStates = {}
        for btn in self.buttonsToManage:
            if isinstance(btn, qt.QComboBox):
                self.previousButtonStates[btn] = btn.isEnabled()
            else:
                self.previousButtonStates[btn] = btn.enabled

    def restoreButtonStates(self):
        for btn, state in self.previousButtonStates.items():
            if isinstance(btn, qt.QComboBox):
                btn.setEnabled(state)
            else:
                btn.enabled = state

    def disableAllButtonsExcept(self, allowedWidget):
        self.storeCurrentButtonStates()
        for btn in self.buttonsToManage:
            if btn != allowedWidget:
                if isinstance(btn, qt.QComboBox):
                    btn.setEnabled(False)
                else:
                    btn.enabled = False

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   NODE CLEANUP
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def removeAllCreatedNodes(self):
        """
        Removes bounding box lines and fiducial nodes from the scene for cleanup.
        """
        self.removeBoundingBoxLines()
        if self.boundingBoxFiducialNode and slicer.mrmlScene.IsNodePresent(self.boundingBoxFiducialNode):
            slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode)
        self.boundingBoxFiducialNode = None

    def cleanup(self):
        """
        Called when closing the module. Saves states and removes nodes.
        """
        self.saveCurrentSetState()
        self.removeAllCreatedNodes()


#
# ??????????????????????????????????????????????????????????????????????????????
#   LOGIC
# ??????????????????????????????????????????????????????????????????????????????
#
class SlicerPhotogrammetryLogic(ScriptedLoadableModuleLogic):
    """
    Encapsulates most of the data-handling logic:
      - Image loading & flipping
      - Volume node creation
      - Bounding box computations
      - Segment Anything
      - Mask volumes creation & removal
      - Saving masked images
    """

    def __init__(self):
        super().__init__()

        # Attempt imports (PIL, segment-anything)
        self.ensurePILInstalled()
        self.ensureSAMInstalled()
        from PIL import Image
        from segment_anything import sam_model_registry, SamPredictor

        # Check or download SAM weights
        weights_filename = "sam_vit_h_4b8939.pth"
        sam_checkpoint = self.checkAndDownloadWeights(weights_filename)

        # Initialize SAM
        self.device = "cpu"
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   PACKAGE INSTALLATION CHECKS
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def ensurePILInstalled(self):
        try:
            import PIL
        except ImportError:
            slicer.util.pip_install("Pillow")

    def ensureSAMInstalled(self):
        try:
            import segment_anything
        except ImportError:
            slicer.util.pip_install("git+https://github.com/facebookresearch/segment-anything.git")

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   WEIGHTS
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def checkAndDownloadWeights(self, filename):
        """
        Ensures SAM weights are present in the module's Resources folder, or downloads them.
        """
        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
        resourcePath = os.path.join(modulePath, 'Resources', filename)

        if not os.path.isfile(resourcePath):
            weights_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            slicer.util.infoDisplay(f"Downloading {filename}. This may take a few minutes...", autoCloseMsec=2000)
            try:
                slicer.util.downloadFile(weights_url, resourcePath)
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to download {filename}: {str(e)}")
                raise RuntimeError("Could not download SAM weights.")

        return resourcePath

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   IMAGE FILE I/O
    # ??????????????????????????????????????????????????????????????????????????????
    #
    @staticmethod
    def get_image_paths_from_folder(folder_path: str,
                                    extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]) -> List[str]:
        """
        Returns a sorted list of image paths in the specified folder.
        """
        folder_path = os.path.abspath(folder_path)
        image_paths = []
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    fp = os.path.join(folder_path, filename)
                    if os.path.isfile(fp):
                        image_paths.append(fp)
        return sorted(image_paths)

    @staticmethod
    def loadAndFlipImageToArray(path):
        """
        Loads image in grayscale, flips vertically to match Slicer's coordinate convention.
        """
        from PIL import Image
        img = Image.open(path).convert('RGB')
        arr = np.asarray(img)
        arr = np.flipud(arr)  # vertical flip
        gray = np.mean(arr, axis=2).astype(np.uint8)
        return gray

    def createVolumeNodesFromArrays(self, filePaths, arrays):
        """
        Given a list of file paths (for naming) and a list of numpy arrays,
        creates scalar volume nodes in the scene.
        """
        volumeNodes = []
        for p, arr in zip(filePaths, arrays):
            vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.util.updateVolumeFromArray(vol, arr)
            vol.SetName(os.path.basename(p))
            volumeNodes.append(vol)
        return volumeNodes

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   BOUNDING BOX & COORDINATES
    # ??????????????????????????????????????????????????????????????????????????????
    #
    @staticmethod
    def computeBoundingBoxFromFiducials(fiducialNode, volumeNode):
        """
        For two fiducials in RAS, compute bounding box in IJK: (x_min, y_min, x_max, y_max).
        """
        if fiducialNode.GetNumberOfDefinedControlPoints() < 2:
            return None

        p1_ras = [0, 0, 0]
        p2_ras = [0, 0, 0]
        fiducialNode.GetNthControlPointPositionWorld(0, p1_ras)
        fiducialNode.GetNthControlPointPositionWorld(1, p2_ras)

        rasToIjkMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjkMatrix)

        def rasToIjk(ras):
            ras4 = [ras[0], ras[1], ras[2], 1.0]
            ijk4 = rasToIjkMatrix.MultiplyPoint(ras4)
            return [int(round(ijk4[0])), int(round(ijk4[1])), int(round(ijk4[2]))]

        p1_ijk = rasToIjk(p1_ras)
        p2_ijk = rasToIjk(p2_ras)
        x_min = min(p1_ijk[0], p2_ijk[0])
        x_max = max(p1_ijk[0], p2_ijk[0])
        y_min = min(p1_ijk[1], p2_ijk[1])
        y_max = max(p1_ijk[1], p2_ijk[1])
        return (x_min, y_min, x_max, y_max)

    @staticmethod
    def ijkToRasFunc(volumeNode):
        """
        Returns a closure that converts (i,j) to RAS for the given volume.
        """
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRasMatrix)

        def ijkToRas(i, j):
            ras4 = ijkToRasMatrix.MultiplyPoint([i, j, 0, 1])
            return [ras4[0], ras4[1], ras4[2]]
        return ijkToRas

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   SEGMENT ANYTHING
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def run_sam_segmentation(self, image_rgb, bounding_box):
        """
        Given a 3-channel image (H,W,3) and an IJK bounding_box=(x_min,y_min,x_max,y_max),
        run SAM to get the mask.
        """
        self.predictor.set_image(image_rgb)
        box = np.array(bounding_box, dtype=np.float32)
        with torch.no_grad():
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )
        mask = masks[0].astype(np.uint8)
        return mask

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   SHOWING VOLUMES IN SLICE VIEWS
    # ??????????????????????????????????????????????????????????????????????????????
    #
    @staticmethod
    def showOriginalInSliceViews(volumeNode, sliceNames=["Red"]):
        layoutManager = slicer.app.layoutManager()
        for sn in sliceNames:
            sw = layoutManager.sliceWidget(sn)
            compNode = sw.sliceLogic().GetSliceCompositeNode()
            compNode.SetBackgroundVolumeID(volumeNode.GetID())
            compNode.SetForegroundVolumeID(None)
            compNode.SetLabelVolumeID(None)
            sw.sliceLogic().FitSliceToAll()

    @staticmethod
    def showMaskedInSliceViews(originalVol, maskNodes, sliceNames=["Red"]):
        """
        Red: background = originalVol, label = maskNodes["labelVol"] (50% opacity)
        Red2: background = maskNodes["maskedVol"]
        """
        if not maskNodes:
            return
        if len(sliceNames) == 0:
            return

        layoutManager = slicer.app.layoutManager()

        # First slice
        redComposite = layoutManager.sliceWidget(sliceNames[0]).sliceLogic().GetSliceCompositeNode()
        redComposite.SetBackgroundVolumeID(originalVol.GetID())
        redComposite.SetForegroundVolumeID(None)
        redComposite.SetLabelVolumeID(maskNodes["labelVol"].GetID())
        redComposite.SetLabelOpacity(0.5)
        layoutManager.sliceWidget(sliceNames[0]).sliceLogic().FitSliceToAll()

        # Second slice (if it exists)
        if len(sliceNames) > 1:
            red2Composite = layoutManager.sliceWidget(sliceNames[1]).sliceLogic().GetSliceCompositeNode()
            red2Composite.SetBackgroundVolumeID(maskNodes["maskedVol"].GetID())
            red2Composite.SetForegroundVolumeID(None)
            red2Composite.SetLabelVolumeID(None)
            layoutManager.sliceWidget(sliceNames[1]).sliceLogic().FitSliceToAll()

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   CREATING / REMOVING MASK VOLUMES
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def createMaskVolumeNodes(self, originalVol, mask):
        """
        Creates labelVol + colorNode + maskedVol from a single-channel mask.
        Returns dict of references.
        """
        # Convert originalVol to 3-channel
        rgbArr = self.volumeToRgbNumpy(originalVol)

        # Create label map
        labelVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        labelVol.CreateDefaultDisplayNodes()
        labelArray = (mask > 0).astype(np.uint8)
        slicer.util.updateVolumeFromArray(labelVol, labelArray)

        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(2)
        colorNode.SetColor(0, "Background", 0, 0, 0, 0)
        colorNode.SetColor(1, "Mask", 1, 0, 0, 1)
        labelVol.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

        # Create masked volume (grayscale)
        maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        maskedArr = rgbArr.copy()
        maskedArr[mask == 0] = 0
        grayMasked = np.mean(maskedArr, axis=2).astype(np.uint8)
        slicer.util.updateVolumeFromArray(maskedVol, grayMasked)
        maskedVol.SetName(originalVol.GetName() + "_masked")

        return {
            "labelVol": labelVol,
            "colorNode": colorNode,
            "maskedVol": maskedVol,
            "labelArray": labelArray,
            "grayMasked": grayMasked
        }

    @staticmethod
    def removeMaskVolumeNodes(maskNodes):
        """
        Removes labelVol, colorNode, maskedVol from the scene.
        """
        for key in ["labelVol", "colorNode", "maskedVol"]:
            node = maskNodes.get(key)
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   HELPER: CONVERT VOLUME TO RGB
    # ??????????????????????????????????????????????????????????????????????????????
    #
    @staticmethod
    def volumeToRgbNumpy(volumeNode):
        """
        Returns (H,W,3) uint8 array from a single-channel volume.
        """
        arr = slicer.util.arrayFromVolume(volumeNode)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            # Make 3 channels
            rgbArr = np.stack([arr, arr, arr], axis=-1)
        else:
            rgbArr = arr
        return rgbArr.astype(np.uint8)

    #
    # ??????????????????????????????????????????????????????????????????????????????
    #   SAVING MASKED IMAGES
    # ??????????????????????????????????????????????????????????????????????????????
    #
    def saveMaskedImage(self, setName, index, imagePaths, outputFolder, grayMasked):
        """
        Save the masked (grayscale) image to disk, flipping it back upright.
        """
        from PIL import Image
        os.makedirs(os.path.join(outputFolder, setName), exist_ok=True)
        filename = os.path.basename(imagePaths[index])

        # Flip it back up
        arr = np.flipud(grayMasked)
        arr_rgb = np.stack([arr, arr, arr], axis=-1)  # keep it 3-channel for easy viewing

        outPath = os.path.join(outputFolder, setName, filename)
        Image.fromarray(arr_rgb.astype(np.uint8)).save(outPath)
