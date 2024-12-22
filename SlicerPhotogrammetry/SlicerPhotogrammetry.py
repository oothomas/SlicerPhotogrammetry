import os
import sys
import qt
import ctk
import vtk
import slicer
import shutil
import numpy as np
import torch
from slicer.ScriptedLoadableModule import *
from typing import List


def getWeights(filename):
    modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
    resourcePath = os.path.join(modulePath, 'Resources', filename)
    return resourcePath


class SlicerPhotogrammetry(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlicerPhotogrammetry"
        self.parent.categories = ["SlicerPhotogrammetry"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name"]
        self.parent.helpText = """NA"""
        self.parent.acknowledgementText = """NA"""


class SlicerPhotogrammetryWidget(ScriptedLoadableModuleWidget):
    try:
        from PIL import Image
    except ImportError:
        slicer.util.pip_install('Pillow')
        from PIL import Image

    try:
        import segment_anything
    except ImportError:
        slicer.util.pip_install("git+https://github.com/facebookresearch/segment-anything.git")
        import segment_anything
    from segment_anything import sam_model_registry, SamPredictor

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = SlicerPhotogrammetryLogic()

        self.setStates = {}  # per-set states
        self.currentSet = None
        self.imagePaths = []
        self.currentImageIndex = 0
        self.originalVolumes = []
        self.processedOnce = False
        self.layoutId = 1003

        self.imageStates = {}
        self.createdNodes = []
        self.currentBboxLineNodes = []
        self.boundingBoxFiducialNode = None
        self.placingBoundingBox = False

        self.buttonsToManage = []
        self.prevButton = None
        self.nextButton = None
        self.maskButton = None
        self.maskAllImagesButton = None
        self.processButton = None
        self.imageSetComboBox = None
        self.placeBoundingBoxButton = None
        self.outputFolderSelector = None
        self.masterFolderSelector = None
        self.previousButtonStates = {}

        # Show loading dialog only on initial load
        self.loadingDialog = qt.QProgressDialog("Loading SlicerPhotogrammetry...", None, 0, 0, self.parent)
        self.loadingDialog.setWindowModality(qt.Qt.WindowModal)
        self.loadingDialog.show()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.setAlignment(qt.Qt.AlignTop)

        self.createCustomLayout()

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Import Image Sets"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.masterFolderSelector = ctk.ctkDirectoryButton()
        self.masterFolderSelector.directory = ""
        parametersFormLayout.addRow("Master Folder:", self.masterFolderSelector)

        self.outputFolderSelector = ctk.ctkDirectoryButton()
        self.outputFolderSelector.directory = ""
        parametersFormLayout.addRow("Output Folder:", self.outputFolderSelector)

        self.processButton = qt.QPushButton("Process Folders")
        parametersFormLayout.addWidget(self.processButton)
        self.processButton.connect('clicked(bool)', self.onProcessFoldersClicked)

        # Add a progress bar for Process Folders
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

        navigationLayout = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("<-")
        self.nextButton = qt.QPushButton("->")
        self.imageIndexLabel = qt.QLabel("[Image 0]")
        self.prevButton.enabled = False
        self.nextButton.enabled = False
        navigationLayout.addWidget(self.prevButton)
        navigationLayout.addWidget(self.imageIndexLabel)
        navigationLayout.addWidget(self.nextButton)
        parametersFormLayout.addRow(navigationLayout)
        self.prevButton.connect('clicked(bool)', self.onPrevImage)
        self.nextButton.connect('clicked(bool)', self.onNextImage)

        self.maskAllImagesButton = qt.QPushButton("Mask All Images")
        self.maskAllImagesButton.enabled = False
        parametersFormLayout.addWidget(self.maskAllImagesButton)
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)

        # Add a progress bar for Mask All Images
        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        self.layout.addStretch(1)

        # Store references to all buttons we will manage
        self.buttonsToManage = [
            self.processButton,
            self.imageSetComboBox,
            self.placeBoundingBoxButton,
            self.maskButton,
            self.maskAllImagesButton,
            self.prevButton,
            self.nextButton,
            self.masterFolderSelector,
            self.outputFolderSelector
        ]

        # Loading complete, hide dialog
        self.loadingDialog.close()

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
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        layoutNode.AddLayoutDescription(self.layoutId, customLayout)
        layoutManager.setLayout(self.layoutId)

    def onProcessFoldersClicked(self):
        # Check if any set has progress made
        if self.anySetHasProgress():
            # Prompt user that all progress will be lost
            if not slicer.util.confirmYesNoDisplay("All progress made so far will be lost. Proceed?"):
                return
            # If yes, clear all data
            self.clearAllData()

        masterFolderPath = self.masterFolderSelector.directory
        outputFolderPath = self.outputFolderSelector.directory
        if not os.path.isdir(masterFolderPath):
            slicer.util.errorDisplay("Please select a valid master folder.")
            return
        if not os.path.isdir(outputFolderPath):
            slicer.util.errorDisplay("Please select a valid output folder.")
            return

        # Show process folders progress bar
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
        # Check if any image in any set is in "bbox" or "masked"
        for setName, setData in self.setStates.items():
            for idx, stateInfo in setData["imageStates"].items():
                if stateInfo["state"] in ["bbox", "masked"]:
                    return True
        return False

    def clearAllData(self):
        # Clear self.setStates and unload any currently loaded set
        self.setStates = {}
        self.currentSet = None
        self.clearAllCreatedNodes()
        self.originalVolumes = []
        self.imageStates = {}
        self.imagePaths = []
        self.currentImageIndex = 0
        self.imageSetComboBox.clear()
        self.imageSetComboBox.enabled = False
        # Reset UI states
        self.placeBoundingBoxButton.enabled = False
        self.doneButton.enabled = False
        self.maskButton.enabled = False
        self.maskAllImagesButton.enabled = False
        self.prevButton.enabled = False
        self.nextButton.enabled = False
        self.imageIndexLabel.setText("[Image 0]")

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

            self.clearAllCreatedNodes()
            self.originalVolumes = []
            self.imageStates = {}

            originalArrays = []
            for i, path in enumerate(self.imagePaths):
                arr = self.loadAndFlipImageToArray(path)
                originalArrays.append(arr)
                self.imageStates[i] = {
                    "state": "none",
                    "bboxCoords": None,
                    "maskNodes": None
                }

            self.currentImageIndex = 0
            self.createVolumesFromArrays(originalArrays)
            self.setStates[self.currentSet] = {
                "imagePaths": self.imagePaths,
                "imageStates": self.imageStates,
                "originalArrays": originalArrays,
            }

            self.updateVolumeDisplay()
            self.placeBoundingBoxButton.enabled = True
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.maskAllImagesButton.enabled = False
            self.prevButton.enabled = (len(self.imagePaths) > 1)
            self.nextButton.enabled = (len(self.imagePaths) > 1)

    def saveCurrentSetState(self):
        if self.currentSet is None or self.currentSet not in self.setStates:
            return

        self.setStates[self.currentSet]["imageStates"] = self.imageStates

        # Remove volumes from scene
        for vol in self.originalVolumes:
            if vol and slicer.mrmlScene.IsNodePresent(vol):
                slicer.mrmlScene.RemoveNode(vol)
        self.originalVolumes = []
        for idx, stateInfo in self.imageStates.items():
            if stateInfo["state"] == "masked" and stateInfo["maskNodes"] is not None:
                m = stateInfo["maskNodes"]
                for nodeKey in ["labelVol", "colorNode", "maskedVol"]:
                    node = m[nodeKey]
                    if node and slicer.mrmlScene.IsNodePresent(node):
                        slicer.mrmlScene.RemoveNode(node)
        self.clearAllCreatedNodes()

    def restoreSetState(self, setName):
        setData = self.setStates[setName]
        self.imagePaths = setData["imagePaths"]
        self.imageStates = setData["imageStates"]
        originalArrays = setData["originalArrays"]

        self.createVolumesFromArrays(originalArrays)

        for idx, stateInfo in self.imageStates.items():
            if stateInfo["state"] == "masked" and stateInfo["maskNodes"] is not None:
                m = stateInfo["maskNodes"]
                labelArray = m["labelArray"]
                grayMasked = m["grayMasked"]

                labelVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                labelVol.CreateDefaultDisplayNodes()
                self.createdNodes.append(labelVol)
                slicer.util.updateVolumeFromArray(labelVol, labelArray)

                colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
                self.createdNodes.append(colorNode)
                colorNode.SetTypeToUser()
                colorNode.SetNumberOfColors(2)
                colorNode.SetColor(0, "Background", 0, 0, 0, 0)
                colorNode.SetColor(1, "Mask", 1, 0, 0, 1)
                labelVol.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

                maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
                self.createdNodes.append(maskedVol)
                slicer.util.updateVolumeFromArray(maskedVol, grayMasked)
                maskedVol.SetName(os.path.basename(self.imagePaths[idx]) + "_masked")

                stateInfo["maskNodes"] = {
                    "labelVol": labelVol,
                    "colorNode": colorNode,
                    "maskedVol": maskedVol,
                    "labelArray": labelArray,
                    "grayMasked": grayMasked
                }

        self.currentImageIndex = 0
        self.updateVolumeDisplay()
        self.placeBoundingBoxButton.enabled = True
        self.refreshButtonStatesBasedOnCurrentState()

    def createVolumesFromArrays(self, originalArrays):
        self.originalVolumes = []
        for i, arr in enumerate(originalArrays):
            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            self.createdNodes.append(volumeNode)
            slicer.util.updateVolumeFromArray(volumeNode, arr)
            volumeNode.SetName(os.path.basename(self.imagePaths[i]))
            self.originalVolumes.append(volumeNode)

    def loadAndFlipImageToArray(self, path):
        from PIL import Image
        img = Image.open(path).convert('RGB')
        arr = np.asarray(img)
        arr = np.flipud(arr)
        gray = np.mean(arr, axis=2).astype(np.uint8)
        return gray

    def refreshButtonStatesBasedOnCurrentState(self):
        stateInfo = self.imageStates[self.currentImageIndex]
        state = stateInfo["state"]
        if state == "none":
            self.doneButton.enabled = False
            self.maskButton.enabled = False
        elif state == "bbox":
            self.doneButton.enabled = False
            self.maskButton.enabled = True
        elif state == "masked":
            self.doneButton.enabled = False
            self.maskButton.enabled = False
        self.enableMaskAllImagesIfPossible()

    def updateVolumeDisplay(self):
        self.imageIndexLabel.setText(f"[Image {self.currentImageIndex}]")
        currentOriginal = self.originalVolumes[self.currentImageIndex]

        self.removeBboxLines()

        stateInfo = self.imageStates[self.currentImageIndex]
        state = stateInfo["state"]

        if state == "none":
            self.showOriginalOnly(currentOriginal)
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.enableMaskAllImagesIfPossible()
        elif state == "bbox":
            self.showOriginalOnly(currentOriginal)
            self.drawBboxLines(stateInfo["bboxCoords"])
            self.doneButton.enabled = False
            self.maskButton.enabled = True
            self.enableMaskAllImagesIfPossible()
        elif state == "masked":
            self.showMaskedState(currentOriginal, stateInfo["maskNodes"])
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.enableMaskAllImagesIfPossible()

    def enableMaskAllImagesIfPossible(self):
        stateInfo = self.imageStates[self.currentImageIndex]
        if stateInfo["state"] == "masked" and stateInfo["bboxCoords"] is not None:
            self.maskAllImagesButton.enabled = True
        else:
            self.maskAllImagesButton.enabled = False

    def showOriginalOnly(self, volNode):
        redComposite = slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComposite.SetBackgroundVolumeID(volNode.GetID())
        redComposite.SetForegroundVolumeID(None)
        redComposite.SetLabelVolumeID(None)

        red2Composite = slicer.app.layoutManager().sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Composite.SetBackgroundVolumeID(None)
        red2Composite.SetForegroundVolumeID(None)
        red2Composite.SetLabelVolumeID(None)

        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().FitSliceToAll()
        slicer.app.layoutManager().sliceWidget('Red2').sliceLogic().FitSliceToAll()

    def showMaskedState(self, originalVol, maskNodes):
        redComposite = slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComposite.SetBackgroundVolumeID(originalVol.GetID())
        redComposite.SetForegroundVolumeID(None)
        redComposite.SetLabelVolumeID(maskNodes["labelVol"].GetID())
        redComposite.SetLabelOpacity(0.5)
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().FitSliceToAll()

        red2Composite = slicer.app.layoutManager().sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Composite.SetBackgroundVolumeID(maskNodes["maskedVol"].GetID())
        red2Composite.SetForegroundVolumeID(None)
        red2Composite.SetLabelVolumeID(None)
        slicer.app.layoutManager().sliceWidget('Red2').sliceLogic().FitSliceToAll()

    def onPrevImage(self):
        if self.currentImageIndex > 0:
            self.currentImageIndex -= 1
            self.updateVolumeDisplay()

    def onNextImage(self):
        if self.currentImageIndex < len(self.originalVolumes) - 1:
            self.currentImageIndex += 1
            self.updateVolumeDisplay()

    def onPlaceBoundingBoxClicked(self):
        self.storeCurrentButtonStates()
        self.disableAllButtonsExceptDone()

        stateInfo = self.imageStates[self.currentImageIndex]
        state = stateInfo["state"]

        if state == "masked":
            if slicer.util.confirmYesNoDisplay(
                    "This image is already masked. Creating a new bounding box will remove the existing mask. Proceed?"):
                self.removeMaskFromCurrentImage()
                self.startPlacingPoints()
            else:
                self.restoreButtonStates()
        elif state == "bbox":
            if slicer.util.confirmYesNoDisplay(
                    "A bounding box already exists. Creating a new one will remove it. Proceed?"):
                self.removeBboxFromCurrentImage()
                self.startPlacingPoints()
            else:
                self.restoreButtonStates()
        elif state == "none":
            self.startPlacingPoints()

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

    def disableAllButtonsExceptDone(self):
        for btn in self.buttonsToManage:
            if btn != self.doneButton:
                if isinstance(btn, qt.QComboBox):
                    btn.setEnabled(False)
                else:
                    btn.enabled = False

    def startPlacingPoints(self):
        self.removeBboxLines()
        if self.boundingBoxFiducialNode:
            slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode)
            self.boundingBoxFiducialNode = None

        self.boundingBoxFiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                                          "BoundingBoxPoints")
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
        self.boundingBoxFiducialNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                                                 self.onBoundingBoxPointPlaced)

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.boundingBoxFiducialNode.GetID())
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self.doneButton.enabled = False
        self.maskButton.enabled = False

    def onBoundingBoxPointPlaced(self, caller, event):
        nPoints = self.boundingBoxFiducialNode.GetNumberOfDefinedControlPoints()
        if nPoints == 2:
            slicer.util.infoDisplay("Two points placed. Click 'Done' to confirm bounding box.")
            self.disablePointSelection()
            self.doneButton.enabled = True

    def disablePointSelection(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        interactionNode.SetPlaceModePersistence(0)

    def onDoneClicked(self):
        if self.boundingBoxFiducialNode is None or self.boundingBoxFiducialNode.GetNumberOfDefinedControlPoints() < 2:
            slicer.util.warningDisplay("You must place two points first.")
            return

        bboxCoords = self.computeBboxFromFiducials()
        self.imageStates[self.currentImageIndex]["state"] = "bbox"
        self.imageStates[self.currentImageIndex]["bboxCoords"] = bboxCoords
        self.imageStates[self.currentImageIndex]["maskNodes"] = None

        if self.boundingBoxFiducialNode.GetDisplayNode():
            slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode.GetDisplayNode())
        slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode)
        self.boundingBoxFiducialNode = None

        self.doneButton.enabled = False
        self.maskButton.enabled = True
        self.updateVolumeDisplay()

        prevIndex = self.currentImageIndex - 1
        nextIndex = self.currentImageIndex + 1
        originalIndex = self.currentImageIndex
        if prevIndex >= 0:
            self.currentImageIndex = prevIndex
            self.updateVolumeDisplay()
            self.currentImageIndex = originalIndex
            self.updateVolumeDisplay()
        elif nextIndex < len(self.imagePaths):
            self.currentImageIndex = nextIndex
            self.updateVolumeDisplay()
            self.currentImageIndex = originalIndex
            self.updateVolumeDisplay()

        self.restoreButtonStates()

    def computeBboxFromFiducials(self):
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0]
        self.boundingBoxFiducialNode.GetNthControlPointPositionWorld(0, p1)
        self.boundingBoxFiducialNode.GetNthControlPointPositionWorld(1, p2)

        currentVol = self.originalVolumes[self.currentImageIndex]
        rasToIjkMatrix = vtk.vtkMatrix4x4()
        currentVol.GetRASToIJKMatrix(rasToIjkMatrix)

        def rasToIjk(ras):
            ras4 = [ras[0], ras[1], ras[2], 1.0]
            ijk4 = rasToIjkMatrix.MultiplyPoint(ras4)
            return [int(round(ijk4[0])), int(round(ijk4[1])), int(round(ijk4[2]))]

        p1_ijk = rasToIjk(p1)
        p2_ijk = rasToIjk(p2)
        x_min = min(p1_ijk[0], p2_ijk[0])
        x_max = max(p1_ijk[0], p2_ijk[0])
        y_min = min(p1_ijk[1], p2_ijk[1])
        y_max = max(p1_ijk[1], p2_ijk[1])
        return (x_min, y_min, x_max, y_max)

    def removeBboxFromCurrentImage(self):
        self.imageStates[self.currentImageIndex]["state"] = "none"
        self.imageStates[self.currentImageIndex]["bboxCoords"] = None
        self.imageStates[self.currentImageIndex]["maskNodes"] = None
        self.removeBboxLines()

    def removeMaskFromCurrentImage(self):
        stateInfo = self.imageStates[self.currentImageIndex]
        if stateInfo["maskNodes"] is not None:
            m = stateInfo["maskNodes"]
            for nodeKey in ["labelVol", "colorNode", "maskedVol"]:
                node = m.get(nodeKey, None)
                if node and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            stateInfo["maskNodes"] = None
        stateInfo["state"] = "none"
        stateInfo["bboxCoords"] = None

    def onMaskClicked(self):
        stateInfo = self.imageStates[self.currentImageIndex]
        if stateInfo["state"] != "bbox":
            slicer.util.warningDisplay("No bounding box defined for this image.")
            return

        bboxCoords = stateInfo["bboxCoords"]
        currentOriginal = self.originalVolumes[self.currentImageIndex]
        arr = slicer.util.arrayFromVolume(currentOriginal)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            rgbArr = np.stack([arr, arr, arr], axis=-1)
        else:
            rgbArr = arr

        mask = self.logic.run_sam_segmentation(rgbArr, bboxCoords)
        labelVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        labelVol.CreateDefaultDisplayNodes()
        self.createdNodes.append(labelVol)
        labelArray = (mask > 0).astype(np.uint8)
        slicer.util.updateVolumeFromArray(labelVol, labelArray)

        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
        self.createdNodes.append(colorNode)
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(2)
        colorNode.SetColor(0, "Background", 0, 0, 0, 0)
        colorNode.SetColor(1, "Mask", 1, 0, 0, 1)
        labelVol.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

        maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        self.createdNodes.append(maskedVol)
        masked = rgbArr.copy()
        masked[mask == 0] = 0
        grayMasked = np.mean(masked, axis=2).astype(np.uint8)
        slicer.util.updateVolumeFromArray(maskedVol, grayMasked)
        maskedVol.SetName(currentOriginal.GetName() + "_masked")

        stateInfo["state"] = "masked"
        stateInfo["maskNodes"] = {
            "labelVol": labelVol,
            "colorNode": colorNode,
            "maskedVol": maskedVol,
            "labelArray": labelArray,
            "grayMasked": grayMasked
        }

        self.removeBboxLines()

        self.processedOnce = True
        self.saveMaskedImage(self.currentImageIndex, mask)
        self.updateVolumeDisplay()

    def removeBboxLines(self):
        for lineNode in self.currentBboxLineNodes:
            if lineNode and slicer.mrmlScene.IsNodePresent(lineNode):
                slicer.mrmlScene.RemoveNode(lineNode)
        self.currentBboxLineNodes = []

    def drawBboxLines(self, bboxCoords):
        currentVol = self.originalVolumes[self.currentImageIndex]
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        currentVol.GetIJKToRASMatrix(ijkToRasMatrix)

        def ijkToRas(i, j):
            p = [i, j, 0, 1]
            ras = ijkToRasMatrix.MultiplyPoint(p)
            return [ras[0], ras[1], ras[2]]

        x_min, y_min, x_max, y_max = bboxCoords
        p1 = ijkToRas(x_min, y_min)
        p2 = ijkToRas(x_max, y_min)
        p3 = ijkToRas(x_max, y_max)
        p4 = ijkToRas(x_min, y_max)

        lineEndpoints = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        for (start, end) in lineEndpoints:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            self.createdNodes.append(lineNode)
            lineNode.AddControlPoint(start)
            lineNode.AddControlPoint(end)
            lengthMeasurement = lineNode.GetMeasurement('length')
            if lengthMeasurement:
                lengthMeasurement.SetEnabled(False)

            dnode = lineNode.GetDisplayNode()
            dnode.SetLineThickness(0.25)
            dnode.SetSelectedColor(1, 1, 0)
            dnode.SetPointLabelsVisibility(False)
            dnode.SetPropertiesLabelVisibility(False)
            dnode.SetTextScale(0)

            self.currentBboxLineNodes.append(lineNode)

    def onMaskAllImagesClicked(self):
        stateInfo = self.imageStates[self.currentImageIndex]
        if stateInfo["state"] != "masked" or stateInfo["bboxCoords"] is None:
            slicer.util.warningDisplay("Current image is not masked or has no bounding box info.")
            return

        bboxCoords = stateInfo["bboxCoords"]
        self.maskAllProgressBar.setVisible(True)
        imagesToMask = [i for i in range(len(self.imagePaths)) if
                        i != self.currentImageIndex and self.imageStates[i]["state"] != "masked"]
        self.maskAllProgressBar.setRange(0, len(imagesToMask))
        self.maskAllProgressBar.setValue(0)

        for count, i in enumerate(imagesToMask):
            self.maskSingleImage(i, bboxCoords)
            self.maskAllProgressBar.setValue(count + 1)
            slicer.app.processEvents()

        slicer.util.infoDisplay("All images masked and saved.")
        self.maskAllProgressBar.setVisible(False)
        self.updateVolumeDisplay()

    def maskSingleImage(self, index, bboxCoords):
        arr = self.setStates[self.currentSet]["originalArrays"][index]
        rgbArr = np.stack([arr, arr, arr], axis=-1)
        mask = self.logic.run_sam_segmentation(rgbArr, bboxCoords)

        labelVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        labelVol.CreateDefaultDisplayNodes()
        self.createdNodes.append(labelVol)
        labelArray = (mask > 0).astype(np.uint8)
        slicer.util.updateVolumeFromArray(labelVol, labelArray)

        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
        self.createdNodes.append(colorNode)
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(2)
        colorNode.SetColor(0, "Background", 0, 0, 0, 0)
        colorNode.SetColor(1, "Mask", 1, 0, 0, 1)
        labelVol.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

        maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        self.createdNodes.append(maskedVol)
        masked = rgbArr.copy()
        masked[mask == 0] = 0
        grayMasked = np.mean(masked, axis=2).astype(np.uint8)
        slicer.util.updateVolumeFromArray(maskedVol, grayMasked)
        maskedVol.SetName(os.path.basename(self.imagePaths[index]) + "_masked")

        self.imageStates[index]["state"] = "masked"
        self.imageStates[index]["bboxCoords"] = bboxCoords
        self.imageStates[index]["maskNodes"] = {
            "labelVol": labelVol,
            "colorNode": colorNode,
            "maskedVol": maskedVol,
            "labelArray": labelArray,
            "grayMasked": grayMasked
        }

        self.saveMaskedImage(index, mask)

    def saveMaskedImage(self, index, mask):
        from PIL import Image
        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        os.makedirs(setOutputFolder, exist_ok=True)

        stateInfo = self.imageStates[index]
        if stateInfo["state"] != "masked" or stateInfo["maskNodes"] is None:
            print("No masked volume available for saving this image.")
            return

        grayMasked = stateInfo["maskNodes"]["grayMasked"]
        arr = np.flipud(grayMasked)
        arr_rgb = np.stack([arr, arr, arr], axis=-1)

        filename = os.path.basename(self.imagePaths[index])
        Image.fromarray(arr_rgb.astype(np.uint8)).save(os.path.join(setOutputFolder, filename))

    def clearAllCreatedNodes(self):
        for node in self.createdNodes:
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.createdNodes = []
        self.removeBboxLines()
        self.boundingBoxFiducialNode = None

    def cleanup(self):
        self.saveCurrentSetState()
        self.clearAllCreatedNodes()


class SlicerPhotogrammetryLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        from segment_anything import sam_model_registry, SamPredictor
        # Ensure weights are present
        weights_filename = "sam_vit_h_4b8939.pth"
        sam_checkpoint = self.check_and_download_weights(weights_filename)
        self.device = "cpu"
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def check_and_download_weights(self, filename):
        """Check if the SAM weights file exists, if not, download it."""
        # Construct the resource path for the file
        modulePath = os.path.dirname(slicer.modules.slicerphotogrammetry.path)
        resourcePath = os.path.join(modulePath, 'Resources', filename)

        if not os.path.isfile(resourcePath):
            # File not present, download from source
            weights_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            slicer.util.infoDisplay(f"Downloading {filename}. This may take a few minutes...", autoCloseMsec=2000)
            try:
                slicer.util.downloadFile(url=weights_url, filepath=resourcePath)
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to download {filename}: {str(e)}")
                raise RuntimeError("Could not download SAM weights.")

        return resourcePath

    @staticmethod
    def get_image_paths_from_folder(folder_path: str,
                                    extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]) -> List[str]:
        folder_path = os.path.abspath(folder_path)
        image_paths = []
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in extensions:
                    full_path = os.path.join(folder_path, filename)
                    if os.path.isfile(full_path):
                        image_paths.append(full_path)
        return sorted(image_paths)

    def run_sam_segmentation(self, image_rgb, bounding_box):
        self.predictor.set_image(image_rgb)
        box = np.array(bounding_box, dtype=np.float32)
        import torch
        with torch.no_grad():
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )
        mask = masks[0].astype(np.uint8)
        return mask
