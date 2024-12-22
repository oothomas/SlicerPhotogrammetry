import os
import sys
import qt
import ctk
import vtk
import slicer
import shutil
import numpy as np
import torch
import time  # for timing
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
        self.parent.contributors = ["Oshane Thomas"]
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
        """
        Removed the old QProgressDialog from the constructor so it no longer appears or stays open.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)

        self.logic = SlicerPhotogrammetryLogic()

        # Per-set states
        self.setStates = {}
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

        # UI elements
        self.buttonsToManage = []
        self.prevButton = None
        self.nextButton = None
        self.maskButton = None
        self.doneButton = None
        self.imageIndexLabel = None
        self.maskAllImagesButton = None
        self.maskAllProgressBar = None
        self.processButton = None
        self.imageSetComboBox = None
        self.placeBoundingBoxButton = None
        self.outputFolderSelector = None
        self.masterFolderSelector = None
        self.processFoldersProgressBar = None
        self.previousButtonStates = {}

        # Model selection controls
        self.samVariantCombo = None
        self.loadModelButton = None
        self.modelLoaded = False  # We'll track if the SAM model is loaded

    def setup(self):
        """
        Called when the module is first opened.
        """
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.setAlignment(qt.Qt.AlignTop)

        # Create the 2-slice custom layout
        self.createCustomLayout()

        # UI Panel
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Import Image Sets"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # 1) SAM Variant dropdown
        self.samVariantCombo = qt.QComboBox()
        # Updated the label text to reflect approximate model sizes
        self.samVariantCombo.addItem("ViT-base (~376 MB)")   # => "vit_b"
        self.samVariantCombo.addItem("ViT-large (~1.03 GB)")  # => "vit_l"
        self.samVariantCombo.addItem("ViT-huge (~2.55 GB)")  # => "vit_h"
        parametersFormLayout.addRow("SAM Variant:", self.samVariantCombo)

        # 2) Load Model button
        self.loadModelButton = qt.QPushButton("Load Model")
        parametersFormLayout.addWidget(self.loadModelButton)
        self.loadModelButton.connect('clicked(bool)', self.onLoadModelClicked)

        # 3) Master/Output Folder controls
        self.masterFolderSelector = ctk.ctkDirectoryButton()
        self.masterFolderSelector.directory = ""
        parametersFormLayout.addRow("Master Folder:", self.masterFolderSelector)

        self.outputFolderSelector = ctk.ctkDirectoryButton()
        self.outputFolderSelector.directory = ""
        parametersFormLayout.addRow("Output Folder:", self.outputFolderSelector)

        # 4) "Process Folders" + progress bar
        self.processButton = qt.QPushButton("Process Folders")
        parametersFormLayout.addWidget(self.processButton)
        self.processButton.connect('clicked(bool)', self.onProcessFoldersClicked)

        self.processFoldersProgressBar = qt.QProgressBar()
        self.processFoldersProgressBar.setVisible(False)
        parametersFormLayout.addWidget(self.processFoldersProgressBar)

        # 5) Image set selector
        self.imageSetComboBox = qt.QComboBox()
        self.imageSetComboBox.enabled = False
        parametersFormLayout.addRow("Image Set:", self.imageSetComboBox)
        self.imageSetComboBox.connect('currentIndexChanged(int)', self.onImageSetSelected)

        # 6) Place bounding box
        self.placeBoundingBoxButton = qt.QPushButton("Place Bounding Box")
        self.placeBoundingBoxButton.enabled = False
        parametersFormLayout.addWidget(self.placeBoundingBoxButton)
        self.placeBoundingBoxButton.connect('clicked(bool)', self.onPlaceBoundingBoxClicked)

        # 7) Done, Mask
        self.doneButton = qt.QPushButton("Done")
        self.doneButton.enabled = False
        parametersFormLayout.addWidget(self.doneButton)
        self.doneButton.connect('clicked(bool)', self.onDoneClicked)

        self.maskButton = qt.QPushButton("Mask Current Image")
        self.maskButton.enabled = False
        parametersFormLayout.addWidget(self.maskButton)
        self.maskButton.connect('clicked(bool)', self.onMaskClicked)

        # 8) Navigation
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

        # 9) Mask All Images In Set + progress bar
        self.maskAllImagesButton = qt.QPushButton("Mask All Images In Set")
        self.maskAllImagesButton.enabled = False
        parametersFormLayout.addWidget(self.maskAllImagesButton)
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)

        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        self.maskAllProgressBar.setTextVisible(True)     # so we can show text
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        # 10) All controls that remain disabled until model is loaded
        self.buttonsToManage = [
            self.masterFolderSelector,
            self.outputFolderSelector,
            self.processButton,
            self.imageSetComboBox,
            self.placeBoundingBoxButton,
            self.doneButton,
            self.maskButton,
            self.maskAllImagesButton,
            self.prevButton,
            self.nextButton
        ]
        for btn in self.buttonsToManage:
            if isinstance(btn, qt.QComboBox):
                btn.setEnabled(False)
            else:
                btn.enabled = False

        self.layout.addStretch(1)

    def createCustomLayout(self):
        """
        Creates a custom layout with two slice viewers: Red and Red2.
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
        layoutNode.AddLayoutDescription(self.layoutId, customLayout)
        layoutManager.setLayout(self.layoutId)

    # ----------------------------------------------------------------------
    #  (A) Model loading methods
    # ----------------------------------------------------------------------
    def onLoadModelClicked(self):
        """
        Called when user clicks the "Load Model" button. We pick the SAM variant
        from samVariantCombo, then attempt to download/load the model.
        Once successful, enable the rest of the UI.
        """
        variant = self.samVariantCombo.currentText
        # Adjusted names to match what's in the combo:
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
        filename = info["filename"]
        url = info["url"]
        registry_key = info["registry_key"]

        success = self.logic.loadSAMModel(variant=registry_key, filename=filename, url=url)
        if success:
            slicer.util.infoDisplay(f"{variant} model loaded successfully.")
            self.modelLoaded = True
            # Enable the rest of the UI that is relevant
            self.processButton.setEnabled(True)
            self.masterFolderSelector.setEnabled(True)
            self.outputFolderSelector.setEnabled(True)
            self.samVariantCombo.setEnabled(False)
            self.loadModelButton.setEnabled(False)

        else:
            slicer.util.errorDisplay("Failed to load the model. Check the error log.")

    # ----------------------------------------------------------------------
    #  (B) "Process Folders" methods
    # ----------------------------------------------------------------------
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
        for setName, setData in self.setStates.items():
            for idx, stateInfo in setData["imageStates"].items():
                if stateInfo["state"] in ["bbox", "masked"]:
                    return True
        return False

    def clearAllData(self):
        self.setStates = {}
        self.currentSet = None
        self.clearAllCreatedNodes()
        self.originalVolumes = []
        self.imageStates = {}
        self.imagePaths = []
        self.currentImageIndex = 0
        self.imageSetComboBox.clear()
        self.imageSetComboBox.enabled = False

        self.placeBoundingBoxButton.enabled = False
        self.doneButton.enabled = False
        self.maskButton.enabled = False
        self.maskAllImagesButton.enabled = False
        self.prevButton.enabled = False
        self.nextButton.enabled = False
        self.imageIndexLabel.setText("[Image 0]")

    # ----------------------------------------------------------------------
    #  (C) onImageSetSelected method is needed for the combo box
    # ----------------------------------------------------------------------
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

    @staticmethod
    def loadAndFlipImageToArray(path):
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
        if not self.originalVolumes or self.currentImageIndex >= len(self.originalVolumes):
            return

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
        stateInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stateInfo:
            return
        if stateInfo["state"] == "masked" and stateInfo["bboxCoords"] is not None:
            self.maskAllImagesButton.enabled = True
        else:
            self.maskAllImagesButton.enabled = False

    @staticmethod
    def showOriginalOnly(volNode):
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

    @staticmethod
    def showMaskedState(originalVol, maskNodes):
        if "labelVol" not in maskNodes or "maskedVol" not in maskNodes:
            return
        labelVol = maskNodes["labelVol"]
        maskedVol = maskNodes["maskedVol"]

        redComposite = slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComposite.SetBackgroundVolumeID(originalVol.GetID())
        redComposite.SetForegroundVolumeID(None)
        redComposite.SetLabelVolumeID(labelVol.GetID())
        redComposite.SetLabelOpacity(0.5)
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().FitSliceToAll()

        red2Composite = slicer.app.layoutManager().sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Composite.SetBackgroundVolumeID(maskedVol.GetID())
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

    # -------------- Place Bbox --------------
    def onPlaceBoundingBoxClicked(self):
        self.storeCurrentButtonStates()
        self.disableAllButtonsExceptDone()

        stateInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stateInfo:
            return
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

    @staticmethod
    def disablePointSelection():
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        interactionNode.SetPlaceModePersistence(0)

    def onDoneClicked(self):
        if (not self.boundingBoxFiducialNode or
                self.boundingBoxFiducialNode.GetNumberOfDefinedControlPoints() < 2):
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

        # Programmatically refresh lines
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
        # In some workflows you might want to re-enable mask button explicitly:
        self.maskButton.enabled = True

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
        return x_min, y_min, x_max, y_max

    def removeBboxFromCurrentImage(self):
        if self.imageStates.get(self.currentImageIndex):
            self.imageStates[self.currentImageIndex]["state"] = "none"
            self.imageStates[self.currentImageIndex]["bboxCoords"] = None
            self.imageStates[self.currentImageIndex]["maskNodes"] = None
        self.removeBboxLines()

    def removeMaskFromCurrentImage(self):
        stateInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stateInfo:
            return
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
        stateInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stateInfo or stateInfo["state"] != "bbox":
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
        if not bboxCoords:
            return
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
        """
        This function now provides time details (elapsed, remain) as it processes each image.
        """
        stateInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stateInfo or stateInfo["state"] != "masked" or not stateInfo["bboxCoords"]:
            slicer.util.warningDisplay("Current image is not masked or has no bounding box info.")
            return

        bboxCoords = stateInfo["bboxCoords"]
        self.maskAllProgressBar.setVisible(True)
        self.maskAllProgressBar.setTextVisible(True)
        imagesToMask = [i for i in range(len(self.imagePaths))
                        if i != self.currentImageIndex and self.imageStates[i]["state"] != "masked"]
        num_images = len(imagesToMask)
        self.maskAllProgressBar.setRange(0, num_images)
        self.maskAllProgressBar.setValue(0)

        if num_images == 0:
            slicer.util.infoDisplay("All images in this set are already masked.")
            self.maskAllProgressBar.setVisible(False)
            return

        start_time = time.time()

        for count, i in enumerate(imagesToMask):
            # Mark start of iteration
            iteration_start = time.time()

            # Mask the single image
            self.maskSingleImage(i, bboxCoords)
            # Update the progress
            processed = count + 1
            self.maskAllProgressBar.setValue(processed)

            # Calculate elapsed time so far
            elapsed_secs = time.time() - start_time
            # Average time per image
            avg_per_image = elapsed_secs / processed
            # Remain time
            remain = avg_per_image * (num_images - processed)

            def format_seconds(sec):
                # Quick helper to format secs as HH:MM:SS
                sec = int(sec)
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                if h > 0:
                    return f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    return f"{m:02d}:{s:02d}"

            elapsed_str = format_seconds(elapsed_secs)
            remain_str = format_seconds(remain)

            # Set progress bar text
            msg = f"Masking image {processed}/{num_images} | Elapsed: {elapsed_str}, Remain: {remain_str}"
            self.maskAllProgressBar.setFormat(msg)

            slicer.app.processEvents()

        slicer.util.infoDisplay("All images in set masked and saved.")
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
        maskBool = (mask > 0)
        self.saveMaskedImage(index, maskBool)

    def saveMaskedImage(self, index, mask):
        from PIL import Image
        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        os.makedirs(setOutputFolder, exist_ok=True)

        stateInfo = self.imageStates.get(index, None)
        if (not stateInfo or stateInfo["state"] != "masked" or
                not stateInfo["maskNodes"]):
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
        self.predictor = None
        self.sam = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA:0")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

    def loadSAMModel(self, variant, filename, url):
        """
        Attempt to download weights if needed, then initialize the SAM model
        with the specified registry key (variant).
        """
        try:
            sam_checkpoint = self.check_and_download_weights(filename, url)
            import segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            self.sam = sam_model_registry[variant](checkpoint=sam_checkpoint)
            self.sam.to(device=self.device)
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
            slicer.util.infoDisplay(f"Downloading {filename}. This may take a few minutes...", autoCloseMsec=2000)
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
            for filename in os.listdir(folder_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in extensions:
                    full_path = os.path.join(folder_path, filename)
                    if os.path.isfile(full_path):
                        image_paths.append(full_path)
        return sorted(image_paths)

    def run_sam_segmentation(self, image_rgb, bounding_box):
        """
        Run SAM segmentation using the current predictor.
        """
        if not self.predictor:
            raise RuntimeError("SAM model is not loaded. Please load model first.")
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
        return masks[0].astype(np.uint8)
