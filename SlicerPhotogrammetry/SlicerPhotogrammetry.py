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
    """Unused in logic; we do check_and_download_weights instead."""
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
    """
    Manages UI for:
     - SAM model loading,
     - Folder processing,
     - Bbox/masking steps,
     - EXIF + color/writing,
     - Creating _mask.png for webODM
    """
    # ALWAYS ADD EXTERNAL IMPORTS HERE
    ##############################################################
    # EXIF helper: read/write using Pillow
    ##############################################################
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        slicer.util.pip_install("Pillow")
        from PIL import Image
        from PIL.ExifTags import TAGS
        
    try:
        import cv2
    except ImportError:
        slicer.util.pip_install("opencv-python")
        import cv2

    try:
        import segment_anything
    except ImportError:
        slicer.util.pip_install("git+https://github.com/facebookresearch/segment-anything.git")
        import segment_anything

    from segment_anything import sam_model_registry, SamPredictor

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

        self.logic = SlicerPhotogrammetryLogic()

        # Each set in self.setStates:
        #   setName -> {
        #     "imagePaths": [...],
        #     "imageStates": { index: {...} },
        #     "originalGrayArrays": [...],
        #     "originalColorArrays": [...],
        #     "exifData": [exif_bytes_or_empty,...],
        #   }
        self.setStates = {}
        self.currentSet = None
        self.imagePaths = []
        self.currentImageIndex = 0
        self.originalVolumes = []  # grayscale volumes
        self.processedOnce = False
        self.layoutId = 1003

        # Per-image state: "none"|"bbox"|"masked"
        self.imageStates = {}
        self.createdNodes = []
        self.currentBboxLineNodes = []
        self.boundingBoxFiducialNode = None
        self.placingBoundingBox = False

        # UI
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

        # Model selection
        self.samVariantCombo = None
        self.loadModelButton = None
        self.modelLoaded = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.setAlignment(qt.Qt.AlignTop)

        self.createCustomLayout()

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Import Image Sets"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # 1) SAM variant
        self.samVariantCombo = qt.QComboBox()
        self.samVariantCombo.addItem("ViT-base (~376 MB)")    # => "vit_b"
        self.samVariantCombo.addItem("ViT-large (~1.03 GB)")  # => "vit_l"
        self.samVariantCombo.addItem("ViT-huge (~2.55 GB)")   # => "vit_h"
        parametersFormLayout.addRow("SAM Variant:", self.samVariantCombo)

        # 2) Load Model
        self.loadModelButton = qt.QPushButton("Load Model")
        parametersFormLayout.addWidget(self.loadModelButton)
        self.loadModelButton.connect('clicked(bool)', self.onLoadModelClicked)

        # 3) Master/Output
        self.masterFolderSelector = ctk.ctkDirectoryButton()
        self.masterFolderSelector.directory = ""
        parametersFormLayout.addRow("Master Folder:", self.masterFolderSelector)

        self.outputFolderSelector = ctk.ctkDirectoryButton()
        self.outputFolderSelector.directory = ""
        parametersFormLayout.addRow("Output Folder:", self.outputFolderSelector)

        # 4) Process Folders
        self.processButton = qt.QPushButton("Process Folders")
        parametersFormLayout.addWidget(self.processButton)
        self.processButton.connect('clicked(bool)', self.onProcessFoldersClicked)

        self.processFoldersProgressBar = qt.QProgressBar()
        self.processFoldersProgressBar.setVisible(False)
        parametersFormLayout.addWidget(self.processFoldersProgressBar)

        # 5) Image Set combo
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
        navLayout = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("<-")
        self.nextButton = qt.QPushButton("->")
        self.imageIndexLabel = qt.QLabel("[Image 0]")
        self.prevButton.enabled = False
        self.nextButton.enabled = False
        navLayout.addWidget(self.prevButton)
        navLayout.addWidget(self.imageIndexLabel)
        navLayout.addWidget(self.nextButton)
        parametersFormLayout.addRow(navLayout)
        self.prevButton.connect('clicked(bool)', self.onPrevImage)
        self.nextButton.connect('clicked(bool)', self.onNextImage)

        # 9) "Mask All Images In Set" + progress
        self.maskAllImagesButton = qt.QPushButton("Mask All Images In Set")
        self.maskAllImagesButton.enabled = False
        parametersFormLayout.addWidget(self.maskAllImagesButton)
        self.maskAllImagesButton.connect('clicked(bool)', self.onMaskAllImagesClicked)

        self.maskAllProgressBar = qt.QProgressBar()
        self.maskAllProgressBar.setVisible(False)
        self.maskAllProgressBar.setTextVisible(True)
        parametersFormLayout.addWidget(self.maskAllProgressBar)

        # 10) Buttons to manage
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
        """Creates a 2-slice layout: Red and Red2 side by side."""
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

    # ----------------- (A) onLoadModelClicked
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
        success = self.logic.loadSAMModel(variant=info["registry_key"],
                                          filename=info["filename"],
                                          url=info["url"])
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

    # ----------------- (B) "Process Folders"
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
            for idx, stInfo in setData["imageStates"].items():
                if stInfo["state"] in ["bbox", "masked"]:
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

    # ----------------- onImageSetSelected
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

            originalGrayArrays = []
            originalColorArrays = []
            exifDataList = []

            # read color + gray + exif
            for path in self.imagePaths:
                colorArr, grayArr, exif_bytes = self.loadImageColorGrayEXIF(path)
                originalColorArrays.append(colorArr)
                originalGrayArrays.append(grayArr)
                exifDataList.append(exif_bytes)

            for i in range(len(self.imagePaths)):
                self.imageStates[i] = {
                    "state": "none",
                    "bboxCoords": None,
                    "maskNodes": None
                }

            self.currentImageIndex = 0
            self.createVolumesFromGrayArrays(originalGrayArrays)

            self.setStates[self.currentSet] = {
                "imagePaths": self.imagePaths,
                "imageStates": self.imageStates,
                "originalGrayArrays": originalGrayArrays,
                "originalColorArrays": originalColorArrays,
                "exifData": exifDataList  # store each image's exif
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

        for idx, stInfo in self.imageStates.items():
            if stInfo["state"] == "masked" and stInfo["maskNodes"]:
                m = stInfo["maskNodes"]
                for nd in ["labelVol", "colorNode", "maskedVol"]:
                    node = m[nd]
                    if node and slicer.mrmlScene.IsNodePresent(node):
                        slicer.mrmlScene.RemoveNode(node)
        self.clearAllCreatedNodes()

    def restoreSetState(self, setName):
        setData = self.setStates[setName]
        self.imagePaths = setData["imagePaths"]
        self.imageStates = setData["imageStates"]

        grayArrays = setData["originalGrayArrays"]

        self.createVolumesFromGrayArrays(grayArrays)

        for idx, stInfo in self.imageStates.items():
            if stInfo["state"] == "masked" and stInfo["maskNodes"]:
                m = stInfo["maskNodes"]
                lbArr = m["labelArray"]
                gMasked = m["grayMasked"]

                lbVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                lbVol.CreateDefaultDisplayNodes()
                self.createdNodes.append(lbVol)
                slicer.util.updateVolumeFromArray(lbVol, lbArr)

                cNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
                self.createdNodes.append(cNode)
                cNode.SetTypeToUser()
                cNode.SetNumberOfColors(2)
                cNode.SetColor(0, "Background", 0, 0, 0, 0)
                cNode.SetColor(1, "Mask", 1, 0, 0, 1)
                lbVol.GetDisplayNode().SetAndObserveColorNodeID(cNode.GetID())

                maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
                self.createdNodes.append(maskedVol)
                slicer.util.updateVolumeFromArray(maskedVol, gMasked)
                maskedVol.SetName(os.path.basename(self.imagePaths[idx]) + "_masked")

                stInfo["maskNodes"] = {
                    "labelVol": lbVol,
                    "colorNode": cNode,
                    "maskedVol": maskedVol,
                    "labelArray": lbArr,
                    "grayMasked": gMasked
                }

        self.currentImageIndex = 0
        self.updateVolumeDisplay()
        self.placeBoundingBoxButton.enabled = True
        self.refreshButtonStatesBasedOnCurrentState()

    def createVolumesFromGrayArrays(self, grayArrays):
        self.originalVolumes = []
        for i, arr in enumerate(grayArrays):
            volNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            self.createdNodes.append(volNode)
            slicer.util.updateVolumeFromArray(volNode, arr)
            volNode.SetName(os.path.basename(self.imagePaths[i]))
            self.originalVolumes.append(volNode)

    def loadImageColorGrayEXIF(self, path):
        """
        Loads color array (flipped) + grayscale + exif bytes if any.
        """
        from PIL import Image
        from PIL.ExifTags import TAGS

        im = Image.open(path)
        exif_bytes = im.info.get("exif", b"")

        im_rgb = im.convert('RGB')
        colorArr = np.asarray(im_rgb)
        colorArr = np.flipud(colorArr)
        grayArr = np.mean(colorArr, axis=2).astype(np.uint8)
        return colorArr, grayArr, exif_bytes

    def refreshButtonStatesBasedOnCurrentState(self):
        st = self.imageStates[self.currentImageIndex]["state"]
        if st == "none":
            self.doneButton.enabled = False
            self.maskButton.enabled = False
        elif st == "bbox":
            self.doneButton.enabled = False
            self.maskButton.enabled = True
        elif st == "masked":
            self.doneButton.enabled = False
            self.maskButton.enabled = False
        self.enableMaskAllImagesIfPossible()

    def updateVolumeDisplay(self):
        self.imageIndexLabel.setText(f"[Image {self.currentImageIndex}]")
        if not self.originalVolumes or self.currentImageIndex >= len(self.originalVolumes):
            return

        currentVol = self.originalVolumes[self.currentImageIndex]
        self.removeBboxLines()

        st = self.imageStates[self.currentImageIndex]["state"]

        if st == "none":
            self.showOriginalOnly(currentVol)
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.enableMaskAllImagesIfPossible()
        elif st == "bbox":
            self.showOriginalOnly(currentVol)
            self.drawBboxLines(self.imageStates[self.currentImageIndex]["bboxCoords"])
            self.doneButton.enabled = False
            self.maskButton.enabled = True
            self.enableMaskAllImagesIfPossible()
        elif st == "masked":
            mNodes = self.imageStates[self.currentImageIndex]["maskNodes"]
            self.showMaskedState(currentVol, mNodes)
            self.doneButton.enabled = False
            self.maskButton.enabled = False
            self.enableMaskAllImagesIfPossible()

    def enableMaskAllImagesIfPossible(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if stInfo and stInfo["state"] == "masked" and stInfo["bboxCoords"] is not None:
            self.maskAllImagesButton.enabled = True
        else:
            self.maskAllImagesButton.enabled = False

    @staticmethod
    def showOriginalOnly(volNode):
        lm = slicer.app.layoutManager()
        redComp = lm.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComp.SetBackgroundVolumeID(volNode.GetID())
        redComp.SetForegroundVolumeID(None)
        redComp.SetLabelVolumeID(None)

        red2Comp = lm.sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Comp.SetBackgroundVolumeID(None)
        red2Comp.SetForegroundVolumeID(None)
        red2Comp.SetLabelVolumeID(None)

        lm.sliceWidget('Red').sliceLogic().FitSliceToAll()
        lm.sliceWidget('Red2').sliceLogic().FitSliceToAll()

    @staticmethod
    def showMaskedState(originalVol, maskNodes):
        if not maskNodes or "labelVol" not in maskNodes or "maskedVol" not in maskNodes:
            return
        lbVol = maskNodes["labelVol"]
        maskedVol = maskNodes["maskedVol"]

        lm = slicer.app.layoutManager()
        redComp = lm.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        redComp.SetBackgroundVolumeID(originalVol.GetID())
        redComp.SetForegroundVolumeID(None)
        redComp.SetLabelVolumeID(lbVol.GetID())
        redComp.SetLabelOpacity(0.5)
        lm.sliceWidget('Red').sliceLogic().FitSliceToAll()

        red2Comp = lm.sliceWidget('Red2').sliceLogic().GetSliceCompositeNode()
        red2Comp.SetBackgroundVolumeID(maskedVol.GetID())
        red2Comp.SetForegroundVolumeID(None)
        red2Comp.SetLabelVolumeID(None)
        lm.sliceWidget('Red2').sliceLogic().FitSliceToAll()

    def onPrevImage(self):
        if self.currentImageIndex > 0:
            self.currentImageIndex -= 1
            self.updateVolumeDisplay()

    def onNextImage(self):
        if self.currentImageIndex < len(self.originalVolumes) - 1:
            self.currentImageIndex += 1
            self.updateVolumeDisplay()

    # -------------- Place Bbox
    def onPlaceBoundingBoxClicked(self):
        self.storeCurrentButtonStates()
        self.disableAllButtonsExceptDone()

        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo:
            return
        s = stInfo["state"]

        if s == "masked":
            if slicer.util.confirmYesNoDisplay(
                    "This image is already masked. Creating a new bounding box will remove the existing mask. Proceed?"):
                self.removeMaskFromCurrentImage()
                self.startPlacingPoints()
            else:
                self.restoreButtonStates()
        elif s == "bbox":
            if slicer.util.confirmYesNoDisplay(
                    "A bounding box already exists. Creating a new one will remove it. Proceed?"):
                self.removeBboxFromCurrentImage()
                self.startPlacingPoints()
            else:
                self.restoreButtonStates()
        elif s == "none":
            self.startPlacingPoints()

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

    def disableAllButtonsExceptDone(self):
        for b in self.buttonsToManage:
            if b != self.doneButton:
                if isinstance(b, qt.QComboBox):
                    b.setEnabled(False)
                else:
                    b.enabled = False

    def startPlacingPoints(self):
        self.removeBboxLines()
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
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onBoundingBoxPointPlaced
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
        if not self.boundingBoxFiducialNode or self.boundingBoxFiducialNode.GetNumberOfDefinedControlPoints() < 2:
            slicer.util.warningDisplay("You must place two points first.")
            return

        coords = self.computeBboxFromFiducials()
        self.imageStates[self.currentImageIndex]["state"] = "bbox"
        self.imageStates[self.currentImageIndex]["bboxCoords"] = coords
        self.imageStates[self.currentImageIndex]["maskNodes"] = None

        if self.boundingBoxFiducialNode.GetDisplayNode():
            slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode.GetDisplayNode())
        slicer.mrmlScene.RemoveNode(self.boundingBoxFiducialNode)
        self.boundingBoxFiducialNode = None

        self.doneButton.enabled = False
        self.maskButton.enabled = True
        self.updateVolumeDisplay()

        # Refresh lines
        pI = self.currentImageIndex - 1
        nI = self.currentImageIndex + 1
        origI = self.currentImageIndex
        if pI >= 0:
            self.currentImageIndex = pI
            self.updateVolumeDisplay()
            self.currentImageIndex = origI
            self.updateVolumeDisplay()
        elif nI < len(self.imagePaths):
            self.currentImageIndex = nI
            self.updateVolumeDisplay()
            self.currentImageIndex = origI
            self.updateVolumeDisplay()

        self.restoreButtonStates()
        self.maskButton.enabled = True

    def computeBboxFromFiducials(self):
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0]
        self.boundingBoxFiducialNode.GetNthControlPointPositionWorld(0, p1)
        self.boundingBoxFiducialNode.GetNthControlPointPositionWorld(1, p2)

        vol = self.originalVolumes[self.currentImageIndex]
        rasToIjkMat = vtk.vtkMatrix4x4()
        vol.GetRASToIJKMatrix(rasToIjkMat)

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
            m = stInfo["maskNodes"]
            for nd in ["labelVol", "colorNode", "maskedVol"]:
                node = m.get(nd, None)
                if node and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            stInfo["maskNodes"] = None
        stInfo["state"] = "none"
        stInfo["bboxCoords"] = None

    def onMaskClicked(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo or stInfo["state"] != "bbox":
            slicer.util.warningDisplay("No bounding box defined for this image.")
            return

        bbox = stInfo["bboxCoords"]
        setData = self.setStates[self.currentSet]
        colorArr = setData["originalColorArrays"][self.currentImageIndex]

        mask = self.logic.run_sam_segmentation(colorArr, bbox)

        labelVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        labelVol.CreateDefaultDisplayNodes()
        self.createdNodes.append(labelVol)
        lbArr = (mask > 0).astype(np.uint8)
        slicer.util.updateVolumeFromArray(labelVol, lbArr)

        cNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
        self.createdNodes.append(cNode)
        cNode.SetTypeToUser()
        cNode.SetNumberOfColors(2)
        cNode.SetColor(0, "Background", 0, 0, 0, 0)
        cNode.SetColor(1, "Mask", 1, 0, 0, 1)
        labelVol.GetDisplayNode().SetAndObserveColorNodeID(cNode.GetID())

        maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        self.createdNodes.append(maskedVol)
        cpy = colorArr.copy()
        cpy[mask == 0] = 0
        grayMasked = np.mean(cpy, axis=2).astype(np.uint8)
        slicer.util.updateVolumeFromArray(maskedVol, grayMasked)
        maskedVol.SetName(self.originalVolumes[self.currentImageIndex].GetName() + "_masked")

        stInfo["state"] = "masked"
        stInfo["maskNodes"] = {
            "labelVol": labelVol,
            "colorNode": cNode,
            "maskedVol": maskedVol,
            "labelArray": lbArr,
            "grayMasked": grayMasked
        }

        self.removeBboxLines()
        self.processedOnce = True

        maskBool = (mask > 0)
        self.saveMaskedImage(self.currentImageIndex, colorArr, maskBool)
        self.updateVolumeDisplay()

    def removeBboxLines(self):
        for ln in self.currentBboxLineNodes:
            if ln and slicer.mrmlScene.IsNodePresent(ln):
                slicer.mrmlScene.RemoveNode(ln)
        self.currentBboxLineNodes = []

    def drawBboxLines(self, coords):
        if not coords:
            return
        vol = self.originalVolumes[self.currentImageIndex]
        ijkToRasMat = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(ijkToRasMat)

        def ijkToRas(i, j):
            p = [i, j, 0, 1]
            ras = ijkToRasMat.MultiplyPoint(p)
            return [ras[0], ras[1], ras[2]]

        x_min, y_min, x_max, y_max = coords
        p1 = ijkToRas(x_min, y_min)
        p2 = ijkToRas(x_max, y_min)
        p3 = ijkToRas(x_max, y_max)
        p4 = ijkToRas(x_min, y_max)
        lines = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]

        for (start, end) in lines:
            ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            self.createdNodes.append(ln)
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

    def onMaskAllImagesClicked(self):
        stInfo = self.imageStates.get(self.currentImageIndex, None)
        if not stInfo or stInfo["state"] != "masked" or not stInfo["bboxCoords"]:
            slicer.util.warningDisplay("Current image is not masked or has no bounding box info.")
            return

        bbox = stInfo["bboxCoords"]
        self.maskAllProgressBar.setVisible(True)
        self.maskAllProgressBar.setTextVisible(True)
        toMask = [i for i in range(len(self.imagePaths)) if i != self.currentImageIndex and self.imageStates[i]["state"] != "masked"]
        n = len(toMask)
        self.maskAllProgressBar.setRange(0, n)
        self.maskAllProgressBar.setValue(0)

        if n == 0:
            slicer.util.infoDisplay("All images in this set are already masked.")
            self.maskAllProgressBar.setVisible(False)
            return

        start_time = time.time()

        for count, i in enumerate(toMask):
            processed = count + 1
            self.maskSingleImage(i, bbox)
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

    def maskSingleImage(self, index, bboxCoords):
        setData = self.setStates[self.currentSet]
        colorArr = setData["originalColorArrays"][index]
        mask = self.logic.run_sam_segmentation(colorArr, bboxCoords)

        # label volume
        labelVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        labelVol.CreateDefaultDisplayNodes()
        self.createdNodes.append(labelVol)
        lbArr = (mask > 0).astype(np.uint8)
        slicer.util.updateVolumeFromArray(labelVol, lbArr)

        # color node
        cNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
        self.createdNodes.append(cNode)
        cNode.SetTypeToUser()
        cNode.SetNumberOfColors(2)
        cNode.SetColor(0, "Background", 0, 0, 0, 0)
        cNode.SetColor(1, "Mask", 1, 0, 0, 1)
        labelVol.GetDisplayNode().SetAndObserveColorNodeID(cNode.GetID())

        # masked vol
        maskedVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        self.createdNodes.append(maskedVol)
        cpy = colorArr.copy()
        cpy[mask == 0] = 0
        grayMasked = np.mean(cpy, axis=2).astype(np.uint8)
        slicer.util.updateVolumeFromArray(maskedVol, grayMasked)
        maskedVol.SetName(os.path.basename(self.imagePaths[index]) + "_masked")

        self.imageStates[index]["state"] = "masked"
        self.imageStates[index]["bboxCoords"] = bboxCoords
        self.imageStates[index]["maskNodes"] = {
            "labelVol": labelVol,
            "colorNode": cNode,
            "maskedVol": maskedVol,
            "labelArray": lbArr,
            "grayMasked": grayMasked
        }

        maskBool = (mask > 0)
        self.saveMaskedImage(index, colorArr, maskBool)

    def saveMaskedImage(self, index, colorArr, maskBool):
        """
        Saves:
          1) The masked color image as PNG, with EXIF if available.
          2) The corresponding _mask.png for webODM (white=255, black=0).
        """
        from PIL import Image

        # Retrieve exif from setStates if present
        setData = self.setStates[self.currentSet]
        exifList = setData.get("exifData", [])
        exif_bytes = exifList[index] if index < len(exifList) else b""

        outputFolder = self.outputFolderSelector.directory
        setOutputFolder = os.path.join(outputFolder, self.currentSet)
        os.makedirs(setOutputFolder, exist_ok=True)

        # color masked
        cpy = colorArr.copy()
        cpy[~maskBool] = 0
        cpy = np.flipud(cpy)  # flip back

        # Save color PNG
        baseName = os.path.splitext(os.path.basename(self.imagePaths[index]))[0]
        colorPngFilename = baseName + ".jpeg"
        colorPngPath = os.path.join(setOutputFolder, colorPngFilename)

        colorPil = Image.fromarray(cpy.astype(np.uint8))
        if exif_bytes:
            colorPil.save(colorPngPath, "jpeg", quality=100, exif=exif_bytes)
        else:
            colorPil.save(colorPngPath, "jpeg")

        # Save the binary mask => baseName_mask.png
        # White=255 => object, black=0 => background
        maskBin = (maskBool.astype(np.uint8) * 255)
        maskBin = np.flipud(maskBin)  # match final orientation
        maskPil = Image.fromarray(maskBin, mode='L')
        maskFilename = f"{baseName}_mask.jpeg"
        maskPath = os.path.join(setOutputFolder, maskFilename)
        maskPil.save(maskPath, "jpeg")

    def clearAllCreatedNodes(self):
        for nd in self.createdNodes:
            if nd and slicer.mrmlScene.IsNodePresent(nd):
                slicer.mrmlScene.RemoveNode(nd)
        self.createdNodes = []
        self.removeBboxLines()
        self.boundingBoxFiducialNode = None

    def cleanup(self):
        self.saveCurrentSetState()
        self.clearAllCreatedNodes()


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
        """
        image_rgb => (H, W, 3),
        bounding_box => (x_min, y_min, x_max, y_max).
        Output => mask => (H, W), 0 or 1
        """
        if not self.predictor:
            raise RuntimeError("SAM model is not loaded.")

        import torch
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