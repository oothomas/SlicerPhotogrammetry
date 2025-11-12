import os
import math
import glob
import shutil

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# ClusterPhotos
#
class ClusterPhotos(ScriptedLoadableModule):
    """
    This class is the 'interface' description of the module to Slicer.
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ClusterPhotos"
        parent.categories = ["SlicerMorph.Photogrammetry"]
        parent.dependencies = []
        parent.contributors = ["Oshane Thomas (SCRI)"]
        parent.helpText = """
This module allows you to cluster images into subfolders using a recursive
spectral clustering approach with a Vision Transformer embedding.
You can visualize clusters using UMAP in a single Plot view and copy the
image files once you're satisfied with the results.
"""
        parent.acknowledgementText = """This module was developed with support from the National Science 
        Foundation under grants DBI/2301405 and OAC/2118240 awarded to AMM at Seattle Children's Research Institute. 
        """



#
# ClusterPhotosWidget
#
class ClusterPhotosWidget(ScriptedLoadableModuleWidget):
    """
    The module GUI class. Creates UI elements (buttons, sliders, etc.)
    and connects to the logic for processing.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

        # UI elements
        self.loadModelButton = None
        self.loadImagesButton = None
        self.clusterPlotButton = None
        self.copyImagesButton = None

        # Parameter controls
        self.umapNNeighborsSpin = None
        self.umapMinDistSpin = None
        self.maxClusterSizeSpin = None
        self.maxEigenSpin = None
        self.kNeighborsSpin = None
        self.modelNameEdit = None
        self.inputDirSelector = None

        # Progress bar for embedding
        self.embeddingProgressBar = None

        # QTableWidget for cluster summary
        self.clusterSummaryTable = None

        # Custom layout ID
        self.chartOnlyLayoutId = 901  # arbitrary unique integer

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.load_dependencies()

        #
        # --- Collapsible Button: Parameters ---
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Image directory
        self.inputDirSelector = ctk.ctkPathLineEdit()
        self.inputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.inputDirSelector.setToolTip("Select the directory containing images.")
        parametersFormLayout.addRow("Image folder:", self.inputDirSelector)

        # Model name
        self.modelNameEdit = qt.QLineEdit("google/vit-large-patch16-224")
        self.modelNameEdit.setToolTip("HuggingFace model name (e.g. google/vit-base-patch16-224).")
        parametersFormLayout.addRow("Model name:", self.modelNameEdit)

        # K neighbors
        self.kNeighborsSpin = qt.QSpinBox()
        self.kNeighborsSpin.setRange(2, 200)
        self.kNeighborsSpin.setValue(10)
        self.kNeighborsSpin.setToolTip("Number of neighbors for k-NN graph.")
        parametersFormLayout.addRow("k-neighbors:", self.kNeighborsSpin)

        # Max eigen
        self.maxEigenSpin = qt.QSpinBox()
        self.maxEigenSpin.setRange(2, 100)
        self.maxEigenSpin.setValue(15)
        self.maxEigenSpin.setToolTip("Max number of eigenvectors used in spectral clustering.")
        parametersFormLayout.addRow("Max eigenvectors:", self.maxEigenSpin)

        # Max cluster size
        self.maxClusterSizeSpin = qt.QSpinBox()
        self.maxClusterSizeSpin.setRange(2, 2000)
        self.maxClusterSizeSpin.setValue(40)
        self.maxClusterSizeSpin.setToolTip("Maximum size of each leaf cluster.")
        parametersFormLayout.addRow("Max cluster size:", self.maxClusterSizeSpin)

        # UMAP n_neighbors
        self.umapNNeighborsSpin = qt.QSpinBox()
        self.umapNNeighborsSpin.setRange(2, 200)
        self.umapNNeighborsSpin.setValue(10)
        self.umapNNeighborsSpin.setToolTip("UMAP parameter: number of neighbors.")
        parametersFormLayout.addRow("UMAP n_neighbors:", self.umapNNeighborsSpin)

        # UMAP min_dist
        self.umapMinDistSpin = ctk.ctkDoubleSpinBox()
        self.umapMinDistSpin.minimum = 0.0
        self.umapMinDistSpin.maximum = 1.0
        self.umapMinDistSpin.singleStep = 0.01
        self.umapMinDistSpin.setValue(0.1)
        self.umapMinDistSpin.setToolTip("UMAP parameter: minimum distance.")
        parametersFormLayout.addRow("UMAP min_dist:", self.umapMinDistSpin)

        #
        # --- Collapsible Button: Actions ---
        #
        actionsCollapsibleButton = ctk.ctkCollapsibleButton()
        actionsCollapsibleButton.text = "Actions"
        self.layout.addWidget(actionsCollapsibleButton)

        actionsFormLayout = qt.QFormLayout(actionsCollapsibleButton)

        # Load Model button
        self.loadModelButton = qt.QPushButton("Load Model")
        self.loadModelButton.toolTip = "Load the ViT model (e.g. google/vit-large-patch16-224)."
        actionsFormLayout.addRow(self.loadModelButton)

        # Load Images & Embed button
        self.loadImagesButton = qt.QPushButton("Load Images & Embed")
        self.loadImagesButton.toolTip = "Load images from folder, compute embeddings, and store results."
        self.loadImagesButton.enabled = False  # only after model loaded
        actionsFormLayout.addRow(self.loadImagesButton)

        # Progress bar for embedding
        self.embeddingProgressBar = qt.QProgressBar()
        self.embeddingProgressBar.hide()  # hidden until embedding starts
        actionsFormLayout.addRow("Embedding progress:", self.embeddingProgressBar)

        # Cluster & Plot button
        self.clusterPlotButton = qt.QPushButton("Cluster & Plot")
        self.clusterPlotButton.toolTip = "Perform recursive spectral clustering, then UMAP, then plot."
        self.clusterPlotButton.enabled = False  # only after embeddings are ready
        actionsFormLayout.addRow(self.clusterPlotButton)

        # Copy images button
        self.copyImagesButton = qt.QPushButton("Copy Clustered Images")
        self.copyImagesButton.toolTip = "Copy images into cluster-labeled subfolders."
        self.copyImagesButton.enabled = False  # only after cluster & plot
        actionsFormLayout.addRow(self.copyImagesButton)

        #
        # --- Collapsible Button: Clustering Results ---
        #
        self.resultsCollapsibleButton = ctk.ctkCollapsibleButton()
        self.resultsCollapsibleButton.text = "Clustering Results"
        self.resultsCollapsibleButton.collapsed = False
        self.layout.addWidget(self.resultsCollapsibleButton)

        resultsFormLayout = qt.QFormLayout(self.resultsCollapsibleButton)

        # A QTableWidget to show cluster name, # images
        self.clusterSummaryTable = qt.QTableWidget()
        self.clusterSummaryTable.setColumnCount(2)
        self.clusterSummaryTable.setHorizontalHeaderLabels(["Cluster", "Num Images"])
        self.clusterSummaryTable.horizontalHeader().setStretchLastSection(True)
        resultsFormLayout.addWidget(self.clusterSummaryTable)

        # Add vertical spacer
        self.layout.addStretch(1)

        #
        #   Initialize Logic
        #
        self.logic = ClusterPhotosLogic()

        # Connections
        self.loadModelButton.connect('clicked(bool)', self.onLoadModel)
        self.loadImagesButton.connect('clicked(bool)', self.onLoadImagesAndEmbed)
        self.clusterPlotButton.connect('clicked(bool)', self.onClusterAndPlot)
        self.copyImagesButton.connect('clicked(bool)', self.onCopyImages)

    def enter(self):
        """
        Called when the user enters this module. We'll create and set a custom
        chart-only layout so the Plot view is maximized.
        """
        self.setChartOnlyLayout()

    def setChartOnlyLayout(self):
        """
        Create a single-plot layout, register it with Slicer?s layout logic,
        and switch to it.
        """
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()

        customLayout = """
        <layout type="horizontal">
          <item>
            <view class="vtkMRMLPlotViewNode" singletontag="PlotView1">
              <property name="viewlabel" action="default">P</property>
            </view>
          </item>
        </layout>
        """

        # Add or overwrite the layout description with our custom layout
        self.chartOnlyLayoutId = 901  # ensure we keep the same ID
        layoutNode.AddLayoutDescription(self.chartOnlyLayoutId, customLayout)
        # Switch to that layout
        layoutNode.SetViewArrangement(self.chartOnlyLayoutId)

    def load_dependencies(self):
        """
        Ensure all needed Python dependencies are installed.
        """
        import logging
        import slicer

        # First check if the PyTorch extension is available
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.messageBox(
                "Photogrammetry requires the PyTorch extension. "
                "Please install it from the Extensions Manager."
            )

        torchLogic = None
        try:
            import PyTorchUtils
            torchLogic = PyTorchUtils.PyTorchUtilsLogic()
            if not torchLogic.torchInstalled():

                if not slicer.util.confirmOkCancelDisplay(
                        f"This module requires installation of additional Python packages. Installation needs network "
                        f"connection and may take several minutes. Click OK to proceed.",
                        "Confirm Python package installation",
                ):
                    raise InstallError("User cancelled.")

                logging.debug('Installing PyTorch via the PyTorch extension...')
                torch = torchLogic.installTorch(askConfirmation=True, forceComputationBackend='cu126')
                if torch:
                    restart = slicer.util.confirmYesNoDisplay(
                        "PyTorch dependencies have been installed. A restart of 3D Slicer is needed. Restart now?"
                    )
                    if restart:
                        slicer.util.restart()
                if torch is None:
                    slicer.util.messageBox('PyTorch must be installed manually to use this module.')
        except Exception as e:
            logging.warning(f"Could not complete PyTorch installation steps: {e}")

        # Pillow
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
        except ImportError:
            slicer.util.pip_install("pillow")
            from PIL import Image
            from PIL.ExifTags import TAGS

        # Transformers
        try:
            import transformers
        except ImportError:
            slicer.util.pip_install("transformers>4.29.2")
            import transformers

        # scikit-learn
        try:
            import sklearn
        except ImportError:
            slicer.util.pip_install("scikit-learn")
            import sklearn

        # umap-learn
        try:
            import umap
        except ImportError:
            slicer.util.pip_install("umap-learn")
            import umap

        # OpenCV (optional)
        try:
            import cv2
            if not hasattr(cv2, 'xfeatures2d'):
                raise ImportError("opencv-contrib-python is not properly installed")
        except ImportError:
            slicer.util.pip_install("opencv-python")
            slicer.util.pip_install("opencv-contrib-python")
            import cv2

    # -------------------------------------------------------------------------
    #   Event handlers
    # -------------------------------------------------------------------------

    def onLoadModel(self):
        """
        1) Loads the specified model into the logic.
        2) Disables 'Load Model' so it can't be run again.
        3) Enables 'Load Images & Embed' button.
        """
        modelName = self.modelNameEdit.text
        logging.info(f"Request to load model: {modelName}")

        try:
            self.logic.loadModel(modelName)
            slicer.util.infoDisplay("Model loaded successfully.")
            self.loadModelButton.enabled = False
            self.loadImagesButton.enabled = True
        except Exception as e:
            slicer.util.errorDisplay(f"Error loading model:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def onLoadImagesAndEmbed(self):
        """
        1) Reads images from folder.
        2) Computes embeddings using the logic's already-loaded model.
        3) Prints the shape of embeddings in the console.
        4) Enables 'Cluster & Plot' on success.
        """
        inputDir = self.inputDirSelector.currentPath
        if not os.path.isdir(inputDir):
            slicer.util.errorDisplay("Please select a valid folder containing images.")
            return

        logging.info(f"Loading images from: {inputDir}")
        try:
            # Show progress bar
            self.embeddingProgressBar.show()
            self.embeddingProgressBar.setValue(0)

            # Actually load and embed
            self.logic.loadImagesAndComputeEmbeddings(
                inputDir,
                kNeighbors=self.kNeighborsSpin.value,
                maxEigen=self.maxEigenSpin.value,
                maxClusterSize=self.maxClusterSizeSpin.value,
                umapNNeighbors=self.umapNNeighborsSpin.value,
                umapMinDist=self.umapMinDistSpin.value,
                progressBar=self.embeddingProgressBar
            )
            # Hide progress bar when done
            self.embeddingProgressBar.hide()

            slicer.util.infoDisplay("Images loaded and embeddings computed.")
            logging.info(f"Embedding shape: {self.logic.embeddings.shape}")

            self.clusterPlotButton.enabled = True  # now user can cluster
        except Exception as e:
            self.embeddingProgressBar.hide()  # hide on failure, too
            slicer.util.errorDisplay(f"Error computing embeddings:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def onClusterAndPlot(self):
        """
        Perform clustering (recursive spectral) and UMAP, then display a plot.
        Also update the cluster summary table. Then enable "Copy Images".
        We re-read the current UI parameters to allow repeated changes.
        """
        # --- Re-read parameters from UI for clustering ---
        # We'll update self.logic.parameters so the new settings are used.
        self.logic.parameters["kNeighbors"] = self.kNeighborsSpin.value
        self.logic.parameters["maxEigen"] = self.maxEigenSpin.value
        self.logic.parameters["maxClusterSize"] = self.maxClusterSizeSpin.value
        self.logic.parameters["umapNNeighbors"] = self.umapNNeighborsSpin.value
        self.logic.parameters["umapMinDist"] = self.umapMinDistSpin.value

        try:
            self.logic.clusterAndPlot()
            slicer.util.infoDisplay("Clustering complete. See Plot view for results.")
            self.updateClusterSummaryTable()
            self.copyImagesButton.enabled = True
        except Exception as e:
            slicer.util.errorDisplay(f"Error during clustering:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def onCopyImages(self):
        """
        Copies images into cluster-labeled subfolders based on the last clustering.
        """
        try:
            self.logic.copyClusteredImages()
        except Exception as e:
            slicer.util.errorDisplay(f"Error copying images:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    #   Helper for cluster summary table
    # -------------------------------------------------------------------------

    def updateClusterSummaryTable(self):
        """
        Clear and re-populate the clusterSummaryTable with cluster names, # images
        from the logic's finalClusters.
        """
        finalClusters = self.logic.finalClusters
        if not finalClusters:
            return

        # Clear existing rows
        self.clusterSummaryTable.clearContents()
        self.clusterSummaryTable.setRowCount(0)

        # We have 2 columns: "Cluster" and "Num Images"
        rowCount = len(finalClusters)
        self.clusterSummaryTable.setRowCount(rowCount)

        for i, clusterIndices in enumerate(finalClusters):
            clusterName = f"Cluster_{i}"
            numImages = len(clusterIndices)

            clusterNameItem = qt.QTableWidgetItem(clusterName)
            clusterCountItem = qt.QTableWidgetItem(str(numImages))

            self.clusterSummaryTable.setItem(i, 0, clusterNameItem)
            self.clusterSummaryTable.setItem(i, 1, clusterCountItem)


#
# ClusterPhotosLogic
#
class ClusterPhotosLogic(ScriptedLoadableModuleLogic):
    """
    Contains functionality for:
      - Loading a ViT model
      - Loading images, computing embeddings
      - Recursive spectral clustering
      - UMAP dimension reduction
      - Plotting results in Slicer
      - Copying final clusters to subfolders
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.parameters = {}

        # Will store model references after load
        self.model = None
        self.processor = None

        # Will store images, embeddings, cluster results
        self.imagePaths = []
        self.embeddings = None
        self.clusterAssignments = None
        self.finalClusters = None

    # -------------------------------------------------------------------------
    #   Step 1: Load Model
    # -------------------------------------------------------------------------

    def loadModel(self, modelName):
        """
        Loads the specified ViT model and sets self.model, self.processor.
        """
        import torch
        from transformers import ViTImageProcessor, ViTModel

        logging.info(f"Loading model: {modelName}")
        self.processor = ViTImageProcessor.from_pretrained(modelName)
        self.model = ViTModel.from_pretrained(modelName)
        self.model.eval()

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Model device: {device}")
        self.model.to(device)

    # -------------------------------------------------------------------------
    #   Step 2: Load Images & Compute Embeddings
    # -------------------------------------------------------------------------

    def loadImagesAndComputeEmbeddings(self, inputDir,
                                       kNeighbors=10,
                                       maxEigen=15,
                                       maxClusterSize=40,
                                       umapNNeighbors=10,
                                       umapMinDist=0.1,
                                       progressBar=None):
        """
        Loads images from the folder, computes embeddings using the loaded model.
        Also stores initial parameters for later use. The optional 'progressBar'
        is updated each time we finish an embedding.
        """
        if not self.model or not self.processor:
            raise RuntimeError("No model loaded. Please load model first.")

        # store parameters used at embedding time
        # (we will update them in onClusterAndPlot if the user changes them)
        self.parameters["inputDir"] = inputDir
        self.parameters["kNeighbors"] = kNeighbors
        self.parameters["maxEigen"] = maxEigen
        self.parameters["maxClusterSize"] = maxClusterSize
        self.parameters["umapNNeighbors"] = umapNNeighbors
        self.parameters["umapMinDist"] = umapMinDist

        # Gather image paths
        self.imagePaths = self.getImageFiles(inputDir)
        if len(self.imagePaths) == 0:
            raise ValueError(f"No images found in {inputDir}.")

        logging.info(f"Found {len(self.imagePaths)} images in {inputDir}")

        # Prepare progress bar
        if progressBar is not None:
            progressBar.setRange(0, len(self.imagePaths))
            progressBar.setValue(0)

        import torch
        import numpy as np

        device = self.model.device
        allEmbeddings = []
        total_images = len(self.imagePaths)
        for idx, path in enumerate(self.imagePaths):
            emb = self.getImageEmbedding(path, self.processor, self.model, device)
            allEmbeddings.append(emb)

            # Print console progress
            logging.info(f"Embedding: {idx+1} / {total_images}")

            # Update progress bar
            if progressBar is not None:
                progressBar.setValue(idx + 1)
                slicer.app.processEvents()

        self.embeddings = np.array(allEmbeddings, dtype=np.float32)
        logging.info(f"Embeddings shape: {self.embeddings.shape}")

    # -------------------------------------------------------------------------
    #   Step 3: Cluster & Plot
    # -------------------------------------------------------------------------

    def clusterAndPlot(self):
        """
        Runs recursive spectral clustering, does UMAP, and plots results,
        using whatever parameters are currently in self.parameters.
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings found. Please load images first.")

        # Read current parameters
        kNeighbors = self.parameters.get("kNeighbors", 10)
        maxEigen = self.parameters.get("maxEigen", 15)
        maxClusterSize = self.parameters.get("maxClusterSize", 40)
        umapNNeighbors = self.parameters.get("umapNNeighbors", 10)
        umapMinDist = self.parameters.get("umapMinDist", 0.1)

        # Perform recursive spectral clustering
        logging.info("Starting recursive spectral clustering...")
        self.finalClusters = self.splitClusterSpectral(
            X=self.embeddings,
            idxs=list(range(self.embeddings.shape[0])),
            kNeighbors=kNeighbors,
            maxEigen=maxEigen,
            maxClusterSize=maxClusterSize
        )
        logging.info(f"Total final clusters: {len(self.finalClusters)}")

        # Build cluster assignment array
        import numpy as np
        self.clusterAssignments = np.zeros(len(self.imagePaths), dtype=int)
        for cIdx, cList in enumerate(self.finalClusters):
            for ix in cList:
                self.clusterAssignments[ix] = cIdx

        # UMAP
        logging.info("Running UMAP for visualization...")
        import umap
        X_2d = umap.UMAP(
            n_neighbors=umapNNeighbors,
            min_dist=umapMinDist,
            n_components=2
        ).fit_transform(self.embeddings)

        # Plot
        self.plotClusteringInSlicer(X_2d, self.clusterAssignments)
        logging.info("Clustering complete.")

    # -------------------------------------------------------------------------
    #   Step 4: Copy Clustered Images
    # -------------------------------------------------------------------------

    def copyClusteredImages(self):
        """
        Copies images into subfolders labeled by cluster.
        """
        if self.finalClusters is None:
            raise RuntimeError("Must run clustering before copying images.")

        inputDir = self.parameters["inputDir"]
        outDir = inputDir.rstrip("/\\") + "_clustered"

        if not os.path.exists(outDir):
            os.makedirs(outDir)

        logging.info(f"Copying final clusters to: {outDir}")
        for i, cluster_idxs in enumerate(self.finalClusters):
            cluster_folder = os.path.join(outDir, f"cluster_{i:03d}")
            os.makedirs(cluster_folder, exist_ok=True)
            for idx in cluster_idxs:
                src_img_path = self.imagePaths[idx]
                filename = os.path.basename(src_img_path)
                dst_img_path = os.path.join(cluster_folder, filename)
                shutil.copy2(src_img_path, dst_img_path)

            logging.info(f"  Cluster {i:03d} - {len(cluster_idxs)} images copied.")

        slicer.util.infoDisplay(f"Copied images into {outDir}")

    # -----------------------------------------------------------------------
    #   Internal helper methods
    # -----------------------------------------------------------------------

    def getImageFiles(self, inputDir, exts=("jpg", "jpeg", "png")):
        """
        Recursively find images in input_dir with specified extensions.
        """
        import glob
        files = []
        for ext in exts:
            pattern = os.path.join(inputDir, "**", f"*.{ext}")
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(files)

    def getImageEmbedding(self, imagePath, processor, model, device):
        """
        Extract [CLS] token from a ViT model for the given image.
        """
        import torch
        from PIL import Image

        im = Image.open(imagePath).convert("RGB")
        inputs = processor(images=im, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[0, 0]
        return cls_emb.cpu().numpy()

    def buildKnnGraph(self, X, k=10):
        """
        Build a kNN-based similarity (affinity) matrix using an RBF kernel
        on distances. Returns a dense NxN matrix W.
        """
        import math
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)

        avg_dist = np.mean(distances[:, 1:])  # ignoring self-dist
        sigma = avg_dist if avg_dist > 0 else 1.0

        W = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            for j_idx in range(k):
                j = indices[i, j_idx]
                dist_ij = distances[i, j_idx]
                sim_ij = math.exp(-(dist_ij ** 2) / (2 * (sigma ** 2)))
                W[i, j] = sim_ij
                W[j, i] = sim_ij
        return W

    def spectralClusteringEigengap(self, W, maxEigen=15):
        """
        1) Compute normalized Laplacian L.
        2) Compute first maxEigen eigenvals/vecs.
        3) Determine #clusters with largest eigengap.
        4) K-means on top-k eigenvectors => cluster labels.
        """
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        n = W.shape[0]
        d = np.sum(W, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-8))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        eigvals, eigvecs = np.linalg.eigh(L)
        sort_idx = np.argsort(eigvals)
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # just take first maxEigen
        eigvals = eigvals[:maxEigen]
        eigvecs = eigvecs[:, :maxEigen]

        # compute the largest gap in consecutive eigenvalues
        gaps = np.diff(eigvals)
        best_k = np.argmax(gaps) + 1  # +1 for 1-based index
        # clamp to [2, maxEigen]
        best_k = max(2, min(best_k, maxEigen))

        # row-normalize the top-k eigenvectors
        U = eigvecs[:, :best_k]
        U = normalize(U, axis=1, norm='l2')

        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(U)
        return labels, best_k

    def splitClusterSpectral(self, X, idxs, kNeighbors=10, maxEigen=15, maxClusterSize=40, depth=0):
        """
        Recursively split the subset of data (X[idxs]) using spectral clustering
        + eigengap until the cluster has <= maxClusterSize images.
        """
        import numpy as np
        n = len(idxs)
        if n <= maxClusterSize:
            return [idxs]

        indent = "  " * depth
        logging.info(f"{indent}Splitting {n} samples with spectral clustering...")

        X_sub = X[idxs]
        W = self.buildKnnGraph(X_sub, k=kNeighbors)

        labels_sub, k_sub = self.spectralClusteringEigengap(W, maxEigen=maxEigen)
        logging.info(f"{indent}  => Found {k_sub} clusters at this level.")

        final_clusters = []
        for c_id in range(k_sub):
            sub_idxs = [idxs[i] for i in range(n) if labels_sub[i] == c_id]
            if len(sub_idxs) > maxClusterSize:
                # recursively split again
                final_clusters.extend(self.splitClusterSpectral(
                    X, sub_idxs, kNeighbors, maxEigen, maxClusterSize, depth + 1
                ))
            else:
                final_clusters.append(sub_idxs)

        return final_clusters

    def plotClusteringInSlicer(self, X_2d, cluster_assignments):
        """
        Show the clustering result in a Slicer Plot View. We create a separate
        table and plot series for each cluster (compatible with older Slicer versions).
        """
        import vtk
        import numpy as np

        num_points = X_2d.shape[0]
        num_clusters = np.max(cluster_assignments) + 1

        # Create a PlotChart node
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "UMAP Chart")
        plotChartNode.SetTitle("UMAP of Recursive Spectral Clustering")
        plotChartNode.SetXAxisTitle("UMAP-1")
        plotChartNode.SetYAxisTitle("UMAP-2")

        # Instead of one big table with filtering, create a separate table per cluster
        for c_idx in range(num_clusters):
            # gather points for cluster c_idx
            c_indices = np.where(cluster_assignments == c_idx)[0]
            cluster_size = len(c_indices)

            # Build a new table just for these points
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"UMAP Table {c_idx}")
            table = tableNode.GetTable()

            arrX = vtk.vtkDoubleArray()
            arrX.SetName("X")
            arrX.SetNumberOfValues(cluster_size)

            arrY = vtk.vtkDoubleArray()
            arrY.SetName("Y")
            arrY.SetNumberOfValues(cluster_size)

            for i, idx in enumerate(c_indices):
                arrX.SetValue(i, float(X_2d[idx, 0]))
                arrY.SetValue(i, float(X_2d[idx, 1]))

            table.AddColumn(arrX)
            table.AddColumn(arrY)
            table.SetNumberOfRows(cluster_size)

            for i in range(cluster_size):
                table.SetValue(i, 0, arrX.GetValue(i))
                table.SetValue(i, 1, arrY.GetValue(i))

            # Create a PlotSeriesNode for this cluster
            plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLPlotSeriesNode", f"Cluster {c_idx}"
            )
            plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
            plotSeriesNode.SetXColumnName("X")
            plotSeriesNode.SetYColumnName("Y")
            plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
            plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
            plotSeriesNode.SetUniqueColor()  # auto-assign color

            # add the series to the chart
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())

        # Switch to our custom chart-only layout (if not already set)
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        layoutNode.SetViewArrangement(901)  # same ID as self.chartOnlyLayoutId

        # Show the chart in a plot widget
        slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)

        logging.info("UMAP plot created in Slicer Plot view.")
