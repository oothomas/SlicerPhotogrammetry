# -*- coding: utf-8 -*-
"""
SamuraiVideoMasking (blocking/main-thread setup)
------------------------------------------------
All work runs on the main Qt thread (no worker threads). UI may freeze during setup.

Features
- Collapsible "SAMURAI Setup"
- Collapsible "Video Prep" (MOV?MP4 + frame extraction)
- Collapsible "ROI & Tracking" (checkpoint/device, Load Frames, Select ROI on First Frame,
  Finalize ROI (Save BBox), Run SAMURAI Masking)
- On module entry, set layout to One Up Red Slice (single viewer).

GPU setup policy (aligns with Photogrammetry approach, using cu126 per request):
- Install torch/torchvision via Slicer's PyTorchUtils with CUDA 12.6:
    PyTorchUtils.PyTorchUtilsLogic().installTorch(..., forceComputationBackend='cu126')
- Do not import torch until AFTER we have exposed wheel-provided CUDA libs to the linker
  (site-packages/nvidia/*/lib + torch/lib) to avoid libcudnn_graph crashes.
"""

import os
import sys
import platform
import subprocess
import shlex
import time
import shutil
from pathlib import Path

import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


class SamuraiVideoMasking(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "SamuraiVideoMasking"
        parent.categories = ["SlicerMorph.Photogrammetry"]
        parent.dependencies = []
        parent.contributors = ["Oshane Thomas (SCRI)"]
        parent.helpText = (
            "Clone and set up the SAMURAI repo for use in this module, prepare a video "
            "(optional MOV?MP4 conversion and frame extraction), then select an ROI and run SAMURAI tracking."
        )
        parent.acknowledgementText = (
            "This module was developed with support from the National Science "
            "Foundation under grants DBI/2301405 and OAC/2118240 awarded to AMM at "
            "Seattle Children's Research Institute."
        )


class SamuraiVideoMaskingWidget(ScriptedLoadableModuleWidget):

    DEFAULT_REPO_URL = "https://github.com/yangchris11/samurai.git"

    SETTINGS_KEY = "SamuraiVideoMasking"
    SETTINGS_INSTALLED = f"{SETTINGS_KEY}/installed"
    SETTINGS_REPO_PATH = f"{SETTINGS_KEY}/repoPath"

    SETTINGS_LAST_VIDEO = f"{SETTINGS_KEY}/lastVideo"
    SETTINGS_LAST_MP4 = f"{SETTINGS_KEY}/lastMp4"
    SETTINGS_LAST_FRAMES = f"{SETTINGS_KEY}/lastFramesDir"

    SETTINGS_CKPT_PATH = f"{SETTINGS_KEY}/ckptPath"
    SETTINGS_DEVICE = f"{SETTINGS_KEY}/device"
    SETTINGS_BBOX = f"{SETTINGS_KEY}/bbox_xywh"  # "x,y,w,h" (in ORIGINAL/orientation coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic = None
        # Session state
        self.framesBuffer = None
        self.masksBuffer = None
        self.bbox_xywh = None

        # Nodes / viewer
        self._firstFrameVectorNode = None
        self._roiNode = None
        self._roiObserverTag = None

        # UI handles
        self.finalizeROIBtn = None
        self.trackBtn = None

        # Early log buffering before logEdit exists
        self._earlyLogs = []

    # ---------- Paths ----------
    def moduleDir(self) -> Path:
        return Path(os.path.dirname(slicer.modules.samuraivideomasking.path))

    def supportDir(self) -> Path:
        return self.moduleDir() / "Support"

    def defaultCloneDir(self) -> Path:
        return self.supportDir() / "samurai"

    # ---------- Logging (safe before UI) ----------
    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        if hasattr(self, "logEdit") and isinstance(getattr(self, "logEdit", None), qt.QPlainTextEdit):
            if getattr(self, "_earlyLogs", None):
                for t, msg in self._earlyLogs:
                    self.logEdit.appendPlainText(f"[{t}] {msg}")
                self._earlyLogs = []
            self.logEdit.appendPlainText(f"[{ts}] {text}")
            self.logEdit.moveCursor(qt.QTextCursor.End)
            slicer.app.processEvents()
        else:
            self._earlyLogs.append((ts, text))
            try:
                print(f"[{ts}] {text}")
            except Exception:
                pass

    # ---------- CUDA/NVIDIA lib path exposure ----------
    def _prepend_path(self, var: str, path: str):
        if not path or not os.path.isdir(path):
            return
        cur = os.environ.get(var, "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if path not in parts:
            os.environ[var] = path + (os.pathsep + cur if cur else "")

    def _discover_nvidia_lib_dirs(self) -> list:
        import site
        roots = []
        try:
            roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            us = site.getusersitepackages()
            if us:
                roots.append(us)
        except Exception:
            pass
        for p in sys.path:
            if "site-packages" in p and p not in roots:
                roots.append(p)

        libdirs = set()
        for r in roots:
            base = Path(r) / "nvidia"
            if not base.is_dir():
                continue
            for sub in base.iterdir():
                d = sub / "lib"
                if d.is_dir():
                    libdirs.add(str(d))
        try:
            import torch  # noqa: F401
            tlib = Path(torch.__file__).parent / "lib"
            if tlib.is_dir():
                libdirs.add(str(tlib))
        except Exception:
            pass
        return sorted(libdirs)

    def _prepare_cuda_runtime_visibility(self, log=True):
        """Expose CUDA *runtime wheels* and the *driver* libcuda before importing torch."""
        if not sys.platform.startswith("linux"):
            return
        # 1) Wheel-provided CUDA libs (cuDNN, cuBLAS, NCCL, etc.)
        for d in self._discover_nvidia_lib_dirs():
            self._prepend_path("LD_LIBRARY_PATH", d)

        # 2) System driver libcuda.so.1 (this is what flips torch.cuda.is_available())
        # Try ldconfig first
        driver_dirs = set()
        try:
            out = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True)
            for line in out.splitlines():
                if "libcuda.so.1" in line:
                    p = line.split("=>")[-1].strip()
                    drv = os.path.dirname(p)
                    if os.path.isdir(drv):
                        driver_dirs.add(drv)
        except Exception:
            pass
        # Fallback common locations
        for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib/wsl/lib", "/usr/local/nvidia/lib64"]:
            if os.path.exists(os.path.join(d, "libcuda.so.1")):
                driver_dirs.add(d)

        for d in sorted(driver_dirs):
            self._prepend_path("LD_LIBRARY_PATH", d)

        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        os.environ.setdefault("NCCL_LAUNCH_MODE", "GROUP")

        if log:
            self._log("CUDA runtime search paths prepared:")
            # print only unique dirs we actually appended
            printed = set()
            for var in ["LD_LIBRARY_PATH"]:
                for part in os.environ.get(var, "").split(os.pathsep):
                    if part and part not in printed and os.path.isdir(part):
                        printed.add(part)
                        if any(seg in part for seg in ("nvidia", "torch/lib", "x86_64-linux-gnu", "lib64")):
                            self._log(f"  ? {part}")

    # ---------- Lifecycle ----------
    def enter(self):
        try:
            lm = slicer.app.layoutManager()
            if lm:
                lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        except Exception as e:
            print(f"[SamuraiVideoMasking] WARN: could not set layout: {e}")

    # ---------- UI ----------
    def setup(self):
        super().setup()
        self.logic = SamuraiVideoMaskingLogic()

        # === SAMURAI Setup ===
        box = ctk.ctkCollapsibleButton()
        box.text = "SAMURAI Setup"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.urlEdit = qt.QLineEdit(self.DEFAULT_REPO_URL)
        form.addRow("Repository URL:", self.urlEdit)

        self.destPathEdit = qt.QLineEdit(str(self.defaultCloneDir()))
        self.destPathEdit.readOnly = True
        form.addRow("Destination:", self.destPathEdit)

        link = qt.QLabel(f'<a href="{self.DEFAULT_REPO_URL}">Open SAMURAI on GitHub</a>')
        link.setOpenExternalLinks(True)
        form.addRow("", link)

        row = qt.QHBoxLayout()
        self.configureBtn = qt.QPushButton("Configure SAMURAI (Blocking)")
        self.verifyBtn = qt.QPushButton("Verify Installation")
        self.openFolderBtn = qt.QPushButton("Open Support Folder")
        row.addWidget(self.configureBtn)
        row.addWidget(self.verifyBtn)
        row.addWidget(self.openFolderBtn)
        form.addRow(row)

        self.statusLabel = qt.QLabel("Status: Idle")
        self.statusLabel.setStyleSheet("color: #BBB;")
        form.addRow(self.statusLabel)

        self.logEdit = qt.QPlainTextEdit()
        self.logEdit.setReadOnly(True)
        mono = qt.QFontDatabase.systemFont(qt.QFontDatabase.FixedFont)
        self.logEdit.setFont(mono)
        self.logEdit.setMinimumHeight(240)
        form.addRow("Log:", self.logEdit)

        # Flush early logs captured before logEdit existed
        if getattr(self, "_earlyLogs", None):
            for ts, msg in self._earlyLogs:
                self.logEdit.appendPlainText(f"[{ts}] {msg}")
            self._earlyLogs = []

        note = qt.QLabel(
            "<i>GPU setup:</i> Configure installs a <b>cu126</b> build of PyTorch via <b>PyTorchUtils</b>, "
            "clones the SAMURAI repo, installs <b>sam2</b> (editable) and dependencies, and runs the checkpoint script if present."
        )
        note.wordWrap = True
        form.addRow(note)

        self.configureBtn.clicked.connect(self.onConfigureClicked)
        self.verifyBtn.clicked.connect(self.onVerifyClicked)
        self.openFolderBtn.clicked.connect(self.onOpenFolderClicked)

        # Persisted destination path
        self.supportDir().mkdir(parents=True, exist_ok=True)
        s = qt.QSettings()
        if s.contains(self.SETTINGS_REPO_PATH):
            saved = s.value(self.SETTINGS_REPO_PATH)
            if saved:
                self.destPathEdit.setText(saved)

        self._log("Click 'Configure SAMURAI' to clone/install everything (cu126). Then 'Verify Installation' to populate devices.")

        # === Video Prep ===
        vbox = ctk.ctkCollapsibleButton()
        vbox.text = "Video Prep (MOV?MP4 + Frame Extraction)"
        self.layout.addWidget(vbox)
        vform = qt.QFormLayout(vbox)

        vrow1 = qt.QHBoxLayout()
        self.videoPathEdit = qt.QLineEdit()
        self.videoPathEdit.setPlaceholderText("Choose a .mov or .mp4 file?")
        self.videoBrowseBtn = qt.QPushButton("Browse?")
        vrow1.addWidget(self.videoPathEdit, 1)
        vrow1.addWidget(self.videoBrowseBtn)
        vform.addRow("Source video:", vrow1)

        self.mp4PathEdit = qt.QLineEdit()
        self.mp4PathEdit.readOnly = True
        vform.addRow("Target .mp4:", self.mp4PathEdit)

        self.framesDirEdit = qt.QLineEdit()
        self.framesDirEdit.readOnly = True
        vform.addRow("Frames folder:", self.framesDirEdit)

        self.loadVideoBtn = qt.QPushButton("Load Video (Convert if MOV, then Extract Frames)")
        vform.addRow(self.loadVideoBtn)

        self.videoBrowseBtn.clicked.connect(self.onBrowseVideo)
        self.loadVideoBtn.clicked.connect(self.onLoadVideo)

        if s.contains(self.SETTINGS_LAST_VIDEO):
            self.videoPathEdit.setText(s.value(self.SETTINGS_LAST_VIDEO))
        self._computeDerivedVideoPaths()
        if s.contains(self.SETTINGS_LAST_MP4):
            self.mp4PathEdit.setText(s.value(self.SETTINGS_LAST_MP4))
        if s.contains(self.SETTINGS_LAST_FRAMES):
            self.framesDirEdit.setText(s.value(self.SETTINGS_LAST_FRAMES))

        # === ROI & Tracking ===
        tbox = ctk.ctkCollapsibleButton()
        tbox.text = "ROI & Tracking (SAMURAI)"
        self.layout.addWidget(tbox)
        tform = qt.QFormLayout(tbox)

        trow1 = qt.QHBoxLayout()
        self.ckptEdit = qt.QLineEdit()
        self.ckptEdit.setPlaceholderText("Select a SAM 2.1 checkpoint (e.g., sam2.1_hiera_large.pt)")
        self.ckptBrowseBtn = qt.QPushButton("Browse?")
        trow1.addWidget(self.ckptEdit, 1)
        trow1.addWidget(self.ckptBrowseBtn)
        tform.addRow("Checkpoint:", trow1)

        if s.contains(self.SETTINGS_CKPT_PATH):
            self.ckptEdit.setText(s.value(self.SETTINGS_CKPT_PATH))

        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItem("cpu")
        if s.contains(self.SETTINGS_DEVICE):
            try:
                self.deviceCombo.setCurrentText(s.value(self.SETTINGS_DEVICE))
            except Exception:
                pass
        tform.addRow("Device:", self.deviceCombo)

        self.keepInMemCheck = qt.QCheckBox("Keep frames in memory (faster post-processing; uses RAM)")
        self.keepInMemCheck.setChecked(True)
        tform.addRow(self.keepInMemCheck)

        trow2 = qt.QHBoxLayout()
        self.loadFramesBtn = qt.QPushButton("Load Frames From Folder")
        self.roiBtn = qt.QPushButton("Select ROI on First Frame")
        self.finalizeROIBtn = qt.QPushButton("Finalize ROI (Save BBox)")
        self.finalizeROIBtn.setEnabled(False)
        self.trackBtn = qt.QPushButton("Run SAMURAI Masking")
        self.trackBtn.setEnabled(False)
        trow2.addWidget(self.loadFramesBtn)
        trow2.addWidget(self.roiBtn)
        trow2.addWidget(self.finalizeROIBtn)
        trow2.addWidget(self.trackBtn)
        tform.addRow(trow2)

        self.ckptBrowseBtn.clicked.connect(self.onBrowseCkpt)
        self.loadFramesBtn.clicked.connect(self.onLoadFrames)
        self.roiBtn.clicked.connect(self.onSelectROI)
        self.finalizeROIBtn.clicked.connect(self.onFinalizeROI)
        self.trackBtn.clicked.connect(self.onRunTracking)

        if s.contains(self.SETTINGS_BBOX):
            try:
                x, y, w_, h_ = [int(v) for v in str(s.value(self.SETTINGS_BBOX)).split(",")]
                self.bbox_xywh = (x, y, w_, h_)
                self._log(f"Restored ROI bbox (x,y,w,h) = {self.bbox_xywh}")
                self.trackBtn.setEnabled(True)
            except Exception:
                pass

        self.layout.addStretch(1)

    # ---------- Helper UI ----------
    def _setBusy(self, busy: bool):
        for w in (self.configureBtn, self.verifyBtn, self.openFolderBtn,
                  self.videoBrowseBtn, self.loadVideoBtn, self.urlEdit,
                  self.ckptBrowseBtn, self.loadFramesBtn, self.roiBtn, self.finalizeROIBtn, self.trackBtn,
                  self.deviceCombo, self.keepInMemCheck):
            w.setEnabled(not busy)
        self.statusLabel.setText(f"Status: {'Working?' if busy else 'Idle'}")
        self.statusLabel.setStyleSheet("color: #f5c542;" if busy else "color: #BBB;")
        slicer.app.processEvents()

    def _comboText(self, combo: qt.QComboBox) -> str:
        try:
            val = combo.currentText
            if callable(val):
                val = val()
            return str(val)
        except Exception:
            try:
                return str(combo.currentText())
            except Exception:
                return str(combo.currentText)

    # ---------- Configure: clone + deps + torch cu126 ----------
    def onOpenFolderClicked(self):
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(self.supportDir())))

    def _run_cmd_blocking(self, args, cwd=None):
        if isinstance(args, str):
            args = shlex.split(args)
        p = subprocess.Popen(
            args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in p.stdout:
            if line:
                self._log(line.rstrip("\n"))
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(args)}")
        return True

    def _clone_or_update_samurai(self, repo_url: str, dest_dir: Path):
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if (dest_dir / ".git").exists():
            self._log("Repo exists. Fetching latest?")
            self._run_cmd_blocking(["git", "-C", str(dest_dir), "fetch", "--all"])
            self._run_cmd_blocking(["git", "-C", str(dest_dir), "pull", "--ff-only"])
        else:
            self._log(f"Cloning {repo_url} ? {dest_dir}")
            self._run_cmd_blocking(["git", "clone", "--depth", "1", repo_url, str(dest_dir)])
        if (dest_dir / ".gitmodules").exists():
            self._log("Initializing submodules?")
            self._run_cmd_blocking(["git", "-C", str(dest_dir), "submodule", "update", "--init", "--recursive"])

    def _pip(self, spec: str, desc: str = None):
        if desc:
            self._log(desc)
        ok, out = True, ""
        try:
            ret = slicer.util.pip_install(spec)
            if isinstance(ret, tuple):
                ok, out = bool(ret[0]), str(ret[1] or "")
            else:
                ok, out = True, str(ret or "")
        except Exception as e:
            ok, out = False, str(e)
        if out:
            for line in str(out).splitlines():
                self._log(line)
        if not ok:
            raise RuntimeError(f"pip install failed: {spec}")
        return True

    def _install_python_deps(self, repo_dir: Path):
        """
        Install SAM-2 (editable) and its runtime deps, including video backends.
        """
        # Install sam2 editable if present
        sam2_dir = repo_dir / "sam2"
        if sam2_dir.is_dir():
            self._pip(f'-e "{sam2_dir}"', desc="Installing SAM2 (editable)?")
        else:
            self._log("WARNING: 'sam2' directory not found; installing repo root as editable instead.")
            self._pip(f'-e "{repo_dir}"', desc="Installing repo (editable)?")

        # Core deps used in your flow
        base = [
            "hydra-core",
            "omegaconf",
            "iopath",
            "loguru",
            "pandas",
            "scipy",
            "opencv-python",
            "jpeg4py",
            "lmdb",
        ]
        for pkg in base:
            self._pip(pkg, desc=f"pip install {pkg}?")

        # ? NEW: video I/O backends (decord preferred)
        self._ensure_video_backends()

        # Quick sanity: sam2 import should succeed now
        try:
            import importlib, sam2  # noqa: F401
            importlib.reload(sam2)
            self._log(f"sam2 OK: {sam2.__file__}")
        except Exception as e:
            self._log(f"sam2 import failed even after install: {e}")
            raise

    def _maybe_run_checkpoint_script(self, repo_dir: Path):
        ckpt_dir = repo_dir / "checkpoints"
        script_sh = ckpt_dir / "download_ckpts.sh"
        if ckpt_dir.exists() and script_sh.exists():
            if platform.system() in ("Linux", "Darwin"):
                self._log("Attempting checkpoint download via bash script?")
                try:
                    self._run_cmd_blocking(["bash", str(script_sh)], cwd=str(ckpt_dir))
                except Exception as e:
                    self._log(f"WARNING: Could not run checkpoint script automatically: {e}")
                    self._log(f"Manual: cd {ckpt_dir} && bash {script_sh.name}")
            else:
                self._log("Windows detected. Run checkpoint script in WSL or follow repo docs.")
                self._log(f"Manual: open folder {ckpt_dir}")
        else:
            self._log("No checkpoint script found. Place checkpoints manually if required by your model.")

    def _ensure_torch_cu126(self) -> bool:
        try:
            import PyTorchUtils  # noqa: F401
        except ModuleNotFoundError:
            slicer.util.messageBox(
                "This module expects the 'PyTorch Utils' extension.\n"
                "Install it from Extensions Manager, then return here."
            )
            self._log("PyTorchUtils not found.")
            return False

        try:
            import PyTorchUtils
            torchLogic = PyTorchUtils.PyTorchUtilsLogic()
            if not torchLogic.torchInstalled():
                if not slicer.util.confirmOkCancelDisplay(
                    "SAMURAI requires PyTorch (cu126). Install via PyTorch Utils now?",
                    "Install PyTorch (cu126)"
                ):
                    self._log("User cancelled PyTorch install.")
                    return False
                self._log("Installing PyTorch via PyTorch Utils (cu126)?")
                try:
                    torch_module = torchLogic.installTorch(askConfirmation=True, forceComputationBackend='cu126')
                except TypeError:
                    slicer.util.messageBox(
                        "This PyTorchUtils build doesn?t support 'cu126'. Update the extension/Slicer Nightly."
                    )
                    self._log("PyTorchUtils lacks cu126 backend.")
                    return False
                if torch_module:
                    if slicer.util.confirmYesNoDisplay(
                        "PyTorch installed. Slicer must restart to finalize. Restart now?"
                    ):
                        self._log("Restarting Slicer to finalize PyTorch install.")
                        slicer.util.restart()
                        return False
                    else:
                        self._log("Restart postponed. Torch unavailable until restart.")
                        return False
                else:
                    self._log("PyTorch install returned no module. Aborting.")
                    return False
            else:
                try:
                    import torch
                    cu = getattr(torch.version, "cuda", None)
                    if cu and not str(cu).startswith("12.6"):
                        self._log(f"WARNING: Torch CUDA {cu} detected; requested cu126.")
                except Exception as e:
                    self._log(f"torch probe failed: {e}")
                return True
        except Exception as e:
            self._log(f"PyTorchUtils failed: {e}")
            return False

    def onConfigureClicked(self):
        repo_url = self.urlEdit.text.strip() or self.DEFAULT_REPO_URL
        dest_dir = Path(self.destPathEdit.text.strip() or str(self.defaultCloneDir()))
        self.supportDir().mkdir(parents=True, exist_ok=True)

        if not self.logic.git_available():
            self._log("ERROR: Git is required. Install Git and ensure it's on PATH.")
            return

        if not slicer.util.confirmOkCancelDisplay(
            "Setup will run on the main thread and may freeze the UI.\nProceed?",
            "Blocking Setup"
        ):
            self._log("User cancelled.")
            return

        self._setBusy(True)
        try:
            # 1) Torch/cu126 first (may require restart)
            if not self._ensure_torch_cu126():
                self._setBusy(False)
                return

            # 2) Clone/update SAMURAI
            self._clone_or_update_samurai(repo_url, dest_dir)

            # 3) Install Python deps + editable sam2
            self._install_python_deps(dest_dir)

            # 4) Optional: run checkpoint script
            self._maybe_run_checkpoint_script(dest_dir)

            # Persist repo path
            qt.QSettings().setValue(self.SETTINGS_REPO_PATH, str(dest_dir))
            qt.QSettings().setValue(self.SETTINGS_INSTALLED, True)
            self._log("Configure complete. Now click 'Verify Installation'.")
        except Exception as e:
            self._log(f"Configuration failed: {e}")
            slicer.util.errorDisplay(f"Configuration failed:\n{e}")
        finally:
            self._setBusy(False)

    # ---------- Verify (safe torch import) ----------
    def _envTorchInfo(self):
        try:
            import torch  # noqa: F401
            import torch.backends.cudnn as cudnn
            cudnn_ver = None
            try:
                cudnn_ver = cudnn.version()
            except Exception:
                pass
            return f"torch={torch.__version__} cuda={getattr(torch.version,'cuda',None)} cudnn={cudnn_ver}"
        except Exception as e:
            return f"(torch unavailable: {e})"

    def _refreshDeviceList(self):
        self._prepare_cuda_runtime_visibility(log=True)  # ensure libs first
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        # Remove old cuda entries
        for i in reversed(range(self.deviceCombo.count)):
            t = self.deviceCombo.itemText(i)
            if isinstance(t, str) and t.lower().startswith("cuda"):
                self.deviceCombo.removeItem(i)
        if has_cuda:
            self.deviceCombo.insertItem(0, "cuda:0")
            cur = self._comboText(self.deviceCombo).strip().lower()
            if cur == "cpu":
                self.deviceCombo.setCurrentIndex(0)
        self._log(f"Devices updated. CUDA available={has_cuda}. {self._envTorchInfo()}")

    def onVerifyClicked(self):
        self._setBusy(True)
        try:
            self._log("Verifying core imports (no early torch import)?")
            self._prepare_cuda_runtime_visibility(log=True)

            checks = [
                ("sam2", "import sam2 as m; getattr(m, '__version__', 'OK')"),
                ("hydra-core", "import hydra as m; getattr(m, '__version__', 'OK')"),
                ("omegaconf", "import omegaconf as m; getattr(m, '__version__', 'OK')"),
                ("iopath", "import iopath as m; getattr(m, '__version__', 'OK')"),
                ("opencv-python", "import cv2 as m; m.__version__"),
                ("decord", "import decord as m; getattr(m, '__version__', 'OK')"),
                ("torch", "import torch as m; (m.__version__, getattr(m.version,'cuda',None))"),
                ("torchvision", "import torchvision as m; m.__version__"),
            ]
            ok_all = True
            for label, code in checks:
                try:
                    ns = {}
                    exec(code, {}, ns)
                    self._log(f"OK: {label} -> {list(ns.values())[-1]}")
                except Exception as e:
                    ok_all = False
                    self._log(f"FAIL: {label} import error -> {e}")

            if ok_all:
                self._refreshDeviceList()
                self._log("Verification passed.")
            else:
                self._log("Verification finished with errors. If 'decord' failed, click Configure again.")
        finally:
            self._setBusy(False)

    # ---------- Video Prep ----------
    def onBrowseVideo(self):
        startDir = os.path.dirname(self.videoPathEdit.text) if self.videoPathEdit.text else str(Path.home())
        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select video",
            startDir,
            "Video files (*.mov *.MOV *.mp4 *.MP4);;All files (*)"
        )
        if not filePath:
            return
        self.videoPathEdit.setText(filePath)
        self._computeDerivedVideoPaths()
        s = qt.QSettings()
        s.setValue(self.SETTINGS_LAST_VIDEO, filePath)
        s.setValue(self.SETTINGS_LAST_MP4, self.mp4PathEdit.text)
        s.setValue(self.SETTINGS_LAST_FRAMES, self.framesDirEdit.text)

    def _computeDerivedVideoPaths(self):
        src = self.videoPathEdit.text.strip()
        if not src:
            self.mp4PathEdit.setText("")
            self.framesDirEdit.setText("")
        else:
            p = Path(src)
            self.mp4PathEdit.setText(str(p.with_suffix(".mp4")))
            self.framesDirEdit.setText(str(p.parent / p.stem))

    def onLoadVideo(self):
        src = self.videoPathEdit.text.strip()
        if not src:
            slicer.util.messageBox("Please choose a source video first.")
            return
        p = Path(src)
        if not p.exists():
            slicer.util.messageBox(f"Video not found:\n{src}")
            return

        target_mp4 = Path(self.mp4PathEdit.text.strip() or (str(p.with_suffix(".mp4"))))
        frames_dir = Path(self.framesDirEdit.text.strip() or (str(p.parent / p.stem)))

        if not slicer.util.confirmOkCancelDisplay(
            "Video prep will run on the main thread and may freeze the UI.\nProceed?",
            "Blocking Video Prep"
        ):
            return

        self._setBusy(True)
        try:
            if p.suffix.lower() == ".mov":
                self._log(f"Converting MOV ? MP4:\n{p}  ?  {target_mp4}")
                self._mov_to_mp4_blocking(str(p), str(target_mp4))
                self._log("Conversion complete.")
            else:
                if p.suffix.lower() == ".mp4":
                    target_mp4 = p
                self._log(f"Using MP4 input: {target_mp4}")

            if frames_dir.exists():
                if slicer.util.confirmYesNoDisplay(
                    f"Frames folder exists:\n{frames_dir}\n\nDelete contents and re-extract?",
                    "Frames folder exists"
                ):
                    self._log(f"Cleaning frames folder: {frames_dir}")
                    self._safe_empty_dir(frames_dir)
                else:
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    frames_dir = frames_dir.parent / f"{frames_dir.name}_{ts}"
                    self._log(f"Using new frames folder: {frames_dir}")

            frames_dir.mkdir(parents=True, exist_ok=True)
            self.framesDirEdit.setText(str(frames_dir))

            self._log(f"Extracting frames ? {frames_dir}")
            n = self._extract_frames_blocking(str(target_mp4), str(frames_dir))
            self._log(f"Done. Extracted {n} frames.")

            s = qt.QSettings()
            s.setValue(self.SETTINGS_LAST_VIDEO, str(p))
            s.setValue(self.SETTINGS_LAST_MP4, str(target_mp4))
            s.setValue(self.SETTINGS_LAST_FRAMES, str(frames_dir))

        except Exception as e:
            self._log(f"Video prep failed: {e}")
            slicer.util.errorDisplay(f"Video prep failed:\n{e}")
        finally:
            self._setBusy(False)

    def _mov_to_mp4_blocking(self, mov_path: str, mp4_path: str):
        import cv2
        cap = cv2.VideoCapture(mov_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open: {mov_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create MP4 writer at: {mp4_path}")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            frame_idx += 1
            if frame_idx % 250 == 0:
                self._log(f"Converted {frame_idx} frames?")
                slicer.app.processEvents()

        cap.release()
        out.release()

    def _extract_frames_blocking(self, video_path: str, frames_dir: str) -> int:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        idx, wrote = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            fpath = str(Path(frames_dir) / f"frame_{idx:07d}.jpg")
            cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            wrote += 1
            if wrote % 250 == 0:
                self._log(f"Extracted {wrote}/{total if total else '?'} frames?")
                slicer.app.processEvents()
        cap.release()
        return wrote

    def _safe_empty_dir(self, path: Path):
        if not path.exists():
            return
        for child in path.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child)
            except Exception as e:
                self._log(f"WARNING: Could not remove {child}: {e}")

    # ---------- ROI & Tracking ----------
    def onBrowseCkpt(self):
        startDir = os.path.dirname(self.ckptEdit.text) if self.ckptEdit.text else str(self.defaultCloneDir() / "checkpoints")
        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select SAM 2.1 checkpoint",
            startDir,
            "Checkpoint (*.pt *.pth);;All files (*)"
        )
        if not filePath:
            return
        self.ckptEdit.setText(filePath)
        qt.QSettings().setValue(self.SETTINGS_CKPT_PATH, filePath)

    def onLoadFrames(self):
        frames_dir = self.framesDirEdit.text.strip()
        if not frames_dir:
            slicer.util.messageBox("Frames folder not set. Please run Video Prep first.")
            return
        p = Path(frames_dir)
        if not p.exists():
            slicer.util.messageBox(f"Frames folder does not exist:\n{frames_dir}")
            return
        self._setBusy(True)
        try:
            self.framesBuffer = self._load_frames_from_folder(p)
            self._log(f"Loaded {len(self.framesBuffer)} frames from {p}")
        except Exception as e:
            self._log(f"Loading frames failed: {e}")
            slicer.util.errorDisplay(f"Loading frames failed:\n{e}")
        finally:
            self._setBusy(False)

    def _load_frames_from_folder(self, frames_dir: Path):
        import cv2
        imgs = []
        files = sorted(
            [f for f in frames_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda fp: int("".join([c for c in fp.stem if c.isdigit()]) or 0)
        )
        if not files:
            raise RuntimeError(f"No images found in {frames_dir}")
        for i, fp in enumerate(files, 1):
            im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if im is None:
                self._log(f"WARNING: Could not read {fp}, skipping.")
                continue
            imgs.append(im)
            if i % 200 == 0:
                self._log(f"Loaded {i} frames?")
                slicer.app.processEvents()
        return imgs

    def _show_frame_in_slice_view(self, bgr_img, nodeName="SVMM_FirstFrameColor"):
        """Show COLOR preview **flipped** (flipud + fliplr) like Photogrammetry."""
        import cv2, numpy as np
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb = np.flipud(np.fliplr(rgb))
        if (self._firstFrameVectorNode is None) or (self._firstFrameVectorNode.GetScene() is None):
            self._firstFrameVectorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", nodeName)
            self._firstFrameVectorNode.CreateDefaultDisplayNodes()
        arr4d = rgb[np.newaxis, ...]
        slicer.util.updateVolumeFromArray(self._firstFrameVectorNode, arr4d)
        self._firstFrameVectorNode.SetSpacing(1.0, 1.0, 1.0)
        try:
            slicer.util.setSliceViewerLayers(background=self._firstFrameVectorNode)
            lm = slicer.app.layoutManager()
            if lm:
                red = lm.sliceWidget('Red')
                if red:
                    red.sliceLogic().FitSliceToAll()
        except Exception as e:
            self._log(f"WARNING: Could not refresh Red slice view: {e}")

    def onSelectROI(self):
        # First frame
        if self.framesBuffer and len(self.framesBuffer) > 0:
            frame0_present = True
        else:
            frames_dir = self.framesDirEdit.text.strip()
            if not frames_dir:
                slicer.util.messageBox("Frames folder not set. Please run Video Prep first.")
                return
            first = None
            p0 = Path(frames_dir) / "frame_0000001.jpg"
            if p0.exists():
                first = str(p0)
            else:
                jpgs = sorted(Path(frames_dir).glob("*.jpg"))
                if jpgs:
                    first = str(jpgs[0])
            if not first:
                slicer.util.messageBox(f"No frames found in {frames_dir}")
                return
            import cv2
            im = cv2.imread(first, cv2.IMREAD_COLOR)
            if im is None:
                slicer.util.messageBox(f"Could not read: {first}")
                return
            self.framesBuffer = [im]
            frame0_present = True

        # Display first frame
        self._show_frame_in_slice_view(self.framesBuffer[0])

        # Remove old ROI
        if self._roiNode and slicer.mrmlScene.IsNodePresent(self._roiNode):
            if self._roiObserverTag:
                try: self._roiNode.RemoveObserver(self._roiObserverTag)
                except Exception: pass
                self._roiObserverTag = None
            slicer.mrmlScene.RemoveNode(self._roiNode)
            self._roiNode = None

        # Add ROI
        self._roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "SVMM_ROI")
        self._roiNode.CreateDefaultDisplayNodes()
        dnode = self._roiNode.GetDisplayNode()
        if dnode:
            dnode.SetHandlesInteractive(True)
            dnode.SetVisibility(True)

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsROINode")
        selectionNode.SetActivePlaceNodeID(self._roiNode.GetID())
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        def _on_point_defined(caller, evt):
            try:
                interactionNode.SetPlaceModePersistence(0)
                interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

                if self._roiNode.GetNumberOfControlPoints() > 0:
                    posRAS = [0.0, 0.0, 0.0]
                    self._roiNode.GetNthControlPointPositionWorld(0, posRAS)
                    try: self._roiNode.SetCenter(*posRAS)
                    except Exception:
                        try: self._roiNode.SetCenter(posRAS)
                        except Exception: pass

                vol = self._firstFrameVectorNode
                if vol and vol.GetImageData():
                    W, H = float(vol.GetImageData().GetDimensions()[0]), float(vol.GetImageData().GetDimensions()[1])
                    rx, ry, rz = max(5.0, 0.10*W), max(5.0, 0.10*H), 0.5
                    try: self._roiNode.SetRadiusXYZ(rx, ry, rz)
                    except Exception:
                        try: self._roiNode.SetSize(2.0*rx, 2.0*ry, 2.0*rz)
                        except Exception: pass

                self.finalizeROIBtn.setEnabled(True)
                slicer.util.infoDisplay(
                    "ROI placement complete. Drag handles to adjust. Click 'Finalize ROI (Save BBox)' to proceed.",
                    autoCloseMsec=5000
                )
            finally:
                pass

        self._roiObserverTag = self._roiNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, _on_point_defined
        )

        self.finalizeROIBtn.setEnabled(True)
        self.trackBtn.setEnabled(False)

    def onFinalizeROI(self):
        if not self._roiNode or not self._firstFrameVectorNode:
            slicer.util.messageBox("No ROI/first frame to finalize. Place an ROI first.")
            return

        bbox_d = self._compute_bbox_from_roi(self._firstFrameVectorNode, self._roiNode)
        if not bbox_d:
            slicer.util.messageBox("Unable to compute ROI bounds. Adjust the ROI and try again.")
            return

        x_d, y_d, w, h = bbox_d
        W, H = self._firstFrameVectorNode.GetImageData().GetDimensions()[0], self._firstFrameVectorNode.GetImageData().GetDimensions()[1]
        x_o = max(0, min(W - 1, W - (x_d + w)))
        y_o = max(0, min(H - 1, H - (y_d + h)))
        w_o = max(1, min(w, W - x_o))
        h_o = max(1, min(h, H - y_o))
        self.bbox_xywh = (int(x_o), int(y_o), int(w_o), int(h_o))

        self._log(f"Finalized ROI: DISPLAYED (x,y,w,h) = {bbox_d}  ?  ORIGINAL (x,y,w,h) = {self.bbox_xywh}")
        self.trackBtn.setEnabled(True)

        s = qt.QSettings()
        s.setValue(self.SETTINGS_BBOX, ",".join(map(str, self.bbox_xywh)))
        frames_dir = self.framesDirEdit.text.strip()
        if frames_dir:
            try:
                (Path(frames_dir) / "bbox_xywh.txt").write_text(",".join(map(str, self.bbox_xywh)))
            except Exception as e:
                self._log(f"WARNING: Could not write bbox file: {e}")

        try:
            if self._roiObserverTag:
                self._roiNode.RemoveObserver(self._roiObserverTag)
        except Exception:
            pass
        self._roiObserverTag = None
        try:
            if self._roiNode and slicer.mrmlScene.IsNodePresent(self._roiNode):
                slicer.mrmlScene.RemoveNode(self._roiNode)
        except Exception:
            pass
        self._roiNode = None
        self.finalizeROIBtn.setEnabled(False)

    def _compute_bbox_from_roi(self, volumeNode, roiNode):
        if volumeNode is None or roiNode is None or volumeNode.GetImageData() is None:
            return None
        bounds = [0.0]*6
        roiNode.GetBounds(bounds)
        p1_ras = [bounds[0], bounds[2], bounds[4], 1.0]
        p2_ras = [bounds[1], bounds[3], bounds[4], 1.0]
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)
        def ras_to_ij(r4):
            ij4 = rasToIjk.MultiplyPoint(r4)
            return [int(round(ij4[0])), int(round(ij4[1])), int(round(ij4[2]))]
        i1, j1, _ = ras_to_ij(p1_ras)
        i2, j2, _ = ras_to_ij(p2_ras)
        x_min, x_max = min(i1, i2), max(i1, i2)
        y_min, y_max = min(j1, j2), max(j1, j2)
        W, H = int(volumeNode.GetImageData().GetDimensions()[0]), int(volumeNode.GetImageData().GetDimensions()[1])
        x_min = max(0, min(x_min, W-1)); x_max = max(0, min(x_max, W-1))
        y_min = max(0, min(y_min, H-1)); y_max = max(0, min(y_max, H-1))
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        return (x_min, y_min, w, h)

    def onRunTracking(self):
        frames_dir = self.framesDirEdit.text.strip()
        if not frames_dir or not Path(frames_dir).exists():
            slicer.util.messageBox("Frames folder not set or missing. Please run Video Prep first.")
            return
        mp4_path = self.mp4PathEdit.text.strip()
        if not mp4_path or not Path(mp4_path).exists():
            slicer.util.messageBox("Target .mp4 missing. Please run Video Prep first.")
            return
        ckpt = self.ckptEdit.text.strip()
        if not ckpt or not Path(ckpt).exists():
            slicer.util.messageBox("Checkpoint not found. Please select a valid .pt/.pth.")
            return
        if not self.bbox_xywh:
            slicer.util.messageBox("Please finalize an ROI first.")
            return

        # Ensure libs visible before any torch import
        self._prepare_cuda_runtime_visibility(log=False)

        device = self._comboText(self.deviceCombo).strip()
        s = qt.QSettings()
        s.setValue(self.SETTINGS_CKPT_PATH, ckpt)
        s.setValue(self.SETTINGS_DEVICE, device)

        try:
            import torch
            cu = getattr(torch.version, "cuda", None)
            if cu and not str(cu).startswith("12.6"):
                slicer.util.messageBox(
                    f"Detected torch CUDA {cu}. This module targets cu126.\n"
                    "Click 'Configure SAMURAI' to install cu126 via PyTorchUtils, then restart."
                )
                return
        except Exception as e:
            slicer.util.messageBox(f"Torch not available: {e}\nClick 'Configure SAMURAI' first.")
            return

        if not slicer.util.confirmOkCancelDisplay(
            "Tracking will run on the main thread and may freeze the UI.\nProceed?",
            "Blocking Tracking"
        ):
            return

        self._setBusy(True)
        try:
            predictor = self._build_samurai_predictor(ckpt, device)

            n_frames = 0
            try:
                if self.framesBuffer is not None:
                    n_frames = len(self.framesBuffer)
                else:
                    n_frames = len([p for p in Path(frames_dir).glob("*.jpg")])
            except Exception:
                pass
            if n_frames <= 0:
                import cv2
                cap = cv2.VideoCapture(mp4_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                cap.release()
            if n_frames <= 1:
                raise RuntimeError("Could not determine the number of frames (>1 required).")

            self._log(f"Running SAMURAI tracking on {n_frames} frames?")
            masks_map = self._run_tracking(mp4_path, self.bbox_xywh, predictor, n_frames, device)

            out_dir = Path(frames_dir + "_masks")
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for fid, mask_list in masks_map.items():
                import numpy as np
                agg = None
                for t in mask_list:
                    m = t.detach().float().cpu().numpy()
                    if m.ndim == 3:
                        m = (m > 0.5).any(axis=0).astype("uint8")
                    else:
                        m = (m > 0.5).astype("uint8")
                    agg = m if agg is None else (agg | m)
                if agg is None:
                    continue
                fpath = str(out_dir / f"mask_{fid+1:07d}.png")
                self._save_png_mask(fpath, (agg*255).astype("uint8"))
                saved += 1
                if saved % 100 == 0:
                    self._log(f"Saved {saved} masks?")
                    slicer.app.processEvents()

            self._log(f"Tracking complete. Saved {saved} masks to {out_dir}")
            if self.keepInMemCheck.isChecked():
                try:
                    import numpy as np
                    self.masksBuffer = {
                        fid: ((self._stack_or_any(mask_list)) > 0).astype(bool)
                        for fid, mask_list in masks_map.items()
                    }
                    self._log(f"In-memory masks available for {len(self.masksBuffer)} frames.")
                except Exception as e:
                    self._log(f"WARNING: Could not keep masks in memory: {e}")

        except Exception as e:
            self._log(f"Tracking failed: {e}")
            slicer.util.errorDisplay(f"Tracking failed:\n{e}")
        finally:
            self._setBusy(False)

    def _save_png_mask(self, path, mask_uint8):
        try:
            from PIL import Image
            Image.fromarray(mask_uint8, mode="L").save(path)
        except Exception:
            import cv2
            cv2.imwrite(path, mask_uint8)

    def _stack_or_any(self, mask_list):
        import numpy as np
        arrs = []
        for t in mask_list:
            m = t.detach().float().cpu().numpy()
            if m.ndim == 3:
                m = (m > 0.5).any(axis=0).astype("uint8")
            else:
                m = (m > 0.5).astype("uint8")
            arrs.append(m)
        if not arrs:
            return np.zeros((1, 1), dtype="uint8")
        return np.any(np.stack(arrs, axis=0), axis=0).astype("uint8")

    def _build_samurai_predictor(self, checkpoint: str, device: str):
        self._prepare_cuda_runtime_visibility(log=False)
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except Exception as e:
            raise RuntimeError(f"Could not import SAM-2 predictor builder: {e}")

        short = self._determine_cfg_from_ckpt(checkpoint)
        for cfg in self._sam2_config_candidates(short):
            try:
                self._log(f"Trying cfg='{cfg}' (device={device})")
                predictor = build_sam2_video_predictor(cfg, checkpoint, device=device)
                self._log(f"Loaded predictor with cfg='{cfg}'")
                return predictor
            except Exception as e:
                self._log(f"Config '{cfg}' failed: {e}")
        self._log_pkg_configs()
        raise RuntimeError("SAM-2 config could not be resolved for this install.")

    def _determine_cfg_from_ckpt(self, ckpt_path: str) -> str:
        name = Path(ckpt_path).name.lower()
        if "large" in name: return "sam2.1_hiera_l"
        if "base" in name:  return "sam2.1_hiera_b"
        if "small" in name or "tiny" in name: return "sam2.1_hiera_s"
        return "sam2.1_hiera_l"

    def _run_tracking(self, video_path: str, bbox_xywh, predictor, n_frames: int, device: str):
        x, y, w, h = [int(v) for v in bbox_xywh]
        self._log(f"Tracking ROI (x,y,w,h) = {x,y,w,h}")
        try:
            import torch
            from contextlib import nullcontext
        except Exception as e:
            raise RuntimeError(f"PyTorch not available: {e}")
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if device.lower().startswith("cuda") else nullcontext()
        masks = {}
        with torch.inference_mode(), autocast_ctx:
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            predictor.add_new_points_or_box(state, box=(x, y, x + w, y + h), frame_idx=0, obj_id=0)
            progressed = 0
            for fid, _, mask_list in predictor.propagate_in_video(state):
                masks[fid] = mask_list
                progressed += 1
                if progressed % 50 == 0:
                    self._log(f"Propagated {progressed}/{max(1, n_frames-1)} frames?")
                    slicer.app.processEvents()
            try:
                if device.lower().startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return masks

    def _sam2_config_candidates(self, short_name: str) -> list[str]:
        from importlib import resources
        rel = f"configs/sam2.1/{short_name}.yaml"
        out = [rel]
        # Some forks flatten or rename groups; include common fallbacks
        for alt in [
            f"configs/{short_name}.yaml",
            f"config/{short_name}.yaml",
            short_name,
            f"sam2.1/{short_name}.yaml",
        ]:
            if alt not in out:
                out.append(alt)
        # Keep only those that actually exist in the installed package when possible
        try:
            base = resources.files("sam2")
            filtered = []
            for c in out:
                p = base / c
                if str(p).endswith(".yaml"):
                    if p.is_file():
                        filtered.append(c)
                else:
                    filtered.append(c)  # name-only; let SAM2 resolve
            if filtered:
                return filtered
        except Exception:
            pass
        return out

    def _log_pkg_configs(self):
        from importlib import resources
        try:
            base = resources.files("sam2") / "configs"
            if not base.exists():
                self._log("No 'configs' directory found in installed sam2 package.")
                return
            lines = ["Available YAMLs under sam2/configs:"]
            for p in base.rglob("*.yaml"):
                rel = p.relative_to(resources.files("sam2"))
                lines.append(f"  - {rel}")
            self._log("\n".join(lines))
        except Exception as e:
            self._log(f"Could not enumerate sam2/configs: {e}")

    def _pip_try_one_of(self, specs: list[str], name: str) -> bool:
        """
        Try several pip specs in order (useful for platform-specific wheels).
        Returns True on first success.
        """
        for spec in specs:
            try:
                self._pip(spec, desc=f"pip install {spec}  ({name})")
                return True
            except Exception as e:
                self._log(f"Attempt '{spec}' failed for {name}: {e}")
        return False

    def _ensure_video_backends(self) -> None:
        """
        Ensure 'decord' is present (preferred by SAM-2). If that fails, install PyAV as a
        best-effort fallback and loudly warn. We also validate import so we fail fast.
        """
        try:
            import decord  # noqa: F401
            from decord import __version__ as _dv
            self._log(f"decord OK: {_dv}")
            return
        except Exception as e:
            self._log(f"decord not available yet: {e}. Installing...")

        # Try common decord wheels (pin before unpinned to avoid source builds)
        if not self._pip_try_one_of(
                ["decord==0.6.0", "decord==0.6.1", "decord"], "decord"
        ):
            self._log("WARNING: Could not install decord wheels. Trying PyAV as a fallback.")
            self._pip_try_one_of(["av==12.2.0", "av"], "PyAV")

        # Validate import again; if still missing, hard fail so user knows early.
        try:
            import decord  # noqa: F401
            from decord import __version__ as _dv
            self._log(f"decord OK after install: {_dv}")
        except Exception:
            try:
                import av  # noqa: F401
                from av import __version__ as _avv
                self._log(f"PyAV installed ({_avv}), but note: SAM-2 video pipeline typically uses decord.")
                self._log("If SAM-2 cannot find decord internally, tracking may still fail.")
            except Exception as ee:
                raise RuntimeError(
                    "Neither 'decord' nor 'av' could be installed. "
                    "Please check networking / wheel availability and try Configure again."
                ) from ee


# -------------------------
# Logic
# -------------------------
class SamuraiVideoMaskingLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        super().__init__()
        self.parameters = {}

    def git_available(self) -> bool:
        try:
            subprocess.check_output(["git", "--version"])
            return True
        except Exception:
            return False
