# -*- coding: utf-8 -*-
"""
SamuraiVideoMasking (blocking/main-thread setup)
------------------------------------------------
All work runs on the main Qt thread (no worker threads). UI may freeze during setup.

Features
- Collapsible "SAMURAI Setup" with:
  ? Pre-populated repo URL
  ? Destination path (Support/ folder)
  ? Buttons:
      - Configure SAMURAI (Blocking)
      - Verify Installation
      - Open Support Folder
  ? Live log console + status line
- Uses PyTorchUtils to ensure Torch (prompt + restart if needed)
- Clones/updates repo, installs SAM2 (editable), installs base deps (no torch/torchvision, no matplotlib pin)
"""

import os
import sys
import platform
import subprocess
import shlex
import time
from pathlib import Path

import qt
import ctk
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
            "Clone and set up the SAMURAI repo for use in this module. "
            "Everything runs on the main thread (UI may freeze during setup)."
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic = None

    # --- Paths ---
    def moduleDir(self) -> Path:
        return Path(os.path.dirname(slicer.modules.samuraivideomasking.path))

    def supportDir(self) -> Path:
        return self.moduleDir() / "Support"

    def defaultCloneDir(self) -> Path:
        return self.supportDir() / "samurai"

    # --- UI ---
    def setup(self):
        super().setup()
        self.logic = SamuraiVideoMaskingLogic()

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

        note = qt.QLabel(
            "<i>Note:</i> SAMURAI expects <b>Python > 3.10</b> and <b>PyTorch > 2.3.1</b>. "
            "Torch is managed via the <b>PyTorch Utils</b> extension."
        )
        note.wordWrap = True
        form.addRow(note)

        self.configureBtn.clicked.connect(self.onConfigureClicked)
        self.verifyBtn.clicked.connect(self.onVerifyClicked)
        self.openFolderBtn.clicked.connect(self.onOpenFolderClicked)

        # Ensure Support/ and reload saved repo path
        self.supportDir().mkdir(parents=True, exist_ok=True)
        s = qt.QSettings()
        if s.contains(self.SETTINGS_REPO_PATH):
            saved = s.value(self.SETTINGS_REPO_PATH)
            if saved:
                self.destPathEdit.setText(saved)

        self._log("Ready. Click 'Configure SAMURAI (Blocking)' to set up, or 'Verify Installation' to test imports.")
        self._envHints()
        self.layout.addStretch(1)

    # --- Helpers ---
    def _setBusy(self, busy: bool):
        self.configureBtn.enabled = not busy
        self.verifyBtn.enabled = not busy
        self.openFolderBtn.enabled = not busy
        self.urlEdit.enabled = not busy
        self.statusLabel.setText(f"Status: {'Working...' if busy else 'Idle'}")
        self.statusLabel.setStyleSheet("color: #f5c542;" if busy else "color: #BBB;")
        slicer.app.processEvents()

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logEdit.appendPlainText(f"[{ts}] {text}")
        self.logEdit.moveCursor(qt.QTextCursor.End)
        slicer.app.processEvents()

    def _envHints(self):
        if sys.version_info < (3, 10):
            self._log("WARNING: Python < 3.10 detected in Slicer; full install may not succeed.")
        if not self.logic.git_available():
            self._log("WARNING: 'git' not found on PATH. Install Git and retry.")
        try:
            import torch  # noqa
            self._log("PyTorch already available in Slicer.")
        except Exception:
            self._log("INFO: PyTorch not currently available in Slicer Python.")

    def _pip(self, spec: str):
        """
        Robust wrapper around slicer.util.pip_install across Slicer versions.
        Returns (ok: bool, out: str). No exception => ok=True.
        """
        try:
            ret = slicer.util.pip_install(spec)
            if isinstance(ret, tuple):  # older returns (ok, out)
                return bool(ret[0]), str(ret[1] or "")
            return True, str(ret or "")
        except Exception as e:
            return False, str(e)

    # --- Torch via PyTorchUtils (main thread) ---
    def _ensureTorchIfPossible(self) -> bool:
        try:
            import PyTorchUtils  # noqa: F401
        except ModuleNotFoundError:
            slicer.util.messageBox(
                "This module expects the 'PyTorch Utils' extension.\n"
                "Install it from Extensions Manager, then return here."
            )
            self._log("PyTorchUtils not found. Install the extension and retry.")
            return False

        try:
            import PyTorchUtils
            torchLogic = PyTorchUtils.PyTorchUtilsLogic()
            if not torchLogic.torchInstalled():
                if not slicer.util.confirmOkCancelDisplay(
                    "SAMURAI requires PyTorch. Install via PyTorch Utils now?",
                    "Install PyTorch"
                ):
                    self._log("User skipped Torch install. Proceeding to clone only.")
                    return True  # allow repo staging; inference can be wired later
                self._log("Installing PyTorch via PyTorch Utils...")
                torch_module = torchLogic.installTorch(askConfirmation=True)
                if torch_module:
                    if slicer.util.confirmYesNoDisplay(
                        "PyTorch installed. Slicer must restart. Restart now?"
                    ):
                        self._log("Restarting Slicer to finalize PyTorch install.")
                        slicer.util.restart()
                        return False
                    else:
                        self._log("Restart postponed. Torch unavailable until restart.")
                        return False
                else:
                    slicer.util.messageBox("PyTorch installation did not complete. Install manually if needed.")
                    self._log("PyTorch install returned no module. Aborting configuration.")
                    return False
            else:
                self._log("PyTorch detected via PyTorch Utils.")
        except Exception as e:
            self._log(f"WARNING: PyTorchUtils check failed: {e}")
            # Still allow clone/update
            return True

        return True

    # --- Actions ---
    def onOpenFolderClicked(self):
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(self.supportDir())))

    def onConfigureClicked(self):
        if not self._ensureTorchIfPossible():
            return

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
            self._configure_samurai_blocking(repo_url, dest_dir)
            # Save settings
            s = qt.QSettings()
            s.setValue(self.SETTINGS_REPO_PATH, str(dest_dir))
            s.setValue(self.SETTINGS_INSTALLED, True)
            self._log("Done. SAMURAI is staged.")
        except Exception as e:
            self._log(f"Configuration failed: {e}")
        finally:
            self._setBusy(False)

    def _configure_samurai_blocking(self, repo_url: str, dest_dir: Path):
        self._log(f"Destination folder: {dest_dir}")
        repo_dir = dest_dir

        # 1) Clone or update
        if repo_dir.exists() and (repo_dir / ".git").exists():
            self._log("Repo exists. Pulling latest...")
            self.logic.run_cmd_blocking(["git", "-C", str(repo_dir), "fetch", "--all"], self._log)
            self.logic.run_cmd_blocking(["git", "-C", str(repo_dir), "pull", "--ff-only"], self._log)
        else:
            self._log(f"Cloning {repo_url} -> {repo_dir}")
            self.logic.run_cmd_blocking(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], self._log)

        # 2) Submodules
        if (repo_dir / ".gitmodules").exists():
            self._log("Initializing submodules...")
            self.logic.run_cmd_blocking(
                ["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"], self._log
            )

        # 3) Install SAM2 (editable) on main thread
        sam2_dir = repo_dir / "sam2"
        if sam2_dir.exists():
            self._log("Installing SAM2 (editable)...")
            ok, out = self._pip(f'-e "{sam2_dir}"')
            if out:
                self._log(out.strip())
            if not ok:
                self._log("WARNING: SAM2 install (editable) reported failure.")

            # Keep things lean: skip notebooks extras by default
            # Uncomment if you truly need the Jupyter stack inside Slicer.
            # self._log("Installing SAM2 extras [notebooks] (editable)...")
            # ok, out = self._pip(f'-e "{sam2_dir}[notebooks]"')
            # if out:
            #     self._log(out.strip())
            # if not ok:
            #     self._log("WARNING: SAM2 [notebooks] extras failed to install.")
        else:
            self._log("WARNING: 'sam2' directory not found; skipping SAM2 install.")

        # 4) Base deps (exclude torch/torchvision/matplotlib pin)
        base_requirements = [
            "tikzplotlib",
            "jpeg4py",
            "opencv-python",
            "lmdb",
            "pandas",
            "scipy",
            "loguru",
        ]
        self._log("Installing base Python dependencies (excluding torch/torchvision/matplotlib pin)...")
        for pkg in base_requirements:
            self._log(f"pip install {pkg} ...")
            ok, out = self._pip(pkg)
            if out:
                self._log(out.strip())
            if not ok:
                self._log(f"WARNING: '{pkg}' reported failure.")

        # 5) Checkpoint download
        ckpt_dir = repo_dir / "checkpoints"
        script_sh = ckpt_dir / "download_ckpts.sh"
        if ckpt_dir.exists() and script_sh.exists():
            if platform.system() in ("Linux", "Darwin"):
                self._log("Attempting checkpoint download via bash script...")
                try:
                    self.logic.run_cmd_blocking(["bash", str(script_sh)], self._log, cwd=str(ckpt_dir))
                except Exception as e:
                    self._log(f"WARNING: Could not run checkpoint script automatically: {e}")
                    self._log(f"Manual: cd {ckpt_dir} && bash {script_sh.name}")
            else:
                self._log("Windows detected. Run checkpoint script in WSL or follow repo docs.")
                self._log(f"Manual: open folder {ckpt_dir}")
        else:
            self._log("No checkpoint script found. Follow repo docs if needed.")

        # 6) Report on torch
        try:
            import torch  # noqa
            self._log(f"PyTorch available: {torch.__version__}")
        except Exception:
            self._log("Reminder: PyTorch will be available only after install/restart.")

        self._log("Setup complete.")

    # --- Verify button: import modules and show versions ---
    def onVerifyClicked(self):
        self._setBusy(True)
        try:
            self._log("Verifying SAMURAI environment imports?")

            checks = [
                ("sam2", "import sam2 as m; getattr(m, '__version__', 'OK')"),
                ("torch", "import torch as m; m.__version__"),
                ("torchvision", "import torchvision as m; m.__version__"),
                ("cv2 (OpenCV)", "import cv2 as m; m.__version__"),
                ("hydra-core", "import hydra as m; getattr(m, '__version__', 'OK')"),
                ("omegaconf", "import omegaconf as m; getattr(m, '__version__', 'OK')"),
                ("iopath", "import iopath as m; getattr(m, '__version__', 'OK')"),
                ("numpy", "import numpy as m; m.__version__"),
                ("pandas", "import pandas as m; m.__version__"),
                ("scipy", "import scipy as m; m.__version__"),
                ("lmdb", "import lmdb as m; getattr(m, '__version__', 'OK')"),
                ("jpeg4py", "import jpeg4py as m; getattr(m, '__version__', 'OK')"),
                ("loguru", "import loguru as m; getattr(m, '__version__', 'OK')"),
            ]

            ok_all = True
            for label, code in checks:
                try:
                    ns = {}
                    exec(code, {}, ns)
                    version = next(iter(ns.values()))
                    self._log(f"OK: {label} -> {version}")
                except Exception as e:
                    ok_all = False
                    self._log(f"FAIL: {label} import error -> {e}")

            if ok_all:
                self._log("Verification passed. All core imports succeeded.")
            else:
                self._log("Verification finished with errors. See failures above.")
        finally:
            self._setBusy(False)


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

    def run_cmd_blocking(self, args, log_fn, cwd=None):
        """
        Run a command on the main thread, streaming stdout to log_fn.
        This blocks the UI (by design) but stays entirely on the UI thread.
        """
        if isinstance(args, str):
            args = shlex.split(args)

        p = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in p.stdout:
            if not line:
                continue
            log_fn(line.rstrip("\n"))
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(args)}")
        return True
