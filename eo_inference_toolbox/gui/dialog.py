"""The guided EO Inference Toolbox dialog.

Three tabs:

* **Backends** — a card per inference backend (Clay, Prithvi, Satlas) showing its
  facts and the *live* install status of PyTorch and the backend's own package,
  with one-click background installs (run in a :class:`QgsTask` so the UI never
  freezes). scikit-learn (for classify/cluster) and planetary-computer (for HLS
  signing) get their own install buttons.
* **Workflow** — the seven algorithms in workflow order; each button opens the
  standard Processing dialog for that algorithm, which runs it on a background
  thread with progress and cancel handled by the Processing framework.
* **About** — licences and pointers.

Only :mod:`eocore.backends` (torch-free) and Qt/QGIS are imported here; GDAL,
torch and scikit-learn are never imported by the GUI itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from qgis.core import Qgis, QgsApplication, QgsMessageLog, QgsTask
from qgis.PyQt import sip
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..eocore.backends import BACKEND_IDS, backend_info

# The GUI is torch-free at import; dependency detection and installs are
# delegated to the single shared implementation in eocore.backends._deps.
from ..eocore.backends._deps import TORCH_INDEX_URLS, module_available, pip_install

# Weight cache inspection + downloads are stdlib-only (no torch), safe at import.
from ..eocore.backends._weights import cache_size_bytes, default_cache_dir, human_bytes

_LOG_TAG = "EO Inference"

# (algorithm id without provider prefix, title, one-line guidance).
_WORKFLOW: tuple[tuple[str, str, str], ...] = (
    ("fetchimagery", "1. Fetch imagery", "Download a low-cloud scene for your AOI."),
    ("embed", "2. Embed imagery", "Run a backend to make a per-pixel embedding raster."),
    ("torgb", "3. Embedding to RGB", "Preview an embedding as a 3-band image."),
    ("similarity", "4. Similarity", "Map similarity to reference points."),
    ("change", "5. Change / trajectory", "Compare embeddings across dates."),
    ("classify", "6. Classify", "Train / apply a supervised classifier."),
    ("cluster", "7. Cluster", "Unsupervised K-means segmentation."),
)

_PROVIDER = "eoinfer"


class _PipInstallTask(QgsTask):
    """Install packages with pip on a background thread (never blocks the UI)."""

    def __init__(
        self,
        packages: Sequence[str],
        on_done: Callable[[bool, str], None],
        *,
        index_url: str | None = None,
    ) -> None:
        super().__init__(f"Installing {', '.join(packages)}", QgsTask.Flag.CanCancel)
        self._packages = list(packages)
        self._on_done = on_done
        self._index_url = index_url
        self._output = ""

    def run(self) -> bool:
        # QgsTask API: runs on a worker thread and must never raise. The actual
        # install is the single shared implementation in eocore.backends._deps.
        try:
            ok, self._output = pip_install(self._packages, timeout=7200, index_url=self._index_url)
        except Exception as exc:  # defensive: a worker thread must never raise.
            self._output = f"pip invocation failed: {exc}"
            return False
        return ok

    def finished(self, result: bool) -> None:
        # QgsTask API: called back on the main thread when run() returns.
        self._on_done(bool(result), self._output)


class _WeightsTask(QgsTask):
    """Pre-download / verify a backend's weights on a background thread (E-5)."""

    def __init__(
        self,
        backend_id: str,
        cache_dir: str | None,
        on_done: Callable[[bool, str], None],
    ) -> None:
        super().__init__(f"Fetching {backend_id} weights", QgsTask.Flag.CanCancel)
        self._backend_id = backend_id
        self._cache_dir = cache_dir
        self._on_done = on_done
        self._message = ""

    def run(self) -> bool:
        # QgsTask API: runs on a worker thread and must never raise.
        try:
            self._message = self._fetch()
        except Exception as exc:  # defensive: a worker thread must never raise.
            self._message = f"Weight download failed: {exc}"
            return False
        return True

    def _fetch(self) -> str:
        from ..eocore.backends._weights import hf_download, sha256_file

        if self._backend_id == "clay":
            from ..eocore.backends.clay import CLAY_CHECKPOINT, CLAY_REPO

            expected = backend_info("clay").weights_sha256
            path = hf_download(
                CLAY_REPO,
                CLAY_CHECKPOINT,
                cache_dir=self._cache_dir or None,
                expected_sha256=expected,
            )
            digest = sha256_file(path)
            verified = " (verified)" if expected else ""
            return f"Cached {path.name}{verified}\nSHA-256: {digest}"
        # Prithvi/Satlas fetch their weights inside their own packages during the
        # first model build; there is no standalone checkpoint to pull here.
        return (
            f"{backend_info(self._backend_id).display_name} downloads its weights "
            "automatically on the first Embed run (via its own package). Install the "
            "backend, then run step 2."
        )

    def finished(self, result: bool) -> None:
        # QgsTask API: called back on the main thread when run() returns.
        self._on_done(bool(result), self._message)


class _CacheSizeTask(QgsTask):
    """Measure the weights-cache size off the GUI thread (rglob+stat can stutter)."""

    def __init__(self, on_done: Callable[[int], None]) -> None:
        super().__init__("Measuring weights cache", QgsTask.Flag.CanCancel)
        self._on_done = on_done
        self._size = 0

    def run(self) -> bool:
        # QgsTask API: runs on a worker thread and must never raise. Touches no
        # GUI object; only the stdlib cache-size helper (recursive rglob + stat).
        try:
            self._size = cache_size_bytes()
        except Exception:  # defensive: a worker thread must never raise.
            return False
        return True

    def finished(self, result: bool) -> None:
        # QgsTask API: called back on the main thread when run() returns.
        self._on_done(self._size)


class EoInferenceDialog(QDialog):
    """Guided front-end for the EO Inference Toolbox."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._status_labels: dict[str, QLabel] = {}
        self._cache_label: QLabel | None = None
        self._install_buttons: list[QPushButton] = []
        self.setWindowTitle("EO Inference Toolbox")
        self.resize(780, 640)

        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        tabs.addTab(self._build_backends_tab(), "Backends")
        tabs.addTab(self._build_workflow_tab(), "Workflow")
        tabs.addTab(self._build_about_tab(), "About")
        layout.addWidget(tabs)

        self._log = QTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setPlaceholderText("Install and status messages appear here.")
        layout.addWidget(self._log)

        self._refresh_status()

    # -- backends tab --------------------------------------------------------
    def _build_backends_tab(self) -> QWidget:
        container = QWidget(self)
        outer = QVBoxLayout(container)

        intro = QLabel(
            "Each backend runs a foundation model over <i>your own</i> imagery. "
            "PyTorch and the backend's package must be installed; model weights "
            "download on first use. Installs run in the background."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        for backend_id in BACKEND_IDS:
            inner_layout.addWidget(self._backend_card(backend_id))
        inner_layout.addWidget(self._extras_card())
        inner_layout.addWidget(self._cache_card())
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        refresh = QPushButton("Refresh status", self)
        refresh.clicked.connect(self._refresh_status)
        outer.addWidget(refresh)
        return container

    def _backend_card(self, backend_id: str) -> QWidget:
        info = backend_info(backend_id)
        box = QFrame(self)
        box.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(box)

        title = QLabel(f"<b>{info.display_name}</b>")
        grid.addWidget(title, 0, 0, 1, 2)

        summary = QLabel(info.summary)
        summary.setWordWrap(True)
        grid.addWidget(summary, 1, 0, 1, 2)

        facts = QLabel(
            f"Bands: {', '.join(info.default_bands)} &nbsp; | &nbsp; "
            f"~{info.embedding_dim}-D &nbsp; | &nbsp; {info.resolution_m:g} m "
            f"&nbsp; | &nbsp; {info.licence}"
        )
        facts.setWordWrap(True)
        grid.addWidget(facts, 2, 0, 1, 2)

        status = QLabel("checking...")
        self._status_labels[backend_id] = status
        grid.addWidget(status, 3, 0)

        install = QPushButton("Install backend", self)
        install.clicked.connect(
            lambda _checked=False, bid=backend_id: self._install(backend_info(bid).pip_packages)
        )
        grid.addWidget(install, 3, 1)
        self._install_buttons.append(install)

        download = QPushButton("Download / verify weights", self)
        download.clicked.connect(lambda _checked=False, bid=backend_id: self._download_weights(bid))
        grid.addWidget(download, 4, 0, 1, 2)
        self._install_buttons.append(download)
        return box

    def _extras_card(self) -> QWidget:
        box = QFrame(self)
        box.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(box)
        grid.addWidget(QLabel("<b>Optional tiers</b>"), 0, 0, 1, 3)

        # PyTorch: offer an explicit CPU-only or CUDA build so users are not
        # silently handed a multi-gigabyte CUDA wheel (or a CPU one on a GPU box).
        torch_status = QLabel("checking...")
        self._status_labels["torch"] = torch_status
        grid.addWidget(torch_status, 1, 0)
        torch_cpu = QPushButton("Install PyTorch (CPU)", self)
        torch_cpu.clicked.connect(
            lambda: self._install(("torch",), index_url=TORCH_INDEX_URLS["cpu"])
        )
        grid.addWidget(torch_cpu, 1, 1)
        self._install_buttons.append(torch_cpu)
        torch_cuda = QPushButton("Install PyTorch (CUDA)", self)
        torch_cuda.clicked.connect(
            lambda: self._install(("torch",), index_url=TORCH_INDEX_URLS["cuda"])
        )
        grid.addWidget(torch_cuda, 1, 2)
        self._install_buttons.append(torch_cuda)

        sk_status = QLabel("checking...")
        self._status_labels["sklearn"] = sk_status
        grid.addWidget(sk_status, 2, 0)
        sk_btn = QPushButton("Install scikit-learn", self)
        # skops rides along so saved classify models can be reloaded through skops'
        # trusted-types gate rather than an arbitrary-code unpickle; it is optional
        # (classify still works, saving via pickle, if skops fails to install).
        sk_btn.clicked.connect(lambda: self._install(("scikit-learn", "skops")))
        grid.addWidget(sk_btn, 2, 1, 1, 2)
        self._install_buttons.append(sk_btn)

        pc_status = QLabel("checking...")
        self._status_labels["planetary_computer"] = pc_status
        grid.addWidget(pc_status, 3, 0)
        pc_btn = QPushButton("Install planetary-computer (HLS)", self)
        pc_btn.clicked.connect(lambda: self._install(("planetary-computer",)))
        grid.addWidget(pc_btn, 3, 1, 1, 2)
        self._install_buttons.append(pc_btn)
        return box

    def _cache_card(self) -> QWidget:
        box = QFrame(self)
        box.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(box)
        grid.addWidget(QLabel("<b>Weights cache</b>"), 0, 0, 1, 2)

        path_label = QLabel(f"Location: <code>{default_cache_dir()}</code>")
        path_label.setWordWrap(True)
        grid.addWidget(path_label, 1, 0, 1, 2)

        self._cache_label = QLabel("checking...")
        grid.addWidget(self._cache_label, 2, 0)
        refresh = QPushButton("Refresh cache size", self)
        refresh.clicked.connect(self._refresh_cache)
        grid.addWidget(refresh, 2, 1)
        return box

    # -- workflow tab --------------------------------------------------------
    def _build_workflow_tab(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        note = QLabel(
            "Run the steps in order. Each opens the Processing dialog for that "
            "algorithm, which runs on a background thread with its own progress bar."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        for alg, title, desc in _WORKFLOW:
            layout.addWidget(self._workflow_row(alg, title, desc))
        layout.addStretch(1)
        return container

    def _workflow_row(self, alg: str, title: str, desc: str) -> QWidget:
        row = QFrame(self)
        row.setFrameShape(QFrame.Shape.StyledPanel)
        inner = QHBoxLayout(row)
        text = QLabel(f"<b>{title}</b><br>{desc}")
        text.setWordWrap(True)
        inner.addWidget(text, 1)
        button = QPushButton("Open...", self)
        button.clicked.connect(lambda _checked=False, a=alg: self._open_algorithm(a))
        inner.addWidget(button)
        return row

    # -- about tab -----------------------------------------------------------
    def _build_about_tab(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        text = QLabel(
            "<h3>EO Inference Toolbox</h3>"
            "<p>Run open Earth-observation foundation models over your own imagery "
            "and explore the embeddings with similarity, change, classification and "
            "clustering tools.</p>"
            "<p><b>Backends &amp; licences:</b> Clay v1.5 (Apache-2.0), "
            "Prithvi-EO-2.0 (Apache-2.0), SatlasPretrain (ODC-BY weights). Model "
            "weights are downloaded from their own sources and are never bundled.</p>"
            "<p>Plugin code: GPL-2.0-or-later. See the NOTICE file for third-party "
            "terms and the docs folder for the methodology and limitations.</p>"
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addStretch(1)
        return container

    # -- actions -------------------------------------------------------------
    def _open_algorithm(self, alg: str) -> None:
        alg_id = f"{_PROVIDER}:{alg}"
        try:
            import processing
        except ImportError:
            try:
                from qgis import processing
            except ImportError:
                self._append_log("Processing framework is not available.")
                return
        # Surface any Processing framework error to the log rather than crashing.
        try:
            processing.execAlgorithmDialog(alg_id, {})
        except Exception as exc:
            self._append_log(f"Could not open {alg_id}: {exc}")

    def _set_installs_enabled(self, enabled: bool) -> None:
        # Guard against a double-click / parallel-backend launch starting concurrent
        # pip tasks into the one shared interpreter. Called only from the main thread.
        for button in self._install_buttons:
            button.setEnabled(enabled)

    def _install(self, packages: Sequence[str], *, index_url: str | None = None) -> None:
        where = f" from {index_url}" if index_url else ""
        self._append_log(f"Installing: {', '.join(packages)}{where} (this can take a while)...")
        # All install/download buttons share one interpreter, so a second
        # concurrent pip task can hang or corrupt it: disable them until done.
        self._set_installs_enabled(False)
        task = _PipInstallTask(packages, self._install_done, index_url=index_url)
        QgsApplication.taskManager().addTask(task)

    def _install_done(self, success: bool, output: str) -> None:
        # A pip task can run for up to ~2h; the dialog may have been closed since.
        if sip.isdeleted(self):
            return
        self._set_installs_enabled(True)
        tail = "\n".join(output.strip().splitlines()[-4:]) if output else ""
        if success:
            self._append_log("Install finished successfully.")
        else:
            self._append_log("Install failed. Last output:\n" + tail)
        self._refresh_status()

    def _download_weights(self, backend_id: str) -> None:
        name = backend_info(backend_id).display_name
        self._append_log(f"Fetching / verifying weights for {name} (runs in the background)...")
        self._set_installs_enabled(False)
        task = _WeightsTask(backend_id, None, self._weights_done)
        QgsApplication.taskManager().addTask(task)

    def _weights_done(self, success: bool, message: str) -> None:
        if sip.isdeleted(self):
            return
        self._set_installs_enabled(True)
        self._append_log(message if message else "Done.")
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        # cache_size_bytes() does a recursive rglob + stat of the whole weights
        # cache; run it off the GUI thread so a large HF cache never stutters the UI.
        if self._cache_label is None:
            return
        self._cache_label.setText("Cached: measuring...")
        QgsApplication.taskManager().addTask(_CacheSizeTask(self._cache_size_done))

    def _cache_size_done(self, size_bytes: int) -> None:
        if sip.isdeleted(self):
            return
        if self._cache_label is not None:
            self._cache_label.setText(f"Cached: {human_bytes(size_bytes)}")

    def _refresh_status(self) -> None:
        torch_ok = module_available("torch")
        for backend_id in BACKEND_IDS:
            info = backend_info(backend_id)
            pkg_ok = all(module_available(name) for name in info.import_names)
            label = self._status_labels.get(backend_id)
            if label is not None:
                label.setText(
                    self._badge("PyTorch", torch_ok)
                    + "  "
                    + self._badge("package", pkg_ok and bool(info.import_names))
                )
        self._set_extra_status("torch", "PyTorch", torch_ok)
        self._set_extra_status("sklearn", "scikit-learn", module_available("sklearn"))
        self._set_extra_status(
            "planetary_computer",
            "planetary-computer",
            module_available("planetary_computer"),
        )
        self._refresh_cache()

    def _set_extra_status(self, key: str, name: str, ok: bool) -> None:
        label = self._status_labels.get(key)
        if label is not None:
            label.setText(self._badge(name, ok))

    @staticmethod
    def _badge(name: str, ok: bool) -> str:
        colour = "#2e7d32" if ok else "#b71c1c"
        mark = "installed" if ok else "missing"
        return f'<span style="color:{colour}">{name}: {mark}</span>'

    def _append_log(self, message: str) -> None:
        if sip.isdeleted(self):
            return
        self._log.append(message)
        QgsMessageLog.logMessage(message, _LOG_TAG, Qgis.MessageLevel.Info)
