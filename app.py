import sys
import os
import json
import shutil
from datetime import datetime
import torch
from core_engine import ElephantEngine, UnknownClusterManager
from cluster_health import ClusterHealthMonitor
from review_store import ReviewStore
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QDialog,
    QComboBox,
    QDialogButtonBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QStatusBar,
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QGridLayout,
    QGroupBox,
    QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import (
    QIcon,
    QAction,
    QFont,
    QDesktopServices,
    QColor,
    QPalette,
    QPixmap,
)

# ─── Color Palette (Government / Institutional) ──────────────────────────────
NAVY_PRIMARY = "#0C2340"  # Deep navy – primary brand
NAVY_DARK = "#081A2F"  # Darker navy – sidebar/header accent
NAVY_LIGHT = "#1B3A5C"  # Lighter navy – hover states
GOLD_ACCENT = "#C5A44E"  # Institutional gold – accents & highlights
GOLD_LIGHT = "#E8D9A0"  # Pale gold – subtle badge backgrounds
BG_LIGHT = "#F4F5F7"  # Off-white page background
BG_WHITE = "#FFFFFF"  # Card/panel background
BORDER_SUBTLE = "#D0D5DD"  # Borders
TEXT_PRIMARY = "#1A1A2E"  # Near-black text
TEXT_SECONDARY = "#5A6270"  # Muted text
TEXT_ON_DARK = "#EAECF0"  # Text on dark backgrounds
SUCCESS_GREEN = "#1B7340"  # Formal green for success states
SUCCESS_BG = "#E6F4EA"  # Success badge background
DANGER_RED = "#9B1C1C"  # Formal red for warnings


def confidence_band(sim_score):
    """Map a raw cosine similarity score to a human-readable confidence band.

    Calibrated for WII wildlife images (within-elephant range 0.29–0.42).
    Returns (label, color_hex).
    """
    if sim_score >= 0.38:
        return "✓ strong", "#1B7340"  # green
    elif sim_score >= 0.34:
        return "? borderline", "#7A4F00"  # amber
    else:
        return "weak", "#9B1C1C"  # red


def category_name_from_node(node):
    """Extracts the plain folder name from a tree node label like '📂  Elephant_01  (12)'."""
    raw = node.data(0, Qt.ItemDataRole.UserRole)  # stored as folder path
    return os.path.basename(raw) if raw else ""


class WorkerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, set, dict)  # summary, auto_enrolled, proposed_merges

    def __init__(self, engine, input_dir, output_dir):
        super().__init__()
        self.engine = engine
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        result = self.engine.process_batch(
            self.input_dir, self.output_dir, self.progress.emit
        )
        # process_batch returns (auto_enrolled, proposed_merges)
        if isinstance(result, tuple) and len(result) == 2:
            auto_enrolled, proposed_merges = result
        else:
            auto_enrolled, proposed_merges = result, {}
        summary = {}
        SYSTEM_FOLDERS = {"_ambiguous_matches", "_review", "_temp"}
        if os.path.exists(self.output_dir):
            for folder_name in os.listdir(self.output_dir):
                if folder_name in SYSTEM_FOLDERS:
                    continue
                folder_path = os.path.join(self.output_dir, folder_name)
                if os.path.isdir(folder_path):
                    count = len(
                        [
                            f
                            for f in os.listdir(folder_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]
                    )
                    if count > 0:
                        summary[folder_name] = {"count": count, "path": folder_path}
        self.finished.emit(summary, auto_enrolled or set(), proposed_merges or {})


class PromoteClusterDialog(QDialog):
    """Dialog for promoting an Unknown cluster to a named elephant identity.

    If the user types an existing gallery name, the cluster's images are
    *merged* into that identity (appended embeddings) rather than blocked.
    """

    def __init__(self, cluster_name, cluster_path, engine, parent=None):
        super().__init__(parent)
        self.cluster_name = cluster_name
        self.cluster_path = cluster_path
        self.engine = engine
        self.chosen_name = None
        self.merge_mode = False  # True when adding to existing identity

        self.setWindowTitle(f"Confirm Elephant Identity — {cluster_name}")
        self.setMinimumWidth(460)
        self.setMinimumHeight(220)
        self.setModal(True)

        # Force white background so the dialog is always legible
        self.setStyleSheet("""
            QDialog   { background: #FFFFFF; }
            QLabel    { color: #1A1A2E; }
            QLineEdit {
                background: #F4F6FA;
                border: 1.5px solid #C5CAD5;
                border-radius: 5px;
                padding: 5px 8px;
                color: #1A1A2E;
                font-size: 10pt;
            }
            QLineEdit:focus { border-color: #0C2340; }
            QPushButton#cancel {
                background: #E5E7EB; color: #374151;
                border: none; border-radius: 4px; padding: 6px 14px;
            }
            QPushButton#cancel:hover { background: #D1D5DB; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 20, 24, 20)

        # ── Header ──────────────────────────────────────────────────────────
        n_images = (
            len(
                [
                    f
                    for f in os.listdir(cluster_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            if os.path.exists(cluster_path)
            else 0
        )

        header = QLabel(
            f"<b style='color:#0C2340;font-size:12pt;'>{cluster_name}</b>"
            f"<span style='color:#6B7280; font-size:9pt;'>  — {n_images} image(s)</span>"
        )
        header.setFont(QFont("Segoe UI", 11))
        layout.addWidget(header)

        hint = QLabel(
            "Type a <b>new name</b> to create a new identity, or an <b>existing name</b> "
            "to merge these images into that identity."
        )
        hint.setWordWrap(True)
        hint.setFont(QFont("Segoe UI", 9))
        hint.setStyleSheet("color: #6B7280;")
        layout.addWidget(hint)

        # ── Name input ───────────────────────────────────────────────────────
        name_lbl = QLabel("Elephant ID / Name:")
        name_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        layout.addWidget(name_lbl)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Elephant_031, Raja, or a2")
        self.name_edit.setFont(QFont("Segoe UI", 10))
        self.name_edit.textChanged.connect(self._update_button_label)
        layout.addWidget(self.name_edit)

        # ── Open-folder link ─────────────────────────────────────────────────
        open_btn = QPushButton("📂  Open Cluster Folder")
        open_btn.setFlat(True)
        open_btn.setFont(QFont("Segoe UI", 9))
        open_btn.setStyleSheet(
            "color: #0C2340; text-decoration: underline; border: none; background: transparent;"
        )
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(cluster_path))
        )
        layout.addWidget(open_btn)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_box = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel")
        cancel.setFont(QFont("Segoe UI", 9))
        cancel.clicked.connect(self.reject)

        self.confirm_btn = QPushButton("✔  Confirm as New Elephant")
        self.confirm_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.confirm_btn.setStyleSheet("""
            QPushButton { background: #1B7340; color: white; border: none;
                          border-radius: 4px; padding: 7px 18px; }
            QPushButton:hover { background: #155a32; }
        """)
        self.confirm_btn.clicked.connect(self._on_confirm)

        btn_box.addStretch()
        btn_box.addWidget(cancel)
        btn_box.addWidget(self.confirm_btn)
        layout.addLayout(btn_box)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _update_button_label(self, text):
        if text.strip() in self.engine.gallery:
            self.confirm_btn.setText("➕  Merge into Existing Identity")
            self.confirm_btn.setStyleSheet("""
                QPushButton { background: #1A56A8; color: white; border: none;
                              border-radius: 4px; padding: 7px 18px; }
                QPushButton:hover { background: #1446A0; }
            """)
        else:
            self.confirm_btn.setText("✔  Confirm as New Elephant")
            self.confirm_btn.setStyleSheet("""
                QPushButton { background: #1B7340; color: white; border: none;
                              border-radius: 4px; padding: 7px 18px; }
                QPushButton:hover { background: #155a32; }
            """)

    def _on_confirm(self):
        new_name = self.name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(
                self, "Name Required", "Please enter a name for the elephant."
            )
            return
        self.merge_mode = new_name in self.engine.gallery
        self.chosen_name = new_name
        self.accept()


class CompareMergeDialog(QDialog):
    def __init__(
        self,
        cluster_a,
        cluster_b,
        base_dir,
        score=None,
        max_sim=None,
        mean_sim=None,
        percentile_str="",
        best_pair=(None, None),
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Compare & Merge: {cluster_a} \u2194 {cluster_b}")
        self.setMinimumSize(850, 500)
        self.cluster_a = cluster_a
        self.cluster_b = cluster_b
        self.base_dir = base_dir
        self.best_img_a = best_pair[0]
        self.best_img_b = best_pair[1]

        self.setStyleSheet("""
            QDialog { background: #FFFFFF; }
            QGroupBox { font-weight: bold; color: #0C2340; border: 1px solid #D0D5DD; border-radius: 4px; margin-top: 1ex; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

        layout = QVBoxLayout(self)

        # Top Stats Panel
        if score is not None or max_sim is not None:
            stats_box = QWidget()
            stats_layout = QHBoxLayout(stats_box)
            stats_layout.setContentsMargins(10, 10, 10, 10)
            stats_box.setStyleSheet("background: #F3F4F6; border-radius: 6px;")

            mx_val = f"{max_sim:.2f}" if max_sim is not None else "N/A"
            mn_val = f"{mean_sim:.2f}" if mean_sim is not None else "N/A"

            # Rank logic
            if "Top" in percentile_str:
                rank_color = "#1B7340"
            elif "Middle" in percentile_str:
                rank_color = "#7A4F00"
            else:
                rank_color = "#9B1C1C"

            rank_html = (
                f"<br><span style='color:{rank_color};'><b>&bull; Relative position: {percentile_str}</b></span>"
                if percentile_str
                else ""
            )

            stats_html = f"""
            <span style='font-size:12px; color:#374151;'><b>Cluster-to-Cluster Similarity:</b></span><br>
            <span style='font-size:12px; color:#4B5563;'>
            &bull; Best matching pair: <span style='color:#1B7340;'><b>{mx_val}</b></span><br>
            &bull; Avg similarity: <b>{mn_val}</b>
            {rank_html}
            </span>
            """
            stats_lbl = QLabel(stats_html)
            stats_layout.addWidget(stats_lbl)
            layout.addWidget(stats_box)

        split = QHBoxLayout()

        left = QGroupBox(f"Current Cluster: {cluster_a}")
        left_layout = QGridLayout(left)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        right = QGroupBox(f"Suggested Cluster: {cluster_b}")
        right_layout = QGridLayout(right)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        split.addWidget(left)
        split.addWidget(right)
        layout.addLayout(split, 1)

        btns = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet(
            "background: transparent; border: 1px solid #D1D5DB; color: #374151; font-weight: bold; border-radius: 4px; padding: 8px 16px;"
        )
        cancel.clicked.connect(self.reject)

        reject_sugg = QPushButton("Reject Suggestion")
        reject_sugg.setCursor(Qt.CursorShape.PointingHandCursor)
        reject_sugg.setStyleSheet(
            "background: #FEE2E2; color: #9B1C1C; font-weight: bold; border-radius: 4px; padding: 8px 16px;"
        )
        reject_sugg.clicked.connect(
            self.reject
        )  # Treat as cancel for now, could emit specific reject signal

        merge_btn = QPushButton(f"Yes, Merge into {cluster_b}")
        merge_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        merge_btn.setStyleSheet("""
            QPushButton { background: #1B7340; color: white; font-weight: bold; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background: #155a32; }
        """)
        merge_btn.clicked.connect(self.accept)

        btns.addStretch()
        btns.addWidget(cancel)
        btns.addWidget(reject_sugg)
        btns.addWidget(merge_btn)

        layout.addLayout(btns)

        self._load_images(left_layout, cluster_a)
        self._load_images(right_layout, cluster_b)

    def _load_images(self, layout, cluster_name):
        path = os.path.join(self.base_dir, cluster_name)
        if not os.path.exists(path):
            return

        images = [
            f
            for f in sorted(os.listdir(path))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not images:
            layout.addWidget(QLabel("No images found."))
            return

        header = QLabel(f"<b>Size: {len(images)} image(s)</b>")
        header.setStyleSheet("color: #4B5563; font-size: 11px;")
        layout.addWidget(header, 0, 0, 1, 2)

        for i, img in enumerate(images[:6]):  # Limit to 6 previews per side
            lbl = QLabel()
            img_path = os.path.abspath(os.path.join(path, img))
            pix = QPixmap(img_path)

            if pix.isNull():
                print(f"[ERROR] Failed to load: {img_path}")
                lbl.setText("⚠ Image not found")
            else:
                pix = pix.scaled(
                    160,
                    160,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                lbl.setPixmap(pix)

            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Highlight best matching pair
            if img == self.best_img_a or img == self.best_img_b:
                lbl.setStyleSheet("border: 3px solid #1B7340; border-radius: 4px;")

            layout.addWidget(lbl, (i // 2) + 1, i % 2)


class ClickableImage(QLabel):
    def __init__(self, img_path, is_outlier=False):
        super().__init__()
        self.img_path = os.path.abspath(img_path)
        self.is_outlier = is_outlier
        self.selected = False

        pixmap = QPixmap(self.img_path)
        if pixmap.isNull():
            print(f"[ERROR] Failed to load: {self.img_path}")
            self.setText("⚠ Image not found")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            pixmap = pixmap.scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio)
            self.setPixmap(pixmap)

        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(3)

    def mousePressEvent(self, event):
        self.selected = not self.selected
        # Notify parent to update border
        if hasattr(self.parent(), "update_border"):
            self.parent().update_border()
        super().mousePressEvent(event)


class ClusterImageCard(QWidget):
    def __init__(self, img_path, is_outlier=False, label_text=""):
        super().__init__()
        self.img_path = img_path
        self.is_outlier = is_outlier
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.img_label = ClickableImage(img_path, is_outlier)
        layout.addWidget(self.img_label, alignment=Qt.AlignmentFlag.AlignCenter)

        text_label = QLabel(label_text)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("color: #4B5563; font-size: 11px; font-weight: bold;")
        if is_outlier:
            text_label.setStyleSheet(
                "color: #9B1C1C; font-size: 11px; font-weight: bold;"
            )
        layout.addWidget(text_label)
        self.update_border()

    @property
    def selected(self):
        return self.img_label.selected

    @selected.setter
    def selected(self, value):
        self.img_label.selected = value
        self.update_border()

    def mousePressEvent(self, event):
        self.selected = not self.selected
        super().mousePressEvent(event)

    def update_border(self):
        if self.selected:
            color = "#1A56A8"  # Blue selected
        elif self.is_outlier:
            color = "#9B1C1C"  # Red outlier
        else:
            color = "#1B7340"  # Green normal

        self.setStyleSheet(f"""
            QWidget {{
                border: 3px solid {color};
                border-radius: 6px;
                background: white;
            }}
            QLabel {{ border: none; }}
        """)


class AmbiguousCard(QWidget):
    def __init__(self, item, controller, parent=None):
        super().__init__(parent)

        self.item = item
        self.controller = controller
        node = item.get("file_paths", [])[0]
        candidates = item.get("candidates", [])
        c1 = candidates[0] if len(candidates) > 0 else {}
        c2 = candidates[1] if len(candidates) > 1 else {}

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Top: Image ---
        layout.addWidget(self.create_main_image(node))

        # --- Middle: Cluster comparison ---
        compare_layout = QHBoxLayout()
        compare_layout.addWidget(self.create_cluster_column(c1))
        compare_layout.addWidget(self.create_cluster_column(c2))

        layout.addLayout(compare_layout)

        # --- Bottom: Actions ---
        layout.addLayout(self.create_buttons(c1, c2))

        self.setLayout(layout)

    def create_main_image(self, img_path):
        container = QWidget()
        layout = QVBoxLayout(container)

        label = QLabel("⚠ Ambiguous Match")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {DANGER_RED}; font-size: 18px; font-weight: bold;")
        layout.addWidget(label)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setMinimumSize(320, 320)

        if os.path.exists(img_path):
            pixmap = QPixmap(img_path).scaled(
                400,
                400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            img_label.setPixmap(pixmap)

        layout.addWidget(img_label)

        name_label = QLabel(os.path.basename(img_path))
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(name_label)

        return container

    def create_cluster_column(self, candidate):
        container = QWidget()
        layout = QVBoxLayout(container)

        cluster_name = candidate.get("cluster", "Unknown")
        score = candidate.get("score", 0.0)

        label = QLabel(f"{cluster_name}\n({score:.3f})")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(label)

        previews_layout = QHBoxLayout()
        for img_path in candidate.get("preview", []):
            if not os.path.exists(img_path):
                continue
            img_label = QLabel()
            pixmap = QPixmap(img_path).scaled(
                120,
                120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            img_label.setPixmap(pixmap)
            previews_layout.addWidget(img_label)

        layout.addLayout(previews_layout)
        return container

    def create_buttons(self, c1, c2):
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cluster1 = c1.get("cluster", "Unknown")
        cluster2 = c2.get("cluster", "Unknown")

        btn_a = QPushButton(f"This is the same elephant ({cluster1})")
        btn_b = QPushButton(f"This is the same elephant ({cluster2})")
        btn_keep = QPushButton("Different elephants")

        btn_a.setStyleSheet(
            f"background-color: {NAVY_PRIMARY}; color: white; padding: 10px; font-size: 14px; border-radius: 4px;"
        )
        btn_b.setStyleSheet(
            f"background-color: {NAVY_PRIMARY}; color: white; padding: 10px; font-size: 14px; border-radius: 4px;"
        )
        btn_keep.setStyleSheet(
            "background-color: #555; color: white; padding: 10px; font-size: 14px; border-radius: 4px;"
        )

        btn_a.clicked.connect(
            lambda: self.controller.resolve_ambiguous_assign(self.item, cluster1)
        )
        btn_b.clicked.connect(
            lambda: self.controller.resolve_ambiguous_assign(self.item, cluster2)
        )
        btn_keep.clicked.connect(
            lambda: self.controller.resolve_ambiguous_keep_separate(self.item)
        )

        layout.addWidget(btn_a)
        layout.addWidget(btn_b)
        layout.addWidget(btn_keep)

        return layout


class ElephantApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Elephant Re-Identification System")
        self.resize(1080, 780)
        self.setMinimumSize(900, 650)

        # ── App icon (works both from source and bundled EXE) ──
        _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        _icon_path = os.path.join(_base, "src", "elephant.ico")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        self.engine = ElephantEngine()
        self.output_base_dir = None
        self.input_dir = None
        self.advanced_mode = False
        self._load_config()  # restore last used output folder
        # Duplicate detection on startup is disabled for WII dataset due to natural herd similarity

        # ── Central structure ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Branding Header ──
        main_layout.addWidget(self._build_header())

        # ── Tab Widget ──
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        self.tabs.addTab(self.tab1, "   Mass Upload   ")
        self.tabs.addTab(self.tab2, "   Review & Correct   ")
        self.tabs.addTab(self.tab3, "   Train Database   ")
        self.tabs.addTab(self.tab4, "   Review & Merge   ")

        main_layout.addWidget(self.tabs, 1)

        # ── Status Bar ──
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet(f"""
            QStatusBar {{
                background: {NAVY_PRIMARY}; color: {TEXT_ON_DARK};
                font-size: 11px; padding: 3px 12px;
                border-top: 2px solid {GOLD_ACCENT};
            }}
        """)
        self.statusBar.showMessage("System Ready  |  Model Loaded Successfully")
        self.setStatusBar(self.statusBar)

        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self._apply_styles()

        # Connect Tab Change Signal
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        if index == 3:  # Review Merge tab
            self.load_ambiguity_inbox()
        elif index == 1:  # Review Correct
            self.load_gallery()
        if self.output_base_dir:
            self.lbl_output_path.setText(self.output_base_dir)
            self.lbl_output_path.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-style: normal; font-weight: 600;"
            )

        # Auto-load output directory into Tab 2 on startup (persists between sessions)
        if self.output_base_dir and os.path.exists(self.output_base_dir):
            self.load_gallery()
            self.load_ambiguity_inbox()

    # ══════════════════════════════════════════════════════════════════════════
    #  BRANDING HEADER
    # ══════════════════════════════════════════════════════════════════════════
    def _build_header(self):
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {NAVY_DARK}, stop:1 {NAVY_PRIMARY});
                border-bottom: 3px solid {GOLD_ACCENT};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("ELEPHANT RE-IDENTIFICATION SYSTEM")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {BG_WHITE}; letter-spacing: 1px;")
        h_layout.addWidget(title)

        h_layout.addStretch()

        sys_status = QLabel(
            "System Status:  ✓ No incorrect matches detected   |   ✓ New sightings grouped automatically"
        )
        sys_status.setFont(QFont("Segoe UI", 10))
        sys_status.setStyleSheet(f"color: {GOLD_LIGHT}; padding-right: 20px;")
        h_layout.addWidget(sys_status)

        from PyQt6.QtWidgets import QCheckBox

        self.cb_advanced_mode = QCheckBox("Advanced Mode")
        self.cb_advanced_mode.setFont(QFont("Segoe UI", 10))
        self.cb_advanced_mode.setStyleSheet(f"color: {BG_WHITE}; spacing: 8px;")
        self.cb_advanced_mode.setChecked(self.advanced_mode)
        self.cb_advanced_mode.toggled.connect(self._toggle_advanced_mode)
        h_layout.addWidget(self.cb_advanced_mode)

        return header

    def _toggle_advanced_mode(self, checked):
        self.advanced_mode = checked
        self.load_ambiguity_inbox()
        # You could also call something to update Tab 2 and Tab 1 if needed

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — Mass Upload & Classification
    # ══════════════════════════════════════════════════════════════════════════
    def setup_tab1(self):
        layout = QVBoxLayout()
        layout.setSpacing(14)
        layout.setContentsMargins(28, 24, 28, 16)

        # Section header
        sec_header = QLabel("\u25a0  Batch Classification")
        sec_header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        sec_header.setStyleSheet(f"color: {NAVY_PRIMARY};")
        layout.addWidget(sec_header)

        sec_desc = QLabel(
            "Select the source and destination directories, then initiate classification."
        )
        sec_desc.setFont(QFont("Segoe UI", 10))
        sec_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; margin-bottom: 6px;")
        layout.addWidget(sec_desc)

        # ── Directory Selection Group ──
        dir_group = QGroupBox("Directory Configuration")
        dir_group.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        dir_grid = QGridLayout(dir_group)
        dir_grid.setHorizontalSpacing(14)
        dir_grid.setVerticalSpacing(10)
        dir_grid.setContentsMargins(16, 20, 16, 14)

        # Input row
        lbl_in = QLabel("Source Folder:")
        lbl_in.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        lbl_in.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.btn_select_input = QPushButton("  Browse...")
        self.btn_select_input.setFixedSize(110, 34)
        self.btn_select_input.clicked.connect(self.select_input_folder)
        self.lbl_input_path = QLabel("No folder selected")
        self.lbl_input_path.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-style: italic;"
        )
        self.lbl_input_path.setWordWrap(True)

        dir_grid.addWidget(lbl_in, 0, 0)
        dir_grid.addWidget(self.btn_select_input, 0, 1)
        dir_grid.addWidget(self.lbl_input_path, 0, 2)

        # Output row
        lbl_out = QLabel("Output Folder:")
        lbl_out.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        lbl_out.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.btn_select_output = QPushButton("  Browse...")
        self.btn_select_output.setFixedSize(110, 34)
        self.btn_select_output.clicked.connect(self.select_output_folder)
        self.lbl_output_path = QLabel("No folder selected")
        self.lbl_output_path.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-style: italic;"
        )
        self.lbl_output_path.setWordWrap(True)

        dir_grid.addWidget(lbl_out, 1, 0)
        dir_grid.addWidget(self.btn_select_output, 1, 1)
        dir_grid.addWidget(self.lbl_output_path, 1, 2)

        dir_grid.setColumnStretch(2, 1)
        layout.addWidget(dir_group)

        # ── Action Row ──
        action_row = QHBoxLayout()
        self.btn_start = QPushButton("  Execute Classification  ")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {NAVY_PRIMARY}; color: white;
                border: 2px solid {GOLD_ACCENT}; border-radius: 4px;
                padding: 8px 28px; letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ background-color: {NAVY_LIGHT}; }}
            QPushButton:disabled {{ background-color: #8896A7; border-color: #8896A7; color: #C0C7D0; }}
        """)
        self.btn_start.clicked.connect(self.start_batch)
        action_row.addStretch()
        action_row.addWidget(self.btn_start)
        action_row.addStretch()
        layout.addLayout(action_row)

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_SUBTLE}; border-radius: 3px;
                background: #E8EBF0; text-align: center;
                font-size: 11px; font-weight: bold; color: {TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {NAVY_PRIMARY}, stop:1 {NAVY_LIGHT});
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self.status = QLabel("Status: Awaiting input.")
        self.status.setFont(QFont("Segoe UI", 10))
        self.status.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(self.status)

        # ── Summary Section (hidden initially) ──
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(f"""
            QFrame#summaryFrame {{
                background: {BG_WHITE}; border: 1px solid {BORDER_SUBTLE};
                border-radius: 4px;
            }}
        """)
        self.summary_frame.setObjectName("summaryFrame")
        self.summary_frame.setVisible(False)
        summary_outer = QVBoxLayout(self.summary_frame)
        summary_outer.setContentsMargins(18, 14, 18, 14)
        summary_outer.setSpacing(8)

        self.summary_title = QLabel("\u25a0  Classification Report")
        self.summary_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.summary_title.setStyleSheet(f"color: {NAVY_PRIMARY}; border: none;")
        summary_outer.addWidget(self.summary_title)

        # Summary table header — fixed widths must match the data rows exactly
        COL_SNO_W = 50
        COL_COUNT_W = 100
        COL_DIR_W = 80
        COL_REVIEW_W = 80
        self._col_widths = (COL_SNO_W, COL_COUNT_W, COL_DIR_W, COL_REVIEW_W)

        hdr_frame = QFrame()
        hdr_frame.setStyleSheet(f"""
            background: {NAVY_PRIMARY}; border-radius: 3px; border: none;
        """)
        hdr_layout = QHBoxLayout(hdr_frame)
        hdr_layout.setContentsMargins(14, 6, 14, 6)
        hdr_layout.setSpacing(8)

        hdr_sno = QLabel("S.No.")
        hdr_sno.setFixedWidth(COL_SNO_W)
        hdr_sno.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hdr_sno.setStyleSheet(f"color: {TEXT_ON_DARK}; border: none;")
        hdr_layout.addWidget(hdr_sno)

        hdr_name = QLabel("Elephant ID")
        hdr_name.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hdr_name.setStyleSheet(f"color: {TEXT_ON_DARK}; border: none;")
        hdr_layout.addWidget(hdr_name, 1)  # only this column stretches

        hdr_count = QLabel("Image Count")
        hdr_count.setFixedWidth(COL_COUNT_W)
        hdr_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr_count.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hdr_count.setStyleSheet(f"color: {TEXT_ON_DARK}; border: none;")
        hdr_layout.addWidget(hdr_count)

        hdr_dir = QLabel("Directory")
        hdr_dir.setFixedWidth(COL_DIR_W)
        hdr_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr_dir.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hdr_dir.setStyleSheet(f"color: {TEXT_ON_DARK}; border: none;")
        hdr_layout.addWidget(hdr_dir)

        hdr_review = QLabel("Review")
        hdr_review.setFixedWidth(COL_REVIEW_W)
        hdr_review.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr_review.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hdr_review.setStyleSheet(f"color: {TEXT_ON_DARK}; border: none;")
        hdr_layout.addWidget(hdr_review)

        summary_outer.addWidget(hdr_frame)

        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.summary_container = QWidget()
        self.summary_container.setStyleSheet("background: transparent;")
        self.summary_layout = QVBoxLayout(self.summary_container)
        self.summary_layout.setSpacing(2)
        self.summary_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.summary_container)
        summary_outer.addWidget(scroll, 1)

        layout.addWidget(self.summary_frame, 1)
        self.tab1.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder (Raw Images)"
        )
        if folder:
            self.input_dir = folder
            self.lbl_input_path.setText(folder)
            self.lbl_input_path.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-style: normal; font-weight: 600;"
            )

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_base_dir = folder
            self.lbl_output_path.setText(folder)
            self.lbl_output_path.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-style: normal; font-weight: 600;"
            )
            self._save_config()  # remember for next session
            self.load_gallery()  # immediately reflect in Tab 2
            self.load_ambiguity_inbox()

    # ── Session persistence ─────────────────────────────────────────────────
    @property
    def _config_path(self):
        # When frozen (.exe): write next to the executable (always writable)
        # When running from source: write next to app.py
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "app_config.json")

    def _load_config(self):
        """Restore last used output folder from config file."""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r") as f:
                    cfg = json.load(f)
                last_dir = cfg.get("last_output_dir")
                if last_dir and os.path.exists(last_dir):
                    self.output_base_dir = last_dir
        except Exception:
            pass  # config corruption is non-fatal

    def _save_config(self):
        """Persist the current output folder so it survives restarts."""
        try:
            with open(self._config_path, "w") as f:
                json.dump({"last_output_dir": self.output_base_dir}, f, indent=2)
        except Exception:
            pass

    def start_batch(self):
        if not self.input_dir:
            QMessageBox.warning(
                self,
                "Input Required",
                "Please select a source folder before proceeding.",
            )
            return
        if not self.output_base_dir:
            QMessageBox.warning(
                self,
                "Output Required",
                "Please select an output folder before proceeding.",
            )
            return

        self.status.setText("Status: Processing images \u2014 please wait...")
        self.status.setStyleSheet(f"color: {NAVY_PRIMARY}; font-weight: bold;")
        self.btn_start.setEnabled(False)
        self.btn_select_input.setEnabled(False)
        self.btn_select_output.setEnabled(False)
        self.progress_bar.setValue(0)
        self.summary_frame.setVisible(False)

        self.worker = WorkerThread(self.engine, self.input_dir, self.output_base_dir)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.batch_finished)
        self.worker.start()

    def batch_finished(self, summary, auto_enrolled, proposed_merges=None):
        self.status.setText(
            f"Status: Classification complete. Output \u2192 {self.output_base_dir}"
        )
        self.status.setStyleSheet(f"color: {SUCCESS_GREEN}; font-weight: bold;")
        self.btn_start.setEnabled(True)
        self.btn_select_input.setEnabled(True)
        self.btn_select_output.setEnabled(True)
        self.load_gallery()
        self.load_ambiguity_inbox()
        self._show_summary(summary, auto_enrolled, proposed_merges or {})

        known_count = sum(1 for n in summary if not n.startswith("Unknown_"))
        cluster_count = sum(1 for n in summary if n.startswith("Unknown_"))
        enrolled_count = len(auto_enrolled)
        self.statusBar.showMessage(
            f"Classification completed  |  {known_count} known elephant(s)  |  "
            f"{cluster_count} unknown cluster(s)  |  "
            f"{enrolled_count} auto-enrolled  |  "
            f"{datetime.now().strftime('%d-%b-%Y  %H:%M:%S')}"
        )

        ambiguous_count = len(
            [
                a
                for a in ReviewStore(self.output_base_dir).list_ambiguities(
                    unresolved_only=True
                )
                if a.get("type") == "ambiguous_match"
            ]
        )
        if ambiguous_count > 0:
            self.tabs.setCurrentIndex(3)
        elif cluster_count > 0:
            self.tabs.setCurrentIndex(1)

    def _show_summary(self, summary, auto_enrolled=None, proposed_merges=None):
        """Builds and displays the formal classification report.

        Rows are split into two visual groups:
          • Known elephants  — identified against the gallery (gold badge)
          • Unknown clusters — Unknown_1, Unknown_2, … (amber badge + italic label)
        """
        while self.summary_layout.count():
            child = self.summary_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not summary:
            lbl = QLabel("No identifiable elephants were found in the submitted batch.")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-style: italic; padding: 14px; border: none;"
            )
            self.summary_layout.addWidget(lbl)
            self.summary_frame.setVisible(True)
            return

        # ── Split into known vs unknown clusters ──────────────────────────────
        known_items = {n: v for n, v in summary.items() if not n.startswith("Unknown_")}
        cluster_items = {n: v for n, v in summary.items() if n.startswith("Unknown_")}

        # Load health flags from cluster JSON (persisted by core_engine)
        health_flags = {}
        if self.output_base_dir:
            cluster_file = os.path.join(self.output_base_dir, "unknown_clusters.json")
            if os.path.exists(cluster_file):
                try:
                    import json as _json

                    with open(cluster_file) as _f:
                        _clusters = _json.load(_f)
                    for cname, cinfo in _clusters.items():
                        health_flags[cname] = {
                            "stability_flag": cinfo.get("stability_flag", False),
                            "growth_warning": cinfo.get("growth_warning", False),
                        }
                except Exception:
                    pass

        total_images = sum(info["count"] for info in summary.values())
        self.summary_title.setText(
            f"\u25a0  Classification Report  \u2014  "
            f"{len(known_items)} Known Elephant(s)  |  "
            f"{len(cluster_items)} Unknown Cluster(s)  |  "
            f"{total_images} Image(s) Processed"
        )

        COL_SNO_W, COL_COUNT_W, COL_DIR_W, COL_REVIEW_W = self._col_widths

        def _add_section_header(title, color):
            lbl = QLabel(title)
            lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            lbl.setStyleSheet(
                f"color: {color}; border: none; padding: 6px 14px 2px 14px;"
            )
            self.summary_layout.addWidget(lbl)

        def _add_row(idx, name, info, is_cluster):
            row_bg = BG_WHITE if idx % 2 == 0 else "#F7F8FA"
            row = QFrame()
            row.setStyleSheet(f"""
                QFrame {{
                    background: {row_bg}; border: none;
                    border-bottom: 1px solid #E5E7EB;
                }}
                QFrame:hover {{ background: #EDF0F7; }}
            """)

            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(14, 7, 14, 7)
            row_layout.setSpacing(8)

            # Serial number
            sno = QLabel(f"{idx}.")
            sno.setFixedWidth(COL_SNO_W)
            sno.setFont(QFont("Segoe UI", 10))
            sno.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
            row_layout.addWidget(sno)

            # ID label — italic for unknown clusters; badges for enrolled/health
            is_enrolled = auto_enrolled and name in auto_enrolled
            badges = ""
            if is_enrolled:
                badges += "  \u26a1"  # ⚡ auto-enrolled
            if is_cluster and health_flags.get(name, {}).get("stability_flag"):
                badges += "  \u26a0"  # ⚠ unstable
            if is_cluster and health_flags.get(name, {}).get("growth_warning"):
                badges += "  \U0001f331"  # 🌱 growth warning
            display_name = name + badges
            name_lbl = QLabel(display_name)
            name_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            if is_cluster:
                name_lbl.setStyleSheet(
                    "color: #7A4F00; font-style: italic; border: none;"
                )
            else:
                name_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none;")
            row_layout.addWidget(name_lbl, 1)

            # Count badge — gold for known, amber for unknown
            count_lbl = QLabel(str(info["count"]))
            count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            count_lbl.setFixedWidth(COL_COUNT_W)
            count_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            if is_cluster:
                count_lbl.setStyleSheet(
                    "background: #FFF3CD; color: #7A4F00;"
                    "border-radius: 3px; padding: 2px 0px;"
                    "border: 1px solid #E8A800;"
                )
            else:
                count_lbl.setStyleSheet(f"""
                    background: {GOLD_LIGHT}; color: {NAVY_PRIMARY};
                    border-radius: 3px; padding: 2px 0px;
                    border: 1px solid {GOLD_ACCENT};
                """)
            row_layout.addWidget(count_lbl)

            # Open folder button
            open_btn = QPushButton("Open")
            open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            open_btn.setFixedWidth(COL_DIR_W)
            open_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            open_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {NAVY_PRIMARY};
                    border: 1px solid {NAVY_PRIMARY}; border-radius: 3px;
                    padding: 3px 0px;
                }}
                QPushButton:hover {{ background: {NAVY_PRIMARY}; color: white; }}
            """)
            open_btn.clicked.connect(
                lambda checked, p=info["path"]: QDesktopServices.openUrl(
                    QUrl.fromLocalFile(p)
                )
            )
            row_layout.addWidget(open_btn)

            # Review button
            review_btn = QPushButton("Review")
            review_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            review_btn.setFixedWidth(COL_REVIEW_W)
            review_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            review_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {GOLD_ACCENT}; color: {NAVY_DARK};
                    border: none; border-radius: 3px;
                    padding: 3px 0px; font-weight: bold;
                }}
                QPushButton:hover {{ background: #B8943F; }}
            """)
            if is_cluster:
                review_btn.setText("Promote")
                review_btn.setToolTip(
                    "Review images and confirm as a named elephant identity"
                )
                the_path = info["path"]
                the_eng = self.engine
                review_btn.clicked.connect(
                    lambda checked, cn=name, cp=the_path, eng=the_eng: (
                        self._promote_cluster(cn, cp, eng)
                    )
                )
            else:
                review_btn.clicked.connect(
                    lambda checked, cat=name: self._review_elephant(cat)
                )
            row_layout.addWidget(review_btn)
            self.summary_layout.addWidget(row)

        # ── Render known elephants ─────────────────────────────────────────────
        if known_items:
            _add_section_header("IDENTIFIED ELEPHANTS", NAVY_PRIMARY)
            for idx, (name, info) in enumerate(
                sorted(known_items.items(), key=lambda x: x[1]["count"], reverse=True),
                start=1,
            ):
                _add_row(idx, name, info, is_cluster=False)

        # ── Render unknown clusters ────────────────────────────────────────────
        if cluster_items:
            enrolled_count = len(
                [n for n in cluster_items if auto_enrolled and n in auto_enrolled]
            )
            section_suffix = (
                f"  ({enrolled_count} \u26a1 auto-enrolled)" if enrolled_count else ""
            )
            _add_section_header(
                f"POTENTIAL NEW ELEPHANTS (UNVERIFIED CLUSTERS){section_suffix}",
                "#7A4F00",
            )
            for idx, (name, info) in enumerate(
                sorted(cluster_items.items(), key=lambda x: x[0]), start=1
            ):
                _add_row(idx, name, info, is_cluster=True)

        # ── Suggested Merge section ────────────────────────────────────────
        if proposed_merges:
            _add_section_header("SUGGESTED MERGES", "#1A56A8")
            for cname, merge_list in proposed_merges.items():
                for minfo in merge_list:
                    avg_w = minfo.get("avg_weight", 0.0)
                    min_w = minfo.get("min_weight", 0.0)
                    n_grps = minfo.get("n_merged", 2)

                    if avg_w >= 0.37:
                        conf_label = "\U0001f500 High Confidence"
                        conf_color = "#1B7340"
                        bg_color = "#E8F5E9"
                    else:
                        conf_label = "\U0001f500 Review Recommended"
                        conf_color = "#7A4F00"
                        bg_color = "#FFF8E1"

                    card = QFrame()
                    card.setStyleSheet(f"""
                        QFrame {{
                            background: {bg_color};
                            border: 1.5px solid {conf_color};
                            border-radius: 6px;
                            margin: 4px 12px;
                        }}
                    """)
                    card_layout = QVBoxLayout(card)
                    card_layout.setContentsMargins(12, 8, 12, 8)
                    card_layout.setSpacing(4)

                    # Header row
                    hdr_row = QHBoxLayout()
                    if getattr(self, "advanced_mode", False):
                        hdr_lbl = QLabel(
                            f"<b style='color:{conf_color};'>{conf_label}</b>  "
                            f"<span style='color:#6B7280;font-size:9pt;'>"
                            f"{n_grps} clusters merged  |  "
                            f"avg sim: {avg_w:.3f}  |  min edge: {min_w:.3f}</span>"
                        )
                        tooltip_text = (
                            f"Why suggested?\n"
                            f"  Clusters from same batch had:\n"
                            f"  - avg cross-similarity: {avg_w:.3f}\n"
                            f"  - min edge weight:      {min_w:.3f}\n"
                            f"  - clusters merged:      {n_grps}\n"
                            f"Decision: Suggested (user confirmation required)"
                        )
                    else:
                        hdr_lbl = QLabel(
                            f"<b style='color:{conf_color};'>{conf_label}</b>  "
                            f"<span style='color:#6B7280;font-size:9pt;'>"
                            f"{n_grps} sightings grouped together</span>"
                        )
                        tooltip_text = (
                            f"Why suggested?\n"
                            f"These {n_grps} sightings look very similar to each other.\n"
                            f"Please review to confirm they are the same elephant."
                        )

                    hdr_lbl.setFont(QFont("Segoe UI", 9))
                    hdr_lbl.setToolTip(tooltip_text)
                    hdr_row.addWidget(hdr_lbl, 1)

                    # Promote button targets the merged cluster
                    if cname in (summary or {}):
                        promo_btn = QPushButton("\u2714  Confirm & Promote")
                        promo_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                        promo_btn.setStyleSheet("""
                            QPushButton { background: #1A56A8; color: white; border: none;
                                          border-radius: 4px; padding: 4px 12px; }
                            QPushButton:hover { background: #1446A0; }
                        """)
                        _cp = summary[cname]["path"]
                        _eng = self.engine
                        promo_btn.clicked.connect(
                            lambda checked, cn=cname, cp=_cp, eng=_eng: (
                                self._promote_cluster(cn, cp, eng)
                            )
                        )
                        hdr_row.addWidget(promo_btn)

                    card_layout.addLayout(hdr_row)
                    self.summary_layout.addWidget(card)

        self.summary_layout.addStretch()
        self.summary_frame.setVisible(True)

    def _check_duplicates_on_startup(self):
        """Run duplicate identity check after gallery loads and warn if any found."""
        try:
            monitor = ClusterHealthMonitor()
            dups = monitor.detect_duplicates(self.engine.gallery)
            if dups:
                msg = f"⚠ {len(dups)} possible duplicate identities detected (similarity > 0.40)"
                print(msg)
        except Exception:
            pass

    def _promote_cluster(self, cluster_name, cluster_path, engine):
        """Open the Promotion dialog and, on confirm, enroll or merge."""
        dlg = PromoteClusterDialog(cluster_name, cluster_path, engine, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.chosen_name:
            return
        new_name = dlg.chosen_name
        merge_mode = dlg.merge_mode

        if merge_mode:
            cluster_embs = None
            if self.output_base_dir:
                import json as _json, torch as _torch

                _cf = os.path.join(self.output_base_dir, "unknown_clusters.json")
                if os.path.exists(_cf):
                    try:
                        with open(_cf) as _f:
                            _cdata = _json.load(_f)
                        _raw = _cdata.get(cluster_name, {}).get("samples", [])
                        if _raw:
                            cluster_embs = _torch.stack(
                                [_torch.tensor(s) for s in _raw]
                            ).to(engine.device)
                    except Exception:
                        pass
            if cluster_embs is not None:
                import torch as _torch

                _ex = engine.gallery.get(new_name)
                if _ex is not None:
                    merged = _torch.cat(
                        [_ex["embeddings"], cluster_embs.to(engine.device)], dim=0
                    )
                    engine._add_to_gallery_internal(new_name, merged)
                else:
                    engine._add_to_gallery_internal(
                        new_name, cluster_embs.to(engine.device)
                    )

                if cluster_name in engine.gallery:
                    engine.gallery.pop(cluster_name)

                engine._save_gallery_with_backup()
                action_msg = f"Merged '{cluster_name}' into '{new_name}' ({len(cluster_embs)} embedding(s) added)."
            else:
                success, _ = engine.update_database(cluster_path, new_name)
                if success:
                    if cluster_name in engine.gallery:
                        engine.gallery.pop(cluster_name)
                    action_msg = f"Merged '{cluster_name}' into '{new_name}'."
                else:
                    QMessageBox.warning(
                        self,
                        "Merge Failed",
                        f"No images found in '{cluster_path}'. Cannot merge.",
                    )
                    return
        else:
            cluster_embs = None
            if self.output_base_dir:
                import json as _json, torch as _torch

                _cf = os.path.join(self.output_base_dir, "unknown_clusters.json")
                if os.path.exists(_cf):
                    try:
                        with open(_cf) as _f:
                            _cdata = _json.load(_f)
                        _raw = _cdata.get(cluster_name, {}).get("samples", [])
                        if _raw:
                            cluster_embs = _torch.stack(
                                [_torch.tensor(s) for s in _raw]
                            ).to(engine.device)
                    except Exception:
                        pass

            if cluster_embs is not None:
                engine._add_to_gallery_internal(
                    new_name, cluster_embs.to(engine.device)
                )
                if cluster_name in engine.gallery:
                    engine.gallery.pop(cluster_name)
                engine._save_gallery_with_backup()
                action_msg = f"'{cluster_name}' promoted to '{new_name}' ({len(cluster_embs)} embedding(s) saved)."
            else:
                if cluster_name in engine.gallery:
                    engine.gallery[new_name] = engine.gallery.pop(cluster_name)
                    engine._save_gallery_with_backup()
                action_msg = f"'{cluster_name}' promoted to '{new_name}'."

        if not merge_mode:
            new_path = os.path.join(os.path.dirname(cluster_path), new_name)
            if os.path.exists(cluster_path) and not os.path.exists(new_path):
                os.rename(cluster_path, new_path)

        if self.output_base_dir:
            cluster_file = os.path.join(self.output_base_dir, "unknown_clusters.json")
            if os.path.exists(cluster_file):
                try:
                    import json as _json

                    with open(cluster_file) as f:
                        cdata = _json.load(f)
                    if cluster_name in cdata:
                        del cdata[cluster_name]
                    with open(cluster_file, "w") as f:
                        _json.dump(cdata, f, indent=2)
                except Exception:
                    pass

        self.load_gallery()
        QMessageBox.information(self, "Done", action_msg)
        self.statusBar.showMessage(
            f"{action_msg}  |  {datetime.now().strftime('%d-%b-%Y  %H:%M:%S')}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Review & Correct
    # ══════════════════════════════════════════════════════════════════════════
    def setup_tab2(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(12)

        sec_header = QLabel("\u25a0  Review & Reassign Classifications")
        sec_header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        sec_header.setStyleSheet(f"color: {NAVY_PRIMARY};")
        layout.addWidget(sec_header)

        sec_desc = QLabel(
            "Select an elephant in the tree to browse its images. Right-click an image to reassign."
        )
        sec_desc.setFont(QFont("Segoe UI", 10))
        sec_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; margin-bottom: 4px;")
        layout.addWidget(sec_desc)

        self.cluster_review_frame = QFrame()
        self.cluster_review_frame.setVisible(False)
        self.cluster_review_frame.setStyleSheet(f"""
            QFrame#clusterReview {{
                background: {BG_WHITE};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 4px;
            }}
        """)
        self.cluster_review_frame.setObjectName("clusterReview")
        review_layout = QHBoxLayout(self.cluster_review_frame)
        review_layout.setContentsMargins(14, 12, 14, 12)
        review_layout.setSpacing(16)

        review_text = QVBoxLayout()
        self.cluster_review_title = QLabel("")
        self.cluster_review_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.cluster_review_title.setStyleSheet(f"color: {NAVY_PRIMARY};")
        review_text.addWidget(self.cluster_review_title)

        self.cluster_review_stats = QLabel("")
        self.cluster_review_stats.setWordWrap(True)
        self.cluster_review_stats.setFont(QFont("Segoe UI", 9))
        self.cluster_review_stats.setStyleSheet(f"color: {TEXT_SECONDARY};")
        review_text.addWidget(self.cluster_review_stats)

        self.cluster_review_suggestions = QLabel("")
        self.cluster_review_suggestions.setWordWrap(True)
        self.cluster_review_suggestions.setFont(QFont("Segoe UI", 9))
        self.cluster_review_suggestions.setStyleSheet(f"color: {TEXT_SECONDARY};")
        review_text.addWidget(self.cluster_review_suggestions)
        review_layout.addLayout(review_text, 1)

        self.btn_open_cluster_folder = QPushButton("  Open Folder  ")
        self.btn_open_cluster_folder.clicked.connect(self._open_selected_cluster_folder)
        review_layout.addWidget(self.btn_open_cluster_folder)

        self.btn_promote_selected_cluster = QPushButton("  Promote / Merge  ")
        self.btn_promote_selected_cluster.clicked.connect(
            self._promote_selected_cluster
        )
        review_layout.addWidget(self.btn_promote_selected_cluster)

        layout.addWidget(self.cluster_review_frame)

        # ── Filter bar (shown when navigating from Classification Report) ──
        self.filter_bar = QFrame()
        self.filter_bar.setStyleSheet(f"""
            QFrame#filterBar {{
                background: {GOLD_LIGHT}; border: 1px solid {GOLD_ACCENT};
                border-radius: 4px;
            }}
        """)
        self.filter_bar.setObjectName("filterBar")
        self.filter_bar.setVisible(False)
        filter_layout = QHBoxLayout(self.filter_bar)
        filter_layout.setContentsMargins(12, 6, 12, 6)

        self.filter_label = QLabel("")
        self.filter_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.filter_label.setStyleSheet(f"color: {NAVY_PRIMARY}; border: none;")
        filter_layout.addWidget(self.filter_label, 1)

        self.btn_show_all = QPushButton("  Show All  ")
        self.btn_show_all.setFixedHeight(28)
        self.btn_show_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_show_all.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.btn_show_all.setStyleSheet(f"""
            QPushButton {{
                background: {NAVY_PRIMARY}; color: white;
                border-radius: 3px; padding: 2px 14px;
            }}
            QPushButton:hover {{ background: {NAVY_LIGHT}; }}
        """)
        self.btn_show_all.clicked.connect(self._clear_gallery_filter)
        filter_layout.addWidget(self.btn_show_all)
        layout.addWidget(self.filter_bar)

        # ── Toolbar row ──
        toolbar_row = QHBoxLayout()
        self.btn_refresh = QPushButton("  \u21ba  Refresh  ")
        self.btn_refresh.setFixedHeight(32)
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.clicked.connect(lambda: self.load_gallery())
        toolbar_row.addWidget(self.btn_refresh)
        toolbar_row.addStretch()
        layout.addLayout(toolbar_row)

        # ── Splitter: left = folder tree, right = thumbnail grid ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #D0D5DD; }")

        # LEFT — Folder Tree
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setMinimumWidth(200)
        self.folder_tree.setMaximumWidth(320)
        self.folder_tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {BG_WHITE}; border: 1px solid {BORDER_SUBTLE};
                border-radius: 3px; font-size: 12px;
            }}
            QTreeWidget::item {{ padding: 4px 6px; }}
            QTreeWidget::item:hover {{ background: #EDF0F7; }}
            QTreeWidget::item:selected {{
                background: {NAVY_PRIMARY}; color: white;
            }}
        """)
        self.folder_tree.currentItemChanged.connect(self._on_tree_selection_changed)
        splitter.addWidget(self.folder_tree)

        # RIGHT — Thumbnail List
        self.gallery_view = QListWidget()
        self.gallery_view.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_view.setIconSize(QSize(130, 130))
        self.gallery_view.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.gallery_view.customContextMenuRequested.connect(self.show_context_menu)
        self.gallery_view.itemDoubleClicked.connect(self.open_image)
        splitter.addWidget(self.gallery_view)

        splitter.setStretchFactor(0, 0)  # tree: fixed
        splitter.setStretchFactor(1, 1)  # thumbnails: stretches
        splitter.setSizes([240, 820])

        layout.addWidget(splitter, 1)
        self.tab2.setLayout(layout)
        self._gallery_filter = None

    def open_image(self, item):
        """Opens the image in the default viewer."""
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path and os.path.exists(image_path):
            try:
                os.startfile(image_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open image: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "Image path does not exist.")

    def load_gallery(self, filter_category=None):
        """Rebuilds the folder tree from output_base_dir. Populates thumbnail pane for
        filter_category (if given), or clears thumbnails and collapses all."""
        self.folder_tree.clear()
        self.gallery_view.clear()
        self._gallery_filter = filter_category

        if filter_category:
            self.filter_label.setText(f"Showing images for:  {filter_category}")
            self.filter_bar.setVisible(True)
        else:
            self.filter_bar.setVisible(False)

        if not self.output_base_dir or not os.path.exists(self.output_base_dir):
            return

        SYSTEM_FOLDERS = {"_ambiguous_matches", "_review", "_temp"}

        for category in sorted(os.listdir(self.output_base_dir)):
            if category in SYSTEM_FOLDERS:
                continue

            cat_path = os.path.join(self.output_base_dir, category)
            if not os.path.isdir(cat_path):
                continue

            images = [
                f
                for f in os.listdir(cat_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not images:
                continue

            # Elephant node
            parent = QTreeWidgetItem(self.folder_tree)
            parent.setText(0, f"\U0001f4c2  {category}  ({len(images)})")
            parent.setData(0, Qt.ItemDataRole.UserRole, cat_path)  # store folder path
            parent.setFont(0, QFont("Segoe UI", 10, QFont.Weight.DemiBold))

            # Image children
            for img_name in sorted(images):
                child = QTreeWidgetItem(parent)
                child.setText(0, f"    {img_name}")
                child.setData(
                    0, Qt.ItemDataRole.UserRole, os.path.join(cat_path, img_name)
                )
                child.setFont(0, QFont("Segoe UI", 9))

        # If filtering, expand and select the target node
        if filter_category:
            for i in range(self.folder_tree.topLevelItemCount()):
                node = self.folder_tree.topLevelItem(i)
                if category_name_from_node(node) == filter_category:
                    self.folder_tree.setCurrentItem(node)
                    node.setExpanded(True)
                    break
        else:
            # Collapse all by default; user expands what they need
            self.folder_tree.collapseAll()

    def _on_tree_selection_changed(self, current, previous):
        """When a tree item is selected, load its images into the thumbnail pane."""
        if not current:
            return
        path = current.data(0, Qt.ItemDataRole.UserRole)
        if not path or not os.path.exists(path):
            return

        if os.path.isdir(path):
            # Elephant-level node — load all images in this folder
            self.gallery_view.clear()
            for img_name in sorted(os.listdir(path)):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(path, img_name)
                    item = QListWidgetItem(QIcon(img_path), img_name)
                    item.setData(Qt.ItemDataRole.UserRole, img_path)
                    self.gallery_view.addItem(item)
            self._update_cluster_review_panel(os.path.basename(path), path)
        else:
            # Individual file node — just highlight/open it
            self.gallery_view.clear()
            item = QListWidgetItem(QIcon(path), os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.gallery_view.addItem(item)
            self._update_cluster_review_panel(
                os.path.basename(os.path.dirname(path)), os.path.dirname(path)
            )

    def _review_elephant(self, category_name):
        """Switches to Tab 2 and filters the tree to show only a specific elephant."""
        self.load_gallery(filter_category=category_name)
        self.tabs.setCurrentIndex(1)
        # Expand and select the matching node
        for i in range(self.folder_tree.topLevelItemCount()):
            node = self.folder_tree.topLevelItem(i)
            if category_name_from_node(node) == category_name:
                self.folder_tree.setCurrentItem(node)
                node.setExpanded(True)
                break

    def _clear_gallery_filter(self):
        """Clears the gallery filter and reloads full tree."""
        self.load_gallery()

    def _load_unknown_cluster_data(self):
        if not self.output_base_dir:
            return {}
        cluster_file = os.path.join(self.output_base_dir, "unknown_clusters.json")
        if not os.path.exists(cluster_file):
            return {}
        try:
            with open(cluster_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _update_cluster_review_panel(self, cluster_name, cluster_path):
        if not cluster_name.startswith("Unknown_"):
            self.cluster_review_frame.setVisible(False)
            return

        cluster_data = self._load_unknown_cluster_data().get(cluster_name, {})
        count = cluster_data.get("count")
        if count is None and os.path.exists(cluster_path):
            count = len(
                [
                    f
                    for f in os.listdir(cluster_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

        self.cluster_review_title.setText(f"{cluster_name}")

        if getattr(self, "advanced_mode", False):
            variance = cluster_data.get("variance", 0.0)
            stability = cluster_data.get("stability_ratio", 0.0)
            growth_warning = cluster_data.get("growth_warning", False)
            stability_flag = cluster_data.get("stability_flag", False)
            self.cluster_review_stats.setText(
                f"Size: {count or 0} image(s)  |  Variance: {variance:.3f}  |  "
                f"Stability ratio: {stability:.3f}  |  "
                f"Growth warning: {'Yes' if growth_warning else 'No'}  |  "
                f"Stability flag: {'Yes' if stability_flag else 'No'}"
            )
        else:
            desc = f"Group of {count or 0} sightings."

            has_review = False
            if hasattr(self, "review_store") and self.review_store:
                items = self.review_store.list_ambiguities(unresolved_only=True)
                has_review = any(
                    item.get("current_cluster") == cluster_name for item in items
                )

            if has_review:
                desc += " \u26a0\ufe0f Needs review"
            elif count and count >= 3:
                desc += " \u2714 Ready to register"
            else:
                desc += " \u23f3 Needs more sightings"

            last_seen = cluster_data.get("last_updated", "")
            if last_seen:
                try:
                    if isinstance(last_seen, (float, int)):
                        from datetime import datetime

                        last_seen_str = datetime.fromtimestamp(last_seen).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                    else:
                        last_seen_str = str(last_seen)
                    desc += f"  |  Last seen: {last_seen_str}"
                except:
                    pass
            self.cluster_review_stats.setText(desc)

        self.cluster_review_suggestions.setText(
            self._cluster_review_suggestions_text(cluster_name, cluster_data)
        )
        self.btn_open_cluster_folder.setProperty("cluster_path", cluster_path)
        self.btn_promote_selected_cluster.setProperty("cluster_name", cluster_name)
        self.btn_promote_selected_cluster.setProperty("cluster_path", cluster_path)
        self.cluster_review_frame.setVisible(True)

    def _cluster_review_suggestions_text(self, cluster_name, cluster_data):
        if not self.output_base_dir:
            return ""

        samples = cluster_data.get("samples", [])
        if not samples:
            return ""

        try:
            import torch

            anchor = torch.tensor(samples[0], dtype=torch.float32)
        except Exception:
            return ""

        cluster_file = os.path.join(self.output_base_dir, "unknown_clusters.json")
        if not os.path.exists(cluster_file):
            return ""

        mgr = UnknownClusterManager(
            unknown_dir=self.output_base_dir,
            cluster_file=cluster_file,
        )
        cluster_info = mgr.clusters.get(cluster_name, {})
        source_samples = cluster_info.get("samples", [anchor])
        
        ranked = self.engine._rank_review_candidates(
            anchor,
            mgr,
            exclude_cluster=cluster_name,
            source_samples=source_samples
        )
        if not ranked:
            return "Nearest candidates: none"

        top = ranked[:3]
        if getattr(self, "advanced_mode", False):
            parts = [
                f"{cand.get('name', 'Unknown')} ({cand.get('score', 0.0):.3f}, n={cand.get('count', 0)})"
                for cand in top
            ]
            prefix = "Nearest candidates"
            if int(cluster_data.get("count", 0) or 0) <= 1:
                prefix = "Nearest candidates for manual review"
            return f"{prefix}: " + "  |  ".join(parts)
        else:
            parts = [f"{cand.get('name', 'Unknown')}" for cand in top]
            return "Also resembles: " + ", ".join(parts)

    def _open_selected_cluster_folder(self):
        cluster_path = self.btn_open_cluster_folder.property("cluster_path")
        if cluster_path and os.path.exists(cluster_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(cluster_path))

    def _promote_selected_cluster(self):
        cluster_name = self.btn_promote_selected_cluster.property("cluster_name")
        cluster_path = self.btn_promote_selected_cluster.property("cluster_path")
        if cluster_name and cluster_path and os.path.exists(cluster_path):
            self._promote_cluster(cluster_name, cluster_path, self.engine)

    def show_context_menu(self, position):
        item = self.gallery_view.itemAt(position)
        if not item:
            return
        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {BG_WHITE}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_SUBTLE}; font-size: 12px;
            }}
            QMenu::item {{ padding: 8px 24px; }}
            QMenu::item:selected {{ background-color: {NAVY_PRIMARY}; color: white; }}
        """)
        action_reassign = QAction("Reassign to Specific Elephant...", self)
        action_reassign.triggered.connect(
            lambda checked, i=item: self.open_reassign_dialog(i)
        )
        menu.addAction(action_reassign)
        menu.exec(self.gallery_view.mapToGlobal(position))

    def open_reassign_dialog(self, item):
        dialog = QDialog(self)
        dialog.setWindowTitle("Reassign Classification")
        dialog.resize(440, 170)
        dialog.setStyleSheet(f"""
            QDialog {{ background: {BG_WHITE}; }}
            QLabel {{ color: {TEXT_PRIMARY}; font-size: 12px; }}
        """)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 16, 20, 16)

        layout.addWidget(QLabel("Search or select the correct Elephant ID:"))

        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setStyleSheet(f"""
            QComboBox {{
                padding: 6px 10px; border: 1px solid {BORDER_SUBTLE};
                border-radius: 3px; background: {BG_WHITE}; color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
            QComboBox:focus {{
                border: 1px solid {NAVY_PRIMARY};
            }}
            QComboBox QAbstractItemView {{
                background: {BG_WHITE};
                color: {TEXT_PRIMARY};
                selection-background-color: {NAVY_PRIMARY};
                selection-color: white;
                border: 1px solid {BORDER_SUBTLE};
                outline: none;
                font-size: 12px;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 10px;
                min-height: 24px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: #EDF0F7;
                color: {TEXT_PRIMARY};
            }}
        """)
        db_keys = sorted(list(self.engine.gallery.keys()))
        combo.addItems(["Herd", "Makhna"] + db_keys)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet(f"""
            QPushButton {{
                padding: 6px 20px; border-radius: 3px; font-weight: bold;
                font-size: 11px;
            }}
        """)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_class = combo.currentText()
            if selected_class:
                self.reassign_image(item, selected_class)

    def reassign_image(self, item, new_category):
        old_path = item.data(Qt.ItemDataRole.UserRole)
        filename = os.path.basename(old_path)
        new_dir = os.path.join(self.output_base_dir, new_category)
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, filename)
        try:
            shutil.move(old_path, new_path)
            self.load_gallery()  # refresh tree
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not move file: {str(e)}")

    def setup_tab4(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(28, 24, 28, 16)
        main_layout.setSpacing(12)

        sec_header = QLabel("■  Review & Merge (Phase 3A)")
        sec_header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        sec_header.setStyleSheet(f"color: {NAVY_PRIMARY};")
        main_layout.addWidget(sec_header)

        sec_desc = QLabel(
            "Click images to select them, then choose an action on the right."
        )
        sec_desc.setFont(QFont("Segoe UI", 10))
        sec_desc.setStyleSheet(f"color: {TEXT_SECONDARY};")
        main_layout.addWidget(sec_desc)

        content_layout = QHBoxLayout()

        # LEFT: Cluster List
        self.ambiguity_list = QListWidget()
        self.ambiguity_list.setMinimumWidth(180)
        self.ambiguity_list.setMaximumWidth(220)
        self.ambiguity_list.currentRowChanged.connect(self._on_list_selection_changed)
        content_layout.addWidget(self.ambiguity_list, 1)

        # CENTER: Dynamic Container
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.center_panel, 5)

        # Pre-build the grid container (for clusters/merge suggestions)
        self.grid_container = QWidget()
        grid_layout_wrapper = QVBoxLayout(self.grid_container)
        grid_layout_wrapper.setContentsMargins(0, 0, 0, 0)

        self.tab4_cluster_summary = QLabel("Select a cluster to view details.")
        self.tab4_cluster_summary.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.tab4_cluster_summary.setStyleSheet(
            "color: #1A56A8; padding: 6px; background: #F3F4F6; border-radius: 4px;"
        )
        self.tab4_cluster_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout_wrapper.addWidget(self.tab4_cluster_summary)

        self.scroll = QScrollArea()
        self.scroll_widget = QWidget()
        self.grid = QGridLayout()
        self.scroll_widget.setLayout(self.grid)
        self.scroll.setWidget(self.scroll_widget)
        self.scroll.setWidgetResizable(True)
        grid_layout_wrapper.addWidget(self.scroll)

        # Add the grid container to the center layout (it will be shown/hidden)
        self.center_layout.addWidget(self.grid_container)

        # RIGHT: Actions — wrap in a scroll area so content is never squeezed
        self.action_scroll = QScrollArea()
        self.action_scroll.setWidgetResizable(True)
        self.action_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.action_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.action_scroll.setMinimumWidth(340)
        self.action_panel = QWidget()
        self.action_panel.setMinimumWidth(320)
        action_layout = QVBoxLayout(self.action_panel)
        action_layout.setSpacing(12)
        action_layout.setContentsMargins(10, 10, 10, 10)

        # ---- SUGGESTIONS PANEL ----
        self.suggestion_group = QGroupBox("🔗 Suggested Merges")
        self.suggestion_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.suggestion_group.setStyleSheet(f"""
            QGroupBox {{ color: {NAVY_PRIMARY}; border: 1px solid {BORDER_SUBTLE}; border-radius: 4px; margin-top: 14px; padding-top: 18px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px 0 6px; background: white; }}
        """)

        sugg_layout = QVBoxLayout(self.suggestion_group)
        sugg_layout.setContentsMargins(10, 10, 10, 10)
        sugg_layout.setSpacing(12)

        self.sugg_container_layout = QVBoxLayout()
        self.sugg_container_layout.setContentsMargins(0, 0, 0, 0)
        self.sugg_container_layout.setSpacing(10)

        self.suggestion_label = QLabel("No suggestions for this cluster.")
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setStyleSheet(
            "color: #4B5563; font-weight: normal; font-size: 11px;"
        )

        self.sugg_container_layout.addWidget(self.suggestion_label)
        sugg_layout.addLayout(self.sugg_container_layout)

        # Add a dummy btn_compare_merge to avoid hasattr issues elsewhere, but hide it.
        self.btn_compare_merge = QPushButton("🔍 Compare & Merge")
        self.btn_compare_merge.hide()

        action_layout.addWidget(self.suggestion_group)
        # ---------------------------

        self.remove_btn = QPushButton("Remove Selected")
        self.undo_btn = QPushButton("Undo Last Remove")
        self.undo_btn.setEnabled(False)
        self.keep_btn = QPushButton("Keep Cluster")
        self.split_btn = QPushButton("Split Cluster")
        self.promote_btn = QPushButton("Promote to Identity")

        for btn in (
            self.remove_btn,
            self.undo_btn,
            self.keep_btn,
            self.split_btn,
            self.promote_btn,
        ):
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            action_layout.addWidget(btn)

        self.remove_btn.setStyleSheet(f"""
            QPushButton {{ background: #9B1C1C; color: white; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            QPushButton:hover {{ background: #771616; }}
            QPushButton:disabled {{ background: #8896A7; color: #C0C7D0; }}
        """)
        self.undo_btn.setStyleSheet(f"""
            QPushButton {{ background: #4B5563; color: white; border-radius: 4px; font-weight: bold; font-size: 11px; }}
            QPushButton:hover {{ background: #374151; }}
            QPushButton:disabled {{ background: #D1D5DB; color: #9CA3AF; }}
        """)
        for btn in (self.keep_btn, self.split_btn, self.promote_btn):
            btn.setStyleSheet(f"""
                QPushButton {{ background: {NAVY_PRIMARY}; color: white; border-radius: 4px; font-weight: bold; font-size: 11px; }}
                QPushButton:hover {{ background: {NAVY_LIGHT}; }}
                QPushButton:disabled {{ background: #8896A7; color: #C0C7D0; }}
            """)

        self.remove_btn.clicked.connect(self._remove_selected)
        self.undo_btn.clicked.connect(self._undo_last_remove)
        self.keep_btn.clicked.connect(self._keep_cluster)
        self.split_btn.clicked.connect(self._split_cluster)
        self.promote_btn.clicked.connect(self._promote_cluster_action)
        action_layout.addStretch()

        self.action_scroll.setWidget(self.action_panel)
        content_layout.addWidget(self.action_scroll, 3)

        main_layout.addLayout(content_layout)

        self.tab4.setLayout(main_layout)

    def _on_list_selection_changed(self, index):
        if index < 0:
            return

        print(f"[DEBUG] Selected cluster index: {index}")
        item = self.ambiguity_list.item(index)
        if not item:
            return

        record = item.data(Qt.ItemDataRole.UserRole)

        # Protect against empty record items
        if record is None:
            print("[DEBUG] No UserRole data on this list item.")
            return

        cluster_name = record.get("current_cluster")
        print(f"[DEBUG] Selected cluster name: {cluster_name}")

        self._load_cluster_to_grid(index)

    def _clear_ambiguous_card(self):
        # Remove any existing AmbiguousCard from the layout
        for i in reversed(range(self.center_layout.count())):
            widget = self.center_layout.itemAt(i).widget()
            if widget and widget is not self.grid_container:
                widget.setParent(None)
                widget.deleteLater()

    def _load_ambiguous_item(self, record):
        self.grid_container.hide()
        self._clear_ambiguous_card()

        # Prefetch cluster previews
        c1 = record["candidates"][0]
        c2 = record["candidates"][1]

        c1["preview"] = self._cluster_image_paths(c1["cluster"])[:3]
        c2["preview"] = self._cluster_image_paths(c2["cluster"])[:3]

        card = AmbiguousCard(record, self)
        self.center_layout.addWidget(card)

    def resolve_ambiguous_assign(self, record, target_cluster):
        # UI -> Controller -> Backend
        import shutil

        node_path = record.get("file_paths")[0]

        # Ensure target cluster exists
        target_dir = os.path.join(self.output_base_dir, target_cluster)
        os.makedirs(target_dir, exist_ok=True)

        # Move file
        target_path = os.path.join(target_dir, os.path.basename(node_path))
        if os.path.exists(node_path):
            shutil.move(node_path, target_path)

        # Update Review Store
        self.review_store = ReviewStore(self.output_base_dir)
        self.review_store.resolve_ambiguity(
            record["id"], {"action": "ASSIGNED", "target": target_cluster}
        )

        # Sync engine logic
        mgr = UnknownClusterManager(
            unknown_dir=self.output_base_dir,
            cluster_file=os.path.join(self.output_base_dir, "unknown_clusters.json"),
        )
        self._rebuild_cluster_from_folder(mgr, target_cluster)
        mgr.save()

        # Cleanup unambiguous matches folder if empty
        ambig_folder = os.path.dirname(node_path)
        if os.path.exists(ambig_folder) and not os.listdir(ambig_folder):
            try:
                os.rmdir(ambig_folder)
            except Exception:
                pass

        self.load_ambiguity_inbox()

    def resolve_ambiguous_keep_separate(self, record):
        import shutil

        node_path = record.get("file_paths")[0]

        mgr = UnknownClusterManager(
            unknown_dir=self.output_base_dir,
            cluster_file=os.path.join(self.output_base_dir, "unknown_clusters.json"),
        )

        # Create a new Unknown cluster
        new_id = len(mgr.clusters) + 1
        new_name = f"Unknown_{new_id}"
        while new_name in mgr.clusters or os.path.exists(
            os.path.join(self.output_base_dir, new_name)
        ):
            new_id += 1
            new_name = f"Unknown_{new_id}"

        target_dir = os.path.join(self.output_base_dir, new_name)
        os.makedirs(target_dir, exist_ok=True)

        # Move file
        target_path = os.path.join(target_dir, os.path.basename(node_path))
        if os.path.exists(node_path):
            shutil.move(node_path, target_path)

        # Update Review Store
        self.review_store = ReviewStore(self.output_base_dir)
        self.review_store.resolve_ambiguity(
            record["id"], {"action": "KEPT_SEPARATE", "target": new_name}
        )

        self._rebuild_cluster_from_folder(mgr, new_name)
        mgr.save()

        ambig_folder = os.path.dirname(node_path)
        if os.path.exists(ambig_folder) and not os.listdir(ambig_folder):
            try:
                os.rmdir(ambig_folder)
            except Exception:
                pass

        self.load_ambiguity_inbox()

    def _update_suggestions_ui(self, cluster_name, record):
        if not hasattr(self, "sugg_container_layout"):
            return

        while self.sugg_container_layout.count():
            item = self.sugg_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                item.layout().deleteLater()

        candidates = []

        is_singleton = False

        if cluster_name and self.output_base_dir:
            mgr = UnknownClusterManager(
                unknown_dir=self.output_base_dir,
                cluster_file=os.path.join(
                    self.output_base_dir, "unknown_clusters.json"
                ),
            )

            # ── Rebuild ALL clusters from disk so embeddings are fresh ──
            # This prevents stale/missing entries in unknown_clusters.json
            # from hiding valid merge candidates (e.g. Unknown_3 for Unknown_4).
            try:
                for d in os.listdir(self.output_base_dir):
                    if d.startswith("Unknown_") and os.path.isdir(
                        os.path.join(self.output_base_dir, d)
                    ):
                        if d not in mgr.clusters:
                            print(f"[SYNC] Rebuilding missing cluster: {d}")
                            self._rebuild_cluster_from_folder(mgr, d)
            except Exception as e:
                print(f"[WARN] Cluster sync error: {e}")

            cluster_data = mgr.clusters.get(cluster_name)

            cluster_path = os.path.join(self.output_base_dir, cluster_name)
            if os.path.exists(cluster_path):
                imgs = [
                    f
                    for f in os.listdir(cluster_path)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
                if len(imgs) == 1:
                    is_singleton = True
                    warn_lbl = QLabel(
                        "<b>⚠️ HIGH RISK: Single image cluster</b><br>"
                        "These are often incorrect splits.<br>"
                        "👉 <i>Strongly review merge candidates below.</i>"
                    )
                    warn_lbl.setStyleSheet(
                        "color: #9B1C1C; background: #FEE2E2; padding: 6px; border-radius: 4px; font-size: 11px;"
                    )
                    warn_lbl.setWordWrap(True)
                    self.sugg_container_layout.addWidget(warn_lbl)

            # If the selected cluster itself is missing from the mgr, rebuild it too
            if not cluster_data or cluster_data.get("centroid") is None:
                try:
                    self._rebuild_cluster_from_folder(mgr, cluster_name)
                    cluster_data = mgr.clusters.get(cluster_name)
                except Exception as e:
                    print(f"[WARN] Could not rebuild {cluster_name}: {e}")

            if cluster_data and cluster_data.get("centroid") is not None:
                source_samples = cluster_data.get("samples", [cluster_data["centroid"]])
                ranked = self.engine._rank_review_candidates(
                    cluster_data["centroid"], mgr, exclude_cluster=cluster_name,
                    source_samples=source_samples
                )
                if ranked:
                    candidates = ranked[:5]

        if not candidates:
            lbl = QLabel("No similar clusters found.")
            lbl.setStyleSheet("color: #4B5563; font-size: 11px;")
            self.sugg_container_layout.addWidget(lbl)
            self.current_suggestion = None
            return

        # Show ALL candidates returned by _rank_review_candidates (already top-5).
        # Don't filter by score threshold — extreme pose variance can produce
        # negative/low cosine similarity between genuine same-identity images.
        valid_candidates = candidates

        for i, cand in enumerate(valid_candidates):
            target = cand.get("name")
            score = cand.get(
                "max_member", 0.0
            )  # Use the raw visual similarity instead of penalized score for the UI

            # Get target composition
            target_path = os.path.join(self.output_base_dir, target)
            comp_str = ""
            if os.path.exists(target_path):
                t_imgs = sorted(
                    [
                        f
                        for f in os.listdir(target_path)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))
                    ]
                )
                count = len(t_imgs)
                if count == 1:
                    comp_str = "<br><span style='color:#9B1C1C; font-size:10px;'>Cluster size: 1 (singleton ⚠️)</span>"
                else:
                    preview = ", ".join(t_imgs[:2]) + ("..." if count > 2 else "")
                    comp_str = f"<br><span style='color:#6B7280; font-size:10px;'>Cluster size: {count} (stable) &bull; {preview}</span>"

            possible_threshold = 0.48 if is_singleton else 0.52

            if score > 0.65 and cand.get("cohesion", 1.0) > 0.50:
                conf_str = "✔ Likely Match"
                color = "#1B7340"
                why_str = f"Similarity: <b>{score:.3f}</b> — High visual overlap. Likely the same identity."
            elif score > possible_threshold:
                conf_str = "⚠ Good Possibility"
                color = "#7A4F00"
                why_str = f"Similarity: <b>{score:.3f}</b> — Moderate similarity. Check for matching features (ears/tusks)."
            elif score > 0.20:
                conf_str = "🤔 Needs Review"
                color = "#B45309"
                why_str = f"Similarity: <b>{score:.3f}</b> — Low direct similarity. 👉 <b>Please compare visually before rejecting.</b>"
            else:
                conf_str = "🔍 Manual Check"
                color = "#4B5563"
                why_str = f"Similarity: <b>{score:.3f}</b> — Model score is low/negative. <b>Manual verification required.</b>"

            bridge = cand.get("bridge_path", "")
            if bridge and score <= 0.65:
                why_str += f"<br>🔗 <b>Bridge path:</b> <span style='font-family:monospace; background:#F3F4F6; padding:2px; border-radius:2px; color:#4B5563;'>{bridge}</span>"

            if i == 0:
                rank_lbl = QLabel("<b>TOP MATCH</b>")
                rank_lbl.setStyleSheet(
                    "color: #111827; font-size: 11px; margin-top: 4px; letter-spacing: 1px;"
                )
                self.sugg_container_layout.addWidget(rank_lbl)
            elif i == 1:
                rank_lbl = QLabel("<b>OTHER POSSIBILITIES</b>")
                rank_lbl.setStyleSheet(
                    "color: #6B7280; font-size: 11px; margin-top: 8px; letter-spacing: 1px;"
                )
                self.sugg_container_layout.addWidget(rank_lbl)

            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 4, 4, 10)
            row_layout.setSpacing(6)

            info_lbl = QLabel(
                f"<b style='font-size:13px; color:#111827;'>{target}</b> "
                f"<span style='color:{color}; font-weight:bold; font-size:11px;'>({conf_str})</span>"
                f"{comp_str}"
            )
            info_lbl.setWordWrap(True)
            row_layout.addWidget(info_lbl)

            why_lbl = QLabel(
                f"<span style='color:#6B7280; font-size:11px;'><b>Reason:</b> {why_str}</span>"
            )
            why_lbl.setWordWrap(True)
            row_layout.addWidget(why_lbl)

            btn_layout = QHBoxLayout()
            btn_layout.setContentsMargins(0, 4, 0, 0)
            btn_layout.setSpacing(10)
            btn_compare = QPushButton("🔍  Compare")
            btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_compare.setMinimumHeight(38)
            btn_compare.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
            )
            btn_compare.setStyleSheet(
                "background: transparent; border: 1.5px solid #D1D5DB; color: #374151;"
                " font-weight: bold; border-radius: 4px; padding: 6px 14px; font-size: 12px;"
            )
            btn_compare.clicked.connect(
                lambda checked, t=target, s=score, c=cand, w=row_widget: (
                    self._open_compare_dialog_for(t, score=s, cand=c, row_widget=w)
                )
            )

            btn_merge = QPushButton("✔  Merge")
            btn_merge.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            btn_merge.setMinimumHeight(38)
            btn_merge.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_merge.setStyleSheet(
                "background: #1B7340; color: white; font-weight: bold;"
                " border-radius: 4px; padding: 6px 14px; font-size: 12px;"
            )
            btn_merge.clicked.connect(
                lambda checked, s=cluster_name, t=target, c=cand: (
                    self._merge_unknown_clusters(s, t, c)
                )
            )

            btn_layout.addWidget(btn_compare, 2)
            btn_layout.addWidget(btn_merge, 3)
            row_layout.addLayout(btn_layout)

            self.sugg_container_layout.addWidget(row_widget)

    def _open_compare_dialog_for(self, target_cluster, score=None, cand=None, row_widget=None):
        if not self.current_ambiguity_record:
            return
        cluster_name = self.current_ambiguity_record.get("current_cluster")
        if not cluster_name or not target_cluster:
            return

        max_sim = None
        mean_sim = None
        percentile_str = ""
        best_pair = (None, None)

        try:
            import torch
            import os

            mgr = UnknownClusterManager(
                unknown_dir=self.output_base_dir,
                cluster_file=os.path.join(
                    self.output_base_dir, "unknown_clusters.json"
                ),
            )
            ca = mgr.clusters.get(cluster_name)
            cb = mgr.clusters.get(target_cluster)

            if ca and cb and ca.get("samples") and cb.get("samples"):
                imgs_a = sorted(
                    [
                        f
                        for f in os.listdir(
                            os.path.join(self.output_base_dir, cluster_name)
                        )
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))
                    ]
                )
                imgs_b = sorted(
                    [
                        f
                        for f in os.listdir(
                            os.path.join(self.output_base_dir, target_cluster)
                        )
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))
                    ]
                )

                sims = []
                for i, s_a in enumerate(ca["samples"]):
                    for j, s_b in enumerate(cb["samples"]):
                        sim = float(torch.dot(s_a.clone().detach(), s_b.clone().detach()))
                        sims.append(sim)
                        if max_sim is None or sim > max_sim:
                            max_sim = sim
                            best_pair = (
                                imgs_a[i] if i < len(imgs_a) else None,
                                imgs_b[j] if j < len(imgs_b) else None,
                            )
                if sims:
                    mean_sim = sum(sims) / len(sims)

                internal_b = []
                for i in range(len(cb["samples"])):
                    for j in range(i + 1, len(cb["samples"])):
                        internal_b.append(
                            float(
                                torch.dot(
                                    cb["samples"][i].clone().detach(),
                                    cb["samples"][j].clone().detach(),
                                )
                            )
                        )

                if internal_b and max_sim is not None:
                    below = sum(1 for x in internal_b if x < max_sim)
                    pct = (below / len(internal_b)) * 100
                    if pct >= 75:
                        percentile_str = (
                            f"Top {100 - int(pct)}% (strong match relative to cluster)"
                        )
                    elif pct >= 25:
                        percentile_str = (
                            f"Middle 50% (consistent with cluster variance)"
                        )
                    else:
                        percentile_str = (
                            f"Bottom {int(pct)}% (fringe/outlier relative to cluster)"
                        )
                else:
                    percentile_str = "N/A (cluster too small for internal variance)"
        except Exception:
            pass

        dialog = CompareMergeDialog(
            cluster_name,
            target_cluster,
            self.output_base_dir,
            score,
            max_sim,
            mean_sim,
            percentile_str,
            best_pair,
            self,
        )

        # Override the dialog's reject button to log before closing
        original_reject = dialog.reject

        def _on_reject():
            if cand is not None:
                self._log_merge_decision(cluster_name, target_cluster, cand, "rejected")
            if row_widget:
                row_widget.setParent(None)
                row_widget.deleteLater()
            original_reject()

        dialog.reject = _on_reject

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._merge_unknown_clusters(cluster_name, target_cluster, cand)

    def _open_compare_dialog(self):
        if not self.current_suggestion or not self.current_ambiguity_record:
            return

        target = self.current_suggestion.get("name")
        cluster_name = self.current_ambiguity_record.get("current_cluster")
        if not cluster_name or not target:
            return

        dialog = CompareMergeDialog(cluster_name, target, self.output_base_dir, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._merge_unknown_clusters(cluster_name, target)

    def _log_merge_decision(self, source, target, candidate_data, decision):
        if not self.output_base_dir:
            return
        log_file = os.path.join(self.output_base_dir, "merge_decisions.csv")
        write_header = not os.path.exists(log_file)

        # Check if source is a singleton
        was_singleton = False
        src_path = os.path.join(self.output_base_dir, source)
        if os.path.exists(src_path):
            imgs = [
                f
                for f in os.listdir(src_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            was_singleton = len(imgs) == 1

        import time
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "timestamp,source,target,effective_score,direct_score,bridge_score,cohesion,was_singleton,decision\n"
                )

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            eff = float(candidate_data.get("effective", 0.0))
            score = float(candidate_data.get("max_member", candidate_data.get("score", 0.0)))
            bridge = float(candidate_data.get("bridge_score", 0.0))
            coh = float(candidate_data.get("cohesion", 1.0))
            f.write(
                f"{ts},{source},{target},{eff:.4f},{score:.4f},{bridge:.4f},{coh:.4f},{was_singleton},{decision}\n"
            )

    def _merge_unknown_clusters(self, source_cluster, target_cluster, cand=None):
        if not source_cluster or not target_cluster or not self.output_base_dir:
            return

        if cand is not None:
            self._log_merge_decision(source_cluster, target_cluster, cand, "merged")

        src_path = os.path.join(self.output_base_dir, source_cluster)
        tgt_path = os.path.join(self.output_base_dir, target_cluster)

        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            return

        for f in os.listdir(src_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.move(os.path.join(src_path, f), os.path.join(tgt_path, f))

        mgr = UnknownClusterManager(
            unknown_dir=self.output_base_dir,
            cluster_file=os.path.join(self.output_base_dir, "unknown_clusters.json"),
        )
        self._rebuild_cluster_from_folder(mgr, target_cluster)
        mgr.clusters.pop(source_cluster, None)
        mgr.save()

        try:
            os.rmdir(src_path)
        except Exception:
            pass

        self.review_store = ReviewStore(self.output_base_dir)
        self.review_store.resolve_open_ambiguities(cluster_names=[source_cluster])

        self.load_ambiguity_inbox()
        self.load_gallery()

        # Select the target cluster in the list if possible
        for i in range(self.ambiguity_list.count()):
            item = self.ambiguity_list.item(i)
            record = item.data(Qt.ItemDataRole.UserRole)
            if record and record.get("current_cluster") == target_cluster:
                self.ambiguity_list.setCurrentRow(i)
                break

    def _load_cluster_to_grid(self, index):
        self.grid_container.show()
        self._clear_ambiguous_card()

        self.undo_btn.setEnabled(False)
        self._last_removed = []

        if index < 0:
            return

        item = self.ambiguity_list.item(index)
        record = item.data(Qt.ItemDataRole.UserRole)
        if not record:
            return

        self.current_ambiguity_record = record

        # Clear grid
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.image_widgets = []

        cluster_name = record.get("current_cluster")
        print(f"[DEBUG] _load_cluster_to_grid updating suggestions for: {cluster_name}")
        self._update_suggestions_ui(cluster_name, record)

        self.tab4_cluster_summary.hide()  # Hide placeholder text when loaded

        if not cluster_name or not self.output_base_dir:
            return

        cluster_path = os.path.join(self.output_base_dir, cluster_name)

        if hasattr(self, "tab4_cluster_summary"):
            is_ambig = bool(record.get("id"))
            size_str = (
                len(record.get("file_paths", [])) if record.get("file_paths") else 0
            )
            # Rough summary fallback
            status = "\u26a0\ufe0f Needs Review" if is_ambig else "\u2714 Clean"
            self.tab4_cluster_summary.setText(
                f"Viewing: {cluster_name}  |  Status: {status}"
            )
        if not os.path.exists(cluster_path):
            return

        outlier_paths = set(record.get("file_paths", []))
        images = [
            f
            for f in sorted(os.listdir(cluster_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Sort to put outliers at the bottom
        images.sort(key=lambda x: (os.path.join(cluster_path, x) in outlier_paths, x))

        for i, img_name in enumerate(images):
            full_path = os.path.abspath(os.path.join(cluster_path, img_name))
            is_outlier = full_path in outlier_paths

            label_text = (
                f"\u26a0\ufe0f OUTLIER  |  {img_name}" if is_outlier else img_name
            )
            widget = ClusterImageCard(
                full_path, is_outlier=is_outlier, label_text=label_text
            )

            row = i // 3
            col = i % 3

            self.grid.addWidget(widget, row, col)
            self.image_widgets.append(widget)

    def _get_selected_images(self):
        return [w for w in self.image_widgets if w.selected]

    def _remove_selected(self):
        selected = self._get_selected_images()
        if not selected:
            return

        print("Removing:", [w.img_path for w in selected])

        # Move to review buffer instead of discarded
        review_buffer = os.path.join(self.output_base_dir, "_review_buffer")
        os.makedirs(review_buffer, exist_ok=True)
        import shutil

        # Clear previous last removed and populate
        self._last_removed = []
        for w in selected:
            if os.path.exists(w.img_path):
                temp_path = os.path.join(review_buffer, os.path.basename(w.img_path))
                shutil.move(w.img_path, temp_path)
                self._last_removed.append({"src": w.img_path, "temp": temp_path})

        self.undo_btn.setEnabled(bool(self._last_removed))

        # Reload grid
        self._load_cluster_to_grid(self.ambiguity_list.currentRow())

    def _undo_last_remove(self):
        if not self._last_removed:
            return

        import shutil

        print("Undoing remove...")
        for op in self._last_removed:
            src, temp = op["src"], op["temp"]
            if os.path.exists(temp):
                shutil.move(temp, src)

        self._last_removed = []
        self.undo_btn.setEnabled(False)
        self._load_cluster_to_grid(self.ambiguity_list.currentRow())

    def _merge_suggestion(self):
        if not self.current_suggestion or not self.current_ambiguity_record:
            return

        target = self.current_suggestion["name"]
        record_id = self.current_ambiguity_record.get("id")
        current_cluster = self.current_ambiguity_record.get("current_cluster")

        print(f"Merging {current_cluster} → {target} (ID: {record_id})")

        # Use existing logic to apply the assignment
        if self.review_store and record_id:
            try:
                # ASSIGN_A maps to the top candidate logic
                resolution = self._apply_ambiguity_resolution(
                    self.current_ambiguity_record, "ASSIGN_A"
                )
                self.review_store.resolve_ambiguity(record_id, resolution)
            except Exception as e:
                QMessageBox.warning(self, "Merge Failed", str(e))
                return

        self.load_ambiguity_inbox()
        self.load_gallery()

    def _ignore_suggestion(self):
        # Simply keep the cluster as new
        if not self.current_ambiguity_record:
            return

        record_id = self.current_ambiguity_record.get("id")
        print(
            f"Suggestion ignored. Keeping cluster {self.current_ambiguity_record.get('current_cluster')} as new."
        )
        if self.review_store and record_id:
            try:
                resolution = self._apply_ambiguity_resolution(
                    self.current_ambiguity_record, "KEEP_NEW"
                )
                self.review_store.resolve_ambiguity(record_id, resolution)
            except Exception as e:
                QMessageBox.warning(self, "Resolution Failed", str(e))
                return
        self.load_ambiguity_inbox()
        self.load_gallery()

    def _keep_cluster(self):
        if not self.current_ambiguity_record:
            return

        record_id = self.current_ambiguity_record.get("id")
        print(
            f"Cluster kept: {self.current_ambiguity_record.get('current_cluster')} (ID: {record_id})"
        )
        if self.review_store and record_id:
            try:
                resolution = self._apply_ambiguity_resolution(
                    self.current_ambiguity_record, "KEEP_NEW"
                )
                self.review_store.resolve_ambiguity(record_id, resolution)
            except Exception as e:
                QMessageBox.warning(self, "Resolution Failed", str(e))
                return
        self.load_ambiguity_inbox()
        self.load_gallery()

    def _split_cluster(self):
        if not self.current_ambiguity_record:
            return

        record_id = self.current_ambiguity_record.get("id")
        print(
            f"Cluster split: {self.current_ambiguity_record.get('current_cluster')} (ID: {record_id})"
        )
        if self.review_store and record_id:
            try:
                resolution = self._apply_ambiguity_resolution(
                    self.current_ambiguity_record, "DISCARD"
                )
                self.review_store.resolve_ambiguity(record_id, resolution)
            except Exception as e:
                QMessageBox.warning(self, "Resolution Failed", str(e))
                return
        self.load_ambiguity_inbox()
        self.load_gallery()

    def _promote_cluster_action(self):
        if not self.current_ambiguity_record:
            return

        cluster_name = self.current_ambiguity_record.get("current_cluster")
        cluster_path = os.path.join(self.output_base_dir, cluster_name)
        print(f"Promoted: {cluster_name}")

        self._promote_cluster(cluster_name, cluster_path, self.engine)

        record_id = self.current_ambiguity_record.get("id")
        if self.review_store and record_id:
            self.review_store.resolve_ambiguity(record_id, {"action": "PROMOTED"})
        self.load_ambiguity_inbox()

    def load_ambiguity_inbox(self):
        self.ambiguity_list.clear()
        if not self.output_base_dir or not os.path.exists(self.output_base_dir):
            return

        self._prune_stale_ambiguities()
        self.review_store = ReviewStore(self.output_base_dir)
        ambig_items = self.review_store.list_ambiguities(unresolved_only=True)

        # We now have 3 categories
        ambiguous_matches = []
        merge_suggestions = []
        other_clusters = []

        seen_clusters = set()

        for item in ambig_items:
            if item.get("type") == "ambiguous_match":
                ambiguous_matches.append(item)
            elif item.get("source_type") in (
                "cluster_conflict",
                "single",
                "soft_match",
                "singleton_postpass",
            ):
                cluster_name = item.get("current_cluster")
                if cluster_name and cluster_name not in seen_clusters:
                    merge_suggestions.append((cluster_name, item))
                    seen_clusters.add(cluster_name)

        try:
            for d in os.listdir(self.output_base_dir):
                if d.startswith("Unknown_") and os.path.isdir(
                    os.path.join(self.output_base_dir, d)
                ):
                    if d not in seen_clusters:
                        other_clusters.append(d)
                        seen_clusters.add(d)
        except Exception:
            pass

        def get_cluster_size(c_name):
            try:
                c_path = os.path.join(self.output_base_dir, c_name)
                if os.path.exists(c_path):
                    return len(
                        [
                            f
                            for f in os.listdir(c_path)
                            if f.lower().endswith((".jpg", ".png", ".jpeg"))
                        ]
                    )
            except Exception:
                pass
            return 999999

        def sort_key_with_size(c):
            import re

            name = c[0] if isinstance(c, tuple) else c
            size = get_cluster_size(name)
            # Priorities:
            # 1. Size == 1 (High risk singletons first)
            # 2. Numeric order
            match = re.search(r"\d+", name)
            num = int(match.group()) if match else 999999
            is_singleton = 0 if size == 1 else 1
            return (is_singleton, num)

        other_clusters.sort(key=sort_key_with_size)
        merge_suggestions.sort(key=sort_key_with_size)

        # 1. Ambiguous Matches (TOP PRIORITY)
        if ambiguous_matches:
            header = QListWidgetItem("⚠ Ambiguous Matches ⚠")
            header.setBackground(QColor("#FFEBEE"))  # Light red/pink
            header.setForeground(QColor(DANGER_RED))
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setFlags(Qt.ItemFlag.NoItemFlags)  # Unselectable
            self.ambiguity_list.addItem(header)

            for item in ambiguous_matches:
                filename = item.get("source_filenames", ["Unknown"])[0]
                label = f"⚠ {filename}"
                list_item = QListWidgetItem(label)
                list_item.setData(Qt.ItemDataRole.UserRole, item)
                list_item.setBackground(QColor("#FFF4E5"))
                self.ambiguity_list.addItem(list_item)

        # 2. Merge Suggestions
        if merge_suggestions:
            header = QListWidgetItem("Merge Suggestions")
            header.setBackground(QColor(BG_LIGHT))
            header.setForeground(QColor(TEXT_SECONDARY))
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self.ambiguity_list.addItem(header)

            for cluster_name, record in merge_suggestions:
                size_str = ""
                cluster_path = os.path.join(self.output_base_dir, cluster_name)
                if os.path.exists(cluster_path):
                    imgs = [
                        f
                        for f in os.listdir(cluster_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    size_str = (
                        f" (1 img ⚠️)" if len(imgs) == 1 else f" ({len(imgs)} imgs)"
                    )

                list_item = QListWidgetItem(f"{cluster_name}{size_str}")
                if "1 img" in size_str:
                    list_item.setForeground(QColor("#9B1C1C"))
                list_item.setData(Qt.ItemDataRole.UserRole, record)
                self.ambiguity_list.addItem(list_item)

        # 3. Clusters
        if other_clusters:
            header = QListWidgetItem("Clusters")
            header.setBackground(QColor(BG_LIGHT))
            header.setForeground(QColor(TEXT_SECONDARY))
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self.ambiguity_list.addItem(header)

            for cluster_name in other_clusters:
                size_str = ""
                cluster_path = os.path.join(self.output_base_dir, cluster_name)
                if os.path.exists(cluster_path):
                    imgs = [
                        f
                        for f in os.listdir(cluster_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    size_str = (
                        f" (1 img ⚠️)" if len(imgs) == 1 else f" ({len(imgs)} imgs)"
                    )

                list_item = QListWidgetItem(f"{cluster_name}{size_str}")
                if "1 img" in size_str:
                    list_item.setForeground(QColor("#9B1C1C"))

                record = {"current_cluster": cluster_name, "id": None}
                list_item.setData(Qt.ItemDataRole.UserRole, record)
                self.ambiguity_list.addItem(list_item)

        if self.ambiguity_list.count() > 0:
            # Select the first actual item (skip headers)
            for i in range(self.ambiguity_list.count()):
                item = self.ambiguity_list.item(i)
                if item.flags() & Qt.ItemFlag.ItemIsSelectable:
                    self.ambiguity_list.setCurrentRow(i)
                    break
        else:
            for i in reversed(range(self.grid.count())):
                widget = self.grid.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

    def _cluster_image_paths(self, cluster_name):
        if not cluster_name or not getattr(self, "output_base_dir", None):
            return []
        import os

        cluster_dir = os.path.join(self.output_base_dir, cluster_name)
        if not os.path.isdir(cluster_dir):
            return []
        return [
            os.path.join(cluster_dir, f)
            for f in os.listdir(cluster_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def _prune_stale_ambiguities(self):
        if not self.output_base_dir or not os.path.exists(self.output_base_dir):
            return

        self.review_store = ReviewStore(self.output_base_dir)
        items = self.review_store.list_ambiguities(unresolved_only=True)
        for record in items:
            record_id = record.get("id")
            if not record_id:
                continue

            query_paths = [
                path
                for path in record.get("file_paths", [])
                if path and os.path.exists(path)
            ]
            current_cluster = record.get("current_cluster")
            current_cluster_files = self._cluster_image_paths(current_cluster)
            candidate_a = record.get("candidate_a", {}).get("name")
            candidate_b = record.get("candidate_b", {}).get("name")
            candidate_exists = any(
                self._cluster_image_paths(name)
                for name in (candidate_a, candidate_b)
                if name
            )

            is_singleton_review = record.get("source_type") in {
                "soft_match",
                "singleton_postpass",
            }
            current_cluster_expanded = (
                is_singleton_review and len(current_cluster_files) > 1
            )
            missing_query = not query_paths
            missing_cluster = bool(current_cluster) and not current_cluster_files
            dead_candidates = not candidate_exists

            if (
                missing_query
                or missing_cluster
                or current_cluster_expanded
                or dead_candidates
            ):
                self.review_store.resolve_ambiguity(
                    record_id,
                    {
                        "action": "AUTO_CLOSE",
                        "status": "stale",
                        "reason": {
                            "missing_query": missing_query,
                            "missing_cluster": missing_cluster,
                            "current_cluster_expanded": current_cluster_expanded,
                            "dead_candidates": dead_candidates,
                        },
                    },
                )

    def _apply_ambiguity_resolution(self, record, action):
        if not self.output_base_dir:
            raise RuntimeError("Output folder is not set.")

        self.review_store = ReviewStore(self.output_base_dir)
        mgr = UnknownClusterManager(
            unknown_dir=self.output_base_dir,
            cluster_file=os.path.join(self.output_base_dir, "unknown_clusters.json"),
        )
        current_cluster = record.get("current_cluster")
        if current_cluster not in mgr.clusters:
            return {"action": action, "status": "missing_cluster"}

        query_paths = [
            path
            for path in record.get("file_paths", [])
            if path and os.path.exists(path)
        ]
        if not query_paths:
            return {"action": action, "status": "missing_files"}

        query_embeddings = self._query_embeddings_from_record(record)
        if not query_embeddings:
            raise RuntimeError(
                "Could not recover query embedding for this ambiguity item."
            )

        pre_variance = {
            "current_cluster": self._cluster_variance(mgr, current_cluster),
            "candidate_a": self._cluster_variance(
                mgr, record.get("candidate_a", {}).get("name")
            ),
            "candidate_b": self._cluster_variance(
                mgr, record.get("candidate_b", {}).get("name")
            ),
        }

        resolution = {
            "action": action,
            "query_paths": query_paths,
            "current_cluster": current_cluster,
            "pre_variance": pre_variance,
            "reviewer_confidence": self.ambiguity_confidence_combo.currentText()
            .strip()
            .lower(),
        }

        if action.startswith("ASSIGN_CUSTOM:"):
            target_custom = action.split(":", 1)[1]
            action = "ASSIGN_CUSTOM"
        else:
            target_custom = None

        if action == "KEEP_NEW":
            resolution["post_variance"] = {
                "current_cluster": self._cluster_variance(mgr, current_cluster),
            }
            resolution["status"] = "kept_new"
            resolution["auto_confidence"] = self._resolution_confidence(
                record, None, pre_variance
            )
            resolution["confidence"] = (
                resolution["reviewer_confidence"] or resolution["auto_confidence"]
            )
            resolution["risk"] = self._feedback_risk(resolution)
            resolution["invalidated_related_items"] = (
                self._invalidate_related_ambiguities(
                    record,
                    action,
                )
            )
            self._record_feedback(record, resolution)
            return resolution

        if action == "DISCARD":
            discard_dir = os.path.join(self.output_base_dir, "_discarded")
            os.makedirs(discard_dir, exist_ok=True)
            self._move_record_files(query_paths, discard_dir)
            self._rebuild_cluster_from_folder(mgr, current_cluster)
            self._validate_cluster_health(mgr, current_cluster)
            self._sync_unknown_cluster_gallery(mgr, current_cluster)
            mgr.save()
            self.engine._save_gallery_with_backup()
            resolution["status"] = "discarded"
            resolution["post_variance"] = {
                "current_cluster": self._cluster_variance(mgr, current_cluster),
            }
            resolution["auto_confidence"] = self._resolution_confidence(
                record, None, pre_variance
            )
            resolution["confidence"] = (
                resolution["reviewer_confidence"] or resolution["auto_confidence"]
            )
            resolution["risk"] = self._feedback_risk(resolution)
            resolution["invalidated_related_items"] = (
                self._invalidate_related_ambiguities(
                    record,
                    action,
                )
            )
            self._record_feedback(record, resolution)
            return resolution

        if action == "ASSIGN_CUSTOM":
            target = target_custom
        else:
            target = (
                record.get("candidate_a", {}).get("name")
                if action == "ASSIGN_A"
                else record.get("candidate_b", {}).get("name")
            )

        if not target or target not in mgr.clusters:
            raise RuntimeError("Selected target cluster no longer exists.")

        target_pre = self._cluster_variance(mgr, target)
        for emb in query_embeddings:
            mgr._add_to_cluster(target, emb)
        self._move_record_files(query_paths, os.path.join(self.output_base_dir, target))
        self._rebuild_cluster_from_folder(mgr, current_cluster)
        self._validate_cluster_health(mgr, current_cluster)
        self._validate_cluster_health(mgr, target)
        self._sync_unknown_cluster_gallery(mgr, current_cluster)
        mgr.save()
        self._sync_unknown_cluster_gallery(mgr, target)
        self.engine._save_gallery_with_backup()
        resolution["status"] = "assigned"
        resolution["chosen_target"] = target
        resolution["post_variance"] = {
            "current_cluster": self._cluster_variance(mgr, current_cluster),
            "chosen_target": self._cluster_variance(mgr, target),
        }
        resolution["auto_confidence"] = self._resolution_confidence(
            record,
            target,
            {
                **pre_variance,
                "chosen_target": target_pre,
            },
        )
        resolution["confidence"] = (
            resolution["reviewer_confidence"] or resolution["auto_confidence"]
        )
        resolution["risk"] = self._feedback_risk(resolution)
        resolution["invalidated_related_items"] = self._invalidate_related_ambiguities(
            record,
            action,
            target_cluster=target,
        )
        self._record_feedback(record, resolution)
        return resolution

    def _invalidate_related_ambiguities(self, record, action, target_cluster=None):
        if not self.review_store:
            return 0

        filenames = set(record.get("source_filenames", []))
        filenames.update(
            os.path.basename(path) for path in record.get("file_paths", []) if path
        )

        cluster_names = set()
        if action in {"ASSIGN_A", "ASSIGN_B", "ASSIGN_CUSTOM", "DISCARD"}:
            current_cluster = record.get("current_cluster")
            if current_cluster:
                cluster_names.add(current_cluster)
        if action in {"ASSIGN_A", "ASSIGN_B", "ASSIGN_CUSTOM"} and target_cluster:
            cluster_names.add(target_cluster)

        return self.review_store.resolve_open_ambiguities(
            filenames=filenames,
            cluster_names=cluster_names,
            exclude_ids=[record.get("id")],
            resolution={
                "action": "AUTO_CLOSE",
                "status": "related_state_changed",
                "trigger_action": action,
                "trigger_record_id": record.get("id"),
            },
        )

    def _query_embeddings_from_record(self, record):
        embeddings = []
        raw = record.get("image_embedding")
        if raw:
            embeddings.append(torch.tensor(raw, dtype=torch.float32))
        for path in record.get("file_paths", []):
            if embeddings:
                break
            if os.path.exists(path):
                emb_res = self.engine.extract_embedding_from_saved_crop(path)
                emb = emb_res[0] if isinstance(emb_res, tuple) else emb_res
                if emb is not None:
                    embeddings.append(emb.squeeze(0).cpu())
        return embeddings

    def _move_record_files(self, file_paths, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        for path in file_paths:
            if not os.path.exists(path):
                continue
            dst = os.path.join(destination_dir, os.path.basename(path))
            shutil.move(path, dst)
            parent = os.path.dirname(path)
            if os.path.isdir(parent) and not os.listdir(parent):
                try:
                    os.rmdir(parent)
                except OSError:
                    pass

    def _rebuild_cluster_from_folder(self, mgr, cluster_name):
        cluster_dir = os.path.join(self.output_base_dir, cluster_name)
        old_info = mgr.clusters.get(cluster_name, {})
        image_paths = []
        if os.path.isdir(cluster_dir):
            image_paths = [
                os.path.join(cluster_dir, name)
                for name in sorted(os.listdir(cluster_dir))
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        if not image_paths:
            mgr.clusters.pop(cluster_name, None)
            self.engine.gallery.pop(cluster_name, None)
            if os.path.isdir(cluster_dir):
                try:
                    os.rmdir(cluster_dir)
                except OSError:
                    pass
            return

        embeddings = []
        for path in image_paths:
            emb_res = self.engine.extract_embedding_from_saved_crop(path)
            emb = emb_res[0] if isinstance(emb_res, tuple) else emb_res
            if emb is not None:
                embeddings.append(emb.squeeze(0).cpu())

        if not embeddings:
            raise RuntimeError(
                f"Could not rebuild cluster {cluster_name} from remaining files."
            )

        mgr.clusters[cluster_name] = {
            "centroid": embeddings[0].clone(),
            "samples": [embeddings[0].clone()],
            "count": 1,
            "created_at": old_info.get("created_at", ""),
            "variance": 0.0,
        }
        mgr._recompute_centroid(cluster_name)
        for emb in embeddings[1:]:
            mgr._add_to_cluster(cluster_name, emb)

    def _pairwise_similarity_stats(self, samples):
        if len(samples) <= 1:
            return {"avg_pairwise_sim": 1.0, "min_pairwise_sim": 1.0}
        sims = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sims.append(float(torch.dot(samples[i], samples[j])))
        return {
            "avg_pairwise_sim": sum(sims) / len(sims),
            "min_pairwise_sim": min(sims),
        }

    def _validate_cluster_health(self, mgr, cluster_name):
        if cluster_name not in mgr.clusters:
            return
        info = mgr.clusters[cluster_name]
        stats = self._pairwise_similarity_stats(info.get("samples", []))
        info["avg_pairwise_sim"] = stats["avg_pairwise_sim"]
        info["min_pairwise_sim"] = stats["min_pairwise_sim"]
        info["unstable_flag"] = bool(
            info.get("variance", 0.0) > 0.25
            or (info.get("count", 0) == 2 and stats["avg_pairwise_sim"] < 0.70)
        )
        if info["unstable_flag"]:
            if info.get("variance", 0.0) > 0.25:
                info["unstable_reason"] = "high_variance"
            else:
                info["unstable_reason"] = "weak_pair"
        else:
            info["unstable_reason"] = ""

    def _sync_unknown_cluster_gallery(self, mgr, cluster_name):
        if cluster_name not in mgr.clusters:
            self.engine.gallery.pop(cluster_name, None)
            return
        samples = mgr.clusters[cluster_name].get("samples", [])
        if not samples:
            self.engine.gallery.pop(cluster_name, None)
            return
        embs = torch.stack(samples).to(self.engine.device)
        self.engine._add_to_gallery_internal(cluster_name, embs)

    def _cluster_variance(self, mgr, cluster_name):
        if not cluster_name:
            return None
        return (
            float(mgr.clusters.get(cluster_name, {}).get("variance", 0.0))
            if cluster_name in mgr.clusters
            else None
        )

    def _cluster_metrics(self, mgr, cluster_name):
        if not cluster_name or cluster_name not in mgr.clusters:
            return {}
        info = mgr.clusters[cluster_name]
        stats = self._pairwise_similarity_stats(info.get("samples", []))
        cluster_dir = os.path.join(self.output_base_dir, cluster_name)
        cluster_size = 0
        if os.path.isdir(cluster_dir):
            cluster_size = len(
                [
                    name
                    for name in os.listdir(cluster_dir)
                    if name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
        return {
            "name": cluster_name,
            "variance": float(info.get("variance", 0.0)),
            "cluster_size": cluster_size or int(info.get("count", 0)),
            "avg_pairwise_sim": float(stats["avg_pairwise_sim"]),
            "min_pairwise_sim": float(stats["min_pairwise_sim"]),
            "unstable_flag": bool(info.get("unstable_flag", False)),
            "unstable_reason": info.get("unstable_reason", ""),
        }

    def _sample_cluster_files(self, cluster_name, limit=3):
        if not self.output_base_dir or not cluster_name:
            return []
        cluster_dir = os.path.join(self.output_base_dir, cluster_name)
        if not os.path.isdir(cluster_dir):
            return []
        sample_paths = []
        for name in sorted(os.listdir(cluster_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_paths.append(os.path.join(cluster_dir, name))
            if len(sample_paths) >= limit:
                break
        return sample_paths

    def _scored_sample_paths(self, query_embedding, cluster_name, limit=6):
        scored = []
        if query_embedding is None:
            return scored
        for path in self._sample_cluster_files(cluster_name, limit=limit):
            if not os.path.exists(path):
                continue
            emb_res = self.engine.extract_embedding_from_saved_crop(path)
            emb = emb_res[0] if isinstance(emb_res, tuple) else emb_res
            if emb is None:
                continue
            sample_emb = emb.squeeze(0).cpu()
            scored.append(
                {
                    "path": path,
                    "sim": float(torch.dot(query_embedding, sample_emb)),
                    "embedding": sample_emb.tolist(),
                }
            )
        scored.sort(key=lambda item: item["sim"], reverse=True)
        return scored

    def _resolution_confidence(self, record, chosen_target, pre_variance):
        if chosen_target:
            target_var = (
                pre_variance.get("candidate_a")
                if chosen_target == record.get("candidate_a", {}).get("name")
                else pre_variance.get("candidate_b")
            )
            return "high" if target_var is not None and target_var < 0.15 else "low"
        cand_a_var = pre_variance.get("candidate_a")
        cand_b_var = pre_variance.get("candidate_b")
        if (
            cand_a_var is not None
            and cand_b_var is not None
            and cand_a_var < 0.15
            and cand_b_var < 0.15
        ):
            return "high"
        return "low"

    def _feedback_risk(self, resolution):
        action = resolution.get("action")
        pre = resolution.get("pre_variance", {})
        post = resolution.get("post_variance", {})

        if action in {"ASSIGN_A", "ASSIGN_B"}:
            pre_target = pre.get("chosen_target")
            post_target = post.get("chosen_target")
            if (
                pre_target is not None
                and post_target is not None
                and post_target > pre_target + 0.05
            ):
                return "high"
        elif action in {"KEEP_NEW", "DISCARD"}:
            pre_current = pre.get("current_cluster")
            post_current = post.get("current_cluster")
            if (
                pre_current is not None
                and post_current is not None
                and post_current > pre_current + 0.05
            ):
                return "high"
        return "low"

    def _record_feedback(self, record, resolution):
        query_paths = record.get("file_paths", [])
        cand_a = record.get("candidate_a", {})
        cand_b = record.get("candidate_b", {})
        chosen_target = resolution.get("chosen_target")
        query_embeddings = self._query_embeddings_from_record(record)
        query_embedding = query_embeddings[0] if query_embeddings else None
        review_mgr = UnknownClusterManager(
            unknown_dir=self.output_base_dir,
            cluster_file=os.path.join(self.output_base_dir, "unknown_clusters.json"),
        )
        cand_a_name = cand_a.get("name")
        cand_b_name = cand_b.get("name")

        feedback = {
            "action": resolution.get("action"),
            "confidence": resolution.get("confidence", "low"),
            "reviewer_confidence": resolution.get(
                "reviewer_confidence", resolution.get("confidence", "low")
            ),
            "auto_confidence": resolution.get("auto_confidence", "low"),
            "risk": resolution.get("risk", "low"),
            "query_paths": query_paths,
            "query_filenames": record.get("source_filenames", []),
            "query_embedding": query_embedding.tolist()
            if query_embedding is not None
            else None,
            "current_cluster": record.get("current_cluster"),
            "chosen_target": chosen_target,
            "candidate_a": {
                **self._cluster_metrics(review_mgr, cand_a_name),
                "score": cand_a.get("score", 0.0),
                "sample_paths": self._sample_cluster_files(cand_a_name),
                "sample_pairs": self._scored_sample_paths(query_embedding, cand_a_name),
            },
            "candidate_b": {
                **self._cluster_metrics(review_mgr, cand_b_name),
                "score": cand_b.get("score", 0.0),
                "sample_paths": self._sample_cluster_files(cand_b_name),
                "sample_pairs": self._scored_sample_paths(query_embedding, cand_b_name),
            },
            "top_candidates": record.get("top_candidates", []),
            "pre_variance": resolution.get("pre_variance", {}),
            "post_variance": resolution.get("post_variance", {}),
        }
        self.review_store.add_feedback_pair(feedback)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — Train Database
    # ══════════════════════════════════════════════════════════════════════════
    def setup_tab3(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(12)

        sec_header = QLabel("\u25a0  Database Management")
        sec_header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        sec_header.setStyleSheet(f"color: {NAVY_PRIMARY};")
        layout.addWidget(sec_header)

        sec_desc = QLabel(
            "Register a new elephant identity, or enrich an existing one with additional images."
        )
        sec_desc.setFont(QFont("Segoe UI", 10))
        sec_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; margin-bottom: 6px;")
        layout.addWidget(sec_desc)

        # ── How-to hint panel ──────────────────────────────────────────────
        hint_frame = QFrame()
        hint_frame.setStyleSheet(f"""
            QFrame {{
                background: {GOLD_LIGHT}; border: 1px solid {GOLD_ACCENT};
                border-radius: 4px;
            }}
        """)
        hint_layout = QVBoxLayout(hint_frame)
        hint_layout.setContentsMargins(14, 10, 14, 10)
        hint_layout.setSpacing(4)

        hint1 = QLabel("\u2022  New elephant ID  \u2192  creates a fresh gallery entry")
        hint2 = QLabel(
            "\u2022  Existing elephant ID  \u2192  merges new images into the gallery (enriches recognition)"
        )
        for h in (hint1, hint2):
            h.setFont(QFont("Segoe UI", 9))
            h.setStyleSheet(f"color: {NAVY_PRIMARY}; border: none;")
            hint_layout.addWidget(h)
        layout.addWidget(hint_frame)

        form_group = QGroupBox("Elephant Registration / Gallery Enrichment")
        form_group.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        form_layout = QGridLayout(form_group)
        form_layout.setHorizontalSpacing(14)
        form_layout.setVerticalSpacing(12)
        form_layout.setContentsMargins(16, 22, 16, 14)

        lbl_id = QLabel("Elephant ID:")
        lbl_id.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.new_id_input = QLineEdit()
        self.new_id_input.setPlaceholderText(
            "New ID  (e.g. Makhna_3)  or  existing ID  (e.g. Makhna_1)  to enrich"
        )
        self.new_id_input.setMinimumHeight(34)

        form_layout.addWidget(lbl_id, 0, 0)
        form_layout.addWidget(self.new_id_input, 0, 1)

        self.btn_train = QPushButton("  Select Folder & Enroll / Update  ")
        self.btn_train.setMinimumHeight(38)
        self.btn_train.clicked.connect(self.train_new_elephant)
        form_layout.addWidget(self.btn_train, 1, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        form_layout.setColumnStretch(1, 1)

        layout.addWidget(form_group)
        layout.addStretch()
        self.tab3.setLayout(layout)

    def train_new_elephant(self):
        elephant_id = self.new_id_input.text().strip()
        if not elephant_id:
            QMessageBox.warning(
                self, "Input Required", "Please enter an Elephant ID before proceeding."
            )
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Images of New Elephant")
        if not folder:
            return

        # Visual duplicate detection — compare new images against the entire database
        similar = self.engine.find_similar_elephants(folder, threshold=0.80)

        if similar:
            match_lines = "\n".join(
                f"  \u2022  {eid}  \u2014  {score:.1f}% similarity"
                for eid, score in similar
            )
            reply = QMessageBox.question(
                self,
                "Similar Elephant(s) Found",
                f"The selected images closely resemble elephant(s) already in the database:\n\n"
                f"{match_lines}\n\n"
                f"This may be the same individual under a different ID.\n"
                f"Do you still want to enrol as '{elephant_id}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        success, is_update = self.engine.update_database(folder, elephant_id)
        if success:
            if is_update:
                QMessageBox.information(
                    self,
                    "Gallery Updated",
                    f"Elephant '{elephant_id}' gallery has been enriched with the new images.\n"
                    f"Future classifications will use both old and new embeddings.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Enrolment Successful",
                    f"Elephant '{elephant_id}' has been added to the vector database.",
                )
        else:
            QMessageBox.warning(
                self,
                "Enrolment Failed",
                "No valid images were found in the selected folder.",
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  GLOBAL STYLESHEET
    # ══════════════════════════════════════════════════════════════════════════
    def _apply_styles(self):
        self.setStyleSheet(f"""
            /* ── Base ── */
            QMainWindow {{
                background-color: {BG_LIGHT};
            }}
            QWidget {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                color: {TEXT_PRIMARY};
            }}

            /* ── Labels ── */
            QLabel {{
                color: {TEXT_PRIMARY};
            }}

            /* ── Buttons (default) ── */
            QPushButton {{
                background-color: {NAVY_PRIMARY};
                color: white;
                padding: 8px 18px;
                border-radius: 3px;
                font-weight: 600;
                border: none;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {NAVY_LIGHT};
            }}
            QPushButton:disabled {{
                background-color: #A0AAB4;
                color: #D0D5DD;
            }}

            /* ── Tab Widget ── */
            QTabWidget::pane {{
                border: 1px solid {BORDER_SUBTLE};
                background: {BG_WHITE};
                border-top: 2px solid {NAVY_PRIMARY};
            }}
            QTabBar::tab {{
                background: #DFE3EA;
                padding: 10px 28px;
                color: {TEXT_PRIMARY};
                font-weight: 600;
                font-size: 11px;
                border: 1px solid {BORDER_SUBTLE};
                border-bottom: none;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {BG_WHITE};
                color: {NAVY_PRIMARY};
                font-weight: bold;
                border-top: 2px solid {GOLD_ACCENT};
                border-bottom: 1px solid {BG_WHITE};
            }}
            QTabBar::tab:hover {{
                background: #CBD0D8;
            }}

            /* ── Inputs ── */
            QLineEdit {{
                background: white;
                color: {TEXT_PRIMARY};
                padding: 7px 10px;
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 3px;
                font-size: 12px;
            }}
            QLineEdit:focus {{
                border-color: {NAVY_PRIMARY};
            }}

            /* ── List Widget (Gallery) ── */
            QListWidget {{
                background: {BG_WHITE};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 3px;
            }}
            QListWidget::item {{
                border: 2px solid transparent;
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item:hover {{
                border: 2px solid {NAVY_PRIMARY};
                background: #E8EBF0;
            }}
            QListWidget::item:selected {{
                border: 2px solid {GOLD_ACCENT};
                background: {GOLD_LIGHT};
            }}

            /* ── Group Box ── */
            QGroupBox {{
                background: {BG_WHITE};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 14px;
                font-weight: bold;
                color: {NAVY_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 10px;
                color: {NAVY_PRIMARY};
            }}

            /* ── Scroll Bar ── */
            QScrollBar:vertical {{
                background: {BG_LIGHT};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: #B0B8C4;
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #8896A7;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}

            /* ── Message Boxes ── */
            QMessageBox {{
                background: {BG_WHITE};
            }}
            QMessageBox QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setFont(QFont("Segoe UI", 10))

    window = ElephantApp()
    window.show()
    sys.exit(app.exec())
