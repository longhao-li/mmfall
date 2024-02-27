from PyQt5.QtWidgets import QWidget, QLabel, QGroupBox, QVBoxLayout, QComboBox, QFormLayout, QGridLayout
from typing import List
from pyqtgraph import glColor
from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLScatterPlotItem, GLMeshItem
import numpy as np


class PointCloudWidget(GLViewWidget):
    def __init__(self, parent = None) -> None:
        super(PointCloudWidget, self).__init__(parent)
        self.setBackgroundColor(70, 72, 79)

        self.grid    = GLGridItem()
        self.scatter = GLScatterPlotItem(size=5)

        evm_size = np.array([0.0625, 0, 0.125])
        vertices = np.empty((2, 3, 3))
        vertices[0, 0, :] = [-evm_size[0], evm_size[1], +evm_size[2]]
        vertices[0, 1, :] = [-evm_size[0], evm_size[1], -evm_size[2]]
        vertices[0, 2, :] = [+evm_size[0], evm_size[1], -evm_size[2]]
        vertices[1, 0, :] = [-evm_size[0], evm_size[1], +evm_size[2]]
        vertices[1, 1, :] = [+evm_size[0], evm_size[1], +evm_size[2]]
        vertices[1, 2, :] = [+evm_size[0], evm_size[1], -evm_size[2]]

        self.evm_box = GLMeshItem(
            vertices  = vertices,
            smooth    = False,
            drawEdges = True,
            edgeColor = glColor("r"),
            drawFaces = False
        )

        self.addItem(self.grid)
        self.addItem(self.scatter)
        self.addItem(self.evm_box)


class StatisticsGroupBox(QGroupBox):
    def __init__(self, parent = None) -> None:
        super(StatisticsGroupBox, self).__init__("Statistics", parent)
        self.frame_count_label = QLabel("Frame: 0")
        self.point_count_label = QLabel("Points: 0")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.frame_count_label)
        self.layout.addWidget(self.point_count_label)
        self.setLayout(self.layout)


class PlotControlPanel(QGroupBox):
    def __init__(self, parent = None) -> None:
        super(PlotControlPanel, self).__init__("Plot Control", parent)

        self.point_color_mode = QComboBox()
        self.point_color_mode.addItems(["SNR", "Height", "Doppler", "Track"])

        self.layout = QFormLayout()
        self.layout.addRow("Color Point Mode:", self.point_color_mode)
        self.setLayout(self.layout)


class MainWindow(QWidget):
    def __init__(self, data: List[np.ndarray], parent = None) -> None:
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("mmFall Pointcloud Virtualizer")

        # Create the pointcloud widget. Use PyQTGraph to render the pointcloud.
        self.pointcloud_widget = PointCloudWidget()

        # Create the statistics box.
        self.statistics_box = StatisticsGroupBox()

        # Create plot control panel.
        self.plot_control_panel = PlotControlPanel()

        # Layout for the main window.
        self.layout = QGridLayout()
        self.layout.addWidget(self.statistics_box, 0, 0, 1, 1)
        self.layout.addWidget(self.pointcloud_widget, 1, 0, 1, 1)
        self.layout.setRowStretch(2, 1)
        self.layout.addWidget(self.plot_control_panel, 0, 1, 3, 1)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 3)

        self.setLayout(self.layout)

        self.data           = data
        self.frame_interval = 100   # ms
        self.current_frame  = 0

