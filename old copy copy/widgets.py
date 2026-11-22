from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView, QMenu, QHeaderView
from PySide6.QtCore import Qt, Signal
from utils import open_file_location

class AudioTableWidget(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Filename", "Duration", "Status"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)

    def show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        menu.addAction("Remove Selected").triggered.connect(self.remove_selected_items)
        menu.addAction("Open Containing Folder").triggered.connect(lambda: open_file_location(self.item(item.row(), 0).data(Qt.UserRole)))
        menu.exec_(self.mapToGlobal(pos))

    def remove_selected_items(self):
        for row in sorted({i.row() for i in self.selectedItems()}, reverse=True): self.removeRow(row)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls()]
            self.files_dropped.emit(paths)
            e.acceptProposedAction()