import sys
import skimage.io
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.uic import loadUi
import cv2
from PyQt5.QtWidgets import QMessageBox
from _UI import Ui_MainWindow
import os
from test import detection
from test import violation_building_detection


class mywin(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywin, self).__init__()
        self.setupUi(self)
        self.new_path = ""
        self.file_path = ""
        self.pushButton.clicked.connect(self.image_detection)
        self.actionfileOpen.triggered.connect(self.open_file)

    def showimage(self, path, image_num):
        pixmap = QPixmap(path)
        if image_num == 1:
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)
        elif image_num == 2:
            self.label_2.setPixmap(pixmap)
            self.label_2.setScaledContents(True)

    def open_file(self):
        # 打开新图片
        self.file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                          "All Files(*);;Text Files(*.txt)")
        # 删除原图
        self.label_2.setPixmap(QPixmap(""))
        # 的到文件名 + 后缀
        split_file_path = os.path.split(self.file_path)
        self.new_path = split_file_path[0] + '/../new/' + split_file_path[1]
        # 显示原图片
        self.showimage(self.file_path, 1)

    def image_detection(self):
        pre_image_path, image_path, violation_building_nums = violation_building_detection(self.file_path, self.new_path)
        self.showimage(pre_image_path, 1)
        self.showimage(image_path, 2)
        if violation_building_nums == 0:
            QMessageBox.information(self, '提示', '检测完成, 未发现违章建筑', QMessageBox.Ok)
        else:
            QMessageBox.information(self, '提示', '检测完成, 发现{}个违章建筑'.format(violation_building_nums), QMessageBox.Ok)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mywin()
    window.show()
    window.setWindowTitle("window")
    sys.exit(app.exec_())
