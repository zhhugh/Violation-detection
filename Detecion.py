import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.uic import loadUi
import cv2
from PyQt5.QtWidgets import QMessageBox
from _UI import Ui_MainWindow
import os
from test import detection


class mywin(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywin, self).__init__()
        self.setupUi(self)
        self.gen_path = ""
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
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                     "All Files(*);;Text Files(*.txt)")
        # 删除原图
        self.label_2.setPixmap(QPixmap(""))
        self.gen_path = file_path
        # 识别原图片
        pre_image_path = detection(file_path, 1)
        print(pre_image_path)
        self.showimage(pre_image_path, 1)
        QMessageBox.information(self, '提示', '图片加载成功', QMessageBox.Ok)

    def image_detection(self):
        image_path = detection(self.gen_path, 2)
        self.showimage(image_path, 2)
        QMessageBox.information(self, '提示', '检测完成', QMessageBox.Ok)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mywin()
    window.show()
    window.setWindowTitle("window")
    sys.exit(app.exec_())
