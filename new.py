from new_ui import *
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QFont, QPixmap, QIcon, QTransform, QImage, QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import (QDockWidget, QApplication, QMainWindow, QGraphicsDropShadowEffect, QDockWidget, QApplication, QMainWindow, QAction, QStatusBar, QFileDialog, QScrollArea, QDoubleSpinBox, QRadioButton, QFrame,
                             QMessageBox, QPushButton, QButtonGroup, QStackedWidget, QFormLayout, QComboBox, QAbstractSpinBox, QHBoxLayout, QGroupBox,   
                            QTextEdit, QToolBar, QGridLayout, QVBoxLayout, QLabel, QWidget, QDesktopWidget, QSpinBox, QCheckBox)
import sys, os, cv2
from PIL import Image, ImageDraw
from superqt import QLabeledSlider, QLabeledRangeSlider, QLabeledDoubleRangeSlider, QLabeledDoubleSlider

class MainWindow_UI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setting()
        self.controlStack()
        self.buttonControl()
        self.show()

    def setting(self):
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(50)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0,92,157,550))
        self.ui.centralwidget.setGraphicsEffect(self.shadow)

        self.ui.display_stack.setCurrentIndex(0)

    def hideMenu(self):
        width = self.ui.menu.width()

        if width == 0:
            newWidth = 250
            self.ui.menu.setMinimumWidth(newWidth)
            self.ui.menu.setMaximumWidth(newWidth)
        else:
            newWidth = 0
            self.ui.menu.setMinimumWidth(newWidth)
            self.ui.menu.setMaximumWidth(newWidth)

    def buttonControl(self):
        self.ui.pushButton_4.clicked.connect(self.close)
        self.ui.pushButton_2.clicked.connect(self.showMinimized)
        self.ui.hide.clicked.connect(self.hideMenu)
        self.ui.toolButton.clicked.connect(self.openImage)
        self.ui.save_img.clicked.connect(self.saveImage)
        self.ui.upload_folder.clicked.connect(self.openFolder)
        self.ui.save_folder.clicked.connect(self.saveFolder)
        self.ui.upload_mask.clicked.connect(self.montage)

    def controlStack(self):
        self.ui.ac.clicked.connect(lambda: self.testLambda(1))
        self.ui.asg.clicked.connect(lambda: self.testLambda(2))
        self.ui.ao.clicked.connect(lambda: self.testLambda(3))

    def testLambda(self, index):
        if index == 1:
            self.ui.state_label.setText("Augmentation for Classification")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
        elif index == 2:
            self.ui.state_label.setText("Augmentation for Segmentation")
            self.ui.display_stack.setCurrentIndex(1)
            self.ui.transform_stack.setCurrentIndex(1)
        elif index == 3:
            self.ui.state_label.setText("Augmentation for Object Detection")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)

    def montage(self):
        im = self.multi_image_viewer(dir=self.rootdir, col=4, row=4)
        pixmap = self.pil2pixmap(im)
        self.ui.label_8.setPixmap(QPixmap(pixmap))

    def openImage(self):
        self.image_file, _ = QFileDialog.getOpenFileName(self, "Open Image", "","")

        if self.image_file:
            self.root_dir = os.path.dirname(self.image_file)
            pixmap = QPixmap(self.image_file)
            self.ui.label_6.setPixmap(pixmap)
        else:
            QMessageBox.information(self, 'Error', 'Unable to open image', QMessageBox.Ok)

    def saveImage(self):
        method_map = {
                "Rotate": "rotated",
                "Crop": "Rcropped",
                "Manual Crop": "Mcropped",
                "Resize": "resized",
                "Horizontal Flip": "hflipped",
                "Vertical Flip": "vflipped",
                "Random Scale": "scaled",
                "Blur": "zoom_blurred",
                "Hue and Saturation": "hue_saturation",
                "Random Brightness Contrast": "brightness_contrast",
                "Solarize": "solarized",
                "Random Gamma": "gamma_adjusted",
                "Noise": "noisy",
                "Channel Shuffle": "shuffled",
                "Coarse Dropout": "dropout"
            }
        frame = cv2.imread(self.image_file)
        frame = self.trans(frame)
        
        method = method_map.get(self.ui.comboBox.currentText(), "NotFound")
        custom_filename = f"{os.path.basename(self.image_file).split('.')[0]}_{method}.jpg"
        cv2.imwrite(os.path.join(self.root_dir, custom_filename), frame)
       
    def openFolder(self):
        self.rootdir = QFileDialog.getExistingDirectory(self, caption='Select a folder')
        if self.rootdir:
            pass
        else:
            QMessageBox.information(self, 'Error', 'Unable to open folder', QMessageBox.Ok)

    def saveFolder(self):
        self.widget = QWidget()
        self.widget.setWindowTitle('Save Folder')
        vbox = QVBoxLayout()
        text = QLabel('How many percentages of the dataset you want to augument?')
        vbox.addWidget(text)
        self.slider = QLabeledSlider()
        self.slider.setRange(1,100)
        vbox.addWidget(self.slider)
        button = QPushButton('Transform the folder')
        vbox.addWidget(button)
        self.widget.setLayout(vbox)
        self.widget.show()
        # button.clicked.connect(self.transform)

    def multi_image_viewer(self, dir: str, col: int, row: int):
    # Get a list of all image files in the folder
        image_files = [f for f in os.listdir(dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Determine the size of the grid image based on the individual image sizes
        grid_width = max(300, min(col * 200, 800))  
        grid_height = 200 * row
        grid_image = Image.new('RGB', (grid_width, grid_height))

        x = 0
        y = 0

        for img in image_files:
            img_path = os.path.join(dir, img)
            current_image = Image.open(img_path)
            current_image.thumbnail((200, 200))  # Resize the image if needed
            grid_image.paste(current_image, (x, y))
            x += 200
            if x >= grid_width:
                x = 0
                y += 200

        return grid_image

    def responsive(self):
        pass

    def apply_changes(self):
        pass

    def updateImageShape(self):
        # if self.state == 0:
        #     image = cv2.imread(self.image_file)
        #     height, width, channels = image.shape
        #     self.widthBar.setRange(0,width)
        #     self.heightBar.setRange(0,height)
        #     self.McropWidth.setRange(0, width)
        #     self.McropHeight.setRange(0,height)
        #     self.widthR.setRange(0,width)
        #     self.heightR.setRange(0,height)
        pass

    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif  im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        # Bild in RGBA konvertieren, falls nicht bereits passiert
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)
        return pixmap

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow_UI()
    sys.exit(app.exec_())