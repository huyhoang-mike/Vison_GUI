from base_ui import *
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QFont, QPixmap, QIcon, QTransform, QImage, QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import (QDockWidget, QApplication, QMainWindow, QGraphicsDropShadowEffect, QDockWidget, QApplication, QMainWindow, QAction, QStatusBar, QFileDialog, QScrollArea, QDoubleSpinBox, QRadioButton, QFrame,
                             QMessageBox, QPushButton, QButtonGroup, QStackedWidget, QFormLayout, QComboBox, QAbstractSpinBox, QHBoxLayout, QGroupBox,   
                            QTextEdit, QToolBar, QGridLayout, QVBoxLayout, QLabel, QWidget, QDesktopWidget, QSpinBox, QCheckBox)
import sys, os, cv2
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from algorithm import Algorithm
from superqt import QLabeledSlider, QLabeledRangeSlider, QLabeledDoubleRangeSlider, QLabeledDoubleSlider


class MainWindow_UI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.algorithms = Algorithm()
        self.mode = 1
        self.view = 1
        self.setup()
        self.controlMainStack()
        self.buttonControl()
        self.show()

    def setup(self):
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(50)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0,92,157,550))
        self.ui.centralwidget.setGraphicsEffect(self.shadow)

        self.ui.param_stack.setCurrentIndex(1)
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
        self.ui.close.clicked.connect(self.close)
        self.ui.minimize.clicked.connect(self.showMinimized)
        self.ui.hide.clicked.connect(self.hideMenu)
        self.ui.upload_img.clicked.connect(self.openImage)
        self.ui.save_img.clicked.connect(self.saveImage)
        self.ui.upload_folder.clicked.connect(self.openFolder)
        self.ui.save_folder.clicked.connect(self.saveFolder)
        self.ui.apply_changes.clicked.connect(self.apply_changes)
        self.ui.setting.clicked.connect(self.setting)

        self.ui.single_view.clicked.connect(lambda: self.viewHandle(1))
        self.ui.multiple_view.clicked.connect(lambda: self.viewHandle(2))

        self.ui.max_factor.valueChanged.connect(self.responsive)
        self.ui.step_factor.valueChanged.connect(self.responsive)
        self.ui.var_limit.valueChanged.connect(self.responsive)
        self.ui.mean_value.valueChanged.connect(self.responsive)
        self.ui.rotate_limit.valueChanged.connect(self.responsive)
        self.ui.solarize_value.valueChanged.connect(self.responsive)

    def controlMainStack(self):
        self.ui.ac.clicked.connect(lambda: self.testLambda(1))
        self.ui.asg.clicked.connect(lambda: self.testLambda(2))
        self.ui.ao.clicked.connect(lambda: self.testLambda(3))
        self.ui.pc.clicked.connect(lambda: self.testLambda(4))
        self.ui.ps.clicked.connect(lambda: self.testLambda(5))
        self.ui.po.clicked.connect(lambda: self.testLambda(6))
        self.ui.mlc.clicked.connect(lambda: self.testLambda(7))
        self.ui.mls.clicked.connect(lambda: self.testLambda(8))
        self.ui.mlo.clicked.connect(lambda: self.testLambda(9))
        self.ui.comboBox.currentIndexChanged.connect(self.controlSubStack)

    def controlSubStack(self):
        options_mapping = {
        "Crop": 0,
        "Rotate": 1,
        "Vertical Flip": 2,
        "Noise": 3,
        "Solarize": 4,
        "Blur": 5,
        "Resize": 6,
        "Horizontal Flip": 7,    
        }
        current_text = self.ui.comboBox.currentText()
        if current_text in options_mapping:
            self.ui.param_stack.setCurrentIndex(options_mapping[current_text])

    def testLambda(self, index):
        if index == 1:
            self.ui.state_label.setText("Augmentation for Classification")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 1
        elif index == 2:
            self.ui.state_label.setText("Augmentation for Segmentation")
            self.ui.display_stack.setCurrentIndex(1)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 2
        elif index == 3:
            self.ui.state_label.setText("Augmentation for Object Detection")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 3
        elif index == 4:
            self.ui.state_label.setText("Preprpcessing for Classification")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 4
        elif index == 5:
            self.ui.state_label.setText("Preprpcessing for Segmentation")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 5
        elif index == 6:
            self.ui.state_label.setText("Preprpcessing for Object Detection")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 6
        elif index == 7:
            self.ui.state_label.setText("Machine Learning for Classification")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 7
        elif index == 8:
            self.ui.state_label.setText("Machine Learning for Segmentation")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 8
        elif index == 9:
            self.ui.state_label.setText("Machine Learning for Object Detection")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 9

    def viewHandle(self, index):

        if self.mode == 1:
            if index == 1:
                self.ui.img_original_oc.setPixmap(self.pixmap_from_cv_image(self.base_img))
                self.ui.img_result_oc.setPixmap(self.pixmap_from_cv_image(self.base_img))
            elif index == 2:
                im_original = self.multi_image_viewer(dir=self.rootdir, col=3, row=2)
                pixmap = self.pil2pixmap(im_original)
                self.ui.img_original_oc.setPixmap(QPixmap(pixmap))

                im_result = self.multi_image_viewer_transform(dir=self.rootdir, col=3, row=2)
                pixmap = self.pil2pixmap(im_result)
                self.ui.img_result_oc.setPixmap(QPixmap(pixmap))

    def setting(self):
        self.widget = QWidget()
        self.layoutt = QVBoxLayout()
        #......................
        self.row = QSpinBox()
        self.row.setFixedWidth(200)
        self.row.setRange(1,10)
        self.row.setValue(3)
        self.column = QSpinBox()
        self.column.setRange(1,10)
        self.column.setValue(3)
        self.button = QPushButton('View')
        self.button.clicked.connect(self.widget.close)

        self.layoutt.addWidget(QLabel('Row'))
        self.layoutt.addWidget(self.row)
        self.layoutt.addWidget(QLabel('Column'))
        self.layoutt.addWidget(self.column)
        self.layoutt.addWidget(self.button)
        self.widget.setLayout(self.layoutt)

        self.widget.show()

    def openImage(self):
        self.image_file, _ = QFileDialog.getOpenFileName(self, "Open Image", "","")

        if self.image_file:
            self.root_dir = os.path.dirname(self.image_file)
            pixmap = QPixmap(self.image_file)
            self.ui.img_original_oc.setPixmap(pixmap)
            self.base_img = cv2.imread(self.image_file)
        else:
            QMessageBox.information(self, 'Error', 'Unable to open image', QMessageBox.Ok)

    def saveImage(self):
        method_map = {
                "Rotate": "rotated",
                "Crop": "Rcropped",
                "Crop": "Mcropped",
                "Resize": "resized",
                "Horizontal Flip": "hflipped",
                "Vertical Flip": "vflipped",
                "Random Scale": "scaled",
                "Blur": "blurred",
                "Hue and Saturation": "hue_saturation",
                "Random Brightness Contrast": "brightness_contrast",
                "Solarize": "solarized",
                "Random Gamma": "gamma_adjusted",
                "Noise": "noisy",
                "Channel Shuffle": "shuffled",
                "Coarse Dropout": "dropout"
            }
        
        method = method_map.get(self.ui.comboBox.currentText(), "NotFound")
        custom_filename = f"{os.path.basename(self.image_file).split('.')[0]}_{method}.jpg"
        cv2.imwrite(os.path.join(self.root_dir, custom_filename), self.base_img)
       
    def openFolder(self):
        self.rootdir = QFileDialog.getExistingDirectory(self, caption='Select a folder')
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                self.image_file = os.path.join(subdir, file)
                break
            break 
        
        if self.rootdir:
            pixmap = QPixmap(self.image_file)
            self.ui.img_original_oc.setPixmap(pixmap)
            self.base_img = cv2.imread(self.image_file)
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

    def multi_image_viewer_transform(self, dir: str, col: int, row: int):
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
            current_image = np.array(current_image)
            current_image = self.transformation(current_image)
            current_image = Image.fromarray(current_image)
            grid_image.paste(current_image, (x, y))
            x += 200
            if x >= grid_width:
                x = 0
                y += 200

        return grid_image

    def responsive(self):
        img = self.transformation(self.base_img)
        self.pixmap = self.pixmap_from_cv_image(img)
        self.ui.img_result_oc.setPixmap(self.pixmap)

    def apply_changes(self):
        self.base_img = self.image_tranformed
        self.ui.img_original_oc.setPixmap(self.pixmap_from_cv_image(self.base_img))

    def transformation(self, image_data):
        if self.mode == 1 or self.mode == 2:
            if self.ui.comboBox.currentText() == "Rotate":
                self.transforms = self.algorithms.rotate(limit=self.ui.rotate_limit.sliderPosition(), border_mode=1)
            elif self.ui.comboBox.currentText() == "Crop":
                self.transforms = self.algorithms.Mcrop(xmin=self.ui.width.sliderPosition()[0], xmax=self.ui.width.sliderPosition()[1],
                                                ymin=self.ui.height.sliderPosition()[0], ymax=self.ui.height.sliderPosition()[1]) 
            elif self.ui.comboBox.currentText() == "Horizontal Flip":
                self.transforms = self.algorithms.hflip()
            elif self.ui.comboBox.currentText() == "Vertical Flip":
                self.transforms = self.algorithms.vflip()
            elif self.ui.comboBox.currentText() == "Resize":
                self.transforms = self.algorithms.resize(width=self.ui.width_resize.value(), height=self.ui.height_resize.value())
            elif self.ui.comboBox.currentText() == "Blur":                   
                self.transforms = self.algorithms.ZoomBlur(max_factor=self.ui.max_factor.value(), step_factor=self.ui.step_factor.sliderPosition())
            elif self.ui.comboBox.currentText() == "Solarize":                   
                self.transforms = self.algorithms.Solarize(threshold=self.ui.solarize_value.value())   
            elif self.ui.comboBox.currentText() == "Noise":                   
                self.transforms = self.algorithms.GaussNoise(var_limit=self.ui.var_limit.value(), mean=self.ui.mean_value.value()) 

        self.image_tranformed = self.transforms(image=image_data)["image"]

        return self.image_tranformed

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

    def pixmap_from_cv_image(self, cv_image):   # check theory
        height, width, _ = cv_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
        return QPixmap(qImg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow_UI()
    sys.exit(app.exec_())