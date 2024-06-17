from frontend.main_ui import *
from PyQt5.QtCore import Qt, QSize, QRect, QThread, QThreadPool
from PyQt5.QtGui import QFont, QPixmap, QIcon, QTransform, QImage, QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import (QDockWidget, QApplication, QMainWindow, QGraphicsDropShadowEffect, QDockWidget, QApplication, QMainWindow, QAction, QStatusBar, QFileDialog, QScrollArea, QDoubleSpinBox, QRadioButton, QFrame,
                             QMessageBox, QPushButton, QButtonGroup, QStackedWidget, QFormLayout, QComboBox, QAbstractSpinBox, QHBoxLayout, QGroupBox,   
                            QTextEdit, QToolBar, QGridLayout, QVBoxLayout, QLabel, QWidget, QDesktopWidget, QSpinBox, QTextBrowser)
import sys, os, cv2, json
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import albumentations as A
from algorithm import Algorithm
from test.train import Trainer, Worker
from test.val import Validation
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Assuming this is your JSON configuration as a string
json_config = """
{
    "Augmentation": {
        "Rotate": {"param": "angle", "value": 30},
        "Blur": {"param": "kernel_size", "value": 5},
        "Solarize": {"param": "threshold", "value": 128},
        "Horizontal Flip": {"param": "probability", "value": 0.5},
        "Vertical Flip": {"param": "probability", "value": 0.5},
        "Crop": {"param": {"width": 100, "height": 100}}
    },
    "Preprocessing": {
        "Decay log": {"param": "decay_rate", "value": 0.01},
        "Scale by square root": {"param": "scale_factor", "value": 2.0},
        "Log": {"param": "base", "value": 10},
        "Normalization": {"param": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
        "Absolute": {"param": "none"}
    }
}
"""

class MainWindow_UI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.algorithms = Algorithm()

        ###
        self.thread_pool = QThreadPool()
        self.validateClass = Validation()
        # Set up the graph in the UI
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.ui.graphWidget.setLayout(layout)
        self.running_losses = []

        self.mode = 1
        self.view = 1
        self.base_img = ''
        self.image_file = ''
        self.rootdir = ''
        self.percentage = 100
        self.setup()
        self.controlMainStack()
        self.buttonControl()

        self._mousePressPos = None
        self._windowPos = None

        self.show()

    def update_training_progress(self, message):
        self.ui.progress_label_training.setWordWrap(True)
        self.ui.progress_label_training.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        current_text = self.ui.progress_label_training.text()
        new_text = current_text + message + "\n"  # Add a newline at the end
        self.ui.progress_label_training.setText(new_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.ui.toolbar.underMouse():
            self._mousePressPos = event.globalPos()
            self._windowPos = self.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._mousePressPos:
            self.move(self._windowPos + (event.globalPos() - self._mousePressPos))

    def mouseReleaseEvent(self, event):
        self._mousePressPos = None
        self._windowPos = None

    def setup(self):
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(50)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0,92,157,550))
        self.ui.centralwidget.setGraphicsEffect(self.shadow)

        self.ui.transform_stack.setCurrentIndex(1)
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
        # self.ui.apply_changes.clicked.connect(self.apply_changes)
        self.ui.apply_changes.clicked.connect(self.add)
        self.ui.upload_mask.clicked.connect(self.openMask)
        self.ui.upload_bbox.clicked.connect(self.openBBox)
        self.ui.toolButton_6.clicked.connect(self.saveBBox)

        self.ui.setting.clicked.connect(self.setting)
        self.ui.help.clicked.connect(self.helpSheet)

        self.ui.single_view.clicked.connect(lambda: self.viewHandle(1))
        self.ui.multiple_view.clicked.connect(lambda: self.viewHandle(2))

        self.ui.blur_limit.valueChanged.connect(self.responsive)
        self.ui.var_limit.valueChanged.connect(self.responsive)
        self.ui.mean_value.valueChanged.connect(self.responsive)
        self.ui.rotate_limit.valueChanged.connect(self.responsive)
        self.ui.solarize_value.valueChanged.connect(self.responsive)

        self.ui.mean_shift.valueChanged.connect(self.responsive)
        self.ui.scale_limit.valueChanged.connect(self.responsive)
        self.ui.scale_factor.valueChanged.connect(self.responsive)
        self.ui.gamma_factor.valueChanged.connect(self.responsive)
        self.ui.decay_factor.valueChanged.connect(self.responsive)
        self.ui.alpha_value.valueChanged.connect(self.responsive)
        self.ui.beta_value.valueChanged.connect(self.responsive)

        self.ui.width_resize.valueChanged.connect(self.responsive)
        self.ui.height_resize.valueChanged.connect(self.responsive)
        self.ui.width.valueChanged.connect(self.responsive)
        self.ui.height.valueChanged.connect(self.responsive)

        ### traning class
        self.ui.pushButton_4.clicked.connect(self.start_training_)
        self.ui.pushButton_3.clicked.connect(self.performValidate)
        self.ui.pushButton_2.clicked.connect(self.openImage_test)

        self.ui.toolButton_7.clicked.connect(self.displayJSON)

    def add(self):
        if self.mode == 1:
            # append the parameter to the json
            """
            if self.ui.comboBox.currentText() == "Rotate" -> param = self.ui.rotate_limit.value()
            and so on
            """
            pass
        elif self.mode == 4:
            pass
        elif self.mode == 7:
            pass

    def displayJSON(self):
        # Find the QTextBrowser widget in your UI.
        self.infoDisplay = QTextBrowser()
        self.infoDisplay.setMinimumSize(800, 600)
        # Load and parse the JSON configuration.
        config = json.loads(json_config)

        # Convert the JSON object to an HTML string.
        html_content = self.json_to_html(config)

        # Set the HTML content to the QTextBrowser.
        self.infoDisplay.setHtml(html_content)
        self.infoDisplay.show()

    def json_to_html(self, json_obj):
        html = "<html><body>"
        for category, techniques in json_obj.items():
            html += f"<h2>{category}</h2>"
            for technique, params in techniques.items():
                html += f"<p><b>{technique}:</b></p><ul>"
                if isinstance(params['param'], dict):
                    for key, value in params['param'].items():
                        html += f"<li>{key}: {value}</li>"
                else:
                    html += f"<li>{params['param']}: {params.get('value', 'N/A')}</li>"
                html += "</ul>"
        html += "</body></html>"
        return html

    def performValidate(self):
        classLabel = self.validateClass.validate(image_path='')
        self.ui.label_31.setText(f"Predited label is {classLabel}")

    def start_training_(self):
        # Here you can define or retrieve batch size and learning rate from the UI
        batch_size = 32  # Example batch size
        lr = 0.001       # Example learning rate
        
        # Create a Worker instance and set up the Trainer with it
        self.worker = Worker(batch_size, lr)
        self.worker.update_signal.connect(self.update_training_progress)

        # Connect the loss signal to update the graph
        self.worker.loss_signal.connect(self.update_graph)

        # Create and start the Trainer QRunnable
        self.trainer = Trainer(self.worker)
        self.thread_pool.start(self.trainer)

    # New method to update the graph
    def update_graph(self, running_loss):
        # Add running loss to some list if you want to store the history
        self.running_losses.append(running_loss)

        # Update the graph
        self.ax.clear()
        self.ax.plot(self.running_losses, 'r-')  # Plot the running loss as a red line
        self.ax.set_title('Running Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.canvas.draw()

    def openImage_test(self):
        self.file, _ = QFileDialog.getOpenFileName(self, "Open Image", "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/test","")

        if self.file:
            pixmap = QPixmap(self.file)
            self.ui.CLS_img.setPixmap(pixmap)
        else:
            QMessageBox.information(self, 'Error', 'Unable to open image', QMessageBox.Ok)

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
        self.ui.comboBox_P.currentIndexChanged.connect(self.controlSubStack_P)

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

        if self.ui.comboBox.currentText() == 'Horizontal Flip' or self.ui.comboBox.currentText() == 'Vertical Flip':
            self.responsive()

        current_text = self.ui.comboBox.currentText()
        if current_text in options_mapping:
            self.ui.param_stack.setCurrentIndex(options_mapping[current_text])

    def controlSubStack_P(self):
        options_mapping_P = {
        "Decayed log": 2,
        "Scale variable by square root": 0,
        "Logarithm": 1,
        "Normalization": 4,
        "Absolute": 3,
        }

        if self.ui.comboBox_P.currentText() == 'Absolute':
            self.responsive()

        current_text = self.ui.comboBox_P.currentText()
        if current_text in options_mapping_P:
            self.ui.stackedPreprocessing.setCurrentIndex(options_mapping_P[current_text])

    def testLambda(self, index):

        self.ui.img_original_oc.setPixmap(QPixmap())
        self.ui.img_result_oc.setPixmap(QPixmap())
        self.ui.o_i_s_label.setPixmap(QPixmap())
        self.ui.o_i_s_label_2.setPixmap(QPixmap())
        self.ui.r_i_s_label.setPixmap(QPixmap())
        self.ui.r_i_s_label_2.setPixmap(QPixmap())

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
            self.ui.transform_stack.setCurrentIndex(0)
            self.ui.comboBox_P.setCurrentText('Scale variable by square root')
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
            self.ui.state_label.setText("Deep Learning for Classification")
            self.ui.display_stack.setCurrentIndex(2)
            self.ui.transform_stack.setCurrentIndex(2)
            self.mode = 7
        elif index == 8:
            self.ui.state_label.setText("Deep Learning for Segmentation")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(0)
            self.mode = 8
        elif index == 9:
            self.ui.state_label.setText("Deep Learning for Object Detection")
            self.ui.display_stack.setCurrentIndex(0)
            self.ui.transform_stack.setCurrentIndex(1)
            self.mode = 9

    def viewHandle(self, index):
        if self.mode == 1:
            if index == 1:
                if self.rootdir == '':
                    QMessageBox.information(self, 'Error', 'Please upload a folder', QMessageBox.Ok)
                else:
                    self.ui.img_original_oc.setPixmap(self.pixmap_from_cv_image(self.base_img))
                    self.ui.img_result_oc.setPixmap(self.pixmap_from_cv_image(self.base_img))
            elif index == 2:
                if self.rootdir == '':
                    QMessageBox.information(self, 'Error', 'Please upload a folder', QMessageBox.Ok)
                else:
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

    def helpSheet(self):
        manual_text = """
        Vision GUI Manual
        -----------------
        
        Creating More Data:
        1. Open the Vision GUI application.
        2. Use the toolbar on the top to start augmentation.
        3. Save your images and necessary files.
        4. ....
        5. ...

        For further details and advanced features, refer to the official documentation.

        Contact huyhoang.nguyen@infineon.com for assistance.
        """

        self.manual_widget = QTextEdit()
        self.manual_widget.setPlainText(manual_text)
        self.manual_widget.setReadOnly(True)
        self.manual_widget.setWindowTitle('Manual')
        self.manual_widget.setGeometry(100, 100, 600, 400)
        self.manual_widget.show()

    def openImage(self):
        self.image_file, _ = QFileDialog.getOpenFileName(self, "Open Image", "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/test","")

        if self.image_file:
            self.root_dir = os.path.dirname(self.image_file)
            pixmap = QPixmap(self.image_file)
            self.ui.img_original_oc.setPixmap(pixmap)
            self.ui.o_i_s_label.setPixmap(pixmap)
            self.base_img = cv2.imread(self.image_file)
            self.updateImageShape()
        else:
            QMessageBox.information(self, 'Error', 'Unable to open image', QMessageBox.Ok)

    def openMask(self):
        self.mask_file, _ = QFileDialog.getOpenFileName(self, "Open a mask", "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/test","JPG Files (*.jpeg *.jpg );;\
                                                           PNG Files (*.png);;Bitmap Files (*.bmp);;GIF Files (*.gif)")
        if self.mask_file:
            pixmap = QPixmap(self.mask_file)
            self.ui.o_i_s_label_2.setPixmap(pixmap)
        else:
            QMessageBox.information(self, 'Error', 'Unable to open mask', QMessageBox.Ok)

    def openBBox(self):  
        self.bbox_file, _ = QFileDialog.getOpenFileName(self, "Open XML file", "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/test","(*.xml)")
        if self.bbox_file:
            tree = ET.parse(self.bbox_file)
            root = tree.getroot()
            # Initialize an empty list to store bounding boxes
            self.bboxes = []
            # Iterate through each "object" node to extract bounding box coordinates
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                label = str(obj.find('name').text)
                self.bboxes.append([xmin, ymin, xmax, ymax, label])
            
            img = cv2.imread(self.image_file)
            result = self.drawBBox(img, self.bboxes)
            pixmap = self.pixmap_from_cv_image(result)
            self.ui.img_original_oc.setPixmap(pixmap)
         
        else:
            QMessageBox.information(self, 'Error', 'Please upload the bounding box', QMessageBox.Ok)

    def drawBBox(self, image, bbox):
        for each in bbox:
            xmin, ymin, xmax, ymax, label = each
            xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))
            image_bbox = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
        return image_bbox

    def saveBBox(self):
        dir = os.path.join(self.root_dir, "bbox.txt")
        with open(dir, 'w') as file:
            for item in self.transformed_bboxes:
                line = ' '.join([str(x) for x in item[::-1]])  # Reversing the tuple and joining its elements
                file.write(f"{line}\n")  # Writing the formatted line to the file
        QMessageBox.information(self, 'Successful', 'Bounding box saved', QMessageBox.Ok)

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
        QMessageBox.information(self, 'Successful', 'Image saved', QMessageBox.Ok)
       
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
        if self.rootdir == '':
            QMessageBox.information(self, 'Error', 'Please upload a folder', QMessageBox.Ok)
        else:
            if self.mode == 1:
                for subdir, dirs, files in os.walk(self.rootdir):
                    base_dir = os.path.dirname(self.rootdir)
                    os.mkdir(os.path.join(base_dir, 'transformed'))
                    total = len(files)
                    counter = float(self.percentage/100)*total
                    for file in files:
                        counter = counter - 1 
                        if counter < 0:
                            break
                        frame = cv2.imread(os.path.join(subdir, file))  
                        original_filename = os.path.basename(os.path.join(subdir, file)).split('.')[0]
                        custom_filename = f"{original_filename}_transformed.jpg"
                        savedDir = os.path.join(base_dir, 'transformed', custom_filename)
                        cv2.imwrite(savedDir, self.transformation(frame))
                    break
                QMessageBox.information(self, 'Completed', 'Augmented folder has been saved', QMessageBox.Ok)
            elif self.mode == 3:
                for subdir, dirs, files in os.walk(self.rootdir):
                    base_dir = os.path.dirname(self.rootdir)
                    os.mkdir(os.path.join(base_dir, 'transformed'))
                    total = len(files)
                    counter = float(self.percentage/100)*total
                    for file in files:
                        counter = counter - 1 
                        if counter < 0:
                            break
                        frame = cv2.imread(os.path.join(subdir, file))  
                        original_filename = os.path.basename(os.path.join(subdir, file)).split('.')[0]
                        custom_img_filename = f"{original_filename}_transformed.jpg"
                        custom_bbox_filename = f"{original_filename}_transformed.txt"
                        savedDir = os.path.join(base_dir, 'transformed', custom_img_filename)
                        cv2.imwrite(savedDir, self.transformation(frame)[0])
                    break
                QMessageBox.information(self, 'Completed', 'Augmented folder has been saved', QMessageBox.Ok)

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
        if self.image_file == '':
            QMessageBox.information(self, 'Error', 'Please upload an image', QMessageBox.Ok)
        elif self.mode == 1:
            img = self.transformation(self.base_img)
            self.pixmap = self.pixmap_from_cv_image(img)
            self.ui.img_result_oc.setPixmap(self.pixmap)
        elif self.mode == 2:
            img, mask = self.transformation(self.base_img)
            self.pixmap = self.pixmap_from_cv_image(img)
            self.ui.r_i_s_label.setPixmap(self.pixmap)
            self.pixmap_mask = self.pixmap_from_cv_image(mask)
            self.ui.r_i_s_label_2.setPixmap(self.pixmap_mask)
        elif self.mode == 3:
            img, bbox = self.transformation(self.base_img)
            result = self.drawBBox(image=img, bbox=bbox)
            self.pixmap = self.pixmap_from_cv_image(result)
            self.ui.img_result_oc.setPixmap(self.pixmap)
        elif self.mode == 4:
            img = self.transformation(self.base_img)
            self.pixmap = self.pixmap_from_cv_image(img)
            self.ui.img_result_oc.setPixmap(self.pixmap)

    def apply_changes(self):
        self.base_img = self.image_tranformed
        self.ui.img_original_oc.setPixmap(self.pixmap_from_cv_image(self.base_img))

    def transformation(self, image_data):
        if self.mode == 1 or self.mode == 2:
            if self.ui.comboBox.currentText() == "Rotate":
                self.transforms = self.algorithms.rotate(limit=self.ui.rotate_limit.sliderPosition(), border=self.ui.rotate_options.currentText())
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
                self.transforms = self.algorithms.Blur(blur_limit=self.ui.blur_limit.value())
            elif self.ui.comboBox.currentText() == "Solarize":                   
                self.transforms = self.algorithms.Solarize(threshold=self.ui.solarize_value.value())   
            elif self.ui.comboBox.currentText() == "Noise":                   
                self.transforms = self.algorithms.GaussNoise(var_limit=self.ui.var_limit.value(), mean=self.ui.mean_value.value()) 
            
            if self.mode == 1:
                self.image_tranformed = self.transforms(image=image_data)["image"]
                return self.image_tranformed
            elif self.mode == 2:
                mask = cv2.imread(self.mask_file)
                transformed = self.transforms(image=image_data, mask=mask)
                self.image_tranformed = transformed['image']
                self.transformed_mask = transformed['mask']
                return self.image_tranformed, self.transformed_mask 
        
        elif self.mode == 3:
            if self.ui.comboBox.currentText() == "Rotate":
                self.transforms = self.algorithms.rotateOD(limit=self.ui.rotate_limit.sliderPosition(), border=self.ui.rotate_options.currentText())
            elif self.ui.comboBox.currentText() == "Crop":
                self.transforms = self.algorithms.McropOD(xmin=self.ui.width.sliderPosition()[0], xmax=self.ui.width.sliderPosition()[1],
                                                ymin=self.ui.height.sliderPosition()[0], ymax=self.ui.height.sliderPosition()[1]) 
            elif self.ui.comboBox.currentText() == "Horizontal Flip":
                self.transforms = self.algorithms.hflipOD()
            elif self.ui.comboBox.currentText() == "Vertical Flip":
                self.transforms = self.algorithms.vflipOD()
            elif self.ui.comboBox.currentText() == "Resize":
                self.transforms = self.algorithms.resizeOD(width=self.ui.width_resize.value(), height=self.ui.height_resize.value())
            elif self.ui.comboBox.currentText() == "Blur":                   
                self.transforms = self.algorithms.BlurOD(blur_limit=self.ui.blur_limit.value())
            elif self.ui.comboBox.currentText() == "Solarize":                   
                self.transforms = self.algorithms.SolarizeOD(threshold=self.ui.solarize_value.value())   
            elif self.ui.comboBox.currentText() == "Noise":                   
                self.transforms = self.algorithms.GaussNoiseOD(var_limit=self.ui.var_limit.value(), mean=self.ui.mean_value.value()) 
            
            transformed = self.transforms(image=image_data, bboxes=self.bboxes)
            self.image_tranformed = transformed['image']
            self.transformed_bboxes = transformed['bboxes']
        
            return self.image_tranformed, self.transformed_bboxes
        
        elif self.mode == 4:
            if self.ui.comboBox_P.currentText() == 'Scale variable by square root':
                self.image_tranformed = self.algorithms.scale_contrast(mean_shift=self.ui.mean_shift.value(), 
                                                                   contrast_scaling=self.ui.scale_limit.value(), img=image_data)
            if self.ui.comboBox_P.currentText() == 'Logarithm':
                self.image_tranformed = self.algorithms.log(c=self.ui.scale_factor.value(), gamma=self.ui.gamma_factor.value(), 
                                                        img=image_data)
            if self.ui.comboBox_P.currentText() == 'Decayed log':
                self.image_tranformed = self.algorithms.decay(decay_factor=self.ui.decay_factor.value(), img=image_data)
            if self.ui.comboBox_P.currentText() == 'Normalization':
                self.image_tranformed = self.algorithms.normalization(alpha=self.ui.alpha_value.value(), beta=self.ui.beta_value.value(), img=image_data)
            if self.ui.comboBox_P.currentText() == 'Absolute':
                self.image_tranformed = self.algorithms.absolute(img=image_data)
            return self.image_tranformed

    def updateImageShape(self):
        image = cv2.imread(self.image_file)
        height, width, channels = image.shape
        self.ui.width_resize.setRange(1,width)
        self.ui.height_resize.setRange(1,height)
        self.ui.width.setRange(1, width)
        self.ui.height.setRange(1,height)

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