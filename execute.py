from MainWindow import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ultralytics import YOLO
import numpy as np
import cv2

import pytesseract
# Đường dẫn trỏ đến công cụ nhận diện ký tự Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import sys 
import string

# Load mô hình nhận diện đối tượng và nhận diện biển số xe
modelObjectDetection = YOLO("modelOD/yolov8m.pt")
modelLicensePlate = YOLO("modelLP/license_plate_detector.pt")


# Hàm đọc ký tự trên biển số xe với đầu vào là ảnh biển số xe đã được nhận diện
def readLicensePlateTesseract(licensePlateCrop):
    options = string.digits + string.ascii_letters + "-."
    gray = cv2.bilateralFilter(licensePlateCrop, 10, 20, 20)
    text = pytesseract.image_to_string(gray, config='-c tessedit_char_whitelist='+options).strip()
    text = text.replace('(','').replace(')','').replace(',','')
    print("Text: "+ text)
    return text

class window(QtWidgets.QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.importImgBtn.clicked.connect(self.selectImageDialog)
        self.ui.executeFunction.clicked.connect(self.objectDetectionExecute)

    # Hàm cho phép người dùng chọn ảnh từ máy tính
    def selectImageDialog(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select image", "D:\\", "Image *.jpg; *.jpeg; *.png", options=options)
        if filePath:
            self.ui.imageUrl.setText(filePath)
            # Hiển thị ảnh người dùng đã chọn trên ứng dụng
            pixmap = QPixmap(filePath)
            scaledInput = pixmap.scaled(self.ui.imageLabel.size(), Qt.KeepAspectRatio)
            self.ui.imageLabel.setPixmap(scaledInput)
            self.ui.resultTableWidget.setColumnCount(0)
            self.ui.resultTableWidget.setRowCount(0)
    
    # Hàm thực thi nhận diện khi người dùng đã chọn ảnh và chọn chức năng, và nhấn nút Thực hiện trên màn hình
    def objectDetectionExecute(self):
        if self.ui.imageUrl.text() != "":
            try:
                # Nạp ảnh đầu vào ứng dụng
                imageInput = cv2.imread(self.ui.imageUrl.text())
                imageInputCopy = imageInput.copy()
                if self.ui.objectDetectionRadioBtn.isChecked():
                    # Lấy ra danh sách kết quả nhận diện bao gồm vị trí đường bao quanh đối tượng và tên đối tượng (car, person,...)
                    results = modelObjectDetection(imageInput, conf=0.7)
                if self.ui.licensePlateRadioBtn.isChecked():
                    # Lấy ra danh sách kết quả nhận diện bao gồm vị trí đường bao quanh đối tượng và tên đối tượng (cụ thể là biển số xe)
                    results = modelLicensePlate(imageInput, conf=0.7)

                for res in results:
                    boxes = res.boxes.xyxy.tolist()
                    classes = res.boxes.cls.tolist()
                    confidences = res.boxes.conf.tolist()
                    names = res.names

                    # List count object
                    listObjectDetected = list()

                    # List license plate detect
                    listLicensePlateDetected = list()
                
                    for box, cls, conf in zip(boxes, classes, confidences):
                        # Xác định vị trí đường bao quanh đối tượng: Điểm góc trên cùng bên trái (x1,y1), điểm góc dưới cùng bên phải (x2,y2)
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        confidence = conf 
                        # Lấy ra tên đối tượng
                        detectedClass = cls 
                        name = names[int(cls)]

                        listObjectDetected.append(name)

                        if self.ui.licensePlateRadioBtn.isChecked():
                            # Cắt vùng ảnh chứa biển số xe
                            licensePlateCrop = imageInputCopy[y1:y2, x1:x2, :]
                            listLicensePlateDetected.append(licensePlateCrop)

                        # Vẽ đường bao quanh đối tượng được nhận diện và hiển thị tên đối tượng
                        cv2.rectangle(imageInput, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(imageInput, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2, cv2.LINE_AA) 

                # Đếm số lượng mỗi loại đối tượng
                counter = {x:listObjectDetected.count(x) for x in listObjectDetected}               
            
                # Convert image BGR to RGB    
                imageMatrixOutput = imageInput[:, :, ::-1].copy()
                qImageOutput = QImage(imageMatrixOutput.data, imageMatrixOutput.shape[1], imageMatrixOutput.shape[0],
                                      imageMatrixOutput.strides[0], QImage.Format_RGB888)
                pixmapOutput = QPixmap(qImageOutput)
                scaledOutput = pixmapOutput.scaled(self.ui.imageLabel.size(), Qt.KeepAspectRatio)
                self.ui.imageLabel.setPixmap(scaledOutput)

                # Trường hợp người dùng chọn chức năng đếm số lượng đối tượng
                if self.ui.objectDetectionRadioBtn.isChecked():
                    self.ui.resultTableWidget.setRowCount(0)
                    self.ui.resultTableWidget.setColumnCount(2)
                    self.ui.resultTableWidget.setRowCount(len(counter))
                    self.ui.resultTableWidget.setHorizontalHeaderLabels(("Đối tượng", "Số lượng"))

                    for index, (key, value) in enumerate(counter.items()):
                        self.ui.resultTableWidget.setItem(index, 0, QTableWidgetItem(key))
                        self.ui.resultTableWidget.setItem(index, 1, QTableWidgetItem(str(value)))
                
                # Trường hợp người dùng chọn chức năng đọc biển số xe
                if self.ui.licensePlateRadioBtn.isChecked():
                    self.ui.resultTableWidget.setRowCount(0)
                    self.ui.resultTableWidget.setColumnCount(2)
                    self.ui.resultTableWidget.setRowCount(len(listLicensePlateDetected))
                    self.ui.resultTableWidget.setHorizontalHeaderLabels(("Biển số xe", "Kết quả đọc"))

                    for index, licensePlateMatrix in enumerate(listLicensePlateDetected):
                        cv2.imwrite("licensePlateMatrix.jpg", licensePlateMatrix)
                        
                        licensePlateMatrix = licensePlateMatrix.copy()
               
                        qImage = QImage(licensePlateMatrix.data, licensePlateMatrix.shape[1], licensePlateMatrix.shape[0], 
                                        licensePlateMatrix.strides[0], QImage.Format_RGB888)
                        pixmap = QPixmap(qImage)
                        
                        # Chuyển ảnh biển số xe từ ảnh màu sang ảnh đen trắng để thực hiện đọc ký tự
                        licensePlateCropGray = cv2.cvtColor(licensePlateMatrix, cv2.COLOR_BGR2GRAY)
                         
                        licensePlateText = ""
                    
                        try:
                            # Thực hiện đọc ký tự trên biển số xe
                            licensePlateText = readLicensePlateTesseract(licensePlateCropGray)    
                          
                        except TypeError as typeError:
                            self.ui.informationError.setText("Không đọc được biển số xe")
                        qlabelImage = QLabel()
                        qlabelImage.setPixmap(pixmap)
                        self.ui.resultTableWidget.setCellWidget(index, 0, qlabelImage)
                        self.ui.resultTableWidget.setItem(index, 1, QTableWidgetItem(licensePlateText))
                        self.ui.resultTableWidget.resizeColumnsToContents()
                        self.ui.resultTableWidget.resizeRowsToContents()
                        
            except FileNotFoundError:
                self.ui.informationError.setText("Không tìm thấy file")
        else:
            self.ui.informationError.setText("Bạn chưa chọn file")

def createApp():
    app = QtWidgets.QApplication(sys.argv)
    win = window()
    win.show()
    sys.exit(app.exec_())

createApp()

