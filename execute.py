from MainWindow import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ultralytics import YOLO
import numpy as np
import cv2

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import sys 
import string

modelObjectDetection = YOLO("modelOD/yolov8m.pt")
modelLicensePlate = YOLO("modelLP/license_plate_detector.pt")


# Pytesseract
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

    def selectImageDialog(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select image", "D:\\", "Image *.jpg; *.jpeg; *.png", options=options)
        if filePath:
            self.ui.imageUrl.setText(filePath)
            pixmap = QPixmap(filePath)
            scaledInput = pixmap.scaled(self.ui.imageLabel.size(), Qt.KeepAspectRatio)
            self.ui.imageLabel.setPixmap(scaledInput)
            self.ui.resultTableWidget.setColumnCount(0)
            self.ui.resultTableWidget.setRowCount(0)
    
    def objectDetectionExecute(self):
        result = None
        if self.ui.imageUrl.text() != "":
            try:

                imageInput = cv2.imread(self.ui.imageUrl.text())
                imageInputCopy = imageInput.copy()
                if self.ui.objectDetectionRadioBtn.isChecked():
                    results = modelObjectDetection(imageInput, conf=0.7)
                if self.ui.licensePlateRadioBtn.isChecked():
                    results = modelLicensePlate(imageInput, conf=0.7)

                for res in results:
                    boxes = res.boxes.xyxy.tolist()
                    classes = res.boxes.cls.tolist()
                    confidences = res.boxes.conf.tolist()
                    names = res.names

                    # List count object
                    listObjectDetected = list()

                    # List license plate detetec
                    listLicensePlateDetected = list()
                
                    for box, cls, conf in zip(boxes, classes, confidences):
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        confidence = conf 
                        detectedClass = cls 
                        name = names[int(cls)]
                        listObjectDetected.append(name)

                        if self.ui.licensePlateRadioBtn.isChecked():
                            licensePlateCrop = imageInputCopy[y1:y2, x1:x2, :]
                            listLicensePlateDetected.append(licensePlateCrop)

                        cv2.rectangle(imageInput, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(imageInput, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2, cv2.LINE_AA) 

                counter = {x:listObjectDetected.count(x) for x in listObjectDetected}               
            
                # Convert image BGR to RGB    
                imageMatrixOutput = imageInput[:, :, ::-1].copy()
                qImageOutput = QImage(imageMatrixOutput.data, imageMatrixOutput.shape[1], imageMatrixOutput.shape[0],
                                      imageMatrixOutput.strides[0], QImage.Format_RGB888)
                pixmapOutput = QPixmap(qImageOutput)
                scaledOutput = pixmapOutput.scaled(self.ui.imageLabel.size(), Qt.KeepAspectRatio)
                self.ui.imageLabel.setPixmap(scaledOutput)

                # Count object detection
                if self.ui.objectDetectionRadioBtn.isChecked():
                    self.ui.resultTableWidget.setRowCount(0)
                    self.ui.resultTableWidget.setColumnCount(2)
                    self.ui.resultTableWidget.setRowCount(len(counter))
                    self.ui.resultTableWidget.setHorizontalHeaderLabels(("Đối tượng", "Số lượng"))

                    for index, (key, value) in enumerate(counter.items()):
                        self.ui.resultTableWidget.setItem(index, 0, QTableWidgetItem(key))
                        self.ui.resultTableWidget.setItem(index, 1, QTableWidgetItem(str(value)))
                
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

                        licensePlateCropGray = cv2.cvtColor(licensePlateMatrix, cv2.COLOR_BGR2GRAY)
                         
                        licensePlateText = ""
                    
                        try:
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

