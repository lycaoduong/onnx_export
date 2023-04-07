from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton, QComboBox, QLabel, QRadioButton, QCheckBox
from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
import traceback, sys
import os
from utils.convert_model import yolov4_darknet_2_onnx, yolov7_pt_2_onnx


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    def __init__(self, path, version=4, with_batch=False):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.save_path = path
        self.version = version
        self.batch = with_batch
        self.signals = WorkerSignals()
    def run(self):
        try:
            result = self.convert_function(self.save_path, self.version, with_batch=self.batch)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def convert_function(self, save_path, version, with_batch=False):
        if version == 4:
            cfg_file = [f for f in os.listdir(save_path) if f.endswith('.cfg')]
            weight_file = [f for f in os.listdir(save_path) if f.endswith('.weights')]
            if len(cfg_file) and len(weight_file):
                cfg = os.path.join(save_path, cfg_file[0])
                weight = os.path.join(save_path, weight_file[0])
                in_s, out_s = yolov4_darknet_2_onnx(cfg, weight, 1, None, save_path,
                                                    onnx_file_name=cfg_file[0][:-4], batch=with_batch)
                return 'Done. In: {}, Out: {}'.format(in_s, out_s)
            else:
                return '-1'
        elif version == 7:
            weight_file = [f for f in os.listdir(save_path) if f.endswith('.pt')]
            cfg_file = [f for f in os.listdir(save_path) if f.endswith('.yaml')]
            if len(weight_file) and len(cfg_file):
                weight = os.path.join(save_path, weight_file[0])
                cfg = os.path.join(save_path, cfg_file[0])
                in_s, out_s = yolov7_pt_2_onnx(cfg, weight, 512, save_path, onnx_file_name=weight_file[0][:-3],dynamic=False)
                return 'Done. In: {}, Out: {}'.format(in_s, out_s)
            else:
                return '-1'


class Ui(QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow.ui', self)

        self.button_load = self.findChild(QPushButton, 'pushButton_dir')
        self.button_load.clicked.connect(self.load_model)

        self.button_export = self.findChild(QPushButton, 'pushButton_xport')
        self.button_export.clicked.connect(self.export_model)

        self.button_cancel = self.findChild(QPushButton, 'pushButton_cancel')
        self.button_cancel.clicked.connect(self.close)

        self.yolov4 = self.findChild(QRadioButton, 'radioButton_yolov4')
        self.yolov7 = self.findChild(QRadioButton, 'radioButton_yolov7')

        self.multi_batch = self.findChild(QCheckBox, 'checkBox_batch')

        self.status = self.findChild(QLabel, 'label_status')
        self.status.setText('...')

        self.path_model = self.findChild(QComboBox, 'comboBox_path')
        self.save_path = ''

        self.threadpool = QThreadPool()

        self.show()

    def load_model(self):
        self.save_path = str(QFileDialog.getExistingDirectory(self, "Select folder"))
        self.path_model.addItem(self.save_path)
        if self.yolov4.isChecked():
            self.status.setText('Select Path included names, cfg, weights with yolov4-Darknet format')
        if self.yolov7.isChecked():
            self.status.setText('Select Path included *.pt with Yolov7-pytorch format')

    def print_output(self, s):
        if s!='-1':
            self.status.setText(s)
            QMessageBox.question(self, 'Message', 'Complete convert to ONNX, check input folder', QMessageBox.Yes)
        else:
            self.status.setText('Failed!')
            QMessageBox.question(self, 'Message', 'Can not load file from input path, please check again!',
                                 QMessageBox.Yes)

    def export_model(self):
        self.status.setText('Exporting ...')
        if self.save_path != '':
            if self.yolov4.isChecked():
                if self.multi_batch.isChecked():
                    worker = Worker(self.save_path, 4, with_batch=True)
                else:
                    worker = Worker(self.save_path, 4)
            elif self.yolov7.isChecked():
                worker = Worker(self.save_path, 7)

            worker.signals.result.connect(self.print_output)

            self.threadpool.start(worker)

        else:
            QMessageBox.question(self, 'Message', 'Input path is empty!', QMessageBox.Yes)

    def close(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui()
    app.exec_()