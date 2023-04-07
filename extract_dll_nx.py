from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
import traceback, sys
import os
import shutil


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    def __init__(self, path_in, path_out):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.path_in = path_in
        self.path_out = path_out
        self.signals = WorkerSignals()
    def run(self):
        try:
            result = self.export_function(self.path_in, self.path_out)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def copy_dll(self, src, target):
        files = os.listdir(src)
        for fname in files:
            shutil.copy2(os.path.join(src, fname), target)

    def export_function(self, path_in, path_out):
        libsurv_dir = os.path.join(path_in, 'libsurv')
        opencv_dir = os.path.join(path_in, 'opencv')
        plugin_dir = os.path.join(path_in, 'plugin')
        license_dir = os.path.join(path_in, 'license')
        onnx_dir = os.path.join(path_in, 'onnx')
        # tensorrt_dir = os.path.join(path_in, 'tensorrt')

        plugin_dll_dir = path_out
        nxserver_dir = os.path.dirname(os.path.dirname(path_out))

        if os.path.isdir(nxserver_dir):
            self.copy_dll(libsurv_dir, plugin_dll_dir)
            self.copy_dll(plugin_dir, plugin_dll_dir)
            self.copy_dll(opencv_dir, plugin_dll_dir)
            self.copy_dll(license_dir, plugin_dll_dir)
            self.copy_dll(onnx_dir, nxserver_dir)
            return 'Copy DLL done'
        else:
            return '-1'


class Ui(QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('copydll.ui', self)

        self.button_load_in = self.findChild(QPushButton, 'pushButton_dir_in')
        self.button_load_in.clicked.connect(self.load_in)

        self.button_load_out = self.findChild(QPushButton, 'pushButton_dir_out')
        self.button_load_out.clicked.connect(self.load_out)

        self.button_export = self.findChild(QPushButton, 'pushButton_xport')
        self.button_export.clicked.connect(self.export_dll)

        self.button_cancel = self.findChild(QPushButton, 'pushButton_cancel')
        self.button_cancel.clicked.connect(self.close)

        self.status = self.findChild(QLabel, 'label_status')
        self.status.setText('...')

        self.path_in_box = self.findChild(QComboBox, 'comboBox_path_in')
        self.path_out_box = self.findChild(QComboBox, 'comboBox_path_out')
        self.path_in = ''
        self.path_out = ''

        self.threadpool = QThreadPool()

        self.show()

    def load_in(self):
        self.path_in = str(QFileDialog.getExistingDirectory(self, "Select folder in"))
        self.path_in_box.addItem(self.path_in)

    def load_out(self):
        self.path_out = str(QFileDialog.getExistingDirectory(self, "Select folder out"))
        self.path_out_box.addItem(self.path_out)

    def print_output(self, s):
        if s!='-1':
            self.status.setText(s)
            QMessageBox.question(self, 'Message', 'Complete Copy DLL, check output folder', QMessageBox.Yes)
        else:
            self.status.setText('Failed!')
            QMessageBox.question(self, 'Message', 'Can not find requirements folder, please check again!',
                                 QMessageBox.Yes)

    def export_dll(self):
        self.status.setText('Exporting ...')
        if (os.path.isdir(self.path_in) and os.path.isdir(self.path_out)):
            worker = Worker(self.path_in, self.path_out)
            worker.signals.result.connect(self.print_output)
            self.threadpool.start(worker)

        else:
            QMessageBox.question(self, 'Message', 'Input or Output path is empty!', QMessageBox.Yes)

    def close(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui()
    app.exec_()