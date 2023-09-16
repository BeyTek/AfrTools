import os
import cv2
import numpy as np
import sys
import uuid
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog, QCheckBox,QSlider,QGridLayout, QTabWidget
from PySide6.QtGui import QFont,QPixmap, QImage
import torch
import torchvision
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.mobilenet_v3_small()
model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = True

epsilon = 0.02
image = None
perturbed_image = None  
random_uuid = str(uuid.uuid1())[:6]
class AFRtools(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.init_ui()
        self.pixel_noise_level = 0
        
    def init_ui(self):
        self.setWindowTitle("AFR Tools")
        
        central_widget = QWidget(self)
        layout = QVBoxLayout()

        title_label = QLabel("Anti Facial Recognition Tools", self)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        bar_label = QLabel("Pixels Intensity", self)
        bar_font = QFont()
        bar_font.setPointSize(17)
        bar_font.setBold(True)
        bar_label.setFont(bar_font)
        bar_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(bar_label)
        
        self.pixel_noise_slider = QSlider(Qt.Horizontal, self)
        self.pixel_noise_slider.setMinimum(0)
        self.pixel_noise_slider.setMaximum(100) 
        self.pixel_noise_slider.setValue(0) 
        self.pixel_noise_slider.valueChanged.connect(self.update_pixel_level)
        self.pixel_noise_slider.valueChanged.connect(self.update_slider_style)
        layout.addWidget(self.pixel_noise_slider)

        checkbox_layout = QGridLayout()
        checkbox_layout.setAlignment(Qt.AlignVCenter)

        self.checkbox_pixels = QCheckBox("Pixels", self)
        self.checkbox_gaussian_blur = QCheckBox("Gaussian Blur", self)
        self.checkbox_resize_eyes = QCheckBox("Resize Eyes", self)
        self.checkbox_random_mask = QCheckBox("Base Noise", self)
        self.checkbox_add_text = QCheckBox("Add Credit Text", self)
        self.checkbox_bw_contrast = QCheckBox("Black and White", self)

        checkbox_layout.addWidget(self.checkbox_pixels, 0, 0)
        checkbox_layout.addWidget(self.checkbox_gaussian_blur, 0, 1)
        checkbox_layout.addWidget(self.checkbox_resize_eyes, 1, 0)
        checkbox_layout.addWidget(self.checkbox_random_mask, 1, 1)
        checkbox_layout.addWidget(self.checkbox_add_text, 2, 0)
        checkbox_layout.addWidget(self.checkbox_bw_contrast, 2, 1)

        layout.addLayout(checkbox_layout)

        choisir_button = QPushButton("Choose Image", self)
        choisir_button.clicked.connect(self.choisir_photo)
        layout.addWidget(choisir_button)

        convertir_button = QPushButton("Anonymize", self)
        convertir_button.clicked.connect(self.convertir_photo)
        layout.addWidget(convertir_button)

        supprimer_exif_button = QPushButton("Clean Metadata", self)
        supprimer_exif_button.clicked.connect(self.supprimer_exif)
        layout.addWidget(supprimer_exif_button)

        self.status_label = QLabel(self)
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


        style = """
                QMainWindow {
                    background-color: #2E2157;
                    color: white;
                    border-radius: 12px; /* Bords arrondis */
                }

                QPushButton {
                    background-color: #6932A2;
                    border-radius: 12px;
                    color: white;
                    padding: 10px 20px;
                    margin: 5px;
                }

                QPushButton:hover {
                    background-color: #9B7FC7;
                }

                QLabel {
                    color: white;
                    margin-top: 10px; 
                }

                QCheckBox {
                    color: white; /* Changer la couleur du texte des cases à cocher en blanc */
                    background-color: #393939; /* Couleur de fond de la case à cocher (gris foncé) */
                    border: 1px solid #000000; /* Bordure de la case à cocher */
                    border-radius: 12px; 
                    padding: 8px; /* Espacement interne */
                }

                QCheckBox::indicator {
                    width: 10px; /* Largeur de la case à cocher */
                    height: 10px; /* Hauteur de la case à cocher */
                }

                QCheckBox::indicator:checked {
                    background-color: #4CAF50; /* Couleur de fond lorsque la case est cochée (vert) */
                    border: 1px solid #4CAF50; /* Bordure de la case cochée */
                    border-radius: 12px; 
                }
     
                        """ 
        self.setStyleSheet(style)
    def update_slider_style(self):
            slider_value = self.pixel_noise_slider.value()
            style = f"""
                QSlider::groove:horizontal {{
                    border: 1px solid #999999;
                    height: 5px;

                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E2157, stop:{slider_value / 100.0} #4CAF50, stop:{slider_value / 100.0} #4CAF50, stop:1.0 #4CAF50);
                }}
                QSlider::handle:horizontal {{
                    background: #2E2157;
                    border: 1px solid #4CAF50;
                    width: 15px;
                    margin: -5px 0px;
                    border-radius: 7px;
                }}
            """
            self.pixel_noise_slider.setStyleSheet(style)

    def choisir_photo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, options=options)
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.List)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.image_path = file_paths[0]
                self.status_label.setText(f"Actual pic: {self.image_path}")

    def convertir_photo(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            
            if self.checkbox_resize_eyes.isChecked():
                image = self.resize_eyes(image, 0.83, 0.83)

            if self.checkbox_random_mask.isChecked():
                image = self.add_random_mask(image)

            if self.checkbox_gaussian_blur.isChecked():
                image = self.add_gaussian_blur(image)

            if self.checkbox_pixels.isChecked():
                image = self.add_pixels(image)

            if self.checkbox_add_text.isChecked():
                image = self.add_text(image) 

            if self.checkbox_bw_contrast.isChecked():
                image = self.add_bw_contrast(image)
            
            current_directory = os.getcwd()
            output_path = os.path.join(current_directory, f"{random_uuid}-clean.jpg")
            cv2.imwrite(output_path, image)
            self.status_label.setText(f"Look Here: {output_path}")
        else:
            self.status_label.setText("Choose picture first.")


    def add_bw_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = 1.5  
        beta = 0.1 
        image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def supprimer_exif(self):
        if self.image_path:
            try:
                image = cv2.imread(self.image_path)
                image_no_exif = image.copy()
                current_directory = os.getcwd()
                output_path = os.path.join(current_directory, f"{random_uuid}-NoExif.jpg")
                cv2.imwrite(output_path, image_no_exif)
                self.status_label.setText("Metadata deleted successfully and image saved without EXIF.")
            except Exception as e:
                self.status_label.setText(f"Error: {str(e)}")
        else:
            self.status_label.setText("Choose a picture first.")
    
    def add_text(self, image):
        if self.checkbox_add_text.isChecked():
            text = 'Made by BeyTek, Anti Facial Recognition tools' 
            position = (50, 50)  
            font_scale = 1
            font_color = (0, 0, 0) 
            font_thickness = 2
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.putText(image, text, position,font_face, font_scale, font_color, font_thickness)
        return image

    def resize_eyes(self, image, factor_x, factor_y):
            eye_cascade = cv2.CascadeClassifier('eyes.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('face.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face)
                for (ex, ey, ew, eh) in eyes:
                    eye = face[ey:ey+eh, ex:ex+ew]
                    new_width = int(ew * factor_x)
                    new_height = int(eh * factor_y)
                    resized_eye = cv2.resize(eye, (new_width, new_height))
                    face[ey:ey+new_height, ex:ex+new_width] = resized_eye
            return image

    def add_pixels(self, image):
        num_pixels = int(self.pixel_noise_level * image.size // 3)
        for _ in range(num_pixels):
                x_rand = np.random.randint(0, image.shape[1])
                y_rand = np.random.randint(0, image.shape[0])
                image[y_rand, x_rand] = np.random.randint(0, 256, 3)
        return image

    def add_random_mask(self, image):
        mask = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        return image

    def add_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def update_pixel_level(self):
        pixel_noise_level = self.pixel_noise_slider.value() / 2000.0  
        self.pixel_noise_level = pixel_noise_level
        self.status_label.setText(f"{self.pixel_noise_level}")

class FGSM(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FGSM Attack")
        central_widget = QWidget(self)
        layout = QVBoxLayout()
        title_label = QLabel("FGSM Attack", self)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)


        button_open_image = QPushButton("Choose Image")
        button_open_image.clicked.connect(self.open_image)
        layout.addWidget(button_open_image)
       
        intensity_label = QLabel("Attack Intensity", self)
        intensity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(intensity_label)

        self.epsilon_slider = QSlider(Qt.Horizontal, self)
        self.epsilon_slider.setMinimum(2) 
        self.epsilon_slider.setMaximum(40)  
        self.epsilon_slider.setValue(20)  
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        layout.addWidget(self.epsilon_slider)
        
        button_attack_image = QPushButton("Attack Image")
        button_attack_image.clicked.connect(self.attack_image)
        layout.addWidget(button_attack_image)

        self.original_image_label = QLabel(self)
        layout.addWidget(self.original_image_label)
        self.perturbed_image_label = QLabel(self)
        layout.addWidget(self.perturbed_image_label)
        
        self.status_label = QLabel(self)
        layout.addWidget(self.status_label)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        style = """
            QMainWindow {
                background-color: #2E2157; 
                color: white; 
                border-radius: 12px; 
            }

            QPushButton {
                background-color: #6932A2;
                border-radius: 12px;
                color: white;
                padding: 10px 20px;
                margin: 5px;
            }

            QPushButton:hover {
                background-color: #9B7FC7; 
            }

            QLabel {
                color: white; 
                margin-top: 10px; 
            }
        """

        self.setStyleSheet(style)

    def open_image(self):
        global image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, options=options)
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.List)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                filename = file_paths[0]
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = img

    def attack_image(self):
        global perturbed_image, image, epsilon 
        if image is not None:
            num_classes = 2
            target = torch.tensor([random.randint(0, num_classes - 1)]).to(device)
            image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
            image_tensor.requires_grad = True
            output = model(image_tensor)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = image_tensor.grad.data
            data_grad = data_grad.detach()
            sign_data_grad = data_grad.sign()
            perturbed_image = image_tensor + epsilon * sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = torchvision.transforms.ToPILImage()(perturbed_image.squeeze(0).cpu())

            if perturbed_image:
                perturbed_image_np = np.array(perturbed_image) 
                current_directory = os.getcwd() 
                output_path = os.path.join(current_directory, f"{random_uuid}-Attackedfgsm.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))
                self.status_label.setText(f"Image saved as {output_path}")
    
    def update_epsilon(self):
        global epsilon
        epsilon_value = self.epsilon_slider.value() / 100.0 
        epsilon = max(0.02, min(0.4, epsilon_value))
        self.status_label.setText(f"{epsilon_value}") 



class CW(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("C&W Attack")
        central_widget = QWidget(self)
        layout = QVBoxLayout()
        title_label = QLabel("C&W Attack", self)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        button_open_image = QPushButton("Choose Image")
        button_open_image.clicked.connect(self.open_image)
        layout.addWidget(button_open_image)

        intensity_label = QLabel("Attack Intensity", self)
        intensity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(intensity_label)

        self.epsilon_slider = QSlider(Qt.Horizontal, self)
        self.epsilon_slider.setMinimum(2) 
        self.epsilon_slider.setMaximum(40)  
        self.epsilon_slider.setValue(20)  
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        layout.addWidget(self.epsilon_slider)

        button_attack_image = QPushButton("Attack Image")
        button_attack_image.clicked.connect(self.attack_image)
        layout.addWidget(button_attack_image)

        self.original_image_label = QLabel(self)
        layout.addWidget(self.original_image_label)
        self.perturbed_image_label = QLabel(self)
        layout.addWidget(self.perturbed_image_label)
        
        self.status_label = QLabel(self)
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        style = """
            QMainWindow {
                background-color: #2E2157; 
                color: white; 
                border-radius: 12px; 
            }

            QPushButton {
                background-color: #6932A2;
                border-radius: 12px;
                color: white;
                padding: 10px 20px;
                margin: 5px;
            }

            QPushButton:hover {
                background-color: #9B7FC7; 
            }

            QLabel {
                color: white; 
                margin-top: 10px; 
            }
        """

        self.setStyleSheet(style)

    def open_image(self):
        global image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, options=options)
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.List)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                filename = file_paths[0]
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = img
      

    def attack_image(self):
        global perturbed_image, image, epsilon
        if image is not None:
            num_classes = 1000
            target = torch.tensor([random.randint(0, num_classes - 1)]).to(device)
            image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
            image_tensor.requires_grad = True
            output = model(image_tensor)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = image_tensor.grad.data
            data_grad = data_grad.detach()
            sign_data_grad = data_grad.sign()
            perturbed_image = image_tensor + epsilon * sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = perturbed_image.squeeze(0).cpu().detach().numpy()
            perturbed_image = np.transpose(perturbed_image, (1, 2, 0))
            perturbed_image = (perturbed_image * 255).astype(np.uint8)
            if perturbed_image.size > 0:
                current_directory = os.getcwd()
                output_path = os.path.join(current_directory, f"{random_uuid}-Attackedcw.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR))
                self.status_label.setText(f"Image saved as {output_path}")
    
    def update_epsilon(self):
        global epsilon
        epsilon_value = self.epsilon_slider.value() / 100.0 
        epsilon = max(0.02, min(0.4, epsilon_value))
        self.status_label.setText(f"{epsilon_value}") 


class PGD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PGD Attack")
        central_widget = QWidget(self)
        layout = QVBoxLayout()
        title_label = QLabel("PGD Attack", self)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        button_open_image = QPushButton("Choose Image")
        button_open_image.clicked.connect(self.open_image)
        layout.addWidget(button_open_image)

        intensity_label = QLabel("Attack Intensity", self)
        intensity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(intensity_label)

        self.epsilon_slider = QSlider(Qt.Horizontal, self)
        self.epsilon_slider.setMinimum(2) 
        self.epsilon_slider.setMaximum(40)  
        self.epsilon_slider.setValue(20)  
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        layout.addWidget(self.epsilon_slider)

        button_attack_image = QPushButton("Attack Image")
        button_attack_image.clicked.connect(self.attack_image)
        layout.addWidget(button_attack_image)

        self.original_image_label = QLabel(self)
        layout.addWidget(self.original_image_label)
        self.perturbed_image_label = QLabel(self)
        layout.addWidget(self.perturbed_image_label)

        self.status_label = QLabel(self)
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        style = """
            QMainWindow {
                background-color: #2E2157; 
                color: white; 
                border-radius: 12px; 
            }

            QPushButton {
                background-color: #6932A2;
                border-radius: 12px;
                color: white;
                padding: 10px 20px;
                margin: 5px;
            }

            QPushButton:hover {
                background-color: #9B7FC7; 
            }

            QLabel {
                color: white; 
                margin-top: 10px; 
            }
        """

        self.setStyleSheet(style)

    def open_image(self):
        global image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, options=options)
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.List)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                filename = file_paths[0]
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = img

    def attack_image(self):
        global perturbed_image, image,epsilon
        if image is not None:
            num_iterations = 10
            alpha = epsilon / num_iterations
            num_classes = 2
            target = torch.tensor([random.randint(0, num_classes - 1)]).to(device)
            image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
            image_tensor.requires_grad = True

            for _ in range(num_iterations):
                output = model(image_tensor)
                loss = -output[0, target]
                loss.backward()
                data_grad = image_tensor.grad.data
                sign_data_grad = data_grad.sign()
                perturbed_image = image_tensor + alpha * sign_data_grad
                perturbed_image = torch.clamp(perturbed_image, 0, 1)
                image_tensor.data = perturbed_image

            perturbed_image = perturbed_image.squeeze(0).cpu().detach().numpy()
            perturbed_image = np.transpose(perturbed_image, (1, 2, 0))
            perturbed_image = (perturbed_image * 255).astype(np.uint8)
            if perturbed_image.size > 0:
                current_directory = os.getcwd()
                output_path = os.path.join(current_directory, f"{random_uuid}-Attackedpgd.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR))
                self.status_label.setText(f"Image saved as {output_path}")
    
    def update_epsilon(self):
        global epsilon
        epsilon_value = self.epsilon_slider.value() / 100.0 
        epsilon = max(0.02, min(0.4, epsilon_value))
        self.status_label.setText(f"{epsilon_value}") 


class FaceImageAdder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Mask Adder")
        central_widget = QWidget(self)
        layout = QVBoxLayout()
        title_label = QLabel("Mask Adder", self)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        explanation_label = QLabel("Add a face image (preferably in PNG format):")
        layout.addWidget(explanation_label)

        self.choose_mask_button = QPushButton("Choose Mask Image")
        self.choose_mask_button.clicked.connect(self.choose_mask_image)
        layout.addWidget(self.choose_mask_button)

        self.add_photo_button = QPushButton("Add Photo")
        self.add_photo_button.clicked.connect(self.add_photo)
        layout.addWidget(self.add_photo_button)

        button_add_face = QPushButton("Generate Face")
        button_add_face.clicked.connect(self.add_image_to_face)
        layout.addWidget(button_add_face)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_image_with_mask)
        
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        layout.addWidget(self.save_button)
        self.save_button.setDisabled(True)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        style = """
            QWidget {
                background-color: #2E2157; 
                color: white; 
            }

            QPushButton {
                background-color: #6932A2;
                border-radius: 12px;
                color: white; 
                padding: 10px 20px;
                margin: 5px;
            }

            QPushButton:hover {
                background-color: #9B7FC7; 
            }

            QLabel {
                color: white; 
                margin-top: 10px; 
            }
        """

        self.setStyleSheet(style)
        
    def choose_mask_image(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog(self, options=options)
        file_dialog.setNameFilter("Images (*.png)")
        file_dialog.setViewMode(QFileDialog.List)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                mask_image_path = file_paths[0]
                self.mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
                self.status_label.setText(f"Mask image selected: {mask_image_path}")
            else:
                self.status_label.setText("No mask image selected.")

    def add_photo(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog(self, options=options)
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.List)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                photo_image_path = file_paths[0]
                self.photo_image = cv2.imread(photo_image_path)
                self.status_label.setText(f"Photo image selected: {photo_image_path}")
            else:
                self.status_label.setText("No photo image selected.")

    def add_image_to_face(self):
        if hasattr(self, 'mask_image') and hasattr(self, 'photo_image'):
            mask_image = self.mask_image
            photo_image = self.photo_image
            if mask_image is not None and photo_image is not None:
                opacity = 0.4
                mask_image = cv2.addWeighted(mask_image, opacity, mask_image, 1 - opacity, 0)
                gray = cv2.cvtColor(photo_image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier('face.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # Ensure the mask image and the face region have the same size
                        mask_image_resized = cv2.resize(mask_image, (w, h))
                        x_offset = x
                        y_offset = y
                        x_offset = max(x_offset, 0)
                        y_offset = max(y_offset, 0)

                        if mask_image_resized.shape[2] == 4:
                            for c in range(0, 3):
                                photo_image[y_offset:y_offset + h, x_offset:x_offset + w, c] = (
                                    photo_image[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1 - mask_image_resized[:, :, 3] / 255.0) +
                                    mask_image_resized[:, :, c] * (mask_image_resized[:, :, 3] / 255.0)
                                )
                        else:
                            photo_face_region = photo_image[y_offset:y_offset + h, x_offset:x_offset + w, :]
                            combined_image = cv2.addWeighted(photo_face_region, 1, mask_image_resized, 1, 0)
                            photo_image[y_offset:y_offset + h, x_offset:x_offset + w, :] = combined_image

                    self.output_image = photo_image
                    self.update_image_display()
                else:
                    self.status_label.setText("No face detected !")
            else:
                self.status_label.setText("Error: Unable to load both mask and photo images.")



    def save_image_with_mask(self):
            current_directory = os.getcwd()
            output_path = os.path.join(current_directory, f"{random_uuid}-Masked.jpg")
            cv2.imwrite(output_path, self.output_image)
            self.status_label.setText(f"Image saved as {output_path}")
    
    def update_image_display(self):
        if hasattr(self, 'output_image'):
            display_image = self.output_image.copy()
            h, w, ch = display_image.shape
            max_width = 700
            max_height = 480
            if w > max_width or h > max_height:
                scale_w = max_width / w
                scale_h = max_height / h
                scale = min(scale_w, scale_h)
                w = int(w * scale)
                h = int(h * scale)
                display_image = cv2.resize(display_image, (w, h))
            cv2.imshow("Masked image", display_image)
            cv2.waitKey(0)
            self.image_label.clear()
            self.save_button.setDisabled(False)

        

def main():
    app = QApplication(sys.argv)
    tab_widget = QTabWidget()

    tab_widget.addTab(AFRtools(), "AFR Tools")
    tab_widget.addTab(FGSM(), "FGSM Attack")
    tab_widget.addTab(CW(), "C&W Attack")
    tab_widget.addTab(PGD(), "PGD Attack")
    tab_widget.addTab(FaceImageAdder(), "Add Mask")

    tab_widget.setStyleSheet("""
        QTabWidget::tab-bar {
            alignment: center;
        }
        QTabBar::tab {
            background-color: #2E2157;
            color: white; 
            border: 2px solid #6932A2; 
            border-radius: 8px; 
            min-width: 150px; 
            padding: 8px;
        }
        QTabBar::tab:selected {
            background-color: #6932A2; 
        }
    """)

    tab_widget.setWindowTitle("AFR Tools")
    tab_widget.resize(880, 750)
    tab_widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()