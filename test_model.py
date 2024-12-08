import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.saving import register_keras_serializable

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

class SignatureDecayModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SignatureDecayModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class SignatureVerificationApp:
    def __init__(self, master):
        self.master = master
        master.title("Signature Verification")
        master.geometry("800x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignatureDecayModel()
        self.model.load_state_dict(torch.load('model/signature_verification_model.pt', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.train_dirs = sorted(os.listdir('train'))
        self.signature_data = self._load_signature_data()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.create_widgets()

    def _load_signature_data(self):
        signature_data = {}
        for cls in self.train_dirs:
            person_id, signature_type = cls.split('-')[0], 'genuine' if cls.split('-')[-1] != 'f' in cls else 'fraudulent'
            if person_id not in signature_data:
                signature_data[person_id] = {'genuine': [], 'fraudulent': []}
            signature_data[person_id][signature_type].append(cls)
        return signature_data

    def create_widgets(self):
        self.image_frame = tk.Frame(self.master, width=800, height=700)
        self.image_frame.pack(pady=20)

        self.image_label = tk.Label(self.image_frame, text="Upload an image", font=("Arial", 16), bg="white")
        self.image_label.pack()

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=10)

        self.upload_button = tk.Button(self.button_frame, text="Upload Signature", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.verify_button = tk.Button(self.button_frame, text="Verify Signature", command=self.verify_signature, state=tk.DISABLED)
        self.verify_button.pack(side=tk.LEFT, padx=10)

        # Result frame
        self.result_frame = tk.Frame(self.master)
        self.result_frame.pack(pady=10)

        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 14))
        self.result_label.pack()

        # Confidence frame
        self.confidence_frame = tk.Frame(self.master)
        self.confidence_frame.pack(pady=10)

        self.confidence_text = tk.Text(self.confidence_frame, height=10, width=70)
        self.confidence_text.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Signature Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            image = Image.open(file_path)

            max_size = (600, 600)
            image.thumbnail(max_size)

            photo = ImageTk.PhotoImage(image)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

            self.master.geometry(f"{image.width + 200}x{image.height + 300}")

            self.image_path = file_path
            self.verify_button.config(state=tk.NORMAL)

    def verify_signature(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = probabilities.topk(1)

            labels = ['Genuine', 'Fraudulent']
            result = labels[top_class[0][0].item()]
            confidence = top_prob[0][0].item() * 100

            if 'f' in result:
                result = "Fraudulent"
            else:
                result = "Genuine"

            input_filename = os.path.basename(self.image_path)
            person_id = input_filename.split('-')[0] if '-' in input_filename else "Unknown"

            self.result_label.config(
                text=f"Person ID: {person_id}\nSignature: {result}\nConfidence: {confidence:.2f}%"
            )

            self.confidence_text.delete(1.0, tk.END)
            for i, label in enumerate(labels):
                self.confidence_text.insert(tk.END, f"{label}: {probabilities[0][i].item() * 100:.2f}%\n")

        except Exception as e:
            messagebox.showerror("Verification Error", str(e))

    def _load_keras_model_weights(self, model_path):
        try:
            keras_model = load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance})
            return keras_model
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")


def main():
    root = tk.Tk()
    app = SignatureVerificationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
