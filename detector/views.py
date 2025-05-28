from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.http import HttpResponse
from django.utils.timezone import localtime
import pytz
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer
import numpy as np
import cv2
import io
import base64
import threading
from .gradcam_utils import generate_grad_cam
from reportlab.lib.utils import ImageReader
from .preprocessing_utils import preprocess_xray_image

MODEL_PATH = os.path.join('detector', 'models', 'new', 'InceptionV3_TB_Detection.h5')  
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def landing_page(request):
    return render(request, 'landing.html')

def load_selected_model():
    model_path = os.path.join('detector', 'models', 'new', "InceptionV3_TB_Detection.h5")
    return tf.keras.models.load_model(model_path, compile=False)

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('xray'):
        image = request.FILES.get('xray')

        # Check file extension
        if not image.name.lower().endswith('.png'):
            return render(request, 'upload.html', {
                'error': "Only .png files are accepted."
            })

        # Save uploaded image
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_url = fs.url(filename)
        img_path = os.path.join("media", filename)

        img = cv2.imread(img_path)
        if img is None:
            return render(request, 'upload.html', {'error': "Failed to read image file."})
        
        # Preprocess the image (CLAHE + resize + normalize)
        try:
            img_for_pred, img_resized = preprocess_xray_image(img_path, size=299)
        except Exception as e:
            return render(request, 'upload.html', {'error': f"Preprocessing error: {str(e)}"})

        # --- Inference using SavedModel format ---
        #inception_model = tf.saved_model.load("detector/models/new/InceptionV3_TB_Detection")
        inception_model = tf.keras.models.load_model("detector/models/new/InceptionV3_TB_Detection.h5", compile=False)
        #tfsml = TFSMLayer("detector/models/new/InceptionV3_TB_Detection", call_endpoint="serving_default") # tfsml works like a Keras Layer for inference

        #img_for_pred = preprocess_xray_image(img_path, size=299)
        #pred_incep = inception_model.serve(img_for_pred).numpy()[0][0]
        pred_incep = inception_model.predict(img_for_pred)[0][0]

       
        model = "InceptionV3"

        if pred_incep > 0.5:
            result_incep = "TB-Positive"
            confidence = pred_incep
            confidence_label = "Confidence (Probability of TB)"
        else:
            result_incep = "TB-Negative"
            confidence = 1 - pred_incep
            confidence_label = "Confidence (Probability of TB-Negative)"


        result = result_incep
        
        # Grad-CAM using .h5 InceptionV3
        last_conv = "mixed10"

        # Generate Grad-CAM always
        """ heatmap = None
        try:
            grad_model = tf.keras.models.load_model('detector/models/new/InceptionV3_TB_Detection.h5', compile=False)

            for layer in grad_model.layers:
                print(layer.name)

            heatmap = generate_grad_cam(
                model=grad_model,
                image_path=img_path,
                size=299,
                last_conv_layer_name=last_conv
            )
           
        except Exception as e:
            print("Grad-CAM generation failed:", e) """
        
        # Only generate Grad-CAM if TB-positive
        heatmap = None
        if result == "TB-Positive":  
            try:
                grad_model = tf.keras.models.load_model('detector/models/new/InceptionV3_TB_Detection.h5', compile=False)
                for layer in grad_model.layers:
                    print(layer.name)
                preprocessed_temp = f"media/preprocessed_{filename}"
                cv2.imwrite(preprocessed_temp, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                heatmap = generate_grad_cam(
                    model=grad_model,
                    image_path=preprocessed_temp,
                    size=299,
                    last_conv_layer_name=last_conv
                )

            except Exception as e:
                print("Grad-CAM generation failed:", e)
        

        # After determining result and confidence
        show_gradcam = result == "TB-Positive"
        show_confidence = result == "TB-Positive"

        # Save to session
        request.session['result'] = result
        request.session['confidence'] = f"{confidence:.2f}"
        request.session['confidence_label'] = confidence_label
        request.session['model_used'] = model
        request.session['file_url'] = file_url
        request.session['uploaded_filename'] = os.path.basename(file_url)
        request.session['heatmap'] = heatmap if show_gradcam else None



        print("Grad-CAM heatmap:", type(heatmap), "length:", len(heatmap) if heatmap is not None else "None")


        # Render results
        return render(request, 'result.html', {
            'file_url': file_url,
            'uploaded_filename': filename,
            'result': result,
            'confidence': f"{confidence:.2f}",
            'confidence_label': confidence_label,
            'model': model,
            'heatmap': heatmap if show_gradcam else None,
            'show_gradcam': show_gradcam,
            'show_confidence': show_confidence,
        })

    return render(request, 'upload.html')


def result_page(request):
    return render(request, 'result.html')

def save_result_pdf(request):
    if request.method == "POST":
        # Get session or POST data
        diagnosis = request.session.get('result', 'N/A')
        confidence = request.session.get('confidence', 'N/A')
        model_used = request.session.get('model_used', 'N/A')
        file_url = request.session.get('file_url', None)
        heatmap_base64 = request.session.get('heatmap', None)
        filename_only = request.session.get('uploaded_filename', 'N/A')
        confidence_label = request.session.get('confidence_label', None)


        # Get current date and time
        manila = pytz.timezone("Asia/Manila")
        timestamp = localtime().astimezone(manila).strftime('%B %d, %Y at %I:%M %p')  # 12-hour with AM/PM
        filename_date = localtime().astimezone(manila).strftime('%Y-%m-%d_%H%M%S')   # 24-hour for filename
        filename = f"TB_Report_{filename_date}.pdf"

        # Start in-memory PDF
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont("Helvetica", 12)

        p.drawString(50, 750, f"Diagnosis: {diagnosis}")
        y_start = 735
        if confidence_label and confidence:
            p.drawString(50, y_start, f"{confidence_label}: {confidence}")
            y_start -= 15
        p.drawString(50, y_start, f"Generated on: {timestamp}")
        p.drawString(50, y_start-15, f"Image Filename: {filename_only}")


        y = 640
        if file_url:
            img_path = os.path.join("media", os.path.basename(file_url))
            if os.path.exists(img_path):
                p.drawString(50, y, "Original Image:")
                p.drawImage(img_path, 50, y-220, width=200, height=200)
                y -= 240

        if heatmap_base64:
            heatmap_data = base64.b64decode(heatmap_base64)
            heatmap_image = io.BytesIO(heatmap_data)
            heatmap_reader = ImageReader(heatmap_image)   # <-- Wrap BytesIO in ImageReader
            p.drawString(300, y+240, "Grad-CAM:")
            p.drawImage(heatmap_reader, 300, y+20, width=200, height=200)

        p.showPage()
        p.save()
        buffer.seek(0)

        # Save to local disk
        local_path = os.path.join("media", filename)
        with open(local_path, "wb") as f:
            f.write(buffer.getbuffer())

        # Send as downloadable response
        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    
    return HttpResponse("Invalid request", status=400)