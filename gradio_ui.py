import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from data_preprocessing import IMAGE_SIZE

# Load the trained model
model_path = "training_results/bone_cancer_hqc_model_epoch_100.keras"
if not os.path.exists(model_path):
    # Fallback to any available model
    model_files = [f for f in os.listdir("training_results") if f.endswith(".keras")]
    if model_files:
        model_path = os.path.join("training_results", model_files[-1])
    else:
        model_path = None

if model_path and os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
else:
    print("No trained model found. Please run train_evaluate.py first.")
    model = None

# Class names
class_names = ["cancerous", "non_cancerous"]

def predict_bone_cancer(image):
    """
    Predict bone cancer from X-ray image and return confidence scores.
    """
    if model is None:
        return "Error: No trained model available. Please train the model first.", {}
    
    try:
        # Preprocess the image
        if image is None:
            return "Error: No image provided.", {}
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize image to match model input
        image = image.resize(IMAGE_SIZE)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_batch, verbose=0)
        
        # Get confidence scores
        confidence_scores = predictions[0]
        
        # Create confidence dictionary
        confidence_dict = {}
        for i, class_name in enumerate(class_names):
            confidence_dict[class_name] = float(confidence_scores[i])
        
        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = class_names[predicted_class_idx]
        confidence = confidence_scores[predicted_class_idx]
        
        # Create result message
        result_message = f"Prediction: {predicted_class.upper()}\nConfidence: {confidence:.2%}"
        
        return result_message, confidence_dict
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", {}

def create_interface():
    """
    Create and configure the Gradio interface.
    """
    # Custom CSS for styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        max-width: 900px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .prediction-output {
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=css, title="Bone Cancer Detection System") as interface:
        gr.HTML("""
        <div class="header">
            <h1>🦴 Bone Cancer Detection System</h1>
            <p>AI-Powered X-ray Analysis with Quantum-Enhanced CNN</p>
            <p><em>Upload an X-ray image to detect potential bone cancer</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Upload X-ray Image")
                image_input = gr.Image(
                    type="pil",
                    label="X-ray Image",
                    height=400
                )
                
                predict_button = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📊 Analysis Results")
                
                prediction_output = gr.Textbox(
                    label="Prediction Result",
                    lines=3,
                    max_lines=5
                )
                
                confidence_plot = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=2
                )
        
        # Model information section
        with gr.Row():
            gr.Markdown("""
            ### ℹ️ Model Information
            - **Architecture**: Hybrid Quantum-Classical CNN
            - **Classes**: Cancerous, Non-cancerous
            - **Input Size**: 128x128 pixels
            - **Quantum Enhancement**: Quantum-inspired processing layer
            
            ### ⚠️ Important Disclaimer
            This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.
            """)
        
        # Set up the prediction function
        predict_button.click(
            fn=predict_bone_cancer,
            inputs=[image_input],
            outputs=[prediction_output, confidence_plot]
        )
        
        # Add examples if available
        example_images = []
        if os.path.exists("bone cancer detection.v1i.multiclass/test"):
            test_dir = "bone cancer detection.v1i.multiclass/test"
            image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            example_images = [os.path.join(test_dir, f) for f in image_files[:3]]
        
        if example_images:
            gr.Examples(
                examples=[[img] for img in example_images],
                inputs=[image_input],
                outputs=[prediction_output, confidence_plot],
                fn=predict_bone_cancer,
                cache_examples=True
            )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    print("Starting Bone Cancer Detection System...")
    print("Model status:", "Loaded" if model is not None else "Not available")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

