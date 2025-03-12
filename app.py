import os
import subprocess
import sys
import re
import numpy as np
from PIL import Image
import gradio as gr
import requests
import json
from dotenv import load_dotenv

# Attempt to install pytesseract if not found
try:
    import pytesseract
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytesseract'])
    import pytesseract

# AFTER importing pytesseract, then set the path
try:
    # First try the default path
    if os.path.exists('/usr/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    # Try to find it on the PATH
    else:
        tesseract_path = subprocess.check_output(['which', 'tesseract']).decode().strip()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
except:
    # If all else fails, try the default installation path
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Load environment variables
load_dotenv()

# Mistral API Key
MISTRAL_API_KEY = "GlrVCBWyvTYjWGKl5jqtK4K41uWWJ79F"

# Import and configure Mistral API
def analyze_ingredients_with_mistral(ingredients_list, health_conditions=None):
    """
    Use Mistral AI to analyze ingredients and provide health insights.
    """
    if not ingredients_list:
        return "No ingredients detected or provided."

    # Prepare the list of ingredients for the prompt
    ingredients_text = ", ".join(ingredients_list)

    # Create a prompt for Mistral
    if health_conditions and health_conditions.strip():
        prompt = f"""
        Analyze the following food ingredients for a person with these health conditions: {health_conditions}
        Ingredients: {ingredients_text}
        For each ingredient:
        1. Provide its potential health benefits
        2. Identify any potential risks
        3. Note if it may affect the specified health conditions
        Then provide an overall assessment of the product's suitability for someone with the specified health conditions.
        Format your response in markdown with clear headings and sections.
        """
    else:
        prompt = f"""
        Analyze the following food ingredients:
        Ingredients: {ingredients_text}
        For each ingredient:
        1. Provide its potential health benefits
        2. Identify any potential risks or common allergens associated with it
        Then provide an overall assessment of the product's general health profile.
        Format your response in markdown with clear headings and sections.
        """

    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-small",  # Or another suitable model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            analysis = response.json()['choices'][0]['message']['content']
        else:
            return dummy_analyze(ingredients_list, health_conditions) + f"\n\n(Using fallback analysis: Mistral API Error - {response.status_code} - {response.text})"

        # Add disclaimer
        disclaimer = """
        ## Disclaimer
        This analysis is provided for informational purposes only and should not replace professional medical advice.
        Always consult with a healthcare provider regarding dietary restrictions, allergies, or health conditions.
        """

        return analysis + disclaimer

    except Exception as e:
        # Fallback to basic analysis if API call fails
        return dummy_analyze(ingredients_list, health_conditions) + f"\n\n(Using fallback analysis: {str(e)})"


# Dummy analysis function for when API is not available
def dummy_analyze(ingredients_list, health_conditions=None):
    ingredients_text = ", ".join(ingredients_list)

    report = f"""
    # Ingredient Analysis Report
    ## Detected Ingredients
    {", ".join([i.title() for i in ingredients_list])}
    ## Overview
    This is a simulated analysis since the Mistral API call failed. In the actual application,
    the ingredients would be analyzed by Mistral for their health implications.
    ## Health Considerations
    """

    if health_conditions:
        report += f"""
        The analysis would specifically consider these health concerns: {health_conditions}
        """
    else:
        report += """
        No specific health concerns were provided, so a general analysis would be performed.
        """

    report += """
    ## Disclaimer
    This analysis is provided for informational purposes only and should not replace professional medical advice.
    Always consult with a healthcare provider regarding dietary restrictions, allergies, or health conditions.
    """

    return report

# Function to extract text from images using OCR
def extract_text_from_image(image):
    try:
        if image is None:
            return "No image captured. Please try again."

        # Verify Tesseract executable is accessible
        try:
            subprocess.run([pytesseract.pytesseract.tesseract_cmd, "--version"],
                          check=True, capture_output=True, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Tesseract OCR is not installed or not properly configured. Please check installation."

        # Import necessary libraries
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance

        # First try with enhanced PIL image processing
        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(2.0)  # Increase contrast
        
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(2.0)  # Increase sharpness
        
        # Try OCR on enhanced image first
        custom_config = r'--oem 3 --psm 4 -l eng --dpi 300'
        text = pytesseract.image_to_string(enhanced_image, config=custom_config)
        
        # If that doesn't work well, try more advanced OpenCV preprocessing
        if not text.strip() or len(text) < 10:
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise image
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Try multiple PSM modes for different text layouts
            psm_modes = [6, 4, 3, 1]  # Single block, Multiple blocks, Auto, Auto with OSD
            best_text = ""
            
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm} -l eng --dpi 300'
                temp_text = pytesseract.image_to_string(Image.fromarray(denoised), config=config)
                
                # Keep the result with the most characters
                if len(temp_text) > len(best_text):
                    best_text = temp_text
            
            # If OpenCV approach yielded better results, use it
            if len(best_text) > len(text):
                text = best_text
                
            # If still no good results, try alternate preprocessing
            if not text.strip() or len(text) < 10:
                # Try different preprocessing: Otsu's thresholding
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Apply dilation to connect nearby text
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(otsu, kernel, iterations=1)
                
                # Try OCR again
                otsu_text = pytesseract.image_to_string(
                    Image.fromarray(dilated), 
                    config=r'--oem 3 --psm 1 -l eng --dpi 300'
                )
                
                if len(otsu_text) > len(text):
                    text = otsu_text

        # Final text cleaning
        text = text.replace('\n\n', '\n')  # Remove excess line breaks
        text = re.sub(r'[^\w\s,;:.()\n-]', '', text)  # Remove non-alphanumeric chars except punctuation
        
        if not text.strip():
            return "No text could be extracted. Ensure image is clear and readable."

        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to parse ingredients from text
def parse_ingredients(text):
    if not text:
        return []

    # Clean up the text
    text = re.sub(r'^ingredients:?\s*', '', text.lower(), flags=re.IGNORECASE)

    # Remove common OCR errors and extraneous characters
    text = re.sub(r'[|\\/@#$%^&*()_+=]', '', text)

    # Replace common OCR errors
    text = re.sub(r'\bngredients\b', 'ingredients', text)

    # Handle common OCR misreads
    replacements = {
        '0': 'o', 'l': 'i', '1': 'i',
        '5': 's', '8': 'b', 'Q': 'g',
    }

    for error, correction in replacements.items():
        text = text.replace(error, correction)

    # Split by common ingredient separators
    ingredients = re.split(r',|;|\n', text)

    # Clean up each ingredient
    cleaned_ingredients = []
    for i in ingredients:
        i = i.strip().lower()
        if i and len(i) > 1:  # Ignore single characters which are likely OCR errors
            cleaned_ingredients.append(i)

    return cleaned_ingredients


# Function to process input based on method (camera, upload, or manual entry)
def process_input(input_method, text_input, camera_input, upload_input, health_conditions):
    if input_method == "Camera":
        if camera_input is not None:
            extracted_text = extract_text_from_image(camera_input)
            # If OCR fails, inform the user they can try manual entry
            if "Error" in extracted_text or "No text could be extracted" in extracted_text:
                return extracted_text + "\n\nPlease try using the 'Manual Entry' option instead."

            ingredients = parse_ingredients(extracted_text)
            return analyze_ingredients_with_mistral(ingredients, health_conditions)
        else:
            return "No camera image captured. Please try again."

    elif input_method == "Image Upload":
        if upload_input is not None:
            extracted_text = extract_text_from_image(upload_input)
            # If OCR fails, inform the user they can try manual entry
            if "Error" in extracted_text or "No text could be extracted" in extracted_text:
                return extracted_text + "\n\nPlease try using the 'Manual Entry' option instead."

            ingredients = parse_ingredients(extracted_text)
            return analyze_ingredients_with_mistral(ingredients, health_conditions)
        else:
            return "No image uploaded. Please try again."

    elif input_method == "Manual Entry":
        if text_input and text_input.strip():
            ingredients = parse_ingredients(text_input)
            return analyze_ingredients_with_mistral(ingredients, health_conditions)
        else:
            return "No ingredients entered. Please try again."

    return "Please provide input using one of the available methods."

# Create the Gradio interface
with gr.Blocks(title="AI Ingredient Scanner") as app:
    gr.Markdown("# AI Ingredient Scanner")
    gr.Markdown("Scan product ingredients and analyze them for health benefits, risks, and potential allergens.")

    with gr.Row():
        with gr.Column():
            input_method = gr.Radio(
                ["Camera", "Image Upload", "Manual Entry"],
                label="Input Method",
                value="Camera"
            )

            # Camera input
            camera_input = gr.Image(label="Capture ingredients with camera", type="pil", visible=True)

            # Image upload
            upload_input = gr.Image(label="Upload image of ingredients label", type="pil", visible=False)

            # Text input
            text_input = gr.Textbox(
                label="Enter ingredients list (comma separated)",
                placeholder="milk, sugar, flour, eggs, vanilla extract",
                lines=3,
                visible=False
            )

            # Health conditions input - now optional and more flexible
            health_conditions = gr.Textbox(
                label="Enter your health concerns (optional)",
                placeholder="diabetes, high blood pressure, peanut allergy, etc.",
                lines=2,
                info="The AI will automatically analyze ingredients for these conditions"
            )

            analyze_button = gr.Button("Analyze Ingredients")

        with gr.Column():
            output = gr.Markdown(label="Analysis Results")
            extracted_text_output = gr.Textbox(label="Extracted Text (for verification)", lines=3)

    # Show/hide inputs based on selection
    def update_visible_inputs(choice):
        return {
            upload_input: gr.update(visible=(choice == "Image Upload")),
            camera_input: gr.update(visible=(choice == "Camera")),
            text_input: gr.update(visible=(choice == "Manual Entry"))
        }

    input_method.change(update_visible_inputs, input_method, [upload_input, camera_input, text_input])

    # Extract and display the raw text (for verification purposes)
    def show_extracted_text(input_method, text_input, camera_input, upload_input):
        if input_method == "Camera" and camera_input is not None:
            return extract_text_from_image(camera_input)
        elif input_method == "Image Upload" and upload_input is not None:
            return extract_text_from_image(upload_input)
        elif input_method == "Manual Entry":
            return text_input
        return "No input detected"

    # Set up event handlers
    analyze_button.click(
        fn=process_input,
        inputs=[input_method, text_input, camera_input, upload_input, health_conditions],
        outputs=output
    )

    analyze_button.click(
        fn=show_extracted_text,
        inputs=[input_method, text_input, camera_input, upload_input],
        outputs=extracted_text_output
    )

    gr.Markdown("### How to use")
    gr.Markdown("""
    1. Choose your input method (Camera, Image Upload, or Manual Entry)
    2. Take a photo of the ingredients label or enter ingredients manually
    3. Optionally enter your health concerns
    4. Click "Analyze Ingredients" to get your personalized analysis
    The AI will automatically analyze the ingredients, their health implications, and their potential impact on your specific health concerns.
    """)

    gr.Markdown("### Examples of what you can ask")
    gr.Markdown("""
    The system can handle a wide range of health concerns, such as:
    - General health goals: "trying to reduce sugar intake" or "watching sodium levels"
    - Medical conditions: "diabetes" or "hypertension"
    - Allergies: "peanut allergy" or "shellfish allergy"
    - Dietary restrictions: "vegetarian" or "gluten-free diet"
    - Multiple conditions: "diabetes, high cholesterol, and lactose intolerance"
    The AI will tailor its analysis to your specific needs.
    """)

    gr.Markdown("### Tips for best results")
    gr.Markdown("""
    - Hold the camera steady and ensure good lighting
    - Focus directly on the ingredients list
    - Make sure the text is clear and readable
    - Be specific about your health concerns for more targeted analysis
    """)

    gr.Markdown("### Disclaimer")
    gr.Markdown("""
    This tool is for informational purposes only and should not replace professional medical advice.
    Always consult with a healthcare provider regarding dietary restrictions, allergies, or health conditions.
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()