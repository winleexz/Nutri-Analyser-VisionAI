import gradio as gr
import torch
import spaces  # For HF ZeroGPU dynamic pooling
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

# Configure ultra-efficient 4-bit quantization to fit safely under the 14GiB VRAM cap (based on Hugging Face's default ZeroGPU tier constraints cap)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model weights in highly-optimized 4-bit mode...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)
print("Initialization completely ready!")

# Tells Hugging Face's backend cluster manager, "Hey, when this specific click event function is called, seamlessly lease me an active Nvidia T4 hardware accelerator pipeline slice for up to 60 seconds to execute the generation matrix.
@spaces.GPU(duration=60)
def analyze_nutrition_images(image_a, image_b, dietary_goal):
    if image_a is None or image_b is None:
        return "⚠️ Please upload images for both Product A and Product B to proceed."

    # Prevent memory fragmentation leaks
    torch.cuda.empty_cache()

    img_a = Image.fromarray(image_a.astype('uint8'), 'RGB')
    img_b = Image.fromarray(image_b.astype('uint8'), 'RGB')

    prompt = f"""
    You are an expert consumer-health nutritionist in Singapore. Analyse the two uploaded images.
    
    CRITICAL BRAND IDENTIFICATION:
    1. Read the brand names directly off the packaging or labels. 
    2. Do NOT output the generic placeholders "Product A" or "Product B". Replace them everywhere with the true discovered brand names.

    MANDATORY 100G NORMALIZATION MATRIX INSTRUCTION:
    Your primary goal is to ensure BOTH items are compared on a completely identical baseline scale of exactly **Per 100g** (or **Per 100ml**).
    Convert ALL nutritional metrics line-by-line using this strict mathematical conversion:
    Calculated_Value_Per_100g = (Value_per_serving / Serving_Size_in_grams) * 100

    SIDE-BY-SIDE MATRIX LAYOUT REQUIREMENT:
    Render the final processed calculations in a side-by-side, horizontal Markdown table.
    
    | Nutrient Metric (Normalized per 100g/ml) | [Extracted Brand Name A] | [Extracted Brand Name B] |
    | :--- | :--- | :--- |
    | **Energy / Calories** | X kcal | Y kcal |
    | **Protein** | X g | Y g |
    | **Total Carbohydrates / Sugars** | X g | Y g |
    | **Total Fat** | X g | Y g |
    | **Sodium** | X mg | Y mg |

    User's Dietary Goal: {dietary_goal}
    
    Please provide your output exactly in this clean Markdown format:
    ### 📊 Label Resolution & Step-by-Step Math Workings
    * **[Brand Name A] Baseline Found:** [e.g., Stated per 135g serving]
    * **[Brand Name A] 100g Scaling Math:** [Explicitly show the calculation fraction for each metric line item]
    * **[Brand Name B] Baseline Found:** [Stated per 100g directly]
    * **[Brand Name B] 100g Scaling Math:** [Stated directly]

    ### 📊 Side-by-Side Comparison Table (Per 100g Baseline)
    [Insert the completed horizontal markdown comparison matrix table here]
    
    ### 🔍 Critical Health Insights
    * **[Brand Name A]:** [Nutritional critique based on the 100g normalized metric data]
    * **[Brand Name B]:** [Nutritional critique based on the 100g normalized metric data]
    
    ### 🏆 The Verdict
    [Provide a definitive recommendation on which product fits the goal: "{dietary_goal}" better and why, using the calculated numbers to substantiate your proof.]
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=[img_a, img_b],
            padding=True,
            return_tensors="pt"
        )
        
        # BitsAndBytes handles device allocation dynamically. We map inputs to the current GPU allocation slice safely.
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1048,
                temperature=0.0,
                do_sample=False
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        return output_text

    except Exception as e:
        return f"⚠️ An error occurred during image processing: {str(e)}"
        
    finally:
        # Explicitly scrub VRAM allocations back to the host system immediately after execution loops - empty the memory
        torch.cuda.empty_cache()

# 3. Interface Layout
with gr.Blocks() as demo:
    gr.Markdown("# 🥛 Nutri-Analyser Vision AI")
    gr.Markdown("Snap photos of two grocery items on your smartphone to instantly run a normalied data comparison.")
    
    with gr.Row():
        with gr.Column():
            image_a_input = gr.Image(label="📸 Take Photo / Upload Product A", type="numpy")
        with gr.Column():
            image_b_input = gr.Image(label="📸 Take Photo / Upload Product B", type="numpy")
            
    with gr.Row():
        goal_input = gr.Dropdown(
            choices=[
                "Muscle Building (High Protein)", 
                "Weight Loss (Low Calorie)", 
                "Diabetic Friendly (Low Sugar)", 
                "General Clean Eating"
            ],
            label="What is your primary health goal?",
            value="Muscle Building (High Protein)"
        )
        
    submit_btn = gr.Button("⚡ Run Vision Comparison", variant="primary")
    output_display = gr.Markdown(label="Analysis Results")
    
    submit_btn.click(
        fn=analyze_nutrition_images, 
        inputs=[image_a_input, image_b_input, goal_input], 
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())