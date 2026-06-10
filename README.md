# 🛒 Nutri-Analyser AI for Smarter Grocery Shopping
While global AI labs are racing to release larger, resource-heavy models capable of generating everything from text to video, the "Small Models, Big Adventures" hackathon challenges us to do the opposite. The test here lies in thinking small and practical – using lightweight models to solve real-world friction points efficiently without massive cloud infrastructure.

## 🔍 Problem & Motivation
I built **Nutri-Analyser AI** to solve a frustrating problem I often face during my grocery trip: decoding confusing food labels. It can be exhausting to stand in a supermarket aisle comparing two similar products, only to find one boasts "Low Fat" while hiding massive amounts of sodium, while the other claims "0% Sugar" but is packed with chemical additives. Because brands use inconsistent nutritional baselines and non-standardised serving sizes, making an optimal choice on the spot becomes a tedious math puzzle instead of a quick decision.

## 🧪 Real-World Supermarket Testing
I put the app to the test during my recent supermarket trip to see how it handled real-world conditions. I snapped photos of two different yogurt tubs, and the lightweight Qwen-7B Vision model instantly extracted the raw label text, calculated the math to normalise their mismatched serving sizes, and evaluated them against my pre-defined health goal. 

Within seconds, a side-by-side comparison matrix and an objective verdict popped up on my phone. I confidently grabbed the winning tub and moved on to the next item on my shopping list 😊

## 🛠️ Tech Stack
1. Frontend UI: Gradio 
2. Core Model Architecture: Qwen2.5-VL-7B-Instruct, a Vision-Language Model
3. Quantisation Framework: NF4/4-bit double quantization tracking
4. Hosting Infrastructure: Hugging Face Spaces

## 🏗️ System Architecture
The Nutri-Analyser application uses a decoupled, resource-optimised client-server pipeline designed to execute heavy multi-modal inference within a low-memory constraints container environment.

![architecture](https://cdn-uploads.huggingface.co/production/uploads/61350aa938a697fa5a1533a2/L4RnqmNewWLHSon8LM6K9.jpeg)

## ⚙️ How It Works Technically
1. **4-Bit Memory Compression**: A raw 7B model running at FP16 precision requires ~17GB–18GB of VRAM to process image tokens and text generation, easily surpassing standard free cluster caps. By enforcing an NF4 quantization configuration, the model footprint is compressed down to a highly efficient ~5.5GB, leaving plenty of memory headroom on an Nvidia T4

2. **Structured Chain-of-Thought Prompting**: the system prompt initiates an explicit, multi-stage logical pipeline including normalising the nutritional labels

3. **Deterministic Evaluation**: Setting temperature=0.0 and do_sample=False forces strict greedy token selection. This shifts the model from a "creative writer" into a deterministic calculator, ensuring accurate fractional arithmetic and table structuring

## 💡 Challenges & Learnings
1. **Defeating the VRAM Cap**: Moving away from traditional dedicated GPU instances meant dealing with tight 14.74GiB memory constraints. Loading and passing the model weights to quantised CUDA layers was a critical exercise in memory management
2. **Mitigating Memory Leaks**: Residual cache blocks can easily trigger an out-of-memory crash on second or third scan. Implementing resource cleanups with torch.cuda.empty_cache() was essential to keep the app robust for continuous multi-product scanning
3. **Prompt Engineering for Precision Math**: Language models are notoriously prone to hallucinating arithmetic. The biggest breakthrough was forcing the model to print its literal step-by-step fractional equations before outputting the final visual table layout – holding the model's logic accountable and dramatically boosting accuracy


***Check out the app and blog below to see how the orchestration works under the hood 👇***
1. https://huggingface.co/spaces/Winnielee/Nutri-Analyser-AI
2. https://huggingface.co/spaces/Winnielee/Nutri-Analyser-AI/tree/main
</table>


#BuildSmallHackathon #BackyardAI #HuggingFace
