"""
Interactive Web Interface for Language Model Text Generation
Academic Project: Gated Linear Attention Transformer with Direct Preference Optimization

Authors:
- Chetan Krishna Kodeboyina (NYU ID: ck3399)
- Bryce Miranda (NYU NetID: bm3986)

Institution: New York University
"""
import gradio as gr
import torch
import tiktoken
import os
from gla_model import GLATransformer, generate_text


class ModelInterface:
    def __init__(self):
        self.model = None
        self.enc = None
        self.device = None
        self.model_config = {}
        
    def load_model(self, checkpoint_path, embed_size, n_blocks, n_heads, block_size, use_pre_norm):
        """Load model from checkpoint."""
        try:
            if not os.path.exists(checkpoint_path):
                return f"Error: Checkpoint file not found at {checkpoint_path}"
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            
            # Initialize tokenizer
            enc = tiktoken.get_encoding("gpt2")
            vocab_size = enc.n_vocab
            self.enc = enc
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if "policy_model_state_dict" in checkpoint:
                state_dict = checkpoint["policy_model_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Create model
            model = GLATransformer(
                vocab_size=vocab_size,
                d_model=embed_size,
                n_heads=n_heads,
                n_blocks=n_blocks,
                block_size=block_size,
                use_pre_norm=use_pre_norm,
            ).to(device)
            
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.model_config = {
                "embed_size": embed_size,
                "n_blocks": n_blocks,
                "n_heads": n_heads,
                "block_size": block_size,
            }
            
            param_count = sum(p.numel() for p in model.parameters())
            return f"Model loaded successfully.\nDevice: {device}\nTotal Parameters: {param_count:,}\nModel Architecture: {n_blocks} blocks, {n_heads} heads, d_model={embed_size}"
            
        except Exception as e:
            return f"Error loading model: {str(e)}\n\nPlease verify:\n- Checkpoint path is correct\n- Model configuration matches training parameters"
    
    def generate(self, prompt, max_new_tokens, top_p, temperature, use_greedy):
        """Generate text from prompt."""
        if self.model is None or self.enc is None:
            return "Error: Model not loaded. Please load a model checkpoint first."
        
        try:
            top_p_val = None if use_greedy else top_p
            
            generated_text = generate_text(
                model=self.model,
                enc=self.enc,
                init_text=prompt,
                max_new_tokens=max_new_tokens,
                device=self.device,
                top_p=top_p_val,
                temperature=temperature,
            )
            
            return generated_text
            
        except Exception as e:
            return f"Error during text generation: {str(e)}"


# Create interface instance
interface = ModelInterface()


def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="GLA Transformer with DPO - Text Generation Interface", theme=gr.themes.Default()) as demo:
        # Header
        gr.Markdown("""
        # Language Model Text Generation Interface
        
        **Gated Linear Attention Transformer with Direct Preference Optimization**
        
        *Academic Project - New York University*  
        *Authors: Chetan Krishna Kodeboyina (NYU ID: ck3399), Bryce Miranda (NYU NetID: bm3986)*
        
        ---
        """)
        
        # Model Information
        with gr.Accordion("Model Architecture Information", open=False):
            gr.Markdown("""
            **Architecture Details:**
            - **Attention Mechanism**: Gated Linear Attention (GLA) - linear complexity attention with learned gates
            - **Training Method**: Direct Preference Optimization (DPO) for alignment
            - **Normalization**: RMSNorm with pre-normalization
            - **Feed-Forward**: SwiGLU activation
            - **Tokenization**: GPT-2 tokenizer (tiktoken)
            
            **Key Features:**
            - Linear complexity attention (O(n) vs O(n²) for standard attention)
            - Efficient KV caching for generation
            - Preference-based fine-tuning without separate reward model
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")
                
                checkpoint_path = gr.Textbox(
                    label="Checkpoint File Path",
                    placeholder="dpo_model_gated_linear_attention.pt",
                    value="dpo_model_gated_linear_attention.pt",
                    info="Path to the trained model checkpoint file"
                )
                
                with gr.Row():
                    embed_size = gr.Number(
                        label="Embedding Dimension (d_model)", 
                        value=1024, 
                        precision=0,
                        info="Model dimension"
                    )
                    n_blocks = gr.Number(
                        label="Number of Transformer Blocks", 
                        value=6, 
                        precision=0,
                        info="Depth of the transformer"
                    )
                
                with gr.Row():
                    n_heads = gr.Number(
                        label="Number of Attention Heads", 
                        value=8, 
                        precision=0,
                        info="Multi-head attention heads"
                    )
                    block_size = gr.Number(
                        label="Maximum Sequence Length", 
                        value=256, 
                        precision=0,
                        info="Context window size"
                    )
                
                use_pre_norm = gr.Checkbox(
                    label="Use Pre-Normalization", 
                    value=True,
                    info="Apply normalization before attention/FFN (recommended for GLA)"
                )
                
                load_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Textbox(
                    label="Model Status", 
                    interactive=False,
                    lines=4
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Text Generation")
                
                prompt = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your text prompt here...",
                    value="Once upon a time",
                    lines=4,
                    info="The initial text sequence to continue"
                )
                
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        label="Maximum New Tokens",
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        info="Maximum number of tokens to generate"
                    )
                    temperature = gr.Slider(
                        label="Sampling Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        info="Controls randomness: higher = more diverse, lower = more deterministic"
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        label="Top-P (Nucleus Sampling)",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        info="Cumulative probability threshold for nucleus sampling"
                    )
                    use_greedy = gr.Checkbox(
                        label="Greedy Decoding", 
                        value=False,
                        info="Select highest probability token (overrides Top-P)"
                    )
                
                generate_btn = gr.Button("Generate Text", variant="primary")
                
                output = gr.Textbox(
                    label="Generated Output",
                    lines=12,
                    interactive=False,
                    info="Complete generated text sequence"
                )
        
        # Example prompts
        gr.Markdown("### Example Prompts")
        examples = gr.Examples(
            examples=[
                ["Once upon a time"],
                ["The cat sat on"],
                ["In a faraway land"],
                ["The scientist discovered"],
                ["Once there was a brave knight"],
            ],
            inputs=prompt,
            label="Click to load example prompts"
        )
        
        # Technical notes
        with gr.Accordion("Technical Notes and Parameters", open=False):
            gr.Markdown("""
            **Parameter Guidelines:**
            
            - **Model Configuration**: The architecture parameters (embedding dimension, number of blocks, etc.) must exactly match the configuration used during training. Mismatched parameters will result in loading errors.
            
            - **Temperature**: Controls the randomness of the output distribution. 
              - Low values (0.1-0.5): More focused, deterministic outputs
              - Medium values (0.7-1.0): Balanced creativity and coherence
              - High values (1.5-2.0): More diverse, potentially less coherent outputs
            
            - **Top-P (Nucleus Sampling)**: Considers only tokens whose cumulative probability mass reaches the threshold. 
              - Lower values (0.5-0.8): More focused generation
              - Higher values (0.9-0.95): More diverse generation
            
            - **Greedy Decoding**: Always selects the token with highest probability. Results in deterministic but potentially repetitive outputs.
            
            **Generation Process:**
            The model uses autoregressive generation with KV caching for efficiency. Each token is generated based on the previous context, maintaining a fixed-size hidden state through the GLA mechanism.
            """)
        
        # Event handlers
        load_btn.click(
            fn=interface.load_model,
            inputs=[checkpoint_path, embed_size, n_blocks, n_heads, block_size, use_pre_norm],
            outputs=load_status
        )
        
        generate_btn.click(
            fn=interface.generate,
            inputs=[prompt, max_new_tokens, top_p, temperature, use_greedy],
            outputs=output
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="localhost", server_port=7860)

