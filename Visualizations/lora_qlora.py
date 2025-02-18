import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_matrix(shape, scale=1):
    """Generates a random weight matrix."""
    return np.random.randn(*shape) * scale

def decompose_lora(W, r):
    """Decomposes W into A and B using low-rank adaptation."""
    d = W.shape[0]
    A = np.random.randn(d, r) * 0.01  # Small random init
    B = np.random.randn(r, d) * 0.01
    delta_W = np.dot(A, B)  # Low-rank update
    return A, B, delta_W

def quantize_matrix(W):
    """Applies 4-bit Normal Float (NF4) quantization."""
    min_val, max_val = W.min(), W.max()
    scale = (max_val - min_val) / 15  # 4-bit (16 levels)
    quantized_W = np.round((W - min_val) / scale).astype(np.int8)
    return quantized_W, min_val, scale

def dequantize_matrix(quantized_W, min_val, scale):
    """Dequantizes the matrix back to floating point values."""
    return (quantized_W * scale) + min_val

def decompose_qlora(W, r):
    """Quantizes W and applies LoRA decomposition."""
    quantized_W, min_val, scale = quantize_matrix(W)
    A, B, delta_W = decompose_lora(W, r)
    return quantized_W, min_val, scale, A, B, delta_W

def plot_matrices(W, A, B, delta_W, W_updated, quantized_W=None):
    """Plots the weight matrices to visualize LoRA and QLoRA effects."""
    fig, axes = plt.subplots(1, 6 if quantized_W is not None else 5, figsize=(20, 4))
    matrices = [W, A, B, delta_W, W_updated]
    titles = ["Original Weights (W)", "A (d×r)", "B (r×d)", "ΔW = A × B", "Updated Weights (W')"]
    
    if quantized_W is not None:
        matrices.insert(1, quantized_W)
        titles.insert(1, "Quantized Weights (4-bit NF4)")
    
    for ax, matrix, title in zip(axes, matrices, titles):
        sns.heatmap(matrix, cmap="coolwarm", ax=ax, cbar=False)
        ax.set_title(title)
        ax.axis("off")
    
    st.pyplot(fig)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["LoRA", "QLoRA"])
    
    if page == "LoRA":
        st.title("🔬 LoRA (Low-Rank Adaptation) Demonstration")
        st.markdown("""
        This interactive visualization shows how **LoRA (Low-Rank Adaptation)** works in fine-tuning Large Language Models (LLMs).
        
        - Instead of updating the **full weight matrix W**, LoRA decomposes it into **two smaller matrices A and B**.
        - This **reduces memory usage** and **improves efficiency**.
        """)
        
        # User inputs
        d = st.slider("Select Model Dimension (d):", 10, 100, 50, 10)
        r = st.slider("Select Low-Rank Dimension (r):", 1, 20, 8, 1)
        
        # Generate original weight matrix
        W = generate_matrix((d, d))
        A, B, delta_W = decompose_lora(W, r)
        W_updated = W + delta_W
        
        # Display parameter count
        full_params = d * d
        lora_params = (d * r) + (r * d)
        st.write(f"**Original Model Parameters:** {full_params:,}")
        st.write(f"**LoRA Parameters (Trainable):** {lora_params:,} ({(lora_params / full_params) * 100:.2f}% of full model)")
        
        # Plot matrices
        plot_matrices(W, A, B, delta_W, W_updated)
    
    elif page == "QLoRA":
        st.title("🚀 QLoRA (Quantized LoRA) Demonstration")
        st.markdown("""
        **QLoRA** combines **4-bit quantization** with **LoRA** to fine-tune massive LLMs on smaller hardware.
        
        - First, the weights are quantized to **4-bit NF4 precision**.
        - A **LoRA adapter** is added on top of the quantized weights.
        - Only the **LoRA parameters** are trained, while the quantized model remains frozen.
        """)
        
        # User inputs
        d = st.slider("Select Model Dimension (d):", 10, 100, 50, 10)
        r = st.slider("Select Low-Rank Dimension (r):", 1, 20, 8, 1)
        
        # Generate original weight matrix
        W = generate_matrix((d, d))
        quantized_W, min_val, scale, A, B, delta_W = decompose_qlora(W, r)
        W_updated = dequantize_matrix(quantized_W, min_val, scale) + delta_W
        
        # Display parameter count
        full_params = d * d
        lora_params = (d * r) + (r * d)
        st.write(f"**Original Model Parameters:** {full_params:,}")
        st.write(f"**QLoRA Parameters (Trainable):** {lora_params:,} ({(lora_params / full_params) * 100:.2f}% of full model)")
        
        # Plot matrices including quantized weights
        plot_matrices(W, A, B, delta_W, W_updated, quantized_W)

if __name__ == "__main__":
    main()