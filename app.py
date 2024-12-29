import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

class ImprovedGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels*8) for _ in range(12)
        ])
        self.attention1 = AttentionBlock(base_channels*8)
        self.attention2 = AttentionBlock(base_channels*4)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            if i == 5:
                x = self.attention1(x)
        
        x = self.up1(x)
        x = self.attention2(x + x3)
        x = self.up2(x)
        x = x + x2
        x = self.up3(x)
        x = x + x1
        return self.final(x)

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedGenerator().to(device)
    model.load_state_dict(torch.load('checkpoints/final_generator.pt', map_location=device))
    model.eval()
    return model, device

def process_sketch(sketch, size=256):
    # Define image transformations
    transform = A.Compose([
        A.Resize(size, size),
        ToTensorV2(),
    ])
    
    # Convert PIL Image to numpy array
    sketch_np = np.array(sketch.convert('L'))
    
    # Apply transformations
    transformed = transform(image=sketch_np)
    processed_sketch = transformed['image'].float() / 127.5 - 1
    return processed_sketch.unsqueeze(0)

def generate_image(model, sketch_tensor, device):
    with torch.no_grad():
        generated = model(sketch_tensor.to(device))
        # Convert from [-1, 1] to [0, 1] range
        generated = (generated * 0.5 + 0.5).cpu()
        # Convert to PIL Image
        generated = generated.squeeze(0).permute(1, 2, 0).numpy()
        generated = (generated * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(generated)

def main():
    st.set_page_config(page_title="Sketch to Image Generator", layout="wide")
    
    st.title("Sketch to Image Generator")
    st.write("Upload a sketch and watch it transform into a detailed image!")

    # Load model
    model, device = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose a sketch...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded sketch
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Sketch")
            sketch = Image.open(uploaded_file)
            st.image(sketch, use_column_width=True)

        # Process and generate image
        with st.spinner('Generating image...'):
            processed_sketch = process_sketch(sketch)
            generated_image = generate_image(model, processed_sketch, device)

        # Display the generated image
        with col2:
            st.subheader("Generated Image")
            st.image(generated_image, use_column_width=True)
            
            # Add download button
            if st.button('Download Generated Image'):
                # Convert PIL image to bytes
                img_byte_arr = BytesIO()
                generated_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                st.download_button(
                    label="Download",
                    data=img_byte_arr,
                    file_name="generated_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()