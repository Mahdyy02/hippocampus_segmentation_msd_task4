import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
encoder_color = '#4A90E2'
decoder_color = '#7ED321'
bottleneck_color = '#F5A623'
head_color = '#D0021B'
skip_color = '#9013FE'

def draw_block(ax, x, y, width, height, text, color, alpha=0.8):
    """Draw a rounded rectangle block"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, 
                         edgecolor='black', 
                         linewidth=2,
                         alpha=alpha)
    ax.add_patch(box)
    
    # Add text
    lines = text.split('\n')
    y_text = y + height/2 + (len(lines)-1) * 0.15
    for line in lines:
        ax.text(x + width/2, y_text, line, 
               ha='center', va='center', 
               fontsize=10, fontweight='bold',
               color='white' if color != '#FFFFFF' else 'black')
        y_text -= 0.3

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->'):
    """Draw arrow between blocks"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           mutation_scale=30,
                           linewidth=2.5,
                           color=color,
                           zorder=1)
    ax.add_patch(arrow)

# Title
ax.text(5, 13.5, 'nnU-Net Architecture for Hippocampus Segmentation', 
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(5, 13, 'Instance Normalization + Leaky ReLU + Skip Connections', 
        ha='center', va='top', fontsize=12, style='italic', color='gray')

# Input
draw_block(ax, 4.5, 11.5, 1, 0.7, 'Input\n1×128×128', '#CCCCCC')
draw_arrow(ax, 5, 11.5, 5, 11, 'black')

# ===== ENCODER =====
# Encoder 1
draw_block(ax, 0.5, 10, 1.2, 0.8, 'Conv 32\nInstNorm\nLeakyReLU', encoder_color)
draw_arrow(ax, 1.1, 9.6, 1.1, 9.2, 'black')
draw_block(ax, 0.5, 8.5, 1.2, 0.6, 'MaxPool 2×2', encoder_color, alpha=0.6)
draw_arrow(ax, 1.1, 8.5, 1.1, 8, 'black')

# Encoder 2
draw_block(ax, 0.5, 7.2, 1.2, 0.8, 'Conv 64\nInstNorm\nLeakyReLU', encoder_color)
draw_arrow(ax, 1.1, 6.8, 1.1, 6.4, 'black')
draw_block(ax, 0.5, 5.7, 1.2, 0.6, 'MaxPool 2×2', encoder_color, alpha=0.6)
draw_arrow(ax, 1.1, 5.7, 1.1, 5.2, 'black')

# Encoder 3
draw_block(ax, 0.5, 4.4, 1.2, 0.8, 'Conv 128\nInstNorm\nLeakyReLU', encoder_color)
draw_arrow(ax, 1.1, 4.0, 1.1, 3.6, 'black')
draw_block(ax, 0.5, 2.9, 1.2, 0.6, 'MaxPool 2×2', encoder_color, alpha=0.6)
draw_arrow(ax, 1.1, 2.9, 1.1, 2.4, 'black')

# Encoder 4
draw_block(ax, 0.5, 1.6, 1.2, 0.8, 'Conv 256\nInstNorm\nLeakyReLU', encoder_color)
draw_arrow(ax, 1.1, 1.2, 1.1, 0.8, 'black')
draw_block(ax, 0.5, 0.1, 1.2, 0.6, 'MaxPool 2×2', encoder_color, alpha=0.6)

# Arrow to bottleneck
draw_arrow(ax, 1.7, 0.4, 3.8, 0.4, 'black')

# ===== BOTTLENECK =====
draw_block(ax, 4, 0, 2, 1, 'Bottleneck\nConv 256\nInstNorm\nLeakyReLU', bottleneck_color)

# Arrow from bottleneck
draw_arrow(ax, 6, 0.5, 7.8, 0.5, 'black')

# ===== DECODER =====
# Decoder 1
draw_block(ax, 8, 0.1, 1.2, 0.6, 'UpConv 2×2', decoder_color, alpha=0.6)
draw_arrow(ax, 8.6, 0.7, 8.6, 1.1, 'black')
draw_block(ax, 8, 1.3, 1.2, 0.8, 'Conv 128\nInstNorm\nLeakyReLU', decoder_color)
draw_arrow(ax, 8.6, 2.1, 8.6, 2.5, 'black')

# Decoder 2
draw_block(ax, 8, 2.7, 1.2, 0.6, 'UpConv 2×2', decoder_color, alpha=0.6)
draw_arrow(ax, 8.6, 3.3, 8.6, 3.7, 'black')
draw_block(ax, 8, 3.9, 1.2, 0.8, 'Conv 64\nInstNorm\nLeakyReLU', decoder_color)
draw_arrow(ax, 8.6, 4.7, 8.6, 5.1, 'black')

# Decoder 3
draw_block(ax, 8, 5.3, 1.2, 0.6, 'UpConv 2×2', decoder_color, alpha=0.6)
draw_arrow(ax, 8.6, 5.9, 8.6, 6.3, 'black')
draw_block(ax, 8, 6.5, 1.2, 0.8, 'Conv 32\nInstNorm\nLeakyReLU', decoder_color)
draw_arrow(ax, 8.6, 7.3, 8.6, 7.7, 'black')

# Decoder 4
draw_block(ax, 8, 7.9, 1.2, 0.6, 'UpConv 2×2', decoder_color, alpha=0.6)
draw_arrow(ax, 8.6, 8.5, 8.6, 8.9, 'black')
draw_block(ax, 8, 9.1, 1.2, 0.8, 'Conv 32\nInstNorm\nLeakyReLU', decoder_color)

# ===== SKIP CONNECTIONS =====
# Skip 1 (from Enc4 to Dec1)
draw_arrow(ax, 1.7, 2.0, 8.0, 1.7, skip_color, style='->')
ax.text(4.85, 1.9, 'concat', ha='center', va='center', 
        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', edgecolor=skip_color))

# Skip 2 (from Enc3 to Dec2)
draw_arrow(ax, 1.7, 4.8, 8.0, 4.3, skip_color, style='->')
ax.text(4.85, 4.6, 'concat', ha='center', va='center', 
        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', edgecolor=skip_color))

# Skip 3 (from Enc2 to Dec3)
draw_arrow(ax, 1.7, 7.6, 8.0, 6.9, skip_color, style='->')
ax.text(4.85, 7.3, 'concat', ha='center', va='center', 
        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', edgecolor=skip_color))

# Skip 4 (from Enc1 to Dec4)
draw_arrow(ax, 1.7, 10.4, 8.0, 9.5, skip_color, style='->')
ax.text(4.85, 10.0, 'concat', ha='center', va='center', 
        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', edgecolor=skip_color))

# ===== SEGMENTATION HEAD =====
draw_arrow(ax, 8.6, 9.9, 8.6, 10.5, 'black')

# Fork for two heads
draw_arrow(ax, 8.6, 10.5, 7.5, 11, 'black', style='-')
draw_arrow(ax, 8.6, 10.5, 9.7, 11, 'black', style='-')

# Standard head
draw_block(ax, 6.5, 11.2, 1.8, 0.8, 'Standard Head\nConv 1×1 → 3', head_color, alpha=0.7)

# INR head (optional)
draw_block(ax, 9, 11.2, 1.8, 0.8, 'INR Head\nSIREN + Coords', head_color, alpha=0.7)
ax.text(9.9, 10.9, '(optional)', ha='center', fontsize=8, style='italic', color='gray')

# Output
draw_arrow(ax, 7.4, 12, 7.4, 12.5, 'black')
draw_arrow(ax, 9.9, 12, 9.9, 12.5, 'black')

# Merge outputs
draw_arrow(ax, 7.4, 12.5, 8.6, 12.8, 'black', style='-')
draw_arrow(ax, 9.9, 12.5, 8.6, 12.8, 'black', style='-')

draw_block(ax, 7.8, 13, 1.6, 0.5, 'Output\n3×128×128', '#CCCCCC')

# ===== LEGEND =====
legend_y = -0.5
legend_elements = [
    mpatches.Patch(facecolor=encoder_color, edgecolor='black', label='Encoder Blocks'),
    mpatches.Patch(facecolor=bottleneck_color, edgecolor='black', label='Bottleneck'),
    mpatches.Patch(facecolor=decoder_color, edgecolor='black', label='Decoder Blocks'),
    mpatches.Patch(facecolor=skip_color, edgecolor='black', label='Skip Connections'),
    mpatches.Patch(facecolor=head_color, edgecolor='black', label='Segmentation Heads')
]

ax.legend(handles=legend_elements, loc='lower center', ncol=5, 
         fontsize=10, frameon=True, bbox_to_anchor=(0.5, legend_y))

# Add annotations
ax.text(1.1, 12.3, 'ENCODER', fontsize=14, fontweight='bold', 
        color=encoder_color, rotation=0)
ax.text(8.6, 12.3, 'DECODER', fontsize=14, fontweight='bold', 
        color=decoder_color, rotation=0)

# Add feature map sizes
feature_sizes = [
    (2, 10.4, '128×128×32'),
    (2, 7.6, '64×64×64'),
    (2, 4.8, '32×32×128'),
    (2, 2.0, '16×16×256'),
    (5, 0.4, '8×8×256'),
    (7, 1.7, '16×16×128'),
    (7, 4.3, '32×32×64'),
    (7, 6.9, '64×64×32'),
    (7, 9.5, '128×128×32'),
]

for x, y, size in feature_sizes:
    ax.text(x, y, size, fontsize=7, style='italic', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                    edgecolor='gray', alpha=0.8))

# Save
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("✓ Architecture diagram saved as 'architecture_diagram.png'")
plt.close()

# Also create a simpler vertical flow diagram
fig2, ax2 = plt.subplots(figsize=(10, 14), facecolor='white')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 16)
ax2.axis('off')

# Title
ax2.text(5, 15.5, 'nnU-Net Architecture Flow', 
        ha='center', va='top', fontsize=18, fontweight='bold')

y_pos = 14.5

def add_flow_block(ax, y, text, color, width=3):
    draw_block(ax, 3.5, y-0.4, width, 0.6, text, color)
    return y - 1

# Flow
y_pos = add_flow_block(ax2, y_pos, 'Input: 1×128×128 MRI Slice', '#CCCCCC')
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↓ Encoder Block 1: Conv 32 + InstNorm + LeakyReLU', encoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↓ Encoder Block 2: Conv 64 + InstNorm + LeakyReLU', encoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↓ Encoder Block 3: Conv 128 + InstNorm + LeakyReLU', encoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↓ Encoder Block 4: Conv 256 + InstNorm + LeakyReLU', encoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, 'Bottleneck: Conv 256 (8×8 feature maps)', bottleneck_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↑ Decoder Block 1: UpConv 128 + Skip Connection', decoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↑ Decoder Block 2: UpConv 64 + Skip Connection', decoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↑ Decoder Block 3: UpConv 32 + Skip Connection', decoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, '↑ Decoder Block 4: UpConv 32 + Skip Connection', decoder_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, 'Segmentation Head: Conv 1×1 or INR', head_color)
draw_arrow(ax2, 5, y_pos+0.5, 5, y_pos, 'black')

y_pos = add_flow_block(ax2, y_pos, 'Output: 3×128×128 (BG, Anterior, Posterior)', '#CCCCCC')

# Add key features box
features_text = """Key Features:
• Instance Normalization (better for medical imaging)
• Leaky ReLU activation (α=0.01)
• U-Net skip connections (preserve spatial info)
• Optional INR head with SIREN
• Mixed precision training (FP16)
• Deep supervision capable"""

ax2.text(5, 1.5, features_text, 
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                 edgecolor='black', linewidth=2))

plt.tight_layout()
plt.savefig('architecture_flow.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ Architecture flow diagram saved as 'architecture_flow.png'")
plt.close()

print("\n✓ Both diagrams generated successfully!")
