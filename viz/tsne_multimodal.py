import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

def get_ellipse_params(points, n_std=2.0):
    """
    Calculate the parameters of an ellipse bounding a set of 2D points.
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # The angle of the ellipse
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Width and height are 2 * n_std * sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(vals)
    return pos, width, height, theta

def generate_simulated_data(num_classes=8, num_samples=30, spread_radius=10.0, noise_std=1.0, modality_gap=3.0):
    """
    Simulate 2D data directly to represent t-SNE outputs.
    Before alignment: high noise_std, large modality_gap.
    After alignment: low noise_std, small modality_gap, larger spread_radius for separation.
    """
    np.random.seed(42) # For reproducibility
    
    data = []
    # Distribute class centers roughly in a circle
    angles = np.linspace(0, 2*np.pi, num_classes, endpoint=False)
    
    for i in range(num_classes):
        # Base cluster center
        center = np.array([np.cos(angles[i]), np.sin(angles[i])]) * spread_radius 
        
        # Image modality
        img_proto = center + np.random.randn(2) * (noise_std * 0.2)
        img_features = img_proto + np.random.randn(num_samples, 2) * noise_std
        
        # Text/Description modality - displaced by modality_gap
        # We push text prototype slightly outward or rotationally
        displacement = np.array([np.cos(angles[i] + 0.5), np.sin(angles[i] + 0.5)]) * modality_gap
        desc_proto = center + displacement
        desc_features = desc_proto + np.random.randn(num_samples // 2, 2) * (noise_std * 0.8)
        
        # Calibrated classifier - usually an interpolation/alignment of both
        calibrated_proto = (float(num_samples) * img_proto + float(num_samples // 2) * desc_proto) / (num_samples * 1.5)
        # Add slight push to calibrated proto
        calibrated_proto += np.random.randn(2) * 0.1
        
        data.append({
            'img_features': img_features,
            'img_proto': img_proto,
            'desc_features': desc_features,
            'desc_proto': desc_proto,
            'calib_proto': calibrated_proto
        })
        
    return data

def plot_multimodal_tsne(ax, data, title):
    # Colors for the 8 classes (using tab10)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, cls_data in enumerate(data):
        c = colors[i]
        
        # 1. Plot Image features (small dots)
        ax.scatter(cls_data['img_features'][:, 0], cls_data['img_features'][:, 1],
                   color=c, marker='o', s=15, alpha=0.3, edgecolors='none', zorder=1)
                   
        # 2. Plot Description features (small squares)
        ax.scatter(cls_data['desc_features'][:, 0], cls_data['desc_features'][:, 1],
                   color=c, marker='s', s=15, alpha=0.3, edgecolors='none', zorder=1)
                   
        # 3. Draw Ellipse around image features
        pos, width, height, theta = get_ellipse_params(cls_data['img_features'], n_std=1.5)
        ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta,
                          edgecolor=c, fc='none', lw=1.5, linestyle='--', alpha=0.7, zorder=2)
        ax.add_patch(ellipse)
        
        # 4. Plot Prototypes
        # Image Prototype (Circle)
        ax.scatter(cls_data['img_proto'][0], cls_data['img_proto'][1],
                   color=c, marker='o', s=120, edgecolors='black', linewidths=1.0, zorder=3)
                   
        # Description Prototype (Square)
        ax.scatter(cls_data['desc_proto'][0], cls_data['desc_proto'][1],
                   color=c, marker='s', s=120, edgecolors='black', linewidths=1.0, zorder=3)
                   
        # Calibrated Classifier (Star)
        ax.scatter(cls_data['calib_proto'][0], cls_data['calib_proto'][1],
                   color=c, marker='*', s=200, edgecolors='black', linewidths=1.0, zorder=4)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    ax.set_aspect('equal') # Keep the circles and ellipses undistorted

def main():
    num_classes = 8
    
    # 1. Simulate "Before alignment"
    # Clusters overlap more -> spread_radius is smaller, noise is higher, gap between modalities is larger
    data_before = generate_simulated_data(
        num_classes=num_classes, 
        spread_radius=8.0, 
        noise_std=1.8, 
        modality_gap=3.5
    )
    
    # 2. Simulate "After alignment"
    # Clusters are tighter and separated -> spread is larger, noise is smaller, gap between modalities is very small
    data_after = generate_simulated_data(
        num_classes=num_classes, 
        spread_radius=12.0, 
        noise_std=0.6, 
        modality_gap=0.5
    )
    
    # Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='white')
    
    plot_multimodal_tsne(ax1, data_before, "Trước khi căn chỉnh")
    plot_multimodal_tsne(ax2, data_after, "Sau khi căn chỉnh")
    
    # Create the custom legend
    # We use gray color elements for the generic legend representations
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, alpha=0.4, label='Đặc trưng Hình ảnh'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=12, label='Mẫu đại diện Hình ảnh'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6, alpha=0.4, label='Đặc trưng Mô tả'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=12, label='Mẫu đại diện Mô tả'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=16, label='Bộ phân loại tinh chỉnh')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=14, 
               frameon=True, shadow=True, borderpad=1, bbox_to_anchor=(0.5, 0.05))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # make room for legend at the bottom
    plt.savefig('tsne_multimodal_alignment.png', dpi=300, facecolor='white', bbox_inches='tight')
    print("Saved figure to tsne_multimodal_alignment.png")
    
    # Optionally, also show the interactive plot:
    # plt.show()

if __name__ == "__main__":
    main()
