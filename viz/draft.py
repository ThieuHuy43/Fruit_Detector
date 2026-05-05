import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image

# Từ điển dịch tên thư mục sang tiếng Việt
TRANSLATIONS = {
    "amaranth": "Rau dền", "apple": "Táo", "asparagus": "Măng tây",
    "banana": "Chuối", "bokchoy": "Cải chíp", "cabbage": "Bắp cải",
    "carrot": "Cà rốt", "cherry": "Cherry", "chili": "Ớt",
    "chive": "Hẹ", "corn": "Ngô", "cucumber": "Dưa chuột",
    "eggplant": "Cà tím", "garlic": "Tỏi", "grape": "Nho",
    "guava": "Ổi", "jackfruit": "Mít", "kiwi": "Kiwi",
    "lettuce": "Xà lách", "lime": "Chanh", "loofah": "Mướp",
    "lychee": "Vải", "mango": "Xoài", "melon": "Dưa lưới",
    "mushroom": "Nấm", "mustardgreens": "Cải bẹ", "okra": "Đậu bắp",
    "onion": "Hành tây", "orange": "Cam", "passionfruit": "Chanh dây",
    "peach": "Đào", "pear": "Lê", "persimmon": "Hồng",
    "pineapple": "Dứa", "pomegranate": "Lựu",
    
    "Vietnam": "Việt Nam", "DaLat": "Đà Lạt", "import": "Nhập khẩu",
    "local": "Nội địa", "HaiDuong": "Hải Dương", "LySon": "Lý Sơn",
    "NinhThuan": "Ninh Thuận", "BenTre": "Bến Tre", "HungYen": "Hưng Yên",
    "DongThap": "Đồng Tháp", "GiaLai": "Gia Lai", "QuangNinh": "Quảng Ninh",
    
    "fake": "Giả", "real": "Thật"
}

# Danh sách các danh mục được hiển thị (lọc bỏ rau củ nhưng thêm Tỏi Lý Sơn)
ALLOWED_CLASSES = {
    "apple", "banana", "cherry", "grape", "guava", "jackfruit", "kiwi", 
    "lime", "lychee", "mango", "melon", "orange", "passionfruit", "peach", 
    "pear", "persimmon", "pineapple", "pomegranate", "garlic"
}

def translate_class_name(cls_name):
    """
    Dịch tên class, ví dụ 'apple_DaLat_fake' -> 'Táo Đà Lạt' 
    """
    parts = cls_name.split('_')
    fruit = TRANSLATIONS.get(parts[0].lower(), parts[0]) if len(parts) > 0 else ""
    origin = TRANSLATIONS.get(parts[1], parts[1]) if len(parts) > 1 else ""
    
    res = f"{fruit} {origin}".strip()
    return res

def visualize_fruit_classes(base_dir="FRUIT", num_samples=6):
    """
    Trực quan hóa một vài class ngẫu nhiên CÓ CHỨA ẢNH trong thư mục dataset.
    """
    if not os.path.exists(base_dir):
        print(f"Thư mục {base_dir} không tồn tại.")
        return

    valid_classes = []
    
    # Chỉ lưu các class có ít nhất 1 ảnh hợp lệ
    for d in os.listdir(base_dir):
        # Tách tên thư mục để lọc
        fruit_name = d.split('_')[0].lower()
        if fruit_name not in ALLOWED_CLASSES:
            continue
            
        cls_dir = os.path.join(base_dir, d)
        if os.path.isdir(cls_dir):
            image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                image_paths.extend(glob.glob(os.path.join(cls_dir, ext)))
            
            if image_paths:
                valid_classes.append((d, image_paths))
                
    if not valid_classes:
        print("Không có thư mục class nào chứa ảnh hợp lệ.")
        return

    # Tách Tỏi Lý Sơn (garlic) ra để đảm bảo luôn được chọn
    garlic_classes = [c for c in valid_classes if c[0].startswith("garlic")]
    other_classes = [c for c in valid_classes if not c[0].startswith("garlic")]

    selected_classes = []
    
    # Lấy 1 class Tỏi ngẫu nhiên nếu có
    if garlic_classes:
        selected_garlic = random.choice(garlic_classes)
        selected_classes.append(selected_garlic)
        
    # Bổ sung thêm các class khác cho đủ số lượng
    num_others_needed = min(num_samples - len(selected_classes), len(other_classes))
    if num_others_needed > 0:
        selected_classes.extend(random.sample(other_classes, num_others_needed))
        
    # Xáo trộn lại để Tỏi không phải lúc nào cũng nằm đầu
    random.shuffle(selected_classes)

    # Tính toán số hàng và cột cho lưới hiển thị
    cols = 3
    rows = (len(selected_classes) + cols - 1) // cols

    fig = plt.figure(figsize=(10, 3.5 * rows))
    fig.suptitle("Một số mẫu trái cây trong bộ dữ liệu", fontsize=16, fontweight='bold', y=1.02)
    
    for i, (cls_name, image_paths) in enumerate(selected_classes):
        # Chọn ngẫu nhiên 1 ảnh trong class đó
        img_path = random.choice(image_paths)
        try:
            img = Image.open(img_path)
            
            # Thay đổi kích thước ảnh (resize) về cùng một độ phân giải để các ô đều nhau
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            
            translated_title = translate_class_name(cls_name)
            
            # Hiện tên class trên mỗi hình
            ax.set_title(translated_title, fontsize=13, fontweight='medium', pad=10)
            
            # Bo viền mỏng và ẩn các trục số cho đẹp
            ax.axis('off')
        except Exception as e:
            print(f"Lỗi khi mở ảnh {img_path}: {e}")

    plt.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    visualize_fruit_classes()
