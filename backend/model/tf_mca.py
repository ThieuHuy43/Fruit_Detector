# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import numpy as np
# import pickle
# import os

# class TFMCAModel:
#     def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, memory_path="memory.pkl"):
#         """
#         Khởi tạo mô hình CLIP và Memory Bank cho TF-MCA.
#         """
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading CLIP model '{model_name}' on {self.device}...")
#         self.model = CLIPModel.from_pretrained(model_name).to(self.device)
#         self.processor = CLIPProcessor.from_pretrained(model_name)
        
#         self.memory_path = memory_path
#         self.class_prototypes = {}  # {class_name: feature_vector}
#         self.load_memory()

#     def load_memory(self):
#         """Khôi phục memory bank từ file nếu có sẵn"""
#         if os.path.exists(self.memory_path):
#             with open(self.memory_path, "rb") as f:
#                 self.class_prototypes = pickle.load(f)
#             print(f"Loaded {len(self.class_prototypes)} classes from memory.")
#         else:
#             print("No memory bank found. Starting fresh.")

#     def save_memory(self):
#         """Lưu trữ memory bank xuống đĩa để không mất sau khi khởi động lại"""
#         with open(self.memory_path, "wb") as f:
#             pickle.dump(self.class_prototypes, f)
#         print("Memory bank saved.")

#     def _extract_image_features(self, image: Image.Image):
#         """Trích xuất features véc-tơ từ một ảnh"""
#         inputs = self.processor(images=image, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             image_features = self.model.get_image_features(**inputs)
#         # Normalize features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         return image_features.cpu().numpy()[0]

#     def add_class(self, class_name: str, images: list[Image.Image]):
#         """
#         Thêm một lớp nông sản mới (Few-Shot) bằng cách lấy trung bình đặc trưng các ảnh mẫu (Prototype).
#         """
#         if len(images) == 0:
#             raise ValueError("Cần ít nhất 1 ảnh để thêm lớp mới.")
            
#         features = []
#         for img in images:
#             feat = self._extract_image_features(img)
#             features.append(feat)
            
#         # Tính trung bình (prototype)
#         features = np.array(features)
#         prototype = np.mean(features, axis=0)
#         # Chuẩn hóa prototype
#         prototype = prototype / np.linalg.norm(prototype)
        
#         self.class_prototypes[class_name] = prototype
#         self.save_memory()
#         return len(self.class_prototypes)

#     def predict(self, image: Image.Image):
#         """
#         Dự đoán lớp của một ảnh đầu vào dựa trên cosine similarity với các prototype đã lưu.
#         """
#         if not self.class_prototypes:
#             return {"error": "Chưa có dữ liệu lớp nào trong hệ thống. Vui lòng thêm lớp mới trước."}
            
#         img_feature = self._extract_image_features(image)
        
#         best_match = None
#         highest_sim = -1.0
        
#         # So sánh cosine similarity
#         for class_name, prototype in self.class_prototypes.items():
#             similarity = np.dot(img_feature, prototype)
#             if similarity > highest_sim:
#                 highest_sim = similarity
#                 best_match = class_name
                
#         # Giả lập logic kiểm tra thật giả (Ví dụ: xem feature img_feature có quá xa cụm chính của class không).
#         # Tạm thời gán mặc định với một score dummy để hoàn thiện sau.
#         is_real = True
#         fake_score = float(1.0 - highest_sim) # Score càng thấp => Similarity càng cao => Thật hơn
#         if highest_sim < 0.2:
#             is_real = False 

#         return {
#             "predicted_class": best_match,
#             "confidence": float(highest_sim),
#             "is_real": is_real,
#             "fake_score": fake_score
#         }
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
import os

FRUIT_TRANSLATION = {
    "apple": "Táo", "banana": "Chuối", "durian": "Sầu riêng", "grape": "Nho", 
    "guava": "Ổi", "jackfruit": "Mít", "lime": "Chanh", "lychee": "Vải", 
    "mango": "Xoài", "melon": "Dưa lê", "muskmelon": "Dưa lưới", "orange": "Cam", 
    "passionfruit": "Chanh leo", "peach": "Đào", "pear": "Lê", "persimmon": "Hồng",
    "pineapple": "Dứa", "pomegranate": "Lựu", "pomelo": "Bưởi", "soursop": "Mãng cầu", 
    "strawberry": "Dâu tây", "tamarind": "Me", "tangerine": "Quýt", "watermelon": "Dưa hấu", 
    "amaranth": "Rau dền", "asparagus": "Măng tây", "bokchoy": "Cải thìa", "cabbage": "Bắp cải",
    "carrot": "Cà rốt", "chive": "Hẹ", "corn": "Ngô", "cucumber": "Dưa chuột", 
    "eggplant": "Cà tím", "garlic": "Tỏi", "greenbean": "Đậu cô ve", "loofah": "Mướp",
    "mushroom": "Nấm", "mustardgreens": "Cải bẹ xanh", "okra": "Đậu bắp", "potato": "Khoai tây", 
    "pumpkin": "Bí đỏ", "shallot": "Hành tím", "sweetpotato": "Khoai lang",
    "tomato": "Cà chua", "waterspinach": "Rau muống"
}

ORIGIN_TRANSLATION = {
    # Tỉnh thành Việt Nam
    "tiengiang": "Tiền Giang", "dalat": "Đà Lạt", "baria": "Bà Rịa", 
    "tayninh": "Tây Ninh", "quangninh": "Quảng Ninh", "langson": "Lạng Sơn", 
    "longan": "Long An", "laocai": "Lào Cai", "bentre": "Bến Tre", 
    "hungyen": "Hưng Yên", "dongthap": "Đồng Tháp", "thanhhoa": "Thanh Hóa", 
    "vinhlong": "Vĩnh Long", "ninhthuan": "Ninh Thuận", "angiang": "An Giang", 
    "lamdong": "Lâm Đồng", "nghean": "Nghệ An", "bacgiang": "Bắc Giang", 
    "daklak": "Đắk Lắk", "cantho": "Cần Thơ", "hanoi": "Hà Nội", 
    "gialai": "Gia Lai", "haiduong": "Hải Dương", "quangngai": "Quảng Ngãi", 
    "binhduong": "Bình Dương", "hagiang" : "Hà Giang",
    # Quốc gia
    "usa": "Mỹ", "china": "Trung Quốc", "india": "Ấn Độ", "korea": "Hàn Quốc",
    # Khác
    "fake": "Giả mạo"
}

# Import thư viện CLIP gốc của project (BiMC sử dụng package 'models.clip')
# Mở rộng đường dẫn để import được từ thư mục gốc
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import models.clip.clip as clip

class TFMCAModel:
    def __init__(self, memory_path=r"C:\Users\PC\OneDrive\Dokumen\Đồ án tốt nghiệp\backend\bimc_memory.pth", backbone="ViT-B/16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_path = memory_path
        
        # 1. Load CLIP chính gốc của OPENAI/BiMC
        print(f"Khởi tạo CLIP Model {backbone}...")
        try:
            model = torch.jit.load(clip._download(clip._MODELS[backbone]), map_location="cpu")
        except RuntimeError:
            model = torch.load(clip._download(clip._MODELS[backbone]), map_location="cpu")
        self.clip_model = clip.build_model(model.state_dict()).to(self.device)
        self.clip_model.eval()

        # Tiền xử lý ảnh (Resize, CenterCrop, Normalize)
        _, self.transform = clip.load(backbone, device=self.device)

        # 2. Load các Thông số BiMC Memory (Prototypes & Covariance)
        self.memory = None
        if os.path.exists(self.memory_path):
            self.memory = torch.load(self.memory_path, map_location=self.device)
            self.class_names = self.memory['class_names']
            print(f"Đã nạp {len(self.class_names)} classes từ BiMC memory.")
        else:
            print("Cảnh báo: Chưa chạy main.py để train model, không tìm thấy file trí nhớ.")

    def predict(self, image: Image.Image):
        if not self.memory:
            return {"error": "Chưa có file mô hình."}

        # 1. Xử lý ảnh và nạp qua CLIP -> lấy Image Feature
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img_tensor)
            img_feat = F.normalize(img_feat, dim=-1)

        # 2. Tính Fused Proto giống `models/bimc.py`
        beta = self.memory['beta']
        lambda_t = self.memory['lambda_t']
        image_proto = self.memory['image_proto']
        desc_proto = self.memory['description_proto']
        txt_feat = self.memory['text_features']

        fused_proto = beta * ((1 - lambda_t) * txt_feat + lambda_t * desc_proto) + (1 - beta) * image_proto        
        fused_proto = F.normalize(fused_proto, dim=-1)  

        # Softmax Probability từ Fused Proto
        # logits_proto = img_feat @ fused_proto.t()
        # prob_fused = F.softmax(logits_proto, dim=-1)
                # Nhân với logit_scale của CLIP (thường ~100.0) để làm sắc nét phân bố xác suất
        logits_proto = img_feat @ fused_proto.t()
        logit_scale = self.clip_model.logit_scale.exp().item() if hasattr(self.clip_model, 'logit_scale') else 100.0
        
        prob_fused = F.softmax(logit_scale * logits_proto, dim=-1)

        # 3. Tính Mahalanobis Covariance Distance (dành cho Base classes)
        cov_image = self.memory['cov_image']
        inv_covmat = torch.pinverse(cov_image.to(torch.float32)).to(img_feat.dtype)
        
        maha_dist = []
        for cl in range(len(self.class_names)):
            distance = img_feat - image_proto[cl]
            left_term = torch.matmul(distance, inv_covmat)
            dist = torch.diag(torch.matmul(left_term, distance.T))
            maha_dist.append(dist)
            
        logits_cov = -torch.stack(maha_dist).T
        prob_cov = F.softmax(logits_cov / 512, dim=-1)

        # 4. Gom Ensemble Alpha
        alpha = self.memory['ensemble_alpha']
        num_base = self.memory['num_base_cls']
        base_probs = alpha * prob_fused[:, :num_base] + (1 - alpha) * prob_cov[:, :num_base]
        
        # NOTE: Đối phó nhanh cho Demo (Nếu bạn có thêm KNN Incremental thì ghép tiếp)
        inc_probs = prob_fused[:, num_base:] 
        final_probs = torch.cat([base_probs, inc_probs], dim=1)[0]
        
        # # Chọn ra class cao nhất
        # max_prob, best_idx = torch.max(final_probs, dim=0)
        # best_match = self.class_names[best_idx.item()]

        # # Tính toán mức độ phân biệt "Giả/Thật" từ Mahalanobis Out-of-Distribution
        # lowest_maha = maha_dist[best_idx].item()
        # fake_score = min(1.0, lowest_maha / 200.0) # Chuẩn hoá ngưỡng tùy thực tế
        # is_real = True if fake_score < 0.6 else False

        # return {
        #     "predicted_class": best_match,
        #     "confidence": float(max_prob.item()),
        #     "is_real": is_real,
        #     "fake_score": float(fake_score),
        #     "mahalanobis_dist": lowest_maha # Output log
        # }
                # Chọn ra class cao nhất (Vẫn giữ nguyên bộ chọn Ensemble chuẩn)
        _, best_idx = torch.max(final_probs, dim=0)
        best_match = self.class_names[best_idx.item()]

        # HƯỚNG 1: Lấy mức độ tin cậy từ luồng CLIP gốc (prob_fused) thay vì luồng bị mix với độ trễ của Covariance
        display_confidence = prob_fused[0, best_idx].item()

        # Tính toán mức độ phân biệt "Giả/Thật" từ Mahalanobis Out-of-Distribution
        lowest_maha = maha_dist[best_idx].item()
        fake_score = min(1.0, lowest_maha / 200.0) # Chuẩn hoá ngưỡng tùy thực tế
        is_real = True if fake_score < 0.6 else False
        
                # Hàm hỗ trợ dịch tên nông sản (Xử lý định dạng có chứa dấu gạch ngang như mango-DongThap)
        # Hàm hỗ trợ dịch tên nông sản và địa danh
        def translate_label(label):
            parts = label.split('-')
            
            # 1. Dịch quả/rau
            fruit_eng = parts[0].lower()
            fruit_vn = FRUIT_TRANSLATION.get(fruit_eng, parts[0].capitalize())
            
            # 2. Dịch xuất xứ (nếu có)
            if len(parts) > 1:
                origin_eng = parts[1].lower()
                origin_vn = ORIGIN_TRANSLATION.get(origin_eng, parts[1]) # Mặc định giữ nguyên nếu ko có trong dict
                return f"{fruit_vn} ({origin_vn})"
                
            return fruit_vn

        # Tạo tên tiếng việt chuẩn dấu
        best_match_vn = translate_label(best_match)

        # Tính toán mức độ phân biệt "Giả/Thật" ... -> Giữ nguyên phần bên dưới

        return {
            "predicted_class": best_match_vn, # <-- Trả về tên tiếng Việt
            "confidence": float(display_confidence),
            "is_real": is_real,
            "fake_score": float(fake_score),
            "mahalanobis_dist": lowest_maha,
            "original_class": best_match # Trả thêm original class nếu front-end cần
        }
    
    def add_class(self, class_name: str, description: str, images: list[Image.Image]):
        """
        Thêm một lớp nông sản mới (Few-Shot) vào cấu trúc BiMC Memory 
        bằng cách trích xuất đặc trưng và cập nhật hệ tensors. Dùng description tùy chọn làm prototype.
        """
        if len(images) == 0:
            raise ValueError("Cần ít nhất 1 ảnh để thêm lớp mới.")
        
        if not self.memory:
            raise ValueError("Chưa nạp Memory gốc, không thể thêm lớp (Incremental).")

        self.clip_model.eval()
        
        # 1. Trích xuất véc-tơ đặc trưng từ các ảnh mới
        features = []
        with torch.no_grad():
            for img in images:
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.clip_model.encode_image(img_tensor)
                features.append(feat)
                
        # 2. Tính trung bình (prototype ảnh) chuẩn hóa cho lớp mới
        # Gom list [Tensor] -> Tensor cục bộ -> Mean -> Normalize
        feat_stack = torch.cat(features, dim=0)
        img_proto_new = torch.mean(feat_stack, dim=0, keepdim=True)
        img_proto_new = F.normalize(img_proto_new, dim=-1)

        # 3. Trích xuất đặc trưng Tên lớp (Text prompt)
        # Nếu người dùng có cung cấp description, sử dụng description đó
        prompt_str = description.strip() if description and description.strip() else f"a photo of a {class_name}"
        prompt = clip.tokenize([prompt_str]).to(self.device)
        with torch.no_grad():
            txt_feat_new = self.clip_model.encode_text(prompt)
            txt_feat_new = F.normalize(txt_feat_new, dim=-1)

        # 4. Cập nhật thẳng vào file trí nhớ (self.memory) trong RAM
        # Chuyển class_names thành list để tránh lỗi tuple.append
        if isinstance(self.class_names, tuple):
            self.class_names = list(self.class_names)
            
        self.class_names.append(class_name)
        self.memory['class_names'] = self.class_names
        
        # Ghép tensor (Nối dòng mới vào ma trận cũ)
        self.memory['image_proto'] = torch.cat([self.memory['image_proto'], img_proto_new], dim=0)
        self.memory['text_features'] = torch.cat([self.memory['text_features'], txt_feat_new], dim=0)
        
        # Với incremental class, description_proto có thể dùng luôn ảnh hoặc text
        # (Ở đây tạm nối text_features vào description_proto để đồng bộ chiều)
        self.memory['description_proto'] = torch.cat([self.memory['description_proto'], txt_feat_new], dim=0)
        
        # Đảm bảo beta và lambda_t mở rộng nếu chúng là tensor riêng cho từng lớp
        if isinstance(self.memory['beta'], torch.Tensor) and self.memory['beta'].shape[0] < self.memory['image_proto'].shape[0]:
            mean_beta = self.memory['beta'].mean().view(1, -1) if self.memory['beta'].dim() > 1 else self.memory['beta'].mean().view(1)
            self.memory['beta'] = torch.cat([self.memory['beta'], mean_beta], dim=0)
            
        if isinstance(self.memory['lambda_t'], torch.Tensor) and self.memory['lambda_t'].shape[0] < self.memory['image_proto'].shape[0]:
            mean_lambda = self.memory['lambda_t'].mean().view(1, -1) if self.memory['lambda_t'].dim() > 1 else self.memory['lambda_t'].mean().view(1)
            self.memory['lambda_t'] = torch.cat([self.memory['lambda_t'], mean_lambda], dim=0)
            
        # 5. (Tùy chọn) Lưu lại xuống đĩa để bảo lưu khi reset Server
        torch.save(self.memory, self.memory_path)
        print(f"Đã cập nhật Memory: Nhận diện được {len(self.class_names)} loại!")
        
        return len(self.class_names)    