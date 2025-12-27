import sys
import torch
from PIL import Image
# Tận dụng class CLIPood từ file clip_ood.py bạn đã gửi
from clip_ood import CLIPood 

def run_mapper():
    # Sử dụng GPU nếu có, nếu không dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Khởi tạo mô hình NegRefine
    # output_folder='./' vì các file .pkl sẽ được Hadoop đẩy vào cùng thư mục chạy
    model = CLIPood(train_dataset='imagenet', arch='ViT-B/16', device=device, output_folder='./', load_saved_labels=True)
    
    # Đọc dữ liệu từ STDIN (Hadoop Streaming truyền vào)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        # Cấu trúc dòng: tên_dataset [Tab] đường_dẫn_ảnh
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        dataset_name, img_path = parts[0], parts[1]
        
        try:
            # Đọc ảnh từ đường dẫn tuyệt đối đã tạo trong input_list.txt
            image = Image.open(img_path).convert('RGB')
            
            # Tiền xử lý ảnh theo chuẩn CLIP
            image_tensor = model.clip_preprocess(image).unsqueeze(0).to(device)
            
            # Tính điểm detection_score (kết hợp NegLabel và SMM)
            score = model.detection_score(image_tensor)
            
            # Xuất ra: tên_dataset [Tab] điểm_số
            # Key là tên dataset để Reducer gom nhóm tính AUROC sau này
            print(f"{dataset_name}\t{score}")
        except Exception:
            # Bỏ qua các ảnh lỗi để không làm dừng Job Hadoop
            continue

if __name__ == "__main__":
    run_mapper()