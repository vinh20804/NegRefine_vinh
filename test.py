import os

# Đường dẫn đến thư mục ImageNet của bạn (thư mục chứa 00999, 00998,...)
dataset_path = '/Users/macintoshhd/Documents/ImageNet' 
output_file = '/Users/macintoshhd/Documents/Bigdata/code/NegRefine/input_list.txt'

with open(output_file, 'w') as f:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                # Lấy đường dẫn tuyệt đối để Hadoop Worker tìm đúng ảnh
                full_path = os.path.abspath(os.path.join(root, file))
                # Ghi định dạng: tên_dataset [Tab] đường_dẫn_ảnh
                f.write(f"imagenet\t{full_path}\n")

print(f"Đã tạo xong file {output_file}")