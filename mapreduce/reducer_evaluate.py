import sys
import time
import numpy as np
# Tận dụng hàm tính toán AUROC/FPR95 từ ood_evaluate.py
from ood_evaluate import get_measures 

def run_reducer():
    # Bắt đầu đo thời gian xử lý tổng hợp
    start_time = time.time()
    data_scores = {}
    
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        parts = line.split('\t')
        if len(parts) < 2: continue
        dataset_name, score = parts[0], float(parts[1])
        
        if dataset_name not in data_scores:
            data_scores[dataset_name] = []
        data_scores[dataset_name].append(score)

    # Toàn bộ nội dung print dưới đây sẽ được Hadoop ghi vào file output trên HDFS
    print("================ FINAL RESULT REPORT ================")
    print(f"Time started at: {time.ctime(start_time)}")
    print("\n--- Starting predictions on testing images")
    
    in_dataset_name = 'imagenet'
    scores_in = np.array(data_scores.get(in_dataset_name, []))
    # In số lượng ảnh In-distribution
    print(f"in ({in_dataset_name}) - number: {len(scores_in)}")

    ood_datasets = [d for d in data_scores.keys() if d != in_dataset_name]
    for i, ood_name in enumerate(ood_datasets):
        # In số lượng ảnh OOD
        print(f"ood{i} ({ood_name}) - number: {len(data_scores[ood_name])}")
    
    # Tính thời gian thực thi của pha đánh giá
    process_time = time.time() - start_time
    print(f"Processing time: {process_time:.2f} seconds")

    print("\n--- Evaluation Metrics")
    for ood_name in ood_datasets:
        print(f"------------------ {ood_name} -------------------")
        scores_ood = np.array(data_scores[ood_name])
        
        if len(scores_in) > 0 and len(scores_ood) > 0:
            # Tính toán AUROC và FPR qua module ood_evaluate.py
            auroc, _, _, fpr = get_measures(scores_in, scores_ood)
            # In kết quả theo định dạng của eval.py
            print(f"score0     auroc: {auroc:.4f}    fpr: {fpr:.4f}")
    
    print("=====================================================")

if __name__ == "__main__":
    run_reducer()