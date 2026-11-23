# Kiểm tra bản bàn giao

Ngày 26/10/2026. Môi trường Linux, Python 3.12, CPU. Đây là xác minh phần mềm; không phải nghiên cứu đánh giá hành vi trên dữ liệu thật.

## Đã chạy

- `python -m pytest -q`: 12 kiểm thử thành công trong lần chạy cuối trước đóng gói.
- `python -m motionguard demo --out runs/demo`: tạo feature giả, train baseline, chọn ngưỡng validation, lưu/nạp checkpoint, đánh giá trên tập test giả độc lập.
- TCN pose-only cấu hình nhỏ: train một epoch CPU, lưu/nạp `.pt`, dự đoán và xuất metrics đủ 7 nhãn.
- `python scripts/smoke_tcn.py --rgb --out runs/final_smoke_rgb`: train TCN + CNN một epoch CPU với crop giả, lưu/nạp checkpoint và đánh giá. Tất cả checkpoint tạo từ demo mới được đánh dấu chỉ để kiểm tra, không cho dùng ở lệnh infer video.
- Adapter Ultralytics YOLO-Pose + tracker thật: tạo kiến trúc `yolo11n-pose.yaml` với trọng số ngẫu nhiên, gọi hai frame trắng 160×160; API chạy thành công, không có detection. Kiểm tra này không tải checkpoint pretrained và không đo chất lượng pose/tracking.
- Video pipeline có pose adapter giả lập: đọc video MP4 thực, trích xuất feature thực, chạy baseline và xuất score log.
- Ghi/đọc lại clip MP4 thành công; kiểm tra metadata pre/post-roll.
- Kiểm tra leakage theo subject, nhãn chưa biết, cấu hình extraction khác training, reset cửa sổ khi thiếu dữ liệu, đa nhãn/hysteresis, matching event một-một.
- Kiểm tra causal TCN cả pose-only và RGB: sửa dữ liệu tương lai không làm thay đổi output quá khứ; backward chạy được.

## Chưa xác minh

- Chưa có video người thật được gán nhãn cho 7 hành vi; không có checkpoint hành vi thực kèm theo.
- Chưa đánh giá pretrained pose trên người thật, crowded scenes, occlusion hoặc domain camera cụ thể.
- Chưa chạy Windows, webcam/RTSP, CUDA, OpenCV GUI hay đo VRAM/FPS/độ trễ thật.
- Chưa có kết quả event sensitivity, false alarms/hour hoặc calibration trên dữ liệu thực.
- Metric demo không được dùng để báo cáo khả năng phát hiện dấu hiệu sức khỏe.

## Dependency trong môi trường kiểm tra

| Gói | Phiên bản |
|---|---|
| numpy | 2.3.5 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.0 |
| torch | 2.14.1 |
| ultralytics | 8.4.189 |
| opencv-python | 4.14.0.94 |
| lap | 0.5.13 |
| pytest | 9.1.1 |
| PyYAML | 6.0.3 |
| joblib | 1.5.3 |

`lap` được khai báo trực tiếp trong extra video sau khi kiểm tra adapter phát hiện tracker cần gói này. Đây là snapshot kiểm thử, không phải lockfile cho mọi OS/GPU. Phiên bản trong `pyproject.toml` là các range được hỗ trợ dự kiến; dùng môi trường ảo và lưu lockfile sau khi xác nhận trên máy triển khai.

## Cập nhật tự chọn đường dẫn — 28/10/2026

`--out` không còn bắt buộc ở `train-tcn` và `train-baseline`. Nếu bỏ qua, CLI tự tạo thư mục duy nhất dưới `runs/tcn/` hoặc `runs/baseline/`, đặt tên checkpoint `model.pt`/`model.joblib`, rồi in đường dẫn đầy đủ. Thời điểm ghi checkpoint vẫn là khi train hoàn tất.

Ba kiểm thử mới chạy bằng `PYTHONPATH=motionguard python -m unittest discover -s motionguard/tests -p test_auto_output.py -v` từ thư mục cha project: đường dẫn tự động cho TCN, baseline, chống trùng và giữ đường dẫn chỉ định. Cả 3 thành công. Các hàm train được thay bằng fixture để kiểm tra CLI; không huấn luyện lại mạng ở lần cập nhật này. Bộ 12 kiểm thử cũ chưa được chạy lại trong môi trường mới do thiếu pytest và kết nối cài đặt bị chặn.
