# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab"; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về **cả bức ảnh** vì: 
-Record này có "task": "image_classification" -> đây là task phân loại ảnh: mô hình đưa ra một nhãn đại diện cho toàn bộ ảnh đầu vào. 
-Record không có các thông tin tọa độ bounding box, đây là thông tin thường xuất hiện khi mô tả một **một vật thể riêng**
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp không phải model tự nghĩ ra mà đi kèm taxonomy/checkpoint. Nó được quyết định từ trước bởi bộ dữ liệu huấn luyện và đã được lưu sẵn bên trong checkpoint.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Để tránh hiểu lầm giữa các bộ từ vựng khác nhau, tránh lỗi chính tả và đồng bộ hóa dễ dàng đồng thời hỗ trợ đa ngôn ngữ và thay đổi hiển thị.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Người gán nhãn phải chọn một lớp duy nhất đại diện tốt nhất cho nội dung chính của toàn bộ bức ảnh. Không gán nhãn riêng cho từng vật thể. Khi có nhiều vật thể thuộc các lớp khác nhau, ưu tiên lớp tương ứng với đối tượng/chủ đề nổi bật nhất, dựa trên mức độ chiếm diện tích, vị trí trung tâm và vai trò chính trong ảnh. Nếu không xác định được đối tượng/chủ đề nào là chính, chọn lớp phù hợp nhất với ngữ cảnh tổng thể của ảnh theo taxonomy được cung cấp.
6. Vì sao không được lấy `score` của model làm ground truth?
Không được lấy score của model làm ground truth vì score chỉ thể hiện mức độ tin tưởng của mô hình đối với dự đoán của chính nó còn ground truth là sự thật khách quan do chính con người xác nhận.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái nằm ở tọa độ (385.33, 69.24) nghĩa là cách biên trái của ảnh 498.92 pixel và cách biên trên của ảnh 348.92 pixel. Hộp rộng 113.58 pixel và cao 279.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
0.2: có 11 prediction
0.6: có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng để giữ lại nhiều box hơn, reviewer phải làm thêm việc kiểm tra và loại bỏ các box sai/không phù hợp mà trước đó ngưỡng cao đã tự động loại bỏ.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Box phải được vẽ bám sát hoàn toàn vào ranh giới ngoài cùng của vật thể ở cả 4 phía (trên, dưới, trái, phải). Không được để thừa khoảng trống của nền quá 2 pixel và không được cắt lẹm vào cơ thể hay bộ phận của đối tượng. Nếu vật thể bị che khuất một phần bởi vật thể khác hoặc bị cắt bởi viền ảnh nhưng vẫn nhận diện được dựa trên ngữ cảnh, annotator vẫn phải vẽ một hộp bao duy nhất bao trọn toàn bộ phần nhìn thấy được của vật thể đó. Tuyệt đối không tự ý chia nhỏ một vật thể bị che khuất thành nhiều hộp riêng biệt trừ khi có hướng dẫn đặc biệt từ reviewer.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Record bowl 0.38 bị record bowl 0.72 che khuất 1 phần. Trong trường hợp này, tôi sẽ viết thành quy tắc trong guideline vì vật thể vẫn có thể xác định được là đối tượng thuộc lớp cần gán nhãn nào, vẫn tạo box cho phần vật thể có thể quan sát được.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ]
      ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho thêm thông tin chính xác về các hình dạng phức tạp của vật thể bằng một chuỗi các điểm tọa độ nối tiếp nhau bám khít theo đường viền thực tế của vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt các cá thể khác nhau xuất hiện trong một bức ảnh cụ thể ngay cả khi chúng thuộc cùng một lớp.
**không** phải class ID: class_id là mã số đại diện cho danh mục/nhãn chung của vật thể (ví dụ: tất cả các chiếc bát đều chung class_id = 45). Trong khi đó, instance_id là mã định danh riêng cho từng chiếc bát cụ thể.
**không** phải tracking ID: tracking_id là ID dùng để theo dõi một đối tượng liên tục xuyên suốt nhiều khung hình của một video (vật thể A ở frame 1 vẫn giữ nguyên ID đó ở frame 2, 3...). Ngược lại, instance_id ở đây chỉ tồn tại độc lập trong phạm vi một bức ảnh tĩnh duy nhất, khi chuyển sang ảnh khác, các ID này sẽ được đánh số lại từ đầu và không liên quan đến ảnh trước đó.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Sử dụng tối thiểu 15–30 điểm neo (anchor points) tùy thuộc vào độ phức tạp của vật thể (ví dụ: các vật thể tròn, uốn lượn như chai, lọ, cơ thể người cần nhiều điểm neo hơn vật thể thẳng như bàn, tủ).
Điểm neo phải đặt chính xác tại pixel ranh giới cuối cùng của vật thể. Không được lấn ra ngoài background quá 1 pixel và không cắt lẹm vào thân vật thể.
Chỉ vẽ mask cho phần nhìn thấy được của vật thể. Tuyệt đối không tự ước lượng hoặc vẽ bù các phần bị che khuất bởi vật thể khác nằm phía trước.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh, record bowl 0.53 bị record bowl 0.74 che một phần. Ở đây, tôi dùng guideline vì trường hợp này không quá mơ hồ và có thể quyết định chắc chắn theo quy tắc đã đề cập trong guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất gán cho toàn bộ bức ảnh từ danh mục định sẵn  |Ảnh traffic có nhiều đối tượng khác nhau (xe bus, người, ô tô) nhưng mô hình chỉ có thể gán một nhãn duy nhất là cab (taxi/xe dịch vụ) với độ tin cậy ~51.1%. Điều này gây tranh cãi vì bỏ sót các thực thể khác | Chọn một nhãn mô tả đúng nhất chủ thể chính của toàn bộ bức ảnh theo quy tắc phân cấp của danh mục | Kiểm tra xem nhãn được chọn có phản ánh đúng nội dung tổng thể của bức ảnh hay không; giải quyết các trường hợp ảnh có nhiều chủ thể gây nhiễu |
| Phát hiện vật thể |Tập hợp Bounding Box xác định bằng tọa độ pixel dạng bbox_xyxy [x_min, y_min, x_max, y_max] đi kèm một nhãn lớp cụ thể cho từng vật thể  | Trong ảnh kitchen, khi giảm ngưỡng tự tin xuống 0.20, mô hình phát hiện ra rất nhiều đối tượng bát và cốc nhỏ chồng chéo, trong khi ở ngưỡng 0.60 nhiều vật thể thực tế bị bỏ sót hoàn toàn | Vẽ hộp bao quanh sát từng đối tượng đích, đảm bảo không cắt lẹm và không vẽ quá rộng; chọn đúng phân loại cho từng hộp | Kiểm tra độ khít của các hộp bao, phát hiện các hộp bị gán sai nhãn hoặc các vật thể bị bỏ sót trong ảnh |
| Instance segmentation | Polygon gồm danh sách các điểm tọa độ liên tiếp polygon_xy [[x1, y1], [x2, y2], ...] bao khít theo đường viền thực tế của từng đối tượng độc lập | Trong ảnh kitchen, các vật thể bị che khuất một phần rất khó vẽ đường viền chính xác. Ranh giới giữa các đối tượng chồng lấn dễ bị nhập nhằng | Click thủ công từng điểm dọc theo ranh giới của vật thể để tạo thành một đa giác khép kín bao bọc chính xác hình dáng thực tế của đối tượng | Kiểm tra độ mịn và độ chính xác của đường viền đa giác, đảm bảo các đối tượng đứng sát nhau không bị gộp chung thành một đa giác đơn lẻ |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ba ảnh COCO công khai có sẵn, tuyệt đối không tải lên hình ảnh cá nhân hoặc dữ liệu nội bộ của doanh nghiệp lên môi trường Colab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng và báo cho lab coach trước.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [v] `REPORT.md` (file này đã điền)
- [v] `day1_lab_outputs/classification_predictions.json`
- [v] `day1_lab_outputs/detection_predictions.json`
- [v] `day1_lab_outputs/segmentation_predictions.json`
- [v] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [v] `day1_lab_outputs/visuals/classification_top5.png`
- [v] `day1_lab_outputs/visuals/detection_predictions.png`
- [v] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [v] Ô validation cuối notebook báo `PASS`
- [v] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v] Đã nộp **link repository** trên LMS
