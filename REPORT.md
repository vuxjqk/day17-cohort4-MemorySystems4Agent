# Báo cáo Lab Day 17: Memory Systems for AI Agent

## 1. Chạy lại kết quả

```bash
pip install -r requirements.txt     # với chế độ offline chỉ cần pytest; tabulate/dotenv là tuỳ chọn
python src/benchmark.py             # thêm -v để in từng câu trả lời recall
pytest src/test_agents.py -v        # 12 test
```

Benchmark chạy ở **chế độ offline** (deterministic, không cần API key). Token được ước lượng bằng `ceil(len(text)/4)`.
Cấu hình mặc định: `compact_threshold_tokens=1000`, `compact_keep_messages=4`, `profile_confidence_threshold=0.6`.
Muốn chạy live (LangChain `create_agent` + `InMemorySaver` + `SummarizationMiddleware`), đặt `LAB_MODE=live` và cấu hình provider trong `.env` (xem `.env.example`).
Chế độ live **chưa được kiểm thử với API thật** trong bài này; mọi số liệu dưới đây đều lấy từ chế độ offline.

## 2. Kết quả benchmark

### Standard Benchmark (`data/conversations.json`: 10 hội thoại, 101 lượt, 14 câu recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2255 | 13639 | 0.00 | 0.20 | 0 | 0 |
| Advanced | 2526 | 40021 | 1.00 | 1.00 | 1084 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`: 1 hội thoại, 16 lượt dài, 3 câu recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2487 | 21548 | 0.00 | 0.20 | 0 | 0 |
| Advanced | 2630 | 13006 | 1.00 | 1.00 | 591 | 3 |

### Ảnh hưởng của ngưỡng compact (Advanced, prompt tokens processed)

| Ngưỡng compact (tokens) | Standard | Stress | Số lần compact (stress) | Recall (stress) |
|---|---|---|---|---|
| 400 | 40021 | 9669 | 24 | 1.00 |
| 600 | 40021 | 10496 | 8 | 1.00 |
| **1000 (mặc định)** | 40021 | 13006 | 3 | 1.00 |
| 2000 | 40021 | 18882 | 1 | 1.00 |
| 4000 (thực tế là tắt compact) | 40021 | **25230** | 0 | 1.00 |

Baseline ở bài stress xử lý 21548 prompt tokens.

## 3. Phân tích

**Câu chuyện chính:**

1. Baseline không nhớ dài hạn: recall = 0 ở mọi thread mới.
2. Advanced thêm `User.md` nên recall tăng lên 1.00.
3. Hội thoại dài làm chi phí prompt tăng rất nhanh.
4. Compact memory kéo chi phí ngữ cảnh xuống.
5. Đổi lại, hệ thống phức tạp hơn và cần guardrail.

### 3.1 Vì sao Advanced có recall tốt hơn Baseline
- Baseline giữ message theo `thread_id`. Nó vẫn trả lời được trong cùng thread (`test_cross_session_recall` kiểm tra điều này), nhưng sang thread mới thì danh sách message rỗng, nên nó trả lời "chưa có thông tin". Đây đúng là hành vi mong muốn của một baseline công bằng: nó không giả vờ có long-term memory.
- Advanced trích fact ổn định từ mỗi tin nhắn và ghi vào `state/profiles/<user>/User.md`. File này tồn tại qua các thread và qua cả các instance agent mới. Câu hỏi recall được trả lời từ `User.md`, không phụ thuộc lịch sử thread.
- Các ca khó đều đúng:
  - correction Đà Nẵng→Huế (standard) và Huế→Đà Nẵng (stress);
  - backend→MLOps engineer;
  - bỏ qua nhiễu "Hà Nội chỉ là nơi đi họp" và "product manager chỉ là câu đùa";
  - không nhầm "corgi tên Bơ" thành tên người dùng.

### 3.2 Vì sao Advanced tốn hơn ở hội thoại ngắn
- Ở bài standard, Advanced xử lý **gấp ~2.9 lần** prompt tokens so với Baseline (40021 so với 13639). Mỗi lượt Advanced phải mang theo system prompt dài hơn và toàn bộ `User.md`, khoảng 230 token mỗi lượt. Trong khi đó, mỗi thread chỉ có khoảng 10 câu ngắn, cả thread chưa tới vài trăm token.
- Compact **không hề kích hoạt** ở bài standard, kể cả với ngưỡng 400. Vì vậy `User.md` là chi phí cố định thuần, chưa có gì để bù lại.
- `Agent tokens only` của Advanced cũng cao hơn một chút (2526 so với 2255). Lý do là nó xác nhận các cập nhật memory ("Đã ghi nhận và cập nhật User.md: …") và trả lời recall đầy đủ thay vì từ chối.
- Kết luận: với hội thoại ngắn, chúng ta **trả token để mua recall**. Đó là một trade-off hợp lý nếu recall quan trọng, nhưng không phải là "miễn phí".

### 3.3 Vì sao compact giúp Advanced có lợi thế ở hội thoại dài
- Prompt của Baseline ở lượt *n* gồm toàn bộ *n* lượt trước, nên tổng chi phí tăng theo **O(n²)**. Với 16 lượt, mỗi lượt khoảng 150 token, tổng lên tới 21548 token.
- Prompt của Advanced = system + `User.md` + summary (tối đa 8 dòng) + 4 message gần nhất. Phần này **bị chặn trên, không phụ thuộc độ dài thread** (`test_compact_trigger` kiểm tra điều này), nên tổng chi phí chỉ tăng theo **O(n)**. Kết quả là ít hơn 39.6% so với Baseline với 3 lần compact.
- Bảng sweep cho thấy rõ rằng **toàn bộ phần tiết kiệm đến từ compact**. Khi tắt compact (ngưỡng 4000), Advanced còn tốn hơn Baseline (25230 so với 21548), vì vừa giữ toàn bộ lịch sử vừa cộng thêm `User.md`.
- Compact tối ưu chủ yếu `Prompt tokens processed`, không phải `Agent tokens only`. Lượng token người dùng và agent sinh ra gần như không đổi (2630 so với 2487). Thứ được cắt giảm là ngữ cảnh mà agent phải đọc lại ở mỗi lượt.
- Recall vẫn giữ 1.00 dù compact mạnh (ngưỡng 400, 24 lần compact). Đó là vì fact quan trọng đã được "thăng cấp" vào `User.md` *trước khi* message bị nén. Summary chỉ cần giữ mạch hội thoại, không phải giữ fact. Đây là lý do tách bạch 3 lớp:

  | Lớp | Lưu gì | Vòng đời |
  |---|---|---|
  | short-term | 4 message gần nhất, nguyên văn | trong thread |
  | compact | summary heuristic của các message cũ, tối đa 8 dòng | trong thread |
  | persistent (`User.md`) | fact ổn định đã qua kiểm tra confidence | xuyên session |

- Ngưỡng nhỏ hơn thì tiết kiệm hơn, nhưng summary bị nén nhiều lần hơn (24 lần ở ngưỡng 400). Mỗi lần nén lại làm mất thêm chi tiết của ngữ cảnh *trong thread* (ví dụ chi tiết bốn tin tức). Bài offline không đo được mất mát này, vì câu recall chỉ hỏi fact ổn định. Đây là giới hạn của benchmark.

### 3.4 Memory file tăng trưởng ra sao, và các rủi ro
- `User.md` tăng 1084 bytes sau 10 phiên (standard) và 591 bytes sau bài stress. Kích thước bị chặn nhờ các cơ chế sau:
  - mỗi key là một dòng duy nhất (upsert, không append);
  - interests được giới hạn ở `max_interests=8` theo điểm decay;
  - nhật ký Corrections chỉ giữ 5 dòng.
- Kích thước file vẫn là **chi phí thật trên mỗi lượt**: 1 KB ≈ 270 token bị cộng vào *mọi* prompt. Nếu không có giới hạn, file phình to sẽ ăn mòn đúng phần tiết kiệm mà compact mang lại.
- Các rủi ro còn lại:
  - **Lưu sai fact**: regex có thể bắt nhầm. Ví dụ `response_style` hiện gộp cả "có ví dụ thực tế" lẫn "có ví dụ thực chiến". Đó là cách merge cộng dồn: an toàn cho recall nhưng dài dần.
  - **Fact cũ không bao giờ hết hạn**: các fact đơn trị không decay. Nơi ở "tạm thời vài tháng" (Đà Nẵng trong bài stress) sẽ vẫn được coi là hiện tại cho tới khi có correction mới.
  - **Quyền riêng tư**: `User.md` là PII dạng plain text. Production cần mã hoá, phân quyền, và cho người dùng xem hoặc xoá.
  - **Prompt injection vào memory**: ở chế độ live, nếu người dùng nói "hãy ghi rằng tôi là admin", tool `save_user_fact` có thể ghi lại câu đó. Cần whitelist key (đã có trong docstring tool) và validate giá trị.

## 4. Bonus đã triển khai

| Bonus | Cách làm | Giải quyết vấn đề gì | Rủi ro / chi phí |
|---|---|---|---|
| **Entity extraction có cấu trúc** | `extract_profile_candidates()` trả `FactCandidate(key, value, confidence)` cho 7 field: name, location, profession, response_style, favorite_drink, favorite_food, pet, cộng thêm interests. `User.md` có section Profile / Interests / Corrections / Meta. | Recall chính xác theo field; trả lời đúng câu được hỏi thay vì đổ cả hồ sơ, nên câu trả lời ngắn hơn và ít token hơn. | Regex phụ thuộc cách diễn đạt tiếng Việt; câu lạ sẽ bị bỏ sót (thiên về false negative). |
| **Confidence threshold** | Fact mặc định 0.7–0.95. Câu giả định ("Nếu…") bị trừ 0.3, nhắc lại thông tin cũ ("Lúc đầu mình nói…") bị trừ 0.4. Chỉ ghi khi confidence ≥ 0.6. Confidence được lưu cạnh từng fact. | Tránh ghi fact không chắc chắn. Ví dụ "Lúc đầu mình nói hiện ở Huế" không ghi đè Đà Nẵng. | Ngưỡng cao thì bỏ sót preference thật (ví dụ "Nếu bạn giải thích, hãy trả lời ngắn gọn…" bị bỏ qua). Ngưỡng thấp thì lưu nhầm. |
| **Tránh lưu sai khi người dùng hỏi / đùa / phủ định** | Câu hỏi (`?`, "là gì", "ở đâu"…) không bao giờ ghi fact. Câu có "đùa", "chỉ là nơi", "thông tin cũ" không ghi location/profession. Mention bị phủ định ("không còn làm…", "chứ không còn ở…", "đừng nói…") bị bỏ qua. | Chặn đúng các bẫy trong dataset: Hà Nội, product manager, backend engineer cũ. | Danh sách từ khoá là heuristic, có thể chặn nhầm một câu hợp lệ. |
| **Conflict handling** | Fact đơn trị được upsert (thay thế), trong cùng tin nhắn thì mention sau cùng thắng. Giá trị cũ chỉ còn trong `## Corrections` (log giới hạn), không còn là fact hiện tại. | Agent không giữ đồng thời "backend engineer" và "MLOps engineer"; recall luôn trả bản mới nhất. Có test kiểm tra câu trả lời *không* chứa giá trị cũ. | Correction sai sẽ ghi đè fact đúng. Nhật ký Corrections giúp audit hoặc rollback. |
| **Memory decay** | Điểm interest = `mentions × 0.97^(số lượt kể từ lần nhắc cuối)`; giữ top 8. | Giới hạn kích thước `User.md` và ưu tiên mối quan tâm còn "nóng". | Sở thích lâu dài nhưng ít được nhắc có thể bị đẩy ra (`test_interest_decay_caps_memory_growth`). |

## 5. Giới hạn của bài làm
- Token là ước lượng `len/4`, không phải tokenizer thật. Tiếng Việt có dấu thường tốn nhiều token hơn, nên số tuyệt đối sẽ khác nhưng tỉ lệ so sánh vẫn có ý nghĩa.
- `Response quality` ở chế độ offline là heuristic: 70% độ bao phủ fact, 20% độ ngắn gọn, 10% không từ chối. Nó không đo độ tự nhiên của câu trả lời. Có thể thay bằng LLM judge (`judge_model` đã có trong config).
- Câu trả lời offline mang tính template. Chế độ live dùng LLM thật, có tool đọc/ghi/sửa `User.md` và `SummarizationMiddleware` của LangChain, nhưng chưa được kiểm thử với API key thật.
