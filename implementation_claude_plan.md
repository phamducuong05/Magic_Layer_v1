# Kế hoạch tích hợp Claude Vision để tự động sinh keyword cho SAM3

## 1. Mục tiêu

Thêm một bước VLM ở đầu pipeline:

```text
Ảnh upload
  → Claude Vision phân tích foreground và occlusion
  → danh sách keyword tiếng Anh có cấu trúc
  → SAM3 segmentation
  → completion / depth ordering / reconstruction
  → matting / layer extraction / background inpainting
```

Claude cần ưu tiên **recall của occluder** nhưng không liệt kê tràn lan:

- Lấy các vật thể foreground quan trọng, có diện tích hoặc vai trò thị giác lớn.
- Bỏ vật thể nhỏ không quan trọng, chi tiết trang trí và vật thể thuộc background.
- Ngoại lệ bắt buộc: vẫn lấy vật thể nhỏ nếu nó che một phần đáng kể của vật thể chính.
- Trả danh từ hoặc cụm danh từ tiếng Anh ngắn, phù hợp làm text prompt cho SAM3.
- Mỗi loại vật thể chỉ xuất hiện một lần; SAM3 hiện có thể trả nhiều mask cho cùng một keyword.

Không thể bảo đảm tuyệt đối rằng một VLM luôn phát hiện mọi occluder từ một ảnh đơn. Phần triển khai tập trung vào prompt rõ ràng, structured output và logging; việc đánh giá chất lượng đầu ra sẽ do chủ dự án thực hiện riêng.

## 2. Phạm vi phiên bản hiện tại và định hướng

### Phiên bản hiện tại

- `keywords` trên `/api/process-image` trở thành optional.
- Nếu request có `keywords`, backend vẫn gọi Claude một lần để:
  1. xét từng target theo `input_index`;
  2. chỉ chấp nhận refinement bảo toàn object type và số ít/số nhiều;
  3. tìm exhaustive các root-level foreground occluder của từng target;
  4. union occluder theo rank rồi ghép target trước occluder.
- Nếu request không có `keywords`, backend bắt buộc gọi Claude để sinh keyword trước khi chạy SAM3.
- Chỉ gọi Claude một lần cho mỗi ảnh.
- Target và occluder có quota riêng: tối đa `10 + 10` keyword.

Manual override chỉ là cơ chế chuyển tiếp, không phải UX đích.

### Định hướng sản phẩm

Trong phiên bản sau, xóa hoàn toàn prompt/keyword do người dùng nhập. Backend luôn sử dụng Claude hoặc một VLM khác để kiểm soát chất lượng nhận diện object và occluder. Thiết kế service trong kế hoạch này phải cô lập nhà cung cấp để có thể thay Claude mà không sửa pipeline SAM3.

Việc gộp các object thành một khối kéo thả được ghi ở mục **Future Work**, chưa triển khai trong phạm vi hiện tại.

## 3. Đánh giá plan cũ

Plan cũ đúng về vị trí tích hợp tổng quát nhưng cần sửa các điểm sau:

1. `claude-3-5-sonnet-20241022` đã cũ và không nên là model mặc định mới.
2. Chuỗi phân cách bằng dấu phẩy không đủ chắc chắn cho đầu vào máy; nên dùng Structured Outputs với JSON Schema.
3. `anthropic>=0.20.0` quá cũ so với API `output_config.format`.
4. Gọi client đồng bộ bên trong endpoint `async` sẽ chặn event loop; cần `AsyncAnthropic`.
5. Không nên khởi tạo service toàn cục theo cách làm server hỏng ngay khi thiếu API key. Manual override và `/health` vẫn phải hoạt động.
6. Không nên log toàn bộ raw response hoặc exception upstream ra client; cần tránh lộ dữ liệu và thông tin cấu hình.
7. Script `backend/test_claude_vision.py` không phải automated test. Cần pytest với mock và endpoint tests.
8. Prompt “main distinct foreground objects” chưa diễn đạt ngoại lệ occluder nhỏ, nên dễ bỏ sót nguyên nhân làm completion/inpainting xấu.
9. Plan cũ không xử lý `stop_reason="refusal"` hoặc `"max_tokens"`, response rỗng, keyword trùng, keyword quá dài và giới hạn tối đa.
10. `python-dotenv` không cần được gọi bằng side effect trong service. API key phải đến từ environment; file `.env` chỉ phục vụ local development và đã được `.gitignore`.

## 4. Kiến trúc đề xuất

### 4.1 Interface độc lập nhà cung cấp

Tạo protocol chung:

```python
@dataclass(frozen=True)
class KeywordExtractionResult:
    keywords: list[str]
    occluders: list[str] = field(default_factory=list)
    target_results: tuple[TargetKeywordExtraction, ...] = ()
    visible_person_count: int | None = None


class KeywordExtractor(Protocol):
    async def extract_keywords(
        self,
        image: Image.Image,
        target_keywords: Sequence[str] | None = None,
    ) -> KeywordExtractionResult:
        ...
```

`ClaudeVisionKeywordExtractor` triển khai protocol này. `main.py` chỉ phụ thuộc vào `KeywordExtractor`, không phụ thuộc trực tiếp vào chi tiết Anthropic. Sau này có thể thêm Gemini, Qwen-VL hoặc VLM local.

### 4.2 Structured output

Nhánh không có keyword người dùng trả:

```json
{
  "visible_person_count": 1,
  "keywords": ["person", "dog", "wooden chair"]
}
```

Nhánh có keyword người dùng trả kết quả indexed theo từng target:

```json
{
  "visible_person_count": 1,
  "target_results": [
    {
      "input_index": 0,
      "refined_keyword": "car",
      "occluders": ["person", "tree"]
    }
  ]
}
```

Backend yêu cầu đúng một `input_index` cho mỗi target. Target một token luôn
được giữ nguyên. Target dài chỉ nhận head noun có trong simple-object
vocabulary hoặc canonical alias có kiểm soát; candidate sai semantic hoặc
cardinality fallback về input gốc. Mỗi `occluders` được phép rỗng và được sắp
xếp từ mức che mạnh nhất đến yếu nhất.

Sau khi parse vẫn phải chuẩn hóa cục bộ:

- `strip()` khoảng trắng.
- Bỏ chuỗi rỗng.
- Bỏ trùng không phân biệt hoa thường nhưng giữ thứ tự.
- Từ chối keyword dài bất thường, câu hoàn chỉnh hoặc kết quả quá giới hạn.
- Không tự suy đoán synonym ngoài alias map.
- Với 1–3 người trong auto/occluder output, cấm `people`; chỉ cho phép
  `people` khi có hơn 3 người. Quy tắc này không rewrite manual target.

### 4.3 Model

Model là cấu hình, không hard-code trong service.

Model mặc định là `claude-haiku-4-5-20251001`: bản pinned này hỗ trợ vision
và structured outputs, đồng thời phù hợp hơn về latency và chi phí cho tác vụ
trích xuất keyword/occluder có output ngắn. Chỉ nâng lên Sonnet nếu kết quả thực
tế cho thấy Haiku bỏ sót các quan hệ occlusion quan trọng.

Structured Outputs hiện dùng API chính thức:

```python
output_config={
    "format": {
        "type": "json_schema",
        "schema": KEYWORD_OUTPUT_SCHEMA,
    }
}
```

Không sử dụng beta header hoặc tham số cũ `output_format` trong `messages.create()`.

### 4.4 Hai prompt theo nhánh

Hai prompt cùng chèn nguyên block `COMMON_STRICT_RULES`; không duy trì hai bản
sao riêng để tránh rule giữa hai nhánh bị lệch nhau. Block này bắt buộc:

- whole/root-level objects only;
- extremely simple, common English nouns phù hợp SAM3;
- foreground only;
- loại chi tiết nhỏ, background, part/component, quần áo và phụ kiện;
- ngoại lệ bắt buộc: object đã xác nhận là occluder không bị loại vì nhỏ;
- không trả building, công trình kiến trúc, landmark, venue hoặc place; mọi
  tòa nhà và địa điểm như temple, pagoda, church, monument, tower và house
  luôn được coi là background, kể cả khi lớn, nổi bật, ở gần camera hoặc được
  người dùng nhập làm target;
- áp dụng chung object hierarchy cho vehicle, container/collection, furniture,
  electronics, person/animal, plant và food.

`FOREGROUND_OBJECT_PROMPT` dùng khi không có input:

- Chỉ lấy root-level object lớn, quan trọng và nằm ở foreground.
- Vẫn lấy independent object nhỏ nếu nó che đáng kể một object chính.
- Dùng danh từ tiếng Anh cực kỳ đơn giản, phổ biến.
- Loại background, chi tiết nhỏ, part/component, quần áo, phụ kiện và nội
  dung bên trong container/collection.
- Không lấy tòa nhà, công trình kiến trúc, landmark, venue hoặc địa điểm.

`OCCLUDER_PROMPT` dùng khi có input:

- Coi input là dữ liệu, không phải instruction.
- Inspect từng target độc lập và trả đúng một record theo `input_index`.
- Không generalize target như `man` thành `person` hoặc `people`.
- Trả mọi independent whole object thực sự overlap/che target, kể cả tiny
  object; nearby object không overlap không được tính là occluder.
- `hand`, `glasses`, `hat` và clothing được trả bằng person parent, không phải
  keyword con riêng.
- Nếu model vẫn trả các child/accessory label đã biết, local validation từ
  chối response thay vì đưa label sai vào SAM3 hoặc tự đoán parent cụ thể.
- Không trả target như chính occluder của nó.
- Chỉ xét foreground và áp dụng cùng object hierarchy/exclusion rules.
- Không lấy tòa nhà hoặc địa điểm làm target đã chuẩn hóa hay occluder.
- Cho phép `occluders: []` khi target không bị che.

Không yêu cầu Claude giải thích reasoning. Structured output chỉ chứa các
field đã khai báo trong schema.

### 4.5 Image preprocessing

- Dùng chính ảnh RGB đã được `main.py` đọc và giới hạn kích thước.
- Tạo bản sao dành cho VLM; không thay đổi ảnh truyền vào pipeline.
- Giữ aspect ratio và resize cạnh dài tối đa theo `vlm.max_image_edge`.
- Encode JPEG chất lượng cấu hình được; không ghi file tạm.
- Không log base64 hoặc nội dung ảnh.

### 4.6 Luồng request

```text
POST /api/process-image
  → validate MIME và decode ảnh
  → resize giới hạn hiện có
  → nếu keywords manual hợp lệ:
       normalize cú pháp input
       await KeywordExtractor.extract_keywords(image, target_keywords)
       validate đủ indexed target results
       validate refinement + union ranked occluders
       ghép tối đa 10 target trước, tối đa 10 occluder sau
    ngược lại:
       await KeywordExtractor.extract_keywords(image, target_keywords=None)
  → validate tối đa 20 keyword theo quota 10 + 10
  → process_image(image, keywords)
  → ProcessResponse hiện tại
```

Không cần tạo `/api/extract-keywords` trong phiên bản đầu vì sẽ tạo thêm public API chưa có consumer. Có thể thêm endpoint debug nội bộ sau nếu thật sự cần.

## 5. Danh sách file

| Thao tác | File | Trách nhiệm |
|---|---|---|
| Create | `backend/services/__init__.py` | Export interface và Claude implementation. |
| Create | `backend/services/keyword_extractor.py` | Khai báo `KeywordExtractor`, lỗi domain và hàm normalize/validate. |
| Create | `backend/services/claude_vision.py` | Encode ảnh, gọi `AsyncAnthropic`, parse structured output. |
| Modify | `backend/config.yaml` | Thêm cấu hình `vlm`; không chứa API key thật. |
| Modify | `backend/config.py` | Thêm `get_vlm_config()` với validation kiểu dữ liệu. |
| Modify | `backend/main.py` | Làm `keywords` optional, resolve extractor và map lỗi HTTP. |
| Modify | `backend/requirements.txt` | Thêm Anthropic SDK hỗ trợ API hiện tại. |
| Create | `.env.example` | Chỉ chứa tên biến và giá trị placeholder. |
| Create | `tests/test_keyword_extractor.py` | Test normalize, deduplicate và validation. |
| Create | `tests/test_claude_vision.py` | Mock Anthropic client; test payload, schema và lỗi. |
| Create | `tests/test_main_claude_keywords.py` | Test routing manual/auto và HTTP error mapping. |
| Create | `tests/integration/test_claude_vision_live.py` | Optional live test, skip khi không có API key. |

Không cần sửa `backend/pipeline/orchestrator.py`: interface hiện tại đã nhận `Sequence[str]`.

## 6. Các bước triển khai

### Bước 1: Dependency và environment

Trong `backend/requirements.txt`:

```text
anthropic>=0.104,<1
```

Lower bound phải được kiểm tra lại khi triển khai nếu code dùng API mới hơn. Không pin API key hoặc model key trong source.

Tạo `.env.example`:

```env
ANTHROPIC_API_KEY=replace-with-your-key
```

Production phải inject secret qua environment/secret manager. Local có thể dùng `uvicorn --env-file .env`; `.env` hiện đã nằm trong `.gitignore`.

### Bước 2: Cấu hình VLM

Thêm vào `backend/config.yaml`:

```yaml
vlm:
  active: claude
  max_keywords: 10
  max_occluders: 10
  max_keyword_length: 80
  claude:
    api_key_env: ANTHROPIC_API_KEY
    model: claude-haiku-4-5-20251001
    max_tokens: 256
    timeout_seconds: 30
    max_retries: 2
    max_image_edge: 1568
    jpeg_quality: 85
```

`ConfigManager.get_vlm_config()` phải:

- Xác nhận `vlm` và provider config là mapping.
- Xác nhận có `active`.
- Merge giới hạn chung và provider config.
- Không đọc hoặc trả API key; chỉ trả tên biến môi trường.

### Bước 3: Domain interface và validation

Trong `backend/services/keyword_extractor.py`:

```python
from typing import Protocol
from PIL import Image


class KeywordExtractionError(RuntimeError):
    """Base error safe for mapping at the API boundary."""


class KeywordExtractorUnavailable(KeywordExtractionError):
    """Extractor is not configured or its upstream service is unavailable."""


class InvalidKeywordExtraction(KeywordExtractionError):
    """VLM returned no usable keyword output."""


class KeywordExtractor(Protocol):
    async def extract_keywords(
        self,
        image: Image.Image,
        target_keywords: Sequence[str] | None = None,
    ) -> KeywordExtractionResult:
        ...
```

Thêm hàm thuần `normalize_keywords(values, *, max_keywords, max_length)`. Hàm này dùng chung cho Claude output và manual override để hai đường có cùng contract.

### Bước 4: Claude implementation

`ClaudeVisionKeywordExtractor`:

- Nhận config và optional client qua constructor để dễ test.
- Tạo `AsyncAnthropic` lazily khi lần đầu thật sự cần gọi.
- Đọc `ANTHROPIC_API_KEY` tại thời điểm tạo client.
- Encode ảnh trong helper thuần riêng.
- Gọi `await client.messages.create(...)`.
- Truyền image block trước text block.
- Truyền `output_config.format` với JSON Schema.
- Kiểm tra `stop_reason`; `refusal` và `max_tokens` không được parse như thành công.
- Lấy đúng text content block, `json.loads`, rồi normalize.
- Không `except Exception: raise e`; giữ traceback bằng `raise` và chỉ chuyển các lỗi Anthropic cụ thể sang lỗi domain.

Pseudo-interface:

```python
class ClaudeVisionKeywordExtractor:
    def __init__(self, settings: dict, client=None):
        ...

    async def extract_keywords(
        self,
        image: Image.Image,
        target_keywords: Sequence[str] | None = None,
    ) -> KeywordExtractionResult:
        ...
```

### Bước 5: Tích hợp FastAPI

Thay đổi tạm thời:

```python
from typing import Optional

keywords: Optional[str] = Form(
    None,
    description="Optional target labels; VLM simplifies them and finds occluders",
)
```

Tạo helper tại boundary:

```python
async def resolve_keywords(
    image: Image.Image,
    supplied_keywords: str | None,
    extractor: KeywordExtractor,
) -> list[str]:
    ...
```

Để test dễ dàng, lấy extractor thông qua dependency/helper có thể monkeypatch, không đóng cứng client Anthropic trong endpoint.

HTTP mapping:

| Trường hợp | Status | Nội dung |
|---|---:|---|
| Ảnh/MIME không hợp lệ | 400 | Giữ hành vi hiện tại. |
| Keyword người dùng không hợp lệ trước khi gọi VLM | 400 | Thông báo input không hợp lệ. |
| Thiếu API key khi cần extraction | 503 | VLM chưa sẵn sàng; không lộ tên/value secret ngoài mức cần thiết. |
| Anthropic timeout/connection/rate limit | 503 | Dịch vụ phân tích ảnh tạm thời không khả dụng. |
| Anthropic authentication/config error | 503 | Cấu hình VLM không hợp lệ. |
| Response refusal/truncated/không có keyword | 502 | Upstream trả kết quả không dùng được. |
| Pipeline sau keyword extraction lỗi | 500 | Giữ boundary hiện tại. |

Không fallback sang keyword đoán cứng vì có thể âm thầm bỏ occluder và làm output xấu.

### Bước 6: Automated tests

#### `tests/test_keyword_extractor.py`

- Chuẩn hóa whitespace.
- Deduplicate case-insensitive, giữ thứ tự đầu tiên.
- Từ chối list rỗng.
- Từ chối quá 10 keyword.
- Từ chối keyword quá dài.

#### `tests/test_claude_vision.py`

- Encode ảnh RGB thành đúng image content block.
- Payload đặt image trước prompt.
- Payload dùng model/config và `output_config.format`.
- Parse JSON hợp lệ thành `KeywordExtractionResult`.
- Nhánh target dùng indexed `target_results`; thiếu, trùng hoặc index ngoài
  phạm vi bị từ chối.
- Cho phép per-target `occluders` rỗng.
- Manual `man` không thể bị đổi thành `people`.
- Auto/occluder `people` bị từ chối khi `visible_person_count <= 3`.
- Prompt bắt buộc tiny-occluder exception và person-parent hierarchy.
- Mock response có keyword trùng và kiểm tra normalize.
- Thiếu `ANTHROPIC_API_KEY` sinh `KeywordExtractorUnavailable`.
- Timeout, connection, rate limit và auth error được map đúng.
- `stop_reason="refusal"` sinh `InvalidKeywordExtraction`.
- `stop_reason="max_tokens"` sinh `InvalidKeywordExtraction`.
- Response rỗng hoặc JSON sai sinh `InvalidKeywordExtraction`.
- Tests không gọi mạng và không cần API key thật.

#### `tests/test_main_claude_keywords.py`

- Có manual keywords: gọi Claude đúng một lần, validate indexed refinement,
  union per-target occluder và truyền danh sách đã ghép vào pipeline.
- Không có manual keywords: await extractor đúng một lần với `target_keywords=None` và truyền kết quả vào `process_image`.
- Blank manual prompt được xem như nhánh tự động.
- Target đã validate luôn đứng trước occluder; quota độc lập cho phép tối đa
  10 target và 10 occluder.
- Auto extraction rỗng/lỗi trả đúng HTTP status.
- Vẫn giữ giới hạn định dạng ảnh và số keyword.
- `/health` hoạt động khi không có API key.

#### Optional live test

`tests/integration/test_claude_vision_live.py`:

- Đánh dấu `integration`.
- Skip nếu thiếu `ANTHROPIC_API_KEY`.
- Chỉ kiểm tra response có 1..10 keyword hợp lệ; không dùng làm CI mặc định.

## 7. Logging và vận hành

Log:

- provider/model
- thời gian VLM
- số keyword trả về
- danh sách keyword đã normalize nếu chính sách dữ liệu cho phép
- loại lỗi và request ID của Anthropic nếu có

Không log:

- API key
- image base64
- raw image
- full upstream exception trả cho client

Retry chỉ dùng cho lỗi transient và giao cho Anthropic SDK với `max_retries` giới hạn. Không retry auth error, refusal hoặc invalid structured output một cách mù quáng.

## 8. Acceptance criteria cho phiên bản hiện tại

1. Request không có `keywords` gọi Claude và truyền danh sách tiếng Anh hợp lệ vào SAM3.
2. Request có manual keywords gọi Claude đúng một lần để validate từng target
   và tìm exhaustive occluder theo target.
3. Claude output luôn đi qua JSON Schema và local validation.
4. Endpoint không block event loop bởi Anthropic sync client.
5. Thiếu key không làm server hoặc `/health` crash.
6. Không có secret trong YAML, source, logs hoặc test fixtures.
7. Unit/endpoint tests chạy offline bằng mock.
8. Không thay đổi contract `process_image(image, Sequence[str])`.
9. Keyword người dùng không bị đổi object type hoặc singular/plural; refinement
   không hợp lệ fallback về input.
10. Không có occluder là kết quả hợp lệ; pipeline vẫn chạy bằng target đã
    validate.
11. Tiny confirmed occluder luôn có keyword root-parent.
12. Target không còn chiếm quota occluder.

## 9. Future Work

### 9.1 Bỏ hoàn toàn prompt do người dùng nhập

Khi chủ dự án sẵn sàng chuyển sang luồng tự động hoàn toàn:

- Xóa trường `keywords` khỏi public request.
- Backend luôn gọi `KeywordExtractor`.
- Có thể giữ override chỉ trong internal/debug tooling có kiểm soát, không đưa ra UX.
- Thêm provider registry/factory để đổi Claude sang VLM khác bằng config.
- Có thể đánh giá chiến lược hai-pass nếu một lần gọi vẫn bỏ sót occluder:
  1. phát hiện foreground chính;
  2. audit riêng các vật thể đang che những foreground đó.

Hai-pass chỉ triển khai nếu chủ dự án xác nhận lợi ích đủ lớn so với latency và chi phí.

### 9.2 Gộp object thành một khối kéo thả

Yêu cầu tương lai: nếu một object nằm trọn hoặc phần lớn bên trong bounding box của object khác, cân nhắc gộp chúng thành một draggable group.

Không nên xóa object con hoặc merge mask quá sớm. Completion, depth ordering và inpainting vẫn cần object/mask riêng để hiểu occluder. Việc gộp nên xảy ra ở **presentation grouping stage sau khi các quan hệ occlusion và reconstruction đã được xử lý**:

```text
individual detected objects
  → occlusion/completion/reconstruction giữ riêng từng object
  → xác định presentation groups
  → trả một draggable group chứa nhiều layer/mask nội bộ
```

Bounding-box containment đơn thuần dễ gộp nhầm, ví dụ người đứng trước cửa sổ hoặc vật thể nằm trong một bbox lớn nhưng không cùng semantic unit. Thiết kế tương lai nên kết hợp:

- `intersection_area(child_bbox) / area(child_bbox)`
- quan hệ mask containment/overlap
- tỷ lệ diện tích object con/object cha
- quan hệ occlusion hiện có
- quy tắc semantic hoặc VLM grouping khi cần

Contract output tương lai nên thêm `group_id` hoặc `LayerGroup`, trong khi vẫn bảo toàn object-level diagnostics. Phần này cần một design/implementation plan riêng trước khi triển khai.

## 10. Tài liệu Anthropic cần kiểm tra khi triển khai

- Model IDs và versioning: <https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions>
- Chọn model: <https://platform.claude.com/docs/en/about-claude/models/choosing-a-model>
- Vision input: <https://platform.claude.com/docs/en/build-with-claude/vision>
- Structured Outputs: <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- Python Messages API: <https://platform.claude.com/docs/en/api/python/messages/create>
- Python SDK: <https://github.com/anthropics/anthropic-sdk-python>

Model và SDK thay đổi theo thời gian; phải kiểm tra lại các trang chính thức này ngay trước khi triển khai dependency/model pin.
