from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image
from model.tf_mca import TFMCAModel
import uvicorn

app = FastAPI(title="TF-MCA Fruit Classification & Authentication API")

# Cấp phép cho mọi kết nối từ trình duyệt Web (Fix lỗi CORS "Failed to fetch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể dùng ["http://localhost:59981", ...] nếu muốn bảo mật
    allow_credentials=True,
    allow_methods=["*"],  # HTTP methods: GET, POST, PUT...
    allow_headers=["*"],
)

model = TFMCAModel() # Load model ngay khi khởi động server

@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    """
    Nhận diện nông sản phân loại / thật giả từ ảnh
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = model.predict(image)
        return {"filename": file.filename, "result": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_class")
async def add_new_fruit_class(
    class_name: str = Form(...), 
    description: str = Form(""), 
    files: list[UploadFile] = File(...)
):
    """
    Hỗ trợ Few-shot Class-Incremental: Thêm 1 class mới với danh sách hình ảnh (không cần huấn luyện)
    """
    try:
        images = []
        for file in files:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(img)
            
        success = model.add_class(class_name, description, images)
        total_classes = len(model.class_names)
        
        return {
            "message": f"Hệ thống đã thêm thành công loại nông sản mới: '{class_name}' với {len(images)} ảnh (prototype memory updated).",
            "total_classes": total_classes
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

