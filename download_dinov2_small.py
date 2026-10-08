from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-with-registers-small"
LOCAL_DIR = "./models/dinov2-with-registers-small"

print("Downloading processor...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
processor.save_pretrained(LOCAL_DIR)

print("Downloading model...")
model = AutoModel.from_pretrained(MODEL_NAME)
model.save_pretrained(LOCAL_DIR)

print("Download complete.")
print(f"Saved to: {LOCAL_DIR}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
