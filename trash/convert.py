import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/best_model.pth"
ONNX_PATH = "best_model.onnx"

INPUT_SIZE = 96


# ============================================================
# 1. RECREATE EXACT MODEL ARCHITECTURE
# ============================================================

def create_model():

    model = models.mobilenet_v2(weights=None)

    # Exact classifier from the trained checkpoint:
    #
    # 1280 -> 256 -> 1
    #
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1)
    )

    return model


# ============================================================
# 2. LOAD CHECKPOINT
# ============================================================

print("[1/5] Loading PyTorch model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded")

# Your checkpoint contains:
# epoch
# model_state_dict
# optimizer_state_dict
# val_acc
# config

state_dict = checkpoint["model_state_dict"]


# ============================================================
# 3. REMOVE "model." PREFIX
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("model."):
        key = key[len("model."):]

    clean_state_dict[key] = value


# ============================================================
# 4. LOAD WEIGHTS
# ============================================================

model = create_model()

model.load_state_dict(clean_state_dict)

model.eval()

print("✓ Model architecture and weights loaded successfully")


# ============================================================
# 5. DUMMY INPUT
# ============================================================

print("[2/5] Preparing dummy input...")

dummy_input = torch.randn(
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE
)

print(f"✓ Input shape: {tuple(dummy_input.shape)}")


# ============================================================
# 6. EXPORT ONNX
# ============================================================

print("[3/5] Exporting to ONNX...")

with torch.no_grad():

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,

        input_names=["images"],
        output_names=["logits"],

        dynamic_axes={
            "images": {
                0: "batch"
            },
            "logits": {
                0: "batch"
            }
        },

        opset_version=17,

        do_constant_folding=True
    )

print(f"✓ ONNX saved to: {ONNX_PATH}")


# ============================================================
# 7. VERIFY ONNX STRUCTURE
# ============================================================

print("[4/5] Checking ONNX model...")

onnx_model = onnx.load(ONNX_PATH)

onnx.checker.check_model(onnx_model)

print("✓ ONNX model is valid")


# ============================================================
# 8. COMPARE PYTORCH VS ONNX
# ============================================================

print("[5/5] Comparing PyTorch vs ONNX output...")

# PyTorch
with torch.no_grad():

    pytorch_output = model(
        dummy_input
    ).numpy()


# ONNX Runtime
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

onnx_output = session.run(
    ["logits"],
    {
        "images": dummy_input.numpy()
    }
)[0]


# ============================================================
# 9. COMPARE
# ============================================================

difference = np.max(
    np.abs(
        pytorch_output -
        onnx_output
    )
)


print()
print("========================================")
print("       ONNX CONVERSION RESULT")
print("========================================")

print(f"PyTorch output : {pytorch_output}")
print(f"ONNX output    : {onnx_output}")
print(f"Max difference : {difference}")

print("========================================")


if difference < 1e-4:

    print("✓ SUCCESS!")
    print("✓ ONNX output matches PyTorch")

else:

    print("⚠ WARNING!")
    print("PyTorch and ONNX outputs differ")


print()
print(f"Output file: {ONNX_PATH}")