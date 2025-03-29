import onnx

model = onnx.load("mapped_models/mapped_sensorA2_and_A0.onnx")
opset_version = model.opset_import[0].version
print(f"ONNX Model Opset Version: {opset_version}")