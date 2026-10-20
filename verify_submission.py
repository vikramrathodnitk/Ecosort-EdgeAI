"""Check an EcoSort ONNX model before submission.

Run these commands from the project folder:
    python verify_submission.py --model model.onnx
    python verify_submission.py --model model.onnx --val_dir data/public/val

The verifier measures CPU latency and validation accuracy. Results depend on
the computer and dataset, so review the printed measurements before submitting.
"""

import argparse
import os
import time

import numpy as np


MAX_FILE_SIZE_BYTES = int(1.50 * 1024 * 1024)
MAX_PARAMETERS = 350_000
MAX_LATENCY_MS = 35.0
MIN_ACCURACY_PERCENT = 75.0
IMAGE_SIZE = 96
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def show_result(name, status, detail):
    """Print one clearly labeled check and return its status."""
    print(f"{status:10} {name}: {detail}")
    return status


def count_model_parameters(model_path):
    """Count the values stored in the ONNX model's weight tensors."""
    import onnx

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    count = 0
    for tensor in model.graph.initializer:
        count += int(np.prod(tensor.dims, dtype=np.int64))
    return count


def prepare_image(image_path):
    """Resize and normalize one image in the same way as the training script."""
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
    # ONNX input expects channels first: (1, 3, 96, 96).
    array = np.transpose(array, (2, 0, 1))
    array = (array - IMAGE_MEAN) / IMAGE_STD
    return np.expand_dims(array, axis=0).astype(np.float32)


def evaluate_validation(session, input_name, output_name, val_dir):
    """Measure accuracy on images arranged in one folder per class."""
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation folder does not exist: {val_dir}")

    found_classes = sorted(
        name for name in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, name))
    )
    if found_classes != CLASS_NAMES:
        raise ValueError(
            "Validation folders must be the six class names in this order: "
            + ", ".join(CLASS_NAMES)
        )

    correct = 0
    total = 0
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(val_dir, class_name)
        image_names = sorted(
            name for name in os.listdir(class_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        for image_name in image_names:
            image_path = os.path.join(class_dir, image_name)
            image_input = prepare_image(image_path)
            scores = session.run([output_name], {input_name: image_input})[0]
            predicted_class = int(np.argmax(scores, axis=1)[0])
            correct += predicted_class == class_index
            total += 1

    if total == 0:
        raise ValueError(f"No supported image files found inside {val_dir}")
    return 100.0 * correct / total, total


def verify_submission(model_path, val_dir=""):
    """Run available model checks and summarize pass, fail, or incomplete."""
    print("=" * 66)
    print("EcoSort Edge-AI | Submission Verification")
    print("=" * 66)
    print(f"Model file: {model_path}\n")

    results = []
    if not os.path.isfile(model_path):
        show_result("Model file", "FAIL", "file not found")
        print("Check that the path and filename are correct.")
        return 1

    # Check the size limit without needing any optional ONNX packages.
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    size_ok = size_bytes <= MAX_FILE_SIZE_BYTES
    results.append(show_result(
        "File size", "PASS" if size_ok else "FAIL",
        f"{size_mb:.2f} MB (limit: 1.50 MB)"
    ))

    # ONNX is used to inspect model weights and validate the model file.
    try:
        parameter_count = count_model_parameters(model_path)
        parameter_ok = parameter_count <= MAX_PARAMETERS
        results.append(show_result(
            "Parameter count", "PASS" if parameter_ok else "FAIL",
            f"{parameter_count:,} (limit: {MAX_PARAMETERS:,})"
        ))
    except ImportError:
        results.append(show_result(
            "Parameter count", "SKIP", "install ONNX with: python -m pip install onnx"
        ))
    except Exception as error:
        results.append(show_result("ONNX model check", "FAIL", str(error)))

    # ONNX Runtime checks model signatures and runs CPU inference.
    session = None
    input_name = None
    output_name = None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one model input and one model output.")

        model_input = inputs[0]
        model_output = outputs[0]
        input_name, output_name = model_input.name, model_output.name
        input_shape, output_shape = model_input.shape, model_output.shape
        print(f"Input:  {input_name}, shape={input_shape}, type={model_input.type}")
        print(f"Output: {output_name}, shape={output_shape}, type={model_output.type}")

        shape_ok = (
            len(input_shape) == 4
            and input_shape[1:] == [3, IMAGE_SIZE, IMAGE_SIZE]
            and len(output_shape) == 2
            and output_shape[1] == len(CLASS_NAMES)
            and model_input.type == "tensor(float)"
            and model_output.type == "tensor(float)"
        )
        results.append(show_result(
            "Input/output format", "PASS" if shape_ok else "FAIL",
            "expected float32 (Batch, 3, 96, 96) -> (Batch, 6)"
        ))

        if shape_ok:
            # Warm up the runtime, then average repeated single-image timings.
            sample = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            for _ in range(5):
                session.run([output_name], {input_name: sample})
            timings = []
            for _ in range(50):
                start = time.perf_counter()
                session.run([output_name], {input_name: sample})
                timings.append((time.perf_counter() - start) * 1000.0)
            average_ms = float(np.mean(timings))
            latency_ok = average_ms <= MAX_LATENCY_MS
            results.append(show_result(
                "Average CPU latency", "PASS" if latency_ok else "FAIL",
                f"{average_ms:.2f} ms/image (target: <= {MAX_LATENCY_MS:.0f} ms)"
            ))
        else:
            results.append(show_result(
                "Average CPU latency", "SKIP", "cannot run with an invalid model signature"
            ))

    except ImportError:
        results.append(show_result(
            "ONNX Runtime checks", "SKIP",
            "install it with: python -m pip install onnxruntime"
        ))
    except Exception as error:
        results.append(show_result("ONNX Runtime checks", "FAIL", str(error)))

    # Local accuracy is only measured when a validation directory is provided.
    if val_dir:
        if session is None or input_name is None or output_name is None:
            results.append(show_result(
                "Validation accuracy", "SKIP", "ONNX Runtime model session is unavailable"
            ))
        else:
            try:
                accuracy, sample_count = evaluate_validation(
                    session, input_name, output_name, val_dir
                )
                accuracy_ok = accuracy >= MIN_ACCURACY_PERCENT
                results.append(show_result(
                    "Validation accuracy", "PASS" if accuracy_ok else "FAIL",
                    f"{accuracy:.2f}% on {sample_count} images (target: >= {MIN_ACCURACY_PERCENT:.0f}%)"
                ))
            except Exception as error:
                results.append(show_result("Validation accuracy", "FAIL", str(error)))
    else:
        results.append(show_result(
            "Validation accuracy", "SKIP", "provide --val_dir to measure local accuracy"
        ))

    print("\n" + "=" * 66)
    if "FAIL" in results:
        print("RESULT: Some checks failed. Review the failures before submission.")
        exit_code = 1
    elif "SKIP" in results:
        print("RESULT: Verification incomplete. Install missing packages or provide validation data.")
        exit_code = 2
    else:
        print("RESULT: All listed checks passed on this computer and validation set.")
        exit_code = 0
    print("Note: local checks do not guarantee results on the organizers' unseen data.")
    print("=" * 66)
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Check EcoSort ONNX size, parameters, format, speed, and optional accuracy."
    )
    parser.add_argument("--model", default="model.onnx", help="Path to the exported ONNX model")
    parser.add_argument(
        "--val_dir", default="",
        help="Optional validation folder containing one subfolder per class"
    )
    args = parser.parse_args()
    return verify_submission(args.model, args.val_dir)


if __name__ == "__main__":
    raise SystemExit(main())
