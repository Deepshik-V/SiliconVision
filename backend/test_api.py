import urllib.request
import json
import uuid

def test_restore_upload():
    print("--> Testing POST /api/restore with file upload...")
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    val_file = r"C:\Users\deeps\Downloads\train\train\NoisyLR\003000.npy"
    with open(val_file, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"003000.npy\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request("http://127.0.0.1:8000/api/restore", data=body, headers=headers, method="POST")

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("POST /api/restore Response:")
        print("  Status:", data["status"])
        print("  File ID:", data["file_id"])
        print("  Latency Inference:", data["latency"]["inference_ms"], "ms")
        print("  Input Shape:", data["input_metadata"]["shape"])
        print("  Output Shape:", data["output_metadata"]["shape"])
        assert data["status"] == "Success"
        assert data["output_metadata"]["shape"] == "256 × 256"

def test_evaluate_upload():
    print("\n--> Testing POST /api/evaluate with paired NoisyLR + GT upload...")
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    lr_path = r"C:\Users\deeps\Downloads\train\train\NoisyLR\003000.npy"
    gt_path = r"C:\Users\deeps\Downloads\train\train\GT\003000.npy"

    with open(lr_path, "rb") as f:
        lr_bytes = f.read()
    with open(gt_path, "rb") as f:
        gt_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"noisy_lr\"; filename=\"003000_lr.npy\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + lr_bytes + (
        f"\r\n--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"ground_truth\"; filename=\"003000_gt.npy\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + gt_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request("http://127.0.0.1:8000/api/evaluate", data=body, headers=headers, method="POST")

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("POST /api/evaluate Response:")
        print("  Status:", data["status"])
        print("  Evaluation Metrics:", data["evaluation_metrics"])
        assert data["evaluation_metrics"]["psnr_db"] > 20.0
        assert data["evaluation_metrics"]["ssim"] > 0.85
        print(f"  Verified PSNR: {data['evaluation_metrics']['psnr_db']} dB | SSIM: {data['evaluation_metrics']['ssim']}")

if __name__ == "__main__":
    test_restore_upload()
    test_evaluate_upload()
    print("\n[+] ALL API ENDPOINT TESTS PASSED SUCCESSFULLY!")
