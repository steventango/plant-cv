import base64
import multiprocessing
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litserve as ls
import numpy as np
import pytest
import requests

# Add app directory to sys.path
sys.path.append(str(Path(__file__).parent.parent / "app"))
from main import EmbeddingsAPI


TEST_DIR = Path(__file__).parent / "test_data"
TEST_IMAGE_PATH = TEST_DIR / "cat.jpg"
PORT = 8803


def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def run_server_proc(port=8000):
    """Run the server in a separate process."""
    # Initialize API with batch settings as in the main app
    api = EmbeddingsAPI(max_batch_size=1, batch_timeout=0.01)
    server = ls.LitServer(api)
    server.run(port=port, num_api_servers=2, generate_client_file=False)


def start_server(port=PORT):
    """Start the server for testing in a background process."""
    server_process = multiprocessing.Process(
        target=run_server_proc,
        args=(port,),
    )
    server_process.start()
    # Give the server a moment to start
    time.sleep(5)
    return server_process


class TestEmbeddings:
    """Test class for Embeddings API integration tests."""

    @pytest.fixture(scope="class")
    def api_url(self):
        server_process = start_server(port=PORT)
        yield f"http://localhost:{PORT}/predict"
        server_process.kill()

    @pytest.fixture(scope="class")
    def image_data(self):
        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")
        return encode_image(TEST_IMAGE_PATH)

    def test_cls_token(self, api_url, image_data):
        """Test retrieving CLS token embedding only."""
        response = requests.post(
            api_url,
            json={
                "image_data": image_data,
                "embedding_types": ["cls_token"],
            },
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        result = response.json()
        assert "cls_token" in result
        cls_token = np.array(result["cls_token"])
        assert cls_token.shape == (768,)

    def test_patch_features(self, api_url, image_data):
        """Test retrieving patch features embedding only."""
        response = requests.post(
            api_url,
            json={
                "image_data": image_data,
                "embedding_types": ["patch_features"],
            },
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        result = response.json()
        assert "patch_features" in result
        patch_features = np.array(result["patch_features"])
        assert patch_features.shape == (196, 768)

    def test_both(self, api_url, image_data):
        """Test retrieving both CLS token and patch features."""
        response = requests.post(
            api_url,
            json={
                "image_data": image_data,
                "embedding_types": ["cls_token", "patch_features"],
            },
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        result = response.json()
        assert "cls_token" in result
        cls_token = np.array(result["cls_token"])
        assert cls_token.shape == (768,)
        assert "patch_features" in result
        patch_features = np.array(result["patch_features"])
        assert patch_features.shape == (196, 768)

    def test_batch(self, api_url, image_data):
        """Test batch processing."""
        num_images = 64
        with ThreadPoolExecutor(max_workers=num_images) as executor:
            futures = [
                executor.submit(
                    requests.post,
                    api_url,
                    json={
                        "image_data": image_data,
                        "embedding_types": ["cls_token"],
                    },
                )
                for _ in range(num_images)
            ]
            for future in futures:
                response = future.result()
                assert response.status_code == 200, f"Failed: {response.text}"
                result = response.json()
                assert "cls_token" in result
                cls_token = np.array(result["cls_token"])
                assert cls_token.shape == (768,)

    def test_benchmark(self, api_url, image_data, benchmark):
        def run():
            return requests.post(
                api_url,
                json={
                    "image_data": image_data,
                    "embedding_types": ["cls_token"],
                },
            )

        response = benchmark(run)
        assert response.status_code == 200, f"Failed: {response.text}"
        result = response.json()
        assert "cls_token" in result
        cls_token = np.array(result["cls_token"])
        assert cls_token.shape == (768,)

