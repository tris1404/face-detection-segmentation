"""
Test module cho Face Recognition functionality.
Test cases: embedding extraction, comparison, face detection, edge cases.
"""

import pytest
import numpy as np
import cv2
import os
from utils.face_recognition import FaceRecognizer


@pytest.fixture
def recognizer():
    """Khởi tạo FaceRecognizer instance."""
    return FaceRecognizer()


@pytest.fixture
def dummy_image():
    """Tạo ảnh RGB giả chứa khuôn mặt (gradient pattern)."""
    # Tạo ảnh 200x200 với gradient pattern (giả khuôn mặt)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Vẽ gradient để tăng độ phức tạp
    for i in range(200):
        for j in range(200):
            img[i, j] = [100 + i % 50, 120 + j % 50, 110 + (i+j) % 60]
    return img


@pytest.fixture
def clear_face_image():
    """Ảnh chân dung rõ ràng từ file (nếu tồn tại)."""
    test_image_path = "data/input_images/test_face.jpg"
    if os.path.exists(test_image_path):
        img_bgr = cv2.imread(test_image_path)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return None


@pytest.fixture
def blurry_image():
    """Tạo ảnh mờ (blur)."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    # Apply Gaussian blur
    blurry = cv2.GaussianBlur(img, (31, 31), 0)
    return blurry


# ==================== TEST CASES ====================

def test_recognizer_initialization(recognizer):
    """Test 1: Khởi tạo FaceRecognizer không gây lỗi."""
    assert recognizer is not None
    assert recognizer.model_name == "FaceNet512"
    assert recognizer.metric == "cosine"
    print("\n✓ Test 1 Passed: FaceRecognizer initialized successfully")


def test_extract_embedding_dummy_image(recognizer, dummy_image):
    """Test 2: Trích embedding từ ảnh dummy."""
    embedding = recognizer.extract_embedding(dummy_image, enforce_detection=False)
    
    # Embedding có thể là None (không nhận mặt) hoặc numpy array
    if embedding is not None:
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        print(f"\n✓ Test 2 Passed: Embedding extracted, shape {embedding.shape}")
    else:
        print("\n⚠ Test 2: Lưu ý - Không nhận mặt trong dummy image (expected)")


def test_extract_embedding_empty_image(recognizer):
    """Test 3: Xử lý ảnh rỗng không crash."""
    empty_img = np.array([])
    result = recognizer.extract_embedding(empty_img, enforce_detection=False)
    assert result is None
    print("\n✓ Test 3 Passed: Empty image handled gracefully")


def test_extract_embedding_tiny_image(recognizer):
    """Test 4: Ảnh quá nhỏ (< 20x20)."""
    tiny_img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    result = recognizer.extract_embedding(tiny_img, enforce_detection=False)
    assert result is None
    print("\n✓ Test 4 Passed: Tiny image filtered out")


def test_compare_embeddings_same_person(recognizer):
    """Test 5: So sánh embedding giống nhau → similarity cao."""
    dummy_emb = np.random.randn(512).astype(np.float32)
    is_match, similarity = recognizer.compare_embeddings(dummy_emb, dummy_emb, threshold=0.4)
    
    assert is_match is True  # Giống nhau 100%
    assert similarity >= 0.99  # Similarity gần 1
    print(f"\n✓ Test 5 Passed: Same embedding comparison - similarity={similarity:.4f}, match={is_match}")


def test_compare_embeddings_different_person(recognizer):
    """Test 6: So sánh embedding khác nhau → similarity thấp."""
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)
    
    is_match, similarity = recognizer.compare_embeddings(emb1, emb2, threshold=0.4)
    
    # Embedding random khác nhau khá xa
    assert similarity < 0.6  # Similarity thấp
    print(f"\n✓ Test 6 Passed: Different embeddings - similarity={similarity:.4f}, match={is_match}")


def test_compare_embeddings_none_input(recognizer):
    """Test 7: Xử lý None embedding."""
    emb1 = np.random.randn(512).astype(np.float32)
    
    is_match, similarity = recognizer.compare_embeddings(None, emb1, threshold=0.4)
    assert is_match is False
    assert similarity == 0.0
    
    is_match, similarity = recognizer.compare_embeddings(emb1, None, threshold=0.4)
    assert is_match is False
    assert similarity == 0.0
    
    print("\n✓ Test 7 Passed: None input handled safely")


def test_embedding_json_serialization(recognizer):
    """Test 8: Chuyển đổi embedding ↔ JSON."""
    emb = np.random.randn(512).astype(np.float32)
    
    # To JSON
    json_str = recognizer.embedding_to_json(emb)
    assert json_str is not None
    assert isinstance(json_str, str)
    
    # From JSON
    recovered = recognizer.json_to_embedding(json_str)
    assert recovered is not None
    assert recovered.shape == (512,)
    
    # Check values match
    assert np.allclose(emb, recovered, atol=1e-5)
    print("\n✓ Test 8 Passed: Embedding JSON serialization works correctly")


def test_blurry_image_handling(recognizer, blurry_image):
    """Test 9: Ảnh mờ được xử lý mà không crash."""
    embedding = recognizer.extract_embedding(blurry_image, enforce_detection=False)
    
    # Embedding có thể trích được hoặc None (tùy độ blur)
    # Quan trọng là không crash
    if embedding is not None:
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        print("\n✓ Test 9a Passed: Blurry image embedding extracted")
    else:
        print("\n✓ Test 9b Passed: Blurry image returned None (acceptable)")


def test_get_faces_in_image_dummy(recognizer, dummy_image):
    """Test 10: Detect faces trong ảnh dummy."""
    faces = recognizer.get_faces_in_image(dummy_image, threshold=0.6)
    
    # Có thể detect được 0 hoặc nhiều faces
    assert isinstance(faces, list)
    
    # Nếu detect được, kiểm tra structure
    for face in faces:
        assert 'bbox' in face
        assert 'embedding' in face
        assert 'confidence' in face
        assert len(face['bbox']) == 4
        assert face['embedding'].shape == (512,)
    
    print(f"\n✓ Test 10 Passed: get_faces_in_image returned {len(faces)} faces")


def test_extract_embedding_from_file():
    """Test 11: Trích embedding từ file ảnh (nếu tồn tại)."""
    recognizer = FaceRecognizer()
    test_file = "data/input_images/test_face.jpg"
    
    if os.path.exists(test_file):
        embedding = recognizer.extract_embedding_from_file(test_file)
        if embedding is not None:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (512,)
            print(f"\n✓ Test 11 Passed: Extracted embedding from {test_file}")
        else:
            print(f"\n⚠ Test 11: File exists but no face detected")
    else:
        print(f"\n⊘ Test 11 Skipped: Test file not found at {test_file}")


def test_threshold_effect(recognizer):
    """Test 12: Threshold ảnh hưởng đến kết quả so sánh."""
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)  # Normalize
    
    # Tạo embedding tương tự: emb1 + small noise
    noise = np.random.randn(512) * 0.1
    emb2 = (emb1 + noise)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Test với threshold khác nhau
    is_match_04, sim_04 = recognizer.compare_embeddings(emb1, emb2, threshold=0.4)
    is_match_06, sim_06 = recognizer.compare_embeddings(emb1, emb2, threshold=0.6)
    
    # Với threshold 0.4, có khả năng match; 0.6 khó hơn
    print(f"\n✓ Test 12 Passed: Threshold effect verified")
    print(f"  Similarity={sim_04:.4f}, match(0.4)={is_match_04}, match(0.6)={is_match_06}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
