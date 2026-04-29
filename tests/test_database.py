"""
Test module cho Database functionality.
Test cases: CRUD operations, database initialization, session management, attendance tracking.
"""

import pytest
import os
import tempfile
import json
import numpy as np
from datetime import datetime
from utils.database import Database


@pytest.fixture
def temp_db():
    """Tạo database tạm thời cho testing."""
    # Tạo temp file
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Khởi tạo database
    db = Database(db_path)
    
    yield db
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_student_data():
    """Dữ liệu sinh viên test."""
    return {
        'mssv': '20210001',
        'ho_ten': 'Nguyễn Văn A',
        'anh_path': 'data/students/20210001.jpg',
        'embedding': json.dumps(np.random.randn(512).tolist())
    }


@pytest.fixture
def sample_students_data():
    """Danh sách sinh viên test."""
    students = []
    for i in range(1, 4):
        embedding = np.random.randn(512)
        students.append({
            'mssv': f'2021000{i}',
            'ho_ten': f'Sinh Viên {i}',
            'anh_path': f'data/students/2021000{i}.jpg',
            'embedding': json.dumps(embedding.tolist())
        })
    return students


# ==================== TEST DATABASE INITIALIZATION ====================

def test_database_initialization(temp_db):
    """Test 1: Database khởi tạo và tạo 3 bảng."""
    # Check tables exist
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check students table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
        assert cursor.fetchone() is not None
        
        # Check sessions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        assert cursor.fetchone() is not None
        
        # Check attendance table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'")
        assert cursor.fetchone() is not None
    
    print("\n✓ Test 1 Passed: Database initialized with 3 tables")


# ==================== TEST STUDENTS CRUD ====================

def test_add_student_success(temp_db, sample_student_data):
    """Test 2: Thêm sinh viên mới thành công."""
    result = temp_db.add_student(
        sample_student_data['mssv'],
        sample_student_data['ho_ten'],
        sample_student_data['anh_path'],
        sample_student_data['embedding']
    )
    
    assert result['success'] is True
    assert result['student_id'] is not None
    assert isinstance(result['student_id'], int)
    
    print(f"\n✓ Test 2 Passed: Student added with ID {result['student_id']}")


def test_add_student_duplicate_skip(temp_db, sample_student_data):
    """Test 3: Thêm sinh viên trùng MSSV - mode skip."""
    # Thêm lần 1
    result1 = temp_db.add_student(
        sample_student_data['mssv'],
        sample_student_data['ho_ten'],
        sample_student_data['anh_path'],
        sample_student_data['embedding']
    )
    assert result1['success'] is True
    
    # Thêm lần 2 (trùng) - skip mode
    result2 = temp_db.add_student(
        sample_student_data['mssv'],
        'Tên khác',
        'path/khac.jpg',
        sample_student_data['embedding'],
        on_duplicate='skip'
    )
    
    assert result2['success'] is False
    assert 'đã tồn tại' in result2['message']
    
    print("\n✓ Test 3 Passed: Duplicate MSSV skipped correctly")


def test_add_student_duplicate_update(temp_db, sample_student_data):
    """Test 4: Thêm sinh viên trùng MSSV - mode update."""
    # Thêm lần 1
    result1 = temp_db.add_student(
        sample_student_data['mssv'],
        sample_student_data['ho_ten'],
        sample_student_data['anh_path'],
        sample_student_data['embedding']
    )
    student_id_1 = result1['student_id']
    
    # Thêm lần 2 (trùng) - update mode
    new_embedding = json.dumps(np.random.randn(512).tolist())
    result2 = temp_db.add_student(
        sample_student_data['mssv'],
        'Tên mới',
        'path/new.jpg',
        new_embedding,
        on_duplicate='update'
    )
    
    assert result2['success'] is True
    assert result2['student_id'] == student_id_1
    
    # Verify data updated
    student = temp_db.get_student_by_mssv(sample_student_data['mssv'])
    assert student['ho_ten'] == 'Tên mới'
    assert student['anh_path'] == 'path/new.jpg'
    
    print("\n✓ Test 4 Passed: Duplicate MSSV updated correctly")


def test_get_all_students(temp_db, sample_students_data):
    """Test 5: Lấy danh sách tất cả sinh viên."""
    # Thêm 3 sinh viên
    for student in sample_students_data:
        result = temp_db.add_student(
            student['mssv'],
            student['ho_ten'],
            student['anh_path'],
            student['embedding']
        )
        assert result['success'] is True
    
    # Get all
    students = temp_db.get_all_students()
    
    assert len(students) == 3
    assert all('mssv' in s for s in students)
    assert all('ho_ten' in s for s in students)
    
    print(f"\n✓ Test 5 Passed: Retrieved {len(students)} students")


def test_get_student_by_mssv(temp_db, sample_student_data):
    """Test 6: Lấy thông tin sinh viên theo MSSV."""
    # Add student
    temp_db.add_student(
        sample_student_data['mssv'],
        sample_student_data['ho_ten'],
        sample_student_data['anh_path'],
        sample_student_data['embedding']
    )
    
    # Get by MSSV
    student = temp_db.get_student_by_mssv(sample_student_data['mssv'])
    
    assert student is not None
    assert student['mssv'] == sample_student_data['mssv']
    assert student['ho_ten'] == sample_student_data['ho_ten']
    
    print("\n✓ Test 6 Passed: Student retrieved by MSSV")


def test_delete_student(temp_db, sample_student_data):
    """Test 7: Xóa sinh viên."""
    # Add student
    result = temp_db.add_student(
        sample_student_data['mssv'],
        sample_student_data['ho_ten'],
        sample_student_data['anh_path'],
        sample_student_data['embedding']
    )
    student_id = result['student_id']
    
    # Verify exists
    students = temp_db.get_all_students()
    assert len(students) == 1
    
    # Delete
    success = temp_db.delete_student(student_id)
    assert success is True
    
    # Verify deleted
    students = temp_db.get_all_students()
    assert len(students) == 0
    
    print("\n✓ Test 7 Passed: Student deleted successfully")


# ==================== TEST SESSIONS CRUD ====================

def test_create_session(temp_db):
    """Test 8: Tạo phiên điểm danh."""
    session_id = temp_db.create_session('2024-03-15', 'data/sessions/class_2024-03-15.jpg')
    
    assert session_id is not None
    assert isinstance(session_id, int)
    
    print(f"\n✓ Test 8 Passed: Session created with ID {session_id}")


def test_get_sessions(temp_db):
    """Test 9: Lấy danh sách sessions."""
    # Create 3 sessions
    for i in range(1, 4):
        temp_db.create_session(f'2024-03-{i:02d}', f'path/class_{i}.jpg')
    
    # Get sessions
    sessions = temp_db.get_sessions(limit=10)
    
    assert len(sessions) == 3
    assert all('ngay_diem_danh' in s for s in sessions)
    
    print(f"\n✓ Test 9 Passed: Retrieved {len(sessions)} sessions")


# ==================== TEST ATTENDANCE CRUD ====================

def test_save_and_get_attendance(temp_db, sample_student_data):
    """Test 10: Lưu và lấy kết quả điểm danh."""
    # Add student
    result = temp_db.add_student(
        sample_student_data['mssv'],
        sample_student_data['ho_ten'],
        sample_student_data['anh_path'],
        sample_student_data['embedding']
    )
    student_id = result['student_id']
    
    # Create session
    session_id = temp_db.create_session('2024-03-15', 'path/class.jpg')
    
    # Save attendance
    success = temp_db.save_attendance(session_id, student_id, 'CÓ MẶT', similarity_score=0.95)
    assert success is True
    
    # Get attendance
    attendance = temp_db.get_attendance_by_session(session_id)
    
    assert len(attendance) == 1
    assert attendance[0]['trang_thai'] == 'CÓ MẶT'
    assert attendance[0]['similarity_score'] == 0.95
    
    print("\n✓ Test 10 Passed: Attendance saved and retrieved")


def test_update_session_stats(temp_db, sample_students_data):
    """Test 11: Cập nhật thống kê phiên."""
    # Add students
    student_ids = []
    for student in sample_students_data:
        result = temp_db.add_student(
            student['mssv'],
            student['ho_ten'],
            student['anh_path'],
            student['embedding']
        )
        student_ids.append(result['student_id'])
    
    # Create session
    session_id = temp_db.create_session('2024-03-15', 'path/class.jpg')
    
    # Save attendance: 2 có mặt, 1 vắng
    temp_db.save_attendance(session_id, student_ids[0], 'CÓ MẶT')
    temp_db.save_attendance(session_id, student_ids[1], 'CÓ MẶT')
    temp_db.save_attendance(session_id, student_ids[2], 'VẮNG MẶT')
    
    # Update stats
    success = temp_db.update_session_stats(session_id)
    assert success is True
    
    # Check session stats
    session = temp_db.get_session_by_id(session_id)
    assert session['tong_so_sv'] == 3
    assert session['co_mat'] == 2
    assert session['vang_mat'] == 1
    
    print("\n✓ Test 11 Passed: Session stats updated correctly")


def test_get_attendance_summary(temp_db, sample_students_data):
    """Test 12: Lấy tóm tắt điểm danh."""
    # Setup
    student_ids = []
    for student in sample_students_data:
        result = temp_db.add_student(
            student['mssv'],
            student['ho_ten'],
            student['anh_path'],
            student['embedding']
        )
        student_ids.append(result['student_id'])
    
    session_id = temp_db.create_session('2024-03-15', 'path/class.jpg')
    
    # Save attendance
    temp_db.save_attendance(session_id, student_ids[0], 'CÓ MẶT')
    temp_db.save_attendance(session_id, student_ids[1], 'CÓ MẶT')
    temp_db.save_attendance(session_id, student_ids[2], 'VẮNG MẶT')
    
    # Get summary
    summary = temp_db.get_attendance_summary(session_id)
    
    assert summary['tong'] == 3
    assert summary['tong_co_mat'] == 2
    assert summary['tong_vang'] == 1
    assert len(summary['co_mat']) == 2
    assert len(summary['vang_mat']) == 1
    
    print("\n✓ Test 12 Passed: Attendance summary retrieved correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
