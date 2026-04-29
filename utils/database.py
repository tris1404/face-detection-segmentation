"""
Module Database - Quản lý SQLite cho hệ thống Điểm Danh Tự Động.
3 bảng: students (sinh viên) | sessions (phiên điểm danh) | attendance (kết quả)
"""

import sqlite3
import os
import json
from datetime import datetime
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Database:
    """
    Lớp quản lý SQLite database cho hệ thống Điểm Danh.
    
    Schema:
        - students: id | mssv (unique) | ho_ten | anh_path | embedding (JSON)
        - sessions: id | ngay_diem_danh (DATE) | anh_lop_path | created_at
        - attendance: id | session_id (FK) | student_id (FK) | trang_thai (enum)
    """
    
    def __init__(self, db_path='data/db/attendance.db'):
        """
        Khởi tạo Database handler.
        
        Args:
            db_path (str): Đường dẫn file SQLite. Default: 'data/db/attendance.db'
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager để kết nối DB an toàn."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Trả về dict-like rows
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Lỗi database: {e}")
            raise
        finally:
            conn.close()

    def init_database(self):
        """Tạo 3 bảng nếu chưa tồn tại."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Bảng students - sinh viên
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mssv TEXT UNIQUE NOT NULL,
                        ho_ten TEXT NOT NULL,
                        anh_path TEXT NOT NULL,
                        embedding TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Bảng sessions - phiên điểm danh
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ngay_diem_danh DATE NOT NULL,
                        anh_lop_path TEXT NOT NULL,
                        tong_so_sv INTEGER DEFAULT 0,
                        co_mat INTEGER DEFAULT 0,
                        vang_mat INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Bảng attendance - kết quả điểm danh
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        student_id INTEGER NOT NULL,
                        trang_thai TEXT CHECK(trang_thai IN ('CÓ MẶT', 'VẮNG MẶT')),
                        similarity_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(session_id) REFERENCES sessions(id),
                        FOREIGN KEY(student_id) REFERENCES students(id)
                    )
                """)
                
                logger.info("✓ Database schema ready")
                
        except Exception as e:
            logger.error(f"Lỗi init_database: {e}")
            raise

    # ==================== STUDENTS CRUD ====================
    
    def add_student(self, mssv, ho_ten, anh_path, embedding=None, on_duplicate='skip'):
        """
        Thêm sinh viên mới vào database.
        
        Args:
            mssv (str): Mã số sinh viên (unique)
            ho_ten (str): Họ và tên
            anh_path (str): Đường dẫn file ảnh chân dung
            embedding (str): Embedding JSON string
            on_duplicate (str): 'skip' = không thêm, 'update' = cập nhật
        
        Returns:
            dict: {'success': bool, 'message': str, 'student_id': int or None}
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Kiểm tra MSSV đã tồn tại
                cursor.execute("SELECT id FROM students WHERE mssv = ?", (mssv,))
                existing = cursor.fetchone()
                
                if existing:
                    student_id = existing['id']
                    if on_duplicate == 'skip':
                        return {
                            'success': False,
                            'message': f'MSSV {mssv} đã tồn tại',
                            'student_id': student_id
                        }
                    elif on_duplicate == 'update':
                        # Update thông tin
                        cursor.execute("""
                            UPDATE students 
                            SET ho_ten = ?, anh_path = ?, embedding = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE mssv = ?
                        """, (ho_ten, anh_path, embedding, mssv))
                        logger.info(f"Cập nhật sinh viên: {mssv}")
                        return {
                            'success': True,
                            'message': f'Cập nhật sinh viên {mssv}',
                            'student_id': student_id
                        }
                
                # Thêm sinh viên mới
                cursor.execute("""
                    INSERT INTO students (mssv, ho_ten, anh_path, embedding)
                    VALUES (?, ?, ?, ?)
                """, (mssv, ho_ten, anh_path, embedding))
                
                student_id = cursor.lastrowid
                logger.info(f"Thêm sinh viên mới: {mssv} (ID: {student_id})")
                
                return {
                    'success': True,
                    'message': f'Thêm sinh viên {mssv}',
                    'student_id': student_id
                }
                
        except sqlite3.IntegrityError as e:
            logger.error(f"Lỗi integrity: {e}")
            return {'success': False, 'message': f'Lỗi: {str(e)}', 'student_id': None}
        except Exception as e:
            logger.error(f"Lỗi add_student: {e}")
            return {'success': False, 'message': f'Lỗi: {str(e)}', 'student_id': None}

    def delete_student(self, student_id):
        """
        Xóa sinh viên khỏi database (cùng attendance records).
        
        Args:
            student_id (int): ID sinh viên
        
        Returns:
            bool: True nếu thành công
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Xóa attendance records
                cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
                
                # Xóa student
                cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
                
                logger.info(f"Xóa sinh viên ID: {student_id}")
                return True
                
        except Exception as e:
            logger.error(f"Lỗi delete_student: {e}")
            return False

    def get_all_students(self):
        """
        Lấy danh sách tất cả sinh viên.
        
        Returns:
            list: Danh sách dict với keys: id, mssv, ho_ten, anh_path, embedding, created_at
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, mssv, ho_ten, anh_path, embedding, created_at 
                    FROM students 
                    ORDER BY created_at DESC
                """)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Lỗi get_all_students: {e}")
            return []

    def get_student_by_mssv(self, mssv):
        """
        Lấy thông tin sinh viên theo MSSV.
        
        Args:
            mssv (str): Mã số sinh viên
        
        Returns:
            dict: Thông tin sinh viên hoặc None nếu không tìm thấy
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, mssv, ho_ten, anh_path, embedding 
                    FROM students WHERE mssv = ?
                """, (mssv,))
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Lỗi get_student_by_mssv: {e}")
            return None

    def get_student_by_id(self, student_id):
        """
        Lấy thông tin sinh viên theo ID.
        
        Args:
            student_id (int): ID sinh viên
        
        Returns:
            dict: Thông tin sinh viên hoặc None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, mssv, ho_ten, anh_path, embedding 
                    FROM students WHERE id = ?
                """, (student_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Lỗi get_student_by_id: {e}")
            return None

    # ==================== SESSIONS CRUD ====================
    
    def create_session(self, ngay_diem_danh, anh_lop_path):
        """
        Tạo phiên điểm danh mới.
        
        Args:
            ngay_diem_danh (str): Ngày dạng 'YYYY-MM-DD' hoặc 'today'
            anh_lop_path (str): Đường dẫn file ảnh lớp học
        
        Returns:
            int: Session ID hoặc None nếu lỗi
        """
        try:
            if ngay_diem_danh == 'today':
                ngay_diem_danh = datetime.now().strftime('%Y-%m-%d')
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (ngay_diem_danh, anh_lop_path)
                    VALUES (?, ?)
                """, (ngay_diem_danh, anh_lop_path))
                
                session_id = cursor.lastrowid
                logger.info(f"Tạo session mới: ID {session_id}, ngày {ngay_diem_danh}")
                
                return session_id
                
        except Exception as e:
            logger.error(f"Lỗi create_session: {e}")
            return None

    def get_sessions(self, limit=10):
        """
        Lấy danh sách phiên điểm danh gần đây.
        
        Args:
            limit (int): Số phiên lấy. Default: 10
        
        Returns:
            list: Danh sách dict với keys: id, ngay_diem_danh, tong_so_sv, co_mat, vang_mat
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, ngay_diem_danh, anh_lop_path, tong_so_sv, co_mat, vang_mat, created_at
                    FROM sessions
                    ORDER BY ngay_diem_danh DESC, created_at DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Lỗi get_sessions: {e}")
            return []

    def get_session_by_id(self, session_id):
        """
        Lấy thông tin phiên điểm danh theo ID.
        
        Args:
            session_id (int): ID phiên
        
        Returns:
            dict: Thông tin session hoặc None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, ngay_diem_danh, anh_lop_path, tong_so_sv, co_mat, vang_mat
                    FROM sessions WHERE id = ?
                """, (session_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Lỗi get_session_by_id: {e}")
            return None

    # ==================== ATTENDANCE CRUD ====================
    
    def save_attendance(self, session_id, student_id, trang_thai, similarity_score=None):
        """
        Lưu kết quả điểm danh cho 1 sinh viên.
        
        Args:
            session_id (int): ID phiên điểm danh
            student_id (int): ID sinh viên
            trang_thai (str): 'CÓ MẶT' hoặc 'VẮNG MẶT'
            similarity_score (float): Độ tương đồng embedding (optional)
        
        Returns:
            bool: True nếu thành công
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO attendance (session_id, student_id, trang_thai, similarity_score)
                    VALUES (?, ?, ?, ?)
                """, (session_id, student_id, trang_thai, similarity_score))
                
                return True
                
        except Exception as e:
            logger.error(f"Lỗi save_attendance: {e}")
            return False

    def get_attendance_by_session(self, session_id):
        """
        Lấy kết quả điểm danh của 1 phiên (tất cả sinh viên).
        
        Args:
            session_id (int): ID phiên
        
        Returns:
            list: Danh sách dict với keys: id, mssv, ho_ten, trang_thai, similarity_score
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        a.id,
                        s.id as student_id,
                        s.mssv,
                        s.ho_ten,
                        s.anh_path,
                        a.trang_thai,
                        a.similarity_score,
                        a.created_at
                    FROM attendance a
                    JOIN students s ON a.student_id = s.id
                    WHERE a.session_id = ?
                    ORDER BY a.created_at DESC
                """, (session_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Lỗi get_attendance_by_session: {e}")
            return []

    def update_session_stats(self, session_id):
        """
        Cập nhật thống kê phiên (tổng, có mặt, vắng mặt).
        
        Args:
            session_id (int): ID phiên
        
        Returns:
            bool: True nếu thành công
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Tính toán stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as tong_so_sv,
                        SUM(CASE WHEN trang_thai = 'CÓ MẶT' THEN 1 ELSE 0 END) as co_mat,
                        SUM(CASE WHEN trang_thai = 'VẮNG MẶT' THEN 1 ELSE 0 END) as vang_mat
                    FROM attendance
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                tong = row['tong_so_sv'] or 0
                co_mat = row['co_mat'] or 0
                vang_mat = row['vang_mat'] or 0
                
                # Update session record
                cursor.execute("""
                    UPDATE sessions
                    SET tong_so_sv = ?, co_mat = ?, vang_mat = ?
                    WHERE id = ?
                """, (tong, co_mat, vang_mat, session_id))
                
                logger.info(f"Update stats session {session_id}: {tong} / {co_mat} / {vang_mat}")
                return True
                
        except Exception as e:
            logger.error(f"Lỗi update_session_stats: {e}")
            return False

    def get_attendance_summary(self, session_id):
        """
        Lấy tóm tắt kết quả điểm danh (có mặt vs vắng mặt).
        
        Args:
            session_id (int): ID phiên
        
        Returns:
            dict: {'co_mat': [...], 'vang_mat': [...]}
        """
        try:
            attendance = self.get_attendance_by_session(session_id)
            
            co_mat = [a for a in attendance if a['trang_thai'] == 'CÓ MẶT']
            vang_mat = [a for a in attendance if a['trang_thai'] == 'VẮNG MẶT']
            
            return {
                'co_mat': co_mat,
                'vang_mat': vang_mat,
                'tong': len(attendance),
                'tong_co_mat': len(co_mat),
                'tong_vang': len(vang_mat)
            }
            
        except Exception as e:
            logger.error(f"Lỗi get_attendance_summary: {e}")
            return {'co_mat': [], 'vang_mat': [], 'tong': 0, 'tong_co_mat': 0, 'tong_vang': 0}
