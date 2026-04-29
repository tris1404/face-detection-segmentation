import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
from datetime import datetime, timedelta
import json
import logging
from utils.detector import FaceDetector
from utils.segmentor import FaceSegmentor
from utils.face_recognition import FaceRecognizer
from utils.database import Database
from utils.export import export_attendance_to_excel

STATUS_PRESENT = "CÓ MẶT"
STATUS_ABSENT = "VẮNG MẶT"
LABEL_TOTAL_STUDENTS = "Tổng SV"
LABEL_PRESENT = "Có Mặt"
LABEL_ABSENT = "Vắng Mặt"
LABEL_ATTENDANCE_HISTORY = "Lịch Sử " + "Điểm Danh"
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument" + ".spreadsheetml.sheet"

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Thiết lập cấu hình trang
st.set_page_config(page_title="Hệ thống Nhận diện, Phân vùng & Điểm Danh Tự Động", layout="wide")
st.title("Hệ thống Nhận diện, Phân vùng & Điểm Danh Tự Động")
st.write("RetinaFace + BiSeNet + DeepFace (FaceNet512)")

# Tạo các thư mục cần thiết
os.makedirs("data/students", exist_ok=True)
os.makedirs("data/db", exist_ok=True)
os.makedirs("data/attendance_reports", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

def draw_text_vietnamese(img, text, position, color=(0, 255, 0), font_size=20):
    """Vẽ chữ tiếng Việt có dấu lên ảnh OpenCV (numpy array)."""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # Thử load font Arial hỗ trợ Unicode trên Windows
    font_path = "C:/Windows/Fonts/arial.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = None # Fallback nếu không tìm thấy font
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)


def save_class_image(image_rgb):
    """Lưu ảnh lớp học vào disk và trả về đường dẫn file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"class_{timestamp}.jpg"
    file_path = os.path.join("data/sessions", file_name)
    Image.fromarray(image_rgb).convert("RGB").save(file_path, format="JPEG", quality=95)
    return file_path

# Khởi tạo các model và database - cache chúng để không bị load lại mỗi khi tương tác UI
@st.cache_resource
def load_models():
    detector = FaceDetector()
    segmentor = FaceSegmentor()
    recognizer = FaceRecognizer()
    database = Database("data/db/attendance.db")
    return detector, segmentor, recognizer, database

detector, segmentor, recognizer, database = load_models()

# Tạo 5 tab chức năng
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Nhận diện (Detection)",
    "Phân vùng (Segmentation)",
    "Quản Lý Sinh Viên",
    "Điểm Danh",
    "Báo Cáo & Lịch Sử"
])

with tab1:
    st.header("Chức năng Nhận diện (Face Detection)")
    st.write("Sử dụng RetinaFace để tìm kiếm và đóng khung khuôn mặt.")
    # Tùy chọn chế độ nhận diện
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_hard_mode = st.checkbox("Bật chế độ HARD (Tiling)", value=False, 
                                    help="Dùng cho ảnh đám đông có hàng trăm khuôn mặt siêu nhỏ. Xử lý sẽ chậm hơn.")
    
    upload_det = st.file_uploader("Tải lên ảnh để Nhận diện", type=["jpg", "jpeg", "png"], key="upload_det")
    
    if upload_det is not None:
        image = Image.open(upload_det).convert('RGB')
        img_rgb = np.array(image)
        
        with st.spinner("Đang xử lý nhận diện..."):
            if use_hard_mode:
                st.warning("Đang chạy chế độ HARD (Tiling) tối ưu. Vui lòng đợi...")
                faces = detector.detect_hard_mode(img_rgb, grid_size=(4, 4))
            else:
                faces = detector.detect(img_rgb)
        
        img_det = detector.draw_faces(img_rgb, faces)
        
        st.success(f"Đã hoàn thành! Phát hiện **{len(faces)}** khuôn mặt.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Ảnh Gốc", use_container_width=True)
        with col2:
            st.image(img_det, caption="Ảnh Nhận diện (Bounding Box)", use_container_width=True)

with tab2:
    st.header("Chức năng Phân vùng (Face Segmentation)")
    st.write("Sử dụng BiSeNet để tô màu vùng da mặt.")
    upload_seg = st.file_uploader("Tải lên ảnh để Phân vùng", type=["jpg", "jpeg", "png"], key="upload_seg")
    
    if upload_seg is not None:
        image = Image.open(upload_seg).convert('RGB')
        img_rgb = np.array(image)
        
        st.write("Đang xử lý phân vùng...")
        
        # Thực hiện phân vùng TRỰC TIẾP toàn bộ ảnh (không dùng detector để crop)
        mask, _ = segmentor.segment(img_rgb, faces=None)
        
        # Vẽ mask lên ảnh gốc (không giới hạn bounding box)
        img_res = segmentor.draw_segmentation(img_rgb, mask, faces=None)
        
        # Chuẩn bị Bản đồ phân vùng màu
        color_mapped = segmentor.colors[mask]
        parts_map = color_mapped.copy()
        parts_map[mask == 0] = [0, 0, 0] # Background màu đen
        
        # Hiển thị 3 ảnh trên cùng 1 hàng
        st.divider()
        st.header("Kết quả Semantic Segmentation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(img_rgb, caption="Ảnh Gốc", use_container_width=True)
        with col2:
            st.image(img_res, caption="Kết quả Overlay", use_container_width=True)
        with col3:
            st.image(parts_map, caption="Bản đồ các bộ phận (Màu)", use_container_width=True)

# =====================================================================
# TAB 3: QUẢN LÝ SINH VIÊN (Student Management)
# =====================================================================
with tab3:
    st.header("Quản Lý Sinh Viên")
    
    # --- FORM THÊM SINH VIÊN ---
    st.subheader("Thêm Sinh Viên Mới")
    col1, col2 = st.columns(2)
    
    with col1:
        mssv = st.text_input("Mã Số Sinh Viên (MSSV):", placeholder="SV001")
    with col2:
        ho_ten = st.text_input("Họ và Tên:", placeholder="Nguyễn Văn A")
    
    upload_anh = st.file_uploader(
        "Upload ảnh chân dung",
        type=["jpg", "jpeg", "png"],
        key="upload_anh_sv"
    )
    
    if st.button("Thêm Sinh Viên", use_container_width=True):
        if not mssv.strip():
            st.error("Vui lòng nhập MSSV")
        elif not ho_ten.strip():
            st.error("Vui lòng nhập Họ tên")
        elif upload_anh is None:
            st.error("Vui lòng upload ảnh chân dung")
        else:
            # Đọc ảnh
            image = Image.open(upload_anh).convert('RGB')
            img_rgb = np.array(image)
            
            with st.spinner("Đang trích xuất embedding từ ảnh..."):
                embedding = recognizer.extract_embedding(img_rgb, enforce_detection=True)
            
            if embedding is None:
                st.error("Không nhận diện được khuôn mặt trong ảnh. Vui lòng thử ảnh khác.")
            else:
                # Lưu ảnh vào disk
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                anh_filename = f"{mssv}_{timestamp}.jpg"
                anh_path = f"data/students/{anh_filename}"
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(anh_path, img_bgr)
                
                # Convert embedding → JSON
                embedding_json = FaceRecognizer.embedding_to_json(embedding)
                
                # Thêm vào DB
                success, message, student_id = database.add_student(
                    mssv=mssv.strip(),
                    ho_ten=ho_ten.strip(),
                    anh_path=anh_path,
                    embedding=embedding_json
                )
                
                if success:
                    st.success(f"Thêm sinh viên thành công! (ID={student_id})")
                else:
                    st.error(f"Lỗi: {message}")
    
    st.divider()
    
    # --- DANH SÁCH SINH VIÊN ---
    st.subheader("Danh Sách Sinh Viên")
    students = database.get_all_students()
    
    if not students:
        st.info("Chưa có sinh viên nào trong hệ thống")
    else:
        st.write(f"Tổng: **{len(students)}** sinh viên")
        
        for sv in students:
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                st.write(f"**{sv['mssv']}** - {sv['ho_ten']}")
            
            with col2:
                if sv['anh_path'] and os.path.exists(sv['anh_path']):
                    img_display = Image.open(sv['anh_path']).convert('RGB')
                    st.image(img_display, width=100)
            
            with col3:
                if st.button("Xóa", key=f"del_{sv['id']}"):
                    success_del, msg_del = database.delete_student(sv['id'])
                    if success_del:
                        st.success("Xóa sinh viên thành công!")
                        st.rerun()

# =====================================================================
# TAB 4: ĐIỂM DANH (Attendance)
# =====================================================================
with tab4:
    st.header("Điểm Danh Tự Động")
    
    # Sidebar: Threshold slider
    st.sidebar.divider()
    st.sidebar.write("### Cài Đặt Điểm Danh")
    threshold = st.sidebar.slider(
        "Ngưỡng nhận diện (Cosine Similarity):",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Càng cao = càng khó khớp. Thấp = dễ nhầm. Mặc định 0.3 (đã thắt chặt)"
    )
    
    # --- UPLOAD ẢNH LỚP ---
    st.subheader("Upload Ảnh Lớp Học")
    upload_class_img = st.file_uploader(
        "Tải lên ảnh cả lớp học",
        type=["jpg", "jpeg", "png"],
        key="upload_class_img"
    )
    
    if upload_class_img is not None:
        image_class = Image.open(upload_class_img).convert('RGB')
        img_class_rgb = np.array(image_class)
        class_image_path = save_class_image(img_class_rgb)
        
        st.image(img_class_rgb, caption="Ảnh Lớp Học", use_container_width=True)
        
        # --- NÚT ĐIỂM DANH ---
        if st.button("Bắt Đầu Điểm Danh", use_container_width=True):
            print("\n" + "="*50)
            print(f"BẮT ĐẦU ĐIỂM DANH - {datetime.now().strftime('%H:%M:%S')}")
            print("="*50)
            
            students = database.get_all_students()
            if not students:
                st.error("Chưa có sinh viên nào trong hệ thống. Vui lòng thêm sinh viên trước!")
                print("LỖI: Chưa có sinh viên trong hệ thống.")
            else:
                # --- BƯỚC 1: Face Detection ---
                with st.status("Đang xử lý bước 1: Nhận diện khuôn mặt...", expanded=True) as status:
                    print("[BƯỚC 1] Đang thực hiện Face Detection...")
                    # Detect faces trong ảnh lớp
                    detected_faces = recognizer.get_faces_in_image(img_class_rgb, threshold=0.5)
                    
                    if not detected_faces:
                        status.update(label="Thất bại ở Bước 1", state="error")
                    else:
                        status.update(label="Bước 1: Hoàn tất", state="complete")

                # Expander phải nằm ngoài st.status để tránh lỗi lồng nhau
                with st.expander("BƯỚC 1 - Kết quả Face Detection", expanded=True):
                    if not detected_faces:
                        msg = "Face Detection không tìm thấy mặt nào -> Thử giảm threshold hoặc đổi ảnh"
                        st.error(msg)
                        print(f"LỖI: {msg}")
                        st.stop()
                    
                    print(f"THÀNH CÔNG: Tìm thấy {len(detected_faces)} khuôn mặt.")
                    st.success(f"Phát hiện **{len(detected_faces)}** khuôn mặt")
                    
                    # Vẽ bounding box lên ảnh
                    img_det = img_class_rgb.copy()
                    h_img, w_img = img_class_rgb.shape[:2]
                    thickness = max(2, int(min(h_img, w_img) / 400))
                    
                    for i, face in enumerate(detected_faces):
                        x1, y1, x2, y2 = face['bbox']
                        cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                        # Đánh số thứ tự mặt
                        cv2.putText(img_det, str(i+1), (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), thickness)
                    
                    st.image(img_det, caption="Ảnh sau khi Face Detection", use_container_width=True)

                # --- BƯỚC 2: Crop khuôn mặt ---
                with st.expander("BƯỚC 2 - Crop khuôn mặt", expanded=False):
                    print("[BƯỚC 2] Đang thực hiện Crop khuôn mặt...")
                    cols = st.columns(6)
                    h, w = img_class_rgb.shape[:2]
                    for i, face in enumerate(detected_faces):
                        x1, y1, x2, y2 = face['bbox']
                        # Thêm padding nhẹ khi crop để nhìn rõ hơn
                        pad = 10
                        face_crop = img_class_rgb[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                        
                        if face_crop.size > 0:
                            cols[i % 6].image(face_crop, caption=f"Mặt số {i+1}", use_container_width=True)
                    print(f"Đã hiển thị grid {len(detected_faces)} khuôn mặt.")

                # --- BƯỚC 3: So sánh embedding ---
                with st.spinner("Đang xử lý bước 3: So sánh embedding..."):
                    print("[BƯỚC 3] Đang thực hiện so sánh embedding với database...")
                    # 2. Tạo session điểm danh trong DB
                    ngay_today = datetime.now().strftime("%Y-%m-%d")
                    session_result = database.create_session(
                        ngay_diem_danh=ngay_today,
                        anh_lop_path=class_image_path
                    )

                    # Hỗ trợ cả kiểu trả về cũ (tuple/dict) lẫn kiểu hiện tại (session_id)
                    if isinstance(session_result, tuple):
                        if len(session_result) == 3:
                            success_session, msg_session, session_id = session_result
                        elif len(session_result) == 2:
                            success_session, session_id = session_result
                            msg_session = "Tạo session thành công"
                        else:
                            success_session, msg_session, session_id = False, "Định dạng kết quả session không hợp lệ", None
                    elif isinstance(session_result, dict):
                        success_session = bool(session_result.get("success", False))
                        msg_session = session_result.get("message", "")
                        session_id = session_result.get("session_id")
                    elif session_result is None:
                        success_session, msg_session, session_id = False, "Không thể tạo session", None
                    else:
                        success_session, msg_session, session_id = True, "Tạo session thành công", session_result

                    if not success_session or session_id is None:
                        st.error(f"Lỗi tạo session: {msg_session}")
                        print(f"LỖI: {msg_session}")
                    else:
                        # 3. Chuẩn bị data từ DB
                        db_students_data = []
                        for sv in students:
                            embedding_json = sv['embedding']
                            embedding_vec = FaceRecognizer.json_to_embedding(embedding_json)
                            if embedding_vec is not None:
                                db_students_data.append({
                                    'id': sv['id'],
                                    'mssv': sv['mssv'],
                                    'ho_ten': sv['ho_ten'],
                                    'anh_path': sv['anh_path'],
                                    'embedding': embedding_vec
                                })
                        
                        print(f"Đang so sánh {len(detected_faces)} mặt với {len(db_students_data)} sinh viên trong DB...")
                        
                        # 4. Tìm tất cả các cặp match tiềm năng
                        potential_matches = []
                        
                        for face_idx, detected_face in enumerate(detected_faces):
                            face_embedding = detected_face['embedding']
                            for db_sv in db_students_data:
                                is_match, similarity = recognizer.compare_embeddings(
                                    face_embedding,
                                    db_sv['embedding'],
                                    threshold=threshold
                                )
                                if is_match:
                                    potential_matches.append({
                                        'face_idx': face_idx,
                                        'student_obj': db_sv,
                                        'similarity': similarity
                                    })
                        
                        # 5. Greedy Matching 1-1: Ưu tiên cặp có similarity cao nhất
                        potential_matches.sort(key=lambda x: x['similarity'], reverse=True)
                        
                        matched_faces = {}
                        used_faces = set()
                        used_students = set()
                        
                        for match in potential_matches:
                            f_idx = match['face_idx']
                            s_id = match['student_obj']['id']
                            
                            if f_idx not in used_faces and s_id not in used_students:
                                matched_faces[f_idx] = {
                                    'student_id': s_id,
                                    'student_name': match['student_obj']['ho_ten'],
                                    'mssv': match['student_obj']['mssv'],
                                    'similarity': match['similarity']
                                }
                                used_faces.add(f_idx)
                                used_students.add(s_id)
                                print(f"Match OK: Face {f_idx+1} -> {match['student_obj']['ho_ten']} ({match['similarity']:.4f})")

                        # Tạo danh sách chi tiết cho UI
                        step3_details = []
                        for face_idx in range(len(detected_faces)):
                            if face_idx in matched_faces:
                                m = matched_faces[face_idx]
                                detail_text = f"✅ Mặt {face_idx+1} → **{m['student_name']}** (score: {m['similarity']:.2f})"
                            else:
                                detail_text = f"❌ Mặt {face_idx+1} → **Không nhận ra**"
                            
                            step3_details.append(detail_text)
                            if "❌" in detail_text:
                                print(f"Kết quả: Face {face_idx+1} -> Unknown")

                        with st.expander("BƯỚC 3 - Chi tiết so sánh Embedding", expanded=False):
                            for detail in step3_details:
                                st.write(detail)

                        matched_student_ids = {
                            match_info['student_id']
                            for match_info in matched_faces.values()
                        }
                        
                        # 5. Lưu kết quả vào DB
                        for sv in db_students_data:
                            trang_thai = STATUS_PRESENT if sv['id'] in matched_student_ids else STATUS_ABSENT
                            database.save_attendance(session_id, sv['id'], trang_thai)

                        database.update_session_stats(session_id)
                        
                        # 6. Vẽ ảnh kết quả tổng hợp
                        img_result = img_class_rgb.copy()
                        h_res, w_res = img_result.shape[:2]
                        thickness_res = max(2, int(min(h_res, w_res) / 400))
                        
                        for face_idx, detected_face in enumerate(detected_faces):
                            x1, y1, x2, y2 = detected_face['bbox']
                            
                            if face_idx in matched_faces:
                                match_info = matched_faces[face_idx]
                                text = f"{match_info['mssv']} - {match_info['student_name']}"
                                color = (0, 255, 0)
                            else:
                                text = "Unknown"
                                color = (0, 0, 255)
                            
                            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, thickness_res)
                            # Dùng PIL để vẽ text tiếng Việt có dấu
                            img_result = draw_text_vietnamese(
                                img_result, 
                                text, 
                                (x1, y1 - 25), 
                                color, 
                                font_size=18
                            )
                        
                        # 7. Hiển thị kết quả cuối cùng
                        st.divider()
                        st.subheader("Kết Quả Điểm Danh Cuối Cùng")
                        
                        st.image(img_result, caption="Ảnh Lớp Với Kết Quả Tổng Hợp", use_container_width=True)
                        
                        # Thống kê
                        stats = database.get_session_by_id(session_id) or {
                            'tong_so_sv': 0,
                            'co_mat': 0,
                            'vang_mat': 0,
                        }
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric(LABEL_TOTAL_STUDENTS, stats['tong_so_sv'])
                        with col_stat2:
                            st.metric(LABEL_PRESENT, stats['co_mat'])
                        with col_stat3:
                            st.metric(LABEL_ABSENT, stats['vang_mat'])
                        
                        # Lưu session_id để dùng trong tab báo cáo
                        st.session_state['last_session_id'] = session_id
                        print(f"HOÀN TẤT ĐIỂM DANH. Session ID: {session_id}")

# =====================================================================
# TAB 5: BÁO CÁO & LỊCH SỬ (Report)
# =====================================================================
with tab5:
    st.header(f"Báo Cáo & {LABEL_ATTENDANCE_HISTORY}")
    
    sessions = database.get_sessions(limit=50)
    
    if not sessions:
        st.info("Chưa có phiên điểm danh nào")
    else:
        st.subheader(LABEL_ATTENDANCE_HISTORY)
        
        session_options = [
            f"{s['ngay_diem_danh']} (Session #{s['id']})" for s in sessions
        ]
        
        selected_session_text = st.selectbox(
            "Chọn buổi điểm danh:",
            session_options,
            key="session_select"
        )
        
        selected_idx = session_options.index(selected_session_text)
        selected_session = sessions[selected_idx]
        session_id = selected_session['id']
        
        st.write(f"**Ngày:** {selected_session['ngay_diem_danh']}")
        st.write(f"**Tạo lúc:** {selected_session['created_at']}")
        
        attendance_list = database.get_attendance_by_session(session_id)
        stats = database.get_session_by_id(session_id) or {
            'tong_so_sv': 0,
            'co_mat': 0,
            'vang_mat': 0,
        }
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric(LABEL_TOTAL_STUDENTS, stats['tong_so_sv'])
        with col_s2:
            st.metric(LABEL_PRESENT, stats['co_mat'])
        with col_s3:
            st.metric(LABEL_ABSENT, stats['vang_mat'])
        
        st.divider()
        
        st.subheader("Chi Tiết Điểm Danh")
        
        if attendance_list:
            for idx, record in enumerate(attendance_list, 1):
                if record['trang_thai'] == "CÓ MẶT":
                    st.success(f"{idx}. {record['mssv']} - {record['ho_ten']}")
                else:
                    st.error(f"{idx}. {record['mssv']} - {record['ho_ten']}")
        else:
            st.info("Không có dữ liệu cho buổi này")
        
        st.divider()
        
        st.subheader("Xuất File Excel")
        
        if st.button("Tải Xuống Excel", use_container_width=True):
            attendance_data = []
            for record in attendance_list:
                attendance_data.append({
                    'mssv': record['mssv'],
                    'ho_ten': record['ho_ten'],
                    'trang_thai': record['trang_thai']
                })
            
            success, msg, filepath = export_attendance_to_excel(
                session_data={
                    'id': session_id,
                    'ngay_diem_danh': selected_session['ngay_diem_danh'],
                    'created_at': selected_session['created_at']
                },
                attendance_list=attendance_data
            )
            
            if success:
                st.success(f"{msg}")
                
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="Tải File Excel",
                        data=f.read(),
                        file_name=os.path.basename(filepath),
                        mime=EXCEL_MIME_TYPE,
                        use_container_width=True
                    )
            else:
                st.error(f"Lỗi: {msg}")

# ============================================================================
# TAB 5: BÁO CÁO & LỊCH SỬ
# ============================================================================

with tab5:
    st.header(f"Báo Cáo & {LABEL_ATTENDANCE_HISTORY}")
    
    report_option = st.radio("Chọn chế độ:", [LABEL_ATTENDANCE_HISTORY, "Xuất Excel"])
    
    # ===== LỊCH SỬ ĐIỂM DANH =====
    if report_option == LABEL_ATTENDANCE_HISTORY:
        st.subheader("Lịch Sử Các Phiên Điểm Danh")
        
        sessions = database.get_sessions(limit=50)
        
        if not sessions:
            st.info("📋 Chưa có phiên điểm danh nào")
        else:
            # Filter theo ngày
            col1, col2 = st.columns(2)
            with col1:
                from_date = st.date_input("Từ ngày", value=datetime.now() - timedelta(days=30))
            with col2:
                to_date = st.date_input("Đến ngày", value=datetime.now())
            
            # Filter sessions
            filtered_sessions = [
                s for s in sessions
                if from_date <= datetime.strptime(s['ngay_diem_danh'], '%Y-%m-%d').date() <= to_date
            ]
            
            st.write(f"**Tìm thấy {len(filtered_sessions)} phiên**")
            st.divider()
            
            # Hiển thị từng phiên
            for session in filtered_sessions:
                with st.expander(f"📅 {session['ngay_diem_danh']} - ID: {session['id']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(LABEL_TOTAL_STUDENTS, session['tong_so_sv'])
                    col2.metric(LABEL_PRESENT, session['co_mat'])
                    col3.metric(LABEL_ABSENT, session['vang_mat'])
                    if session['tong_so_sv'] > 0:
                        rate = (session['co_mat'] / session['tong_so_sv']) * 100
                        col4.metric("Tỷ Lệ", f"{rate:.1f}%")
                    
                    st.divider()
                    
                    # Danh sách chi tiết
                    attendance = database.get_attendance_by_session(session['id'])
                    
                    if attendance:
                        # Chia thành có mặt / vắng
                        co_mat_list = [a for a in attendance if a['trang_thai'] == STATUS_PRESENT]
                        vang_list = [a for a in attendance if a['trang_thai'] == STATUS_ABSENT]
                        
                        # Danh sách có mặt
                        st.write(f"**✅ {LABEL_PRESENT} ({len(co_mat_list)})**")
                        if co_mat_list:
                            co_mat_df = []
                            for a in co_mat_list:
                                co_mat_df.append({
                                    'MSSV': a['mssv'],
                                    'Họ Tên': a['ho_ten'],
                                    'Độ Tương Đồng': f"{a['similarity_score']:.4f}" if a['similarity_score'] else '-'
                                })
                            st.dataframe(co_mat_df, use_container_width=True)
                        
                        # Danh sách vắng
                        st.write(f"**❌ {LABEL_ABSENT} ({len(vang_list)})**")
                        if vang_list:
                            vang_df = []
                            for a in vang_list:
                                vang_df.append({
                                    'MSSV': a['mssv'],
                                    'Họ Tên': a['ho_ten']
                                })
                            st.dataframe(vang_df, use_container_width=True)
    
    # ===== XUẤT EXCEL =====
    elif report_option == "Xuất Excel":
        st.subheader("Xuất Báo Cáo Excel")
        
        sessions = database.get_sessions(limit=50)
        
        if not sessions:
            st.info("📋 Chưa có phiên điểm danh nào để xuất")
        else:
            col1, col2 = st.columns(2)
            
            # Option 1: Export single session
            with col1:
                st.write("**Export 1 Phiên**")
                session_options = {f"{s['ngay_diem_danh']} (ID: {s['id']})": s['id'] for s in sessions}
                selected_session = st.selectbox("Chọn phiên:", session_options.keys())
                
                if selected_session:
                    session_id = session_options[selected_session]
                    if st.button("📥 Tải Excel", key="export_single"):
                        # Export
                        output_path = f"data/attendance_reports/attendance_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        success = export_attendance_to_excel(database, session_id, output_path)
                        
                        if success:
                            with open(output_path, 'rb') as file:
                                st.download_button(
                                    label="📥 Tải File Excel",
                                    data=file.read(),
                                    file_name=os.path.basename(output_path),
                                    mime=EXCEL_MIME_TYPE
                                )
                        else:
                            st.error("❌ Xuất Excel thất bại")
            
            # Option 2: Export month report
            with col2:
                st.write("**Export Báo Cáo Tháng**")
                month = st.number_input("Tháng", 1, 12, datetime.now().month)
                year = st.number_input("Năm", 2020, 2050, datetime.now().year)
                
                if st.button("📥 Tải Báo Cáo Tháng", key="export_month"):
                    # Filter sessions của tháng/năm
                    month_sessions = [
                        s for s in sessions
                        if datetime.strptime(s['ngay_diem_danh'], '%Y-%m-%d').month == month
                        and datetime.strptime(s['ngay_diem_danh'], '%Y-%m-%d').year == year
                    ]
                    
                    if not month_sessions:
                        st.warning(f"⚠ Không có dữ liệu cho tháng {month}/{year}")
                    else:
                        from utils.export import ExcelExporter
                        output_path = f"data/attendance_reports/report_{year}_{month:02d}_{datetime.now().strftime('%H%M%S')}.xlsx"
                        
                        success = ExcelExporter.export_month_report(month, year, month_sessions, output_path)
                        
                        if success:
                            with open(output_path, 'rb') as file:
                                st.download_button(
                                    label=f"📥 Tải Báo Cáo Tháng {month}/{year}",
                                    data=file.read(),
                                    file_name=os.path.basename(output_path),
                                    mime=EXCEL_MIME_TYPE
                                )
                        else:
                            st.error("❌ Xuất báo cáo tháng thất bại")

