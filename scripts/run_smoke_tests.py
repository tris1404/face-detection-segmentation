from utils.face_recognition import FaceRecognizer
from utils.database import Database
import numpy as np, os

print('Testing embedding JSON roundtrip...')
emb = np.arange(512,dtype=np.float32)
js = FaceRecognizer.embedding_to_json(emb)
back = FaceRecognizer.json_to_embedding(js)
if back is not None and back.shape==(512,) and all(abs(back - emb) < 1e-6):
    print('OK')
else:
    print('FAIL')

print('\nTesting Database add/delete...')
db_path = 'data/db/test_attendance_tmp.db'
if os.path.exists(db_path):
    os.remove(db_path)

try:
    db = Database(db_path)
    res = db.add_student('TST001','Test User','data/students/tst.jpg',embedding=FaceRecognizer.embedding_to_json(emb), on_duplicate='skip')
    print('add_student:', res)
    students = db.get_all_students()
    print('students count:', len(students))
    if res.get('student_id'):
        ok = db.delete_student(res['student_id'])
        print('delete_student:', ok)
    else:
        print('No student_id returned')
except Exception as e:
    print('DB test error:', e)

# cleanup
try:
    if os.path.exists(db_path):
        os.remove(db_path)
except Exception:
    pass
