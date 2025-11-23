import sqlite3
import json
from db_utils import init_db, save_work_history, get_work_history

# 데이터베이스 초기화
print("데이터베이스 초기화 중...")
init_db()

# 테이블 확인
conn = sqlite3.connect('similarity_results.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = c.fetchall()
print(f"생성된 테이블: {tables}")

# work_history 테이블 구조 확인
c.execute("PRAGMA table_info(work_history)")
columns = c.fetchall()
print(f"work_history 테이블 구조: {columns}")

# 테스트 데이터 삽입
print("테스트 데이터 삽입 중...")
test_work_id = save_work_history(
    work_type="color_comparison",
    title="테스트 색상 비교",
    description="테스트용 데이터",
    input_images=json.dumps(["test1.jpg", "test2.jpg"]),
    parameters=json.dumps({"test": "value"}),
    results=json.dumps({"similarity_score": 85.5})
)

print(f"삽입된 작업 ID: {test_work_id}")

# 데이터 조회 테스트
print("데이터 조회 테스트 중...")
history = get_work_history("color_comparison")
print(f"조회된 데이터 수: {len(history)}")

if len(history) > 0:
    print("첫 번째 데이터:")
    print(history.iloc[0].to_dict())

conn.close()
print("테스트 완료!")