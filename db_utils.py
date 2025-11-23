import sqlite3
import datetime
import uuid
import pandas as pd
import os

# 결과 저장을 위한 폴더 생성
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# SQLite DB 초기화
def init_db():
    ensure_dir("Result")  # Result 폴더 생성
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    
    # 기존 테이블의 존재 여부 확인
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    table_exists = c.fetchone()
    
    if not table_exists:
        # 테이블이 없으면 새로 생성
        c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            real_image_path TEXT,
            ai_image_path TEXT,
            ssim_score REAL,
            psnr_score REAL,
            vgg_score REAL,
            avg_score REAL,
            comment TEXT DEFAULT NULL
        )
        ''')
    else:
        # 테이블이 존재하면 comment 컬럼이 있는지 확인
        c.execute("PRAGMA table_info(results)")
        columns = c.fetchall()
        column_names = [column[1] for column in columns]
        
        # comment 컬럼이 없으면 추가
        if 'comment' not in column_names:
            c.execute("ALTER TABLE results ADD COLUMN comment TEXT DEFAULT NULL")
    
    # 모든 작업 유형을 저장하는 새로운 테이블 생성
    c.execute('''
    CREATE TABLE IF NOT EXISTS work_history (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        work_type TEXT,
        title TEXT,
        description TEXT,
        input_images TEXT,
        output_images TEXT,
        parameters TEXT,
        results TEXT,
        comment TEXT DEFAULT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# DB에 결과 저장
def save_results(real_image_path, ai_image_path, ssim_score, psnr_score, vgg_score, avg_score):
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    result_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    c.execute('''
    INSERT INTO results (id, timestamp, real_image_path, ai_image_path, ssim_score, psnr_score, vgg_score, avg_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (result_id, timestamp, real_image_path, ai_image_path, ssim_score, psnr_score, vgg_score, avg_score))
    conn.commit()
    conn.close()
    return result_id

# DB에서 결과 가져오기
def get_results():
    conn = sqlite3.connect('similarity_results.db')
    query = "SELECT * FROM results ORDER BY timestamp DESC"
    try:
        results = pd.read_sql(query, conn)
    except:
        results = pd.DataFrame(columns=["id", "timestamp", "real_image_path", "ai_image_path", "ssim_score", "psnr_score", "vgg_score", "avg_score", "comment"])
    conn.close()
    return results

# 결과에 코멘트 저장 함수 추가
def save_comment(result_id, comment):
    """
    결과에 코멘트를 저장하는 함수
    
    Args:
        result_id (str): 결과 ID
        comment (str): 저장할 코멘트
    
    Returns:
        bool: 저장 성공 여부
    """
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    c.execute("UPDATE results SET comment = ? WHERE id = ?", (comment, result_id))
    conn.commit()
    conn.close()
    return True

# 작업 히스토리 저장 함수
def save_work_history(work_type, title, description, input_images=None, output_images=None, parameters=None, results=None, comment=None):
    """
    작업 히스토리를 저장하는 함수
    
    Args:
        work_type (str): 작업 유형 ('image_comparison', 'color_analysis', 'image_upscaling')
        title (str): 작업 제목
        description (str): 작업 설명
        input_images (str): 입력 이미지 경로 (JSON 문자열)
        output_images (str): 출력 이미지 경로 (JSON 문자열)
        parameters (str): 작업 매개변수 (JSON 문자열)
        results (str): 작업 결과 (JSON 문자열)
        comment (str): 코멘트
    
    Returns:
        str: 작업 ID
    """
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    work_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    c.execute('''
    INSERT INTO work_history (id, timestamp, work_type, title, description, input_images, output_images, parameters, results, comment)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (work_id, timestamp, work_type, title, description, input_images, output_images, parameters, results, comment))
    
    conn.commit()
    conn.close()
    return work_id

# 작업 히스토리 조회 함수
def get_work_history(work_type=None):
    """
    작업 히스토리를 조회하는 함수
    
    Args:
        work_type (str): 필터링할 작업 유형 (None이면 모든 유형)
    
    Returns:
        pandas.DataFrame: 작업 히스토리 데이터
    """
    conn = sqlite3.connect('similarity_results.db')
    
    if work_type:
        query = "SELECT * FROM work_history WHERE work_type = ? ORDER BY timestamp DESC"
        try:
            results = pd.read_sql(query, conn, params=[work_type])
        except:
            results = pd.DataFrame(columns=["id", "timestamp", "work_type", "title", "description", "input_images", "output_images", "parameters", "results", "comment"])
    else:
        query = "SELECT * FROM work_history ORDER BY timestamp DESC"
        try:
            results = pd.read_sql(query, conn)
        except:
            results = pd.DataFrame(columns=["id", "timestamp", "work_type", "title", "description", "input_images", "output_images", "parameters", "results", "comment"])
    
    conn.close()
    return results

# 작업 히스토리 코멘트 저장 함수
def save_work_comment(work_id, comment):
    """
    작업 히스토리에 코멘트를 저장하는 함수
    
    Args:
        work_id (str): 작업 ID
        comment (str): 저장할 코멘트
    
    Returns:
        bool: 저장 성공 여부
    """
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    c.execute("UPDATE work_history SET comment = ? WHERE id = ?", (comment, work_id))
    conn.commit()
    conn.close()
    return True