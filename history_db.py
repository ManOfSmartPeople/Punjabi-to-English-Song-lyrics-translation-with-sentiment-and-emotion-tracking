import os
import sqlite3
import json
from datetime import datetime

class HistoryDB:
    def __init__(self, db_path=None):
        base=os.path.dirname(os.path.abspath(__file__))
        self.db_path=db_path or os.path.join(base,"data","history.db")
        os.makedirs(os.path.dirname(self.db_path),exist_ok=True)
        self.conn=sqlite3.connect(self.db_path,check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT,
            original_text TEXT NOT NULL,
            filtered_text TEXT,
            translation TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            emotion_label TEXT,
            emotion_score REAL,
            confidence REAL,
            explicit_count INTEGER DEFAULT 0,
            result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            line_no INTEGER NOT NULL,
            original_line TEXT,
            translated_line TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            emotion_label TEXT,
            emotion_score REAL,
            FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        );
        """)
        self.conn.commit()

    def save_analysis(self,result,original_text=None,source="manual",title=""):
        if original_text is None: original_text=result.get("original_text","")
        cur=self.conn.cursor(); now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        s=result.get("sentiment",{}); e=result.get("emotion",{}); ex=result.get("explicit",{})
        cur.execute("""INSERT INTO analyses(created_at,source,title,original_text,filtered_text,translation,sentiment_label,sentiment_score,emotion_label,emotion_score,confidence,explicit_count,result_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(
            now,source,title,original_text,result.get("filtered_input",""),result.get("translation",""),s.get("label","neutral"),s.get("score",0),e.get("label","neutral"),e.get("score",0),result.get("overall_confidence",0),ex.get("count",0),json.dumps(result,ensure_ascii=False)))
        aid=cur.lastrowid
        for line in result.get("line_analysis",[]) or []:
            ls=line.get("sentiment",{}); le=line.get("emotion",{})
            cur.execute("""INSERT INTO analysis_lines(analysis_id,line_no,original_line,translated_line,sentiment_label,sentiment_score,emotion_label,emotion_score) VALUES(?,?,?,?,?,?,?,?)""",(aid,line.get("line_no",0),line.get("original",""),line.get("translation",""),ls.get("label","neutral"),ls.get("score",0),le.get("label","neutral"),le.get("score",0)))
        self.conn.commit(); return aid

    def recent(self,limit=100):
        cur=self.conn.cursor(); cur.execute("SELECT id,created_at,source,title,sentiment_label,emotion_label,confidence,explicit_count FROM analyses ORDER BY id DESC LIMIT ?",(int(limit),)); return cur.fetchall()
    def get(self,analysis_id):
        cur=self.conn.cursor(); cur.execute("SELECT result_json FROM analyses WHERE id=?",(int(analysis_id),)); row=cur.fetchone(); return json.loads(row[0]) if row else None
    def delete(self,analysis_id): self.conn.execute("DELETE FROM analyses WHERE id=?",(int(analysis_id),)); self.conn.commit()
    def clear(self): self.conn.execute("DELETE FROM analyses"); self.conn.commit()
    def close(self):
        try:self.conn.close()
        except Exception:pass
