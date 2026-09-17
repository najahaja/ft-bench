import time
from sqlalchemy import Column, Float, Integer, String, Text, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = "sqlite:///serving/requests.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time)
    system = Column(String, index=True)
    utterance = Column(Text)
    raw_output = Column(Text)
    is_json_valid = Column(Boolean, default=False)
    is_schema_valid = Column(Boolean, default=False)
    latency_ms = Column(Float)
    tokens_generated = Column(Integer, default=0)
    throughput_tok_s = Column(Float, default=0.0)
    parse_error = Column(Text, nullable=True)


Base.metadata.create_all(bind=engine)


def log_request(system, utterance, raw_output, is_json_valid, is_schema_valid,
                latency_ms, tokens_generated, throughput_tok_s, parse_error=None):
    db = SessionLocal()
    try:
        record = InferenceLog(
            timestamp=time.time(), system=system, utterance=utterance,
            raw_output=raw_output, is_json_valid=is_json_valid,
            is_schema_valid=is_schema_valid, latency_ms=latency_ms,
            tokens_generated=tokens_generated, throughput_tok_s=throughput_tok_s,
            parse_error=parse_error,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def get_recent_logs(limit: int = 50) -> list:
    db = SessionLocal()
    try:
        rows = db.query(InferenceLog).order_by(InferenceLog.id.desc()).limit(limit).all()
        return [
            {"id": r.id, "system": r.system, "utterance": r.utterance,
             "is_json_valid": r.is_json_valid, "is_schema_valid": r.is_schema_valid,
             "latency_ms": r.latency_ms, "throughput_tok_s": r.throughput_tok_s,
             "timestamp": r.timestamp}
            for r in rows
        ]
    finally:
        db.close()
