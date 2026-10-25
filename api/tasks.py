"""Background job tracker and async ingestion tasks for large financial documents."""

import uuid
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestionJobStatus(BaseModel):
    """Schema for tracking async document ingestion lifecycle."""
    job_id: str
    filename: str
    status: str = Field(description="queued | processing | completed | failed")
    chunks_created: int = 0
    pages_parsed: int = 0
    doc_id: Optional[str] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None


class JobManager:
    """Thread-safe job state tracker."""

    def __init__(self):
        self._jobs: Dict[str, IngestionJobStatus] = {}

    def create_job(self, filename: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = IngestionJobStatus(
            job_id=job_id,
            filename=filename,
            status="queued"
        )
        return job_id

    def update_job(self, job_id: str, **kwargs) -> Optional[IngestionJobStatus]:
        if job_id not in self._jobs:
            return None
        job = self._jobs[job_id]
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJobStatus]:
        return self._jobs.get(job_id)


job_manager = JobManager()
