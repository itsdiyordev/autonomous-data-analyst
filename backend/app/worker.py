import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta

from sqlalchemy import select, update

from .config import settings
from .db import AnalysisRun, SessionLocal, init_db, now
from .ml import train_run

logger = logging.getLogger(__name__)


def claim_job():
    with SessionLocal() as db:
        job_id = db.scalar(select(AnalysisRun.id).where(AnalysisRun.status == "queued").order_by(AnalysisRun.created_at).limit(1))
        if not job_id:
            return None
        result = db.execute(update(AnalysisRun).where(AnalysisRun.id == job_id, AnalysisRun.status == "queued").values(status="running", stage="Starting analysis", updated_at=now()))
        db.commit()
        return job_id if result.rowcount == 1 else None


def recover_stale():
    # Failed workers never leave a job looking active forever. Do not requeue automatically:
    # a user can rerun without publishing duplicate artifacts from an uncertain execution.
    cutoff = now() - timedelta(minutes=30)
    with SessionLocal() as db:
        db.execute(update(AnalysisRun).where(AnalysisRun.status == "running", AnalysisRun.updated_at < cutoff).values(status="failed", error="The worker stopped responding. Please start a new analysis.", stage="Worker interrupted", finished_at=now()))
        db.commit()


async def dispatcher():
    pool = ProcessPoolExecutor(max_workers=settings.max_workers, mp_context=multiprocessing.get_context("spawn"))
    active = {}
    last_recovery = 0.0
    try:
        while True:
            done = {future for future in active if future.done()}
            broken = False
            for future in done:
                try:
                    future.result()
                except Exception as exc:
                    logger.exception("Worker process failed")
                    broken = broken or isinstance(exc, BrokenProcessPool)
                    with SessionLocal() as db:
                        db.execute(update(AnalysisRun).where(
                            AnalysisRun.id == active[future], AnalysisRun.status == "running"
                        ).values(status="failed", error="The worker process was interrupted. Please retry the analysis.",
                                 stage="Worker interrupted", finished_at=now()))
                        db.commit()
                del active[future]
            if broken:
                pool.shutdown(wait=False, cancel_futures=True)
                pool = ProcessPoolExecutor(max_workers=settings.max_workers, mp_context=multiprocessing.get_context("spawn"))
            if time.monotonic() - last_recovery > 60:
                recover_stale()
                last_recovery = time.monotonic()
            while len(active) < settings.max_workers:
                job_id = claim_job()
                if not job_id:
                    break
                active[pool.submit(train_run, job_id)] = job_id
            await asyncio.sleep(1)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    asyncio.run(dispatcher())
