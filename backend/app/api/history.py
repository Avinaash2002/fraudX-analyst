"""
FraudX Analyst - History API
==============================
GET    /api/v1/history              → list simulation history for a device
GET    /api/v1/history/{id}         → single simulation detail
DELETE /api/v1/history/{id}         → delete a simulation record
DELETE /api/v1/history              → clear all history for a device
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import Optional
import json

from app.models import SimulationHistory
from app.database import get_db, ensure_device

router = APIRouter()


# ── GET /history ───────────────────────────────────────────────────────────────
@router.get("/history")
async def get_history(
    device_id : str = Query(..., description="Device ID to fetch history for"),
    limit     : int = Query(50, description="Max number of records to return"),
    db        : AsyncSession = Depends(get_db)
):
    """
    Returns simulation history for a specific device, newest first.
    """
    await ensure_device(db, device_id)

    result = await db.execute(
        select(SimulationHistory)
        .where(SimulationHistory.device_id == device_id)
        .order_by(SimulationHistory.timestamp.desc())
        .limit(limit)
    )
    records = result.scalars().all()

    return {
        "history": [
            {
                "simulation_id"     : str(r.simulation_id),
                "timestamp"         : r.timestamp.isoformat() if r.timestamp else None,
                "transaction_amount": r.transaction_amount,
                "prediction_result" : r.prediction_result,
                "risk_score"        : r.risk_score,
                "location"          : r.location,
                "card_number"       : r.card_number,
            }
            for r in records
        ],
        "total": len(records)
    }


# ── GET /history/{id} ──────────────────────────────────────────────────────────
@router.get("/history/{simulation_id}")
async def get_simulation_detail(
    simulation_id : str,
    db            : AsyncSession = Depends(get_db)
):
    """
    Returns full detail for a single simulation including SHAP and AI explanation.
    Used by the Simulation Result detail screen in Flutter.
    """
    result = await db.execute(
        select(SimulationHistory)
        .where(SimulationHistory.simulation_id == simulation_id)
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Simulation not found")

    top_features = []
    try:
        top_features = json.loads(record.top_features) if record.top_features else []
    except Exception:
        pass

    return {
        "simulation_id"     : str(record.simulation_id),
        "timestamp"         : record.timestamp.isoformat() if record.timestamp else None,
        "transaction_amount": record.transaction_amount,
        "location"          : record.location,
        "prediction_result" : record.prediction_result,
        "risk_score"        : record.risk_score,
        "confidence_score"  : record.confidence_score,
        "processing_time"   : record.processing_time,
        "xai_explanation"   : record.xai_explanation,
        "top_features"      : top_features,
        "card_number"       : record.card_number,
    }


# ── DELETE /history/{id} ───────────────────────────────────────────────────────
@router.delete("/history/{simulation_id}")
async def delete_simulation(
    simulation_id : str,
    db            : AsyncSession = Depends(get_db)
):
    """Deletes a single simulation record."""
    result = await db.execute(
        select(SimulationHistory)
        .where(SimulationHistory.simulation_id == simulation_id)
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Simulation not found")

    await db.delete(record)
    await db.commit()
    return {"message": "Simulation deleted successfully"}


# ── DELETE /history ────────────────────────────────────────────────────────────
@router.delete("/history")
async def clear_history(
    device_id : str = Query(..., description="Device ID to clear history for"),
    db        : AsyncSession = Depends(get_db)
):
    """Clears all simulation history for a device."""
    await db.execute(
        delete(SimulationHistory)
        .where(SimulationHistory.device_id == device_id)
    )
    await db.commit()
    return {"message": "History cleared successfully"}
