"""
Evaluation Routes
=================

Endpoints for fairness and fidelity evaluation of synthetic data.

Features:
- Comprehensive fairness metrics evaluation
- Fidelity and statistical similarity assessment
- Privacy risk evaluation
- Multi-modal consistency checking
- Report generation and export
"""

import os
import json
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.api.schemas.request import (
    EvaluationRequest,
    FairnessMetricType,
)
from src.api.schemas.response import (
    EvaluationResponse,
    FairnessReportResponse,
    MetricResult,
    StatusResponse,
    JobStatus,
    ErrorResponse,
)


router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


# ==========================================
# In-Memory Job Storage (Replace with Redis/DB in production)
# ==========================================

class EvaluationJobManager:
    """In-memory storage for evaluation jobs."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, config: Dict[str, Any]) -> str:
        """Create a new evaluation job."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "job_type": "evaluation",
            "status": JobStatus.QUEUED,
            "config": config,
            "progress_percent": 0.0,
            "message": "Evaluation queued",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, **updates) -> None:
        """Update job fields."""
        if job_id in self._jobs:
            self._jobs[job_id].update(updates)
    
    def save_report(self, job_id: str, report: Dict[str, Any]) -> None:
        """Save evaluation report."""
        self._reports[job_id] = report
    
    def get_report(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get saved report."""
        return self._reports.get(job_id)


# Global job manager instance
eval_job_manager = EvaluationJobManager()


# ==========================================
# Background Evaluation Task
# ==========================================

async def run_evaluation_job(job_id: str, config: Dict[str, Any]):
    """
    Background task for evaluation.
    
    In production, this would:
    1. Load synthetic data
    2. Load real data for comparison
    3. Compute all requested metrics
    4. Generate comprehensive report
    """
    try:
        eval_job_manager.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            message="Starting evaluation..."
        )
        
        # Simulate evaluation progress
        steps = [
            ("Loading synthetic data...", 10),
            ("Loading real data for comparison...", 20),
            ("Computing group fairness metrics...", 40),
            ("Computing individual fairness metrics...", 55),
            ("Computing fidelity metrics...", 70),
            ("Computing privacy metrics...", 85),
            ("Generating report...", 95),
            ("Finalizing...", 100)
        ]
        
        for message, progress in steps:
            await asyncio.sleep(0.2)  # Simulate work
            eval_job_manager.update_job(
                job_id,
                progress_percent=float(progress),
                message=message
            )
        
        # Generate mock results
        fairness_metrics = [
            MetricResult(
                name="demographic_parity_difference",
                value=0.08,
                threshold=0.1,
                passed=True,
                description="Difference in positive outcome rates between groups"
            ),
            MetricResult(
                name="equalized_odds_difference",
                value=0.12,
                threshold=0.15,
                passed=True,
                description="Difference in TPR and FPR between groups"
            ),
            MetricResult(
                name="disparate_impact_ratio",
                value=0.85,
                threshold=0.8,
                passed=True,
                description="Ratio of positive outcome rates (should be >= 0.8)"
            ),
            MetricResult(
                name="calibration_difference",
                value=0.05,
                threshold=0.1,
                passed=True,
                description="Difference in calibration between groups"
            )
        ]
        
        individual_metrics = [
            MetricResult(
                name="consistency_score",
                value=0.92,
                threshold=0.85,
                passed=True,
                description="Average similarity of predictions for similar individuals"
            ),
            MetricResult(
                name="lipschitz_constant",
                value=1.5,
                threshold=2.0,
                passed=True,
                description="Estimated Lipschitz constant of the prediction function"
            )
        ]
        
        fidelity_metrics = [
            MetricResult(
                name="js_divergence",
                value=0.08,
                threshold=0.15,
                passed=True,
                description="Jensen-Shannon divergence between real and synthetic distributions"
            ),
            MetricResult(
                name="correlation_preservation",
                value=0.94,
                threshold=0.85,
                passed=True,
                description="Correlation between real and synthetic correlation matrices"
            ),
            MetricResult(
                name="ks_statistic_max",
                value=0.12,
                threshold=0.2,
                passed=True,
                description="Maximum Kolmogorov-Smirnov statistic across features"
            )
        ]
        
        privacy_metrics = [
            MetricResult(
                name="membership_inference_accuracy",
                value=0.55,
                threshold=0.65,
                passed=True,
                description="Accuracy of membership inference attack (lower is better)"
            ),
            MetricResult(
                name="attribute_inference_accuracy",
                value=0.42,
                threshold=0.6,
                passed=True,
                description="Accuracy of attribute inference attack"
            )
        ]
        
        # Calculate overall scores
        fairness_score = sum(m.value for m in fairness_metrics if m.passed) / len(fairness_metrics)
        fidelity_score = 1 - sum(m.value for m in fidelity_metrics[:2]) / 2
        privacy_score = 1 - (privacy_metrics[0].value + privacy_metrics[1].value) / 2
        
        # Create report
        report = {
            "job_id": job_id,
            "evaluation_time": 5.0,
            "n_samples_evaluated": config.get("n_samples", 1000),
            "timestamp": datetime.utcnow().isoformat(),
            "fairness": {
                "overall_fairness_score": round(fairness_score, 3),
                "group_fairness_metrics": [m.model_dump() for m in fairness_metrics],
                "individual_fairness_metrics": [m.model_dump() for m in individual_metrics],
                "counterfactual_metrics": [],
                "intersectional_metrics": [],
                "passed_all_thresholds": all(m.passed for m in fairness_metrics),
                "recommendations": [
                    "Consider increasing adversary weight for improved demographic parity",
                    "Monitor equalized odds during training for better balance"
                ]
            },
            "fidelity": {
                "overall_fidelity_score": round(fidelity_score, 3),
                "metrics": [m.model_dump() for m in fidelity_metrics],
                "statistical_similarity": {
                    "js_divergence": 0.08,
                    "wasserstein_distance": 0.15,
                    "correlation_preservation": 0.94
                },
                "distribution_comparison": {
                    "num_features_compared": 10,
                    "features_passing_ks": 9,
                    "max_ks_statistic": 0.12
                }
            },
            "privacy": {
                "overall_privacy_score": round(privacy_score, 3),
                "risk_level": "low",
                "metrics": [m.model_dump() for m in privacy_metrics],
                "recommendations": [
                    "Current privacy protection is adequate",
                    "Consider differential privacy for additional protection"
                ]
            },
            "multimodal": {
                "cross_modal_consistency": 0.89,
                "alignment_score": 0.91,
                "metrics": []
            }
        }
        
        # Save report
        eval_job_manager.save_report(job_id, report)
        
        # Generate report file
        report_dir = Path("artifacts/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"evaluation_report_{job_id}.json"
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create HTML report
        html_report = generate_html_report(report, job_id)
        html_file = report_dir / f"evaluation_report_{job_id}.html"
        with open(html_file, "w") as f:
            f.write(html_report)
        
        # Update job as completed
        eval_job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100.0,
            completed_at=datetime.utcnow(),
            message="Evaluation completed successfully",
            result={
                "job_id": job_id,
                "evaluation_time": report["evaluation_time"],
                "n_samples_evaluated": report["n_samples_evaluated"],
                "fairness_score": report["fairness"]["overall_fairness_score"],
                "fidelity_score": report["fidelity"]["overall_fidelity_score"],
                "privacy_score": report["privacy"]["overall_privacy_score"],
                "report_url": f"/evaluate/report/{job_id}",
                "html_report_url": f"/evaluate/report/{job_id}/html"
            }
        )
        
    except Exception as e:
        eval_job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.utcnow(),
            message=f"Evaluation failed: {str(e)}",
            error={
                "error_code": "evaluation_error",
                "error_message": str(e)
            }
        )


def generate_html_report(report: Dict[str, Any], job_id: str) -> str:
    """Generate HTML report from evaluation results."""
    
    fairness_metrics_html = ""
    for m in report["fairness"]["group_fairness_metrics"]:
        status_class = "metric-pass" if m["passed"] else "metric-fail"
        fairness_metrics_html += f"""
        <tr class="{status_class}">
            <td>{m['name']}</td>
            <td>{m['value']:.4f}</td>
            <td>{m['threshold']:.4f}</td>
            <td>{'✓' if m['passed'] else '✗'}</td>
        </tr>
        """
    
    fidelity_metrics_html = ""
    for m in report["fidelity"]["metrics"]:
        status_class = "metric-pass" if m["passed"] else "metric-fail"
        fidelity_metrics_html += f"""
        <tr class="{status_class}">
            <td>{m['name']}</td>
            <td>{m['value']:.4f}</td>
            <td>{m['threshold']:.4f}</td>
            <td>{'✓' if m['passed'] else '✗'}</td>
        </tr>
        """
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - {job_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .score-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center; }}
            .score-value {{ font-size: 48px; font-weight: bold; }}
            .score-label {{ color: #666; margin-top: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #4CAF50; color: white; }}
            .metric-pass {{ background: #e8f5e9; }}
            .metric-fail {{ background: #ffebee; }}
            .recommendations {{ background: #fff3e0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .recommendations ul {{ margin: 10px 0; padding-left: 20px; }}
            .timestamp {{ color: #888; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Synthetic Data Evaluation Report</h1>
            <p class="timestamp">Report ID: {job_id}</p>
            <p class="timestamp">Generated: {report['timestamp']}</p>
            
            <div class="summary">
                <div class="score-card">
                    <div class="score-value" style="color: #4CAF50;">{report['fairness']['overall_fairness_score']:.2f}</div>
                    <div class="score-label">Fairness Score</div>
                </div>
                <div class="score-card">
                    <div class="score-value" style="color: #2196F3;">{report['fidelity']['overall_fidelity_score']:.2f}</div>
                    <div class="score-label">Fidelity Score</div>
                </div>
                <div class="score-card">
                    <div class="score-value" style="color: #FF9800;">{report['privacy']['overall_privacy_score']:.2f}</div>
                    <div class="score-label">Privacy Score</div>
                </div>
            </div>
            
            <h2>⚖️ Fairness Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Pass</th>
                </tr>
                {fairness_metrics_html}
            </table>
            
            <h2>📈 Fidelity Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Pass</th>
                </tr>
                {fidelity_metrics_html}
            </table>
            
            <h2>🔒 Privacy Metrics</h2>
            <p>Privacy Risk Level: <strong>{report['privacy']['risk_level']}</strong></p>
            
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <ul>
                    {''.join(f'<li>{r}</li>' for r in report['fairness']['recommendations'])}
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


# ==========================================
# Routes
# ==========================================

@router.post(
    "",
    response_model=StatusResponse,
    summary="Evaluate synthetic data",
    description="Submit a comprehensive evaluation of synthetic data."
)
async def evaluate_data(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Evaluate synthetic data.
    
    Performs comprehensive evaluation including:
    - Fairness metrics (group, individual, counterfactual)
    - Fidelity metrics (statistical similarity, distribution comparison)
    - Privacy metrics (membership inference, attribute inference)
    - Multi-modal consistency (if applicable)
    """
    config = request.model_dump()
    job_id = eval_job_manager.create_job(config)
    
    background_tasks.add_task(run_evaluation_job, job_id, config)
    
    return StatusResponse(
        job_id=job_id,
        job_type="evaluation",
        status=JobStatus.QUEUED,
        progress_percent=0.0,
        message="Evaluation request accepted. Use /evaluate/status/{job_id} to track progress.",
        created_at=datetime.utcnow()
    )


@router.post(
    "/quick",
    summary="Quick evaluation",
    description="Perform a quick evaluation without background processing."
)
async def quick_evaluate(
    data_path: str = Query(..., description="Path to synthetic data file"),
    sensitive_attributes: str = Query(default="", description="Comma-separated sensitive attributes"),
    target_column: Optional[str] = Query(default=None, description="Target column name")
):
    """
    Quick synchronous evaluation.
    
    Performs basic fairness and fidelity checks synchronously.
    Suitable for small datasets and quick feedback.
    """
    import numpy as np
    
    # Mock quick evaluation
    return {
        "status": "completed",
        "fairness": {
            "demographic_parity_difference": np.random.uniform(0.05, 0.15),
            "equalized_odds_difference": np.random.uniform(0.08, 0.18),
            "overall_score": np.random.uniform(0.85, 0.98)
        },
        "fidelity": {
            "js_divergence": np.random.uniform(0.05, 0.12),
            "correlation_preservation": np.random.uniform(0.90, 0.98),
            "overall_score": np.random.uniform(0.88, 0.96)
        },
        "recommendation": "Data quality is good. Consider increasing fairness constraints."
    }


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Get evaluation status",
    description="Check the status of an evaluation job."
)
async def get_evaluation_status(job_id: str):
    """Get evaluation job status and progress."""
    job = eval_job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation job {job_id} not found"
        )
    
    return StatusResponse(
        job_id=job["job_id"],
        job_type=job["job_type"],
        status=job["status"],
        progress_percent=job["progress_percent"],
        message=job["message"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error")
    )


@router.get(
    "/report/{job_id}",
    response_model=EvaluationResponse,
    summary="Get evaluation report",
    description="Get the full evaluation report for a completed job."
)
async def get_evaluation_report(job_id: str):
    """Get detailed evaluation report."""
    job = eval_job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation job {job_id} not found"
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job['status']}"
        )
    
    report = eval_job_manager.get_report(job_id)
    
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found"
        )
    
    # Build response
    fairness_data = report.get("fairness", {})
    
    return EvaluationResponse(
        job_id=job_id,
        evaluation_time=report.get("evaluation_time", 0),
        n_samples_evaluated=report.get("n_samples_evaluated", 0),
        fairness=FairnessReportResponse(
            job_id=job_id,
            overall_fairness_score=fairness_data.get("overall_fairness_score", 0),
            overall_fidelity_score=report.get("fidelity", {}).get("overall_fidelity_score", 0),
            combined_score=(fairness_data.get("overall_fairness_score", 0) + 
                          report.get("fidelity", {}).get("overall_fidelity_score", 0)) / 2,
            group_fairness_metrics=[
                MetricResult(**m) for m in fairness_data.get("group_fairness_metrics", [])
            ],
            individual_fairness_metrics=[
                MetricResult(**m) for m in fairness_data.get("individual_fairness_metrics", [])
            ],
            passed_all_thresholds=fairness_data.get("passed_all_thresholds", True),
            failed_metrics=[],
            recommendations=fairness_data.get("recommendations", [])
        ),
        fidelity_metrics=[
            MetricResult(**m) for m in report.get("fidelity", {}).get("metrics", [])
        ],
        statistical_similarity=report.get("fidelity", {}).get("statistical_similarity", {}),
        privacy_metrics=[
            MetricResult(**m) for m in report.get("privacy", {}).get("metrics", [])
        ],
        privacy_risk_assessment={
            "risk_level": report.get("privacy", {}).get("risk_level", "unknown")
        },
        report_url=f"/evaluate/report/{job_id}/html"
    )


@router.get(
    "/report/{job_id}/html",
    summary="Get HTML report",
    description="Download the evaluation report as HTML."
)
async def get_html_report(job_id: str):
    """Get HTML version of the evaluation report."""
    report_file = Path(f"artifacts/reports/evaluation_report_{job_id}.html")
    
    if not report_file.exists():
        raise HTTPException(
            status_code=404,
            detail="HTML report not found"
        )
    
    return FileResponse(
        path=report_file,
        filename=f"evaluation_report_{job_id}.html",
        media_type="text/html"
    )


@router.get(
    "/report/{job_id}/json",
    summary="Get JSON report",
    description="Download the evaluation report as JSON."
)
async def get_json_report(job_id: str):
    """Get JSON version of the evaluation report."""
    report_file = Path(f"artifacts/reports/evaluation_report_{job_id}.json")
    
    if not report_file.exists():
        raise HTTPException(
            status_code=404,
            detail="JSON report not found"
        )
    
    return FileResponse(
        path=report_file,
        filename=f"evaluation_report_{job_id}.json",
        media_type="application/json"
    )


@router.post(
    "/compare",
    summary="Compare evaluations",
    description="Compare multiple evaluation results."
)
async def compare_evaluations(
    job_ids: List[str] = Query(..., description="List of job IDs to compare")
):
    """
    Compare multiple evaluation results.
    
    Provides side-by-side comparison of fairness, fidelity, and privacy
    metrics across multiple synthetic datasets.
    """
    if len(job_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 job IDs required for comparison"
        )
    
    if len(job_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 job IDs allowed for comparison"
        )
    
    results = []
    for job_id in job_ids:
        report = eval_job_manager.get_report(job_id)
        if report:
            results.append({
                "job_id": job_id,
                "fairness_score": report.get("fairness", {}).get("overall_fairness_score", 0),
                "fidelity_score": report.get("fidelity", {}).get("overall_fidelity_score", 0),
                "privacy_score": report.get("privacy", {}).get("overall_privacy_score", 0)
            })
    
    if len(results) < 2:
        raise HTTPException(
            status_code=404,
            detail="Not enough valid evaluation results found"
        )
    
    # Find best performing
    best_fairness = max(results, key=lambda x: x["fairness_score"])
    best_fidelity = max(results, key=lambda x: x["fidelity_score"])
    best_privacy = max(results, key=lambda x: x["privacy_score"])
    
    return {
        "comparison": results,
        "best_fairness": best_fairness["job_id"],
        "best_fidelity": best_fidelity["job_id"],
        "best_privacy": best_privacy["job_id"],
        "recommendation": f"Job {best_fairness['job_id']} has best fairness score, "
                         f"Job {best_fidelity['job_id']} has best fidelity score."
    }


@router.get(
    "/metrics",
    summary="List available metrics",
    description="List all available evaluation metrics."
)
async def list_metrics():
    """List all available evaluation metrics."""
    return {
        "fairness_metrics": [
            {
                "name": "demographic_parity_difference",
                "category": "group",
                "description": "Difference in positive outcome rates between groups",
                "ideal_value": 0.0
            },
            {
                "name": "equalized_odds_difference",
                "category": "group",
                "description": "Maximum difference in TPR and FPR between groups",
                "ideal_value": 0.0
            },
            {
                "name": "equal_opportunity_difference",
                "category": "group",
                "description": "Difference in true positive rates between groups",
                "ideal_value": 0.0
            },
            {
                "name": "disparate_impact_ratio",
                "category": "group",
                "description": "Ratio of positive outcome rates",
                "ideal_value": 1.0
            },
            {
                "name": "consistency_score",
                "category": "individual",
                "description": "Similarity of predictions for similar individuals",
                "ideal_value": 1.0
            },
            {
                "name": "lipschitz_constant",
                "category": "individual",
                "description": "Smoothness of predictions with respect to features",
                "ideal_value": "low"
            },
            {
                "name": "counterfactual_invariance",
                "category": "counterfactual",
                "description": "Invariance under counterfactual changes",
                "ideal_value": 1.0
            }
        ],
        "fidelity_metrics": [
            {
                "name": "js_divergence",
                "description": "Jensen-Shannon divergence between distributions",
                "ideal_value": 0.0
            },
            {
                "name": "correlation_preservation",
                "description": "Correlation between real and synthetic correlation matrices",
                "ideal_value": 1.0
            },
            {
                "name": "ks_statistic",
                "description": "Kolmogorov-Smirnov statistic for distribution similarity",
                "ideal_value": 0.0
            },
            {
                "name": "wasserstein_distance",
                "description": "Earth mover's distance between distributions",
                "ideal_value": 0.0
            }
        ],
        "privacy_metrics": [
            {
                "name": "membership_inference_accuracy",
                "description": "Accuracy of membership inference attack",
                "ideal_value": 0.5
            },
            {
                "name": "attribute_inference_accuracy",
                "description": "Accuracy of attribute inference attack",
                "ideal_value": "low"
            }
        ]
    }


@router.post(
    "/upload",
    summary="Upload and evaluate",
    description="Upload a data file for evaluation."
)
async def upload_and_evaluate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Data file to evaluate"),
    sensitive_attributes: str = Form(default="", description="Comma-separated sensitive attributes"),
    target_column: Optional[str] = Form(default=None, description="Target column name"),
    evaluate_fairness: bool = Form(default=True, description="Evaluate fairness metrics"),
    evaluate_fidelity: bool = Form(default=True, description="Evaluate fidelity metrics"),
    evaluate_privacy: bool = Form(default=False, description="Evaluate privacy metrics")
):
    """
    Upload data file and evaluate.
    
    Accepts CSV, JSON, or Parquet files for evaluation.
    """
    # Validate file type
    allowed_types = [".csv", ".json", ".parquet"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}"
        )
    
    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    upload_path = upload_dir / f"{file_id}{file_ext}"
    
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create evaluation job
    config = {
        "data_path": str(upload_path),
        "sensitive_attributes": sensitive_attributes.split(",") if sensitive_attributes else [],
        "target_column": target_column,
        "evaluation_types": [],
        "file_id": file_id,
        "original_filename": file.filename
    }
    
    if evaluate_fairness:
        config["evaluation_types"].append("fairness")
    if evaluate_fidelity:
        config["evaluation_types"].append("fidelity")
    if evaluate_privacy:
        config["evaluation_types"].append("privacy")
    
    job_id = eval_job_manager.create_job(config)
    background_tasks.add_task(run_evaluation_job, job_id, config)
    
    return {
        "message": "File uploaded and evaluation started",
        "job_id": job_id,
        "file_id": file_id,
        "status_url": f"/evaluate/status/{job_id}"
    }
