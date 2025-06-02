from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import json
import asyncio
import httpx
from datetime import datetime, timedelta
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import our NLP processor (modified for async)
from nlp_processor import SupportTicketNLPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Attribution Model NLP Service",
    description="Automated support ticket analysis and behavioral insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables
nlp_processor = None
scheduler = None
processing_status = {
    "last_run": None,
    "status": "idle",
    "tickets_processed": 0,
    "errors": 0
}

# Pydantic models
class ProcessingResult(BaseModel):
    status: str
    tickets_processed: int
    processing_time: float
    insights: Dict
    timestamp: datetime

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    uptime: str
    last_processing: Optional[datetime]

class TriggerRequest(BaseModel):
    force_reprocess: bool = False
    date_range_days: int = 1

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    expected_token = os.getenv("API_TOKEN", "dev-token-123")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials

# Initialize services
async def initialize_services():
    """Initialize NLP processor and scheduler"""
    global nlp_processor, scheduler
    
    try:
        # Initialize NLP processor
        project_id = os.getenv("GCP_PROJECT_ID", "mops-lab")
        dataset_id = os.getenv("GCP_DATASET_ID", "user_signals")
        
        nlp_processor = SupportTicketNLPProcessor(project_id, dataset_id)
        logger.info("✅ NLP processor initialized")
        
        # Initialize scheduler
        scheduler = BackgroundScheduler()
        
        # Schedule daily processing at 6 AM UTC
        scheduler.add_job(
            func=scheduled_processing,
            trigger=CronTrigger(hour=6, minute=0),
            id='daily_nlp_processing',
            replace_existing=True
        )
        
        # Schedule weekly full reprocessing (Sundays at 2 AM)
        scheduler.add_job(
            func=weekly_full_processing,
            trigger=CronTrigger(day_of_week=6, hour=2, minute=0),
            id='weekly_full_processing',
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("✅ Scheduler started")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {str(e)}")
        raise

async def scheduled_processing():
    """Daily scheduled processing of new tickets"""
    logger.info("🔄 Starting scheduled NLP processing...")
    
    try:
        # Process tickets from last 24 hours
        result = await process_support_tickets(days_back=1)
        
        # Trigger dbt refresh
        await trigger_dbt_refresh()
        
        # Send success notification
        await send_processing_notification(result, success=True)
        
        logger.info(f"✅ Scheduled processing complete: {result['tickets_processed']} tickets")
        
    except Exception as e:
        logger.error(f"❌ Scheduled processing failed: {str(e)}")
        await send_processing_notification({"error": str(e)}, success=False)

async def weekly_full_processing():
    """Weekly full reprocessing of recent tickets"""
    logger.info("🔄 Starting weekly full reprocessing...")
    
    try:
        # Process tickets from last 30 days
        result = await process_support_tickets(days_back=30, force_reprocess=True)
        
        # Trigger full dbt refresh
        await trigger_dbt_refresh(full_refresh=True)
        
        logger.info(f"✅ Weekly processing complete: {result['tickets_processed']} tickets")
        
    except Exception as e:
        logger.error(f"❌ Weekly processing failed: {str(e)}")

async def process_support_tickets(days_back: int = 1, force_reprocess: bool = False) -> Dict:
    """Core processing function"""
    global processing_status
    
    start_time = datetime.now()
    processing_status["status"] = "processing"
    
    try:
        # Load and process tickets
        results = await asyncio.to_thread(nlp_processor.process_tickets_timeframe, days_back, force_reprocess)
        
        # Save to BigQuery
        await asyncio.to_thread(nlp_processor.save_to_bigquery, results)
        
        # Generate insights
        insights = await asyncio.to_thread(nlp_processor.generate_insights_summary, results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update status
        processing_status.update({
            "last_run": datetime.now(),
            "status": "completed",
            "tickets_processed": len(results),
            "errors": 0
        })
        
        return {
            "status": "success",
            "tickets_processed": len(results),
            "processing_time": processing_time,
            "insights": insights,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        processing_status.update({
            "status": "error",
            "errors": processing_status["errors"] + 1
        })
        raise e

async def trigger_dbt_refresh(full_refresh: bool = False):
    """Trigger dbt Cloud job refresh"""
    dbt_webhook_url = os.getenv("DBT_WEBHOOK_URL")
    dbt_token = os.getenv("DBT_API_TOKEN")
    
    if not dbt_webhook_url or not dbt_token:
        logger.warning("⚠️ dbt configuration missing, skipping refresh")
        return
    
    try:
        headers = {
            "Authorization": f"Token {dbt_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "cause": "NLP processing completed",
            "git_sha": None,
            "schema_override": None,
            "dbt_version_override": None,
            "timeout_seconds": 3600,
            "generate_docs": True,
            "run_steps": [
                "dbt run --models engagement_fatigue_scores behavioral_segments"
            ]
        }
        
        if full_refresh:
            payload["run_steps"] = ["dbt run --full-refresh"]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(dbt_webhook_url, headers=headers, json=payload)
            
        if response.status_code == 200:
            logger.info("✅ dbt refresh triggered successfully")
        else:
            logger.error(f"❌ dbt refresh failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ dbt refresh error: {str(e)}")

async def send_processing_notification(result: Dict, success: bool = True):
    """Send processing notifications to Slack/email"""
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    
    if not slack_webhook:
        return
    
    try:
        if success:
            message = f"""
🤖 *NLP Processing Complete*
✅ Status: Success
📊 Tickets Processed: {result.get('tickets_processed', 0)}
⏱️ Processing Time: {result.get('processing_time', 0):.1f}s
🎯 Growth Signals: {result.get('insights', {}).get('growth_signals_detected', 0)}
😤 Frustrated Customers: {result.get('insights', {}).get('frustrated_customers', 0)}
            """
        else:
            message = f"""
🚨 *NLP Processing Failed*
❌ Error: {result.get('error', 'Unknown error')}
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        
        payload = {"text": message}
        
        async with httpx.AsyncClient() as client:
            await client.post(slack_webhook, json=payload)
            
    except Exception as e:
        logger.error(f"❌ Notification failed: {str(e)}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_services()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if scheduler:
        scheduler.shutdown()

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - datetime.fromtimestamp(os.path.getctime(__file__))
    
    return HealthCheck(
        status="healthy" if processing_status["status"] != "error" else "degraded",
        timestamp=datetime.now(),
        uptime=str(uptime),
        last_processing=processing_status.get("last_run")
    )

@app.post("/process", response_model=ProcessingResult)
async def manual_trigger(
    background_tasks: BackgroundTasks,
    request: TriggerRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Manually trigger NLP processing"""
    
    if processing_status["status"] == "processing":
        raise HTTPException(status_code=409, detail="Processing already in progress")
    
    # Run processing in background
    background_tasks.add_task(
        process_support_tickets, 
        request.date_range_days, 
        request.force_reprocess
    )
    
    return ProcessingResult(
        status="started",
        tickets_processed=0,
        processing_time=0.0,
        insights={},
        timestamp=datetime.now()
    )

@app.get("/status")
async def get_processing_status(credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    """Get current processing status"""
    return processing_status

@app.post("/dbt/trigger")
async def trigger_dbt_manual(
    full_refresh: bool = False,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Manually trigger dbt refresh"""
    await trigger_dbt_refresh(full_refresh)
    return {"status": "triggered", "full_refresh": full_refresh}

@app.get("/insights/latest")
async def get_latest_insights(credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    """Get latest processing insights"""
    
    try:
        # Query latest insights from BigQuery
        insights = await asyncio.to_thread(nlp_processor.get_latest_insights)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/process-sample")
async def test_process_sample(credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    """Test endpoint with sample data"""
    
    try:
        # Process a small sample for testing
        result = await process_support_tickets(days_back=1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates (optional)
@app.websocket("/ws/status")
async def websocket_status(websocket):
    """WebSocket for real-time status updates"""
    await websocket.accept()
    
    try:
        while True:
            await websocket.send_json(processing_status)
            await asyncio.sleep(5)  # Send update every 5 seconds
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Attribution Model NLP Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process": "/process",
            "status": "/status",
            "insights": "/insights/latest"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
