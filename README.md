# Attribution Model NLP Service
Automated support ticket analysis and behavioral insights for segmentation and attribution modeling.

## Features
- Daily automated NLP processing of support tickets
- Growth signal detection and intent classification
- Integration with dbt Cloud for model refreshes
- Real-time API endpoints for manual triggers
- Comprehensive monitoring and alerting

## API Endpoints
- `GET /health` - Health check
- `POST /process` - Manual processing trigger
- `GET /status` - Processing status
- `GET /insights/latest` - Latest insights

## Environment Variables
See `.env.example` for required configuration.

## Deployment
This service is deployed on DigitalOcean App Platform with automatic GitHub integration.
