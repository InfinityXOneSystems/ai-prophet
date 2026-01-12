#!/bin/bash
#
# AI PROPHET - GOOGLE CLOUD RUN DEPLOYMENT
# =========================================
# Deploys AI Prophet to Google Cloud Run for 24/7 autonomous operation
#
# Usage: bash deploy_cloud_run.sh
#

set -e

PROJECT_ID="infinity-x-one-systems"
SERVICE_NAME="ai-prophet-autonomous"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "============================================"
echo "AI PROPHET - CLOUD RUN DEPLOYMENT"
echo "============================================"
echo ""
echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"
echo "Image: $IMAGE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Set project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    firestore.googleapis.com

# Build container image
echo ""
echo "Building container image..."
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 1 \
    --min-instances 1 \
    --no-allow-unauthenticated \
    --set-env-vars "TZ=America/New_York" \
    --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,GCP_SA_KEY=GCP_SA_KEY:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "============================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "============================================"
echo ""
echo "Service URL: $SERVICE_URL"
echo "Region: $REGION"
echo "Min Instances: 1 (always running)"
echo ""
echo "View logs:"
echo "  gcloud run logs read $SERVICE_NAME --region $REGION --limit 50"
echo ""
echo "View service:"
echo "  gcloud run services describe $SERVICE_NAME --region $REGION"
echo ""
echo "Update service:"
echo "  bash deploy_cloud_run.sh"
echo ""
echo "Delete service:"
echo "  gcloud run services delete $SERVICE_NAME --region $REGION"
echo ""
echo "============================================"
