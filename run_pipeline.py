import logging
from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Get and display the tracking URI from the active experiment tracker
        client = Client()
        tracker = client.active_stack.experiment_tracker
        
        # Call get_tracking_uri if it exists
        if hasattr(tracker, 'get_tracking_uri'):
            tracking_uri = tracker.get_tracking_uri()  # type: ignore
            logger.info(f"MLflow Tracking URI: {tracking_uri}")
            print(f"✓ Tracking URI: {tracking_uri}")
        else:
            logger.warning(f"Experiment tracker {type(tracker).__name__} does not support get_tracking_uri")
            
    except Exception as e:
        logger.error(f"Error retrieving tracking URI: {e}")
    
    # Run the training pipeline (ZenML will handle MLflow experiment tracking)
    logger.info("Starting training pipeline...")
    try:
        results = training_pipeline(data_path="data\\olist_customers_dataset.csv")
        logger.info("Training pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")