"""
Custom YOLO Model Training Script
==================================

Train a custom YOLOv8 model on American Football datasets for improved
player and ball detection accuracy.

This script automates:
1. Dataset download from Roboflow
2. Model training with configurable parameters
3. Model evaluation and export
4. Integration with tracker.py

Usage:
    python train.py --dataset <roboflow_dataset> --epochs 100

Author: Computer Vision Engineer
Date: October 2025
"""

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml


class FootballModelTrainer:
    """
    Handles custom YOLO model training for football detection.
    """
    
    def __init__(self, base_model='yolov8n.pt'):
        """
        Initialize trainer.
        
        Args:
            base_model: Base YOLO model to start from
        """
        self.base_model = base_model
        self.model = None
        self.results = None
        
        # Create directories
        self.project_dir = Path('runs/train')
        self.models_dir = Path('custom_models')
        self.models_dir.mkdir(exist_ok=True)
    
    def download_dataset(self, roboflow_url=None, dataset_name='american-football'):
        """
        Download dataset from Roboflow.
        
        Args:
            roboflow_url: Full Roboflow dataset URL
            dataset_name: Dataset identifier
            
        Returns:
            Path to dataset yaml file
        """
        print("="*60)
        print("Downloading Dataset from Roboflow")
        print("="*60)
        
        if roboflow_url:
            print(f"\nUsing provided URL: {roboflow_url}")
            # Use roboflow Python API
            try:
                from roboflow import Roboflow
                
                # Extract workspace, project, version from URL
                # Format: https://app.roboflow.com/workspace/project/version
                parts = roboflow_url.split('/')
                workspace = parts[-3]
                project = parts[-2]
                version = int(parts[-1]) if parts[-1].isdigit() else 1
                
                # Initialize Roboflow
                rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY', ''))
                
                # Download dataset
                project_obj = rf.workspace(workspace).project(project)
                dataset = project_obj.version(version).download("yolov8")
                
                dataset_path = Path(dataset.location) / 'data.yaml'
                
            except ImportError:
                print("\nError: roboflow package not installed.")
                print("Install with: pip install roboflow")
                print("\nAlternatively, manually download the dataset:")
                print("1. Go to Roboflow")
                print("2. Export dataset in YOLOv8 format")
                print("3. Place in ./datasets/ folder")
                print("4. Update DATASET_PATH in this script")
                return None
        
        else:
            # Use pre-configured dataset
            print(f"\nLooking for dataset: {dataset_name}")
            dataset_path = Path(f'datasets/{dataset_name}/data.yaml')
            
            if not dataset_path.exists():
                print(f"\nDataset not found at: {dataset_path}")
                print("\nTo download dataset:")
                print("1. Visit: https://universe.roboflow.com/")
                print("2. Search for 'American Football' datasets")
                print("3. Recommended: 'American Football Players Detection'")
                print("4. Export in YOLOv8 format")
                print("5. Place in ./datasets/ folder")
                print("\nOr run with --roboflow_url parameter")
                return None
        
        print(f"\n✓ Dataset ready: {dataset_path}")
        return str(dataset_path)
    
    def train(self,
              data_yaml,
              epochs=100,
              batch_size=16,
              imgsz=640,
              device='cpu',
              patience=50,
              save_period=10,
              **kwargs):
        """
        Train YOLO model on custom dataset.
        
        Args:
            data_yaml: Path to dataset yaml file
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Image size for training
            device: 'cpu' or 'cuda' or specific GPU id
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("Starting Model Training")
        print("="*60)
        
        print(f"\nTraining Configuration:")
        print(f"  Base model: {self.base_model}")
        print(f"  Dataset: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {imgsz}")
        print(f"  Device: {device}")
        print(f"  Patience: {patience}")
        print()
        
        # Load model
        self.model = YOLO(self.base_model)
        
        # Train
        self.results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            patience=patience,
            save_period=save_period,
            project=str(self.project_dir),
            name='football_custom',
            exist_ok=True,
            **kwargs
        )
        
        print("\n✓ Training complete!")
        return self.results
    
    def evaluate(self):
        """
        Evaluate trained model on validation set.
        
        Returns:
            Validation metrics
        """
        if self.model is None:
            print("Error: No trained model available")
            return None
        
        print("\n" + "="*60)
        print("Evaluating Model")
        print("="*60)
        
        # Validate
        metrics = self.model.val()
        
        print("\nValidation Results:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, model_name='custom_football_yolov8.pt'):
        """
        Export trained model to custom_models directory.
        
        Args:
            model_name: Name for the exported model
            
        Returns:
            Path to exported model
        """
        print("\n" + "="*60)
        print("Exporting Model")
        print("="*60)
        
        # Find best weights
        weights_dir = self.project_dir / 'football_custom' / 'weights'
        best_weights = weights_dir / 'best.pt'
        
        if not best_weights.exists():
            print(f"Error: Best weights not found at {best_weights}")
            return None
        
        # Copy to custom_models directory
        destination = self.models_dir / model_name
        shutil.copy(best_weights, destination)
        
        print(f"\n✓ Model exported to: {destination}")
        print(f"\nTo use this model in tracker.py:")
        print(f"  1. Edit tracker_config.py")
        print(f"  2. Set: YOLO_MODEL_PATH = '{destination}'")
        print(f"  3. Run: python tracker.py")
        
        return str(destination)
    
    def create_dataset_yaml_template(self, output_path='dataset_template.yaml'):
        """
        Create a template dataset.yaml file.
        
        Args:
            output_path: Where to save the template
        """
        template = {
            'path': '../datasets/football',  # Dataset root
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'person',
                1: 'football'
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False)
        
        print(f"Dataset template created: {output_path}")
        print("Customize this file with your dataset paths and classes")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train custom YOLO model for football detection')
    
    # Dataset arguments
    parser.add_argument('--roboflow_url', type=str, default=None,
                       help='Roboflow dataset URL')
    parser.add_argument('--dataset', type=str, default='american-football',
                       help='Dataset name or path to data.yaml')
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='Base YOLO model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Training device (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--model_name', type=str, default='custom_football_yolov8.pt',
                       help='Output model name')
    
    # Additional options
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip dataset download (use existing)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (evaluate only)')
    parser.add_argument('--create_template', action='store_true',
                       help='Create dataset.yaml template and exit')
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        trainer = FootballModelTrainer()
        trainer.create_dataset_yaml_template()
        return
    
    print("="*60)
    print("Custom Football YOLO Training")
    print("="*60)
    
    # Initialize trainer
    trainer = FootballModelTrainer(base_model=args.base_model)
    
    # Download dataset
    if not args.skip_download:
        dataset_path = trainer.download_dataset(
            roboflow_url=args.roboflow_url,
            dataset_name=args.dataset
        )
        if dataset_path is None:
            print("\nError: Dataset not available")
            print("Run with --create_template to create a dataset config template")
            return
    else:
        # Use provided dataset path
        dataset_path = args.dataset if args.dataset.endswith('.yaml') else f'datasets/{args.dataset}/data.yaml'
        if not Path(dataset_path).exists():
            print(f"Error: Dataset not found: {dataset_path}")
            return
    
    # Train model
    if not args.skip_training:
        trainer.train(
            data_yaml=dataset_path,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience
        )
        
        # Evaluate
        trainer.evaluate()
    
    # Export model
    trainer.export_model(model_name=args.model_name)
    
    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review training results in runs/train/football_custom/")
    print("2. Update tracker_config.py to use the new model")
    print("3. Run tracker.py to test the custom model")


if __name__ == '__main__':
    main()

