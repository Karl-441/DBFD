import os
import shutil
import logging
import torch
from ultralytics import YOLO
import optuna
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_model_path = config['training']['base_model_path']
        self.output_dir = config['training']['output_model_dir']
        self.improvement_threshold = config['training']['map_improvement_threshold']
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config['training']['tensorboard_dir'], exist_ok=True)
        self.writer = SummaryWriter(config['training']['tensorboard_dir'])

    def update_config(self, config):
        self.config = config
        self.base_model_path = config['training']['base_model_path']
        self.output_dir = config['training']['output_model_dir']
        self.improvement_threshold = config['training']['map_improvement_threshold']
        os.makedirs(self.output_dir, exist_ok=True)

    def run_training(self):
        """Main training loop with validation and possible rollback."""
        try:
            # 1. Prepare dataset (assume it's already exported or we trigger it)
            # For YOLO, we need a data.yaml
            data_yaml_path = self._prepare_yolo_data_yaml()
            
            # 2. Evaluate base model for baseline mAP
            self.logger.info(f"Evaluating base model: {self.base_model_path}")
            base_model = YOLO(self.base_model_path)
            results_base = base_model.val(data=data_yaml_path)
            base_map = results_base.box.map50 # mAP@0.5
            self.logger.info(f"Base model mAP@0.5: {base_map:.4f}")
            
            # 3. Fine-tune (with auto-tuning if enabled)
            if self.config['training']['auto_tune']:
                best_params = self._auto_tune(data_yaml_path)
                self.logger.info(f"Best parameters found: {best_params}")
            else:
                best_params = {
                    'epochs': self.config['training']['epochs'],
                    'batch': self.config['training']['batch_size'],
                    'lr0': self.config['training']['learning_rate']
                }
            
            # 4. Final training
            model = YOLO(self.base_model_path)
            results = model.train(
                data=data_yaml_path,
                epochs=best_params['epochs'],
                batch=best_params['batch'],
                lr0=best_params['lr0'],
                patience=self.config['training']['early_stopping_patience'],
                project=self.output_dir,
                name="new_run",
                save=True
            )
            
            # 5. Evaluate new model
            new_model_path = os.path.join(self.output_dir, "new_run", "weights", "best.pt")
            new_model = YOLO(new_model_path)
            results_new = new_model.val(data=data_yaml_path)
            new_map = results_new.box.map50
            self.logger.info(f"New model mAP@0.5: {new_map:.4f}")
            
            # 6. Check improvement (≥ 2%)
            improvement = (new_map - base_map) / (base_map + 1e-6)
            if improvement >= self.improvement_threshold:
                # Success: Overwrite old model (ensure extension matches)
                ext = os.path.splitext(self.base_model_path)[1]
                target_path = os.path.splitext(self.base_model_path)[0] + "_new" + ext
                shutil.copy2(new_model_path, target_path)
                self.logger.info(f"Improvement met ({improvement:.2%}). New model saved to {target_path}")
                return True, f"mAP@0.5 improved from {base_map:.4f} to {new_map:.4f} ({improvement:.2%})."
            else:
                # Rollback/Fail
                self.logger.warning(f"Improvement ({improvement:.2%}) below threshold ({self.improvement_threshold:.2%}). Rolling back.")
                return False, f"mAP@0.5 improvement ({improvement:.2%}) insufficient. Model not updated."
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False, str(e)

    def _prepare_yolo_data_yaml(self):
        """Create data.yaml for YOLO training."""
        # Simple implementation: assume exported data is in the configured export dir
        export_dir = self.config['export']['output_dir']
        # Find the latest export folder
        exports = sorted([d for d in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, d))])
        if not exports:
            raise ValueError("No exported datasets found for training.")
        
        latest_export = os.path.join(export_dir, exports[-1])
        yaml_content = f"""
train: {os.path.abspath(os.path.join(latest_export, 'images'))}
val: {os.path.abspath(os.path.join(latest_export, 'images'))} # Simplified for now

nc: 2
names: ['fire', 'non_fire']
"""
        yaml_path = os.path.join(latest_export, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        return yaml_path

    def _auto_tune(self, data_yaml_path):
        """Use Optuna to find best hyperparameters."""
        def objective(trial):
            model = YOLO(self.base_model_path)
            lr = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
            batch = trial.suggest_categorical("batch", [8, 16, 32])
            
            results = model.train(
                data=data_yaml_path,
                epochs=5, # Short trials
                batch=batch,
                lr0=lr,
                patience=3,
                project=self.output_dir,
                name=f"trial_{trial.number}",
                save=False
            )
            return results.box.map50

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        
        best = study.best_params
        best['epochs'] = self.config['training']['epochs'] # Use full epochs for final
        return best
