import os
import json
import threading
import time
import uuid
import random
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class TrainStatus:
    job_id: str
    status: str = "idle"  # idle | running | completed | error | stopped
    message: str = ""
    phase: str = ""       # "download"|"prepare"|"build"|"train"|"evaluate"|"save"
    epoch: int = 0
    total_epochs: int = 0
    progress: int = 0
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0

class TrainingManager:
    def __init__(self, models_dir="models", seed=123):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}
        self._statuses: Dict[str, TrainStatus] = {}
        self._stop_flags: Dict[str, bool] = {}
        self.seed = seed
        self.active_model_path: Optional[str] = None
        self.active_classes_path: Optional[str] = None

    def get_status(self, job_id: str) -> Optional[TrainStatus]:
        with self._lock:
            return self._statuses.get(job_id)

    def list_jobs(self):
        with self._lock:
            return list(self._statuses.keys())

    def stop(self, job_id: str):
        with self._lock:
            if job_id in self._statuses and self._statuses[job_id].status == "running":
                self._stop_flags[job_id] = True
                self._statuses[job_id].message = "Stop requested; finishing current batch"
                return True
        return False
        
    def get_active_artifacts(self):
        """Returns the last successfully trained/activated artifacts."""
        with self._lock:
            return self.active_model_path, self.active_classes_path

    def start_training(self, config: Dict[str, Any]) -> str:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        status = TrainStatus(job_id=job_id, status="running", started_at=time.time(), message="Starting", phase="prepare")
        with self._lock:
            self._statuses[job_id] = status
            self._stop_flags[job_id] = False
        th = threading.Thread(target=self._train_loop, args=(job_id, config), daemon=True)
        self._threads[job_id] = th
        th.start()
        return job_id

    def _train_loop(self, job_id: str, config: Dict[str, Any]):
        st = self._statuses[job_id]
        try:
            # Determinism
            os.environ["TF_DETERMINISTIC_OPS"] = "1"; os.environ["TF_CUDNN_DETERMINISTIC"] = "1"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            import random as pyrand
            pyrand.seed(self.seed); np.random.seed(self.seed); tf.random.set_seed(self.seed)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)

            import pandas as pd
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from math import ceil

            # Read config
            ds_conf = config.get("dataset", {}); epochs = int(config.get("epochs", 10)); batch_size = int(config.get("batchSize", 16)); lr = float(config.get("lr", 1e-3)); augment = bool(config.get("augment", True)); model_name = config.get("modelName") or f"bt_{time.strftime('%Y%m%d_%H%M%S')}"; backbone = config.get("backbone", "B4"); max_steps_train = int(config.get("maxStepsTrain", 0)); max_steps_valid = int(config.get("maxStepsValid", 0)); dry_run = bool(config.get("dryRun", False))

            # 1) Dataset
            with self._lock: st.phase = "prepare"; st.message = "Preparing dataset"; st.total_epochs = epochs
            ds_type = ds_conf.get("type", "local"); root = ds_conf.get("localPath")
            if ds_type == "kaggle":
                with self._lock: st.phase = "download"; st.message = "Downloading Kaggle dataset"
                import kagglehub
                kaggle_id = ds_conf.get("kaggleId", "masoudnickparvar/brain-tumor-mri-dataset")
                root = kagglehub.dataset_download(kaggle_id)

            if not root or not os.path.isdir(root): raise ValueError(f"Dataset path not found: {root}")
            train_dir = os.path.join(root, "Training"); test_dir = os.path.join(root, "Testing")
            if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)): raise ValueError("Dataset must have 'Training' and 'Testing' subfolders")

            # 2) Dataframes
            with self._lock: st.phase = "prepare"; st.message = "Indexing images"
            def build_df(base_dir):
                paths, labels = [], []
                for cls in sorted(os.listdir(base_dir)):
                    cls_path = os.path.join(base_dir, cls)
                    if not os.path.isdir(cls_path): continue
                    for img in os.listdir(cls_path): paths.append(os.path.join(cls_path, img)); labels.append(cls)
                return pd.DataFrame({"imagepaths": paths, "labels": labels})

            train_df = build_df(train_dir); test_all_df = build_df(test_dir)
            idx = np.arange(len(test_all_df)); rng = np.random.default_rng(self.seed); rng.shuffle(idx); mid = len(idx) // 2
            valid_df = test_all_df.iloc[idx[:mid]].reset_index(drop=True)
            test_df = test_all_df.iloc[idx[mid:]].reset_index(drop=True)

            # 3) Generators
            img_size = (224, 224)
            tr_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode="nearest") if augment else ImageDataGenerator()
            ts_gen = ImageDataGenerator()
            
            train_flow = tr_gen.flow_from_dataframe(train_df, x_col="imagepaths", y_col="labels", target_size=img_size, batch_size=batch_size, class_mode="categorical", color_mode="rgb", shuffle=True, seed=self.seed)
            valid_flow = ts_gen.flow_from_dataframe(valid_df, x_col="imagepaths", y_col="labels", target_size=img_size, batch_size=batch_size, class_mode="categorical", color_mode="rgb", shuffle=False)
            test_flow = ts_gen.flow_from_dataframe(test_df, x_col="imagepaths", y_col="labels", target_size=img_size, batch_size=batch_size, class_mode="categorical", color_mode="rgb", shuffle=False)
            
            class_indices = train_flow.class_indices; classes = [None] * len(class_indices)
            for label, idx in class_indices.items(): classes[idx] = label

            steps_train = max(1, ceil(train_flow.n / batch_size)); steps_valid = max(1, ceil(valid_flow.n / batch_size))
            if max_steps_train > 0: steps_train = min(steps_train, max_steps_train)
            if max_steps_valid > 0: steps_valid = min(steps_valid, max_steps_valid)
            if dry_run: steps_train, steps_valid, epochs = 5, 2, 1; st.total_epochs = 1

            # 4) Model
            with self._lock: st.phase = "build"; st.message = f"Building model ({backbone})"
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224,224,3), pooling="max") if backbone.upper() == "B0" else tf.keras.applications.EfficientNetB4(include_top=False, weights=None, input_shape=(224,224,3), pooling="max")
            model = Sequential([base, Dense(256, activation="relu"), Dropout(rate=0.45, seed=self.seed), Dense(len(classes), activation="softmax")])
            model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])

            # 5) Callbacks
            class Stopper(tf.keras.callbacks.Callback):
                def __init__(self, manager): super().__init__(); self.manager = manager
                def on_epoch_begin(self, epoch, logs=None):
                    with self.manager._lock: s = self.manager._statuses[job_id]; s.phase = "train"; s.message = f"Training epoch {epoch+1}/{epochs}"
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    with self.manager._lock:
                        s = self.manager._statuses[job_id]; s.epoch = epoch + 1; s.progress = int((s.epoch / s.total_epochs) * 100)
                        s.metrics_history.append({"epoch": s.epoch, "loss": float(logs.get("loss", 0)), "accuracy": float(logs.get("accuracy", 0)), "val_loss": float(logs.get("val_loss", 0)), "val_accuracy": float(logs.get("val_accuracy", 0))})
                    if self.manager._stop_flags.get(job_id): self.model.stop_training = True
            
            # 6) Train
            history = model.fit(train_flow, epochs=epochs, steps_per_epoch=steps_train, validation_data=valid_flow, validation_steps=steps_valid, shuffle=False, callbacks=[Stopper(self)], verbose=0)
            
            # 7) Evaluate
            with self._lock: st.phase = "evaluate"; st.message = "Evaluating"
            train_score = model.evaluate(train_flow, steps=min(steps_train, 10), verbose=0)
            valid_score = model.evaluate(valid_flow, steps=min(steps_valid, 10), verbose=0)
            test_score = model.evaluate(test_flow, steps=min(max(1, ceil(test_flow.n / batch_size)), 10), verbose=0)

            # 8) Save artifacts
            with self._lock: st.phase = "save"; st.message = "Saving model"
            out_dir = os.path.join(self.models_dir, model_name); os.makedirs(out_dir, exist_ok=True)
            model_path = os.path.join(out_dir, "model.keras"); classes_path = os.path.join(out_dir, "classes.json"); history_path = os.path.join(out_dir, "history.json"); summary_path = os.path.join(out_dir, "summary.json")
            model.save(model_path)
            with open(classes_path, "w") as f: json.dump(classes, f)
            with open(history_path, "w") as f: json.dump(history.history, f)
            summary = {"train_loss": train_score[0], "train_accuracy": train_score[1], "valid_loss": valid_score[0], "valid_accuracy": valid_score[1], "test_loss": test_score[0], "test_accuracy": test_score[1], "epochs": epochs, "batch_size": batch_size, "lr": lr, "backbone": backbone, "classes": classes}
            with open(summary_path, "w") as f: json.dump(summary, f, indent=2)

            with self._lock:
                st.status = "completed"; st.progress = 100; st.ended_at = time.time()
                st.artifacts = {"model": model_path, "classes": classes_path, "history": history_path, "summary": summary_path}
                self.active_model_path = model_path; self.active_classes_path = classes_path
        except Exception as e:
            import traceback; traceback.print_exc()
            with self._lock: st.status = "error"; st.message = str(e); st.ended_at = time.time()