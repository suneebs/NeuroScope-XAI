  import { useEffect, useRef, useState } from "react";
  import { motion } from "framer-motion";
  import { 
    Download, Upload, Cpu, PlayCircle, PauseCircle, 
    CheckCircle2, AlertCircle, Loader2, Sparkles, 
    Database, Zap, TrendingUp 
  } from "lucide-react";
  import { useModel } from "../context/ModelContext";
  import { api } from "../services/api";

  export default function ModelManager() {
    const { state, dispatch } = useModel();
    const fileRef = useRef(null);
    const [training, setTraining] = useState({ progress: 0 });
    const [config, setConfig] = useState({ epochs: 10, batchSize: 16, lr: 0.0001, augment: true });

    useEffect(() => {
      (async () => {
        const res = await api.listModels();
        dispatch({
          type: "SET_MODELS",
          payload: res.models,
        });
      })();
    }, [dispatch]);

    const handleDownload = async (id) => {
      dispatch({ type: "STATUS", payload: "downloading" });
      dispatch({ type: "PROGRESS", payload: 0 });
      try {
        await api.downloadModel(id, (p) => dispatch({ type: "PROGRESS", payload: p }));
        dispatch({ type: "STATUS", payload: "idle" });
      } catch (e) {
        dispatch({ type: "STATUS", payload: "error" });
        dispatch({ type: "ERROR", payload: e.message });
      }
    };

    const handleUpload = async (file) => {
      dispatch({ type: "STATUS", payload: "downloading" });
      try {
        const up = await api.uploadModel(file, (p) => dispatch({ type: "PROGRESS", payload: p }));
        dispatch({
          type: "SET_MODELS",
          payload: [{ id: up.id, name: up.name, source: "uploaded" }, ...state.availableModels],
        });
        dispatch({ type: "STATUS", payload: "idle" });
      } catch (e) {
        dispatch({ type: "STATUS", payload: "error" });
        dispatch({ type: "ERROR", payload: e.message });
      }
    };

    const handleLoad = async () => {
      if (!state.selectedModel) return;
      try {
        await api.loadModel(state.selectedModel.id);
        dispatch({ type: "READY", payload: true });
      } catch (e) {
        dispatch({ type: "ERROR", payload: e.message });
      }
    };

    const startTraining = async () => {
      dispatch({ type: "STATUS", payload: "training" });
      const res = await api.startTraining(config);
      
      const interval = setInterval(async () => {
        const s = await api.trainingStatus(res.jobId);
        setTraining({
          progress: s.progress,
          epoch: s.epoch,
          totalEpochs: s.totalEpochs,
          loss: s.metrics.loss,
          valLoss: s.metrics.valLoss,
        });
        dispatch({ type: "PROGRESS", payload: s.progress });
        if (s.status === "completed") {
          clearInterval(interval);
          dispatch({ type: "STATUS", payload: "idle" });
          dispatch({ type: "READY", payload: true });
        }
      }, 1000);
    };

    return (
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Model Catalog */}
        <motion.section 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card-pro p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Database className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-xl font-semibold">Model Catalog</h2>
            </div>
            <button
              className="btn-secondary flex items-center gap-2"
              onClick={() => fileRef.current?.click()}
            >
              <Upload className="w-4 h-4" />
              Upload
            </button>
          </div>

          <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
            {state.availableModels.map((model, idx) => (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={`
                  p-4 rounded-xl border transition-all cursor-pointer
                  ${state.selectedModel?.id === model.id
                    ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950/20"
                    : "border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700"
                  }
                `}
                onClick={() => dispatch({ type: "SELECT_MODEL", payload: model })}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium">{model.name}</h3>
                      {model.version && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                          v{model.version}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 mt-2 text-sm text-slate-600 dark:text-slate-400">
                      <span className="flex items-center gap-1">
                        <Cpu className="w-3 h-3" />
                        {model.sizeMB}MB
                      </span>
                      <span className="flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        {model.source}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(model.id);
                    }}
                    className="btn-primary py-2 px-4 text-sm"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>
            ))}

            {!state.availableModels.length && (
              <div className="text-center py-12">
                <Database className="w-12 h-12 mx-auto text-slate-300 dark:text-slate-700 mb-3" />
                <p className="text-slate-600 dark:text-slate-400">No models available</p>
                <p className="text-sm text-slate-500 dark:text-slate-500 mt-1">
                  Upload a model to get started
                </p>
              </div>
            )}
          </div>

          {/* Progress bar */}
          {state.status === "downloading" && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 rounded-xl bg-indigo-50 dark:bg-indigo-950/20"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-indigo-700 dark:text-indigo-400">
                  Downloading model...
                </span>
                <span className="text-sm font-medium text-indigo-700 dark:text-indigo-400">
                  {state.progress}%
                </span>
              </div>
              <div className="w-full h-2 rounded-full bg-indigo-200 dark:bg-indigo-900/50 overflow-hidden">
                <motion.div
                  className="h-full gradient-brand"
                  initial={{ width: 0 }}
                  animate={{ width: `${state.progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </motion.div>
          )}

          <input
            type="file"
            accept=".onnx,.pt,.pth,.pb,.h5,.json,.bin"
            ref={fileRef}
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            className="hidden"
          />
        </motion.section>

        {/* Training Control */}
        <motion.section 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card-pro p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-semibold">Model Training</h2>
          </div>

          {/* Model selection status */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 mb-6">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Selected Model
              </span>
              {state.isModelReady && (
                <span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                  <CheckCircle2 className="w-3 h-3" />
                  Ready
                </span>
              )}
            </div>
            <div className="font-medium">
              {state.selectedModel ? state.selectedModel.name : "No model selected"}
            </div>
            {state.selectedModel && !state.isModelReady && (
              <button onClick={handleLoad} className="btn-primary mt-3 w-full">
                Load Model
              </button>
            )}
          </div>

          {/* Training Configuration */}
          <div className="space-y-4 mb-6">
            <h3 className="font-medium text-sm text-slate-700 dark:text-slate-300">
              Training Configuration
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-1 block">
                  Epochs
                </label>
                <input
                  type="number"
                  min={1}
                  className="input-field"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value || "1") })}
                />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-1 block">
                  Batch Size
                </label>
                <input
                  type="number"
                  min={1}
                  className="input-field"
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value || "1") })}
                />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-1 block">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.00001"
                  min={0}
                  className="input-field"
                  value={config.lr}
                  onChange={(e) => setConfig({ ...config, lr: parseFloat(e.target.value || "0.0001") })}
                />
              </div>
              <div className="flex items-center">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.augment}
                    onChange={(e) => setConfig({ ...config, augment: e.target.checked })}
                    className="w-4 h-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                    Data Augmentation
                  </span>
                </label>
              </div>
            </div>
          </div>

          <button
            onClick={startTraining}
            disabled={!state.isModelReady || state.status === "training"}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {state.status === "training" ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <PlayCircle className="w-4 h-4" />
                Start Training
              </>
            )}
          </button>

          {/* Training progress */}
          {state.status === "training" && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 p-4 rounded-xl bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-medium">
                    Epoch {training.epoch || 0} / {training.totalEpochs || 0}
                  </span>
                </div>
                <span className="text-sm font-medium text-purple-600 dark:text-purple-400">
                  {training.progress}%
                </span>
              </div>
              <div className="w-full h-2 rounded-full bg-purple-200 dark:bg-purple-900/50 overflow-hidden mb-3">
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${training.progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600 dark:text-slate-400">Loss</span>
                  <span className="font-mono font-medium">
                    {training.loss?.toFixed(4) || "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-600 dark:text-slate-400">Val Loss</span>
                  <span className="font-mono font-medium">
                    {training.valLoss?.toFixed(4) || "—"}
                  </span>
                </div>
              </div>
            </motion.div>
          )}

          {state.error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 p-3 rounded-lg bg-red-50 dark:bg-red-950/20 flex items-start gap-2"
            >
              <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400 mt-0.5" />
              <p className="text-sm text-red-600 dark:text-red-400">{state.error}</p>
            </motion.div>
          )}
        </motion.section>
      </div>
    );
  }