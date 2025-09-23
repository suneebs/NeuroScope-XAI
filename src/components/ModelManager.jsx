import { useEffect, useRef, useState } from "react";
import { useModel } from "../context/ModelContext";
import { api } from "../services/api";

export default function ModelManager() {
  const { state, dispatch } = useModel();
  const fileRef = useRef(null);
  const [jobId, setJobId] = useState(null);
  const [training, setTraining] = useState({ progress: 0 });

  useEffect(() => {
    (async () => {
      const res = await api.listModels();
      dispatch({
        type: "SET_MODELS",
        payload: res.models.map((m) => ({
          id: m.id,
          name: m.name,
          version: m.version,
          sizeMB: m.sizeMB,
          source: m.source,
        })),
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

  const [config, setConfig] = useState({ epochs: 10, batchSize: 16, lr: 0.0001, augment: true });

  const startTraining = async () => {
    dispatch({ type: "STATUS", payload: "training" });
    const res = await api.startTraining(config);
    setJobId(res.jobId);

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
    <div className="grid md:grid-cols-2 gap-6">
      <section className="card p-5">
        <h2 className="font-semibold text-lg mb-4">Model Catalog</h2>
        <div className="space-y-3 max-h-[320px] overflow-auto pr-2">
          {state.availableModels.map((m) => (
            <div
              key={m.id}
              className={[
                "p-4 rounded-md border cursor-pointer transition",
                state.selectedModel?.id === m.id
                  ? "bg-indigo-50 dark:bg-indigo-950/20 border-brand"
                  : "bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 hover:bg-white/60 dark:hover:bg-gray-800",
              ].join(" ")}
              onClick={() => dispatch({ type: "SELECT_MODEL", payload: m })}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{m.name}</div>
                  <div className="text-xs text-gray-500">
                    {m.version ? `v${m.version}` : "custom"} • {m.source}
                    {m.sizeMB ? ` • ${m.sizeMB}MB` : ""}
                  </div>
                </div>
                <button
                  className="text-sm px-3 py-1.5 rounded-md bg-brand text-white hover:bg-brand-dark"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownload(m.id);
                  }}
                >
                  Download
                </button>
              </div>
            </div>
          ))}
          {!state.availableModels.length && (
            <div className="text-sm text-gray-500">No models found. Download or upload one.</div>
          )}
        </div>

        {state.status === "downloading" && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm mb-2">
              <span>Downloading...</span>
              <span>{state.progress}%</span>
            </div>
            <div className="w-full h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden">
              <div
                className="h-2 bg-brand transition-[width]"
                style={{ width: `${state.progress}%` }}
              />
            </div>
          </div>
        )}

        <div className="mt-4">
          <input
            type="file"
            accept=".onnx,.pt,.pth,.pb,.h5,.json,.bin"
            ref={fileRef}
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            className="hidden"
          />
          <button
            className="px-4 py-2 rounded-md border border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
            onClick={() => fileRef.current?.click()}
          >
            Upload custom model
          </button>
        </div>
      </section>

      <section className="card p-5">
        <h2 className="font-semibold text-lg mb-4">Load & Train</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Selected model</label>
            <div className="p-3 rounded border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
              {state.selectedModel ? state.selectedModel.name : "None"}
            </div>
            <button
              disabled={!state.selectedModel}
              className="px-4 py-2 rounded-md bg-brand text-white disabled:opacity-50 hover:bg-brand-dark"
              onClick={handleLoad}
            >
              Load Model
            </button>
            <div className="text-sm text-gray-500">
              Status: {state.isModelReady ? "Ready" : "Not loaded"}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Training config</label>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="text-xs mb-1">Epochs</div>
                <input
                  type="number"
                  min={1}
                  className="w-full px-3 py-2 rounded border bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value || "1") })}
                />
              </div>
              <div>
                <div className="text-xs mb-1">Batch size</div>
                <input
                  type="number"
                  min={1}
                  className="w-full px-3 py-2 rounded border bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700"
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value || "1") })}
                />
              </div>
              <div>
                <div className="text-xs mb-1">Learning rate</div>
                <input
                  type="number"
                  step="0.00001"
                  min={0}
                  className="w-full px-3 py-2 rounded border bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700"
                  value={config.lr}
                  onChange={(e) => setConfig({ ...config, lr: parseFloat(e.target.value || "0.0001") })}
                />
              </div>
              <label className="flex items-center gap-2 text-sm mt-2">
                <input
                  type="checkbox"
                  checked={config.augment}
                  onChange={(e) => setConfig({ ...config, augment: e.target.checked })}
                />
                Data augmentation
              </label>
            </div>
            <button
              onClick={startTraining}
              className="px-4 py-2 rounded-md border border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              Start Training
            </button>
          </div>
        </div>

        {state.status === "training" && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm mb-2">
              <span>Training... (epoch {training.epoch ?? 0}/{training.totalEpochs ?? 0})</span>
              <span>{training.progress}%</span>
            </div>
            <div className="w-full h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden">
              <div className="h-2 bg-brand transition-[width]" style={{ width: `${training.progress}%` }} />
            </div>
            <div className="text-xs text-gray-500 mt-2">
              loss: {training.loss?.toFixed?.(3)} • val: {training.valLoss?.toFixed?.(3)}
            </div>
          </div>
        )}

        {state.error && (
          <div className="mt-4 text-sm text-red-600 dark:text-red-400">
            Error: {state.error}
          </div>
        )}
      </section>
    </div>
  );
}