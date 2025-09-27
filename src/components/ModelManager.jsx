import { useEffect, useRef, useState } from "react";
import { Loader2, Database, Rocket, CheckCircle2, Play, Square } from "lucide-react";
import { useTrainingStatus } from "../hooks/useTrainingStatus"; // Make sure you have this hook from the previous step

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function ModelManager() {
  const [dataset, setDataset] = useState({ type: "kaggle", kaggleId: "masoudnickparvar/brain-tumor-mri-dataset", localPath: "" });
  // Set safer defaults for CPU training
  const [config, setConfig] = useState({ 
    epochs: 10, 
    batchSize: 16, // Lowered batch size
    lr: 0.001, 
    augment: true, 
    modelName: "bt_efficientnet_b0_v1", 
    backbone: "B0", // Default to the lighter B0 model
    dryRun: false 
  });
  const [jobId, setJobId] = useState(null);
  const [startError, setStartError] = useState("");
  
  const { status, error: pollError, isPolling } = useTrainingStatus(jobId);

  const startTraining = async () => {
    setStartError("");
    setJobId(null); 
    try {
      const payload = { dataset, ...config };
      const res = await fetch(`${BASE}/api/train/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setJobId(data.jobId); 
    } catch (e) {
      setStartError(e.message || "Failed to start training");
    }
  };
  
  const stopTraining = async () => {
    if (!jobId) return;
    try {
      await fetch(`${BASE}/api/train/stop`, { 
        method: "POST", 
        headers: { "Content-Type": "application/json" }, 
        body: JSON.stringify({ jobId: jobId }) 
      });
    } catch (e) {
      console.error("Failed to send stop request", e);
    }
  };

  const setActiveModel = async () => {
    if (!status?.artifacts?.model) return;
    try {
      const res = await fetch(`${BASE}/api/models/use`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modelPath: status.artifacts.model,
          classesPath: status.artifacts.classes,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      await res.json();
      alert("Active model updated for inference! You can now go to the 'Analyze' tab.");
    } catch (e) {
      alert(e.message || "Failed to set active model");
    }
  };

  const isTrainingRunning = status?.status === 'running';

  return (
    <div className="space-y-8">
      {/* Config Section */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm p-6">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Dataset Column */}
          <div className="space-y-3">
            <h3 className="font-semibold flex items-center gap-2"><Database className="w-4 h-4" /> Dataset</h3>
            <select
              className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm"
              value={dataset.type}
              onChange={(e) => setDataset({ ...dataset, type: e.target.value })}
            >
              <option value="kaggle">Kaggle (Default)</option>
              <option value="local">Local Path</option>
            </select>
            {dataset.type === "kaggle" ? (
              <div>
                <label className="text-xs font-medium">Kaggle ID</label>
                <input className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" value={dataset.kaggleId} onChange={(e) => setDataset({ ...dataset, kaggleId: e.target.value })} />
                <p className="text-xs text-slate-500 mt-1">Ensure Kaggle API is configured on backend.</p>
              </div>
            ) : (
              <div>
                <label className="text-xs font-medium">Local Path</label>
                <input className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" value={dataset.localPath} onChange={(e) => setDataset({ ...dataset, localPath: e.target.value })} />
                <p className="text-xs text-slate-500 mt-1">Must contain 'Training' and 'Testing' subfolders.</p>
              </div>
            )}
          </div>

          {/* Training Column */}
          <div className="space-y-3">
            <h3 className="font-semibold flex items-center gap-2"><Rocket className="w-4 h-4" /> Training Parameters</h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium">Epochs</label>
                <input type="number" min={1} className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" value={config.epochs} onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value || "1") })} />
              </div>
              <div>
                <label className="text-xs font-medium">Batch Size</label>
                <input type="number" min={1} className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" value={config.batchSize} onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value || "1") })} />
              </div>
              <div>
                <label className="text-xs font-medium">Learning Rate</label>
                <input type="number" step="0.00001" min={0} className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" value={config.lr} onChange={(e) => setConfig({ ...config, lr: parseFloat(e.target.value || "0.001") })} />
              </div>
              <div className="flex items-center gap-2 pt-6">
                <input id="augment" type="checkbox" className="rounded" checked={config.augment} onChange={(e) => setConfig({ ...config, augment: e.target.checked })} />
                <label htmlFor="augment" className="text-sm font-medium">Augment</label>
              </div>
            </div>
            <div>
              <label className="text-xs font-medium">Model Backbone</label>
              <select 
                className="w-full mt-1 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" 
                value={config.backbone} 
                onChange={(e) => setConfig({ ...config, backbone: e.target.value, modelName: `bt_efficientnet_${e.target.value.toLowerCase()}_v1` })}
              >
                <option value="B0">EfficientNetB0 (Fast, good for CPU)</option>
                <option value="B4">EfficientNetB4 (Powerful, needs GPU or lots of RAM)</option>
              </select>
            </div>
          </div>

          {/* Action Column */}
          <div className="space-y-3">
            <h3 className="font-semibold">Action</h3>
            <div>
              <label className="text-xs font-medium">Save Model As</label>
              <input className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm" value={config.modelName} onChange={(e) => setConfig({ ...config, modelName: e.target.value })} />
            </div>
            <div className="flex items-center gap-2 pt-2">
              <input id="dryRun" type="checkbox" className="rounded" checked={config.dryRun} onChange={(e) => setConfig({ ...config, dryRun: e.target.checked })} />
              <label htmlFor="dryRun" className="text-sm font-medium">Quick Test Run (1 epoch, few steps)</label>
            </div>
            <button
              onClick={startTraining}
              disabled={isTrainingRunning}
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
            >
              <Play className="w-4 h-4" /> Start Training
            </button>
            {startError && (
              <p className="text-xs text-red-500">{startError}</p>
            )}
          </div>
        </div>
      </section>

      {/* Progress Section */}
      {jobId && (
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm p-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-lg">Training Progress (Job ID: {jobId.slice(4)})</h3>
            {isTrainingRunning && (
              <button onClick={stopTraining} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-white bg-red-600 hover:bg-red-700">
                <Square className="w-4 h-4" /> Stop
              </button>
            )}
          </div>
          
          {!status && isPolling && (
             <div className="flex items-center gap-2 text-sm text-slate-500">
                <Loader2 className="w-4 h-4 animate-spin" />
                Connecting to training job...
             </div>
          )}

          {status && (
            <>
              <div className="grid sm:grid-cols-2 gap-4">
                <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div>
                      Status:{" "}
                      <span className={`font-medium ${status.status === "completed" ? "text-emerald-600" : status.status === "error" ? "text-red-600" : "text-indigo-600"}`}>{status.status}</span>
                    </div>
                    <div>{status.epoch}/{status.totalEpochs} epochs</div>
                  </div>
                  <div className="w-full h-2 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                    <div className="h-2 bg-indigo-500 transition-[width]" style={{ width: `${status.progress || 0}%` }} />
                  </div>
                  <div className="text-xs text-slate-500">
                    Phase: {status.phase} - {status.message}
                  </div>
                </div>

                {status.metrics && Object.keys(status.metrics).length > 0 && (
                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Loss: <span className="font-mono">{status.metrics.loss?.toFixed(4)}</span></div>
                      <div>Val Loss: <span className="font-mono">{status.metrics.val_loss?.toFixed(4)}</span></div>
                      <div>Accuracy: <span className="font-mono">{status.metrics.accuracy?.toFixed(4)}</span></div>
                      <div>Val Accuracy: <span className="font-mono">{status.metrics.val_accuracy?.toFixed(4)}</span></div>
                    </div>
                  </div>
                )}
              </div>

              {status.status === "completed" && (
                <div className="mt-4 p-4 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900">
                  <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">Training Complete</h4>
                  <ul className="text-sm list-disc pl-5">
                    {Object.entries(status.artifacts).map(([k, v]) => (
                      <li key={k}><span className="font-medium">{k}:</span> <span className="text-slate-600">{v}</span></li>
                    ))}
                  </ul>
                  <button onClick={setActiveModel} className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-white bg-emerald-600 hover:bg-emerald-700">
                    <CheckCircle2 className="w-4 h-4" /> Use this model for inference
                  </button>
                </div>
              )}
              {(status.status === "error" || pollError) && (
                <div className="mt-4 p-4 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900">
                  <h4 className="font-medium text-red-800 dark:text-red-300">Error</h4>
                  <p className="text-sm text-red-700 dark:text-red-400">{status.message || pollError}</p>
                </div>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
}