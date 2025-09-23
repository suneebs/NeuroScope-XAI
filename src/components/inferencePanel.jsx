import { useState } from "react";
import { useModel } from "../context/ModelContext";
import FileDropzone from "./FileDropzone";
import { api } from "../services/api";
import HeatmapOverlay from "./HeatmapOverlay";

export default function InferencePanel() {
  const { state } = useModel();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);

  const [opacity, setOpacity] = useState(0.5);
  const [threshold, setThreshold] = useState(0.2);

  const onDrop = (f) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const run = async () => {
    if (!file) return;
    setRunning(true);
    try {
      const res = await api.infer(file);
      setResult({
        label: res.prediction.label,
        prob: res.prediction.probability,
        heatmap: res.heatmap,
      });
    } catch (e) {
      // TODO: add toast/error UI
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      <section className="card p-5">
        <h2 className="font-semibold text-lg mb-4">Upload MRI</h2>
        <FileDropzone onFile={onDrop} accept="image/*" />
        {preview && (
          <div className="mt-4">
            <img src={preview} alt="preview" className="w-full rounded border border-gray-200 dark:border-gray-700" />
          </div>
        )}
        <button
          disabled={!file || !state.isModelReady || running}
          onClick={run}
          className="mt-4 px-4 py-2 rounded-md bg-brand text-white disabled:opacity-50 hover:bg-brand-dark"
        >
          {running ? "Running..." : state.isModelReady ? "Run Inference" : "Load a model first"}
        </button>
      </section>

      <section className="card p-5">
        <h2 className="font-semibold text-lg mb-4">Prediction & Explainability</h2>
        {!result ? (
          <div className="text-sm text-gray-500">Results will appear here.</div>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500">Prediction</div>
                <div className="text-xl font-semibold">
                  {result.label} <span className="text-sm text-gray-500">({(result.prob * 100).toFixed(1)}%)</span>
                </div>
              </div>
            </div>

            {preview && (
              <div className="mt-4">
                <HeatmapOverlay image={preview} heatmap={result.heatmap} opacity={opacity} threshold={threshold} />
                <div className="mt-3 grid grid-cols-2 gap-4 items-center">
                  <label className="text-sm">
                    Opacity
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={opacity}
                      onChange={(e) => setOpacity(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </label>
                  <label className="text-sm">
                    Threshold
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </label>
                </div>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}