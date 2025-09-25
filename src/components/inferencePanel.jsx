import { useEffect, useState } from "react";
import { api } from "../services/api";
import { AlertTriangle, Image as ImageIcon, Loader2, ShieldCheck, BarChart2 } from "lucide-react";

export default function InferencePanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [modelReady, setModelReady] = useState(false);
  const [phase, setPhase] = useState(""); // "quick" | "full" | ""

  useEffect(() => {
    (async () => {
      try {
        const s = await api.status();
        setModelReady(!!s.model_loaded);
      } catch {
        setModelReady(false);
      }
    })();
  }, []);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  };

  const run = async () => {
    if (!file) return;
    setRunning(true);
    setError("");
    setPhase("quick");
    try {
      if (!modelReady) {
        setError("Backend model not loaded. Place model files in backend/models and reload.");
        setRunning(false);
        setPhase("");
        return;
      }
      const quick = await api.infer(file, "quick");
      setResult(quick);
      setPhase("full");
      api.infer(file, "full")
        .then((full) => {
          setResult(full);
          setPhase("");
        })
        .catch(() => setPhase(""));
    } catch (err) {
      setError(err.message || "Failed to analyze image.");
      setPhase("");
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setPhase("");
  };

  return (
    <div className="space-y-8">
      {/* Upload */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300">
            <ImageIcon className="w-4 h-4" />
            <span className="text-sm">Upload MRI</span>
          </div>
        </div>

        {!modelReady && (
          <div className="mb-4 p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-sm">
            Backend model not loaded. Put your model files in backend/models and restart the server, or POST /api/models/reload.
          </div>
        )}

        <div className="grid lg:grid-cols-2 gap-6">
          <div>
            <div
              className="border-2 border-dashed rounded-xl p-6 text-center hover:border-indigo-500 transition-colors"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (f) onFileChange({ target: { files: [f] } });
              }}
            >
              <p className="text-sm text-slate-700 dark:text-slate-300">Drag & drop an image here</p>
              <p className="text-xs text-slate-500 mt-1">or use the file picker below</p>
            </div>

            <input
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="mt-3 block w-full text-sm text-slate-700
                        file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0
                        file:text-sm file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
            />

            {preview && (
              <div className="mt-4">
                <img src={preview} alt="preview" className="w-full max-w-md rounded-lg border border-slate-200 dark:border-slate-800" />
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            <div className="mt-4 flex items-center gap-3">
              <button
                onClick={run}
                disabled={!file || running}
                className="inline-flex items-center justify-center px-5 py-2.5 rounded-lg font-medium
                           text-white bg-indigo-600 hover:bg-indigo-700 transition disabled:opacity-50"
              >
                {running ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {phase === "full" ? "Running full analysis..." : "Analyzing..."}
                  </span>
                ) : (
                  "Analyze MRI"
                )}
              </button>
              <button
                onClick={reset}
                className="inline-flex items-center justify-center px-4 py-2 rounded-lg font-medium
                           border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Live status */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
            <h3 className="font-medium mb-2 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              Analysis Status
            </h3>
            <ul className="text-sm space-y-1">
              <li>
                • Backend:{" "}
                <span className={modelReady ? "text-emerald-600" : "text-amber-600"}>
                  {modelReady ? "Model Loaded" : "Not Loaded"}
                </span>
              </li>
              <li>• Stage: {phase ? (phase === "quick" ? "Quick (fast)" : "Full (comprehensive)") : "Idle"}</li>
              <li>• Time depends on CPU/GPU; full analysis may take longer</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Results */}
      {result && (
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm p-6">
          <h2 className="font-semibold text-lg mb-4">AI Analysis Results</h2>

          {/* Stats */}
          <div className="grid sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Diagnosis</div>
              <div className="text-xl font-semibold mt-1">{result?.prediction?.label || "—"}</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Confidence</div>
              <div className="text-xl font-semibold mt-1">
                {result?.prediction?.probability != null ? `${(result.prediction.probability * 100).toFixed(1)}%` : "—"}
              </div>
            </div>
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Validation</div>
              <div
                className={`inline-flex items-center px-2 py-0.5 rounded-md text-sm font-medium mt-1 ${
                  result?.summary?.validation === "PASSED"
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                    : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                }`}
              >
                {result?.summary?.validation || "—"}
              </div>
            </div>
          </div>

          {/* Visuals */}
          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <div>
              <h3 className="font-medium mb-2">Original</h3>
              {result?.visualizations?.original ? (
                <img src={result.visualizations.original} alt="original" className="w-full rounded-lg border border-slate-200 dark:border-slate-800" />
              ) : (
                <div className="h-56 rounded-lg bg-slate-200 dark:bg-slate-800 animate-pulse" />
              )}
            </div>
            <div>
              <h3 className="font-medium mb-2">AI Focus Areas</h3>
              {result?.visualizations?.sensitivity_overlay ? (
                <img src={result.visualizations.sensitivity_overlay} alt="sensitivity" className="w-full rounded-lg border border-slate-200 dark:border-slate-800" />
              ) : (
                <div className="h-56 rounded-lg bg-slate-200 dark:bg-slate-800 animate-pulse" />
              )}
            </div>
            <div>
              <h3 className="font-medium mb-2">Analyzed Regions</h3>
              {result?.visualizations?.regions_marked ? (
                <img src={result.visualizations.regions_marked} alt="regions" className="w-full rounded-lg border border-slate-200 dark:border-slate-800" />
              ) : (
                <div className="h-56 rounded-lg bg-slate-200 dark:bg-slate-800 animate-pulse" />
              )}
            </div>
            <div>
              <h3 className="font-medium mb-2 flex items-center gap-2">
                <BarChart2 className="w-4 h-4" /> Probability Distribution
              </h3>
              <div className="space-y-2">
                {result?.all_predictions ? (
                  Object.entries(result.all_predictions).map(([label, p]) => (
                    <div key={label}>
                      <div className="flex items-center justify-between text-sm">
                        <span>{label}</span>
                        <span>{(p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                        <div className="h-2 bg-indigo-500" style={{ width: `${p * 100}%` }} />
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-slate-500">No data</div>
                )}
              </div>
            </div>
          </div>

          {/* Findings */}
          <div className="mt-6 grid md:grid-cols-2 gap-6">
            <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900">
              <h4 className="font-medium mb-2">Pattern Analysis</h4>
              <p className="text-sm">
                {result?.summary?.pattern_analysis || result?.interpretation?.pattern_description || "—"}
              </p>
            </div>
            <div className="p-4 rounded-lg bg-purple-50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-900">
              <h4 className="font-medium mb-2">Spatial Distribution</h4>
              <p className="text-sm">
                {result?.summary?.location_analysis || result?.interpretation?.location_analysis || "—"}
              </p>
            </div>
          </div>

          {/* Region table */}
          {result?.regional_analysis?.length > 0 && (
            <div className="mt-6 overflow-x-auto">
              <h3 className="font-medium mb-3">Detailed Regional Analysis</h3>
              <table className="w-full text-sm border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden">
                <thead className="bg-slate-50 dark:bg-slate-800/50">
                  <tr>
                    <th className="text-left p-2">Region</th>
                    <th className="text-left p-2">Size (px)</th>
                    <th className="text-left p-2">Size %</th>
                    <th className="text-left p-2">Location</th>
                    <th className="text-left p-2">Importance</th>
                    <th className="text-left p-2">Eccentricity</th>
                    <th className="text-left p-2">Solidity</th>
                  </tr>
                </thead>
                <tbody>
                  {result.regional_analysis.map((r, idx) => (
                    <tr key={idx} className="border-t border-slate-200 dark:border-slate-800">
                      <td className="p-2">Region {r.region_number}</td>
                      <td className="p-2">{r.size_pixels}</td>
                      <td className="p-2">{r.size_percentage}</td>
                      <td className="p-2">{r.location}</td>
                      <td className="p-2">{r.importance}</td>
                      <td className="p-2">{r.eccentricity}</td>
                      <td className="p-2">{r.solidity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Medical */}
          {result?.interpretation?.medical_interpretation && (
            <div className="mt-6 p-4 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900">
              <h3 className="font-medium mb-2">Medical Interpretation</h3>
              <div className="text-sm font-medium mb-1">
                {result.interpretation.medical_interpretation.assessment}
              </div>
              {result.interpretation.medical_interpretation.observations?.length > 0 && (
                <ul className="list-disc pl-5 text-sm">
                  {result.interpretation.medical_interpretation.observations.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
              )}
              {result.interpretation.medical_interpretation.recommendations?.length > 0 && (
                <div className="mt-3">
                  <div className="font-medium mb-1">Recommendations</div>
                  <ul className="list-disc pl-5 text-sm">
                    {result.interpretation.medical_interpretation.recommendations.map((t, i) => (
                      <li key={i}>{t.replace(/[✓⚠]/g, "").trim()}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}