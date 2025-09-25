// frontend/src/components/InferencePanel.jsx
import { useEffect, useState } from "react";
import { api } from "../services/api";

export default function InferencePanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [modelReady, setModelReady] = useState(false);

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
    try {
      if (!modelReady) {
        setError("Backend model not loaded. Put your model files in backend/models and restart the server, or POST /api/models/reload.");
        setRunning(false);
        return;
      }
      // Quick first
      const quick = await api.infer(file, "quick");
      setResult(quick);
      // Upgrade to full (non-blocking)
      api.infer(file, "full").then(setResult).catch(() => {});
    } catch (err) {
      setError(err.message || "Failed to analyze image.");
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="space-y-8">
      <section className="bg-white border rounded-xl p-6">
        <h2 className="font-semibold text-lg mb-4">Upload Brain MRI</h2>

        {!modelReady && (
          <div className="mb-4 p-3 rounded-lg bg-yellow-50 border border-yellow-200 text-sm text-yellow-700">
            Model not loaded. Place model in backend/models (brain_tumor_model.h5 or model_architecture.json + model.weights.h5), restart backend, or POST /api/models/reload.
          </div>
        )}

        <input
          type="file"
          accept="image/*"
          onChange={onFileChange}
          className="block w-full text-sm text-slate-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
        />

        {preview && (
          <div className="mt-4">
            <img src={preview} alt="preview" className="w-full max-w-md rounded-lg border" />
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
            {error}
          </div>
        )}

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={run}
            disabled={!file || running}
            className="px-5 py-2.5 rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
          >
            {running ? "Analyzing..." : "Analyze MRI"}
          </button>
          <button onClick={reset} className="px-5 py-2.5 rounded-lg border">Clear</button>
        </div>
      </section>

      {result && (
        <section className="bg-white border rounded-xl p-6">
          <h2 className="font-semibold text-lg mb-4">AI Analysis Results</h2>

          <div className="mb-4 p-4 rounded-lg bg-slate-50">
            <div className="flex items-center justify-between">
              <div className="text-xl font-semibold">{result?.prediction?.label || "N/A"}</div>
              <div className="text-lg">
                {result?.prediction?.probability != null
                  ? `${(result.prediction.probability * 100).toFixed(1)}%`
                  : "—"}
              </div>
            </div>
            <div className="text-sm text-slate-600 mt-1">
              {result?.summary?.validation ? `Validation: ${result.summary.validation}` : ""}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium mb-2">Original</h3>
              {result?.visualizations?.original ? (
                <img src={result.visualizations.original} alt="original" className="w-full rounded-lg border" />
              ) : (
                <div className="h-56 rounded-lg bg-slate-100 grid place-items-center text-slate-500">Not available</div>
              )}
            </div>

            <div>
              <h3 className="font-medium mb-2">AI Focus Areas</h3>
              {result?.visualizations?.sensitivity_overlay ? (
                <img src={result.visualizations.sensitivity_overlay} alt="focus" className="w-full rounded-lg border" />
              ) : (
                <div className="h-56 rounded-lg bg-slate-100 grid place-items-center text-slate-500">Not available</div>
              )}
            </div>

            <div>
              <h3 className="font-medium mb-2">Analyzed Regions</h3>
              {result?.visualizations?.regions_marked ? (
                <img src={result.visualizations.regions_marked} alt="regions" className="w-full rounded-lg border" />
              ) : (
                <div className="h-56 rounded-lg bg-slate-100 grid place-items-center text-slate-500">Not available</div>
              )}
            </div>

            <div>
              <h3 className="font-medium mb-2">Probability Distribution</h3>
              <div className="space-y-2">
                {result?.all_predictions ? (
                  Object.entries(result.all_predictions).map(([label, p]) => (
                    <div key={label}>
                      <div className="flex items-center justify-between text-sm">
                        <span>{label}</span>
                        <span>{(p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 rounded bg-slate-200 overflow-hidden">
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

          <div className="mt-6 grid md:grid-cols-2 gap-6">
            <div className="p-4 rounded-lg bg-blue-50 border">
              <h4 className="font-medium mb-2">Pattern Analysis</h4>
              <p className="text-sm">
                {result?.summary?.pattern_analysis || result?.interpretation?.pattern_description || "—"}
              </p>
            </div>
            <div className="p-4 rounded-lg bg-purple-50 border">
              <h4 className="font-medium mb-2">Spatial Distribution</h4>
              <p className="text-sm">
                {result?.summary?.location_analysis || result?.interpretation?.location_analysis || "—"}
              </p>
            </div>
          </div>

          {result?.regional_analysis?.length > 0 && (
            <div className="mt-6 overflow-x-auto">
              <h3 className="font-medium mb-3">Detailed Regional Analysis</h3>
              <table className="w-full text-sm border">
                <thead>
                  <tr className="bg-slate-50">
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
                  {result.regional_analysis.map((r) => (
                    <tr key={r.region_number} className="border-t">
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

          {result?.interpretation?.medical_interpretation && (
            <div className="mt-6 p-4 rounded-lg bg-amber-50 border">
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