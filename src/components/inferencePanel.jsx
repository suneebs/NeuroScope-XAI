import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Upload, Brain, Loader2, ChevronRight, 
  Info, Activity, Layers, Sliders,
  FileImage, CheckCircle, XCircle
} from "lucide-react";
import { useModel } from "../context/ModelContext";
import { api } from "../services/api";

export default function InferencePanel() {
  const { state } = useModel();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.7);

  const handleFileSelect = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
    }
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
        topK: res.topK || [],
      });
    } catch (e) {
      console.error(e);
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="grid lg:grid-cols-2 gap-8">
      {/* Upload Section */}
      <motion.section
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="card-pro p-6"
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center">
            <FileImage className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-xl font-semibold">MRI Upload</h2>
        </div>

        {!preview ? (
          <div
            className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-2xl p-12 text-center hover:border-indigo-500 dark:hover:border-indigo-500 transition-colors cursor-pointer"
            onClick={() => document.getElementById("file-input").click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <Upload className="w-12 h-12 mx-auto text-slate-400 mb-4" />
            <h3 className="font-medium text-lg mb-2">Drop your MRI scan here</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
              or click to browse files
            </p>
            <p className="text-xs text-slate-500">
              Supports: PNG, JPG, JPEG, DICOM • Max 10MB
            </p>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative rounded-2xl overflow-hidden bg-slate-100 dark:bg-slate-900">
              <img
                src={preview}
                alt="MRI scan"
                className="w-full h-auto"
              />
              {result?.heatmap && showHeatmap && (
                <motion.img
                  initial={{ opacity: 0 }}
                  animate={{ opacity }}
                  src={result.heatmap}
                  alt="Heatmap overlay"
                  className="absolute inset-0 w-full h-full object-contain mix-blend-multiply"
                />
              )}
            </div>

            <div className="flex items-center gap-3">
              <div className="flex-1 text-sm">
                <p className="font-medium">{file.name}</p>
                <p className="text-slate-600 dark:text-slate-400">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button onClick={reset} className="btn-secondary">
                Change
              </button>
            </div>

            {result && (
              <div className="space-y-3 p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
                <label className="flex items-center justify-between">
                  <span className="text-sm font-medium">Show Heatmap</span>
                  <input
                    type="checkbox"
                    checked={showHeatmap}
                    onChange={(e) => setShowHeatmap(e.target.checked)}
                    className="w-4 h-4"
                  />
                </label>
                {showHeatmap && (
                  <label className="block">
                    <span className="text-sm font-medium">Opacity</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      value={opacity}
                      onChange={(e) => setOpacity(parseFloat(e.target.value))}
                      className="w-full mt-1"
                    />
                  </label>
                )}
              </div>
            )}

            <button
              onClick={run}
              disabled={!state.isModelReady || running}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {running ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  {state.isModelReady ? "Analyze MRI" : "Model not loaded"}
                </>
              )}
            </button>
          </div>
        )}
      </motion.section>

      {/* Results Section */}
      <motion.section
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="card-pro p-6"
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-xl font-semibold">Analysis Results</h2>
        </div>

        {!result ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-slate-100 dark:bg-slate-900 flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-slate-400" />
            </div>
            <p className="text-slate-600 dark:text-slate-400">
              Upload an MRI scan to see results
            </p>
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              {/* Primary Result */}
              <div className={`p-6 rounded-2xl ${
                result.label === "Tumor" 
                  ? "bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-950/20 dark:to-orange-950/20"
                  : "bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20"
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-lg">Primary Diagnosis</h3>
                  {result.label === "Tumor" ? (
                    <XCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
                  ) : (
                    <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
                  )}
                </div>
                <div className="flex items-end justify-between">
                  <div>
                    <p className="text-3xl font-bold">{result.label}</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                      Detected with high confidence
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-4xl font-bold">
                      {(result.prob * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      confidence
                    </p>
                  </div>
                </div>
              </div>

              {/* Confidence Breakdown */}
              <div className="space-y-3">
                <h3 className="font-medium text-sm text-slate-700 dark:text-slate-300">
                  Confidence Distribution
                </h3>
                {result.topK?.map((item, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{item.label}</span>
                      <span className="text-slate-600 dark:text-slate-400">
                        {(item.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${item.probability * 100}%` }}
                        transition={{ duration: 0.5, delay: idx * 0.1 }}
                        className={`h-full ${
                          idx === 0 
                            ? "bg-gradient-to-r from-indigo-500 to-purple-600"
                            : "bg-slate-400 dark:bg-slate-600"
                        }`}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Explainability Info */}
              <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                  <div className="text-sm">
                    <p className="font-medium text-blue-900 dark:text-blue-300 mb-1">
                      About the Heatmap
                    </p>
                    <p className="text-blue-700 dark:text-blue-400">
                      The overlay shows which regions of the MRI most influenced the AI's decision. 
                      Brighter areas indicate higher importance.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        )}
      </motion.section>
    </div>
  );
}