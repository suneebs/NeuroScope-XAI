import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, Brain, Loader2, FileImage } from "lucide-react";
import { api } from "../services/api";
import ComprehensiveAnalysis from "./ComprehensiveAnalysis";

export default function InferencePanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);

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
      setResult(res);
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
    <div className="space-y-8">
      {/* Upload Section */}
      {!result && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-pro p-6 max-w-4xl mx-auto"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center">
              <FileImage className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-semibold">Upload Brain MRI for Complete Analysis</h2>
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
                Supports: PNG, JPG, JPEG • Max 10MB
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
              <div className="max-w-md mx-auto">
                <img
                  src={preview}
                  alt="MRI scan"
                  className="w-full rounded-xl shadow-lg"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="text-sm">
                  <p className="font-medium">{file.name}</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <button onClick={reset} className="btn-secondary">
                  Change Image
                </button>
              </div>

              <button
                onClick={run}
                disabled={running}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                {running ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Performing Complete TrueExplainableAI Analysis...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    Start Comprehensive Analysis
                  </>
                )}
              </button>

              <div className="text-sm text-slate-600 dark:text-slate-400 text-center">
                This will perform occlusion analysis, Grad-CAM visualization, 
                pattern recognition, and generate a complete medical interpretation.
              </div>
            </div>
          )}
        </motion.section>
      )}

      {/* Results */}
      {result && (
        <>
          <div className="text-center mb-4">
            <button onClick={reset} className="btn-secondary">
              Analyze Another MRI
            </button>
          </div>
          <ComprehensiveAnalysis result={result} />
        </>
      )}
    </div>
  );
}