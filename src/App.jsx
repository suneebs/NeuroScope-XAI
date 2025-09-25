import { useState, useEffect } from "react";
import { Brain, Activity, Zap, FileImage, Settings, Github, Twitter, AlertCircle, RefreshCw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ModelManager from "./components/ModelManager";
import InferencePanel from "./components/InferencePanel";
import { useModel } from "./context/ModelContext";

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const { state, dispatch } = useModel();
  const [backendStatus, setBackendStatus] = useState({ online: true, modelLoaded: false, checking: true });

  // Poll backend status
  useEffect(() => {
    let stop = false;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${BASE}/api/status`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        if (!stop) setBackendStatus({ online: true, modelLoaded: !!data.model_loaded, checking: false });
      } catch {
        if (!stop) setBackendStatus({ online: false, modelLoaded: false, checking: false });
      }
    };
    fetchStatus();
    const t = setInterval(fetchStatus, 5000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, []);

  // Demo mode (optional)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!state.isModelReady) dispatch({ type: "READY", payload: true });
    }, 2000);
    return () => clearTimeout(timer);
  }, [state.isModelReady, dispatch]);

  const reloadModel = async () => {
    try {
      await fetch(`${BASE}/api/models/reload`, { method: "POST" });
      const s = await fetch(`${BASE}/api/status`).then((r) => r.json());
      setBackendStatus({ online: true, modelLoaded: !!s.model_loaded, checking: false });
    } catch {}
  };

  const tabs = [
    { id: "analyze", label: "Analyze", icon: FileImage },
    { id: "models", label: "Models", icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply blur-3xl opacity-20 animate-[float_6s_ease-in-out_infinite]" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply blur-3xl opacity-20 animate-[float_6s_ease-in-out_infinite_1s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-300 rounded-full mix-blend-multiply blur-3xl opacity-10 animate-[float_6s_ease-in-out_infinite_2s]" />
      </div>

      {/* Header */}
      <header className="relative bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200/70 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.div
                initial={{ rotate: -180, scale: 0 }}
                animate={{ rotate: 0, scale: 1 }}
                transition={{ type: "spring", duration: 0.8 }}
                className="relative"
              >
                <div className="w-12 h-12 rounded-2xl p-[1px] bg-gradient-to-br from-indigo-500 to-purple-600">
                  <div className="w-full h-full rounded-2xl bg-white dark:bg-slate-900 grid place-items-center">
                    <Brain className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                  </div>
                </div>
                <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-slate-900" />
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  NeuroScope XAI
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">Advanced Brain MRI Analysis Platform</p>
              </div>
            </div>

            <nav className="flex items-center gap-3">
              {/* Backend status */}
              <div
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
                  backendStatus.online
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                    : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                }`}
              >
                <Activity className="w-4 h-4" />
                <span>{backendStatus.online ? "Backend Online" : "Backend Offline"}</span>
              </div>

              {/* Model status */}
              <div
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
                  backendStatus.modelLoaded
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                    : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                }`}
              >
                {backendStatus.modelLoaded ? <Activity className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                <span>{backendStatus.modelLoaded ? "Model Loaded" : "No Model Loaded"}</span>
              </div>

              {/* Reload model */}
              <button
                onClick={reloadModel}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 text-sm"
                title="Reload model on backend"
              >
                <RefreshCw className="w-4 h-4" />
                Reload
              </button>

              <a className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100" href="#" aria-label="GitHub">
                <Github className="w-5 h-5" />
              </a>
              <a className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100" href="#" aria-label="Twitter">
                <Twitter className="w-5 h-5" />
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative max-w-7xl mx-auto px-6 py-8">
        {/* Tabs */}
        <div className="flex items-center justify-center mb-8">
          <div className="inline-flex p-1 bg-white/70 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 rounded-2xl shadow-sm">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative px-6 py-3 rounded-xl font-medium transition ${
                    isActive
                      ? "text-white shadow"
                      : "text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100"
                  }`}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                >
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600"
                      transition={{ type: "spring", duration: 0.5 }}
                    />
                  )}
                  <span className="relative flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    {tab.label}
                  </span>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Demo message */}
        {state.isModelReady && !backendStatus.modelLoaded && (
          <div className="mb-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <p className="text-sm text-blue-700 dark:text-blue-300">
                Demo mode enabled. Upload an image to see AI output. Use the Models tab or Reload button to load a real model.
              </p>
            </div>
          </div>
        )}

        {/* Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -18 }}
            transition={{ duration: 0.25 }}
          >
            {activeTab === "models" ? <ModelManager /> : <InferencePanel />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="relative mt-20 border-t border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <Zap className="w-4 h-4" />
              <span>Powered by EfficientNet & Explainable AI</span>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              © {new Date().getFullYear()} NeuroScope. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}