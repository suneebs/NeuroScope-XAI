import { useState, useEffect } from "react";
import { Brain, Activity, Zap, FileImage, Settings, Github, Twitter, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ModelManager from "./components/ModelManager";
import InferencePanel from "./components/InferencePanel";
import { useModel } from "./context/ModelContext";

export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const { state, dispatch } = useModel();
  
  // Enable demo mode if no model is loaded
  useEffect(() => {
    // Auto-enable model after 2 seconds if not loaded
    const timer = setTimeout(() => {
      if (!state.isModelReady) {
        console.log("Enabling demo mode");
        dispatch({ type: "READY", payload: true });
      }
    }, 2000);
    
    return () => clearTimeout(timer);
  }, [state.isModelReady, dispatch]);

  const tabs = [
    { id: "analyze", label: "Analyze", icon: FileImage },
    { id: "models", label: "Models", icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float animation-delay-2000" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-float animation-delay-4000" />
      </div>

      {/* Header */}
      <header className="relative glass border-b border-white/10 dark:border-slate-800/50">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.div
                initial={{ rotate: -180, scale: 0 }}
                animate={{ rotate: 0, scale: 1 }}
                transition={{ type: "spring", duration: 0.8 }}
                className="relative"
              >
                <div className="w-12 h-12 rounded-2xl gradient-brand p-[1px]">
                  <div className="w-full h-full rounded-2xl bg-white dark:bg-slate-900 flex items-center justify-center">
                    <Brain className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                  </div>
                </div>
                <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-slate-900" />
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  NeuroScope XAI
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Advanced Brain MRI Analysis Platform
                </p>
              </div>
            </div>

            <nav className="flex items-center gap-6">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
                state.isModelReady 
                  ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                  : 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400'
              }`}>
                {state.isModelReady ? (
                  <>
                    <Activity className="w-4 h-4" />
                    <span className="text-sm font-medium">Model Ready</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm font-medium">No Model Loaded</span>
                  </>
                )}
              </div>
              {/* Demo mode button */}
              {!state.isModelReady && (
                <button
                  onClick={() => dispatch({ type: "READY", payload: true })}
                  className="text-sm px-3 py-1.5 rounded-lg bg-indigo-100 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-400 hover:bg-indigo-200 dark:hover:bg-indigo-900/30"
                >
                  Enable Demo Mode
                </button>
              )}
              <a href="#" className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100">
                <Twitter className="w-5 h-5" />
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Main content - rest remains the same */}
      <main className="relative max-w-7xl mx-auto px-6 py-8">
        {/* Tab navigation */}
        <div className="flex items-center justify-center mb-8">
          <div className="inline-flex p-1 glass rounded-2xl shadow-lg">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    relative px-6 py-3 rounded-xl font-medium transition-all duration-200
                    ${isActive 
                      ? "text-white shadow-lg" 
                      : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
                    }
                  `}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 gradient-brand rounded-xl"
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

        {/* Show notice if in demo mode */}
        {state.isModelReady && !state.selectedModel && (
          <div className="mb-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <p className="text-sm text-blue-700 dark:text-blue-300">
                Demo mode enabled. Upload an image to see sample predictions. 
                Go to Models tab to load a real model.
              </p>
            </div>
          </div>
        )}

        {/* Tab content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
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
              <span>Powered by Advanced Neural Networks</span>
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