import { useState } from "react";
import Tabs from "./components/Tabs";
import ModelManager from "./components/ModelManager";
import InferencePanel from "./components/InferencePanel";

export default function App() {
  const [activeTab, setActiveTab] = useState("models");

  return (
    <div className="min-h-screen">
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white/70 dark:bg-gray-900/70 backdrop-blur">
        <div className="container-pro py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-brand text-white grid place-items-center font-semibold">
              NS
            </div>
            <div>
              <h1 className="font-semibold text-xl">NeuroScope XAI</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Explainable Brain MRI Analysis
              </p>
            </div>
          </div>
          <a
            className="text-sm text-brand hover:text-brand-dark underline underline-offset-4"
            href="#"
            onClick={(e) => e.preventDefault()}
            title="Docs coming soon"
          >
            Docs
          </a>
        </div>
      </header>

      <main className="container-pro py-8">
        <Tabs
          tabs={[
            { id: "models", label: "Models" },
            { id: "analyze", label: "Analyze" },
          ]}
          activeId={activeTab}
          onChange={(id) => setActiveTab(id)}
        />

        <div className="mt-6">
          {activeTab === "models" ? <ModelManager /> : <InferencePanel />}
        </div>
      </main>

      <footer className="container-pro py-10 text-sm text-gray-500 dark:text-gray-400">
        Built with React, Vite, and Tailwind • © {new Date().getFullYear()} NeuroScope
      </footer>
    </div>
  );
}