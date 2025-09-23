const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function toJson(res) {
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const api = {
  async listModels() {
    try {
      return await toJson(
        await fetch(`${BASE}/api/models`, { credentials: "include" })
      );
    } catch {
      // Mock fallback
      return {
        models: [
          { id: "bt-1", name: "BrainTumorNet", version: "1.0", sizeMB: 45, source: "remote" },
          { id: "bt-2", name: "BrainTumorNet+ (XAI)", version: "1.1", sizeMB: 52, source: "remote" },
        ],
      };
    }
  },

  async downloadModel(id, onProgress) {
    // Simulated streaming download; replace with real streaming fetch
    for (let p = 0; p <= 100; p += 10) {
      await new Promise((r) => setTimeout(r, 120));
      onProgress && onProgress(p);
    }
    return { ok: true };
  },

  async uploadModel(file, onProgress) {
    // Simulated upload; replace with multipart/form-data
    for (let p = 0; p <= 100; p += 20) {
      await new Promise((r) => setTimeout(r, 120));
      onProgress && onProgress(p);
    }
    return { id: `upload-${Date.now()}`, name: file.name, source: "uploaded" };
  },

  async loadModel(id) {
    await new Promise((r) => setTimeout(r, 500));
    return { loaded: true };
  },

  async startTraining(config) {
    await new Promise((r) => setTimeout(r, 300));
    return { jobId: `job-${Date.now()}` };
  },

  async trainingStatus(jobId) {
    // Simulated curve
    const epoch = Math.min(Math.floor((Date.now() / 1000) % 20), 19);
    const progress = Math.min(100, Math.round((epoch / 20) * 100));
    return {
      epoch,
      totalEpochs: 20,
      progress,
      metrics: { loss: Math.max(0.1, 1.5 - epoch * 0.06), valLoss: Math.max(0.12, 1.8 - epoch * 0.07) },
      status: progress >= 100 ? "completed" : "running",
    };
  },

  async infer(file) {
    // Replace with POST /api/infer multipart/form-data
    await new Promise((r) => setTimeout(r, 800));
    return {
      prediction: { label: "Tumor", probability: 0.93 },
      topK: [
        { label: "Tumor", probability: 0.93 },
        { label: "No Tumor", probability: 0.07 },
      ],
      // 1x1 PNG as placeholder. Replace with Grad-CAM heatmap data URL.
      heatmap:
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQYGWNgYGD4DwABGgE25dQb9gAAAABJRU5ErkJggg==",
    };
  },
};