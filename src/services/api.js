// frontend/src/services/api.js
const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function toJson(res) {
  const text = await res.text();
  if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
  return JSON.parse(text);
}

export const api = {
  async infer(file, mode = "quick") {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/api/infer?mode=${mode}`, {
      method: "POST",
      body: form,
    });
    return toJson(res);
  },
  async status() {
    const res = await fetch(`${BASE}/api/status`);
    return toJson(res);
  },
};