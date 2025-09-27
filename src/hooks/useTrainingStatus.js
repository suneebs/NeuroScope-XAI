import { useState, useEffect, useRef } from 'react';

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export function useTrainingStatus(jobId) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const intervalRef = useRef(null);

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPolling(false);
  };

  useEffect(() => {
    if (!jobId) {
      stopPolling();
      setStatus(null);
      return;
    }

    const tick = async () => {
      try {
        const res = await fetch(`${BASE}/api/train/status?jobId=${encodeURIComponent(jobId)}`);
        if (!res.ok) {
          throw new Error(`Server responded with ${res.status}`);
        }
        const data = await res.json();
        setStatus(data);

        if (["completed", "error", "stopped"].includes(data.status)) {
          stopPolling();
        }
      } catch (e) {
        setError(e.message || "Failed to fetch training status.");
        stopPolling();
      }
    };

    // Start polling
    setIsPolling(true);
    tick(); // Run immediately
    intervalRef.current = setInterval(tick, 2500); // Poll every 2.5 seconds

    // Cleanup function
    return () => {
      stopPolling();
    };
  }, [jobId]); // This effect ONLY re-runs if jobId changes

  return { status, error, isPolling };
}