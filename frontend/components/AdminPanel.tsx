'use client';
import React, { useState, useEffect } from 'react';
import { Database } from 'lucide-react';
import { apiIngest, apiMetrics } from '../lib/api';

export default function AdminPanel() {
  const [metrics, setMetrics] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  // ---------------- Actions ---------------- //
  const refreshMetrics = async () => {
    const m = await apiMetrics();
    setMetrics(m);
  };

  const handleIngest = async () => {
    setBusy(true);
    try {
      await apiIngest();
      await refreshMetrics();
    } finally {
      setBusy(false);
    }
  };

  // ---------------- Effects ---------------- //
  useEffect(() => {
    refreshMetrics();
  }, []);

  // ---------------- Render ---------------- //
  return (
    <div className="bg-white p-6 rounded-xl shadow-xl flex flex-col transition-all duration-300">

      {/* Header */}
      <div className="flex items-center mb-4 space-x-3">
        <Database className="w-10 h-10 text-purple-600" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Admin Panel</h2>
      </div>

      {/* Buttons */}
      <div className="flex flex-wrap gap-4 mb-6">
        <button
          onClick={handleIngest}
          disabled={busy}
          className={`px-5 py-2 rounded-lg font-semibold transition duration-200 shadow-md ${
            busy
              ? 'bg-indigo-300 text-gray-800 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white'
          }`}
        >
          {busy ? 'Indexing...' : 'Ingest Sample Docs'}
        </button>

        <button
          onClick={refreshMetrics}
          className="px-5 py-2 rounded-lg font-semibold bg-gray-100 border border-gray-300 hover:bg-gray-200 text-gray-800 transition duration-200 shadow-sm"
        >
          Refresh Metrics
        </button>
      </div>

      {/* Metrics Display */}
      {metrics && (
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 overflow-x-auto shadow-inner">
          <pre className="text-sm text-gray-800 font-mono">
            {JSON.stringify(metrics, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
