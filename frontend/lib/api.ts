export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export async function apiAsk(query: string, k: number = 6) {
  const r = await fetch(`${API_BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k })
  });
  if (!r.ok) throw new Error('Ask failed');
  return r.json();
}

export async function apiIngest() {
  const r = await fetch(`${API_BASE}/api/ingest`, { method: 'POST' });
  if (!r.ok) throw new Error('Ingest failed');
  return r.json();
}

export async function apiMetrics() {
  const r = await fetch(`${API_BASE}/api/metrics`);
  if (!r.ok) throw new Error('Metrics failed');
  return r.json();
}
