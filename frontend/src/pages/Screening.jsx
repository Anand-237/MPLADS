import React, { useState } from 'react';

export default function Screening() {
  const [formData, setFormData] = useState({
    work_description: '',
    category: 'ROADS, PATHWAYS AND BRIDGES',
    recommended_amount: 1500000,
    mp_name: 'Hon\'ble Member of Parliament',
    constituency: 'Varanasi',
    state: 'Uttar Pradesh'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('/api/screen-work', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await res.json();
      setResult(data);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div className="page-body">
      <h2 style={{ marginBottom: '1.5rem' }}>🆕 New Work AI Screening Engine</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2.0rem' }}>
        <form className="form-card" onSubmit={handleSubmit}>
          <h3 style={{ marginBottom: '1.0rem' }}>Proposed Work Parameters</h3>
          <div className="form-group">
            <label className="form-label">Work Description:</label>
            <textarea
              className="form-textarea"
              rows="3"
              value={formData.work_description}
              onChange={(e) => setFormData({ ...formData, work_description: e.target.value })}
              placeholder="e.g. Construction of RCC drain and CC road..."
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Category:</label>
            <input
              type="text"
              className="form-input"
              value={formData.category}
              onChange={(e) => setFormData({ ...formData, category: e.target.value })}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Recommended Amount (₹):</label>
            <input
              type="number"
              className="form-input"
              value={formData.recommended_amount}
              onChange={(e) => setFormData({ ...formData, recommended_amount: parseFloat(e.target.value) })}
              required
            />
          </div>

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Screening Proposed Work...' : '⚡ Screen New Work'}
          </button>
        </form>

        <div>
          {result ? (
            <div className="form-card">
              <h3 style={{ marginBottom: '1.0rem' }}>AI Screening Telemetry</h3>

              <div className="cards-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className="metric-card-web">
                  <div className="metric-label-web">Proposed Amount</div>
                  <div className="metric-value-web">₹{result.proposed_amount.toLocaleString()}</div>
                </div>
                <div className="metric-card-web">
                  <div className="metric-label-web">Historical Median</div>
                  <div className="metric-value-web">₹{result.historical_median.toLocaleString()}</div>
                </div>
                <div className="metric-card-web">
                  <div className="metric-label-web">Cost Difference %</div>
                  <div className="metric-value-web" style={{ color: result.cost_difference_pct > 50 ? '#EF4444' : '#10B981' }}>
                    {result.cost_difference_pct}%
                  </div>
                </div>
                <div className="metric-card-web">
                  <div className="metric-label-web">Final AI Risk Score</div>
                  <div className="metric-value-web">{result.final_risk_score} / 100</div>
                  <span className={result.risk_level === 'HIGH' ? 'badge-web-high' : 'badge-web-low'}>
                    {result.risk_level} PRIORITY
                  </span>
                </div>
              </div>

              <h4 style={{ margin: '1.5rem 0 0.8rem 0' }}>NLP Description Matches (Top Candidate)</h4>
              <div className="table-container">
                <table className="custom-table">
                  <thead>
                    <tr>
                      <th>Match ID</th>
                      <th>Description</th>
                      <th>Similarity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.similar_matches.map(m => (
                      <tr key={m.work_id}>
                        <td>#{m.work_id}</td>
                        <td>{m.work_description}</td>
                        <td><b>{m.similarity_pct}%</b></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="form-card" style={{ textAlign: 'center', padding: '4.0rem' }}>
              <div style={{ fontSize: '3.0rem', marginBottom: '1.0rem' }}>🔍</div>
              <h3>Ready to Screen Proposed MPLADS Work</h3>
              <p style={{ color: '#64748B' }}>Enter work details on the left and click 'Screen New Work' to run real-time cost Z-score, NLP text similarity, and Isolation Forest ML scoring.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
