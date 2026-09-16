import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Overview() {
  const [overview, setOverview] = useState(null);

  useEffect(() => {
    fetch('/api/overview')
      .then(res => res.json())
      .then(data => setOverview(data))
      .catch(err => console.error(err));
  }, []);

  if (!overview) {
    return <div className="page-body">Loading Executive Overview...</div>;
  }

  const pieData = [
    { name: 'HIGH Risk', value: overview.high_risk_count, color: '#EF4444' },
    { name: 'MEDIUM Risk', value: overview.medium_risk_count, color: '#F59E0B' },
    { name: 'LOW Risk', value: overview.low_risk_count, color: '#10B981' }
  ];

  return (
    <div className="page-body">
      <h2 style={{ marginBottom: '1.5rem' }}>📊 Executive Anomaly & Risk Overview</h2>

      <div className="cards-grid">
        <div className="metric-card-web">
          <div className="metric-label-web">Total Projects Monitored</div>
          <div className="metric-value-web">{overview.total_projects.toLocaleString()}</div>
        </div>
        <div className="metric-card-web">
          <div className="metric-label-web">HIGH Risk Priority</div>
          <div className="metric-value-web" style={{ color: '#EF4444' }}>{overview.high_risk_count.toLocaleString()}</div>
          <div style={{ fontSize: '0.85rem', color: '#EF4444', fontWeight: '700' }}>{overview.high_risk_pct}% of Corpus</div>
        </div>
        <div className="metric-card-web">
          <div className="metric-label-web">MEDIUM Risk Priority</div>
          <div className="metric-value-web" style={{ color: '#F59E0B' }}>{overview.medium_risk_count.toLocaleString()}</div>
          <div style={{ fontSize: '0.85rem', color: '#F59E0B', fontWeight: '700' }}>{overview.medium_risk_pct}% of Corpus</div>
        </div>
        <div className="metric-card-web">
          <div className="metric-label-web">Average Risk Score</div>
          <div className="metric-value-web">{overview.average_risk_score} / 100</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div className="form-card">
          <h3 style={{ marginBottom: '1.0rem' }}>Risk Category Distribution</h3>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="form-card">
          <h3 style={{ marginBottom: '1.0rem' }}>Top State Risk Scores</h3>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer>
              <BarChart data={overview.top_state_risks}>
                <XAxis dataKey="state" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="median_risk_score" fill="#2563EB" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
