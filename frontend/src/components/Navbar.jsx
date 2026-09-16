import React from 'react';

export default function Navbar({ activeTab }) {
  const titles = {
    'risk-map': '🗺️ Administrative India AI Risk Map',
    'overview': '📊 Executive Anomaly & Risk Overview',
    'screening': '🆕 Proposed MPLADS Work Screening Engine',
    'explorer': '🔎 Project Explorer & Data Warehouse',
    'audit': '📋 Human Verification Audit Logs'
  };

  return (
    <header className="navbar">
      <div className="navbar-title">
        {titles[activeTab] || 'MPLADS SENTINEL'}
      </div>

      <div className="navbar-status">
        <span className="status-dot"></span>
        <span>127,263 Projects Monitored</span>
      </div>
    </header>
  );
}
