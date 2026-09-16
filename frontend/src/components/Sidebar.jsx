import React from 'react';
import { Map, BarChart3, PlusCircle, Search, FileCheck, Layers } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const menuItems = [
    { id: 'risk-map', label: 'AI Risk Map', icon: Map },
    { id: 'overview', label: 'Executive Overview', icon: BarChart3 },
    { id: 'screening', label: 'New Work Screening', icon: PlusCircle },
    { id: 'explorer', label: 'Project Explorer', icon: Search },
    { id: 'audit', label: 'Audit & Verification', icon: FileCheck }
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand-icon">🛡️</div>
        <div className="sidebar-brand-title">MPLADS SENTINEL</div>
        <div className="sidebar-brand-sub">AI Oversight Platform</div>
      </div>

      <nav className="sidebar-menu">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <div
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon className="nav-icon" />
              <span>{item.label}</span>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
