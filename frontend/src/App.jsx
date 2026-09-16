import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import RiskMap from './pages/RiskMap';
import Overview from './pages/Overview';
import Screening from './pages/Screening';
import AuditLogs from './pages/AuditLogs';

export default function App() {
  const [activeTab, setActiveTab] = useState('risk-map');

  const renderContent = () => {
    switch (activeTab) {
      case 'risk-map':
        return <RiskMap />;
      case 'overview':
        return <Overview />;
      case 'screening':
        return <Screening />;
      case 'audit':
        return <AuditLogs />;
      default:
        return <RiskMap />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        <Navbar activeTab={activeTab} />
        {renderContent()}
      </main>
    </div>
  );
}
