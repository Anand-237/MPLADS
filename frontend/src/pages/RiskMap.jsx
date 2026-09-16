import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

export default function RiskMap() {
  const [stateData, setStateData] = useState([]);
  const [geoJson, setGeoJson] = useState(null);
  const [selectedState, setSelectedState] = useState('Gujarat');
  const [constituencyData, setConstituencyData] = useState([]);
  const [selectedConstituency, setSelectedConstituency] = useState('');
  const [projectsData, setProjectsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('state');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [resStates, resGeo] = await Promise.all([
        fetch('/api/risk-map/states').then(r => r.json()),
        fetch('/api/risk-map/geojson').then(r => r.json())
      ]);
      setStateData(resStates);
      setGeoJson(resGeo);
      if (resStates.length > 0) {
        setSelectedState(resStates[0].state);
      }
      setLoading(false);
    } catch (err) {
      console.error("Error fetching risk map data:", err);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedState) {
      fetchConstituencies(selectedState);
      fetchProjects(selectedState);
    }
  }, [selectedState]);

  const fetchConstituencies = async (stName) => {
    try {
      const res = await fetch(`/api/risk-map/constituencies?state=${encodeURIComponent(stName)}`).then(r => r.json());
      setConstituencyData(res);
      if (res.length > 0) {
        setSelectedConstituency(res[0].constituency);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchProjects = async (stName) => {
    try {
      const res = await fetch(`/api/projects?state=${encodeURIComponent(stName)}&limit=10`).then(r => r.json());
      setProjectsData(res.projects || []);
    } catch (err) {
      console.error(err);
    }
  };

  const stRow = stateData.find(s => s.state === selectedState) || stateData[0] || {};
  const cRow = constituencyData.find(c => c.constituency === selectedConstituency) || constituencyData[0] || {};

  // Map Plot Data
  const mapPlotData = [{
    type: 'choropleth',
    geojson: geoJson,
    locations: stateData.map(s => {
      if (s.state === 'The Dadra And Nagar Haveli And Daman And Diu' || s.state.includes('Dadra')) return 'Dadra and Nagar Haveli';
      if (s.state === 'Andaman & Nicobar Islands' || s.state === 'Andaman And Nicobar Islands') return 'Andaman and Nicobar Islands';
      if (s.state === 'Jammu & Kashmir' || s.state === 'Jammu And Kashmir') return 'Jammu and Kashmir';
      return s.state;
    }),
    z: stateData.map(s => s.area_risk_score),
    featureidkey: 'properties.st_nm',
    colorscale: [
      [0.0, '#10B981'],
      [0.4, '#F59E0B'],
      [0.65, '#F97316'],
      [1.0, '#EF4444']
    ],
    zmin: 20,
    zmax: 80,
    hoverinfo: 'text',
    text: stateData.map(s => `<b>${s.state}</b><br>Area Risk Score: ${s.area_risk_score}<br>Total Works: ${(s.total_works || 0).toLocaleString()}<br>High Risk Works: ${(s.high_risk_works || 0).toLocaleString()}<br>High Risk %: ${s.high_risk_pct}%`)
  }];

  const mapLayout = {
    geo: {
      fitbounds: 'locations',
      visible: false,
      bgcolor: 'rgba(248, 250, 252, 1)'
    },
    margin: { r: 0, t: 0, l: 0, b: 0 },
    height: 650,
    autosize: true
  };

  const handleMapClick = (evt) => {
    if (evt && evt.points && evt.points.length > 0) {
      const clickedLocation = evt.points[0].location;
      const matchingState = stateData.find(s => {
        if (s.state === clickedLocation) return true;
        if (clickedLocation === 'Dadra and Nagar Haveli' && (s.state.includes('Dadra') || s.state.includes('Daman'))) return true;
        if (clickedLocation === 'Andaman and Nicobar Islands' && s.state.includes('Andaman')) return true;
        if (clickedLocation === 'Jammu and Kashmir' && s.state.includes('Jammu')) return true;
        return false;
      });
      if (matchingState) {
        setSelectedState(matchingState.state);
      }
    }
  };

  return (
    <div className="page-body">
      {/* Legend Bar */}
      <div style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E2E8F0',
        borderRadius: '12px',
        padding: '0.8rem 1.5rem',
        marginBottom: '1.5rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: '0 2px 6px rgba(15,23,42,0.04)'
      }}>
        <div style={{ fontWeight: '800', fontSize: '1.1rem', color: '#0F172A' }}>
          🛡️ Area Risk Level Scale
        </div>
        <div style={{ display: 'flex', gap: '1.5rem', fontSize: '1.0rem', fontWeight: '600' }}>
          <span style={{ color: '#10B981' }}>🟢 Low (0–39)</span>
          <span style={{ color: '#F59E0B' }}>🟡 Moderate (40–54)</span>
          <span style={{ color: '#F97316' }}>🟠 Elevated (55–69)</span>
          <span style={{ color: '#EF4444' }}>🔴 High (70–100)</span>
        </div>
      </div>

      {/* Main Split: Map + State Inspector */}
      <div style={{ display: 'grid', gridTemplateColumns: '2.3fr 1fr', gap: '1.5rem', marginBottom: '2.0rem' }}>
        {/* Left: Plotly Map */}
        <div className="table-container" style={{ padding: '1.0rem' }}>
          <h3 style={{ marginBottom: '0.8rem', fontSize: '1.4rem' }}>Administrative India Risk Map</h3>
          {loading ? (
            <div style={{ padding: '4.0rem', textAlign: 'center', fontSize: '1.2rem' }}>Loading India Map Data...</div>
          ) : (
            <Plot
              data={mapPlotData}
              layout={mapLayout}
              useResizeHandler={true}
              style={{ width: '100%', height: '650px' }}
              onClick={handleMapClick}
            />
          )}
        </div>

        {/* Right: State Statistics Inspector Card */}
        <div>
          <div className="form-card" style={{ padding: '1.5rem', marginBottom: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '2.2rem' }}>📍</div>
            <h3 style={{ fontSize: '1.25rem', margin: '0.4rem 0' }}>State Statistics Inspector</h3>
            <p style={{ color: '#64748B', fontSize: '0.95rem', marginBottom: '1.0rem' }}>Click any state on the map to view statistics</p>

            <select
              className="form-select"
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              style={{ fontWeight: '700', fontSize: '1.1rem', marginBottom: '1.0rem' }}
            >
              {stateData.map(s => (
                <option key={s.state} value={s.state}>{s.state}</option>
              ))}
            </select>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.0rem', marginBottom: '1.0rem' }}>
              <div style={{ background: '#F8FAFC', padding: '0.8rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.85rem', color: '#64748B' }}>Area Risk Score</div>
                <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#0F172A', fontFamily: 'JetBrains Mono' }}>
                  {stRow.area_risk_score || 'N/A'}
                </div>
              </div>

              <div style={{ background: '#F8FAFC', padding: '0.8rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.85rem', color: '#64748B' }}>High-Risk %</div>
                <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#EF4444', fontFamily: 'JetBrains Mono' }}>
                  {stRow.high_risk_pct !== undefined ? stRow.high_risk_pct : 0}%
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.95rem', color: '#475569' }}>
              Total Works: <b>{(stRow.total_works || 0).toLocaleString()}</b> | High Risk: <b>{(stRow.high_risk_works || 0).toLocaleString()}</b>
            </div>
          </div>

          {/* Top Risk States List */}
          <div className="table-container" style={{ padding: '1.2rem' }}>
            <h4 style={{ color: '#475569', marginBottom: '1.0rem' }}>⚠️ TOP RISK STATES</h4>
            {stateData.slice(0, 6).map(ts => (
              <div
                key={ts.state}
                onClick={() => setSelectedState(ts.state)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.6rem 0.8rem',
                  borderBottom: '1px solid #F1F5F9',
                  cursor: 'pointer',
                  borderRadius: '6px',
                  backgroundColor: selectedState === ts.state ? '#EFF6FF' : 'transparent'
                }}
              >
                <span style={{ fontWeight: '600', color: '#334155' }}>{ts.state}</span>
                <span className="badge-web-medium">{ts.total_works.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Separate Detailed Tabs */}
      <div className="form-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '12px', borderBottom: '2px solid #E2E8F0', paddingBottom: '12px', marginBottom: '1.5rem' }}>
          <button
            className={`btn-primary ${activeTab === 'state' ? '' : 'ghost'}`}
            style={{ backgroundColor: activeTab === 'state' ? '#0F172A' : '#F1F5F9', color: activeTab === 'state' ? '#FFF' : '#334155' }}
            onClick={() => setActiveTab('state')}
          >
            🏛️ State Telemetry ({selectedState})
          </button>
          <button
            className={`btn-primary ${activeTab === 'constituency' ? '' : 'ghost'}`}
            style={{ backgroundColor: activeTab === 'constituency' ? '#0F172A' : '#F1F5F9', color: activeTab === 'constituency' ? '#FFF' : '#334155' }}
            onClick={() => setActiveTab('constituency')}
          >
            📌 Constituency Inspection
          </button>
          <button
            className={`btn-primary ${activeTab === 'works' ? '' : 'ghost'}`}
            style={{ backgroundColor: activeTab === 'works' ? '#0F172A' : '#F1F5F9', color: activeTab === 'works' ? '#FFF' : '#334155' }}
            onClick={() => setActiveTab('works')}
          >
            📋 Top Flagged Works
          </button>
        </div>

        {activeTab === 'state' && (
          <div>
            <h3 style={{ marginBottom: '1.0rem' }}>🏛️ State Overview & Telemetry: {selectedState}</h3>
            <div className="cards-grid">
              <div className="metric-card-web">
                <div className="metric-label-web">Median Risk Score</div>
                <div className="metric-value-web">{stRow.area_risk_score} / 100</div>
              </div>
              <div className="metric-card-web">
                <div className="metric-label-web">Total Works</div>
                <div className="metric-value-web">{(stRow.total_works || 0).toLocaleString()}</div>
              </div>
              <div className="metric-card-web">
                <div className="metric-label-web">High Risk Works</div>
                <div className="metric-value-web" style={{ color: '#EF4444' }}>{(stRow.high_risk_works || 0).toLocaleString()}</div>
              </div>
              <div className="metric-card-web">
                <div className="metric-label-web">High Risk %</div>
                <div className="metric-value-web">{stRow.high_risk_pct}%</div>
              </div>
            </div>

            <h4 style={{ margin: '1.5rem 0 1.0rem 0' }}>📋 Constituencies within {selectedState}</h4>
            <div className="table-container">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>Constituency</th>
                    <th>Total Works</th>
                    <th>High Risk Works</th>
                    <th>High Risk %</th>
                    <th>Area Risk Score</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {constituencyData.map(c => (
                    <tr key={c.constituency}>
                      <td><b>{c.constituency}</b></td>
                      <td>{c.total_works.toLocaleString()}</td>
                      <td>{c.high_risk_works}</td>
                      <td>{c.high_risk_pct}%</td>
                      <td><b>{c.area_risk_score}</b></td>
                      <td>
                        <span className={c.area_status === 'HIGH' ? 'badge-web-high' : (c.area_status === 'MEDIUM' ? 'badge-web-medium' : 'badge-web-low')}>
                          {c.area_status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'constituency' && (
          <div>
            <h3 style={{ marginBottom: '1.0rem' }}>📌 Constituency Deep-Dive</h3>
            <div style={{ marginBottom: '1.5rem', maxWidth: '400px' }}>
              <label className="form-label">Select Constituency:</label>
              <select
                className="form-select"
                value={selectedConstituency}
                onChange={(e) => setSelectedConstituency(e.target.value)}
              >
                {constituencyData.map(c => (
                  <option key={c.constituency} value={c.constituency}>{c.constituency}</option>
                ))}
              </select>
            </div>

            <div className="cards-grid">
              <div className="metric-card-web">
                <div className="metric-label-web">Constituency Risk Score</div>
                <div className="metric-value-web">{cRow.area_risk_score || 'N/A'}</div>
              </div>
              <div className="metric-card-web">
                <div className="metric-label-web">Total Works</div>
                <div className="metric-value-web">{(cRow.total_works || 0).toLocaleString()}</div>
              </div>
              <div className="metric-card-web">
                <div className="metric-label-web">High-Risk Works</div>
                <div className="metric-value-web" style={{ color: '#EF4444' }}>{cRow.high_risk_works || 0}</div>
              </div>
              <div className="metric-card-web">
                <div className="metric-label-web">High-Risk %</div>
                <div className="metric-value-web">{cRow.high_risk_pct || 0}%</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'works' && (
          <div>
            <h3 style={{ marginBottom: '1.0rem' }}>📋 Top Flagged Works in {selectedState}</h3>
            <div className="table-container">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>Work ID</th>
                    <th>Constituency</th>
                    <th>Description</th>
                    <th>Category</th>
                    <th>Amount (₹)</th>
                    <th>Risk Score</th>
                    <th>Risk Level</th>
                  </tr>
                </thead>
                <tbody>
                  {projectsData.map(w => (
                    <tr key={w.work_id}>
                      <td><b>#{w.work_id}</b></td>
                      <td>{w.constituency}</td>
                      <td>{w.work_description}</td>
                      <td>{w.category}</td>
                      <td>₹{w.effective_amount ? w.effective_amount.toLocaleString() : 0}</td>
                      <td><b>{w.final_risk_score}</b></td>
                      <td>
                        <span className={w.risk_category === 'HIGH' ? 'badge-web-high' : (w.risk_category === 'MEDIUM' ? 'badge-web-medium' : 'badge-web-low')}>
                          {w.risk_category}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
