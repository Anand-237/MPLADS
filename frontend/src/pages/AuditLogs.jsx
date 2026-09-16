import React, { useState, useEffect } from 'react';

export default function AuditLogs() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    fetch('/api/audit-logs')
      .then(res => res.json())
      .then(data => setLogs(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="page-body">
      <h2 style={{ marginBottom: '1.5rem' }}>📋 Human Verification Audit Logs</h2>

      <div className="table-container">
        <table className="custom-table">
          <thead>
            <tr>
              <th>Log ID</th>
              <th>Work ID</th>
              <th>Review Status</th>
              <th>Reviewer Notes</th>
              <th>Reviewer Officer</th>
              <th>Timestamp</th>
              <th>AI Score</th>
            </tr>
          </thead>
          <tbody>
            {logs.length > 0 ? (
              logs.map(log => (
                <tr key={log.log_id || log.review_timestamp}>
                  <td><b>{log.log_id}</b></td>
                  <td>#{log.work_id}</td>
                  <td>
                    <span className={log.review_status.includes('Needs') ? 'badge-web-medium' : 'badge-web-low'}>
                      {log.review_status}
                    </span>
                  </td>
                  <td>{log.reviewer_notes}</td>
                  <td>{log.reviewer_id}</td>
                  <td style={{ fontFamily: 'JetBrains Mono', fontSize: '0.85rem' }}>{log.review_timestamp}</td>
                  <td><b>{log.ai_risk_score_at_review}</b></td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '2.0rem' }}>No audit reviews recorded yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
