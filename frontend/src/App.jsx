import { useState, useEffect, useRef } from 'react'
import './App.css'

const API = 'http://localhost:8000/api'

function App() {
  const [page, setPage] = useState('dashboard')
  const [status, setStatus] = useState(null)
  const [stats, setStats] = useState(null)
  const [logs, setLogs] = useState({ download: '', build_features: '' })
  const [showLogs, setShowLogs] = useState({ download: false, build_features: false })
  const [preview, setPreview] = useState(null)
  const [previewDataset, setPreviewDataset] = useState(null)
  const logEndRef = useRef({})

  // Poll status every 2 seconds
  useEffect(() => {
    const fetchStatus = () => {
      fetch(`${API}/status`).then(r => r.json()).then(setStatus).catch(() => {})
      fetch(`${API}/data-stats`).then(r => r.json()).then(setStats).catch(() => {})
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const streamLogs = (step) => {
    setShowLogs(prev => ({ ...prev, [step]: true }))
    setLogs(prev => ({ ...prev, [step]: '' }))

    const eventSource = new EventSource(`${API}/logs/${step}`)
    eventSource.onmessage = (e) => {
      setLogs(prev => ({ ...prev, [step]: prev[step] + e.data }))
    }
    eventSource.addEventListener('status', (e) => {
      eventSource.close()
    })
    eventSource.onerror = () => {
      eventSource.close()
    }
  }

  const runStep = (endpoint, step) => {
    fetch(`${API}/${endpoint}`, { method: 'POST' })
    setTimeout(() => streamLogs(step), 500)
  }

  const loadPreview = (dataset) => {
    setPreviewDataset(dataset)
    fetch(`${API}/data-preview/${dataset}?rows=15`)
      .then(r => r.json())
      .then(setPreview)
      .catch(() => setPreview(null))
  }

  const getStepStatus = (step) => status?.steps?.[step]?.status || 'idle'
  const getStepFiles = (step) => status?.steps?.[step]?.files || []
  const hasData = (step) => status?.steps?.[step]?.has_data || false

  const statusLabel = (s) => {
    if (s === 'running') return 'Running'
    if (s === 'done') return 'Complete'
    if (s === 'error') return 'Error'
    return 'Ready'
  }

  const statusIcon = (s) => {
    if (s === 'done') return '\u2713'
    if (s === 'running') return '\u25CB'
    if (s === 'error') return '!'
    return '\u2022'
  }

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">SB</div>
          <div>
            <div className="logo-text">SportsBetting</div>
            <div className="logo-sub">ML Pipeline</div>
          </div>
        </div>

        <div className="sidebar-label">Menu</div>
        <nav className="sidebar-nav">
          <button className={`nav-item ${page === 'dashboard' ? 'active' : ''}`} onClick={() => setPage('dashboard')}>
            <span className="nav-icon">{'\u25A6'}</span> Dashboard
          </button>
          <button className={`nav-item ${page === 'pipeline' ? 'active' : ''}`} onClick={() => setPage('pipeline')}>
            <span className="nav-icon">{'\u25B6'}</span> Data Pipeline
          </button>
          <button className={`nav-item ${page === 'data' ? 'active' : ''}`} onClick={() => setPage('data')}>
            <span className="nav-icon">{'\u2630'}</span> Data Explorer
          </button>
          <button className={`nav-item ${page === 'training' ? 'active' : ''}`} onClick={() => setPage('training')}>
            <span className="nav-icon">{'\u2699'}</span> Training
          </button>
          <button className={`nav-item ${page === 'predict' ? 'active' : ''}`} onClick={() => setPage('predict')}>
            <span className="nav-icon">{'\u2605'}</span> Predictions
          </button>
        </nav>

        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '16px', marginTop: '16px' }}>
          <div className="sidebar-label">Status</div>
          <div style={{ padding: '0 12px', fontSize: '12px', color: 'var(--text-muted)' }}>
            {status ? 'Backend connected' : 'Connecting...'}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main">
        {page === 'dashboard' && (
          <DashboardPage stats={stats} status={status} setPage={setPage} />
        )}
        {page === 'pipeline' && (
          <PipelinePage
            status={status}
            logs={logs}
            showLogs={showLogs}
            runStep={runStep}
            getStepStatus={getStepStatus}
            getStepFiles={getStepFiles}
            hasData={hasData}
            statusLabel={statusLabel}
            statusIcon={statusIcon}
          />
        )}
        {page === 'data' && (
          <DataExplorerPage
            preview={preview}
            previewDataset={previewDataset}
            loadPreview={loadPreview}
            hasData={hasData}
          />
        )}
        {page === 'training' && <ComingSoonPage title="Training" desc="Train models on your engineered features" />}
        {page === 'predict' && <ComingSoonPage title="Predictions" desc="Generate player prop predictions" />}
      </main>
    </div>
  )
}

function DashboardPage({ stats, status, setPage }) {
  return (
    <>
      <div className="header">
        <div className="header-title">
          <h1>Dashboard</h1>
          <p>NFL Player Props ML Pipeline</p>
        </div>
      </div>

      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon blue">{'\u2630'}</div>
          </div>
          <div className="stat-label">Total Rows</div>
          <div className="stat-value">{stats?.total_rows?.toLocaleString() || '---'}</div>
          <div className="stat-sub">Player-game records</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon green">{'\u2605'}</div>
          </div>
          <div className="stat-label">Seasons</div>
          <div className="stat-value">{stats?.seasons?.length || '---'}</div>
          <div className="stat-sub">{stats?.seasons ? `${stats.seasons[0]} - ${stats.seasons[stats.seasons.length-1]}` : 'No data yet'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon orange">{'\u263A'}</div>
          </div>
          <div className="stat-label">Players</div>
          <div className="stat-value">{stats?.players?.toLocaleString() || '---'}</div>
          <div className="stat-sub">Unique players</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon red">{'\u2699'}</div>
          </div>
          <div className="stat-label">Features</div>
          <div className="stat-value">{stats?.total_columns || '---'}</div>
          <div className="stat-sub">Engineered columns</div>
        </div>
      </div>

      {/* Pipeline Status Overview */}
      <div className="section-card">
        <div className="section-header">
          <span className="section-title">Pipeline Status</span>
          <button className="btn btn-primary" onClick={() => setPage('pipeline')}>
            Go to Pipeline
          </button>
        </div>
        <div className="section-body">
          <div style={{ display: 'flex', gap: '24px' }}>
            <StepStatusChip label="Download Data" status={status?.steps?.download?.status || 'idle'} hasData={status?.steps?.download?.has_data} />
            <div style={{ color: 'var(--text-muted)', alignSelf: 'center' }}>{'\u2192'}</div>
            <StepStatusChip label="Build Features" status={status?.steps?.build_features?.status || 'idle'} hasData={status?.steps?.build_features?.has_data} />
            <div style={{ color: 'var(--text-muted)', alignSelf: 'center' }}>{'\u2192'}</div>
            <StepStatusChip label="Train Model" status="idle" hasData={false} />
            <div style={{ color: 'var(--text-muted)', alignSelf: 'center' }}>{'\u2192'}</div>
            <StepStatusChip label="Predict" status="idle" hasData={false} />
          </div>
        </div>
      </div>

      {/* Top Passers 2025 */}
      {stats?.top_passers_2025 && Object.keys(stats.top_passers_2025).length > 0 && (
        <div className="section-card">
          <div className="section-header">
            <span className="section-title">Top Passers — 2025 Season</span>
          </div>
          <div className="section-body">
            <div className="data-table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Player</th>
                    <th>Passing Yards</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.top_passers_2025).map(([name, yards], i) => (
                    <tr key={name}>
                      <td>{i + 1}</td>
                      <td style={{ fontWeight: 600 }}>{name}</td>
                      <td>{yards.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function StepStatusChip({ label, status, hasData }) {
  const s = hasData && status === 'idle' ? 'done' : status
  return (
    <div style={{ textAlign: 'center' }}>
      <div className={`status-badge ${s}`}>
        {s === 'running' && <span className="spinner" />}
        {s === 'done' ? '\u2713' : ''} {label}
      </div>
    </div>
  )
}

function PipelinePage({ status, logs, showLogs, runStep, getStepStatus, getStepFiles, hasData, statusLabel, statusIcon }) {
  return (
    <>
      <div className="header">
        <div className="header-title">
          <h1>Data Pipeline</h1>
          <p>Download, merge, and engineer features step by step</p>
        </div>
      </div>

      <div className="section-card">
        <div className="section-body">
          <div className="pipeline-steps">
            {/* Step 1: Download */}
            <div className="pipeline-step">
              <div className={`step-number ${hasData('download') ? 'done' : getStepStatus('download')}`}>
                {hasData('download') ? '\u2713' : '1'}
              </div>
              <div className="step-info">
                <div className="step-name">Download NFL Data</div>
                <div className="step-desc">
                  Pull weekly stats, schedules, snap counts, Next Gen Stats, and injuries from nfl_data_py (2016-2025)
                </div>
                {getStepFiles('download').length > 0 && (
                  <div className="step-files">
                    {getStepFiles('download').map(f => (
                      <span key={f} className="file-tag">{f}</span>
                    ))}
                  </div>
                )}
                <span className={`status-badge ${hasData('download') ? 'done' : getStepStatus('download')}`} style={{ marginTop: '8px' }}>
                  {getStepStatus('download') === 'running' && <span className="spinner" />}
                  {statusLabel(hasData('download') ? 'done' : getStepStatus('download'))}
                </span>
                {showLogs.download && logs.download && (
                  <div className="log-viewer">
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{logs.download}</pre>
                  </div>
                )}
              </div>
              <div className="step-action">
                <button
                  className="btn btn-primary"
                  onClick={() => runStep('download', 'download')}
                  disabled={getStepStatus('download') === 'running'}
                >
                  {getStepStatus('download') === 'running' ? 'Running...' : hasData('download') ? 'Re-download' : 'Download'}
                </button>
              </div>
            </div>

            {/* Step 2: Build Features */}
            <div className="pipeline-step">
              <div className={`step-number ${hasData('build_features') ? 'done' : getStepStatus('build_features')}`}>
                {hasData('build_features') ? '\u2713' : '2'}
              </div>
              <div className="step-info">
                <div className="step-name">Merge & Build Features</div>
                <div className="step-desc">
                  Join all datasets, compute rolling averages (3/5/8 games), opponent defense stats, trends, and temporal features
                </div>
                {getStepFiles('build_features').length > 0 && (
                  <div className="step-files">
                    {getStepFiles('build_features').map(f => (
                      <span key={f} className="file-tag">{f}</span>
                    ))}
                  </div>
                )}
                <span className={`status-badge ${hasData('build_features') ? 'done' : getStepStatus('build_features')}`} style={{ marginTop: '8px' }}>
                  {getStepStatus('build_features') === 'running' && <span className="spinner" />}
                  {statusLabel(hasData('build_features') ? 'done' : getStepStatus('build_features'))}
                </span>
                {showLogs.build_features && logs.build_features && (
                  <div className="log-viewer">
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{logs.build_features}</pre>
                  </div>
                )}
              </div>
              <div className="step-action">
                <button
                  className="btn btn-primary"
                  onClick={() => runStep('build-features', 'build_features')}
                  disabled={!hasData('download') || getStepStatus('build_features') === 'running'}
                >
                  {getStepStatus('build_features') === 'running' ? 'Running...' : hasData('build_features') ? 'Rebuild' : 'Build Features'}
                </button>
              </div>
            </div>

            {/* Step 3: Train (coming soon) */}
            <div className="pipeline-step" style={{ opacity: 0.5 }}>
              <div className="step-number idle">3</div>
              <div className="step-info">
                <div className="step-name">Train Model</div>
                <div className="step-desc">Train a feed-forward neural net for each stat type (passing yards, rushing yards, receptions, etc.)</div>
                <span className="status-badge idle">Coming Soon</span>
              </div>
              <div className="step-action">
                <button className="btn btn-primary" disabled>Train</button>
              </div>
            </div>

            {/* Step 4: Predict (coming soon) */}
            <div className="pipeline-step" style={{ opacity: 0.5 }}>
              <div className="step-number idle">4</div>
              <div className="step-info">
                <div className="step-name">Generate Predictions</div>
                <div className="step-desc">Predict player stat values for upcoming games and compare against posted lines</div>
                <span className="status-badge idle">Coming Soon</span>
              </div>
              <div className="step-action">
                <button className="btn btn-primary" disabled>Predict</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

function DataExplorerPage({ preview, previewDataset, loadPreview, hasData }) {
  const datasets = [
    { key: 'weekly', label: 'Weekly Stats', source: 'raw' },
    { key: 'schedules', label: 'Schedules', source: 'raw' },
    { key: 'snap_counts', label: 'Snap Counts', source: 'raw' },
    { key: 'ngs_passing', label: 'NGS Passing', source: 'raw' },
    { key: 'ngs_rushing', label: 'NGS Rushing', source: 'raw' },
    { key: 'ngs_receiving', label: 'NGS Receiving', source: 'raw' },
    { key: 'injuries', label: 'Injuries', source: 'raw' },
    { key: 'merged', label: 'Merged', source: 'processed' },
    { key: 'features', label: 'Features', source: 'processed' },
  ]

  return (
    <>
      <div className="header">
        <div className="header-title">
          <h1>Data Explorer</h1>
          <p>Browse raw and processed datasets</p>
        </div>
      </div>

      <div className="section-card">
        <div className="section-header">
          <span className="section-title">Select Dataset</span>
        </div>
        <div className="section-body">
          <div className="preview-controls">
            {datasets.map(d => (
              <button
                key={d.key}
                className={`preview-btn ${previewDataset === d.key ? 'active' : ''}`}
                onClick={() => loadPreview(d.key)}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {preview && !preview.error && (
        <div className="section-card">
          <div className="section-header">
            <span className="section-title">{previewDataset}</span>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              {preview.shape.rows.toLocaleString()} rows x {preview.shape.columns} columns
            </span>
          </div>
          <div className="section-body">
            <div className="data-table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    {preview.columns.slice(0, 12).map(col => (
                      <th key={col}>{col}</th>
                    ))}
                    {preview.columns.length > 12 && <th>...</th>}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview.map((row, i) => (
                    <tr key={i}>
                      {preview.columns.slice(0, 12).map(col => (
                        <td key={col}>{String(row[col] ?? '').substring(0, 20)}</td>
                      ))}
                      {preview.columns.length > 12 && <td style={{ color: 'var(--text-muted)' }}>+{preview.columns.length - 12} cols</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {preview?.error && (
        <div className="section-card">
          <div className="section-body" style={{ color: 'var(--warning)' }}>
            {preview.error} — run the pipeline first.
          </div>
        </div>
      )}
    </>
  )
}

function ComingSoonPage({ title, desc }) {
  return (
    <>
      <div className="header">
        <div className="header-title">
          <h1>{title}</h1>
          <p>{desc}</p>
        </div>
      </div>
      <div className="section-card">
        <div className="section-body" style={{ textAlign: 'center', padding: '60px 24px', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>{'\u2699'}</div>
          <div style={{ fontSize: '16px', fontWeight: 600 }}>Coming Soon</div>
          <div style={{ fontSize: '13px', marginTop: '4px' }}>This feature will be wired up in the next iteration.</div>
        </div>
      </div>
    </>
  )
}

export default App
