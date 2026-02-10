import { useState, useEffect, useRef } from 'react'
import './App.css'

const API = 'http://localhost:8001/api'

function App() {
  const [page, setPage] = useState('dashboard')
  const [status, setStatus] = useState(null)
  const [stats, setStats] = useState(null)
  const [logs, setLogs] = useState({ download: '', build_features: '', train: '' })
  const [showLogs, setShowLogs] = useState({ download: false, build_features: false, train: false })
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

        <div className="sidebar-status">
          <div className="sidebar-label">Status</div>
          <div style={{ padding: '0 12px', fontSize: '12px', color: 'var(--text-sidebar)' }}>
            <span className={`sidebar-status-dot ${status ? 'connected' : 'disconnected'}`} />
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
            setPage={setPage}
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
        {page === 'training' && (
          <TrainingPage
            logs={logs}
            showLogs={showLogs}
            runStep={runStep}
            streamLogs={streamLogs}
            hasFeatures={hasData('build_features')}
          />
        )}
        {page === 'predict' && <PredictionsPage />}
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
          <div className="pipeline-status-flow">
            <StepStatusChip label="Download Data" status={status?.steps?.download?.status || 'idle'} hasData={status?.steps?.download?.has_data} />
            <span className="pipeline-status-arrow">{'\u2192'}</span>
            <StepStatusChip label="Build Features" status={status?.steps?.build_features?.status || 'idle'} hasData={status?.steps?.build_features?.has_data} />
            <span className="pipeline-status-arrow">{'\u2192'}</span>
            <StepStatusChip label="Train Model" status="idle" hasData={false} />
            <span className="pipeline-status-arrow">{'\u2192'}</span>
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

function PipelinePage({ status, logs, showLogs, runStep, getStepStatus, getStepFiles, hasData, statusLabel, statusIcon, setPage }) {
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

            {/* Step 3: Train */}
            <div className="pipeline-step">
              <div className={`step-number ${hasData('build_features') ? 'idle' : 'idle'}`}>3</div>
              <div className="step-info">
                <div className="step-name">Train Model</div>
                <div className="step-desc">Train a feed-forward neural net for each stat type (passing yards, rushing yards, receptions, etc.)</div>
              </div>
              <div className="step-action">
                <button className="btn btn-primary" onClick={() => setPage('training')} disabled={!hasData('build_features')}>
                  Go to Training
                </button>
              </div>
            </div>

            {/* Step 4: Predict */}
            <div className="pipeline-step">
              <div className="step-number idle">4</div>
              <div className="step-info">
                <div className="step-name">Generate Predictions</div>
                <div className="step-desc">Predict player stat values for upcoming games and compare against posted lines</div>
              </div>
              <div className="step-action">
                <button className="btn btn-primary" onClick={() => setPage('predict')}>
                  Go to Predictions
                </button>
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

const TARGET_STATS = [
  'passing_yards', 'passing_tds', 'rushing_yards', 'carries',
  'receptions', 'receiving_yards', 'receiving_tds',
]

const STAT_LABELS = {
  passing_yards: 'Passing Yards',
  passing_tds: 'Passing TDs',
  rushing_yards: 'Rushing Yards',
  carries: 'Carries',
  receptions: 'Receptions',
  receiving_yards: 'Receiving Yards',
  receiving_tds: 'Receiving TDs',
}

function TrainingPage({ logs, showLogs, runStep, streamLogs, hasFeatures }) {
  const [models, setModels] = useState([])
  const [selectedStat, setSelectedStat] = useState(TARGET_STATS[0])
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [hyperparams, setHyperparams] = useState({
    learning_rate: '',
    hidden_sizes: '',
    dropout: '',
    patience: '',
  })
  const [training, setTraining] = useState(false)
  const [trainError, setTrainError] = useState('')

  const fetchModels = () => {
    fetch(`${API}/models`).then(r => r.json()).then(d => setModels(d.models || [])).catch(() => {})
  }

  useEffect(() => {
    fetchModels()
    const interval = setInterval(fetchModels, 5000)
    return () => clearInterval(interval)
  }, [])

  const startTraining = async () => {
    setTrainError('')
    setTraining(true)

    const body = { stat: selectedStat }
    if (hyperparams.learning_rate) body.learning_rate = parseFloat(hyperparams.learning_rate)
    if (hyperparams.hidden_sizes) {
      try {
        body.hidden_sizes = hyperparams.hidden_sizes.split(',').map(s => parseInt(s.trim()))
      } catch { /* ignore parse errors */ }
    }
    if (hyperparams.dropout) body.dropout = parseFloat(hyperparams.dropout)
    if (hyperparams.patience) body.patience = parseInt(hyperparams.patience)

    try {
      const res = await fetch(`${API}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (data.error) {
        setTrainError(data.error)
        setTraining(false)
        return
      }
      // Stream logs
      setTimeout(() => streamLogs('train'), 500)

      // Poll for completion
      const pollInterval = setInterval(() => {
        fetch(`${API}/status`).then(r => r.json()).catch(() => null)
        // Check training state via logs endpoint status events
      }, 2000)

      // Use an event source to detect when done
      const es = new EventSource(`${API}/logs/train`)
      es.addEventListener('status', () => {
        es.close()
        setTraining(false)
        fetchModels()
        clearInterval(pollInterval)
      })
      es.onerror = () => {
        es.close()
        setTraining(false)
        clearInterval(pollInterval)
      }
    } catch {
      setTrainError('Failed to connect to backend')
      setTraining(false)
    }
  }

  return (
    <>
      <div className="header">
        <div className="header-title">
          <h1>Training</h1>
          <p>Train and manage neural network models</p>
        </div>
      </div>

      {/* Model Cards Grid */}
      {models.length > 0 && (
        <div className="section-card">
          <div className="section-header">
            <span className="section-title">Trained Models</span>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              {models.length} model{models.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="section-body">
            <div className="model-cards-grid">
              {models.map(m => (
                <div key={m.stat} className="model-card">
                  <div className="model-card-header">
                    <span className="model-card-stat">{STAT_LABELS[m.stat] || m.stat}</span>
                    <span className="status-badge done">{'\u2713'} Trained</span>
                  </div>
                  <div className="model-card-metrics">
                    <div className="model-metric">
                      <span className="model-metric-label">Test MAE</span>
                      <span className="model-metric-value">{m.test_mae != null ? m.test_mae.toFixed(1) : '—'}</span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">Improvement</span>
                      <span className="model-metric-value" style={{ color: 'var(--success)' }}>
                        {m.improvement_pct != null ? `${parseFloat(m.improvement_pct).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">Epochs</span>
                      <span className="model-metric-value">{m.epochs_trained ?? '—'}</span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">Features</span>
                      <span className="model-metric-value">{m.n_features ?? '—'}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Train New / Retrain */}
      <div className="section-card">
        <div className="section-header">
          <span className="section-title">Train Model</span>
        </div>
        <div className="section-body">
          {!hasFeatures && (
            <div className="error-banner">
              Features not built yet. Go to Data Pipeline and run the pipeline first.
            </div>
          )}

          {trainError && <div className="error-banner">{trainError}</div>}

          <div className="form-row">
            <label className="form-label">Stat to Train</label>
            <select
              className="form-select"
              value={selectedStat}
              onChange={e => setSelectedStat(e.target.value)}
              disabled={training}
            >
              {TARGET_STATS.map(s => (
                <option key={s} value={s}>{STAT_LABELS[s] || s}</option>
              ))}
            </select>
          </div>

          <div className="form-row">
            <button
              className="collapsible-toggle"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? '\u25BC' : '\u25B6'} Advanced Settings
            </button>
          </div>

          {showAdvanced && (
            <div className="advanced-settings">
              <div className="form-grid">
                <div className="form-row">
                  <label className="form-label">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    className="form-input"
                    placeholder="0.001"
                    value={hyperparams.learning_rate}
                    onChange={e => setHyperparams(p => ({ ...p, learning_rate: e.target.value }))}
                    disabled={training}
                  />
                </div>
                <div className="form-row">
                  <label className="form-label">Hidden Sizes</label>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="128, 64, 32"
                    value={hyperparams.hidden_sizes}
                    onChange={e => setHyperparams(p => ({ ...p, hidden_sizes: e.target.value }))}
                    disabled={training}
                  />
                </div>
                <div className="form-row">
                  <label className="form-label">Dropout</label>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="0.9"
                    className="form-input"
                    placeholder="0.3"
                    value={hyperparams.dropout}
                    onChange={e => setHyperparams(p => ({ ...p, dropout: e.target.value }))}
                    disabled={training}
                  />
                </div>
                <div className="form-row">
                  <label className="form-label">Patience</label>
                  <input
                    type="number"
                    className="form-input"
                    placeholder="10"
                    value={hyperparams.patience}
                    onChange={e => setHyperparams(p => ({ ...p, patience: e.target.value }))}
                    disabled={training}
                  />
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: '16px' }}>
            <button
              className="btn btn-primary"
              onClick={startTraining}
              disabled={training || !hasFeatures}
            >
              {training ? 'Training...' : models.find(m => m.stat === selectedStat) ? 'Retrain Model' : 'Train Model'}
            </button>
            {training && <span className="spinner" style={{ marginLeft: '12px' }} />}
          </div>

          {showLogs.train && logs.train && (
            <div className="log-viewer">
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{logs.train}</pre>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

function PredictionsPage() {
  const [models, setModels] = useState([])
  const [selectedStat, setSelectedStat] = useState('')
  const [games, setGames] = useState([])
  const [selectedGame, setSelectedGame] = useState('')
  const [gameContext, setGameContext] = useState({
    home_team: '', away_team: '', spread: 0, total_line: 0,
    is_dome: 0, temp: 72, wind: 0,
  })
  const [predictions, setPredictions] = useState([])
  const [props, setProps] = useState({})
  const [bookmaker, setBookmaker] = useState('')
  const [predError, setPredError] = useState('')
  const [loading, setLoading] = useState(false)

  // Fetch models on mount
  useEffect(() => {
    fetch(`${API}/models`).then(r => r.json()).then(d => {
      const m = d.models || []
      setModels(m)
      if (m.length > 0) setSelectedStat(m[0].stat)
    }).catch(() => {})
  }, [])

  // Fetch schedule games
  useEffect(() => {
    fetch(`${API}/schedule-games`).then(r => r.json()).then(d => {
      setGames(d.games || [])
    }).catch(() => {})
  }, [])

  const handleGameSelect = (idx) => {
    setSelectedGame(idx)
    if (idx === '') return
    const game = games[parseInt(idx)]
    if (!game) return
    setGameContext({
      home_team: game.home_team,
      away_team: game.away_team,
      spread: game.spread_line,
      total_line: game.total_line,
      is_dome: game.roof === 'dome' || game.roof === 'closed' ? 1 : 0,
      temp: game.temp,
      wind: game.wind,
    })
  }

  // Auto-run batch prediction + fetch sportsbook props when stat + both teams are set
  useEffect(() => {
    if (!selectedStat || !gameContext.home_team || !gameContext.away_team) return
    setPredError('')
    setPredictions([])
    setProps({})
    setBookmaker('')
    setLoading(true)

    // Fetch predictions and props in parallel
    const predFetch = fetch(`${API}/predict-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        stat: selectedStat,
        home_team: gameContext.home_team,
        away_team: gameContext.away_team,
        spread: gameContext.spread,
        total_line: gameContext.total_line,
        is_dome: gameContext.is_dome,
        temp: gameContext.temp,
        wind: gameContext.wind,
      }),
    }).then(r => r.json())

    const propsFetch = fetch(
      `${API}/props?stat=${selectedStat}&home_team=${gameContext.home_team}&away_team=${gameContext.away_team}`
    ).then(r => r.json()).catch(() => ({ props: [] }))

    Promise.all([predFetch, propsFetch]).then(([predData, propsData]) => {
      if (predData.error) {
        setPredError(predData.error)
      } else {
        setPredictions(predData.predictions || [])
      }
      // Build props lookup by player name (normalized for matching)
      const propsMap = {}
      for (const p of (propsData.props || [])) {
        propsMap[p.player_name.toLowerCase()] = p
      }
      setProps(propsMap)
      setBookmaker(propsData.bookmaker || '')
      setLoading(false)
    }).catch(() => {
      setPredError('Failed to connect to backend')
      setLoading(false)
    })
  }, [selectedStat, gameContext])

  // Match a prediction player name to a props player name
  // Our data: "X.Worthy", Odds API: "Xavier Worthy"
  const findProp = (playerName) => {
    if (!playerName) return null
    const lower = playerName.toLowerCase()
    if (props[lower]) return props[lower]
    const parts = lower.split(/[.\s]+/)
    const lastName = parts[parts.length - 1]
    for (const [key, val] of Object.entries(props)) {
      if (key.endsWith(lastName) && key[0] === lower[0]) return val
    }
    return null
  }

  // Convert American odds to implied probability (0-1)
  const oddsToProb = (odds) => {
    if (odds < 0) return (-odds) / (-odds + 100)
    return 100 / (odds + 100)
  }

  // Confidence score (0-100) using edge, DK odds, and player std
  const calcConfidence = (pred, prop, rollingStd) => {
    if (!prop || prop.line == null) return null
    const edge = pred - prop.line
    const isOver = edge > 0

    // 1. Edge z-score: how many player-std is the edge?
    const std = rollingStd && rollingStd > 0 ? rollingStd : prop.line * 0.3
    const zScore = Math.abs(edge) / std

    // 2. Market alignment: does the DK line agree with our lean?
    const relevantOdds = isOver ? (prop.over_odds || -110) : (prop.under_odds || -110)
    const marketProb = oddsToProb(relevantOdds)

    // 3. Combine — z-score drives magnitude, market alignment modulates
    // z=0 → 0, z=1 → ~0.39, z=2 → ~0.63, z=3 → ~0.78
    const zComponent = 1 - Math.exp(-0.5 * zScore)
    const raw = (zComponent * 0.65 + marketProb * 0.35) * 100
    return Math.min(99, Math.max(1, Math.round(raw)))
  }

  const homePlayers = predictions.filter(p => p.team === gameContext.home_team)
  const awayPlayers = predictions.filter(p => p.team === gameContext.away_team)
  const hasProps = Object.keys(props).length > 0

  return (
    <>
      <div className="header">
        <div className="header-title">
          <h1>Predictions</h1>
          <p>Generate player prop predictions for upcoming games</p>
        </div>
      </div>

      {predError && <div className="error-banner" style={{ marginBottom: '20px' }}>{predError}</div>}

      {/* Controls */}
      <div className="section-card">
        <div className="section-body">
          <div className="form-grid">
            <div className="form-row">
              <label className="form-label">Model</label>
              {models.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No trained models. Train one first.</div>
              ) : (
                <select
                  className="form-select"
                  value={selectedStat}
                  onChange={e => setSelectedStat(e.target.value)}
                >
                  {models.map(m => (
                    <option key={m.stat} value={m.stat}>
                      {STAT_LABELS[m.stat] || m.stat} (MAE: {m.test_mae?.toFixed(1)})
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="form-row">
              <label className="form-label">Game</label>
              <select className="form-select" value={selectedGame} onChange={e => handleGameSelect(e.target.value)}>
                <option value="">-- Select Game --</option>
                {games.map((g, i) => (
                  <option key={i} value={i}>
                    {g.game_type} Wk{g.week}: {g.away_team} @ {g.home_team}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {gameContext.home_team && (
            <div className="form-grid" style={{ marginTop: '12px' }}>
              <div className="form-row">
                <label className="form-label">Spread (home)</label>
                <input type="number" step="0.5" className="form-input" value={gameContext.spread}
                  onChange={e => setGameContext(p => ({ ...p, spread: parseFloat(e.target.value) || 0 }))} />
              </div>
              <div className="form-row">
                <label className="form-label">Total Line</label>
                <input type="number" step="0.5" className="form-input" value={gameContext.total_line}
                  onChange={e => setGameContext(p => ({ ...p, total_line: parseFloat(e.target.value) || 0 }))} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="section-card">
          <div className="section-body" style={{ textAlign: 'center', padding: '40px' }}>
            <span className="spinner" style={{ width: '24px', height: '24px' }} />
            <div style={{ marginTop: '12px', color: 'var(--text-secondary)', fontSize: '14px' }}>
              Running predictions for all starters...
            </div>
          </div>
        </div>
      )}

      {/* Results — split by team */}
      {!loading && predictions.length > 0 && (
        <>
          {bookmaker && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
              Lines from: <strong style={{ color: 'var(--text-secondary)' }}>{bookmaker}</strong>
            </div>
          )}
          {[
            { team: gameContext.home_team, players: homePlayers, label: 'Home' },
            { team: gameContext.away_team, players: awayPlayers, label: 'Away' },
          ].map(({ team, players: teamPreds, label }) => (
            <div className="section-card" key={team}>
              <div className="section-header">
                <span className="section-title">{team} ({label})</span>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  {teamPreds.length} player{teamPreds.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="section-body" style={{ padding: 0 }}>
                <div className="data-table-wrapper" style={{ border: 'none', margin: 0 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Player</th>
                        <th>Pos</th>
                        <th>Predicted</th>
                        {hasProps && <th>Line</th>}
                        {hasProps && <th>Edge</th>}
                        {hasProps && <th>Confidence</th>}
                        <th>3-Game Avg</th>
                      </tr>
                    </thead>
                    <tbody>
                      {teamPreds.map(p => {
                        const prop = findProp(p.player_name)
                        const edge = prop ? (p.predicted_value - prop.line) : null
                        const confidence = prop ? calcConfidence(p.predicted_value, prop, p.rolling_std) : null
                        return (
                          <tr key={p.player_name}>
                            <td style={{ fontWeight: 600 }}>{p.player_name}</td>
                            <td>{p.position}</td>
                            <td>
                              <span className="prediction-inline-value">{p.predicted_value}</span>
                            </td>
                            {hasProps && (
                              <td style={{ color: 'var(--text-secondary)' }}>
                                {prop ? prop.line : '—'}
                              </td>
                            )}
                            {hasProps && (
                              <td>
                                {edge != null ? (
                                  <span className={`edge-value ${edge > 0 ? 'over' : 'under'}`}>
                                    {edge > 0 ? '+' : ''}{edge.toFixed(1)}
                                  </span>
                                ) : '—'}
                              </td>
                            )}
                            {hasProps && (
                              <td>
                                {confidence != null ? (
                                  <span className={`confidence-badge ${confidence >= 70 ? 'high' : confidence >= 40 ? 'mid' : 'low'}`}>
                                    {confidence}
                                  </span>
                                ) : '—'}
                              </td>
                            )}
                            <td style={{ color: 'var(--text-secondary)' }}>{p.rolling_avg ?? '—'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </>
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
