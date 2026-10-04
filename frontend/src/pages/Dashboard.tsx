import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { Activity, ArrowRight, ArrowUpRight, Check, Database, FileSpreadsheet, FlaskConical, Layers, Plus, ShieldCheck, Sparkles, UploadCloud } from '../icons'
import { api, date, label, number } from '../api'
import type { DashboardData, Dataset } from '../types'
import { useWorkspace } from '../workspace'
import { DistributionChart } from '../components/charts'
import { Empty, ErrorState, Loading, PageHeading, Panel, Status, ViewLink } from '../components/ui'
import { goalText } from '../presentation'

export default function Dashboard() {
  const { user, openAnalysis, openUpload } = useWorkspace()
  const navigate = useNavigate()
  const [datasetId, setDatasetId] = useState('')
  const [columnName, setColumnName] = useState('')
  const query = useQuery({ queryKey: ['dashboard'], queryFn: () => api<DashboardData>('/dashboard') })
  const selected = query.data?.datasets.find(dataset => dataset.id === datasetId) || query.data?.datasets[0]
  const detail = useQuery({ queryKey: ['dataset', selected?.id], queryFn: () => api<Dataset>(`/datasets/${selected!.id}`), enabled: !!selected })
  if (query.isLoading) return <Loading />
  if (query.error) return <ErrorState error={query.error} retry={() => query.refetch()} />

  const { stats, datasets, runs } = query.data!
  const profile = detail.data?.profile
  const distribution = profile?.columns.find(column => column.name === columnName)
    || profile?.columns.find(column => column.kind === 'numeric') || profile?.columns[0]
  const readyModel = runs.find(run => run.status === 'completed')
  const activeRun = runs.find(run => ['queued', 'running'].includes(run.status))
  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  return <>
    <PageHeading eyebrow="YOUR ANALYTICS WORKSPACE" title={`${greeting}, ${user.name.split(' ')[0]}`} text="A clear view of your data, your models, and what to do next." actions={<>
      <button className="button secondary" onClick={openUpload}><UploadCloud size={17} />Upload dataset</button>
      <button className="button primary" onClick={() => openAnalysis()}><Plus size={17} />Train a model</button>
    </>} />

    <section className="dashboard-hero">
      <div className="hero-copy">
        <span className="hero-eyebrow"><span className="hero-status-dot" />YOUR AUTONOMOUS DATA ANALYST</span>
        <h2>Your data.<br /><span>Clearly understood.</span></h2>
        <p>Explore the patterns. Build a prediction model.<br />Turn what you learn into your next decision.</p>
        <div className="hero-actions">
          <button className="button hero-primary" onClick={() => datasets.length ? openAnalysis(selected?.id) : openUpload()}>
            {datasets.length ? <Sparkles size={17} /> : <UploadCloud size={17} />}{datasets.length ? 'Build your next model' : 'Upload your first dataset'}<ArrowRight size={17} />
          </button>
          {datasets.length > 0 && <Link to={`/datasets/${selected!.id}`} className="hero-secondary">Explore your data<ArrowUpRight size={16} /></Link>}
        </div>
        <div className="hero-proof"><ShieldCheck size={14} /><span>Real calculations. Reproducible models. Useful answers.</span></div>
      </div>
      <div className="hero-journey">
        <div className="journey-heading"><span>FROM UPLOAD TO PREDICTION</span><span className="journey-pill">3 simple steps</span></div>
        {[
          { title: 'Bring your data', text: 'Upload a CSV or Excel spreadsheet.', icon: Database, done: datasets.length > 0 },
          { title: 'Understand the patterns', text: 'Explore charts, columns, and data quality.', icon: ChartIcon, done: false },
          { title: 'Build a useful model', text: 'Compare models and make predictions.', icon: FlaskConical, done: stats.models > 0 },
        ].map((step, index) => <div className="journey-step" key={step.title}>
          <span className={`journey-icon ${step.done ? 'done' : ''}`}>{step.done ? <Check size={18} /> : <step.icon size={19} />}</span>
          <div><small>STEP 0{index + 1}</small><strong>{step.title}</strong><p>{step.text}</p></div>
          <span className="journey-number">0{index + 1}</span>
        </div>)}
        <div className="journey-footer"><Sparkles size={14} />You choose the goal. We handle the workflow.</div>
      </div>
    </section>

    <div className="stats-grid">
      {[
        { name: 'Datasets', value: stats.datasets, icon: Database, description: 'Files in your workspace', color: 'purple' },
        { name: 'Total records', value: stats.rows, icon: Layers, description: 'Rows available to explore', color: 'blue' },
        { name: 'Trained models', value: stats.models, icon: FlaskConical, description: `${stats.active_runs} currently in training`, color: 'green' },
        { name: 'Average data quality', value: datasets.length ? `${stats.average_quality}%` : '—', icon: ShieldCheck, description: 'Completeness minus duplicates', color: 'amber' },
      ].map(stat => <div className="stat-card" key={stat.name}><div className="stat-card-top"><span>{stat.name}</span><span className={`stat-icon ${stat.color}`}><stat.icon size={19} /></span></div><strong>{typeof stat.value === 'number' ? number(stat.value) : stat.value}</strong><small>{stat.description}</small></div>)}
    </div>

    {activeRun && <Link to={`/runs/${activeRun.id}`} className="active-run-banner">
      <span className="active-run-icon"><Activity size={20} /></span>
      <div><strong>Your model is being built</strong><p>{goalText(activeRun)} · {activeRun.stage}</p></div>
      <div className="active-run-progress"><span>{activeRun.progress}%</span><div className="progress-track"><i style={{ width: `${activeRun.progress}%` }} /></div></div>
      <ArrowRight size={18} />
    </Link>}

    <div className="dashboard-grid overview-grid">
      <Panel title="A quick look at your data" subtitle="See how values are distributed in a selected column." action={datasets.length > 0 && <select className="compact-select dataset-select" aria-label="Dashboard dataset" value={selected?.id || ''} onChange={event => { setDatasetId(event.target.value); setColumnName('') }}>{datasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select>}>
        {selected ? detail.isLoading ? <Loading text="Preparing the chart…" /> : detail.error ? <ErrorState error={detail.error} retry={() => detail.refetch()} /> : distribution && <>
          <div className="chart-toolbar"><div><span className="chart-feature-name">{label(distribution.name)}</span><small>{number(profile!.sampled_rows)} sampled rows</small></div><select className="compact-select" aria-label="Dashboard chart column" value={distribution.name} onChange={event => setColumnName(event.target.value)}>{profile!.columns.map(column => <option key={column.name} value={column.name}>{label(column.name)}</option>)}</select></div>
          <DistributionChart data={distribution.distribution} />
          <div className="panel-bottom-note"><span><i className="legend-dot" />Number of records</span><ViewLink to={`/datasets/${selected.id}`}>Open full explorer</ViewLink></div>
        </> : <Empty title="Your first chart starts with a file" text="Upload a dataset to see its distributions and relationships." action={<button className="button secondary" onClick={openUpload}>Choose a file<ArrowRight size={16} /></button>} />}
      </Panel>
      <Panel title="What would you like to do?" subtitle="Pick a starting point. We’ll guide you from there." className="quick-actions-panel">
        {[
          { icon: UploadCloud, title: 'Upload a dataset', text: 'Add a CSV or Excel spreadsheet.', action: openUpload, color: 'purple' },
          { icon: SearchIcon, title: 'Explore your data', text: 'View charts and check data quality.', action: () => selected ? navigate(`/datasets/${selected.id}`) : openUpload(), color: 'blue' },
          { icon: FlaskConical, title: 'Train a prediction model', text: 'Choose a target and compare models.', action: () => openAnalysis(selected?.id), color: 'green' },
        ].map(action => <button className="quick-action" key={action.title} onClick={action.action}><span className={`stat-icon ${action.color}`}><action.icon size={21} /></span><span><strong>{action.title}</strong><small>{action.text}</small></span><ArrowUpRight size={17} /></button>)}
        <div className="next-step-card"><span className="eyebrow">RECOMMENDED NEXT STEP</span><strong>{readyModel ? 'Put your trained model to work' : datasets.length ? 'Choose what you want to predict' : 'Start with a dataset'}</strong><p>{readyModel ? 'Try a new record and see your model’s prediction.' : datasets.length ? 'For example: customer churn, a price, or sales revenue.' : 'Upload your own spreadsheet or add a sample from the datasets page.'}</p><button className="text-link" onClick={() => readyModel ? navigate(`/runs/${readyModel.id}`) : datasets.length ? openAnalysis(selected?.id) : openUpload()}>{readyModel ? 'Open model results' : datasets.length ? 'Set up a model' : 'Upload dataset'}<ArrowRight size={15} /></button></div>
      </Panel>
    </div>

    <div className="dashboard-grid lower">
      <Panel title="Recent datasets" subtitle="Your files, ready to explore." action={<ViewLink to="/datasets">View all datasets</ViewLink>}>
        {datasets.length ? <div className="dataset-mini-list">{datasets.slice(0, 4).map(dataset => <Link key={dataset.id} to={`/datasets/${dataset.id}`}><span className="dataset-file-icon"><FileSpreadsheet size={21} /></span><div><strong>{dataset.name}</strong><small>{number(dataset.row_count)} rows · {dataset.column_count} columns</small></div><span className="quality-mini">{dataset.quality_score}%<small>quality</small></span><ArrowUpRight size={17} /></Link>)}</div> : <Empty title="No datasets yet" text="Your uploaded files will appear here." />}
      </Panel>
      <Panel title="Recent model builds" subtitle="From training to a ready-to-use solution." action={<ViewLink to="/analyses">View history</ViewLink>}>
        {runs.length ? <div className="activity-list">{runs.slice(0, 4).map(run => <Link to={`/runs/${run.id}`} key={run.id}><span className={`activity-symbol ${run.status}`}><FlaskConical size={17} /></span><div><strong>{goalText(run)}</strong><small>{run.result?.model_name || run.stage} · {date(run.created_at)}</small></div><Status status={run.status} /></Link>)}</div> : <div className="activity-empty"><span><FlaskConical size={26} /></span><h4>Ready when you are</h4><p>Your model builds will appear here, with live progress and final results.</p><button className="text-link" onClick={() => openAnalysis()}>Train your first model<ArrowRight size={15} /></button></div>}
      </Panel>
    </div>
  </>
}

function ChartIcon({ size }: { size?: number }) { return <Activity size={size} /> }
function SearchIcon({ size }: { size?: number }) { return <Database size={size} /> }
