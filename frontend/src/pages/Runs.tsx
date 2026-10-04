import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Activity, ArrowRight, ArrowUpRight, Clock3, FlaskConical, Plus, Search, Sparkles } from '../icons'
import { api, date, label, metric } from '../api'
import type { Dataset, Run } from '../types'
import { goalText, metrics, statusText, taskText } from '../presentation'
import { useWorkspace } from '../workspace'
import { Empty, ErrorState, Loading, PageHeading, Panel, Status } from '../components/ui'

export default function Runs({ modelsOnly = false }: { modelsOnly?: boolean }) {
  const { openAnalysis } = useWorkspace()
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const query = useQuery({ queryKey: ['runs'], queryFn: () => api<Run[]>('/analysis-runs'), refetchInterval: value => value.state.data?.some(run => ['running', 'queued'].includes(run.status)) ? 2500 : false })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  const data = query.data?.filter(run => (!modelsOnly || run.status === 'completed') && (filter === 'all' || run.status === filter) && `${run.objective} ${run.target} ${run.result?.model_name || ''}`.toLowerCase().includes(search.toLowerCase())) || []

  return <>
    <PageHeading eyebrow={modelsOnly ? 'STEP 4 · MAKE PREDICTIONS' : 'STEP 3 · MODEL TRAINING'} title={modelsOnly ? 'Your trained models.' : 'Every model build, in one view.'} text={modelsOnly ? 'Open a model to understand its scores, try predictions, or download the solution.' : 'Track live progress and revisit the results of previous training runs.'} actions={<button className="button primary" onClick={() => openAnalysis()}><Plus size={18} />Train a model</button>} />
    <div className="page-guide"><span className="guide-icon"><FlaskConical size={21} /></span><div><strong>{modelsOnly ? 'Each model includes a ready-to-use prediction form.' : 'We prepare, compare, test, and explain — automatically.'}</strong><p>{modelsOnly ? 'The same saved preprocessing pipeline is used for every new prediction.' : 'Open a build to see its timeline. Completed builds become available in Models & predictions.'}</p></div></div>
    <div className="collection-toolbar"><div className="search-input"><Search size={18} /><input placeholder={modelsOnly ? 'Search models or prediction targets…' : 'Search training runs…'} aria-label="Search analyses" value={search} onChange={event => setSearch(event.target.value)} /></div><div className="toolbar-right"><span className="collection-count">{data.length} {modelsOnly ? 'models' : 'builds'}</span>{!modelsOnly && <select className="compact-select" aria-label="Filter status" value={filter} onChange={event => setFilter(event.target.value)}>{['all', 'completed', 'running', 'queued', 'failed', 'cancelled'].map(status => <option key={status} value={status}>{status === 'all' ? 'All statuses' : statusText(status)}</option>)}</select>}</div></div>
    {query.isLoading ? <Loading /> : query.error ? <ErrorState error={query.error} retry={() => query.refetch()} /> : !data.length ? <Panel><Empty title={search || filter !== 'all' ? 'No matching model builds' : modelsOnly ? 'Your first trained model will appear here' : 'Let’s build your first model'} text={search || filter !== 'all' ? 'Try a different search or status filter.' : 'Choose a dataset and a prediction target. We’ll take care of the training workflow.'} action={<button className="button primary" onClick={() => openAnalysis()}><Sparkles size={17} />Train a model</button>} /></Panel> : modelsOnly ?
      <div className="model-grid">{data.map(run => {
        const result = run.result!
        return <Link className="model-card" key={run.id} to={`/runs/${run.id}`}>
          <div className="model-card-header"><span className="model-icon"><FlaskConical size={25} /></span><Status status="completed" /><ArrowUpRight size={18} /></div>
          <span className="eyebrow">{run.task === 'clustering' ? 'DISCOVERED GROUPS' : 'PREDICTING'}</span><h3>{run.task === 'clustering' ? `${result.metrics.clusters} similar groups` : label(run.target)}</h3><p>{result.model_name} · {taskText(run.task)}</p>
          <div className="model-score"><span title={metrics[result.primary_metric]?.explanation}>{metrics[result.primary_metric]?.title || label(result.primary_metric)}<small>Score on unseen test data</small></span><strong>{metric(result.primary_metric, result.metrics[result.primary_metric])}</strong></div>
          <div className="model-card-footer"><span>{datasets.data?.find(dataset => dataset.id === run.dataset_id)?.name || 'Dataset'}</span><span><Clock3 size={14} />{date(run.created_at)}</span></div>
          <div className="model-card-cta">View results & try predictions<ArrowRight size={16} /></div>
        </Link>
      })}</div> :
      <Panel><div className="table-scroll"><table className="data-table runs-table"><thead><tr><th>Prediction goal</th><th>Dataset</th><th>Type</th><th>Status</th><th>Started</th><th /></tr></thead><tbody>{data.map(run => <tr key={run.id}>
        <td><Link className="run-title" to={`/runs/${run.id}`}><span className={`run-table-icon ${run.status}`}><Activity size={18} /></span><div><strong>{goalText(run)}</strong><small>{run.objective}</small></div></Link></td>
        <td>{datasets.data?.find(dataset => dataset.id === run.dataset_id)?.name || 'Dataset'}</td><td><span className="type-badge">{taskText(run.task)}</span></td>
        <td><Status status={run.status} />{run.status === 'running' && <div className="mini-progress"><i style={{ width: `${run.progress}%` }} /></div>}</td>
        <td className="muted">{date(run.created_at)}</td><td><Link to={`/runs/${run.id}`} className="icon-button" aria-label={`View ${run.target} model build`}><ArrowUpRight size={18} /></Link></td>
      </tr>)}</tbody></table></div></Panel>}
  </>
}
