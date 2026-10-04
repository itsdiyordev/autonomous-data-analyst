import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { api, date, metric } from '../api'
import type { Dataset, ExperimentGroup } from '../types'
import { useWorkspace } from '../workspace'
import { Empty, ErrorState, Loading, PageHeading, Panel, Status } from '../components/ui'

export default function Experiments() {
  const { openAutonomous } = useWorkspace()
  const [selected, setSelected] = useState('')
  const [name, setName] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: ['experiments'], queryFn: () => api<ExperimentGroup[]>('/experiments') })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  const create = useMutation({ mutationFn: () => api<ExperimentGroup>('/experiments', { method: 'POST', body: JSON.stringify({ name, dataset_id: datasetId }) }), onSuccess: experiment => { queryClient.invalidateQueries({ queryKey: ['experiments'] }); setSelected(experiment.id); setName(''); toast.success('Experiment created') }, onError: (error: Error) => toast.error(error.message) })
  const experiment = query.data?.find(item => item.id === selected) || query.data?.[0]
  return <><PageHeading eyebrow="REPRODUCIBLE INVESTIGATIONS" title="Experiments, with context." text="Group related runs on the same immutable dataset. Compare development scores and revisit every decision." /><Panel title="Create an experiment"><form onSubmit={event => { event.preventDefault(); create.mutate() }}><div className="form-grid"><label className="field">Experiment name<input required minLength={2} maxLength={150} value={name} onChange={event => setName(event.target.value)} placeholder="Customer retention investigation" /></label><label className="field">Dataset<select required value={datasetId} onChange={event => setDatasetId(event.target.value)}><option value="">Choose dataset</option>{datasets.data?.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label></div><button className="button secondary" disabled={create.isPending}>Create experiment</button></form></Panel>
    {query.isLoading ? <Loading /> : query.error ? <ErrorState error={query.error} /> : !query.data?.length ? <Empty title="Your first experiment starts with an objective" text="Autonomous analyses create an experiment automatically. Re-runs remain in the same experiment." action={<button className="button primary" onClick={() => openAutonomous()}>Analyze a dataset</button>} /> : <Panel title={experiment!.name} subtitle={experiment!.selection_note} action={<select className="compact-select" aria-label="Select experiment" value={experiment!.id} onChange={event => setSelected(event.target.value)}>{query.data.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select>}><button className="button primary" onClick={() => openAutonomous(experiment!.dataset_id, experiment!.id)}>Add analytical run</button><div className="table-scroll"><table className="data-table"><thead><tr><th>Run</th><th>Status</th><th>Model</th><th>CV mean ± SD</th><th>Test score</th><th>Duration</th><th>Dataset hash</th></tr></thead><tbody>{experiment!.runs.map(run => <tr key={run.id}><td className="wrapping-cell"><Link className="text-link" to={run.config.analysis_mode === 'autonomous' ? `/analysis/${run.id}` : `/runs/${run.id}`}>{run.objective}</Link><small className="cell-secondary">{date(run.created_at)}{experiment!.best_run_id === run.id ? ' · BEST DEVELOPMENT SCORE' : ''}</small></td><td><Status status={run.status} /></td><td>{run.result?.model_name || 'Descriptive investigation'}</td><td>{run.result?.cross_validation?.mean !== undefined ? `${run.result.cross_validation.mean.toFixed(3)} ± ${run.result.cross_validation.std?.toFixed(3)}` : 'Not applicable'}</td><td>{run.result?.primary_metric ? metric(run.result.primary_metric, run.result.metrics[run.result.primary_metric]) : 'Not applicable'}</td><td>{run.result?.duration_seconds?.toFixed(1) || '—'}s</td><td><code>{run.result?.dataset_fingerprint?.slice(0, 12) || 'Pending'}</code></td></tr>)}</tbody></table></div></Panel>}
  </>
}
