import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { ArrowLeft, ArrowRight, LoaderCircle, Sparkles } from '../icons'
import { api, label } from '../api'
import type { AnalysisPlan, Dataset, Run } from '../types'
import { Empty, ErrorState, Loading, Modal } from './ui'
import { PlanView } from './AnalyticalEvidence'

export default function AutonomousModal({ initialDataset, experimentId, close }: { initialDataset?: string; experimentId?: string; close: () => void }) {
  const [step, setStep] = useState(0)
  const [datasetId, setDatasetId] = useState(initialDataset || '')
  const [objective, setObjective] = useState('Understand the main patterns, relationships and data quality.')
  const [target, setTarget] = useState('')
  const [budget, setBudget] = useState(180)
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  useEffect(() => { if (!datasetId && datasets.data?.length) setDatasetId(datasets.data[0].id) }, [datasetId, datasets.data])
  const detail = useQuery({ queryKey: ['dataset', datasetId], queryFn: () => api<Dataset>(`/datasets/${datasetId}`), enabled: !!datasetId })
  const request = { dataset_id: datasetId, objective, target: target || null, task: 'auto', analysis_mode: 'autonomous', budget_seconds: budget, experiment_id: experimentId || null }
  const plan = useQuery({ queryKey: ['plan', datasetId, objective, target], queryFn: () => api<AnalysisPlan>('/analysis-plans', { method: 'POST', body: JSON.stringify(request) }), enabled: step === 2 && !!datasetId && objective.length >= 8, retry: false })
  const create = useMutation({
    mutationFn: () => api<Run>('/analysis-runs', { method: 'POST', body: JSON.stringify(request) }),
    onSuccess: run => { queryClient.invalidateQueries({ queryKey: ['runs'] }); queryClient.invalidateQueries({ queryKey: ['dashboard'] }); queryClient.invalidateQueries({ queryKey: ['experiments'] }); close(); toast.success('Autonomous analysis started'); navigate(`/analysis/${run.id}`) },
    onError: (error: Error) => toast.error(error.message),
  })
  return <Modal title="Ask your autonomous analyst" subtitle="Describe the question. Inspect the plan. Follow the evidence." close={close} wide>
    {datasets.isLoading ? <Loading /> : datasets.error ? <ErrorState error={datasets.error} /> : !datasets.data?.length ? <Empty title="Add a dataset first" text="Upload a spreadsheet from Datasets to start an analytical investigation." /> : <form onSubmit={event => { event.preventDefault(); if (step < 2) setStep(step + 1); else create.mutate() }}>
      <ol className="wizard-steps">{['Choose your data', 'Analytical objective', 'Inspect the plan'].map((title, index) => <li key={title} className={index === step ? 'current' : index < step ? 'done' : ''}><span>{index + 1}</span><strong>{title}</strong></li>)}</ol>
      {step === 0 && <div className="wizard-page"><div className="wizard-heading"><h3>What should we investigate?</h3><p>The original dataset stays intact. Each decision and finding gets an evidence record.</p></div><label className="field">Dataset<select value={datasetId} onChange={event => { setDatasetId(event.target.value); setTarget('') }} required disabled={!!experimentId}>{datasets.data.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>{detail.data && <div className="info-note">{detail.data.row_count.toLocaleString()} rows · {detail.data.column_count} columns · {detail.data.quality_score}/100 quality</div>}</div>}
      {step === 1 && <div className="wizard-page"><div className="wizard-heading"><h3>What would you like to understand?</h3><p>Descriptive questions use EDA and statistics. Predictive objectives activate evaluated ML when appropriate.</p></div><label className="field">Analytical objective<textarea rows={4} required minLength={8} maxLength={2000} value={objective} onChange={event => setObjective(event.target.value)} /></label><div className="suggested-questions">{['Why did sales decrease?', 'Predict customer churn.', 'Find customer segments.', 'Forecast future sales.'].map(text => <button type="button" key={text} onClick={() => setObjective(text)}>{text}</button>)}</div><details className="advanced-options"><summary>Optional target and search budget</summary><div><label className="field">Outcome column<select value={target} onChange={event => setTarget(event.target.value)}><option value="">Let the planner assess candidates</option>{detail.data?.profile?.columns.map(column => <option key={column.name} value={column.name}>{label(column.name)}</option>)}</select><small className="field-help">Selecting an outcome does not force ML for a descriptive question.</small></label><label className="field">ML search budget<select value={budget} onChange={event => setBudget(Number(event.target.value))}><option value={60}>Quick · 1 minute</option><option value={180}>Balanced · 3 minutes</option><option value={600}>Thorough · 10 minutes</option></select></label></div></details></div>}
      {step === 2 && <div>{plan.isLoading ? <Loading text="Building an objective-driven plan…" /> : plan.error ? <ErrorState error={plan.error} /> : plan.data && <><div className="info-note">{plan.data.ml_decision.reason}</div><PlanView plan={plan.data} /></>}</div>}
      {create.error && <div className="inline-error" role="alert">{create.error.message}</div>}
      <div className="modal-footer wizard-footer"><button type="button" className="button secondary" onClick={() => step ? setStep(step - 1) : close()}>{step > 0 && <ArrowLeft size={16} />}{step ? 'Back' : 'Cancel'}</button><span>Step {step + 1} of 3</span><button className="button primary" disabled={!datasetId || detail.isLoading || create.isPending || step === 2 && (!plan.data || !!plan.error)}>{create.isPending ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}{create.isPending ? 'Starting analysis…' : step === 2 ? 'Start autonomous analysis' : step === 0 ? 'Continue to objective' : 'Preview analysis plan'}<ArrowRight size={16} /></button></div>
    </form>}
  </Modal>
}
