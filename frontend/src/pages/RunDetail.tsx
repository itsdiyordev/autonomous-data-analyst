import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, Check, CheckCircle2, Clock3, Code2, Copy, Download, FileText, FlaskConical, LoaderCircle, Play, ShieldCheck, Sparkles, Trophy, X } from '../icons'
import { toast } from 'sonner'
import { api, download, label, metric, number } from '../api'
import type { ModelResult, Run } from '../types'
import { metrics as metricDescriptions, taskText } from '../presentation'
import { ClusterPlot, CurveChart, ImportanceChart, ScatterPlot } from '../components/charts'
import Pipeline from '../components/Pipeline'
import { ClusterProfiles, ErrorEvidence, ShapEvidence } from '../components/ModelEvidence'
import { ErrorState, Loading, PageHeading, Panel, Status } from '../components/ui'
import { useWorkspace } from '../workspace'

const resultTabs = [
  { id: 'overview', title: 'Summary' }, { id: 'comparison', title: 'Model comparison' },
  { id: 'explainability', title: 'Understand the model' }, { id: 'predictions', title: 'Try predictions' },
  { id: 'report', title: 'Final report' },
]

export default function RunDetail() {
  const { id } = useParams()
  const [tab, setTab] = useState('overview')
  const [downloading, setDownloading] = useState('')
  const queryClient = useQueryClient()
  const { openAnalysis } = useWorkspace()
  const query = useQuery({ queryKey: ['run', id], queryFn: () => api<Run>(`/analysis-runs/${id}`), refetchInterval: value => ['running', 'queued'].includes(value.state.data?.status || '') ? 1500 : false })
  const cancel = useMutation({
    mutationFn: () => api<Run>(`/analysis-runs/${id}/cancel`, { method: 'POST' }),
    onSuccess: run => { queryClient.setQueryData(['run', id], run); queryClient.invalidateQueries({ queryKey: ['dashboard'] }); toast.success('Training cancelled') },
    onError: (error: Error) => toast.error(error.message),
  })
  useEffect(() => {
    if (query.data?.status === 'completed') {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] }); queryClient.invalidateQueries({ queryKey: ['runs'] })
    }
  }, [query.data?.status, queryClient])
  useEffect(() => { setTab('overview') }, [id])
  if (query.isLoading) return <Loading text="Loading the model build…" />
  if (query.error) return <ErrorState error={query.error} retry={() => query.refetch()} />
  const run = query.data!
  const result = run.result
  const secondaryMetric = result?.task === 'classification' ? 'accuracy' : result?.task === 'clustering' ? 'davies_bouldin' : 'mae'
  async function getArtifact(report = false) {
    setDownloading(report ? 'report' : 'model')
    try {
      await download(report ? `/analysis-runs/${id}/report` : `/models/${id}/download`, report ? 'analytiq-report.html' : 'analytiq-solution.zip')
      toast.success(report ? 'Report downloaded' : 'Model package downloaded')
    } catch (error) { toast.error((error as Error).message) } finally { setDownloading('') }
  }

  return <>
    <Link to="/analyses" className="back-link"><ArrowLeft size={16} />Back to training history</Link>
    <PageHeading eyebrow={result ? 'STEP 4 · YOUR TRAINED SOLUTION' : 'STEP 3 · TRAINING YOUR MODEL'} title={result ? 'Model results' : 'Building your prediction model'} text={run.objective} actions={result && <>
      <button className="button secondary" disabled={!!downloading} onClick={() => getArtifact(true)}><FileText size={17} />Export report</button>
      <button className="button primary" disabled={!!downloading} onClick={() => getArtifact()}>{downloading === 'model' ? <LoaderCircle size={17} className="spin" /> : <Download size={17} />}Download model</button>
    </>} />
    <div className="run-meta"><Status status={run.status} /><span><FlaskConical size={15} />{taskText(run.task)}</span>{run.task !== 'clustering' && <span>Predicting <strong>{label(run.target)}</strong></span>}<span><Clock3 size={15} />{result ? `${result.duration_seconds.toFixed(1)} seconds` : 'Runs in the background'}</span><span className="run-id">BUILD / {run.id.slice(0, 8)}</span></div>

    {!result && <>
      <Panel className="execution-panel">
        <div className="execution-heading"><span className={`execution-icon ${run.status}`}>{['failed', 'cancelled'].includes(run.status) ? <X size={28} /> : <Sparkles size={28} />}</span><div><h2>{run.stage}</h2><p>{run.error || (run.status === 'cancelled' ? 'This build was stopped. You can start a new one whenever you’re ready.' : 'You can leave this page and return from Training history.')}</p></div>{['running', 'queued'].includes(run.status) && <span className="execution-percent">{run.progress}%</span>}</div>
        <div className="progress-track large"><i style={{ width: `${run.progress}%` }} /></div>
        <Pipeline run={run} />
        <div className="execution-footer"><span><ShieldCheck size={16} />Final scores use data kept separate from model training.</span>{['running', 'queued'].includes(run.status) ? <button className="button subtle" disabled={cancel.isPending} onClick={() => cancel.mutate()}>Cancel training</button> : <button className="button primary" onClick={() => openAnalysis(run.dataset_id)}>Start a new build<ArrowRight size={16} /></button>}</div>
      </Panel>
      <EventLog run={run} />
    </>}

    {result && <>
      <div className="tabs" aria-label="Model result views">{resultTabs.map(item => <button key={item.id} onClick={() => setTab(item.id)} className={tab === item.id ? 'active' : ''} aria-pressed={tab === item.id}>{item.id === 'predictions' && <Play size={15} />}{item.title}</button>)}</div>
      {tab === 'overview' && <>
        <div className="model-winner"><span className="winner-icon"><Trophy size={27} /></span><div><span>SELECTED MODEL</span><h3>{result.model_name}</h3><p>{result.cross_validation?.status === 'completed' ? `Selected using ${result.cross_validation.folds}-fold ${result.cross_validation.strategy}.` : 'Selected on validation results.'} Final scores use unseen test data.</p></div><div className="winner-score"><small>{metricDescriptions[result.primary_metric]?.title || label(result.primary_metric)}</small><strong>{metric(result.primary_metric, result.metrics[result.primary_metric])}</strong></div></div>
        <div className="results-help"><ShieldCheck size={19} /><div><strong>How to read these results.</strong> These scores describe performance on the separate test set. Hover an underlined metric name to see what it measures.</div></div>
        <div className="metrics-grid">{Object.entries(result.metrics).filter(([name]) => name !== 'f1_macro').map(([name, value]) => <div className="metric-card" key={name}><span><abbr title={metricDescriptions[name]?.explanation}>{metricDescriptions[name]?.title || label(name)}</abbr></span><strong>{metric(name, value)}</strong><small>{name === 'clusters' ? 'Assigned test groups' : ['mae', 'rmse', 'davies_bouldin'].includes(name) ? 'Lower is better · test data' : 'Higher is better · test data'}</small></div>)}</div>
        <div className="dashboard-grid equal-grid">
          <Panel title="What influences the prediction?" subtitle="A larger bar means the model relies more on that column."><ImportanceChart data={result.feature_importance} /><div className="chart-explanation">Measured by shuffling one column at a time in test data and checking the score change.</div></Panel>
          <Panel title={result.task === 'clustering' ? 'See the discovered groups' : result.task === 'classification' ? 'How well does it separate outcomes?' : 'Predictions versus actual values'} subtitle={result.task === 'clustering' ? 'A two-dimensional PCA projection of test rows, colored by their assigned group.' : result.task === 'classification' ? (result.positive_class ? `ROC curve · positive outcome: ${result.positive_class}` : 'Correct and incorrect predicted categories') : 'Each dot compares a test prediction with its actual value.'}>
            {result.diagnostics.cluster_projection ? <ClusterPlot data={result.diagnostics.cluster_projection} /> : result.diagnostics.roc_curve ? <CurveChart data={result.diagnostics.roc_curve} xKey="fpr" yKey="tpr" xLabel="False positive rate" yLabel="True positive rate" /> : result.diagnostics.predicted_vs_actual ? <ScatterPlot data={result.diagnostics.predicted_vs_actual.map(value => ({ x: value.actual, y: value.predicted }))} xLabel="Actual value" yLabel="Predicted value" /> : <ConfusionMatrix result={result} />}
          </Panel>
        </div>
        {result.task === 'clustering' && <ClusterProfiles result={result} />}
        {result.pipeline && <Panel title="Your completed ML pipeline" subtitle={result.problem_detection?.reason}><Pipeline run={run} /></Panel>}
        <Panel title="How your data was used" subtitle="The test set stayed separate during model selection."><div className="split-summary">{[{ name: 'Training rows', value: result.split.train }, { name: 'Validation rows', value: result.split.validation }, { name: 'Separate test rows', value: result.split.test }].map(part => <div key={part.name}><span>{part.name}</span><strong>{number(part.value)}</strong></div>)}<div><span>Split strategy</span><strong>{label(result.split.strategy)}</strong></div></div><div className="info-note"><ShieldCheck size={18} /><span>{result.selection_note}</span></div></Panel>
      </>}

      {tab === 'comparison' && <Panel title="Compare the approaches" subtitle="Selection uses cross-validation means when available. Validation and test scores are kept distinct.">
        <div className="table-scroll"><table className="data-table"><thead><tr><th>Model / family</th><th>Selection score</th><th>Validation {metricDescriptions[result.primary_metric]?.title || label(result.primary_metric)}</th><th>{metricDescriptions[secondaryMetric]?.title}</th><th>Training time</th><th>Configuration</th></tr></thead><tbody>{[...result.experiments].sort((a, b) => {
          if (a.status !== b.status) return a.status === 'completed' ? -1 : b.status === 'completed' ? 1 : 0
          return (result.task === 'regression' ? 1 : -1) * ((a.selection_score ?? a.validation_metrics?.[result.primary_metric] ?? 0) - (b.selection_score ?? b.validation_metrics?.[result.primary_metric] ?? 0))
        }).map(experiment => <tr key={experiment.name} className={experiment.name === result.model_name ? 'winner-row' : ''}>
          <td><strong>{experiment.name}</strong>{experiment.name === result.model_name && <span className="selected-badge"><Trophy size={11} />SELECTED</span>}<small className="cell-secondary">{label(experiment.family || 'Candidate model')}</small></td>
          <td>{experiment.status === 'completed' ? <>{metric(result.primary_metric, experiment.selection_score ?? experiment.validation_metrics[result.primary_metric])}<small className="cell-secondary">{experiment.cross_validation?.mean !== undefined ? `${experiment.cross_validation.folds} folds · SD ${experiment.cross_validation.std !== undefined ? metric(result.primary_metric, experiment.cross_validation.std) : '—'}` : 'Validation selection'}</small></> : <span className="text-amber">{experiment.status === 'skipped' ? 'Budget reached' : 'Could not train'}</span>}</td>
          <td>{experiment.status === 'completed' ? metric(result.primary_metric, experiment.validation_metrics[result.primary_metric]) : '—'}</td>
          <td>{experiment.status === 'completed' ? metric(secondaryMetric, experiment.validation_metrics[secondaryMetric]) : '—'}</td>
          <td>{experiment.duration_seconds?.toFixed(2) || '—'}s</td><td><code className="params-code">{experiment.reason || experiment.error || JSON.stringify(experiment.parameters)}</code></td>
        </tr>)}</tbody></table></div>
        <div className="info-note"><CheckCircle2 size={18} /><span>{result.selection_note}</span></div>
      </Panel>}

      {tab === 'explainability' && <>
        <ShapEvidence result={result} />
        <div className="dashboard-grid equal-grid"><Panel title="Which columns matter most?" subtitle="Score change after shuffling a column, averaged over 3 repeats."><ImportanceChart data={result.feature_importance} /></Panel>{result.task === 'clustering' ? <Panel title="Group separation" subtitle="A visual projection of the assigned test groups."><ClusterPlot data={result.diagnostics.cluster_projection || []} /></Panel> : result.task === 'classification' ? <Panel title="Correct and incorrect predictions" subtitle="Rows show the actual category; columns show the prediction."><ConfusionMatrix result={result} /></Panel> : <Panel title="Where are the prediction errors?" subtitle="Residual = actual value minus predicted value."><ScatterPlot data={result.diagnostics.predicted_vs_actual?.map(value => ({ x: value.predicted, y: value.residual })) || []} xLabel="Predicted value" yLabel="Error (residual)" /></Panel>}</div>
        <ErrorEvidence result={result} />
        {result.feature_engineering && <Panel title="Feature engineering and data handling" subtitle="Transformations saved inside the reusable prediction pipeline."><div className="feature-engineering-summary"><div><strong>Missing values</strong><p>Numeric medians, categorical modes, and learned missing-value indicators.</p></div><div><strong>Encoding and scaling</strong><p>Unknown-safe categorical encoding, numeric scaling, and whitespace normalization.</p></div><div><strong>Date features</strong><p>{result.feature_engineering.date_columns.length ? `${result.feature_engineering.date_columns.join(', ')} → year, month, weekday, and cyclic month features.` : 'No date columns were detected.'}</p></div><div><strong>Outlier policy</strong><p>{result.feature_engineering.outlier_policy}</p></div></div></Panel>}
        <Panel title="Columns left out of the model" subtitle="Identifiers and unsuitable columns are excluded automatically.">{result.excluded_features.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Column</th><th>Reason</th></tr></thead><tbody>{result.excluded_features.map(feature => <tr key={feature.feature}><td><strong>{feature.feature}</strong></td><td>{feature.reason}</td></tr>)}</tbody></table></div> : <div className="info-note"><Check size={17} />All non-target columns were kept.</div>}<div className="panel-bottom-note">Predictive importance describes how the model uses a column, not a causal effect.</div></Panel>
      </>}
      {tab === 'predictions' && <PredictionPlayground run={run} />}
      {tab === 'report' && <>
        <Panel title="Your final analysis report" subtitle={result.report_source === 'llm' ? 'AI-written explanation based on computed evidence' : 'A report built from the actual analysis and model results'} action={<button className="button secondary" disabled={!!downloading} onClick={() => getArtifact(true)}><Download size={16} />Download report</button>}><div className="report-content">{result.report}</div><div className="fingerprint"><ShieldCheck size={16} /><span>Dataset fingerprint <code>{result.dataset_fingerprint}</code></span></div></Panel>
        <EventLog run={run} />
      </>}
    </>}
  </>
}

function EventLog({ run }: { run: Run }) {
  return <Panel title="What happened during this build" subtitle="A recorded timeline of every completed stage."><div className="event-log">{run.events.map((event, index) => <div key={index}><span className="event-marker"><Check size={12} /></span><div><strong>{event.stage}</strong><p>{event.detail}</p></div><time>{new Date(event.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div>)}</div></Panel>
}

function ConfusionMatrix({ result }: { result: ModelResult }) {
  const matrix = result.diagnostics.confusion_matrix || []
  const classes = result.diagnostics.classes || []
  const max = Math.max(1, ...matrix.flat())
  return <div className="confusion-wrap"><table className="confusion-table"><thead><tr><th>Actual ↓ / Predicted →</th>{classes.map(name => <th key={name}>{name}</th>)}</tr></thead><tbody>{matrix.map((row, index) => <tr key={index}><th>{classes[index]}</th>{row.map((value, column) => <td key={column} style={{ background: index === column ? `rgba(139,115,192,${0.1 + value / max * 0.75})` : `rgba(211,151,147,${0.08 + value / max * 0.45})`, color: index === column && value / max > 0.6 ? 'white' : undefined }}>{value}</td>)}</tr>)}</tbody></table></div>
}

function PredictionPlayground({ run }: { run: Run }) {
  const result = run.result!
  const initial = Object.fromEntries(result.input_schema.map(feature => [feature.name, feature.example]))
  const [values, setValues] = useState<Record<string, unknown>>(initial)
  const [batch, setBatch] = useState(false)
  const [json, setJson] = useState(JSON.stringify([initial], null, 2))
  const prediction = useMutation({
    mutationFn: () => {
      let records: unknown
      if (batch) {
        try { records = JSON.parse(json) } catch { throw new Error('Enter a valid JSON array of records.') }
        if (!Array.isArray(records)) throw new Error('Batch input must be a JSON array.')
      } else records = [Object.fromEntries(result.input_schema.map(feature => [feature.name, values[feature.name] === '' ? null : feature.type === 'number' ? Number(values[feature.name]) : values[feature.name]]))]
      return api<{ predictions: { label?: string; value?: number; cluster?: number; probabilities?: Record<string, number> }[] }>(`/models/${run.id}/predict`, { method: 'POST', body: JSON.stringify({ records }) })
    }, onError: (error: Error) => toast.error(error.message),
  })
  const exampleCode = `POST /api/models/${run.id}/predict\nAuthorization: Bearer <your-session-token>\nContent-Type: application/json\n\n${JSON.stringify({ records: [initial] }, null, 2)}`

  return <div className="prediction-layout">
    <Panel title="Try your model on a new record" subtitle="Enter values below. The saved pipeline handles preprocessing." action={<div className="view-toggle"><button className={!batch ? 'active' : ''} onClick={() => setBatch(false)}>One record</button><button className={batch ? 'active' : ''} onClick={() => setBatch(true)}>Batch JSON</button></div>}>
      <form onSubmit={event => { event.preventDefault(); prediction.mutate() }}>
        {batch ? <label className="field">Records to predict (up to 1,000)<textarea className="code-input" rows={14} value={json} onChange={event => setJson(event.target.value)} required /></label> : <div className="form-grid prediction-fields">{result.input_schema.map(feature => <label className="field" key={feature.name}>{label(feature.name)}{feature.categories ? <select value={String(values[feature.name] ?? '')} onChange={event => setValues({ ...values, [feature.name]: event.target.value })}><option value="">Missing — fill automatically</option>{feature.categories.map(category => <option key={category}>{category}</option>)}</select> : <input type={feature.type === 'number' ? 'number' : 'text'} step="any" value={String(values[feature.name] ?? '')} onChange={event => setValues({ ...values, [feature.name]: event.target.value })} placeholder="Leave blank to fill automatically" />}</label>)}</div>}
        {prediction.error && <div className="inline-error" role="alert">{prediction.error.message}</div>}
        <button className="button primary full" disabled={prediction.isPending}>{prediction.isPending ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />}{prediction.isPending ? 'Making a prediction…' : 'Generate prediction'}</button>
      </form>
    </Panel>
    <div>
      <Panel title="Your prediction" subtitle="Live output from this trained model.">
        {prediction.data ? <div className="prediction-results" aria-live="polite">{prediction.data.predictions.slice(0, 20).map((item, index) => <div className="prediction-result" key={index}><span className="eyebrow">{prediction.data.predictions.length > 1 ? `RECORD ${index + 1}` : 'PREDICTED OUTCOME'}</span><strong>{item.label ?? number(item.value!)}</strong>{item.probabilities && <><div className="probability-bars">{Object.entries(item.probabilities).map(([name, value]) => <div key={name}><div><span>{name}</span><b>{(value * 100).toFixed(1)}%</b></div><div className="progress-track"><i style={{ width: `${value * 100}%` }} /></div></div>)}</div><p className="prediction-estimate-note">Model estimates for each possible outcome.</p></>}</div>)}{prediction.data.predictions.length > 20 && <p className="muted">Showing the first 20 of {prediction.data.predictions.length} predictions.</p>}</div> : <div className="prediction-placeholder"><FlaskConical size={35} /><h4>Your next prediction will appear here</h4><p>Complete the form and click Generate prediction.</p></div>}
      </Panel>
      <Panel title="Connect the model to your app" subtitle="Use the same prediction pipeline through the API." action={<button className="icon-button" aria-label="Copy API example" onClick={() => navigator.clipboard.writeText(exampleCode).then(() => toast.success('API example copied')).catch(() => toast.error('Clipboard is unavailable'))}><Copy size={17} /></button>}><details className="advanced-options"><summary>View API request example<span><Code2 size={15} /></span></summary><div><pre className="api-code">{exampleCode}</pre></div></details></Panel>
    </div>
  </div>
}
