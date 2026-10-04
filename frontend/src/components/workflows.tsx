import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Check, Database, FileSpreadsheet, FlaskConical, LoaderCircle, ShieldCheck, Sparkles, UploadCloud } from '../icons'
import { toast } from 'sonner'
import { api, label, number } from '../api'
import type { Dataset, Run } from '../types'
import { useWorkspace } from '../workspace'
import { Empty, ErrorState, Loading, Modal } from './ui'

export function UploadModal({ close }: { close: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  function selectFile(selected?: File) {
    if (!selected) return
    if (!/\.(csv|xlsx|xlsm)$/i.test(selected.name)) return toast.error('Choose a CSV or Excel spreadsheet.')
    if (selected.size > 50 * 1024 * 1024) return toast.error('The maximum file size is 50 MB.')
    setFile(selected)
  }
  const upload = useMutation({
    mutationFn: () => { const form = new FormData(); form.append('file', file!); return api<Dataset>('/datasets', { method: 'POST', body: form }) },
    onSuccess: dataset => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Dataset uploaded. Your charts are ready.')
      close(); navigate(`/datasets/${dataset.id}`)
    },
    onError: (error: Error) => toast.error(error.message),
  })
  return <Modal title="Upload a dataset" subtitle="Choose a spreadsheet. We’ll check it and prepare your charts." close={close}>
    <div className={`dropzone ${dragging ? 'dragging' : ''}`} role="button" tabIndex={0} aria-label="Choose dataset file"
      onClick={() => input.current?.click()} onKeyDown={event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); input.current?.click() } }}
      onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)}
      onDrop={event => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]) }}>
      <input ref={input} type="file" accept=".csv,.xlsx,.xlsm" hidden onChange={event => selectFile(event.target.files?.[0])} />
      <span className="drop-icon">{file ? <FileSpreadsheet size={31} /> : <UploadCloud size={31} />}</span>
      <h3>{file ? file.name : 'Drop your file here'}</h3>
      <p>{file ? `${(file.size / 1024).toFixed(1)} KB · Click to choose a different file` : 'or click to browse your computer'}</p>
      <span className="file-types">CSV or Excel · Up to 50 MB</span>
    </div>
    <div className="upload-outcomes"><h4>What happens next?</h4>{['Your original data is saved.', 'Columns and data quality are checked.', 'You can explore charts or train a model.'].map((text, index) => <div key={text}><span>{index + 1}</span>{text}</div>)}</div>
    {upload.error && <div className="inline-error" role="alert">{upload.error.message}</div>}
    <div className="modal-footer"><button className="button secondary" onClick={close}>Cancel</button><button className="button primary" disabled={!file || upload.isPending} onClick={() => upload.mutate()}>{upload.isPending ? <LoaderCircle size={17} className="spin" /> : <UploadCloud size={17} />}{upload.isPending ? 'Uploading & checking…' : 'Upload and explore'}<ArrowRight size={16} /></button></div>
  </Modal>
}

export function AnalysisModal({ initialDataset, close }: { initialDataset?: string; close: () => void }) {
  const { openUpload } = useWorkspace()
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  const [step, setStep] = useState(0)
  const [datasetId, setDatasetId] = useState(initialDataset || '')
  const [target, setTarget] = useState('')
  const [objective, setObjective] = useState('')
  const [task, setTask] = useState('auto')
  const [split, setSplit] = useState('random')
  const [splitColumn, setSplitColumn] = useState('')
  const [budget, setBudget] = useState(180)
  const [cvFolds, setCvFolds] = useState(3)
  const initializedDataset = useRef('')
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  useEffect(() => { if (!datasetId && datasets.data?.length) setDatasetId(datasets.data[0].id) }, [datasetId, datasets.data])
  const detail = useQuery({ queryKey: ['dataset', datasetId], queryFn: () => api<Dataset>(`/datasets/${datasetId}`), enabled: !!datasetId })
  useEffect(() => {
    if (!detail.data?.profile || initializedDataset.current === detail.data.id) return
    initializedDataset.current = detail.data.id
    const columns = detail.data.profile.columns
    const candidate = columns.find(column => ['churn', 'monthly_revenue', 'price', 'target', 'label'].includes(column.name.toLowerCase())) || columns[columns.length - 1]
    setTarget(candidate.name)
    setObjective(`Predict ${candidate.name.replaceAll('_', ' ')} and explain the most important factors.`)
    setSplitColumn('')
  }, [detail.data])
  const create = useMutation({
    mutationFn: () => api<Run>('/analysis-runs', { method: 'POST', body: JSON.stringify({ dataset_id: datasetId, target: task === 'clustering' ? null : target || null, objective, task, split_strategy: split, split_column: splitColumn || null, budget_seconds: budget, cv_folds: cvFolds }) }),
    onSuccess: run => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] }); queryClient.invalidateQueries({ queryKey: ['runs'] })
      close(); toast.success('Model training started'); navigate(`/runs/${run.id}`)
    },
    onError: (error: Error) => toast.error(error.message),
  })
  const selected = detail.data
  const targetColumn = selected?.profile?.columns.find(column => column.name === target)
  const predictedType = task === 'clustering' || (task === 'auto' && !target) ? 'clustering' : task === 'auto' ? targetColumn?.kind !== 'numeric' || (targetColumn.unique <= 15 && targetColumn.unique / selected!.row_count < 0.1) ? 'classification' : 'regression' : task
  const titles = ['Choose your data', 'Set your prediction goal', 'Review and start']

  return <Modal title="Train a prediction model" subtitle="Three simple steps. We handle preparation, model comparison, and testing." close={close} wide>
    {datasets.isLoading ? <Loading text="Loading your datasets…" /> : datasets.error ? <ErrorState error={datasets.error} retry={() => datasets.refetch()} /> : !datasets.data?.length ? <Empty title="First, add a dataset" text="Upload a CSV or Excel file so there’s data to learn from." action={<button className="button primary" onClick={() => { close(); openUpload() }}><UploadCloud size={17} />Upload dataset</button>} /> : <form onSubmit={event => { event.preventDefault(); if (step < 2) setStep(step + 1); else create.mutate() }}>
      <ol className="wizard-steps">{titles.map((title, index) => <li key={title} className={step === index ? 'current' : step > index ? 'done' : ''} aria-current={step === index ? 'step' : undefined}><span>{step > index ? <Check size={15} /> : index + 1}</span><strong>{title}</strong></li>)}</ol>

      {step === 0 && <div className="wizard-page">
        <div className="wizard-heading"><span className="eyebrow">STEP 1 OF 3</span><h3>Which dataset should we learn from?</h3><p>Choose a file you’ve already added to this workspace.</p></div>
        <label className="field">Dataset<select value={datasetId} onChange={event => { setDatasetId(event.target.value); setTarget('') }} required><option value="">Choose a dataset</option>{datasets.data.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>
        {detail.isLoading ? <Loading text="Checking the dataset…" /> : detail.error ? <ErrorState error={detail.error} /> : selected && <div className="selected-dataset-card"><span className="dataset-file-icon"><FileSpreadsheet size={25} /></span><div><strong>{selected.name}</strong><small>{selected.filename}</small></div><div className="selected-dataset-stats"><span><strong>{number(selected.row_count)}</strong>rows</span><span><strong>{selected.column_count}</strong>columns</span></div></div>}
        <div className="info-note"><Database size={18} /><span>The original file stays saved. Cleaning and encoding are included in the trained pipeline.</span></div>
      </div>}

      {step === 1 && <div className="wizard-page">
        <div className="wizard-heading"><span className="eyebrow">STEP 2 OF 3</span><h3>What would you like the model to do?</h3><p>Predict a category or number, or discover groups of similar rows.</p></div>
        {task !== 'clustering' && <label className="field">Prediction target<select value={target} onChange={event => { setTarget(event.target.value); if (!event.target.value) setObjective('Find groups of similar records and explain each group.') }} required={task !== 'auto'}><option value="">No target — discover groups</option>{selected?.profile?.columns.map(column => <option key={column.name} value={column.name}>{label(column.name)} ({column.name})</option>)}</select><small className="field-help">For example: churn, house price, or monthly revenue. Without a target, auto-detection chooses clustering.</small></label>}
        <label className="field">Describe your goal<textarea value={objective} onChange={event => setObjective(event.target.value)} minLength={8} maxLength={2000} rows={3} required placeholder="Predict customer churn and explain the main risk factors." /></label>
        <fieldset className="task-options"><legend>Prediction type</legend>{[
          { value: 'auto', title: 'Choose for me', text: 'Detect from the target column' },
          { value: 'classification', title: 'A category', text: 'Yes / no, a label, or a class' },
          { value: 'regression', title: 'A number', text: 'A price, amount, or quantity' },
          { value: 'clustering', title: 'Find groups', text: 'Similar rows, without labels' },
        ].map(option => <label key={option.value} className={task === option.value ? 'selected' : ''}><input type="radio" name="task" value={option.value} checked={task === option.value} onChange={() => { setTask(option.value); if (option.value === 'clustering') setObjective('Find groups of similar records and explain each group.'); else if (task === 'clustering' && target) setObjective(`Predict ${target.replaceAll('_', ' ')} and explain the most important factors.`) }} /><strong>{option.title}</strong><small>{option.text}</small></label>)}</fieldset>
      </div>}

      {step === 2 && <div className="wizard-page">
        <div className="wizard-heading"><span className="eyebrow">STEP 3 OF 3</span><h3>Your model is ready to build.</h3><p>We’ll compare several approaches and select the best validation result.</p></div>
        <div className="training-summary"><div><span>Dataset</span><strong>{selected?.name}</strong></div><div><span>Goal</span><strong>{predictedType === 'clustering' ? 'Discover similar groups' : label(target)}</strong></div><div><span>Task</span><strong>{predictedType === 'clustering' ? 'Clustering' : predictedType === 'classification' ? 'Category prediction' : 'Numeric prediction'}</strong></div><div><span>Validation</span><strong>{cvFolds}-fold cross-validation</strong></div></div>
        <div className="training-plan"><span className="stat-icon purple"><FlaskConical size={21} /></span><div><strong>Profile → detect → prepare → train → validate → explain</strong><p>{predictedType === 'clustering' ? 'Compare K-Means and Gaussian mixture groups.' : 'Compare linear, tree, ensemble, and neural-network candidates.'} Get predictions, feature importance, SHAP, error analysis, and a final report.</p></div></div>
        <details className="advanced-options"><summary>Advanced settings<span>Optional</span></summary><div>
          <div className="form-grid"><label className="field">Model search budget<select value={budget} onChange={event => setBudget(Number(event.target.value))}><option value={60}>Quick · 1 minute</option><option value={180}>Balanced · 3 minutes</option><option value={600}>Thorough · 10 minutes</option></select></label><label className="field">How to split the data<select value={split} onChange={event => setSplit(event.target.value)}><option value="random">Random / balanced classes</option><option value="chronological">By date (earlier → later)</option><option value="group">Keep related groups together</option></select></label></div>
          {split !== 'random' && <label className="field">{split === 'group' ? 'Group identifier column' : 'Date / time column'}<select required value={splitColumn} onChange={event => setSplitColumn(event.target.value)}><option value="">Select a column</option>{selected?.profile?.columns.filter(column => column.name !== target).map(column => <option key={column.name} value={column.name}>{label(column.name)}</option>)}</select></label>}
          <label className="field">Cross-validation folds<select value={cvFolds} onChange={event => setCvFolds(Number(event.target.value))}>{[2, 3, 5].map(value => <option key={value} value={value}>{value} folds</option>)}</select><small className="field-help">Preprocessing is refitted inside each fold. Group/time splits use matching fold strategies.</small></label>
          <p className="field-help">The budget is checked between models. Final training, testing, and packaging may take longer.</p>
        </div></details>
        <div className="info-note"><ShieldCheck size={18} /><span>A separate test set measures performance on data the model hasn’t seen.</span></div>
      </div>}

      {create.error && <div className="inline-error" role="alert">{create.error.message}</div>}
      <div className="modal-footer wizard-footer">
        <button type="button" className="button secondary" onClick={() => step > 0 ? setStep(step - 1) : close()}>{step > 0 && <ArrowLeft size={16} />}{step > 0 ? 'Back' : 'Cancel'}</button>
        <span>Step {step + 1} of 3</span>
        <button className="button primary" disabled={!datasetId || (task !== 'auto' && task !== 'clustering' && !target) || detail.isLoading || create.isPending}>{create.isPending ? <LoaderCircle className="spin" size={17} /> : step === 2 ? <Sparkles size={17} /> : null}{create.isPending ? 'Starting training…' : step === 2 ? 'Start training' : step === 0 ? 'Continue to goal' : 'Review model setup'}<ArrowRight size={16} /></button>
      </div>
    </form>}
  </Modal>
}
