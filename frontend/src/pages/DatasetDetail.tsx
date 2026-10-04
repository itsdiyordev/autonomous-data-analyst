import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, Bot, CheckCircle2, ChevronLeft, ChevronRight, Database, Hash, LoaderCircle, MessageSquare, Send, ShieldCheck, Sparkles } from '../icons'
import { api, label, number } from '../api'
import type { Dataset } from '../types'
import { useWorkspace } from '../workspace'
import { CorrelationHeatmap, DistributionChart, ScatterPlot } from '../components/charts'
import { Empty, ErrorState, Loading, PageHeading, Panel } from '../components/ui'
import DataQuality from '../components/DataQuality'

const tabs = [
  { id: 'explore', title: 'Charts & insights' }, { id: 'table', title: 'Data table' },
   { id: 'schema', title: 'Columns' }, { id: 'assistant', title: 'Ask a question', icon: MessageSquare },
   { id: 'quality', title: 'Data quality' },
]

export default function DatasetDetail() {
  const { id } = useParams()
  const { openAnalysis, openAutonomous } = useWorkspace()
  const [tab, setTab] = useState('explore')
  const [columnName, setColumnName] = useState('')
  const [x, setX] = useState('')
  const [y, setY] = useState('')
  const [offset, setOffset] = useState(0)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<{ role: string; text: string; source?: string }[]>([])
  const chatEnd = useRef<HTMLDivElement>(null)
  const query = useQuery({ queryKey: ['dataset', id], queryFn: () => api<Dataset>(`/datasets/${id}`) })
  const profile = query.data?.profile
  useEffect(() => {
    if (!profile) return
    const numeric = profile.columns.filter(column => column.kind === 'numeric')
    setColumnName(numeric[0]?.name || profile.columns[0].name)
    setX(numeric[0]?.name || ''); setY(numeric[1]?.name || '')
  }, [profile])
  useEffect(() => { setMessages([]); setOffset(0); setTab('explore') }, [id])
  useEffect(() => { if (messages.length) chatEnd.current?.scrollIntoView({ block: 'nearest' }) }, [messages])
  const preview = useQuery({
    queryKey: ['preview', id, offset], queryFn: () => api<{ columns: string[]; records: Record<string, unknown>[]; total: number }>(`/datasets/${id}/preview?offset=${offset}&limit=20`), enabled: tab === 'table',
  })
  const scatter = useQuery({
    queryKey: ['scatter', id, x, y], queryFn: () => api<{ points: { x: number; y: number }[] }>(`/datasets/${id}/scatter?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`), enabled: !!x && !!y && x !== y && tab === 'explore',
  })
  const chat = useMutation({
    mutationFn: (text: string) => api<{ answer: string; source: string }>(`/datasets/${id}/chat`, { method: 'POST', body: JSON.stringify({ question: text }) }),
    onSuccess: response => setMessages(old => [...old, { role: 'assistant', text: response.answer, source: response.source }]),
    onError: (error: Error) => setMessages(old => [...old, { role: 'assistant', text: error.message, source: 'error' }]),
  })
  function send(text: string) {
    if (text.trim().length < 3 || chat.isPending) return
    setMessages(old => [...old, { role: 'user', text }]); setQuestion(''); chat.mutate(text)
  }
  if (query.isLoading) return <Loading text="Loading your dataset…" />
  if (query.error) return <ErrorState error={query.error} retry={() => query.refetch()} />
  const dataset = query.data!
  const selected = profile!.columns.find(column => column.name === columnName)
  const numeric = profile!.columns.filter(column => column.kind === 'numeric')

  return <>
    <Link to="/datasets" className="back-link"><ArrowLeft size={16} />Back to datasets</Link>
    <PageHeading eyebrow="STEP 2 · EXPLORE YOUR DATA" title={dataset.name} text={`${dataset.filename} · Understand the data before you build a model.`} actions={<><button className="button secondary" onClick={() => openAutonomous(id)}><Sparkles size={17} />Analyze data</button><button className="button primary" onClick={() => openAnalysis(id)}><Sparkles size={17} />Train a model<ArrowRight size={16} /></button></>} />
    <div className="dataset-summary">
      <div><span className="summary-icon"><Database size={19} /></span><span><strong>{number(dataset.row_count)}</strong><small>Total rows</small></span></div>
      <div><span className="summary-icon"><Hash size={19} /></span><span><strong>{dataset.column_count}</strong><small>Columns</small></span></div>
      <div><span className="summary-icon"><ShieldCheck size={19} /></span><span><strong>{profile!.quality_score}%</strong><small>Data quality</small></span></div>
      <div><span className="summary-icon"><CheckCircle2 size={19} /></span><span><strong>{profile!.outlier_rows !== undefined ? number(profile!.outlier_rows) : '—'}</strong><small>Potential outlier rows</small></span></div>
    </div>
    <div className="tabs" aria-label="Dataset views">{tabs.map(item => <button key={item.id} className={tab === item.id ? 'active' : ''} aria-pressed={tab === item.id} onClick={() => setTab(item.id)}>{item.icon && <item.icon size={16} />}{item.title}</button>)}</div>

    {tab === 'explore' && <>
      <div className="dashboard-grid equal-grid">
        <Panel title="How are the values distributed?" subtitle={`Frequency counts from ${number(profile!.sampled_rows)} sampled rows.`} action={<select className="compact-select" aria-label="Distribution feature" value={columnName} onChange={event => setColumnName(event.target.value)}>{profile!.columns.map(column => <option key={column.name} value={column.name}>{label(column.name)}</option>)}</select>}>
          {selected && <DistributionChart data={selected.distribution} />}
          <div className="feature-stat-row"><span>Column type<strong>{selected?.kind === 'numeric' ? 'Numbers' : selected?.kind === 'datetime' ? 'Dates' : 'Categories'}</strong></span><span>Distinct values<strong>{number(selected?.unique || 0)}</strong></span><span>Missing values<strong>{selected?.missing_pct}%</strong></span></div>
        </Panel>
        <Panel title="Which columns move together?" subtitle="Correlation runs from −1 (opposite) to +1 (together)."><CorrelationHeatmap profile={profile!} /><div className="chart-explanation">Darker cells show a stronger relationship. Hover a cell for its exact value.</div></Panel>
      </div>
      <div className="dashboard-grid equal-grid">
        <Panel title="Compare two numeric columns" subtitle="Each dot is one row. Showing up to 600 sampled rows." action={numeric.length >= 2 && <div className="axis-selectors"><label>X<select className="compact-select" aria-label="Scatter X" value={x} onChange={event => setX(event.target.value)}>{numeric.map(column => <option key={column.name} value={column.name}>{label(column.name)}</option>)}</select></label><label>Y<select className="compact-select" aria-label="Scatter Y" value={y} onChange={event => setY(event.target.value)}>{numeric.map(column => <option key={column.name} value={column.name}>{label(column.name)}</option>)}</select></label></div>}>
          {numeric.length < 2 ? <Empty title="Two numeric columns are needed" text="Scatter plots compare the values in a pair of numeric columns." /> : x === y ? <div className="chart-empty">Select a different column for one of the axes.</div> : scatter.isLoading ? <Loading text="Plotting the rows…" /> : scatter.error ? <ErrorState error={scatter.error} /> : <ScatterPlot data={scatter.data?.points || []} xLabel={label(x)} yLabel={label(y)} />}
        </Panel>
        <Panel title="A few things worth knowing" subtitle="Observations calculated directly from your data."><div className="insight-list">{profile!.insights.map((insight, index) => <div key={insight.title} className={`insight-item ${insight.kind}`}><span>{index === 0 ? <Database size={18} /> : <Sparkles size={18} />}</span><div><strong>{insight.title}</strong><p>{insight.text}</p></div></div>)}</div></Panel>
      </div>
      <div className="explorer-next-step"><div><span className="eyebrow">READY FOR THE NEXT STEP?</span><h3>Use this dataset to make predictions.</h3><p>Choose a target column. We’ll prepare the data and compare the models.</p></div><button className="button primary" onClick={() => openAnalysis(id)}>Set up a model<ArrowRight size={17} /></button></div>
    </>}

    {tab === 'table' && <Panel title="Your original records" subtitle="These values are shown before model preprocessing." action={<span className="small-tag">{number(dataset.row_count)} ROWS</span>}>
      {preview.isLoading ? <Loading /> : preview.error ? <ErrorState error={preview.error} /> : <>
        <div className="table-scroll"><table className="data-table"><thead><tr><th>Row</th>{preview.data?.columns.map(column => <th key={column}>{column}</th>)}</tr></thead><tbody>{preview.data?.records.map((record, index) => <tr key={index}><td className="muted">{offset + index + 1}</td>{preview.data?.columns.map(column => <td key={column}>{record[column] == null ? <span className="null-value">Missing</span> : String(record[column])}</td>)}</tr>)}</tbody></table></div>
        <div className="table-pagination"><span>Rows {offset + 1}–{Math.min(offset + 20, dataset.row_count)} of {number(dataset.row_count)}</span><div><button className="button secondary" aria-label="Previous rows" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 20))}><ChevronLeft size={16} />Previous</button><button className="button secondary" aria-label="Next rows" disabled={offset + 20 >= dataset.row_count} onClick={() => setOffset(offset + 20)}>Next<ChevronRight size={16} /></button></div></div>
      </>}
    </Panel>}

    {tab === 'schema' && <Panel title="Understand every column" subtitle="Data types, missing values, and a few examples.">
      <div className="table-scroll"><table className="data-table"><thead><tr><th>Column</th><th>Values</th><th>Distinct values</th><th>Missing values</th><th>IQR outliers</th><th>Examples</th></tr></thead><tbody>{profile!.columns.map(column => <tr key={column.name}><td><strong>{column.name}</strong></td><td><span className={`type-badge ${column.kind}`}>{column.kind === 'numeric' ? 'Numbers' : column.kind === 'datetime' ? 'Dates' : 'Categories'}</span></td><td>{number(column.unique)}</td><td><span className={column.missing_pct > 0 ? 'text-amber' : 'text-green'}>{column.missing_pct}%</span><small className="cell-secondary">{number(column.missing)} rows</small></td><td>{column.outliers !== undefined ? number(column.outliers) : '—'}</td><td className="example-cell">{column.examples.join(' · ')}</td></tr>)}</tbody></table></div>
      <div className="fingerprint"><ShieldCheck size={16} /><span>Dataset fingerprint <code>{dataset.fingerprint.slice(0, 32)}…</code></span></div>
    </Panel>}
    {tab === 'quality' && <DataQuality dataset={dataset} />}

    {tab === 'assistant' && <Panel className="chat-panel">
      <div className="chat-intro"><span className="chat-bot-icon"><Bot size={29} /></span><span className="eyebrow">ASK YOUR DATA</span><h2>What would you like to understand?</h2><p>Ask about missing values, relationships, or a particular column.<br />Answers are based on the dataset’s calculated profile.</p></div>
      {!messages.length && <div className="suggested-questions">{['What are the missing values?', 'Which features are most correlated?', `Tell me about ${profile!.columns[1]?.name || profile!.columns[0].name}`].map(text => <button key={text} onClick={() => send(text)}>{text}<ArrowRight size={15} /></button>)}</div>}
      <div className="chat-messages" aria-live="polite">{messages.map((message, index) => <div key={index} className={`chat-message ${message.role}`}><span className="chat-role-icon">{message.role === 'assistant' ? <Sparkles size={17} /> : 'You'}</span><div><p>{message.text}</p>{message.source && <small>{message.source === 'llm' ? 'AI explanation based on computed evidence' : message.source === 'error' ? 'The request could not complete' : 'Answer from the computed dataset profile'}</small>}</div></div>)}{chat.isPending && <div className="chat-thinking"><LoaderCircle size={17} className="spin" />Checking your data…</div>}<div ref={chatEnd} /></div>
      <form className="chat-composer" onSubmit={event => { event.preventDefault(); send(question) }}><input aria-label="Question about your data" placeholder="Ask a question about this dataset…" value={question} onChange={event => setQuestion(event.target.value)} maxLength={2000} /><button className="button primary" aria-label="Send question" disabled={chat.isPending || question.trim().length < 3}><Send size={18} /></button></form>
    </Panel>}
  </>
}
