import { useState } from 'react'
import { label, metric, number } from '../api'
import type { ModelResult } from '../types'
import { ImportanceChart } from './charts'
import { Empty, Panel } from './ui'

export function ShapEvidence({ result }: { result: ModelResult }) {
  const [exampleIndex, setExampleIndex] = useState(0)
  const shap = result.shap
  if (shap?.status !== 'completed') return <Panel title="SHAP contributions"><Empty title="No SHAP explanation for this build" text={shap?.reason || 'This earlier model does not contain SHAP evidence. New builds include sampled SHAP explanations.'} /></Panel>
  const example = shap.examples[exampleIndex]
  return <div className="dashboard-grid equal-grid">
    <Panel title="SHAP: feature contributions" subtitle={`${shap.method} · ${shap.sample_rows} sampled test rows`} action={<span className="small-tag">SHAP</span>}>
      <ImportanceChart data={shap.feature_importance} /><div className="chart-explanation">Average absolute contribution, combined back into the original input columns. Output: {shap.output}.</div>
    </Panel>
    <Panel title="Explain one test prediction" subtitle="See which inputs pushed the selected model output up or down." action={<select className="compact-select" aria-label="SHAP example" value={exampleIndex} onChange={event => setExampleIndex(Number(event.target.value))}>{shap.examples.map((item, index) => <option key={index} value={index}>Original row {item.row_index + 1}</option>)}</select>}>
      {example && <ImportanceChart data={example.contributions.map(item => ({ feature: item.feature, importance: item.value }))} />}
      <div className="chart-explanation">Positive contributions increase the explained output; negative contributions decrease it. These are model contributions, not causal effects.</div>
    </Panel>
  </div>
}

export function ErrorEvidence({ result }: { result: ModelResult }) {
  const analysis = result.error_analysis
  if (!analysis) return null
  const grouping = analysis.kind === 'cluster_ambiguity'
  return <Panel title={grouping ? 'Which group assignments need a closer look?' : 'Learn from the difficult predictions'} subtitle={analysis.note || (analysis.kind === 'classification' ? `${number(analysis.error_count || 0)} incorrect test predictions. Showing the most confident mistakes.` : `95% of absolute test errors are at or below ${analysis.p95_absolute_error?.toFixed(3)} target units.`)}>
    {analysis.examples.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Original row</th>{!grouping && <th>Actual value</th>}<th>{grouping ? 'Assigned group' : 'Predicted value'}</th><th>{grouping ? 'Assignment ambiguity' : analysis.kind === 'classification' ? 'Wrong-guess probability' : 'Absolute error'}</th><th>Input values</th></tr></thead><tbody>{analysis.examples.map(item => <tr key={item.row_index}><td>{item.row_index + 1}</td>{!grouping && <td>{typeof item.actual === 'number' ? number(item.actual) : item.actual}</td>}<td>{typeof item.predicted === 'number' ? number(item.predicted) : item.predicted}</td><td>{analysis.kind === 'classification' ? `${(item.error * 100).toFixed(1)}%` : item.error.toFixed(3)}</td><td><details className="record-details"><summary>View row</summary><pre>{JSON.stringify(item.record, null, 2)}</pre></details></td></tr>)}</tbody></table></div> : <div className="info-note">No incorrect predictions were found in this test set.</div>}
  </Panel>
}

export function ClusterProfiles({ result }: { result: ModelResult }) {
  if (!result.cluster_profiles?.length) return null
  return <Panel title="What makes each group different?" subtitle="Profiles summarize the held-out rows assigned to each group. Numeric values are means; categories are the most common value.">
    <div className="cluster-profile-grid">{result.cluster_profiles.map(group => <article className="cluster-profile" key={group.cluster}><div><h4>{group.name}</h4><span>{metric('accuracy', group.share)} of test rows</span></div><strong>{number(group.rows)} <small>rows</small></strong><dl>{Object.entries(group.features).slice(0, 8).map(([name, value]) => <div key={name}><dt>{label(name)}</dt><dd>{typeof value === 'number' ? number(value) : value}</dd></div>)}</dl></article>)}</div>
  </Panel>
}
