import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import type { Dataset, Leakage } from '../types'
import { ErrorState, Loading, Panel } from './ui'

export default function DataQuality({ dataset }: { dataset: Dataset }) {
  const profile = dataset.profile!
  const [target, setTarget] = useState('')
  const review = useQuery({ queryKey: ['quality', dataset.id, target], queryFn: () => api<{ leakage: Leakage }>(`/datasets/${dataset.id}/quality${target ? `?target=${encodeURIComponent(target)}` : ''}`), enabled: !!target })
  const issues = [
    { name: 'Missing values', value: `${profile.missing_cells.toLocaleString()} cells (${profile.missing_pct}%)`, action: 'Confirm why values are missing. Learned imputation applies only inside predictive pipelines.' },
    { name: 'Exact duplicates', value: profile.duplicates.toLocaleString(), action: 'Confirm whether repeats are legitimate. Original records remain saved; ML duplicate handling is explicitly recorded.' },
    { name: 'Potential IQR outliers', value: (profile.outlier_rows || 0).toLocaleString(), action: 'Review unusual observations. Outliers and anomalies are never silently deleted.' },
    { name: 'Invalid/non-finite values', value: (profile.invalid_values || 0).toLocaleString(), action: 'Recorded non-finite inputs are represented as missing values; verify the ingestion note.' },
    { name: 'Constant / near-constant columns', value: [...new Set([...(profile.constant_columns || []), ...(profile.near_constant_columns || [])])].join(', ') || 'None detected', action: 'Review whether these fields carry useful information. Structural ML exclusions are explained per run.' },
    { name: 'Potential identifiers', value: profile.possible_ids?.join(', ') || 'None detected', action: 'Identifier detection is heuristic. Confirm entity identifiers and use group splits when records are related.' },
    { name: 'High-cardinality / free text', value: [...new Set([...(profile.high_cardinality_columns || []), ...(profile.free_text_columns || [])])].join(', ') || 'None detected', action: 'High-cardinality features can inflate encoding; unsupported text features are explicitly listed in model exclusions.' },
  ]
  return <><Panel title="Data quality center" subtitle={`Overall quality: ${profile.quality_score}/100. The score summarizes completeness and duplicates; it is not a full analytical reliability score.`}><div className="quality-issues">{issues.map(issue => <article key={issue.name}><div><strong>{issue.name}</strong><span>{issue.value}</span></div><p>{issue.action}</p></article>)}</div></Panel><Panel title="Potential leakage by analytical outcome" subtitle="Choose a target to review features that may contain outcome-derived or post-event information."><label className="field">Review target<select value={target} onChange={event => setTarget(event.target.value)}><option value="">Choose an outcome</option>{profile.columns.map(column => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>{review.isFetching ? <Loading text="Reviewing potential leakage…" /> : review.error ? <ErrorState error={review.error} /> : review.data && <>{review.data.leakage.flags.length ? review.data.leakage.flags.map(flag => <article className="evidence-record" key={flag.feature}><strong>{flag.feature}</strong><p>{flag.reasons.join(' ')}</p><small>{flag.action}</small></article>) : <p className="field-help">No supported checks flagged a feature. Domain confirmation of prediction-time availability is still required.</p>}<div className="info-note">{review.data.leakage.policy}</div></>}</Panel></>
}
