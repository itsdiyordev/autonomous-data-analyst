import { Check, LoaderCircle } from '../icons'
import type { Run } from '../types'

const stages = [
  { name: 'Data profiler', short: 'Profile your data', at: 5 },
  { name: 'Problem detector', short: 'Detect the task', at: 12 },
  { name: 'Preprocessing engine', short: 'Prepare the features', at: 20 },
  { name: 'Model selector', short: 'Choose model candidates', at: 25 },
  { name: 'Training engine', short: 'Train the models', at: 28 },
  { name: 'Validation / Cross-validation', short: 'Validate across folds', at: 32 },
  { name: 'Model comparison', short: 'Compare the results', at: 67 },
  { name: 'Best model', short: 'Refit the best model', at: 73 },
  { name: 'Prediction', short: 'Predict unseen rows', at: 79 },
  { name: 'Explainability', short: 'Explain and review errors', at: 84 },
  { name: 'Final analysis report', short: 'Create your final report', at: 94 },
]

export default function Pipeline({ run }: { run: Run }) {
  return <div className="pipeline-view" aria-label="Autonomous ML pipeline">
    <div className="pipeline-view-heading"><strong>Your autonomous ML pipeline</strong><span>11 recorded stages</span></div>
    <ol>{stages.map((stage, index) => {
      const current = run.status === 'running' && run.stage === stage.name
      const completed = run.status === 'completed' || (run.progress > stage.at && !current)
      return <li key={stage.name} className={current ? 'current' : completed ? 'completed' : ''} title={stage.name} aria-current={current ? 'step' : undefined}>
        <span>{current ? <LoaderCircle size={13} className="spin" /> : completed ? <Check size={12} /> : index + 1}</span><strong>{stage.short}</strong>
      </li>
    })}</ol>
  </div>
}
