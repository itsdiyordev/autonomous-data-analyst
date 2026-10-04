import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { CheckCircle2, Cpu, ExternalLink, LogOut, Moon, ShieldCheck, Sun, UserRound } from '../icons'
import { api } from '../api'
import { useWorkspace } from '../workspace'
import { PageHeading, Panel } from '../components/ui'

export default function Settings() {
  const { user, signOut } = useWorkspace()
  const [dark, setDark] = useState(localStorage.getItem('analytiq-theme') === 'dark')
  const health = useQuery({ queryKey: ['health'], queryFn: () => api<{ status: string; version: string; llm_enabled: boolean; demo_enabled: boolean }>('/health') })
  function theme(value: boolean) { setDark(value); localStorage.setItem('analytiq-theme', value ? 'dark' : 'light'); document.documentElement.dataset.theme = value ? 'dark' : 'light' }
  return <>
    <PageHeading eyebrow="MAKE YOURSELF AT HOME" title="A workspace that feels like you." text="Your profile, preferences, and connected intelligence." />
    <div className="settings-grid">
      <Panel title="Your profile" subtitle="The person behind the discoveries.">
        <div className="profile-card"><div className="profile-avatar"><UserRound size={28} /></div><div><h3>{user.name}</h3><p>{user.email}</p></div></div>
        <div className="settings-detail"><span>Workspace</span><strong>Personal analytics</strong></div>
        <div className="settings-detail"><span>Session</span><strong><ShieldCheck size={14} />Authenticated · 24-hour token</strong></div>
        <button className="button secondary" onClick={signOut}><LogOut size={15} />Sign out of workspace</button>
      </Panel>
      <Panel title="The right atmosphere" subtitle="Choose a look for your next deep dive.">
        <div className="theme-options">
          <button className={!dark ? 'selected' : ''} onClick={() => theme(false)}><div className="theme-preview light"><i /><span /><span /><b /></div><span><Sun size={16} />Light & focused{!dark && <CheckCircle2 size={15} />}</span></button>
          <button className={dark ? 'selected' : ''} onClick={() => theme(true)}><div className="theme-preview dark"><i /><span /><span /><b /></div><span><Moon size={16} />Dark & immersive{dark && <CheckCircle2 size={15} />}</span></button>
        </div>
      </Panel>
      <Panel title="Your intelligence engine" subtitle="A clear view of what powers the workspace.">
        <div className="settings-detail"><span><Cpu size={15} />Backend API</span><strong className={health.data ? 'text-green' : 'text-amber'}>{health.data ? 'Connected' : health.error ? 'Unavailable' : 'Checking…'}</strong></div>
        <div className="settings-detail"><span>Machine learning</span><strong>scikit-learn · CPU optimized</strong></div>
        <div className="settings-detail"><span>Language model</span><strong>{health.data?.llm_enabled ? 'Connected' : 'Computed-evidence mode'}</strong></div>
        <div className="settings-detail"><span>Application version</span><strong>{health.data?.version || '1.0.0'}</strong></div>
        <a className="button secondary" href={`${import.meta.env.VITE_API_URL || ''}/docs`} target="_blank" rel="noreferrer">Explore API documentation<ExternalLink size={15} /></a>
      </Panel>
      <Panel title="Designed around your data" subtitle="Transparency, at every step.">
        <div className="settings-principles">{['Original datasets are preserved as versioned artifacts.', 'Preprocessing is fitted on development data, never the test set.', 'Model decisions include a reproducible evaluation record.', 'Your exported solution includes its complete input contract.'].map(text => <div key={text}><ShieldCheck size={18} /><p>{text}</p></div>)}</div>
      </Panel>
    </div>
  </>
}
