import { useState } from 'react'
import { ArrowRight, ChartNoAxesCombined, Check, Database, FlaskConical, LoaderCircle, ShieldCheck, Sparkles } from '../icons'
import { api } from '../api'
import type { User } from '../types'

export default function Auth({ authenticated }: { authenticated: (token: string, user: User) => void }) {
  const [register, setRegister] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  async function submit(demo = false) {
    setError('')
    setBusy(demo ? 'demo' : 'auth')
    try {
      const result = await api<{ token: string; user: User }>(`/auth/${demo ? 'demo' : register ? 'register' : 'login'}`, {
        method: 'POST', body: demo ? undefined : JSON.stringify({ name, email, password }),
      })
      authenticated(result.token, result.user)
    } catch (err) { setError((err as Error).message) } finally { setBusy('') }
  }

  return <div className="auth-page">
    <section className="auth-story">
      <div className="brand"><span className="brand-symbol"><ChartNoAxesCombined size={24} /></span>analytiq<span className="brand-dot">.</span></div>
      <div className="auth-story-content">
        <span className="auth-story-tag"><Sparkles size={14} />YOUR DATA, WITH A LITTLE MORE CLARITY</span>
        <h1>Understand your data.<br /><em>Build something useful.</em></h1>
        <p>One workspace to explore your spreadsheets, train prediction models, and turn the results into action.</p>
        <div className="auth-workflow-card">
          <div className="auth-preview-heading"><span><ChartNoAxesCombined size={17} />A clearer way to work with data</span><span className="auth-preview-badge">ALL IN ONE PLACE</span></div>
          {[
            { title: 'Upload your spreadsheet', text: 'Start with a CSV or Excel file.', icon: Database },
            { title: 'Explore the story', text: 'Understand your columns, charts, and relationships.', icon: ChartNoAxesCombined },
            { title: 'Make your first prediction', text: 'Compare models, explain results, and try new data.', icon: FlaskConical },
          ].map((step, index) => <div className="auth-preview-step" key={step.title}><span className="auth-preview-icon"><step.icon size={20} /></span><div><small>0{index + 1}</small><strong>{step.title}</strong><p>{step.text}</p></div><ArrowRight size={16} /></div>)}
          <div className="auth-preview-footer"><ShieldCheck size={15} />Every answer is grounded in your data.</div>
        </div>
      </div>
      <div className="auth-story-footer"><span>EXPLORE · TRAIN · PREDICT</span><span>Your next decision starts here.</span></div>
    </section>

    <section className="auth-form-side">
      <div className="auth-form">
        <span className="auth-welcome-icon"><ChartNoAxesCombined size={27} /></span>
        <div className="eyebrow">WELCOME TO ANALYTIQ</div>
        <h2>{register ? 'Create your workspace' : 'Let’s get you started.'}</h2>
        <p>{register ? 'Keep your data, models, and results in one place.' : 'Sign in to your account, or explore a ready-made demo.'}</p>
        <form onSubmit={event => { event.preventDefault(); submit() }}>
          {register && <label className="field">Full name<input value={name} onChange={event => setName(event.target.value)} required minLength={2} autoComplete="name" placeholder="Alex Morgan" /></label>}
          <label className="field">Email address<input type="email" value={email} onChange={event => setEmail(event.target.value)} autoComplete="email" required placeholder="you@example.com" /></label>
          <label className="field">Password<input type="password" value={password} onChange={event => setPassword(event.target.value)} autoComplete={register ? 'new-password' : 'current-password'} required minLength={register ? 8 : 1} placeholder={register ? 'At least 8 characters' : 'Enter your password'} /></label>
          {error && <div className="inline-error" role="alert">{error}</div>}
          <button className="button primary full" disabled={!!busy}>{busy === 'auth' && <LoaderCircle className="spin" size={17} />}{register ? 'Create account' : 'Sign in'}<ArrowRight size={17} /></button>
        </form>
        <div className="auth-divider"><span>Want to explore first?</span></div>
        <button className="button demo-button full" disabled={!!busy} onClick={() => submit(true)}>{busy === 'demo' ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}{busy === 'demo' ? 'Preparing your demo…' : 'Try the live demo'}<ArrowRight size={16} /></button>
        <div className="auth-note"><Check size={14} />Two sample datasets. No account setup needed.</div>
        <p className="auth-switch">{register ? 'Already have an account?' : 'New here?'}<button onClick={() => { setRegister(!register); setError('') }}>{register ? 'Sign in' : 'Create an account'}</button></p>
      </div>
      <span className="auth-bottom-note"><ShieldCheck size={14} />Your own workspace. Your own discoveries.</span>
    </section>
  </div>
}
