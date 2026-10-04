import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, ArrowRight, ArrowUpRight, Bell, ChartNoAxesCombined, Command, Database, FlaskConical, LayoutDashboard, LogOut, Menu, Search, Settings as SettingsIcon, Sparkles, X } from './icons'
import { toast } from 'sonner'
import { api } from './api'
import type { DashboardData, User } from './types'
import { Workspace } from './workspace'
import { AnalysisModal, UploadModal } from './components/workflows'
import { Empty, Loading, Modal, Status } from './components/ui'
import Auth from './pages/Auth'
import { goalText } from './presentation'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Datasets = lazy(() => import('./pages/Datasets'))
const DatasetDetail = lazy(() => import('./pages/DatasetDetail'))
const Runs = lazy(() => import('./pages/Runs'))
const RunDetail = lazy(() => import('./pages/RunDetail'))
const Settings = lazy(() => import('./pages/Settings'))

const navigation = [
  { to: '/', label: 'Dashboard', description: 'Your next step', icon: LayoutDashboard },
  { to: '/datasets', label: 'Datasets', description: 'Explore and visualize', icon: Database },
  { to: '/analyses', label: 'Training history', description: 'Follow your model builds', icon: Activity },
  { to: '/models', label: 'Models & predictions', description: 'Use your trained models', icon: FlaskConical },
]

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('analytiq-token'))
  const [upload, setUpload] = useState(false)
  const [analysis, setAnalysis] = useState<{ datasetId?: string } | null>(null)
  const [sidebar, setSidebar] = useState(false)
  const [search, setSearch] = useState(false)
  const [term, setTerm] = useState('')
  const [notifications, setNotifications] = useState(false)
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const location = useLocation()
  const notificationRef = useRef<HTMLDivElement>(null)
  const user = useQuery({ queryKey: ['me', token], queryFn: () => api<User>('/auth/me'), enabled: !!token, retry: false })
  const dashboard = useQuery({
    queryKey: ['dashboard'], queryFn: () => api<DashboardData>('/dashboard'), enabled: !!user.data,
    refetchInterval: query => query.state.data?.runs.some(run => ['queued', 'running'].includes(run.status)) ? 2500 : false,
  })
  const health = useQuery({
    queryKey: ['health'], queryFn: () => api<{ status: string; version: string; llm_enabled: boolean; demo_enabled: boolean }>('/health'),
    enabled: !!user.data, refetchInterval: 60_000,
  })
  const signOut = useCallback(() => {
    localStorage.removeItem('analytiq-token')
    setToken(null)
    queryClient.clear()
    navigate('/')
  }, [queryClient, navigate])
  useEffect(() => {
    const handler = () => { signOut(); toast.error('Your session expired. Please sign in again.') }
    window.addEventListener('session-expired', handler)
    return () => window.removeEventListener('session-expired', handler)
  }, [signOut])
  useEffect(() => { if (user.error) signOut() }, [user.error, signOut])
  useEffect(() => { setSidebar(false); setNotifications(false) }, [location.pathname])
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') { event.preventDefault(); setSearch(value => !value) }
      if (event.key === 'Escape') { setSidebar(false); setNotifications(false) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])
  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (!notificationRef.current?.contains(event.target as Node)) setNotifications(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])
  const closeUpload = useCallback(() => setUpload(false), [])
  const closeAnalysis = useCallback(() => setAnalysis(null), [])
  const closeSearch = useCallback(() => { setSearch(false); setTerm('') }, [])

  if (!token) return <Auth authenticated={(value, profile) => {
    localStorage.setItem('analytiq-token', value)
    setToken(value)
    queryClient.setQueryData(['me', value], profile)
    navigate('/')
  }} />
  if (!user.data) return <Loading />

  const current = location.pathname.startsWith('/runs') ? 'Model results'
    : navigation.find(item => item.to !== '/' && location.pathname.startsWith(item.to))?.label
    || (location.pathname === '/settings' ? 'Settings' : 'Dashboard')
  const datasets = dashboard.data?.datasets || []
  const runs = dashboard.data?.runs || []
  const matchingDatasets = datasets.filter(dataset => `${dataset.name} ${dataset.filename}`.toLowerCase().includes(term.toLowerCase()))
  const matchingRuns = runs.filter(run => `${run.objective} ${run.target}`.toLowerCase().includes(term.toLowerCase())).slice(0, 6)
  const startModel = (datasetId?: string) => setAnalysis({ datasetId })

  return <Workspace.Provider value={{ user: user.data, openUpload: () => setUpload(true), openAnalysis: startModel, signOut }}>
    <div className="app-shell">
      {sidebar && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setSidebar(false)} />}
      <aside className={`sidebar ${sidebar ? 'open' : ''}`}>
        <NavLink to="/" className="brand"><span className="brand-symbol"><ChartNoAxesCombined size={23} /></span>analytiq<span className="brand-dot">.</span></NavLink>
        <div className="workspace-switch">
          <span className="workspace-avatar">{user.data.name[0].toUpperCase()}</span>
          <div><strong>{user.data.name}</strong><small>Personal workspace</small></div>
          <span className="workspace-indicator" />
        </div>
        <div className="nav-label">YOUR WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map(item => <NavLink key={item.to} to={item.to} end={item.to === '/'} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <item.icon size={20} /><span><strong>{item.label}</strong><small>{item.description}</small></span>
            {item.to === '/datasets' && datasets.length > 0 && <b>{datasets.length}</b>}
          </NavLink>)}
        </nav>
        <div className="sidebar-insight">
          <span className="sidebar-insight-icon"><Sparkles size={19} /></span>
          <h4>Make your first prediction</h4>
          <p>Choose your data. Tell us what to predict. We’ll compare the models.</p>
          <button onClick={() => startModel()}>Train a model<ArrowRight size={15} /></button>
        </div>
        <div className="sidebar-bottom">
          <NavLink to="/settings" className={({ isActive }) => `nav-item compact ${isActive ? 'active' : ''}`}><SettingsIcon size={19} /><strong>Settings</strong></NavLink>
          <button className="nav-item compact" onClick={signOut}><LogOut size={19} /><strong>Sign out</strong></button>
          <div className="sidebar-version"><span><i className={`live-dot ${health.error ? 'offline' : ''}`} />{health.error ? 'Connection unavailable' : health.data ? 'Connected to your engine' : 'Connecting…'}</span><span>v1.0</span></div>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setSidebar(true)}><Menu size={21} /></button>
            <span className="breadcrumb">Workspace <span>/</span><strong>{current}</strong></span>
          </div>
          <div className="topbar-right">
            <button className="search-trigger" onClick={() => setSearch(true)} aria-label="Search workspace"><Search size={17} /><span>Search datasets and models</span><kbd><Command size={11} /> K</kbd></button>
            <div className="notifications" ref={notificationRef}>
              <button className="icon-button" aria-label="Notifications" aria-expanded={notifications} onClick={() => setNotifications(!notifications)}><Bell size={20} />{!!dashboard.data?.stats.active_runs && <i className="notification-dot" />}</button>
              {notifications && <div className="notification-menu"><h4>Latest training activity</h4>
                {runs.length ? runs.slice(0, 5).map(run => <button key={run.id} onClick={() => navigate(`/runs/${run.id}`)}>
                  <span className={`activity-dot ${run.status}`} /><div><strong>{goalText(run)}</strong><small>{run.stage}</small></div><Status status={run.status} />
                </button>) : <p>No models in training yet. Start with a dataset.</p>}
              </div>}
            </div>
            <div className="topbar-divider" />
            <NavLink className="user-avatar" to="/settings" aria-label="Your profile">{user.data.name.split(' ').map(name => name[0]).slice(0, 2).join('').toUpperCase()}</NavLink>
          </div>
        </header>
        <main className="page-content">
          <Suspense fallback={<Loading text="Opening this page…" />}>
            <Routes>
              <Route path="/" element={<Dashboard />} /><Route path="/datasets" element={<Datasets />} />
              <Route path="/datasets/:id" element={<DatasetDetail />} /><Route path="/analyses" element={<Runs />} />
              <Route path="/models" element={<Runs modelsOnly />} /><Route path="/runs/:id" element={<RunDetail />} />
              <Route path="/settings" element={<Settings />} /><Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
          <footer className="page-footer"><span><ChartNoAxesCombined size={14} />From data to decisions, one clear step at a time.</span><span>ANALYTIQ · YOUR DATA WORKSPACE</span></footer>
        </main>
      </div>
    </div>

    {upload && <UploadModal close={closeUpload} />}
    {analysis && <AnalysisModal initialDataset={analysis.datasetId} close={closeAnalysis} />}
    {search && <Modal title="Search your workspace" subtitle="Find a dataset or a previous model build." close={closeSearch}>
      <div className="search-input full-search"><Search size={18} /><input autoFocus placeholder="Search by name, file, or prediction target…" value={term} onChange={event => setTerm(event.target.value)} /><button className="icon-button" aria-label="Clear search" onClick={() => setTerm('')}><X size={16} /></button></div>
      <div className="search-results">
        {matchingDatasets.length > 0 && <div className="search-group-label">DATASETS</div>}
        {matchingDatasets.map(dataset => <button key={dataset.id} onClick={() => { closeSearch(); navigate(`/datasets/${dataset.id}`) }}><Database size={19} /><div><strong>{dataset.name}</strong><small>{dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns</small></div><ArrowUpRight size={16} /></button>)}
        {matchingRuns.length > 0 && <div className="search-group-label">MODEL BUILDS</div>}
        {matchingRuns.map(run => <button key={run.id} onClick={() => { closeSearch(); navigate(`/runs/${run.id}`) }}><Activity size={19} /><div><strong>{goalText(run)}</strong><small>{run.objective}</small></div><Status status={run.status} /></button>)}
        {!matchingDatasets.length && !matchingRuns.length && <Empty title="No matching results" text="Try a dataset name or the column you want to predict." />}
      </div>
    </Modal>}
  </Workspace.Provider>
}
