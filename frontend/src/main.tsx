import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import App from './App'
import './styles.css'

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 15000, refetchOnWindowFocus: false } } })
const Feedback = import.meta.env.DEV ? React.lazy(() => import('agentation').then(module => ({ default: module.Agentation }))) : null
if (localStorage.getItem('analytiq-theme') === 'dark') document.documentElement.dataset.theme = 'dark'

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed ? <div className="fatal"><h1>Something went wrong.</h1><p>Reload the workspace to recover.</p><button className="button primary" onClick={() => location.reload()}>Reload workspace</button></div> : this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><ErrorBoundary><QueryClientProvider client={client}><BrowserRouter><App /></BrowserRouter><Toaster richColors position="bottom-right" />{Feedback && <React.Suspense fallback={null}><Feedback appName="Analytiq UI redesign" /></React.Suspense>}</QueryClientProvider></ErrorBoundary></React.StrictMode>,
)
