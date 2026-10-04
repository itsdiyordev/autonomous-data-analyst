import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import { AlertCircle, ArrowUpRight, Database, LoaderCircle, X } from '../icons'
import { Link } from 'react-router-dom'
import { statusText } from '../presentation'

export function Loading({ text = 'Loading your workspace…' }: { text?: string }) {
  return <div className="loading"><LoaderCircle className="spin" size={25} /><span>{text}</span></div>
}
export function ErrorState({ error, retry }: { error: Error | null; retry?: () => void }) {
  return <div className="empty"><AlertCircle size={34} /><h3>We couldn’t complete that</h3><p>{error?.message || 'Please try again.'}</p>{retry && <button className="button secondary" onClick={retry}>Try again</button>}</div>
}
export function Empty({ title, text, action }: { title: string; text: string; action?: ReactNode }) {
  return <div className="empty"><div className="empty-icon"><Database size={28} /></div><h3>{title}</h3><p>{text}</p>{action}</div>
}
export function Status({ status }: { status: string }) {
  return <span className={`status ${status}`}><i />{statusText(status)}</span>
}
export function PageHeading({ eyebrow, title, text, actions }: { eyebrow?: string; title: string; text: string; actions?: ReactNode }) {
  return <div className="page-heading"><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h1>{title}</h1><p>{text}</p></div><div className="heading-actions">{actions}</div></div>
}
export function Panel({ title, subtitle, action, children, className = '' }: { title?: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{title && <div className="panel-heading"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{action}</div>}{children}</section>
}
export function Modal({ title, subtitle, close, children, wide = false }: { title: string; subtitle: string; close: () => void; children: ReactNode; wide?: boolean }) {
  const dialog = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const oldOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialog.current?.querySelector<HTMLElement>('button, input, select, textarea')?.focus()
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close()
      if (event.key === 'Tab') {
        const nodes = dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, textarea, a[href]')
        if (!nodes?.length) return
        const first = nodes[0], last = nodes[nodes.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
      }
    }
    window.addEventListener('keydown', handler)
    return () => { document.body.style.overflow = oldOverflow; window.removeEventListener('keydown', handler); previous?.focus() }
  }, [close])
  return <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) close() }}><div ref={dialog} className={`modal ${wide ? 'wide' : ''}`} role="dialog" aria-modal="true" aria-label={title}><div className="modal-header"><div><h2>{title}</h2><p>{subtitle}</p></div><button className="icon-button" onClick={close} aria-label="Close dialog"><X size={20} /></button></div>{children}</div></div>
}
export function ViewLink({ to, children }: { to: string; children: ReactNode }) {
  return <Link to={to} className="text-link">{children}<ArrowUpRight size={15} /></Link>
}
