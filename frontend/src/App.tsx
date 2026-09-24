import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { sendMessage, type ChatResponse } from './api'
import './App.css'

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  metadata?: ChatResponse
}

const suggestions = [
  'Recommend thoughtful sci-fi movies',
  'What movies are trending this week?',
  'Where can I watch Interstellar?',
]

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [region, setRegion] = useState('US')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading, error])

  async function submit(message: string) {
    const trimmed = message.trim()
    if (!trimmed || isLoading) return

    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: 'user', content: trimmed },
    ])
    setDraft('')
    setError(null)
    setIsLoading(true)

    try {
      const result = await sendMessage({ message: trimmed, region })
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          metadata: result,
        },
      ])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The request failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submit(draft)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submit(draft)
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true">✦</span>
          <div>
            <h1>Movie Blasters</h1>
            <p>Find your next great watch</p>
          </div>
        </div>
        <label className="region-picker">
          Watch region
          <select value={region} onChange={(event) => setRegion(event.target.value)}>
            <option value="US">United States</option>
            <option value="CA">Canada</option>
            <option value="GB">United Kingdom</option>
            <option value="IN">India</option>
          </select>
        </label>
      </header>

      <main className="conversation" aria-label="Movie conversation">
        {messages.length === 0 ? (
          <div className="welcome">
            <span className="welcome-icon" aria-hidden="true">✦</span>
            <h2>What are you in the mood for?</h2>
            <p>Ask for movie recommendations, explore similar films, or check what’s trending.</p>
            <div className="suggestions">
              {suggestions.map((suggestion) => (
                <button
                  className="suggestion"
                  key={suggestion}
                  onClick={() => void submit(suggestion)}
                  disabled={isLoading}
                  type="button"
                >
                  {suggestion} <span aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
            {isLoading && <p className="welcome-status" role="status">Finding movies…</p>}
            {error && <p className="error welcome-error" role="alert">{error}</p>}
          </div>
        ) : (
          <div className="message-list" aria-live="polite">
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <span className="message-label">
                  {message.role === 'user' ? 'You' : 'Movie Blasters'}
                </span>
                <div className="message-content">{message.content}</div>
                {message.metadata && (
                  <div className="message-meta">
                    <span>
                      {message.metadata.route === 'rag'
                        ? 'Movie index + Gemma'
                        : message.metadata.route === 'recommendation'
                          ? 'TMDB + Movie index + diversity ranking'
                        : message.metadata.route === 'mcp'
                          ? 'Live TMDB'
                          : message.metadata.route === 'mcp_error'
                            ? 'TMDB unavailable'
                            : 'More detail needed'}
                    </span>
                    {message.metadata.tools_used.length > 0 && (
                      <span>{message.metadata.tools_used.join(', ')}</span>
                    )}
                    <span>{(message.metadata.latency_ms / 1000).toFixed(1)}s</span>
                  </div>
                )}
              </article>
            ))}
            {isLoading && (
              <div className="thinking" role="status">
                <span className="spinner" aria-hidden="true" />
                Finding movies…
              </div>
            )}
            {error && <p className="error" role="alert">{error}</p>}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      <div className="composer-area">
        <form className="composer" onSubmit={handleSubmit}>
          <label htmlFor="movie-query" className="sr-only">Ask about movies</label>
          <textarea
            id="movie-query"
            placeholder="Ask about movies…"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            maxLength={1000}
            rows={2}
          />
          <button type="submit" disabled={!draft.trim() || isLoading} aria-label="Send message">
            ↑
          </button>
        </form>
        <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
      </div>
    </div>
  )
}

export default App
