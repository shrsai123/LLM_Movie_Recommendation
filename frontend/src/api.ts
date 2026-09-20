export type ChatRequest = {
  message: string
  region: string
}

export type ChatResponse = {
  answer: string
  route: string
  intent: string | null
  tools_used: string[]
  sources: string[]
  latency_ms: number
}

// Vite proxies /api to FastAPI in development. Set VITE_API_URL for a separate
// production API origin; never place API secrets in Vite environment variables.
const apiUrl = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

export async function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
  let response: Response

  try {
    response = await fetch(`${apiUrl}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new Error('Cannot reach the movie API. Check that FastAPI is running on port 8000.')
  }

  if (!response.ok) {
    throw new Error(
      response.status === 422
        ? 'Enter a movie question and try again.'
        : 'The movie API could not process your request. Please try again.',
    )
  }

  return (await response.json()) as ChatResponse
}
