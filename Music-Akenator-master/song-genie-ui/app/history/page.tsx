"use client"

import Link from "next/link"
import { useEffect, useState } from "react"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000"

type SessionSummary = {
  session_id: string
  final_song_id: number | null
  confidence: number | null
  success: boolean | null
  created_at?: string
  correct_song_title?: string
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_URL}/sessions`)
        const data = await res.json()
        setSessions(data.sessions ?? [])
      } catch {
        setSessions([])
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return (
    <main className="min-h-screen py-10 flex justify-center">
      <div className="w-full max-w-3xl px-4 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-serif">Past Sessions</h1>
          <Link
            href="/"
            className="text-sm text-indigo-300 hover:text-indigo-200 transition"
          >
            Back
          </Link>
        </div>

        {loading && <p>Loading...</p>}
        {!loading && sessions.length === 0 && (
          <p className="text-muted-foreground">No sessions logged yet.</p>
        )}

        <ul className="space-y-4">
          {sessions.map((s) => (
            <li
              key={s.session_id}
              className="border border-slate-700 rounded-lg p-4 flex flex-col gap-2 bg-slate-900/40"
            >
              <div className="flex justify-between items-center gap-4">
                <span className="font-mono text-xs text-slate-400">
                  {s.session_id.slice(0, 8)}…
                </span>
                <span className="text-sm">
                  {s.success === true
                    ? "Correct"
                    : s.success === false
                    ? "Incorrect"
                    : "Unknown"}
                </span>
              </div>
              <div className="text-sm text-slate-200">
                <span className="font-semibold">Confidence:</span>{" "}
                {s.confidence != null ? `${Math.round(s.confidence * 100)}%` : "—"}
              </div>
              {s.created_at && (
                <div className="text-xs text-slate-400">
                  {s.created_at}
                </div>
              )}
              {s.correct_song_title && (
                <div className="text-sm text-slate-200">
                  <span className="font-semibold">Correct song:</span>{" "}
                  {s.correct_song_title}
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </main>
  )
}

