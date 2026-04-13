"use client"

import Link from "next/link"
import { useEffect, useState } from "react"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000"

type QuestionInsight = {
  feature: string
  value: string
  count: number
  success_rate: number
  avg_position: number
}

export default function InsightsPage() {
  const [questions, setQuestions] = useState<QuestionInsight[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_URL}/insights`)
        const data = await res.json()
        setQuestions(data.questions ?? [])
      } catch {
        setQuestions([])
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return (
    <main className="min-h-screen py-10 flex justify-center">
      <div className="w-full max-w-5xl px-4 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-serif">Question Insights</h1>
          <Link
            href="/"
            className="text-sm text-indigo-300 hover:text-indigo-200 transition"
          >
            Back
          </Link>
        </div>

        <p className="text-sm text-muted-foreground">
          Learned from past games. Higher success rate and earlier average
          position generally indicates better questions.
        </p>

        {loading && <p>Loading...</p>}
        {!loading && questions.length === 0 && (
          <p className="text-muted-foreground">No data yet.</p>
        )}

        {!loading && questions.length > 0 && (
          <div className="overflow-auto border border-slate-800 rounded-lg">
            <table className="w-full text-sm border-collapse">
              <thead className="bg-slate-950/60">
                <tr className="border-b border-slate-800">
                  <th className="py-3 px-3 text-left">Feature</th>
                  <th className="py-3 px-3 text-left">Value</th>
                  <th className="py-3 px-3 text-right">Asked</th>
                  <th className="py-3 px-3 text-right">Success</th>
                  <th className="py-3 px-3 text-right">Avg pos</th>
                </tr>
              </thead>
              <tbody>
                {questions.slice(0, 50).map((q) => (
                  <tr
                    key={`${q.feature}:${q.value}`}
                    className="border-b border-slate-900 hover:bg-slate-900/30 transition"
                  >
                    <td className="py-2 px-3">{q.feature}</td>
                    <td className="py-2 px-3">{q.value}</td>
                    <td className="py-2 px-3 text-right">
                      {Number.isFinite(q.count) ? q.count.toFixed(0) : "0"}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {(q.success_rate * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right">
                      {Number.isFinite(q.avg_position)
                        ? q.avg_position.toFixed(1)
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  )
}

