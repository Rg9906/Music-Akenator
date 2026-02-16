"use client"

import { useEffect, useState } from "react"

interface QuestionDisplayProps {
  question: string
  questionNumber: number
  totalQuestions: number
}

export function QuestionDisplay({
  question,
  questionNumber,
  totalQuestions,
}: QuestionDisplayProps) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    setVisible(false)
    const timer = setTimeout(() => setVisible(true), 100)
    return () => clearTimeout(timer)
  }, [question])

  return (
    <div className="text-center px-4">
      {/* Progress indicator */}
      <div className="flex items-center justify-center gap-2 mb-4">
        <span
          className="font-sans text-xs tracking-widest uppercase"
          style={{ color: "rgba(139, 92, 246, 0.6)" }}
        >
          Question {questionNumber} of {totalQuestions}
        </span>
      </div>

      {/* Progress bar */}
      <div
        className="mx-auto mb-6 h-0.5 rounded-full overflow-hidden"
        style={{
          maxWidth: "200px",
          background: "rgba(139, 92, 246, 0.15)",
        }}
        role="progressbar"
        aria-valuenow={questionNumber}
        aria-valuemin={1}
        aria-valuemax={totalQuestions}
      >
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${(questionNumber / totalQuestions) * 100}%`,
            background: "linear-gradient(90deg, hsl(260, 70%, 60%), hsl(260, 80%, 75%))",
            boxShadow: "0 0 10px rgba(139, 92, 246, 0.5)",
          }}
        />
      </div>

      {/* Question text */}
      <h2
        className={`
          font-serif text-xl sm:text-2xl md:text-3xl leading-relaxed text-balance
          transition-all duration-500
          ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"}
        `}
        style={{
          color: "rgba(220, 210, 255, 0.95)",
          textShadow: "0 0 30px rgba(139, 92, 246, 0.3)",
        }}
      >
        {question}
      </h2>
    </div>
  )
}
