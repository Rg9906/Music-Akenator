"use client"

import { useEffect, useState } from "react"

interface SpeechBubbleProps {
  message: string
  isThinking: boolean
}

export function SpeechBubble({ message, isThinking }: SpeechBubbleProps) {
  const [displayedMessage, setDisplayedMessage] = useState("")
  const [isTyping, setIsTyping] = useState(false)

  useEffect(() => {
    if (!message) {
      setDisplayedMessage("")
      return
    }

    setIsTyping(true)
    setDisplayedMessage("")
    let index = 0

    const interval = setInterval(() => {
      if (index < message.length) {
        setDisplayedMessage(message.slice(0, index + 1))
        index++
      } else {
        setIsTyping(false)
        clearInterval(interval)
      }
    }, 30)

    return () => clearInterval(interval)
  }, [message])

  return (
    <div className="animate-genie-float" style={{ animationDuration: "5s", animationDelay: "1s" }}>
      <div
        className="relative px-6 py-4 rounded-2xl max-w-sm mx-auto text-center"
        style={{
          background: "rgba(139, 92, 246, 0.1)",
          border: "1px solid rgba(139, 92, 246, 0.25)",
          backdropFilter: "blur(12px)",
          boxShadow: "0 0 30px rgba(139, 92, 246, 0.1), inset 0 0 20px rgba(139, 92, 246, 0.05)",
        }}
      >
        {/* Small triangle pointing up toward genie */}
        <div
          className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 rotate-45"
          style={{
            background: "rgba(139, 92, 246, 0.15)",
            border: "1px solid rgba(139, 92, 246, 0.25)",
            borderBottom: "none",
            borderRight: "none",
          }}
          aria-hidden="true"
        />

        {isThinking ? (
          <div className="flex items-center justify-center gap-2 py-1" role="status" aria-label="Genie is thinking">
            <span className="sr-only">Thinking</span>
            {[0, 1, 2].map((i) => (
              <span
                key={`dot-${i}`}
                className="inline-block w-2.5 h-2.5 rounded-full animate-thinking-dots"
                style={{
                  backgroundColor: "hsl(260, 70%, 65%)",
                  animationDelay: `${i * 0.2}s`,
                }}
              />
            ))}
          </div>
        ) : (
          <p
            className="font-sans text-sm leading-relaxed tracking-wide"
            style={{
              color: "rgba(200, 180, 255, 0.9)",
              minHeight: "1.5em",
            }}
          >
            {displayedMessage}
            {isTyping && (
              <span
                className="inline-block w-0.5 h-4 ml-0.5 align-text-bottom"
                style={{
                  backgroundColor: "hsl(260, 70%, 65%)",
                  animation: "pulse-glow 0.8s ease-in-out infinite",
                }}
                aria-hidden="true"
              />
            )}
          </p>
        )}
      </div>
    </div>
  )
}
