"use client"

import { useState } from "react"

interface AnswerButtonsProps {
  onAnswer: (answer: "yes" | "no" | "unsure") => void
  disabled: boolean
}

const BUTTON_CONFIG = [
  {
    key: "yes" as const,
    label: "Yes",
    color: "hsl(145, 80%, 50%)",
    glowColor: "rgba(34, 197, 94, 0.4)",
    hoverGlow: "rgba(34, 197, 94, 0.6)",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    ),
  },
  {
    key: "no" as const,
    label: "No",
    color: "hsl(0, 85%, 55%)",
    glowColor: "rgba(239, 68, 68, 0.4)",
    hoverGlow: "rgba(239, 68, 68, 0.6)",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
      </svg>
    ),
  },
  {
    key: "unsure" as const,
    label: "Unsure",
    color: "hsl(40, 90%, 55%)",
    glowColor: "rgba(245, 158, 11, 0.4)",
    hoverGlow: "rgba(245, 158, 11, 0.6)",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
  },
]

export function AnswerButtons({ onAnswer, disabled }: AnswerButtonsProps) {
  const [flashKey, setFlashKey] = useState<string | null>(null)

  const handleClick = (key: "yes" | "no" | "unsure") => {
    if (disabled) return
    setFlashKey(key)
    onAnswer(key)
    setTimeout(() => setFlashKey(null), 400)
  }

  return (
    <div className="flex items-center justify-center gap-4 sm:gap-6" role="group" aria-label="Answer options">
      {BUTTON_CONFIG.map((btn) => {
        const isFlashing = flashKey === btn.key

        return (
          <button
            key={btn.key}
            type="button"
            onClick={() => handleClick(btn.key)}
            disabled={disabled}
            aria-label={btn.label}
            className="group relative flex flex-col items-center gap-2 transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 rounded-2xl"
            style={{
              ["--btn-glow" as string]: btn.glowColor,
              outline: "none",
              focusRingColor: btn.color,
            }}
          >
            {/* Button circle */}
            <div
              className={`
                relative flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 rounded-full
                transition-all duration-300
                ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}
                ${!disabled ? "group-hover:scale-110" : ""}
                ${isFlashing ? "scale-125" : ""}
              `}
              style={{
                background: `radial-gradient(circle at 40% 35%, ${btn.color}dd, ${btn.color}88)`,
                boxShadow: isFlashing
                  ? `0 0 40px 12px ${btn.hoverGlow}, inset 0 0 20px rgba(255,255,255,0.2)`
                  : `0 0 20px 4px ${btn.glowColor}, inset 0 0 10px rgba(255,255,255,0.1)`,
                border: `2px solid ${btn.color}`,
                animation: !disabled && !isFlashing ? "button-pulse 2.5s ease-in-out infinite" : "none",
                animationDelay: btn.key === "no" ? "0.8s" : btn.key === "unsure" ? "1.6s" : "0s",
                transition: "all 0.3s ease, transform 0.2s ease",
              }}
            >
              <span className="text-foreground" style={{ color: "#0a0a1a" }}>
                {btn.icon}
              </span>
            </div>

            {/* Label */}
            <span
              className={`
                font-serif text-xs sm:text-sm tracking-widest uppercase
                transition-all duration-300
                ${disabled ? "opacity-40" : ""}
              `}
              style={{
                color: btn.color,
                textShadow: `0 0 10px ${btn.glowColor}`,
              }}
            >
              {btn.label}
            </span>
          </button>
        )
      })}
    </div>
  )
}
