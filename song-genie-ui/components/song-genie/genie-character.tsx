"use client"

import Image from "next/image"
import type { GenieState } from "./types"

interface GenieCharacterProps {
  state: GenieState
}

export function GenieCharacter({ state }: GenieCharacterProps) {
  const isThinking = state === "thinking"
  const isYes = state === "yes"
  const isNo = state === "no"
  const isUnsure = state === "unsure"

  const glowColor = isYes
    ? "rgba(34, 197, 94, 0.5)"
    : isNo
      ? "rgba(239, 68, 68, 0.5)"
      : isUnsure
        ? "rgba(245, 158, 11, 0.5)"
        : "rgba(139, 92, 246, 0.45)"

  const borderGlow = isYes
    ? "rgba(34, 197, 94, 0.7)"
    : isNo
      ? "rgba(239, 68, 68, 0.7)"
      : isUnsure
        ? "rgba(245, 158, 11, 0.7)"
        : "rgba(139, 92, 246, 0.6)"

  const auraColor = isYes
    ? "#22c55e"
    : isNo
      ? "#ef4444"
      : isUnsure
        ? "#f59e0b"
        : "#8b5cf6"

  return (
    <div
      className={`relative flex items-center justify-center ${
        isUnsure ? "animate-genie-wobble" : "animate-genie-float"
      }`}
      style={{
        animationDuration: isThinking ? "2s" : "4s",
      }}
    >
      {/* Outer aura ring */}
      <div
        className="absolute rounded-full animate-pulse-glow"
        style={{
          width: "340px",
          height: "380px",
          background: `radial-gradient(circle, ${glowColor} 0%, transparent 65%)`,
          filter: "blur(40px)",
          transition: "all 0.6s ease",
        }}
        aria-hidden="true"
      />

      {/* Inner concentrated glow */}
      <div
        className="absolute rounded-full animate-pulse-glow"
        style={{
          width: "220px",
          height: "260px",
          background: `radial-gradient(circle, ${glowColor} 0%, transparent 50%)`,
          filter: "blur(20px)",
          animationDelay: "0.5s",
          transition: "all 0.6s ease",
        }}
        aria-hidden="true"
      />

      {/* SVG smoke tail behind portrait */}
      <svg
        width="260"
        height="160"
        viewBox="0 0 260 160"
        fill="none"
        className="absolute z-0"
        style={{ bottom: "-80px" }}
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="smokeTail" x1="130" y1="0" x2="130" y2="160" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor={auraColor} stopOpacity="0.6" />
            <stop offset="50%" stopColor={auraColor} stopOpacity="0.2" />
            <stop offset="100%" stopColor={auraColor} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path
          d="M90 0 Q75 40 85 80 Q100 120 130 150 Q160 120 175 80 Q185 40 170 0"
          fill="url(#smokeTail)"
        >
          <animate
            attributeName="d"
            dur={isThinking ? "1.5s" : "4s"}
            repeatCount="indefinite"
            values="
              M90 0 Q75 40 85 80 Q100 120 130 150 Q160 120 175 80 Q185 40 170 0;
              M85 0 Q68 45 88 85 Q105 125 130 155 Q155 125 172 85 Q192 45 175 0;
              M90 0 Q75 40 85 80 Q100 120 130 150 Q160 120 175 80 Q185 40 170 0
            "
          />
        </path>
        {/* secondary wisp */}
        <path
          d="M105 5 Q95 50 100 90 Q115 130 130 145 Q145 130 160 90 Q165 50 155 5"
          fill={auraColor}
          opacity="0.15"
        >
          <animate
            attributeName="d"
            dur="3.5s"
            repeatCount="indefinite"
            values="
              M105 5 Q95 50 100 90 Q115 130 130 145 Q145 130 160 90 Q165 50 155 5;
              M100 5 Q88 55 102 95 Q118 135 130 148 Q142 135 158 95 Q172 55 160 5;
              M105 5 Q95 50 100 90 Q115 130 130 145 Q145 130 160 90 Q165 50 155 5
            "
          />
        </path>
      </svg>

      {/* Main genie portrait */}
      <div
        className="relative z-10 overflow-hidden"
        style={{
          width: "240px",
          height: "300px",
          borderRadius: "30% 30% 50% 50% / 20% 20% 40% 40%",
          border: `2px solid ${borderGlow}`,
          boxShadow: `
            0 0 30px 8px ${glowColor},
            inset 0 0 40px 5px rgba(0, 0, 0, 0.5)
          `,
          transition: "border-color 0.5s ease, box-shadow 0.6s ease",
          maskImage: "linear-gradient(to bottom, black 55%, transparent 100%)",
          WebkitMaskImage: "linear-gradient(to bottom, black 55%, transparent 100%)",
        }}
      >
        <Image
          src="/genie-face.jpg"
          alt="Mystical genie oracle with turban and glowing eyes"
          width={240}
          height={300}
          className="object-cover object-top"
          style={{
            filter: isThinking
              ? "brightness(0.7) saturate(1.4) hue-rotate(10deg)"
              : isYes
                ? "brightness(1.15) saturate(1.1) hue-rotate(-10deg)"
                : isNo
                  ? "brightness(0.85) saturate(1.3) hue-rotate(5deg)"
                  : isUnsure
                    ? "brightness(1.05) saturate(1.2) sepia(0.15)"
                    : "brightness(1) saturate(1.15)",
            transition: "filter 0.5s ease",
          }}
          priority
        />

        {/* Color overlay for state reactions */}
        <div
          className="absolute inset-0"
          style={{
            background: isYes
              ? "linear-gradient(to top, rgba(34, 197, 94, 0.2) 0%, transparent 50%)"
              : isNo
                ? "linear-gradient(to top, rgba(239, 68, 68, 0.2) 0%, transparent 50%)"
                : isUnsure
                  ? "linear-gradient(to top, rgba(245, 158, 11, 0.2) 0%, transparent 50%)"
                  : isThinking
                    ? "linear-gradient(to top, rgba(139, 92, 246, 0.3) 0%, transparent 60%)"
                    : "linear-gradient(to top, rgba(139, 92, 246, 0.15) 0%, transparent 40%)",
            transition: "background 0.5s ease",
          }}
          aria-hidden="true"
        />

        {/* Thinking overlay -- pulsing vignette */}
        {isThinking && (
          <div
            className="absolute inset-0 animate-pulse-glow"
            style={{
              background: "radial-gradient(circle at 50% 40%, transparent 30%, rgba(88, 28, 175, 0.35) 100%)",
            }}
            aria-hidden="true"
          />
        )}
      </div>

      {/* Sparkle particles around portrait */}
      {[...Array(10)].map((_, i) => (
        <div
          key={`spark-${i}`}
          className="absolute rounded-full animate-drift"
          style={{
            width: `${2 + Math.random() * 3}px`,
            height: `${2 + Math.random() * 3}px`,
            backgroundColor: auraColor,
            opacity: 0.4,
            top: `${10 + Math.random() * 65}%`,
            left: `${10 + Math.random() * 80}%`,
            animationDelay: `${i * 1.5}s`,
            animationDuration: `${12 + Math.random() * 12}s`,
            filter: "blur(0.5px)",
            transition: "background-color 0.5s ease",
          }}
          aria-hidden="true"
        />
      ))}

      {/* Glowing orbs floating near hands/sides */}
      <div
        className="absolute z-20 rounded-full animate-pulse-glow"
        style={{
          width: "8px",
          height: "8px",
          backgroundColor: auraColor,
          left: "-10px",
          top: "50%",
          boxShadow: `0 0 12px 4px ${glowColor}`,
          transition: "background-color 0.5s ease, box-shadow 0.5s ease",
          animationDelay: "0.3s",
        }}
        aria-hidden="true"
      />
      <div
        className="absolute z-20 rounded-full animate-pulse-glow"
        style={{
          width: "8px",
          height: "8px",
          backgroundColor: auraColor,
          right: "-10px",
          top: "50%",
          boxShadow: `0 0 12px 4px ${glowColor}`,
          transition: "background-color 0.5s ease, box-shadow 0.5s ease",
          animationDelay: "0.8s",
        }}
        aria-hidden="true"
      />

      {/* Smaller accent orbs */}
      <div
        className="absolute z-20 rounded-full animate-pulse-glow"
        style={{
          width: "5px",
          height: "5px",
          backgroundColor: auraColor,
          left: "5px",
          top: "35%",
          boxShadow: `0 0 8px 3px ${glowColor}`,
          transition: "background-color 0.5s ease",
          animationDelay: "1.2s",
        }}
        aria-hidden="true"
      />
      <div
        className="absolute z-20 rounded-full animate-pulse-glow"
        style={{
          width: "5px",
          height: "5px",
          backgroundColor: auraColor,
          right: "5px",
          top: "35%",
          boxShadow: `0 0 8px 3px ${glowColor}`,
          transition: "background-color 0.5s ease",
          animationDelay: "1.7s",
        }}
        aria-hidden="true"
      />
    </div>
  )
}
