"use client"

import { useEffect, useRef } from "react"

export function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationId: number

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener("resize", resize)

    const stars: {
      x: number
      y: number
      size: number
      speed: number
      opacity: number
      phase: number
    }[] = []

    for (let i = 0; i < 150; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 0.5,
        speed: Math.random() * 0.3 + 0.05,
        opacity: Math.random() * 0.5 + 0.1,
        phase: Math.random() * Math.PI * 2,
      })
    }

    const nebulaClouds: {
      x: number
      y: number
      radius: number
      hue: number
      speed: number
      phase: number
    }[] = []

    for (let i = 0; i < 5; i++) {
      nebulaClouds.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 200 + 100,
        hue: Math.random() > 0.5 ? 260 : 220,
        speed: Math.random() * 0.0005 + 0.0002,
        phase: Math.random() * Math.PI * 2,
      })
    }

    let time = 0

    const draw = () => {
      time += 0.01
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      for (const cloud of nebulaClouds) {
        const alpha = 0.03 + Math.sin(time * 0.5 + cloud.phase) * 0.015
        const gradient = ctx.createRadialGradient(
          cloud.x + Math.sin(time * 0.2 + cloud.phase) * 20,
          cloud.y + Math.cos(time * 0.15 + cloud.phase) * 15,
          0,
          cloud.x,
          cloud.y,
          cloud.radius
        )
        gradient.addColorStop(0, `hsla(${cloud.hue}, 60%, 40%, ${alpha})`)
        gradient.addColorStop(1, "transparent")
        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }

      for (const star of stars) {
        const twinkle =
          star.opacity +
          Math.sin(time * star.speed * 10 + star.phase) * 0.3
        ctx.beginPath()
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(200, 210, 255, ${Math.max(0, twinkle)})`
        ctx.fill()

        if (star.size > 1.3) {
          ctx.beginPath()
          ctx.arc(star.x, star.y, star.size * 2.5, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(180, 190, 255, ${Math.max(0, twinkle * 0.15)})`
          ctx.fill()
        }
      }

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      window.removeEventListener("resize", resize)
      cancelAnimationFrame(animationId)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  )
}
