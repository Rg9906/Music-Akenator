"use client"

import { useEffect, useRef } from "react"

export function SmokeParticles() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationId: number

    const resize = () => {
      const parent = canvas.parentElement
      if (!parent) return
      canvas.width = parent.clientWidth
      canvas.height = parent.clientHeight
    }
    resize()

    const observer = new ResizeObserver(resize)
    if (canvas.parentElement) observer.observe(canvas.parentElement)

    interface SmokeParticle {
      x: number
      y: number
      vx: number
      vy: number
      size: number
      opacity: number
      life: number
      maxLife: number
      hue: number
    }

    const particles: SmokeParticle[] = []

    function spawnParticle() {
      if (!canvas) return
      const cx = canvas.width / 2
      const bottomY = canvas.height * 0.65
      particles.push({
        x: cx + (Math.random() - 0.5) * 40,
        y: bottomY + Math.random() * 20,
        vx: (Math.random() - 0.5) * 0.4,
        vy: -(Math.random() * 0.6 + 0.3),
        size: Math.random() * 15 + 8,
        opacity: 0,
        life: 0,
        maxLife: 120 + Math.random() * 100,
        hue: Math.random() > 0.5 ? 260 : 220,
      })
    }

    let time = 0
    function draw() {
      if (!canvas || !ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      time++

      if (time % 6 === 0 && particles.length < 30) {
        spawnParticle()
      }

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.life++
        p.x += p.vx + Math.sin(time * 0.02 + i) * 0.2
        p.y += p.vy
        p.size += 0.15

        const lifeRatio = p.life / p.maxLife
        if (lifeRatio < 0.15) {
          p.opacity = lifeRatio / 0.15
        } else if (lifeRatio > 0.6) {
          p.opacity = 1 - (lifeRatio - 0.6) / 0.4
        }
        p.opacity = Math.max(0, Math.min(0.25, p.opacity * 0.25))

        ctx.beginPath()
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size)
        gradient.addColorStop(0, `hsla(${p.hue}, 60%, 50%, ${p.opacity})`)
        gradient.addColorStop(0.5, `hsla(${p.hue}, 50%, 35%, ${p.opacity * 0.5})`)
        gradient.addColorStop(1, `hsla(${p.hue}, 40%, 20%, 0)`)
        ctx.fillStyle = gradient
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()

        if (p.life >= p.maxLife) {
          particles.splice(i, 1)
        }
      }

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animationId)
      observer.disconnect()
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      aria-hidden="true"
    />
  )
}
