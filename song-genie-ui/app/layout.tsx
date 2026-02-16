import React from "react"
import type { Metadata, Viewport } from 'next'
import { Cinzel, Raleway } from 'next/font/google'

import './globals.css'

const cinzel = Cinzel({
  subsets: ['latin'],
  variable: '--font-cinzel',
})

const raleway = Raleway({
  subsets: ['latin'],
  variable: '--font-raleway',
})

export const metadata: Metadata = {
  title: 'Song Genie - The Mystical Song Guessing Oracle',
  description:
    'A magical AI genie that guesses your song through mystical yes/no questions. Experience the dark-fantasy song guessing oracle.',
}

export const viewport: Viewport = {
  themeColor: '#0a0a1a',
  userScalable: false,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${cinzel.variable} ${raleway.variable}`}>
      <body className="font-sans antialiased overflow-hidden">
        {children}
      </body>
    </html>
  )
}
