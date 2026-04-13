import Link from "next/link"

export default function Page() {
  return (
    <main className="relative w-full h-screen flex flex-col items-center justify-center overflow-hidden">
      <div className="relative z-10 flex flex-col items-center gap-6 w-full max-w-2xl px-4 text-center">
        <h1 className="text-5xl font-serif">Song Genie</h1>
        <p className="text-lg text-muted-foreground max-w-xl">
          A learning oracle that adapts its questions from past games.
        </p>
        <div className="flex flex-wrap justify-center gap-4 mt-4">
          <Link
            href="/game"
            className="px-6 py-3 rounded-full bg-indigo-600 text-white hover:bg-indigo-500 transition"
          >
            Start a Game
          </Link>
          <Link
            href="/history"
            className="px-6 py-3 rounded-full border border-indigo-500 text-indigo-200 hover:bg-indigo-900/40 transition"
          >
            View Past Sessions
          </Link>
          <Link
            href="/insights"
            className="px-6 py-3 rounded-full border border-slate-500 text-slate-200 hover:bg-slate-900/40 transition"
          >
            Question Insights
          </Link>
        </div>
      </div>
    </main>
  )
}
