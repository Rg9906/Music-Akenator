"use client"

import { useCallback, useEffect, useState } from "react"

import { StarField } from "./star-field"
import { GenieCharacter } from "./genie-character"
import { SpeechBubble } from "./speech-bubble"
import { QuestionDisplay } from "./question-display"
import { AnswerButtons } from "./answer-buttons"
import { SmokeParticles } from "./smoke-particles"

import { GENIE_COMMENTS } from "./types"
import type { GenieState } from "./types"


const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000"
const MAX_QUESTIONS = 20


function getRandomComment(type: keyof typeof GENIE_COMMENTS): string {

  const comments = GENIE_COMMENTS[type]

  return comments[Math.floor(Math.random() * comments.length)]

}


export function SongGenie() {

  // Mount safety (prevents hydration errors)
  const [mounted, setMounted] = useState(false)


  // Visual state
  const [genieState, setGenieState] = useState<GenieState>("idle")

  const [genieMessage, setGenieMessage] = useState(
    "I am the Song Genie. Think of a song..."
  )

  const [isThinking, setIsThinking] = useState(false)


  // Game state
  const [gameStarted, setGameStarted] = useState(false)

  const [gameFinished, setGameFinished] = useState(false)
  const [awaitingFeedback, setAwaitingFeedback] = useState(false)


  // Backend state
  const [sessionId, setSessionId] = useState<string | null>(null)

  const [currentQuestion, setCurrentQuestion] = useState("")

  const [currentFeature, setCurrentFeature] = useState("")

  const [currentValue, setCurrentValue] = useState("")

  const [questionNumber, setQuestionNumber] = useState(0)

  // Learning state
  const [showSongInput, setShowSongInput] = useState(false)
  const [songInput, setSongInput] = useState("")
  const [isLearning, setIsLearning] = useState(false)


  // Run once after mount
  useEffect(() => {

    setMounted(true)

  }, [])



  // Start game
  const handleStart = async () => {

    try {

      setGameStarted(true)

      setGenieState("thinking")

      setIsThinking(true)

      setGenieMessage("Consulting the spirits...")


      const res = await fetch(`${API_URL}/start`)


      if (!res.ok) {

        throw new Error("Backend failed")

      }


      const data = await res.json()


      setSessionId(data.session_id)

      setCurrentQuestion(data.text)

      setCurrentFeature(data.feature)

      setCurrentValue(data.value)

      setQuestionNumber(1)


      setGenieState("idle")

      setIsThinking(false)

      setGenieMessage("Answer truthfully.")


    }

    catch {

      setGenieMessage("Backend unreachable.")

      setGenieState("idle")

      setIsThinking(false)

    }

  }



  // Handle answer
  const handleAnswer = useCallback(

    async (answer: "yes" | "no" | "unsure") => {

      if (!sessionId) return


      setGenieState(answer)


      setTimeout(async () => {

        try {

          setGenieState("thinking")

          setIsThinking(true)


          const res = await fetch(`${API_URL}/answer`, {

            method: "POST",

            headers: {

              "Content-Type": "application/json"

            },

            body: JSON.stringify({

              session_id: sessionId,

              feature: currentFeature,

              value: currentValue,

              answer: answer

            })

          })


          if (!res.ok) {

            throw new Error("Backend failed")

          }


          const data = await res.json()


          setGenieState("idle")

          setIsThinking(false)


          if (data.type === "result") {

            setGameFinished(true)

            setAwaitingFeedback(true)

            // Format top candidates display
            const topCandidatesList = data.top_songs
              .slice(0, 3)
              .map((item: any, index: number) => 
                `${index + 1}. "${item.song.title}" (${Math.round(item.probability * 100)}%)`
              ).join('\n')

            setGenieMessage(
              `My top guesses are:\n\n${topCandidatesList}\n\nWas #1 correct?`
            )

          }

          else if (data.type === "question") {

            setCurrentQuestion(data.text)

            setCurrentFeature(data.feature)

            setCurrentValue(data.value)

            setQuestionNumber(prev => prev + 1)

            setGenieMessage(getRandomComment(answer))

          }

          else if (data.type === "learn") {

            setGameFinished(true)

            // Show top candidates and ask for song input
            const topCandidatesList = data.top_songs
              ? data.top_songs
                  .slice(0, 3)
                  .map((item: any, index: number) => 
                    `${index + 1}. "${item.song.title}" (${Math.round(item.probability * 100)}%)`
                  ).join('\n')
              : "No candidates found"

            setGenieMessage(
              `I couldn't guess your song in 20 questions.\n\nMy best guesses were:\n\n${topCandidatesList}\n\nWhat song were you thinking of?`
            )

            setShowSongInput(true)

          }

          else {

            setGameFinished(true)

            setGenieMessage("The spirits are uncertain.")

          }

        }

        catch {

          setGenieMessage("Connection lost.")

          setGenieState("idle")

          setIsThinking(false)

        }

      }, 500)

    },

    [sessionId, currentFeature, currentValue]

  )



  // Restart
  const handleRestart = () => {

    setGameStarted(false)

    setGameFinished(false)

    setSessionId(null)

    setCurrentQuestion("")

    setQuestionNumber(0)

    setGenieState("idle")

    setGenieMessage("Think of another song.")
    setAwaitingFeedback(false)
    setShowSongInput(false)
    setSongInput("")
    setIsLearning(false)

  }


  const handleFeedback = async (correct: boolean, correctSongTitle?: string) => {
    if (!sessionId) return
    
    if (!correct) {
      // Show song input for wrong guesses
      setGenieMessage("What song were you actually thinking of?")
      setShowSongInput(true)
      return
    }

    try {
      const res = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          correct,
          correct_song_title: correctSongTitle,
        }),
      })
      
      const data = await res.json()
      
      // Show learning feedback if available
      if (data.learning && data.learning.status === "learned") {
        const summary = data.learning.analysis_summary
        setGenieMessage(`Thanks! I learned from your answers. You got ${summary.confirms} questions right and had ${summary.mismatches} mismatches. Quality score: ${Math.round(data.learning.quality_score * 100)}%`)
      }
      
    } catch {
      // ignore feedback failures
    } finally {
      setAwaitingFeedback(false)
    }
  }

  const handleSongSubmit = async () => {
    if (!songInput.trim() || !sessionId) return
    
    setIsLearning(true)
    try {
      // Send feedback first with the correct song title (this triggers smart learning)
      const feedbackRes = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          correct: false,
          correct_song_title: songInput.trim(),
        }),
      })
      
      const feedbackData = await feedbackRes.json()
      
      if (feedbackData.learning && feedbackData.learning.status === "learned") {
        const summary = feedbackData.learning.analysis_summary
        setGenieMessage(`Smart learning complete! I analyzed your answers vs "${songInput}" and found ${summary.confirms} correct answers and ${summary.mismatches} mismatches. Quality: ${Math.round(feedbackData.learning.quality_score * 100)}%`)
      } else if (feedbackData.learning && feedbackData.learning.status === "rejected") {
        setGenieMessage(`I couldn't learn reliably from your answers (${feedbackData.learning.reason}), but thanks for playing!`)
      } else {
        setGenieMessage(`Thanks for playing with "${songInput}"!`)
      }
      
    } catch {
      setGenieMessage("Failed to analyze the learning, but thanks for playing!")
    } finally {
      setIsLearning(false)
      setShowSongInput(false)
      setSongInput("")
      setAwaitingFeedback(false)
    }
  }



  const isDisabled = genieState !== "idle"



  // Prevent hydration mismatch
  if (!mounted) return null



  return (

    <main className="relative w-full h-screen flex flex-col items-center justify-center overflow-hidden">


      <StarField />


      <div className="relative z-10 flex flex-col items-center gap-6 w-full max-w-lg px-4">


        <h1 className="text-4xl font-serif">

          Song Genie

        </h1>


        <SmokeParticles />


        <GenieCharacter state={genieState} />


        <SpeechBubble

          message={genieMessage}

          isThinking={isThinking}

        />


        {!gameStarted && (

          <button onClick={handleStart}>

            Begin the Divination

          </button>

        )}


        {gameStarted && !gameFinished && (

          <>

            <QuestionDisplay

              question={currentQuestion}

              questionNumber={questionNumber}

              totalQuestions={MAX_QUESTIONS}

            />


            <AnswerButtons

              onAnswer={handleAnswer}

              disabled={isDisabled}

            />

          </>

        )}


        {gameFinished && awaitingFeedback && (
          <div className="flex flex-wrap justify-center gap-3">
            <button onClick={() => handleFeedback(true)}>
              Yes, correct
            </button>
            <button onClick={() => handleFeedback(false)}>
              No, wrong
            </button>
          </div>
        )}

        {showSongInput && (
          <div className="flex flex-col gap-3 w-full max-w-sm">
            <input
              type="text"
              value={songInput}
              onChange={(e) => setSongInput(e.target.value)}
              placeholder="Enter song name..."
              className="px-4 py-2 rounded-lg border border-gray-300 bg-white text-black placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              disabled={isLearning}
              onKeyPress={(e) => e.key === 'Enter' && handleSongSubmit()}
            />
            <div className="flex gap-3">
              <button 
                onClick={handleSongSubmit}
                disabled={isLearning || !songInput.trim()}
                className="px-4 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLearning ? "Learning..." : "Teach Me"}
              </button>
              <button 
                onClick={handleRestart}
                className="px-4 py-2 rounded-lg border border-gray-400 text-gray-300 hover:bg-gray-800"
              >
                Skip
              </button>
            </div>
          </div>
        )}

        {gameFinished && !awaitingFeedback && (

          <button onClick={handleRestart}>

            Summon Again

          </button>

        )}


      </div>


    </main>

  )

}
