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


const API_URL = "http://127.0.0.1:5000"


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


  // Backend state
  const [sessionId, setSessionId] = useState<string | null>(null)

  const [currentQuestion, setCurrentQuestion] = useState("")

  const [currentFeature, setCurrentFeature] = useState("")

  const [currentValue, setCurrentValue] = useState("")

  const [questionNumber, setQuestionNumber] = useState(0)


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

            setGenieMessage(
              `Your song is "${data.song.title}" (${Math.round(data.confidence * 100)}%)`
            )

          }

          else if (data.type === "question") {

            setCurrentQuestion(data.text)

            setCurrentFeature(data.feature)

            setCurrentValue(data.value)

            setQuestionNumber(prev => prev + 1)

            setGenieMessage(getRandomComment(answer))

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

              totalQuestions={20}

            />


            <AnswerButtons

              onAnswer={handleAnswer}

              disabled={isDisabled}

            />

          </>

        )}


        {gameFinished && (

          <button onClick={handleRestart}>

            Summon Again

          </button>

        )}


      </div>


    </main>

  )

}
