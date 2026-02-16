export type GenieState = "idle" | "thinking" | "yes" | "no" | "unsure"

export interface Question {
  id: number
  text: string
}

export const QUESTIONS: Question[] = [
  { id: 1, text: "Is this song from the 21st century?" },
  { id: 2, text: "Does the song have a strong beat?" },
  { id: 3, text: "Is the lead vocalist male?" },
  { id: 4, text: "Was this song a number one hit?" },
  { id: 5, text: "Does the song feature any instruments other than guitar?" },
  { id: 6, text: "Is the song in English?" },
  { id: 7, text: "Does the song have a music video?" },
  { id: 8, text: "Is the song considered a classic?" },
  { id: 9, text: "Does the song belong to the pop genre?" },
  { id: 10, text: "Has the song been covered by another artist?" },
  { id: 11, text: "Is the song longer than 4 minutes?" },
  { id: 12, text: "Does the song feature a collaboration?" },
  { id: 13, text: "Was the song released as a single?" },
  { id: 14, text: "Does the song have a dance-friendly tempo?" },
  { id: 15, text: "Is the song associated with a movie or TV show?" },
  { id: 16, text: "Does the song have a distinctive opening riff or melody?" },
  { id: 17, text: "Is the song about love or relationships?" },
  { id: 18, text: "Has the song been performed live at a major event?" },
  { id: 19, text: "Does the song feature electronic production?" },
  { id: 20, text: "Is the artist or band still active today?" },
]

export const GENIE_COMMENTS = {
  yes: [
    "Hmm... interesting.",
    "That narrows it down considerably.",
    "Yes, yes... I can feel it.",
    "The melody grows clearer...",
    "My musical senses are tingling.",
  ],
  no: [
    "That rules out many possibilities.",
    "Interesting... not what I expected.",
    "The fog clears a little more.",
    "Good... good... eliminating options.",
    "Every answer brings us closer.",
  ],
  unsure: [
    "Uncertain, are we? No matter.",
    "The mists remain cloudy on this one.",
    "Even uncertainty tells me something.",
    "A vague answer, but useful nonetheless.",
    "The spirits whisper mixed signals...",
  ],
  thinking: [
    "Let me consult the musical spirits...",
    "The melodies are swirling...",
    "I sense a rhythm forming...",
    "The oracle is contemplating...",
  ],
  guess: [
    "I believe I know your song!",
    "The spirits have revealed the answer!",
    "The melody is crystal clear now!",
  ],
}
