export type GenieState = "idle" | "thinking" | "yes" | "no" | "unsure"

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
