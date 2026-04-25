// Behavioral coding manual based on interpersonal synchrony literature
// Provides structured guides for each behavioral event type

export interface CodingLevel {
  level: number;
  label: string;
  description: string;
  examples: string[];
  color: string; // semantic indication
}

export interface BehavioralCodingGuide {
  type: string;
  label: string;
  definition: string;
  whatToLookFor: string[];
  commonMistakes: string[];
  levels: CodingLevel[];
  references: string[];
  keyboardShortcut: string;
}

export const BEHAVIORAL_CODING_GUIDES: BehavioralCodingGuide[] = [
  {
    type: "gaze_coordination",
    label: "Gaze Coordination",
    definition:
      "Temporal alignment of visual attention between interacting partners, including mutual gaze (eye contact), joint attention (looking at the same object), and gaze following (one partner directing gaze after the other).",
    whatToLookFor: [
      "Both partners looking at each other's eyes simultaneously (mutual gaze)",
      "One partner follows the other's gaze to an object or location (gaze following)",
      "Both partners attend to the same referent within ~500ms (joint attention)",
      "Gaze alternation patterns during turn-taking",
    ],
    commonMistakes: [
      "Confusing looking at the same general area with genuine joint attention — the referent must match",
      "Missing brief mutual gaze episodes (<300ms) that are still socially meaningful",
      "Coding gaze at camera/screen as mutual gaze in video-mediated interactions",
      "Over-scoring during monologue segments where listener gaze is expected",
    ],
    levels: [
      { level: 0, label: "No Coordination", description: "Partners gaze at unrelated targets; no alignment", examples: ["Looking away from each other", "Distracted by phone"], color: "hsl(var(--muted-foreground))" },
      { level: 1, label: "Brief Alignment", description: "Fleeting (<500ms) mutual gaze or gaze following", examples: ["Quick glance during speech", "Brief check-in look"], color: "hsl(185, 55%, 50%)" },
      { level: 2, label: "Moderate Coordination", description: "Sustained joint attention (0.5-2s) or clear gaze following", examples: ["Both looking at a shared task", "Following a pointing gesture"], color: "hsl(185, 55%, 40%)" },
      { level: 3, label: "Strong Coordination", description: "Extended mutual gaze (>2s) or rapid turn-taking gaze patterns", examples: ["Deep eye contact during emotional sharing", "Synchronized gaze shifts during collaborative tasks"], color: "hsl(185, 70%, 35%)" },
    ],
    references: [
      "Behrens et al. (2020) — WCC for gaze coordinates",
      "Prochazkova et al. (2018) — Pupil mimicry and gaze dynamics",
    ],
    keyboardShortcut: "1",
  },
  {
    type: "head_movement",
    label: "Head Movement Sync",
    definition:
      "Coordination of head orientation changes, nodding patterns, and head tilting between interacting partners. Includes mirroring (same direction) and complementary movements (responsive, different direction).",
    whatToLookFor: [
      "Simultaneous or closely timed head nods during agreement",
      "Head tilting in the same or opposite direction (mirroring vs. complementary)",
      "Rhythmic head movements aligned with speech prosody of either partner",
      "Head turning coordination when attending to the same stimulus",
    ],
    commonMistakes: [
      "Coding incidental same-direction head turns (e.g., both turning to a sound) as synchrony",
      "Ignoring complementary movements (head tilt in opposite direction can still be synchronous)",
      "Not distinguishing between social nodding and self-regulatory movements",
      "Over-scoring head shaking during disagreement as asynchrony — it can still be temporally coordinated",
    ],
    levels: [
      { level: 0, label: "No Sync", description: "Independent head movements with no temporal relation", examples: ["Random head positions", "One still while other moves"], color: "hsl(var(--muted-foreground))" },
      { level: 1, label: "Occasional Match", description: "Sporadic head nods or tilts near each other in time", examples: ["One nod returned after delay", "Brief tilt match"], color: "hsl(200, 55%, 50%)" },
      { level: 2, label: "Rhythmic Alignment", description: "Repeated patterns of coordinated head movement", examples: ["Matched nodding during storytelling", "Head tilts during emotional exchange"], color: "hsl(200, 55%, 40%)" },
      { level: 3, label: "Tight Coupling", description: "Near-simultaneous, sustained head movement coordination", examples: ["Locked nodding during enthusiastic agreement", "Mirror-image head positions held >3s"], color: "hsl(200, 70%, 35%)" },
    ],
    references: [
      "Ramseyer & Tschacher (2011) — Movement synchrony in psychotherapy",
    ],
    keyboardShortcut: "2",
  },
  {
    type: "gesture_mirroring",
    label: "Gesture Mirroring",
    definition:
      "Temporal alignment and imitation of hand/arm gestures between partners, including iconic gestures (representational), beat gestures (rhythmic), and self-touching patterns.",
    whatToLookFor: [
      "One partner adopts a similar hand position or gesture shape as the other",
      "Beat gestures aligned with the other's speech rhythm",
      "Gesture mimicry with a short delay (automatic imitation, typically <2s)",
      "Coordinated reaching or pointing toward shared objects",
    ],
    commonMistakes: [
      "Confusing functional similarity with temporal coordination — same gesture at different times ≠ synchrony",
      "Over-scoring self-touching mimicry (hair touch → hair touch) which may be coincidental",
      "Not noticing subtle gestural adaptations (gesture size, speed matching)",
      "Ignoring gesture absence synchrony (both partners still at the same time)",
    ],
    levels: [
      { level: 0, label: "No Mirroring", description: "Gestures are unrelated in form and timing", examples: ["One gestures while other is still", "Different gestures at random times"], color: "hsl(var(--muted-foreground))" },
      { level: 1, label: "Partial Echo", description: "Similar gesture produced after noticeable delay (>2s)", examples: ["Delayed adoption of crossed arms", "Late echo of a wave"], color: "hsl(160, 50%, 50%)" },
      { level: 2, label: "Active Mirroring", description: "Clear gesture imitation within 0.5-2s", examples: ["Hand-on-chin matching", "Mirrored lean-forward"], color: "hsl(160, 50%, 40%)" },
      { level: 3, label: "Synchronized Gesturing", description: "Near-simultaneous gesture production or tightly coordinated exchange", examples: ["Matched pointing during explanation", "Rhythmic hand movements in sync"], color: "hsl(160, 60%, 35%)" },
    ],
    references: [
      "Chartrand & Bargh (1999) — The chameleon effect",
      "Koul et al. (2023) — Spontaneous dyadic behavior and neural synchrony",
    ],
    keyboardShortcut: "3",
  },
  {
    type: "facial_expression",
    label: "Facial Expression Sync",
    definition:
      "Temporal coordination of facial muscle activations (Action Units) between partners, including emotional mimicry (matching affect), complementary expressions (responsive affect), and facial movement rhythm.",
    whatToLookFor: [
      "Smile matching: one partner smiles → other smiles within ~1s",
      "Eyebrow raises coordinated with speech emphasis or surprise",
      "Emotional contagion moments (laughter spreading, shared concern expressions)",
      "Micro-expression matching during empathic listening",
    ],
    commonMistakes: [
      "Confusing polite smiling (social display) with genuine emotional synchrony",
      "Missing subtle AU activations that are visible in slow playback but not real-time",
      "Scoring display-rule expressions (masking) as genuine facial sync",
      "Not accounting for cultural differences in facial expressivity baseline",
    ],
    levels: [
      { level: 0, label: "No Match", description: "Facial expressions are unrelated or neutral", examples: ["One smiles while other is blank", "Mismatched affect"], color: "hsl(var(--muted-foreground))" },
      { level: 1, label: "Weak Mimicry", description: "Subtle or delayed facial matching (>1s)", examples: ["Slow smile response", "Faint eyebrow echo"], color: "hsl(35, 80%, 60%)" },
      { level: 2, label: "Clear Mimicry", description: "Recognizable facial matching within 0.5-1s", examples: ["Quick smile return", "Matched concern expression"], color: "hsl(35, 80%, 50%)" },
      { level: 3, label: "Emotional Resonance", description: "Rapid, intense, and sustained facial coordination", examples: ["Shared laughter", "Simultaneous surprise reaction", "Mutual tearfulness"], color: "hsl(35, 90%, 40%)" },
    ],
    references: [
      "Behrens et al. (2020) — WCC for AU intensities",
      "Chartrand & Bargh (1999) — Automatic mimicry",
    ],
    keyboardShortcut: "4",
  },
  {
    type: "vocal_sync",
    label: "Vocal Synchrony",
    definition:
      "Temporal coordination of vocal features between partners, including turn-taking rhythm, pitch matching, speech rate entrainment, and simultaneous vocalization (e.g., laughter, backchannels).",
    whatToLookFor: [
      "Smooth turn-taking with consistent inter-turn intervals",
      "Pitch convergence: partners' fundamental frequency (F0) becoming more similar over time",
      "Speech rate matching: one partner adjusting pace to match the other",
      "Synchronized backchannels ('mm-hm', 'yeah') timed to partner's speech boundaries",
    ],
    commonMistakes: [
      "Scoring simultaneous speech (overlap) as synchrony — it can be either coordinated or disruptive",
      "Not distinguishing cooperative overlap (e.g., finishing sentences) from competitive interruptions",
      "Missing prosodic matching that occurs over longer time windows (30s+)",
      "Confusing volume matching with true vocal synchrony",
    ],
    levels: [
      { level: 0, label: "Discoordinated", description: "Frequent interruptions, awkward silences, mismatched rhythm", examples: ["Talking over each other", "Long uncomfortable pauses"], color: "hsl(var(--muted-foreground))" },
      { level: 1, label: "Basic Turn-Taking", description: "Functional but unpolished conversational flow", examples: ["Standard pauses between turns", "Occasional overlap"], color: "hsl(262, 60%, 60%)" },
      { level: 2, label: "Fluid Exchange", description: "Smooth turn-taking with well-timed backchannels", examples: ["Quick, natural transitions", "Supportive 'mm-hm' timing"], color: "hsl(262, 60%, 50%)" },
      { level: 3, label: "Vocal Entrainment", description: "Pitch, rhythm, and rate converge; highly coordinated vocalization", examples: ["Finishing each other's sentences", "Shared laughter onset", "Pitch matching during emotional speech"], color: "hsl(262, 70%, 40%)" },
    ],
    references: [
      "Ramseyer & Tschacher (2011) — Nonverbal synchrony measurement",
    ],
    keyboardShortcut: "5",
  },
];

export function getGuideForEvent(eventType: string): BehavioralCodingGuide | undefined {
  return BEHAVIORAL_CODING_GUIDES.find((g) => g.type === eventType);
}
