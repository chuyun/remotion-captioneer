/**
 * Caption translation helpers (OpenAI).
 * Preserves per-segment word count and timing; only text fields are updated.
 */

import type { CaptionData, CaptionSegment, Word } from "./types.js";

export type TranslateCaptionsOptions = {
  /** BCP-47 language code, e.g. "es", "fr", "de", "ar", "he" */
  targetLanguage: string;
  /** Defaults to OPENAI_API_KEY */
  apiKey?: string;
  /** Chat model (default gpt-4o-mini) */
  model?: string;
};

type OpenAIChatResponse = {
  choices?: Array<{ message?: { content?: string | null } }>;
  error?: { message?: string };
};

function parseJsonObject(content: string): Record<string, unknown> {
  const trimmed = content.trim();
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("Model did not return a JSON object");
  }
  return JSON.parse(trimmed.slice(start, end + 1)) as Record<string, unknown>;
}

async function translateWordLists(
  segments: Array<{ words: string[] }>,
  opts: TranslateCaptionsOptions
): Promise<string[][]> {
  const apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "OpenAI API key missing. Set OPENAI_API_KEY or pass apiKey in options."
    );
  }

  const model = opts.model ?? "gpt-4o-mini";
  const payload = {
    model,
    temperature: 0.2,
    response_format: { type: "json_object" as const },
    messages: [
      {
        role: "system" as const,
        content: `You translate subtitle/caption words for video. Target language: ${opts.targetLanguage}.
Return JSON: {"segments":[{"words":["..."]}]}
Rules:
- Same number of segments as input, same order.
- Each segment has the same number of words as input, same order.
- Preserve punctuation attached to words (e.g. "hello," stays one token).
- Do not add explanations.`,
      },
      {
        role: "user" as const,
        content: JSON.stringify({ segments }),
      },
    ],
  };

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const body = (await res.json()) as OpenAIChatResponse;
  if (!res.ok) {
    throw new Error(
      body.error?.message ?? `OpenAI request failed (${res.status})`
    );
  }

  const content = body.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("Empty response from OpenAI");
  }

  const parsed = parseJsonObject(content);
  const outSegs = parsed.segments;
  if (!Array.isArray(outSegs)) {
    throw new Error('Invalid response: expected "segments" array');
  }

  return outSegs.map((seg, i) => {
    if (!seg || typeof seg !== "object") {
      throw new Error(`Invalid segment at index ${i}`);
    }
    const words = (seg as { words?: unknown }).words;
    if (!Array.isArray(words) || !words.every((w) => typeof w === "string")) {
      throw new Error(`Invalid words array at segment ${i}`);
    }
    return words as string[];
  });
}

/**
 * Returns new CaptionData with translated `text` and each `word.word`;
 * timings, confidence, and language code are preserved (language is not auto-updated).
 */
export async function translateCaptionData(
  captionData: CaptionData,
  opts: TranslateCaptionsOptions
): Promise<CaptionData> {
  const input = captionData.segments.map((s) => ({
    words: s.words.map((w) => w.word),
  }));

  const translatedLists = await translateWordLists(input, opts);

  if (translatedLists.length !== captionData.segments.length) {
    throw new Error("Translation returned wrong segment count");
  }

  const segments: CaptionSegment[] = captionData.segments.map((seg, i) => {
    const newWordsStr = translatedLists[i];
    if (newWordsStr.length !== seg.words.length) {
      throw new Error(
        `Segment ${i}: expected ${seg.words.length} words, got ${newWordsStr.length}`
      );
    }
    const words: Word[] = seg.words.map((w, j) => ({
      ...w,
      word: newWordsStr[j]!,
    }));
    const text = words.map((w) => w.word).join(" ");
    return { ...seg, text, words };
  });

  return { ...captionData, segments };
}
