import type { RoundPlayer } from "@/lib/api";

export function playerDisplayName(
  players: RoundPlayer[], id: string | null, fallback: string,
): string {
  const normalized = fallback.trim().toLowerCase();
  const player = players.find((candidate) => id
    ? candidate.id === id
    : candidate.id.toLowerCase() === normalized || candidate.name.toLowerCase() === normalized,
  );
  if (!player) return fallback;
  return player.kind === "human" ? `${player.name} (You)` : player.name;
}
